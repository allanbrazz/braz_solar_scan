#core/views/fdd
from __future__ import annotations
from core.views._imports import *
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
from django.db.models import Count
from zoneinfo import ZoneInfo
import inspect

from core.services.fdd_mismatch import (
    MismatchThresholds,
    classify_mismatch_series,
    CODE_INVALID,
)


# Models
from core.models import (
    PVPlant,
    PVPlantMergedRecord15m,
    PlantDiagnostic15m,
)
# ----------------------------
# ----------- S A N I D A D E  D O  S I S T E M A  (Mismatch FDD)
# ----------------------------



logger = logging.getLogger(__name__)


# ----------------------------
# JSON strict/robusto
# ----------------------------
def _json_response_strict(payload: Any, *, status: int = 200) -> JsonResponse:
    """JsonResponse com serializer robusto (datetime, date, numpy, etc.)."""

    def _default(o: Any):
        if isinstance(o, (datetime, date)):
            return o.isoformat()

        # numpy (opcional)
        try:
            import numpy as np  # type: ignore

            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass

        # dataclasses
        if is_dataclass(o):
            return asdict(o)

        return str(o)

    safe = isinstance(payload, dict)
    return JsonResponse(
        payload,
        status=status,
        safe=safe,
        json_dumps_params={"ensure_ascii": False, "default": _default},
    )


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _sum_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    ok = False
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        ok = True
    return acc if ok else None


def _mean_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    n = 0
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        n += 1
    return (acc / n) if n else None


def _pick_best_sources(
    plant_id: int,
    dt0_utc: datetime,
    dt1_utc: datetime,
) -> Tuple[Optional[str], Optional[str]]:
    """Escolhe (source_oper, source_meteo) com maior n no range."""
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper", "source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    if not row:
        return None, None
    return row.get("source_oper"), row.get("source_meteo")


def _upsert_diag15m(
    *,
    plant: PVPlant,
    times_utc: List[datetime],
    codes: List[int],
    labels: List[str],
    valid: List[bool],
    g_poa: List[Optional[float]],
    tcell_c: List[Optional[float]],
    pac_real_w: List[Optional[float]],
    pac_model_w: List[Optional[float]],
    mismatch_rel: List[Optional[float]],
) -> Dict[str, Any]:
    """
    Upsert em PlantDiagnostic15m para o range.
    FIX importante:
      - bulk_create/bulk_update NÃO disparam auto_now/auto_now_add.
      - Portanto, setamos updated_at (e created_at se existir) manualmente.
    """
    assert (
        len(times_utc)
        == len(codes)
        == len(labels)
        == len(valid)
        == len(g_poa)
        == len(tcell_c)
        == len(pac_real_w)
        == len(pac_model_w)
        == len(mismatch_rel)
    )

    existing: Dict[datetime, PlantDiagnostic15m] = {}
    chunk = 1000
    for i in range(0, len(times_utc), chunk):
        ts_chunk = times_utc[i : i + chunk]
        qs = PlantDiagnostic15m.objects.filter(plant=plant, ts_utc__in=ts_chunk)
        for obj in qs:
            existing[obj.ts_utc] = obj

    to_create: List[PlantDiagnostic15m] = []
    to_update: List[PlantDiagnostic15m] = []

    now = timezone.now()

    for i, ts in enumerate(times_utc):
        obj = existing.get(ts)
        is_new = obj is None
        if is_new:
            obj = PlantDiagnostic15m(plant=plant, ts_utc=ts)
            to_create.append(obj)
        else:
            to_update.append(obj)

        obj.rca_code = int(codes[i])
        obj.rca_label = str(labels[i] or "invalid")
        obj.valid = bool(valid[i])

        obj.g_poa = g_poa[i]
        obj.tcell_c = tcell_c[i]
        obj.pac_real_w = pac_real_w[i]
        obj.pac_model_w = pac_model_w[i]
        obj.mismatch_rel = mismatch_rel[i]

        # timestamps (bulk_* não chama save())
        if hasattr(obj, "updated_at"):
            setattr(obj, "updated_at", now)
        if is_new and hasattr(obj, "created_at") and getattr(obj, "created_at", None) is None:
            setattr(obj, "created_at", now)

    with transaction.atomic():
        if to_create:
            PlantDiagnostic15m.objects.bulk_create(to_create, ignore_conflicts=True)
        if to_update:
            fields = [
                "rca_code",
                "rca_label",
                "valid",
                "g_poa",
                "tcell_c",
                "pac_real_w",
                "pac_model_w",
                "mismatch_rel",
            ]
            if hasattr(PlantDiagnostic15m, "updated_at"):
                fields.append("updated_at")

            PlantDiagnostic15m.objects.bulk_update(to_update, fields=fields)

    return {"created": len(to_create), "updated": len(to_update)}


# ----------------------------
# View (página)
# ----------------------------
@require_GET
@login_required
def mismatch_fdd_view(request: HttpRequest):
    """Página: Heatmap de mismatch + detalhes ao clicar."""
    qs = (
        PVPlant.objects.all().order_by("nome")
        if request.user.is_superuser
        else PVPlant.objects.filter(owner=request.user).order_by("nome")
    )
    plants = list(qs)

    d_end = date.today()
    d_start = d_end - timedelta(days=7)

    # FIX: adiciona "today" para templates antigos que usam default:today
    return render(
        request,
        "dashboard/mismatch_fdd.html",
        {
            "plants": plants,
            "default_start": d_start.isoformat(),
            "default_end": d_end.isoformat(),
            "today": d_end.isoformat(),
            "api_url": reverse("mismatch_fdd_api"),
        },
    )


# ----------------------------
# API (GET/POST)
# ----------------------------
@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_api(request: HttpRequest) -> JsonResponse:
    data = request.POST if request.method == "POST" else request.GET

    try:
        plant_id = int((data.get("plant_id") or "0").strip())
    except Exception:
        return _json_response_strict({"ok": False, "error": "plant_id inválido"}, status=400)

    plant = (
        PVPlant.objects.filter(id=plant_id)
        .select_related("details", "details__module", "details__inverter")
        .first()
    )
    if not plant:
        return _json_response_strict({"ok": False, "error": "Planta não encontrada"}, status=404)

    if (not request.user.is_superuser) and plant.owner_id and (plant.owner_id != request.user.id):
        return _json_response_strict({"ok": False, "error": "Sem permissão para esta planta"}, status=403)

    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"
        tz = ZoneInfo("UTC")

    d0 = _parse_date(data.get("start") or "")
    d1 = _parse_date(data.get("end") or "")
    if not d0 or not d1:
        return _json_response_strict({"ok": False, "error": "start/end (YYYY-MM-DD) são obrigatórios"}, status=400)
    if d1 < d0:
        return _json_response_strict({"ok": False, "error": "end < start"}, status=400)

    dt0_local = datetime.combine(d0, time.min, tzinfo=tz)
    dt1_local = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    src_oper_raw = (data.get("source_oper") or data.get("src_oper") or "").strip()
    src_meteo = (data.get("source_meteo") or data.get("src_meteo") or "").strip() or None

    if not src_meteo:
        _, best_m = _pick_best_sources(plant_id, dt0_utc, dt1_utc)
        src_meteo = best_m

    if not src_meteo:
        return _json_response_strict({"ok": False, "error": "Sem registros no range (PVPlantMergedRecord15m)."}, status=404)

    src_oper_rows = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=src_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    source_oper_list = [r["source_oper"] for r in src_oper_rows if r.get("source_oper")]
    if not source_oper_list:
        return _json_response_strict({"ok": False, "error": "Sem source_oper para a fonte meteo selecionada no range."}, status=404)

    want_all = (not src_oper_raw) or (src_oper_raw.upper() == "ALL")
    if want_all:
        selected_sources = list(source_oper_list)
    else:
        if src_oper_raw not in source_oper_list:
            return _json_response_strict({"ok": False, "error": f"source_oper '{src_oper_raw}' não existe no range."}, status=404)
        selected_sources = [src_oper_raw]

    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=src_meteo,
            source_oper__in=selected_sources,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .order_by("ts_utc")
        .values(
            "ts_utc",
            "source_oper",
            "p_ac_w",
            "p_dc_w",
            "e_ac_wh_15",
            "v_dc_v",
            "i_dc_a",
            "v_ac_v",
            "i_ac_a",
            "inv_coverage",
            "flag_inv_missing",
            "gti",
            "ghi",
            "dni",
            "dhi",
            "temp_air",
            "wind_speed",
            "rh",
            "flag_meteo_missing",
        )
    )
    rows = list(qs)
    if not rows:
        return _json_response_strict({"ok": False, "error": "Sem registros no range para as fontes selecionadas."}, status=404)

    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        ts = r["ts_utc"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt_tz.utc)
        src = r.get("source_oper") or ""
        if not src:
            continue
        per_ts.setdefault(ts, {})[src] = r

    times_utc = sorted(per_ts.keys())
    n = len(times_utc)

    p_ac_w: List[Optional[float]] = [None] * n
    p_dc_w: List[Optional[float]] = [None] * n
    e_ac_wh_15: List[Optional[float]] = [None] * n
    v_dc_v: List[Optional[float]] = [None] * n
    i_dc_a: List[Optional[float]] = [None] * n
    v_ac_v: List[Optional[float]] = [None] * n
    i_ac_a: List[Optional[float]] = [None] * n
    inv_cov: List[Optional[float]] = [None] * n
    flag_inv_missing: List[bool] = [False] * n

    gti: List[Optional[float]] = [None] * n
    ghi: List[Optional[float]] = [None] * n
    dni: List[Optional[float]] = [None] * n
    dhi: List[Optional[float]] = [None] * n
    temp_air: List[Optional[float]] = [None] * n
    wind_speed: List[Optional[float]] = [None] * n
    rh: List[Optional[float]] = [None] * n
    flag_meteo_missing: List[bool] = [False] * n

    series_by_source: Dict[str, Dict[str, List[Optional[float]]]] = {
        src: {
            "p_ac_w": [None] * n,
            "p_dc_w": [None] * n,
            "e_ac_wh_15": [None] * n,
            "v_dc_v": [None] * n,
            "i_dc_a": [None] * n,
            "v_ac_v": [None] * n,
            "i_ac_a": [None] * n,
        }
        for src in selected_sources
    }

    for i, ts in enumerate(times_utc):
        by_src = per_ts.get(ts, {})

        pac_l: List[Optional[float]] = []
        pdc_l: List[Optional[float]] = []
        e15_l: List[Optional[float]] = []
        vdc_l: List[Optional[float]] = []
        idc_l: List[Optional[float]] = []
        vac_l: List[Optional[float]] = []
        iac_l: List[Optional[float]] = []
        cov_l: List[Optional[float]] = []

        inv_missing_any = False
        first_row: Optional[Dict[str, Any]] = None

        for src in selected_sources:
            r = by_src.get(src)
            if r is None:
                continue
            if first_row is None:
                first_row = r

            pac = _as_float(r.get("p_ac_w"))
            pdc = _as_float(r.get("p_dc_w"))
            e15 = _as_float(r.get("e_ac_wh_15"))
            vdc = _as_float(r.get("v_dc_v"))
            idc = _as_float(r.get("i_dc_a"))
            vac = _as_float(r.get("v_ac_v"))
            iac = _as_float(r.get("i_ac_a"))
            cov = _as_float(r.get("inv_coverage"))
            inv_missing_any = inv_missing_any or bool(r.get("flag_inv_missing") or False)

            pac_l.append(pac)
            pdc_l.append(pdc)
            e15_l.append(e15)
            vdc_l.append(vdc)
            idc_l.append(idc)
            vac_l.append(vac)
            iac_l.append(iac)
            cov_l.append(cov)

            sb = series_by_source.get(src)
            if sb is not None:
                sb["p_ac_w"][i] = pac
                sb["p_dc_w"][i] = pdc
                sb["e_ac_wh_15"][i] = e15
                sb["v_dc_v"][i] = vdc
                sb["i_dc_a"][i] = idc
                sb["v_ac_v"][i] = vac
                sb["i_ac_a"][i] = iac

        p_ac_w[i] = _sum_none(pac_l)
        p_dc_w[i] = _sum_none(pdc_l)
        e_ac_wh_15[i] = _sum_none(e15_l)
        v_dc_v[i] = _mean_none(vdc_l)
        i_dc_a[i] = _sum_none(idc_l)
        v_ac_v[i] = _mean_none(vac_l)
        i_ac_a[i] = _sum_none(iac_l)
        inv_cov[i] = _mean_none(cov_l)
        flag_inv_missing[i] = inv_missing_any

        if first_row is not None:
            gti[i] = _as_float(first_row.get("gti"))
            ghi[i] = _as_float(first_row.get("ghi"))
            dni[i] = _as_float(first_row.get("dni"))
            dhi[i] = _as_float(first_row.get("dhi"))
            temp_air[i] = _as_float(first_row.get("temp_air"))
            wind_speed[i] = _as_float(first_row.get("wind_speed"))
            rh[i] = _as_float(first_row.get("rh"))
            flag_meteo_missing[i] = bool(first_row.get("flag_meteo_missing") or False)

    # timestamps locais/utc (strings)
    x_local_dt = [t.astimezone(tz) for t in times_utc]
    x_local = [t.isoformat() for t in x_local_dt]
    x_utc = [t.astimezone(dt_tz.utc).isoformat() for t in times_utc]

    # extras p/ heatmap sem depender de Date() no browser
    hm_day_local = [t.date().isoformat() for t in x_local_dt]
    hm_minute_local = [t.hour * 60 + t.minute for t in x_local_dt]

    g_poa_used = [gti_i if gti_i is not None else ghi_i for gti_i, ghi_i in zip(gti, ghi)]

    def _gf(key: str, default: float) -> float:
        raw = (data.get(key) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _gi(key: str, default: int) -> int:
        raw = (data.get(key) or "").strip()
        if not raw:
            return int(default)
        try:
            return int(float(raw))
        except Exception:
            return int(default)

    thr = MismatchThresholds(
        gpoa_gate_wm2=_gf("gpoa_gate", 200.0),
        warn_abs=_gf("warn_abs", 0.35),
        fault_abs=_gf("fault_abs", 0.80),
        meteo_pos_abs=_gf("meteo_pos_abs", 0.25),
        shading_std_abs=_gf("shading_std_abs", 0.22),
        shading_window_points=_gi("shading_window_points", 6),
        dt_minutes=15.0,
        max_gap_minutes=_gf("max_gap_minutes", 30.0),
    )

    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
        return _json_response_strict(
            {"ok": False, "error": "PVPlantDetails.module não configurado. Cadastre o módulo em 'Planta > Detalhes'."},
            status=400,
        )

    n_mod = int(getattr(details, "modules_total", 0) or 0)
    if n_mod <= 0:
        return _json_response_strict(
            {"ok": False, "error": "PVPlantDetails.modules_total inválido. Configure strings/módulos totais."},
            status=400,
        )

    try:
        import numpy as np
        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
        )

        mod = module_from_pvmodule(details.module)
        inv = getattr(details, "inverter", None)
        pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

        pld = asdict(pl) if is_dataclass(pl) else dict(getattr(pl, "__dict__", {}))
        if pld.get("lat_deg") is None:
            pld["lat_deg"] = _as_float(getattr(plant, "latitude", None))
        if pld.get("lon_deg") is None:
            pld["lon_deg"] = _as_float(getattr(plant, "longitude", None))
        if pld.get("tilt_deg") is None:
            pld["tilt_deg"] = _as_float(getattr(details, "tilt_deg", None))
        if pld.get("azimuth_deg") is None:
            pld["azimuth_deg"] = _as_float(getattr(details, "azimuth_deg", None))
        pl = pl.__class__(**pld)

        def list_to_np_nan(xs):
            out = np.empty(len(xs), dtype=np.float64)
            for j, v in enumerate(xs):
                try:
                    out[j] = np.nan if v is None else float(v)
                except Exception:
                    out[j] = np.nan
            return out

        gpoa_np = list_to_np_nan(g_poa_used)
        tamb_np = list_to_np_nan(temp_air)
        pac_real_np = list_to_np_nan(p_ac_w)

        sig = inspect.signature(expected_and_mismatch)
        kwargs: Dict[str, Any] = dict(
            g_poa=gpoa_np,
            tamb_c=tamb_np,
            pac_real_w=pac_real_np,
            module=mod,
            plant=pl,
            g_min_valid=0.0,
            n_points=60,
            eps_w=50.0,
        )
        if "dt_minutes" in sig.parameters:
            kwargs["dt_minutes"] = 15.0
        if "window_minutes" in sig.parameters:
            kwargs["window_minutes"] = 60.0
        if "times_utc" in sig.parameters:
            kwargs["times_utc"] = times_utc

        out_model = expected_and_mismatch(**kwargs) or {}
        pac_expected = out_model.get("pac_expected_w")
        mismatch = out_model.get("mismatch_rel")
        valid_model_np = out_model.get("valid")
        tcell_np = out_model.get("tcell_c")

        if pac_expected is None:
            return _json_response_strict({"ok": False, "error": "power_model não retornou pac_expected_w."}, status=500)

        pac_model_w = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(pac_expected, dtype=float).tolist()]

        if mismatch is None:
            eps = 50.0
            mm: List[Optional[float]] = []
            for pr, pm in zip(p_ac_w, pac_model_w):
                if pr is None or pm is None:
                    mm.append(None)
                    continue
                den = max(abs(pm), eps)
                mm.append((float(pr) - float(pm)) / float(den))
            mismatch_rel = mm
        else:
            mismatch_rel = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(mismatch, dtype=float).tolist()]

        if valid_model_np is None:
            valid_model = [False if (m is None) else True for m in mismatch_rel]
        else:
            valid_model = [bool(v) for v in list(np.asarray(valid_model_np, dtype=bool).tolist())]

        if tcell_np is None:
            tcell_c = [None] * n
        else:
            tcell_c = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(tcell_np, dtype=float).tolist()]

    except Exception as e:
        logger.exception("Falha no power_model (mismatch_fdd_api) plant_id=%s", plant_id)
        return _json_response_strict({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)

    out_cls = classify_mismatch_series(
        times_utc=times_utc,
        mismatch_rel=mismatch_rel,
        g_poa_wm2=g_poa_used,
        valid=valid_model,
        thresholds=thr,
    )
    codes = out_cls["codes"]
    labels = out_cls["labels"]

    persist = (data.get("persist") or "").strip().lower() in ("1", "true", "yes", "on")
    upsert = None
    if persist:
        upsert = _upsert_diag15m(
            plant=plant,
            times_utc=times_utc,
            codes=codes,
            labels=labels,
            valid=valid_model,
            g_poa=g_poa_used,
            tcell_c=tcell_c,
            pac_real_w=p_ac_w,
            pac_model_w=pac_model_w,
            mismatch_rel=mismatch_rel,
        )

    def _mean_abs_valid() -> Optional[float]:
        buf: List[float] = []
        for v_ok, mi, code in zip(valid_model, mismatch_rel, codes):
            if (not v_ok) or int(code) == CODE_INVALID:
                continue
            if mi is None:
                continue
            buf.append(abs(float(mi)))
        return (sum(buf) / len(buf)) if buf else None

    payload = {
        "ok": True,
        "plant": {"id": plant.id, "nome": plant.nome, "tz": tz_name},
        "range": {
            "start": d0.isoformat(),
            "end": d1.isoformat(),
            "start_utc": dt0_utc.isoformat(),
            "end_utc_excl": dt1_utc.isoformat(),
            "source_meteo": src_meteo,
            "selected_sources": selected_sources,
        },
        "sources": {
            "source_meteo": src_meteo,
            "source_oper_list": source_oper_list,
            "selected_sources": selected_sources,
        },
        "x_local": x_local,
        "x_utc": x_utc,
        "series": {
            # aliases esperados pelo front
            "t_local": x_local,
            "t_utc": x_utc,
            "g_poa": g_poa_used,
            "labels": labels,
            "codes": codes,

            # extras p/ heatmap sem Date()
            "hm_day_local": hm_day_local,
            "hm_minute_local": hm_minute_local,

            # mantém nomes atuais
            "x_local": x_local,
            "x_utc": x_utc,

            "p_ac_w": p_ac_w,
            "p_ac_real_w": p_ac_w,
            "p_dc_w": p_dc_w,
            "e_ac_wh_15": e_ac_wh_15,
            "v_dc_v": v_dc_v,
            "i_dc_a": i_dc_a,
            "v_ac_v": v_ac_v,
            "i_ac_a": i_ac_a,
            "inv_coverage": inv_cov,
            "flag_inv_missing": flag_inv_missing,

            "gti": gti,
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "wind_speed": wind_speed,
            "rh": rh,
            "flag_meteo_missing": flag_meteo_missing,

            "p_ac_model_w": pac_model_w,
            "tcell_c": tcell_c,
            "mismatch_rel": mismatch_rel,

            "g_poa_used": g_poa_used,
            "valid_model": valid_model,
            "valid": valid_model,

            "rca_code": codes,
            "rca_label": labels,
        },
        "series_by_source": series_by_source,
        "summary": {
            "counts": out_cls.get("summary", {}),
            "events": out_cls.get("events", []),
            "mean_abs_mismatch_valid": _mean_abs_valid(),
            "n_points": n,
        },
        "thresholds": out_cls.get("thresholds", {}),
        "persist": upsert,
    }

    return _json_response_strict(payload)

