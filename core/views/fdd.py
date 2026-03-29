# core/views/fdd.py
from __future__ import annotations

from core.views._imports import *  # mantém o teu padrão
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import inspect
import math

from zoneinfo import ZoneInfo
from django.db.models import Count
from django.db import transaction
from django.utils import timezone
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods
from django.contrib.auth.decorators import login_required

from core.services.fdd_mismatch import (
    MismatchThresholds,
    classify_mismatch_series,  # legado (opcional)
    CODE_INVALID,
)

from core.models import (
    PVPlant,
    PVPlantMergedRecord15m,
    PlantDiagnostic15m,
)
from core.services.fdd.reliability import (
    compute_data_reliability,
    compute_detection_confidence,
    compute_diagnosis_confidence,
)

try:
    from core.services.fdd.report_pdf import build_mismatch_pdf_report  # type: ignore
except Exception:
    build_mismatch_pdf_report = None  # type: ignore

logger = logging.getLogger(__name__)

MISMATCH_VERSION_SUMMARY = {
    "detector_version": "mismatch_runtime_v1",
    "event_classifier_version": None,
    "trained_model_version": None,
    "detector_note": "Detector runtime desta tela: modelo físico de potência + limiares/heurísticas de mismatch configurados na UI.",
    "event_classifier_note": "Não aplicável no dashboard Mismatch.",
    "trained_model_note": "Não aplicável no dashboard Mismatch.",
}


# ----------------------------
# helpers source (MPPT vs AGG)
# ----------------------------
def _is_mppt_source(src: str) -> bool:
    u = (src or "").upper()
    return "|MPPT" in u


def _is_agg_source(src: str) -> bool:
    """
    AGG:
      - sem "|" (ex: SHINEMONITOR)
      - OU termina com |AGG
    """
    s = (src or "").strip()
    if not s:
        return False
    u = s.upper()
    if "|" not in u:
        return True
    if u.endswith("|AGG"):
        return True
    return False


# ----------------------------
# JSON strict/robusto
# ----------------------------
def _json_sanitize(x: Any) -> Any:
    """Converte NaN/Inf -> None recursivamente (para allow_nan=False)."""
    if x is None:
        return None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    try:
        import numpy as np  # type: ignore
        if isinstance(x, np.generic):
            v = x.item()
            if isinstance(v, float) and (not math.isfinite(v)):
                return None
            return _json_sanitize(v)
        if isinstance(x, np.ndarray):
            return [_json_sanitize(v) for v in x.tolist()]
    except Exception:
        pass
    if is_dataclass(x):
        return _json_sanitize(asdict(x))
    return x


def _json_response_strict(payload: Any, *, status: int = 200) -> JsonResponse:
    safe = isinstance(payload, dict)
    payload = _json_sanitize(payload)
    return JsonResponse(
        payload,
        status=status,
        safe=safe,
        json_dumps_params={"ensure_ascii": False, "allow_nan": False},
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
        if not math.isfinite(v):
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


def _pick_best_sources(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Tuple[Optional[str], Optional[str]]:
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
    FIX:
      - bulk_create/bulk_update NÃO disparam auto_now/auto_now_add
      - setamos updated_at/created_at manualmente.
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


@require_GET
@login_required
def mismatch_fdd_view(request: HttpRequest):
    """Página: Heatmap tipo GitHub (dia x bin) + drawer com dump ao clicar."""
    qs = (
        PVPlant.objects.all().order_by("nome")
        if request.user.is_superuser
        else PVPlant.objects.filter(owner=request.user).order_by("nome")
    )
    plants = list(qs)

    d_end = date.today()
    d_start = d_end - timedelta(days=7)

    plant_id = request.GET.get("plant_id") or request.GET.get("pk") or request.GET.get("plant_pk")
    if not plant_id and plants:
        plant_id = str(plants[0].id)

    return render(
        request,
        "dashboard/mismatch_fdd.html",  # ✅ alinhado com o template que você mandou
        {
            "plants": plants,
            "plant_id": plant_id,
            "start": request.GET.get("start") or d_start.isoformat(),
            "end": request.GET.get("end") or d_end.isoformat(),
            "dt_minutes": int(float(request.GET.get("dt_minutes") or 15)),
            "warn_abs": float((request.GET.get("warn_abs") or 0.35)),
            "fault_abs": float((request.GET.get("fault_abs") or 0.90)),
            "gpoa_min": float((request.GET.get("gpoa_min") or 50)),
            "pmin_w": float((request.GET.get("pmin_w") or 0)),
            "api_url": reverse("mismatch_fdd_api"),
            "export_pdf_url": reverse("mismatch_fdd_export_pdf"),
            "version_summary": MISMATCH_VERSION_SUMMARY,
        },
    )


@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_api(request: HttpRequest) -> JsonResponse:
    data = request.POST if request.method == "POST" else request.GET

    try:
        plant_id = int((data.get("plant_id") or data.get("plant_pk") or data.get("pk") or "0").strip())
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

    # available source_oper no range (para dropdown)
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

    # ----------------------------
    # Query merged_15m (values dinâmico — não quebra se campo não existir)
    # ----------------------------
    field_names = {ff.name for ff in PVPlantMergedRecord15m._meta.get_fields() if hasattr(ff, "name")}

    base_values = [
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
        "meteo_qc_score",
        "flag_meteo_low_confidence",
        "flag_meteo_interpolated",
        "flag_meteo_outlier",
        "flag_meteo_artifact",
        "flag_meteo_missing",
    ]

    optional_values = [
        # MPPT (se existirem no model)
        "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
        "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
        # alarmes (se existirem)
        "alarm_code", "alarm_sev",
    ]

    values_fields = list(base_values)
    for k in optional_values:
        if k in field_names:
            values_fields.append(k)

    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=src_meteo,
            source_oper__in=selected_sources,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .order_by("ts_utc", "source_oper")
        .values(*values_fields)
    )
    rows = list(qs)
    if not rows:
        return _json_response_strict({"ok": False, "error": "Sem registros no range para as fontes selecionadas."}, status=404)

    # ----------------------------
    # Agrupa por timestamp e source
    # ----------------------------
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        ts = r["ts_utc"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt_tz.utc)
        src = r.get("source_oper") or ""
        if not src:
            continue
        per_ts.setdefault(ts, {})[src] = r

    times_utc = sorted(per_ts.keys())
    n = len(times_utc)
    if n == 0:
        return _json_response_strict({"ok": False, "error": "Sem timestamps válidos no range."}, status=404)

    # ----------------------------
    # Séries agregadas (POLICY: prefer ΣMPPT, fallback AGG)
    # ----------------------------
    p_ac_w: List[Optional[float]] = [None] * n
    p_dc_w: List[Optional[float]] = [None] * n
    e_ac_wh_15: List[Optional[float]] = [None] * n
    v_dc_v: List[Optional[float]] = [None] * n
    i_dc_a: List[Optional[float]] = [None] * n
    v_ac_v: List[Optional[float]] = [None] * n
    i_ac_a: List[Optional[float]] = [None] * n
    inv_cov: List[Optional[float]] = [None] * n

    # ✅ flags: "all missing" e "partial missing" (evita cinza indevido)
    flag_inv_missing_all: List[bool] = [False] * n
    flag_inv_missing_partial: List[bool] = [False] * n

    # meteo (um row por ts)
    gti: List[Optional[float]] = [None] * n
    ghi: List[Optional[float]] = [None] * n
    dni: List[Optional[float]] = [None] * n
    dhi: List[Optional[float]] = [None] * n
    temp_air: List[Optional[float]] = [None] * n
    wind_speed: List[Optional[float]] = [None] * n
    rh: List[Optional[float]] = [None] * n
    meteo_qc_score: List[Optional[float]] = [None] * n
    flag_meteo_low_confidence: List[bool] = [False] * n
    flag_meteo_interpolated: List[bool] = [False] * n
    flag_meteo_outlier: List[bool] = [False] * n
    flag_meteo_artifact: List[bool] = [False] * n
    flag_meteo_missing: List[bool] = [False] * n

    # comparativos (debug / UX)
    p_ac_mppt_sum_w: List[Optional[float]] = [None] * n
    p_ac_agg_w: List[Optional[float]] = [None] * n
    policy_used: List[str] = [""] * n

    # por source (inclui MPPT + alarmes)
    series_by_source: Dict[str, Dict[str, List[Any]]] = {
        src: {
            "p_ac_w": [None] * n,
            "p_dc_w": [None] * n,
            "e_ac_wh_15": [None] * n,
            "v_dc_v": [None] * n,
            "i_dc_a": [None] * n,
            "v_ac_v": [None] * n,
            "i_ac_a": [None] * n,

            "mppt1_vdc_v": [None] * n, "mppt2_vdc_v": [None] * n, "mppt3_vdc_v": [None] * n, "mppt4_vdc_v": [None] * n,
            "mppt1_idc_a": [None] * n, "mppt2_idc_a": [None] * n, "mppt3_idc_a": [None] * n, "mppt4_idc_a": [None] * n,

            "alarm_code": [None] * n,
            "alarm_sev": [None] * n,

            "inv_coverage": [None] * n,
            "flag_inv_missing": [None] * n,
        }
        for src in selected_sources
    }

    for i, ts in enumerate(times_utc):
        by_src = per_ts.get(ts, {})

        present = [s for s in selected_sources if s in by_src]
        present_mppt = [s for s in present if _is_mppt_source(s)]
        present_agg = [s for s in present if _is_agg_source(s)]

        if present_mppt:
            chosen = present_mppt
            policy = "mppt_sum"
        elif present_agg:
            chosen = present_agg
            policy = "agg_fallback"
        else:
            chosen = present
            policy = "any_fallback"

        policy_used[i] = policy

        # meteo: usa qualquer row do timestamp (primeiro válido)
        first_row: Optional[Dict[str, Any]] = None
        for s0 in present:
            rr = by_src.get(s0)
            if rr is not None:
                first_row = rr
                break

        if first_row is not None:
            gti[i] = _as_float(first_row.get("gti"))
            ghi[i] = _as_float(first_row.get("ghi"))
            dni[i] = _as_float(first_row.get("dni"))
            dhi[i] = _as_float(first_row.get("dhi"))
            temp_air[i] = _as_float(first_row.get("temp_air"))
            wind_speed[i] = _as_float(first_row.get("wind_speed"))
            rh[i] = _as_float(first_row.get("rh"))
            meteo_qc_score[i] = _as_float(first_row.get("meteo_qc_score"))
            flag_meteo_low_confidence[i] = bool(first_row.get("flag_meteo_low_confidence") or False)
            flag_meteo_interpolated[i] = bool(first_row.get("flag_meteo_interpolated") or False)
            flag_meteo_outlier[i] = bool(first_row.get("flag_meteo_outlier") or False)
            flag_meteo_artifact[i] = bool(first_row.get("flag_meteo_artifact") or False)
            flag_meteo_missing[i] = bool(first_row.get("flag_meteo_missing") or False)

        # preenche series_by_source
        for src in present:
            r = by_src.get(src)
            if r is None:
                continue

            pac = _as_float(r.get("p_ac_w"))
            pdc = _as_float(r.get("p_dc_w"))
            e15 = _as_float(r.get("e_ac_wh_15"))
            vdc = _as_float(r.get("v_dc_v"))
            idc = _as_float(r.get("i_dc_a"))
            vac = _as_float(r.get("v_ac_v"))
            iac = _as_float(r.get("i_ac_a"))
            cov = _as_float(r.get("inv_coverage"))
            inv_miss = bool(r.get("flag_inv_missing") or False)

            sb = series_by_source.get(src)
            if sb is not None:
                sb["p_ac_w"][i] = pac
                sb["p_dc_w"][i] = pdc
                sb["e_ac_wh_15"][i] = e15
                sb["v_dc_v"][i] = vdc
                sb["i_dc_a"][i] = idc
                sb["v_ac_v"][i] = vac
                sb["i_ac_a"][i] = iac
                sb["inv_coverage"][i] = cov
                sb["flag_inv_missing"][i] = inv_miss

                if "mppt1_vdc_v" in values_fields:
                    sb["mppt1_vdc_v"][i] = _as_float(r.get("mppt1_vdc_v"))
                    sb["mppt2_vdc_v"][i] = _as_float(r.get("mppt2_vdc_v"))
                    sb["mppt3_vdc_v"][i] = _as_float(r.get("mppt3_vdc_v"))
                    sb["mppt4_vdc_v"][i] = _as_float(r.get("mppt4_vdc_v"))
                    sb["mppt1_idc_a"][i] = _as_float(r.get("mppt1_idc_a"))
                    sb["mppt2_idc_a"][i] = _as_float(r.get("mppt2_idc_a"))
                    sb["mppt3_idc_a"][i] = _as_float(r.get("mppt3_idc_a"))
                    sb["mppt4_idc_a"][i] = _as_float(r.get("mppt4_idc_a"))

                if "alarm_code" in values_fields:
                    sb["alarm_code"][i] = r.get("alarm_code")
                    sb["alarm_sev"][i] = r.get("alarm_sev")

        # agregados de comparação
        pac_mppt = _sum_none([_as_float(by_src[s].get("p_ac_w")) for s in present_mppt]) if present_mppt else None
        pac_agg = _sum_none([_as_float(by_src[s].get("p_ac_w")) for s in present_agg]) if present_agg else None
        p_ac_mppt_sum_w[i] = pac_mppt
        p_ac_agg_w[i] = pac_agg

        # agregação principal (chosen)
        pac_l = [_as_float(by_src[s].get("p_ac_w")) for s in chosen] if chosen else []
        pdc_l = [_as_float(by_src[s].get("p_dc_w")) for s in chosen] if chosen else []
        e15_l = [_as_float(by_src[s].get("e_ac_wh_15")) for s in chosen] if chosen else []
        vdc_l = [_as_float(by_src[s].get("v_dc_v")) for s in chosen] if chosen else []
        idc_l = [_as_float(by_src[s].get("i_dc_a")) for s in chosen] if chosen else []
        vac_l = [_as_float(by_src[s].get("v_ac_v")) for s in chosen] if chosen else []
        iac_l = [_as_float(by_src[s].get("i_ac_a")) for s in chosen] if chosen else []
        cov_l = [_as_float(by_src[s].get("inv_coverage")) for s in chosen] if chosen else []

        miss_flags = [bool(by_src[s].get("flag_inv_missing") or False) for s in chosen] if chosen else []

        p_ac_w[i] = _sum_none(pac_l)
        p_dc_w[i] = _sum_none(pdc_l)
        e_ac_wh_15[i] = _sum_none(e15_l)
        v_dc_v[i] = _mean_none(vdc_l)
        i_dc_a[i] = _sum_none(idc_l)
        v_ac_v[i] = _mean_none(vac_l)
        i_ac_a[i] = _sum_none(iac_l)
        inv_cov[i] = _mean_none(cov_l)

        if not miss_flags:
            flag_inv_missing_all[i] = True
            flag_inv_missing_partial[i] = False
        else:
            all_miss = all(miss_flags)
            any_miss = any(miss_flags)
            flag_inv_missing_all[i] = bool(all_miss)
            flag_inv_missing_partial[i] = bool(any_miss and (not all_miss))

    # timestamps locais/utc (strings)
    x_local_dt = [t.astimezone(tz) for t in times_utc]
    x_local = [t.isoformat() for t in x_local_dt]
    x_utc = [t.astimezone(dt_tz.utc).isoformat() for t in times_utc]

    hm_day_local = [t.date().isoformat() for t in x_local_dt]
    hm_minute_local = [t.hour * 60 + t.minute for t in x_local_dt]

    # ----------------------------
    # thresholds (UI)
    # ----------------------------
    def _gf(key: str, default: float) -> float:
        raw = (data.get(key) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(str(raw).replace(",", "."))
        except Exception:
            return float(default)

    def _gi(key: str, default: int) -> int:
        raw = (data.get(key) or "").strip()
        if not raw:
            return int(default)
        try:
            return int(float(str(raw).replace(",", ".")))
        except Exception:
            return int(default)

    gpoa_gate = _gf("gpoa_gate", _gf("gpoa_min", 50.0))
    pmin_w = _gf("pmin_w", 0.0)

    thr = MismatchThresholds(
        gpoa_gate_wm2=gpoa_gate,
        warn_abs=_gf("warn_abs", 0.35),
        fault_abs=_gf("fault_abs", 0.90),
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

    # ----------------------------
    # power_model: P_expected + mismatch + valid + (tcell)
    # ----------------------------
    try:
        import numpy as np
        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
            transpose_ghi_to_poa_isotropic,
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

        gti_np = list_to_np_nan(gti)
        ghi_np = list_to_np_nan(ghi)
        dni_np = list_to_np_nan(dni)
        dhi_np = list_to_np_nan(dhi)

        mask_gti = np.isfinite(gti_np)
        has_any_gti = bool(mask_gti.any())

        ghi_arg = ghi_np if np.isfinite(ghi_np).any() else None
        dni_arg = dni_np if np.isfinite(dni_np).any() else None
        dhi_arg = dhi_np if np.isfinite(dhi_np).any() else None

        # transposição (condicional por assinatura)
        g_poa_transpo = None
        if ghi_arg is not None:
            lat = getattr(pl, "lat_deg", None)
            lon = getattr(pl, "lon_deg", None)
            tilt = getattr(pl, "tilt_deg", None)
            azs = getattr(pl, "azimuth_deg", None)
            if None not in (lat, lon, tilt, azs):
                trans_sig = inspect.signature(transpose_ghi_to_poa_isotropic)
                trans_kwargs = dict(
                    ghi=ghi_arg,
                    dhi=dhi_arg,
                    dni=dni_arg,
                    times_utc=times_utc,
                    lat_deg=float(lat),
                    lon_deg=float(lon),
                    tilt_deg=float(tilt),
                    azimuth_deg=float(azs),
                    albedo=float(getattr(pl, "albedo", 0.20) or 0.20),
                )
                if "times_shift_minutes" in trans_sig.parameters:
                    trans_kwargs["times_shift_minutes"] = float(getattr(pl, "meteo_time_shift_minutes", 0.0) or 0.0)

                trans = transpose_ghi_to_poa_isotropic(**trans_kwargs)
                g_poa_transpo = np.asarray(trans.get("g_poa"), dtype=float)

        if has_any_gti:
            if g_poa_transpo is not None and g_poa_transpo.size == gti_np.size:
                g_poa_used_np = np.where(mask_gti, gti_np, g_poa_transpo)
            else:
                g_poa_used_np = gti_np
        else:
            if g_poa_transpo is not None and g_poa_transpo.size == gti_np.size:
                g_poa_used_np = g_poa_transpo
            else:
                g_poa_used_np = ghi_arg if ghi_arg is not None else np.full_like(gti_np, np.nan)

        g_poa_used = [None if (not np.isfinite(v)) else float(v) for v in g_poa_used_np.tolist()]

        tamb_np = list_to_np_nan(temp_air)
        pac_real_np = list_to_np_nan(p_ac_w)

        sig = inspect.signature(expected_and_mismatch)

        kwargs: Dict[str, Any] = dict(
            g_poa=g_poa_used_np,  # ✅ sempre passa POA usado
            tamb_c=tamb_np,
            pac_real_w=pac_real_np,
            module=mod,
            plant=pl,
            g_min_valid=0.0,
            n_points=60,
            eps_w=50.0,
        )

        if "times_utc" in sig.parameters:
            kwargs["times_utc"] = times_utc
        if "dt_minutes" in sig.parameters:
            kwargs["dt_minutes"] = 15.0
        if "window_minutes" in sig.parameters:
            kwargs["window_minutes"] = 60.0

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

    # ----------------------------
    # PIPELINE: Detecção + RCA
    # ----------------------------
    coarse_period = [False] * n
    fine_period = [False] * n
    meteo_quality_ok = [False] * n
    irradiance_tier = ["N"] * n
    rca: Dict[str, Any] = {}

    def _pick_diag_row_for_ts(ts_utc: datetime) -> Optional[Dict[str, Any]]:
        by_src = per_ts.get(ts_utc, {})
        if not by_src:
            return None
        for s in selected_sources:
            rr = by_src.get(s)
            if rr is not None:
                return rr
        try:
            return next(iter(by_src.values()))
        except Exception:
            return None

    use_legacy = (data.get("legacy") or data.get("use_legacy") or "").strip().lower() in ("1", "true", "yes", "on")

    # ✅ gate usa flag_inv_missing_all (não “any”), para não matar pontos parcialmente disponíveis
    base_gate: List[bool] = []
    for i in range(n):
        gp = g_poa_used[i]
        pr = p_ac_w[i]
        ok = bool(valid_model[i])
        ok = ok and (gp is not None) and (float(gp) >= float(gpoa_gate))
        ok = ok and (pr is not None) and (float(pr) >= float(pmin_w))
        ok = ok and (not bool(flag_meteo_missing[i]))
        ok = ok and (not bool(flag_inv_missing_all[i]))
        base_gate.append(ok)

    if use_legacy:
        out_cls = classify_mismatch_series(
            times_utc=times_utc,
            mismatch_rel=mismatch_rel,
            g_poa_wm2=g_poa_used,
            valid=base_gate,
            thresholds=thr,
        )
        codes = [int(c) for c in out_cls["codes"]]
        labels = [str(x) for x in out_cls["labels"]]
        valid_period = [bool(v) for v in base_gate]
        anomaly = [False] * n
        stable_sky = [False] * n
        coarse_period = valid_period[:]
        fine_period = valid_period[:]
        meteo_quality_ok = [not bool(v) for v in flag_meteo_missing]
        irradiance_tier = ["C" if bool(v) else "N" for v in valid_period]
        det_dbg = {}
        rca_dbg = {}
        pipeline_name = "legacy_mismatch_classifier"

    else:
        try:
            from core.services.fdd.detection import DetectionParams, detect_anomalies
            from core.services.fdd.rca import RCAParams, diagnose_rca_series
        except Exception as e:
            return _json_response_strict(
                {"ok": False, "error": f"ImportError fdd/detection ou fdd/rca: {type(e).__name__}: {e}"},
                status=500,
            )

        # ----------------------------
        # DetectionParams: compatível com API antiga e nova
        # ----------------------------
        det_sig = inspect.signature(DetectionParams)
        det_param_names = set(det_sig.parameters.keys())

        if "gpoa_gate_wm2" in det_param_names:
            # API antiga
            det_params = DetectionParams(
                gpoa_gate_wm2=float(gpoa_gate),
                stable_cv_max=_gf("stable_cv_max", 0.08),
                stable_window_points=_gi("stable_window_points", 6),
                ewma_lambda=_gf("ewma_lambda", 0.20),
                ewma_L=_gf("ewma_L", 3.0),
                cusum_k=_gf("cusum_k", 0.50),
                cusum_h=_gf("cusum_h", 8.0),
                min_baseline_points=_gi("min_baseline_points", 24),
                inv_cov_min=_gf("inv_cov_min", 0.30),
            )
        else:
            # API nova híbrida
            det_params = DetectionParams(
                sun_available_gpoa_wm2=_gf("sun_available_gpoa_wm2", max(150.0, float(gpoa_gate))),
                coarse_diag_gpoa_wm2=_gf("coarse_diag_gpoa_wm2", max(700.0, float(gpoa_gate))),
                fine_diag_gpoa_wm2=_gf("fine_diag_gpoa_wm2", max(800.0, float(gpoa_gate))),
                stable_cv_max=_gf("stable_cv_max", 0.08),
                stable_ramp_max_wm2=_gf("stable_ramp_max_wm2", 120.0),
                stable_window_points=_gi("stable_window_points", 6),
                ewma_lambda=_gf("ewma_lambda", 0.20),
                ewma_L=_gf("ewma_L", 3.0),
                cusum_k=_gf("cusum_k", 0.50),
                cusum_h=_gf("cusum_h", 8.0),
                min_baseline_points=_gi("min_baseline_points", 24),
                inv_cov_min=_gf("inv_cov_min", 0.30),
            )

        det = detect_anomalies(
            mismatch_rel=mismatch_rel,
            g_poa_wm2=g_poa_used,
            valid_model=base_gate,
            flag_meteo_missing=flag_meteo_missing,
            flag_meteo_low_confidence=flag_meteo_low_confidence,
            flag_meteo_interpolated=flag_meteo_interpolated,
            flag_inv_missing=flag_inv_missing_all,
            inv_coverage=inv_cov,
            params=det_params,
        ) or {}

        valid_period = [bool(v) for v in (det.get("valid_period") or base_gate)]
        anomaly = [bool(v) for v in (det.get("anomaly") or [False] * n)]
        stable_sky = [bool(v) for v in (det.get("stable_sky") or [False] * n)]

        # Campos novos da API híbrida
        coarse_period = [bool(v) for v in (det.get("coarse_period") or valid_period)]
        fine_period = [bool(v) for v in (det.get("fine_period") or [False] * n)]
        meteo_quality_ok = [bool(v) for v in (det.get("meteo_quality_ok") or stable_sky)]
        irradiance_tier = [str(v) for v in (det.get("irradiance_tier") or ["N"] * n)]

        det_dbg = {
            "z": det.get("z"),
            "ewma_z": det.get("ewma_z"),
            "cusum": det.get("cusum"),
            "baseline": det.get("baseline"),
            "coarse_period": coarse_period,
            "fine_period": fine_period,
            "meteo_quality_ok": meteo_quality_ok,
            "meteo_qc_score": meteo_qc_score,
            "flag_meteo_low_confidence": flag_meteo_low_confidence,
            "flag_meteo_interpolated": flag_meteo_interpolated,
            "flag_meteo_outlier": flag_meteo_outlier,
            "flag_meteo_artifact": flag_meteo_artifact,
            "irradiance_tier": irradiance_tier,
        }

        pac_cap_w = None
        try:
            inv_obj = getattr(details, "inverter", None)
            for attr in ("pac_nom_w", "p_ac_nom_w", "rated_power_w", "pnom_w", "pac_nom_kw", "rated_power_kw"):
                vv = getattr(inv_obj, attr, None) if inv_obj is not None else None
                if vv is None:
                    continue
                pac_cap_w = float(vv) * (1000.0 if str(attr).endswith("_kw") else 1.0)
                break
        except Exception:
            pac_cap_w = None

        # ----------------------------
        # RCAParams: compatível com API antiga e nova
        # ----------------------------
        rca_sig = inspect.signature(RCAParams)
        rca_param_names = set(rca_sig.parameters.keys())

        if "warn_abs" in rca_param_names and "fault_abs" in rca_param_names:
            # API antiga
            rca_params = RCAParams(
                warn_abs=float(thr.warn_abs),
                fault_abs=float(thr.fault_abs),
                min_baseline_points=_gi("rca_min_baseline_points", 24),
            )
        else:
            # API nova híbrida
            rca_params = RCAParams(
                sun_available_gpoa_wm2=_gf("sun_available_gpoa_wm2", max(150.0, float(gpoa_gate))),
                expected_power_min_w=float(pmin_w),
                zero_abs_w=_gf("zero_abs_w", 100.0),
                zero_rel_model=_gf("zero_rel_model", 0.05),
                degraded_rel=_gf("degraded_rel", 0.25),
                severe_rel=_gf("severe_rel", 0.50),
                low_i_ratio_warn=_gf("low_i_ratio_warn", 0.35),
                low_i_ratio_crit=_gf("low_i_ratio_crit", 0.15),
                low_v_ratio_warn=_gf("low_v_ratio_warn", 0.80),
                low_v_ratio_crit=_gf("low_v_ratio_crit", 0.60),
                vac_low_ratio=_gf("vac_low_ratio", 0.90),
                vac_high_ratio=_gf("vac_high_ratio", 1.10),
                vac_abs_margin_v=_gf("vac_abs_margin_v", 10.0),
                freq_abs_tol_hz=_gf("freq_abs_tol_hz", 1.0),
                clip_margin=_gf("clip_margin", 0.98),
                clip_model_margin=_gf("clip_model_margin", 1.02),
                min_baseline_points=_gi("rca_min_baseline_points", 24),
            )

        # ----------------------------
        # diagnose_rca_series: compatível com API antiga e nova
        # ----------------------------
        diag_sig = inspect.signature(diagnose_rca_series)
        diag_param_names = set(diag_sig.parameters.keys())

        diag_kwargs = dict(
            anomaly=anomaly,
            valid_period=valid_period,
            mismatch_rel=mismatch_rel,
            v_dc_v=v_dc_v,
            i_dc_a=i_dc_a,
            pac_real_w=p_ac_w,
            pac_model_w=pac_model_w,
            flag_inv_missing=flag_inv_missing_all,
            flag_meteo_missing=flag_meteo_missing,
            inv_coverage=inv_cov,
            pac_cap_w=pac_cap_w,
            params=rca_params,
        )

        if "g_poa_wm2" in diag_param_names:
            diag_kwargs["g_poa_wm2"] = g_poa_used
        if "coarse_period" in diag_param_names:
            diag_kwargs["coarse_period"] = coarse_period
        if "fine_period" in diag_param_names:
            diag_kwargs["fine_period"] = fine_period
        if "meteo_quality_ok" in diag_param_names:
            diag_kwargs["meteo_quality_ok"] = meteo_quality_ok
        if "irradiance_tier" in diag_param_names:
            diag_kwargs["irradiance_tier"] = irradiance_tier
        if "v_ac_v" in diag_param_names:
            diag_kwargs["v_ac_v"] = v_ac_v
        if "i_ac_a" in diag_param_names:
            diag_kwargs["i_ac_a"] = i_ac_a

        # frequência pode nem existir nesta tela; envia nulo se necessário
        if "freq_hz" in diag_param_names:
            diag_kwargs["freq_hz"] = [None] * n

        def _pick_diag_row_for_ts(ts_utc: datetime) -> Optional[Dict[str, Any]]:
            by_src = per_ts.get(ts_utc, {})
            if not by_src:
                return None
            for s in selected_sources:
                rr = by_src.get(s)
                if rr is not None:
                    return rr
            try:
                return next(iter(by_src.values()))
            except Exception:
                return None

        if "alarm_code" in diag_param_names:
            alarm_code_series = []
            for ts_utc in times_utc:
                row = _pick_diag_row_for_ts(ts_utc)
                alarm_code_series.append(None if row is None else row.get("alarm_code"))
            diag_kwargs["alarm_code"] = alarm_code_series

        if "alarm_sev" in diag_param_names:
            alarm_sev_series = []
            for ts_utc in times_utc:
                row = _pick_diag_row_for_ts(ts_utc)
                alarm_sev_series.append(None if row is None else row.get("alarm_sev"))
            diag_kwargs["alarm_sev"] = alarm_sev_series

        rca = diagnose_rca_series(**diag_kwargs) or {}

        rca_codes_raw = rca.get("codes") or [0] * n
        rca_labels_raw = rca.get("labels") or ["normal"] * n

        codes: List[int] = [CODE_INVALID] * n
        labels: List[str] = ["invalid"] * n

        for i in range(n):
            if not valid_period[i]:
                codes[i] = CODE_INVALID
                labels[i] = "invalid"
                continue
            if not anomaly[i]:
                codes[i] = 0
                labels[i] = "normal"
                continue
            try:
                c = int(rca_codes_raw[i])
            except Exception:
                c = 2
            lbl = str(rca_labels_raw[i] or "anom")
            codes[i] = c
            labels[i] = lbl

        rca_dbg = {"baseline": rca.get("baseline")}
        pipeline_name = "ewma_cusum_detection + rca_patterns"

    # ----------------------------
    # Camada explícita de confiabilidade (runtime da view)
    # ----------------------------
    diag_state_labels = [str(v) for v in (rca.get("state_labels") or (["unknown"] * n))]
    diag_domain_labels = [str(v) for v in (rca.get("domain_labels") or (["unknown"] * n))]
    diag_diagnosis_labels = [str(v) for v in (rca.get("diagnosis_labels") or labels)]
    diag_base_conf = list(rca.get("diagnosis_confidence") or [None] * n)
    diag_direct_grid = [bool(v) for v in (rca.get("direct_grid_evidence") or ([False] * n))]
    diag_zero_inj = [bool(v) for v in (rca.get("zero_injection_flag") or ([False] * n))]
    diag_evidence_json = list(rca.get("evidence_json") or ([{}] * n))

    data_reliability_score: List[Optional[float]] = [None] * n
    data_reliability_level: List[str] = [""] * n
    detection_confidence_score: List[Optional[float]] = [None] * n
    detection_confidence_level: List[str] = [""] * n
    diagnosis_confidence_score: List[Optional[float]] = [None] * n
    diagnosis_confidence_level: List[str] = [""] * n
    confidence_notes: List[Dict[str, Any]] = [{} for _ in range(n)]

    for i, ts_utc in enumerate(times_utc):
        row_ref = _pick_diag_row_for_ts(ts_utc) or {}
        mismatch_i = mismatch_rel[i]
        pac_real_i = p_ac_w[i]
        pac_model_i = pac_model_w[i]
        ewma_i = None
        cusum_i = None
        try:
            ewma_seq = det_dbg.get("ewma_z") if isinstance(det_dbg, dict) else None
            if isinstance(ewma_seq, list) and i < len(ewma_seq):
                ewma_i = ewma_seq[i]
        except Exception:
            ewma_i = None
        try:
            cusum_seq = det_dbg.get("cusum") if isinstance(det_dbg, dict) else None
            if isinstance(cusum_seq, list) and i < len(cusum_seq):
                cusum_i = cusum_seq[i]
        except Exception:
            cusum_i = None

        row_runtime = dict(row_ref)
        row_runtime.setdefault("flag_inv_missing", flag_inv_missing_all[i])
        row_runtime.setdefault("flag_low_coverage", flag_inv_missing_partial[i])
        row_runtime.setdefault("flag_meteo_missing", flag_meteo_missing[i])
        row_runtime.setdefault("flag_meteo_low_confidence", flag_meteo_low_confidence[i])
        row_runtime.setdefault("flag_meteo_interpolated", flag_meteo_interpolated[i])
        row_runtime.setdefault("flag_meteo_outlier", flag_meteo_outlier[i])
        row_runtime.setdefault("flag_meteo_artifact", flag_meteo_artifact[i])
        row_runtime.setdefault("inv_coverage", inv_cov[i])
        row_runtime.setdefault("meteo_qc_score", meteo_qc_score[i])
        row_runtime.setdefault("gti", gti[i])
        row_runtime.setdefault("ghi", ghi[i])

        anomaly_final = bool(anomaly[i]) or bool(diag_direct_grid[i]) or str(diag_diagnosis_labels[i] or "") not in {"normal", "ok", "invalid"}

        data_rel = compute_data_reliability(
            row=row_runtime,
            pac_real_w=pac_real_i,
            pac_model_w=pac_model_i,
            mismatch_rel=mismatch_i,
        )
        det_rel = compute_detection_confidence(
            data_reliability_score=data_rel["score"],
            valid_period=bool(valid_period[i]),
            coarse_period=bool(coarse_period[i]),
            fine_period=bool(fine_period[i]),
            meteo_quality_ok=bool(meteo_quality_ok[i]),
            stable_sky=bool(stable_sky[i]),
            anomaly_flag=bool(anomaly_final),
            mismatch_rel=mismatch_i,
            ewma_z=ewma_i,
            cusum_score=cusum_i,
        )
        diag_rel = compute_diagnosis_confidence(
            diagnosis_label=str(diag_diagnosis_labels[i] or labels[i] or "invalid"),
            base_diagnosis_confidence=(diag_base_conf[i] if i < len(diag_base_conf) else None),
            data_reliability_score=data_rel["score"],
            detection_confidence_score=det_rel["score"],
            fine_diag_allowed=bool(fine_period[i]),
            meteo_quality_ok=bool(meteo_quality_ok[i]),
            direct_grid_evidence=bool(diag_direct_grid[i]),
            zero_injection_flag=bool(diag_zero_inj[i]),
            irradiance_tier=str(irradiance_tier[i] or "N"),
        )

        data_reliability_score[i] = data_rel["score"]
        data_reliability_level[i] = str(data_rel["level"] or "")
        detection_confidence_score[i] = det_rel["score"]
        detection_confidence_level[i] = str(det_rel["level"] or "")
        diagnosis_confidence_score[i] = diag_rel["score"]
        diagnosis_confidence_level[i] = str(diag_rel["level"] or "")
        confidence_notes[i] = {
            "data_reliability": data_rel,
            "detection_confidence": det_rel,
            "diagnosis_confidence": diag_rel,
            "diagnostic_context": {
                "state_label": diag_state_labels[i],
                "domain_label": diag_domain_labels[i],
                "diagnosis_label": diag_diagnosis_labels[i],
                "direct_grid_evidence": bool(diag_direct_grid[i]),
                "zero_injection_flag": bool(diag_zero_inj[i]),
                "irradiance_tier": str(irradiance_tier[i] or "N"),
                "evidence_json": diag_evidence_json[i] if i < len(diag_evidence_json) else {},
            },
        }

    # ----------------------------
    # Série de plot do mismatch (visual)
    # Evita explosões em baixa irradiância / baixa potência
    # e preserva gaps reais quando faltam dados operativos.
    # ----------------------------
    gpoa_plot_min = _gf("gpoa_plot_min", max(700.0, float(gpoa_gate)))
    pmodel_plot_min = _gf("pmodel_plot_min", max(200.0, float(pmin_w)))
    mismatch_clip_abs = _gf("mismatch_clip_abs", 2.0)

    mismatch_rel_raw: List[Optional[float]] = mismatch_rel[:]
    mismatch_rel_plot: List[Optional[float]] = []

    for i in range(n):
        mm = mismatch_rel[i]
        gp = g_poa_used[i]
        pm = pac_model_w[i]
        pr = p_ac_w[i]

        ok_plot = (
            (mm is not None)
            and (gp is not None) and (float(gp) >= float(gpoa_plot_min))
            and (pm is not None) and (abs(float(pm)) >= float(pmodel_plot_min))
            and (pr is not None)
            and (not bool(flag_meteo_missing[i]))
            and (not bool(flag_inv_missing_all[i]))
        )

        if not ok_plot:
            mismatch_rel_plot.append(None)
            continue

        v = float(mm)
        if np.isfinite(v):
            v = max(-float(mismatch_clip_abs), min(float(mismatch_clip_abs), v))
            mismatch_rel_plot.append(v)
        else:
            mismatch_rel_plot.append(None)

    # ----------------------------
    # Persistência
    # ----------------------------
    persist = (data.get("persist") or data.get("save") or "").strip().lower() in ("1", "true", "yes", "on")
    upsert = None
    if persist:
        upsert = _upsert_diag15m(
            plant=plant,
            times_utc=times_utc,
            codes=codes,
            labels=labels,
            valid=valid_period,
            g_poa=g_poa_used,
            tcell_c=tcell_c,
            pac_real_w=p_ac_w,
            pac_model_w=pac_model_w,
            mismatch_rel=mismatch_rel,
        )

    # ----------------------------
    # DUMP por tkey (usa dump_by_tkey no template)
    # ----------------------------
    dump_fields = [
        "p_ac_w", "p_dc_w", "e_ac_wh_15", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a",
        "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
        "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
        "alarm_code", "alarm_sev",
        "inv_coverage", "flag_inv_missing",
        "gti", "ghi", "dni", "dhi", "temp_air", "wind_speed", "rh", "flag_meteo_missing",
    ]

    dump_by_tkey: Dict[str, Any] = {}
    for i, ts_utc in enumerate(times_utc):
        tloc = ts_utc.astimezone(tz)
        tkey = tloc.strftime("%Y-%m-%dT%H:%M")

        by_src = per_ts.get(ts_utc, {})

        meteo_dump: Dict[str, Any] = {}
        src_dump: Dict[str, Any] = {}

        any_row = None
        for sname in selected_sources:
            rr = by_src.get(sname)
            if rr is not None:
                any_row = rr
                break

        if any_row is not None:
            for k in ["gti", "ghi", "dni", "dhi", "temp_air", "wind_speed", "rh", "flag_meteo_missing"]:
                meteo_dump[k] = any_row.get(k)

        for sname in selected_sources:
            rr = by_src.get(sname)
            if rr is None:
                continue
            src_dump[sname] = {
                "p_ac_w": rr.get("p_ac_w"),
                "p_dc_w": rr.get("p_dc_w"),
                "e_ac_wh_15": rr.get("e_ac_wh_15"),
                "v_dc_v": rr.get("v_dc_v"),
                "i_dc_a": rr.get("i_dc_a"),
                "v_ac_v": rr.get("v_ac_v"),
                "i_ac_a": rr.get("i_ac_a"),

                "mppt1_vdc_v": rr.get("mppt1_vdc_v"),
                "mppt2_vdc_v": rr.get("mppt2_vdc_v"),
                "mppt3_vdc_v": rr.get("mppt3_vdc_v"),
                "mppt4_vdc_v": rr.get("mppt4_vdc_v"),
                "mppt1_idc_a": rr.get("mppt1_idc_a"),
                "mppt2_idc_a": rr.get("mppt2_idc_a"),
                "mppt3_idc_a": rr.get("mppt3_idc_a"),
                "mppt4_idc_a": rr.get("mppt4_idc_a"),

                "alarm_code": rr.get("alarm_code"),
                "alarm_sev": rr.get("alarm_sev"),

                "inv_coverage": rr.get("inv_coverage"),
                "flag_inv_missing": rr.get("flag_inv_missing"),
            }

        dump_by_tkey[tkey] = {
            "ts_local": tloc.isoformat(),
            "ts_utc": ts_utc.astimezone(dt_tz.utc).isoformat(),
            "source_meteo": src_meteo,
            "policy": policy_used[i],
            "confidence": {
                "data_reliability_score": data_reliability_score[i],
                "data_reliability_level": data_reliability_level[i],
                "detection_confidence_score": detection_confidence_score[i],
                "detection_confidence_level": detection_confidence_level[i],
                "diagnosis_confidence_score": diagnosis_confidence_score[i],
                "diagnosis_confidence_level": diagnosis_confidence_level[i],
                "notes": confidence_notes[i],
            },
            "chosen_total": {
                "p_ac_w": p_ac_w[i],
                "p_ac_mppt_sum_w": p_ac_mppt_sum_w[i],
                "p_ac_agg_w": p_ac_agg_w[i],
                "inv_coverage": inv_cov[i],
                "flag_inv_missing_all": flag_inv_missing_all[i],
                "flag_inv_missing_partial": flag_inv_missing_partial[i],
            },
            "sources": src_dump,
            "meteo": meteo_dump,
        }

    # severidade (para heatmap)
    rca_code_to_sev = {
        str(CODE_INVALID): "none",
        "0": "ok",
        "1": "warn",
        "2": "warn",
        "3": "crit",
        "4": "crit",
    }

    sev_counts = {"none": 0, "ok": 0, "warn": 0, "crit": 0}
    for c, v in zip(codes, valid_period):
        if not v or int(c) == int(CODE_INVALID):
            sev_counts["none"] += 1
            continue
        sev = rca_code_to_sev.get(str(int(c)), "warn")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    payload = {
        "ok": True,
        "pipeline": pipeline_name,
        "plant": {"id": plant.id, "nome": plant.nome, "tz": tz_name},
        "range": {
            "start": d0.isoformat(),
            "end": d1.isoformat(),
            "start_utc": dt0_utc.isoformat(),
            "end_utc_excl": dt1_utc.isoformat(),
            "source_meteo": src_meteo,
            "selected_sources": selected_sources,
        },
        "versions": MISMATCH_VERSION_SUMMARY,
        "confidence_summary": {
            "data_reliability_mean": _mean_none(data_reliability_score),
            "detection_confidence_mean": _mean_none(detection_confidence_score),
            "diagnosis_confidence_mean": _mean_none(diagnosis_confidence_score),
        },
        "sources": {
            "source_meteo": src_meteo,
            "source_oper_list": source_oper_list,   # ✅ disponíveis
            "selected_sources": selected_sources,   # ✅ selecionadas
            "total_policy": "prefer_mppt_sum",
        },
        "x_local": x_local,
        "x_utc": x_utc,
        "rca_code_to_sev": rca_code_to_sev,
        "dump_fields": dump_fields,
        "dump_by_tkey": dump_by_tkey,
        "series": {
            "t_local": x_local,
            "t_utc": x_utc,

            # meteo
            "g_poa": g_poa_used,
            "g_poa_used": g_poa_used,
            "gti": gti,
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "wind_speed": wind_speed,
            "rh": rh,
            "meteo_qc_score": meteo_qc_score,
            "flag_meteo_low_confidence": flag_meteo_low_confidence,
            "flag_meteo_interpolated": flag_meteo_interpolated,
            "flag_meteo_outlier": flag_meteo_outlier,
            "flag_meteo_artifact": flag_meteo_artifact,
            "flag_meteo_missing": flag_meteo_missing,

            # oper (TOTAL sem dupla contagem)
            "p_ac_w": p_ac_w,
            "p_ac_real_w": p_ac_w,
            "p_dc_w": p_dc_w,
            "e_ac_wh_15": e_ac_wh_15,
            "v_dc_v": v_dc_v,
            "i_dc_a": i_dc_a,
            "v_ac_v": v_ac_v,
            "i_ac_a": i_ac_a,
            "inv_coverage": inv_cov,

            # ✅ flags ajustadas
            "flag_inv_missing": flag_inv_missing_all,
            "flag_inv_missing_all": flag_inv_missing_all,
            "flag_inv_missing_partial": flag_inv_missing_partial,

            # comparação
            "p_ac_mppt_sum_w": p_ac_mppt_sum_w,
            "p_ac_agg_w": p_ac_agg_w,
            "policy_used": policy_used,

            # model
            "p_ac_model_w": pac_model_w,
            "tcell_c": tcell_c,
            "mismatch_rel": mismatch_rel_plot,
            "mismatch_rel_raw": mismatch_rel_raw,
            "gpoa_plot_min": [float(gpoa_plot_min)] * n,

            # confiabilidade em camadas
            "data_reliability_score": data_reliability_score,
            "data_reliability_level": data_reliability_level,
            "detection_confidence_score": detection_confidence_score,
            "detection_confidence_level": detection_confidence_level,
            "diagnosis_confidence_score": diagnosis_confidence_score,
            "diagnosis_confidence_level": diagnosis_confidence_level,
            "state_label": diag_state_labels,
            "domain_label": diag_domain_labels,
            "diagnosis_label": diag_diagnosis_labels,
            "direct_grid_evidence": diag_direct_grid,
            "zero_injection_flag": diag_zero_inj,
            "irradiance_tier": irradiance_tier,
            "pmodel_plot_min": [float(pmodel_plot_min)] * n,

            # detecção + validade
            "valid_model": valid_model,
            "valid_period": valid_period,
            "valid": valid_period,
            "stable_sky": stable_sky,
            "anomaly": anomaly,

            # rca
            "rca_code": codes,
            "rca_label": labels,
            "codes": codes,
            "labels": labels,

            # heatmap helpers (local)
            "hm_day_local": hm_day_local,
            "hm_minute_local": hm_minute_local,
        },
        "series_by_source": series_by_source,
        "summary": {
            "counts": sev_counts,
            "events": [],
            "n_points": n,
        },
        "thresholds": {
            "gpoa_gate": gpoa_gate,
            "pmin_w": pmin_w,
            "warn_abs": float(thr.warn_abs),
            "fault_abs": float(thr.fault_abs),
        },
        "debug": {
            "det": det_dbg,
            "rca": rca_dbg,
        },
        "persist": upsert,
    }

    return _json_response_strict(payload)

@require_GET
@login_required
def mismatch_fdd_export_pdf(request: HttpRequest) -> HttpResponse:
    try:
        if build_mismatch_pdf_report is None:
            return HttpResponse("Serviço de geração PDF não disponível.", content_type="text/plain; charset=utf-8", status=500)

        plant_id = int(request.GET.get("plant_id") or request.GET.get("plant_pk") or request.GET.get("pk") or 0)
        if not plant_id:
            return HttpResponse("plant_id obrigatório", content_type="text/plain; charset=utf-8", status=400)

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return HttpResponse("Planta não encontrada", content_type="text/plain; charset=utf-8", status=404)
        if (not request.user.is_superuser) and plant.owner_id and (plant.owner_id != request.user.id):
            return HttpResponse("Sem permissão para esta planta", content_type="text/plain; charset=utf-8", status=403)

        api_response = mismatch_fdd_api(request)
        payload = json.loads(api_response.content.decode("utf-8"))
        if not payload.get("ok"):
            return HttpResponse(str(payload.get("error") or "Falha ao montar payload do relatório."), content_type="text/plain; charset=utf-8", status=400)

        tz_name = getattr(plant, "timezone", "UTC") or "UTC"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")

        d0 = _parse_date((request.GET.get("start") or "").strip())
        d1 = _parse_date((request.GET.get("end") or "").strip())
        if not d0 or not d1:
            return HttpResponse("start/end são obrigatórios", content_type="text/plain; charset=utf-8", status=400)
        if d1 < d0:
            d0, d1 = d1, d0

        filters = {
            "warn_abs": request.GET.get("warn_abs") or payload.get("thresholds", {}).get("warn_abs"),
            "fault_abs": request.GET.get("fault_abs") or payload.get("thresholds", {}).get("fault_abs"),
            "gpoa_min": request.GET.get("gpoa_min") or payload.get("thresholds", {}).get("gpoa_gate"),
            "pmin_w": request.GET.get("pmin_w") or payload.get("thresholds", {}).get("pmin_w"),
            "dt_minutes": request.GET.get("dt_minutes") or payload.get("series", {}).get("dt_minutes") or request.GET.get("bin_minutes"),
            "source_oper": request.GET.get("source_oper") or request.GET.get("src_oper") or None,
            "source_meteo": request.GET.get("source_meteo") or request.GET.get("src_meteo") or payload.get("sources", {}).get("source_meteo"),
            "pipeline": payload.get("pipeline"),
        }

        generated_at_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        pdf_bytes = build_mismatch_pdf_report(
            plant_name=str(getattr(plant, "nome", f"Plant {plant_id}")),
            payload=payload,
            filters=filters,
            generated_at_local=generated_at_local,
            user_label=str(getattr(request.user, "username", "") or getattr(request.user, "email", "") or request.user.pk),
        )

        filename = f"mismatch_fdd_report_plant{plant_id}_{d0.isoformat()}_{d1.isoformat()}.pdf"
        resp = HttpResponse(pdf_bytes, content_type="application/pdf")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        logger.exception("mismatch_fdd_export_pdf failed")
        return HttpResponse(f"Erro ao gerar PDF: {e}", content_type="text/plain; charset=utf-8", status=500)
