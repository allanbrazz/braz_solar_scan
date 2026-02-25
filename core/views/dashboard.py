# core/views/dashboard.py
from __future__ import annotations

from core.views._imports import *  # mantém seu padrão (HttpRequest, JsonResponse, render, login_required, require_GET etc.)
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, date, time, timezone as dt_timezone
from typing import Any, Dict, List, Optional
from django.apps import apps

import logging
import math


# Models
from core.models import (
    PVPlant,
    PVPlantMergedRecord15m,
)

UTC = dt_timezone.utc
logger = logging.getLogger(__name__)


# ---------------------------
# TZ helpers
# ---------------------------
def _safe_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "UTC")
    except Exception:
        return ZoneInfo("UTC")


def _local_dates_to_utc_range(start_date: date, end_date: date, tz_name: str) -> tuple[datetime, datetime]:
    """
    Converte [start_date, end_date] (datas locais) -> intervalo UTC [start, end).
    end é exclusivo (end_date + 1 dia, 00:00 local).
    """
    tz = _safe_zoneinfo(tz_name)
    start_local = datetime.combine(start_date, time.min, tzinfo=tz)
    end_local_excl = datetime.combine(end_date, time.min, tzinfo=tz) + timedelta(days=1)
    return start_local.astimezone(UTC), end_local_excl.astimezone(UTC)


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _get_merged15m_model():
    return apps.get_model("core", "PVPlantMergedRecord15m")


def _pick_latest_sources_for_plant(plant: PVPlant) -> tuple[Optional[str], Optional[str]]:
    """
    Descobre automaticamente quais sources (oper/meteo) existem na base merged para a planta,
    usando o registro mais recente como referência.
    """
    M = _get_merged15m_model()

    last = (
        M.objects
        .filter(plant=plant, interval_min=15)
        .exclude(source_oper__isnull=True)
        .exclude(source_meteo__isnull=True)
        .order_by("-ts_utc")
        .values("source_oper", "source_meteo")
        .first()
    )
    if not last:
        return None, None
    return last.get("source_oper"), last.get("source_meteo")


# ----------------------------
# JSON estrito (evita NaN/Inf)
# ----------------------------
try:
    import numpy as np  # type: ignore
except Exception:
    np = None


def _json_safe(x: Any) -> Any:
    if x is None:
        return None

    if np is not None:
        if isinstance(x, (np.floating,)):
            xf = float(x)
            return xf if math.isfinite(xf) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.ndarray,)):
            return [_json_safe(v) for v in x.tolist()]

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    if isinstance(x, (datetime, date)):
        return x.isoformat()

    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    return x


def _json_response_strict(payload: Dict[str, Any], *, status: int = 200) -> JsonResponse:
    payload = _json_safe(payload)
    return JsonResponse(
        payload,
        status=status,
        json_dumps_params={"ensure_ascii": False, "allow_nan": False},
    )


# ---------------------------
# Source classification (MPPT vs AGG)
# ---------------------------
def _is_mppt_source(src: str) -> bool:
    u = (src or "").upper()
    return "|MPPT" in u


def _is_agg_source(src: str) -> bool:
    """
    Considera AGG:
      - sem separador "|" (ex: SHINEMONITOR)
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


# ---------------------------
# Views
# ---------------------------
@login_required
def pv_dashboard_view(request: HttpRequest) -> HttpResponse:
    qs = PVPlant.objects.all().order_by("nome")
    if not request.user.is_superuser:
        qs = qs.filter(owner=request.user)

    plants = list(qs.values("id", "nome", "timezone"))

    today = date.today()
    default_start = today - timedelta(days=2)
    default_end = today

    default_plant_id = plants[0]["id"] if plants else None

    return render(
        request,
        "dashboard/pv_dashboard.html",
        {
            "plants": plants,
            "default_plant_id": default_plant_id,
            "default_start": default_start.isoformat(),
            "default_end": default_end.isoformat(),
            "heatmap_year": today.year,
        },
    )


@require_GET
@login_required
def pv_dashboard_timeseries_api(request: HttpRequest) -> JsonResponse:
    """
    Retorna JSON com séries e KPIs (eixo X em horário local), baseado em PVPlantMergedRecord15m.

    ✅ Política anti-dupla-contagem:
      - Se existir MPPT no timestamp -> TOTAL = Σ(MPPTs) e ignora AGG no total.
      - Se não existir MPPT -> TOTAL = AGG (fallback).
    Mantém:
      - series_by_source com todas as curvas (MPPTs + AGG).
      - series.p_ac_agg_w / p_dc_agg_w / e_ac_wh_15_agg para comparação visual.
      - sources.available_oper para popular dropdown sem “sumir opções”.
    """
    import inspect
    from collections import OrderedDict

    # ----------------------------
    # Helpers locais
    # ----------------------------
    def f(v: Any) -> Optional[float]:
        return _float_or_none(v)

    def s_or_none(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _safe_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    def _safe_float(v: Any) -> Optional[float]:
        return _float_or_none(v)

    def _mean_none(vals: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in vals if x is not None and math.isfinite(x)]
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    def _sum_none(vals: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in vals if x is not None and math.isfinite(x)]
        if not xs:
            return None
        return float(sum(xs))

    def _build_plant_info(plant_obj: PVPlant) -> Dict[str, Any]:
        try:
            details_obj = plant_obj.details
        except Exception:
            details_obj = None

        module_obj = getattr(details_obj, "module", None) if details_obj else None
        inverter_obj = getattr(details_obj, "inverter", None) if details_obj else None

        strings_count = _safe_int(getattr(details_obj, "strings_count", None)) if details_obj else None
        mps = _safe_int(getattr(details_obj, "modules_per_string", None)) if details_obj else None
        modules_total = _safe_int(getattr(details_obj, "modules_total", None)) if details_obj else None

        string_groups = []
        if details_obj and getattr(details_obj, "pk", None):
            try:
                string_groups = list(
                    details_obj.string_configs.order_by("order", "id").values(
                        "id", "name", "order", "mppt", "strings_qty", "modules_per_string"
                    )
                )
            except Exception:
                string_groups = []

        if string_groups:
            sc2 = 0
            mt2 = 0
            for g in string_groups:
                sq = _safe_int(g.get("strings_qty")) or 0
                ns = _safe_int(g.get("modules_per_string")) or 0
                sc2 += sq
                mt2 += sq * ns
            if sc2 > 0:
                strings_count = sc2
            if mt2 > 0:
                modules_total = mt2

            mps_set = {int(g["modules_per_string"]) for g in string_groups if g.get("modules_per_string")}
            mps = (mps_set.pop() if len(mps_set) == 1 else None)

        if (modules_total is None or modules_total == 0) and (strings_count is not None) and (mps is not None):
            modules_total = strings_count * mps

        tilt = _safe_float(getattr(details_obj, "tilt_deg", None)) if details_obj else None
        az = _safe_float(getattr(details_obj, "azimuth_deg", None)) if details_obj else None
        ksys = _safe_float(getattr(details_obj, "k_sys", None)) if details_obj else None
        noct = _safe_float(getattr(details_obj, "noct_c", None)) if details_obj else None

        mod_model = s_or_none(getattr(module_obj, "model", None) or getattr(module_obj, "nome", None) or getattr(module_obj, "name", None))
        mod_mfr = s_or_none(getattr(module_obj, "manufacturer", None) or getattr(module_obj, "marca", None))
        mod_pstc = getattr(module_obj, "p_stc_w", None) or getattr(module_obj, "pmp_stc_w", None) or getattr(module_obj, "p_stc", None)

        inv_model = s_or_none(getattr(inverter_obj, "model", None) or getattr(inverter_obj, "nome", None) or getattr(inverter_obj, "name", None))
        inv_mfr = s_or_none(getattr(inverter_obj, "manufacturer", None) or getattr(inverter_obj, "marca", None))
        inv_pac = getattr(inverter_obj, "p_ac_rated_w", None) or getattr(inverter_obj, "pac_rated_w", None) or getattr(inverter_obj, "p_nom_w", None)

        return {
            "module": {
                "id": getattr(module_obj, "id", None),
                "model": mod_model,
                "manufacturer": mod_mfr,
                "p_stc_w": _safe_float(mod_pstc),
            },
            "inverter": {
                "id": getattr(inverter_obj, "id", None),
                "model": inv_model,
                "manufacturer": inv_mfr,
                "p_ac_rated_w": _safe_float(inv_pac),
            },
            "electrical": {
                "strings_count": strings_count,
                "modules_per_string": mps,
                "modules_total": modules_total,
                "string_groups": string_groups,
            },
            "geometry": {
                "tilt_deg": tilt,
                "azimuth_deg": az,
                "k_sys": ksys,
                "noct_c": noct,
            },
        }

    def _empty_payload(
        *,
        plant: Optional[PVPlant],
        tz_name: str,
        start_s: str,
        end_s: str,
        dt0_utc=None,
        dt1_utc=None,
        src_oper=None,
        src_oper_list=None,
        src_meteo=None,
        available_oper=None,
        message: str = "",
    ):
        plant_info = {}
        if plant is not None:
            try:
                plant_info = _build_plant_info(plant)
            except Exception:
                plant_info = {}

        return {
            "ok": True,
            "empty": True,
            "message": message or "Sem dados.",
            "plant": {"id": getattr(plant, "id", None), "nome": getattr(plant, "nome", None), "tz": tz_name},
            "plant_info": plant_info,
            "range": {
                "start_local": start_s or None,
                "end_local": end_s or None,
                "dt0_utc": dt0_utc.isoformat() if dt0_utc else None,
                "dt1_utc": dt1_utc.isoformat() if dt1_utc else None,
            },
            "sources": {
                "source_oper": src_oper,
                "source_oper_list": src_oper_list or [],
                "available_oper": available_oper or [],
                "source_meteo": src_meteo,
                "total_policy": "prefer_mppt_sum",
                "mppt_sources": [],
                "agg_sources": [],
            },
            "x": [],
            "x_label": [],
            "series": {},
            "series_by_source": {},
            "charts": {
                "gauge": {"score": None, "gcv_stat": None, "label": "Indisponível"},
                "scatter": {"times": [], "x_gcv": [], "y_mismatch": [], "code": [], "code_name": []},
                "sankey": {"nodes": [], "links": [], "values_kwh": {}},
                "timeline": {"times": [], "code": [], "code_hyst": [], "dt_minutes": 15.0, "min_persist_minutes": 60.0},
            },
            "kpis": {"points": 0},
            "audit": {"model_ok": False, "model_error": None, "model_meta": {}, "g_used": "—"},
            "debug": {"has_model": False, "model_error": None},
        }

    # ----------------------------
    # Inputs
    # ----------------------------
    try:
        plant_id = int(request.GET.get("plant_id", "0"))
    except Exception:
        return _json_response_strict({"ok": False, "error": "plant_id inválido"}, status=400)

    start_s = (request.GET.get("start") or "").strip()
    end_s = (request.GET.get("end") or "").strip()

    if not start_s or not end_s:
        return _json_response_strict({"ok": False, "error": "start e end são obrigatórios (YYYY-MM-DD)"}, status=400)

    try:
        start_date = date.fromisoformat(start_s)
        end_date = date.fromisoformat(end_s)
    except Exception:
        return _json_response_strict({"ok": False, "error": "start/end devem estar no formato YYYY-MM-DD"}, status=400)

    if end_date < start_date:
        return _json_response_strict({"ok": False, "error": "end não pode ser menor que start"}, status=400)

    # ----------------------------
    # Planta + permissão
    # ----------------------------
    plant = PVPlant.objects.filter(id=plant_id).first()
    if not plant:
        return _json_response_strict({"ok": False, "error": "Planta não encontrada"}, status=404)

    if (not request.user.is_superuser) and (plant.owner_id != request.user.id):
        return _json_response_strict({"ok": False, "error": "Sem permissão para esta planta"}, status=403)

    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    tz = _safe_zoneinfo(tz_name)

    dt0_utc, dt1_utc = _local_dates_to_utc_range(start_date, end_date, tz_name)

    try:
        plant_info = _build_plant_info(plant)
    except Exception:
        plant_info = {}

    # ----------------------------
    # Sources (multi source_oper)
    # ----------------------------
    src_oper_raw = (request.GET.get("source_oper") or "").strip() or None
    src_meteo = (request.GET.get("source_meteo") or "").strip() or None

    if (not src_oper_raw) or (not src_meteo):
        _, auto_meteo = _pick_latest_sources_for_plant(plant)
        src_meteo = src_meteo or auto_meteo

    if not src_meteo:
        return _json_response_strict(
            _empty_payload(
                plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
                dt0_utc=dt0_utc, dt1_utc=dt1_utc,
                src_oper=None, src_oper_list=[], src_meteo=None, available_oper=[],
                message="Não há dados merged_15m para esta planta ainda (source_meteo ausente).",
            )
        )

    # disponíveis (para dropdown)
    avail_oper = list(
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_meteo=src_meteo,
            interval_min=15,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).values_list("source_oper", flat=True).distinct()
    )
    avail_oper = [s for s in avail_oper if s]  # remove None/''

    if not avail_oper:
        avail_oper = list(
            PVPlantMergedRecord15m.objects.filter(
                plant=plant,
                interval_min=15,
                ts_utc__gte=dt0_utc,
                ts_utc__lt=dt1_utc,
            ).values_list("source_oper", flat=True).distinct()
        )
        avail_oper = [s for s in avail_oper if s]

    # seleção
    src_oper_list: List[str] = []
    if src_oper_raw:
        if src_oper_raw.strip().upper() == "ALL":
            src_oper_list = list(avail_oper)
        else:
            src_oper_list = [s.strip() for s in src_oper_raw.split(",") if s.strip()]
    else:
        src_oper_list = list(avail_oper)

    if avail_oper:
        avail_set = set(avail_oper)
        src_oper_list = [s for s in src_oper_list if s in avail_set]

    if not src_oper_list:
        return _json_response_strict(
            _empty_payload(
                plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
                dt0_utc=dt0_utc, dt1_utc=dt1_utc,
                src_oper=src_oper_raw, src_oper_list=[], src_meteo=src_meteo, available_oper=avail_oper,
                message="Sem dados merged_15m no intervalo para as fontes operativas selecionadas.",
            )
        )

    src_oper = src_oper_list[0] if src_oper_list else None

    # ----------------------------
    # Query merged_15m (values dinâmico: não quebra se campo não existir)
    # ----------------------------
    field_names = {ff.name for ff in PVPlantMergedRecord15m._meta.get_fields() if hasattr(ff, "name")}

    base_values = [
        "ts_utc",
        "source_oper",
        "p_ac_w", "p_dc_w", "e_ac_wh_15",
        "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a",
        "ghi", "gti", "dni", "dhi",
        "temp_air", "wind_speed", "rh",
        "inv_coverage",
        "flag_meteo_missing", "flag_inv_missing",
    ]

    optional_values = [
        # MPPT fields (se existirem no model)
        "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
        "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
        # alarmes (se existirem no model)
        "alarm_code", "alarm_sev",
    ]

    values_fields = list(base_values)
    for k in optional_values:
        if k in field_names:
            values_fields.append(k)

    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_oper__in=src_oper_list,
            source_meteo=src_meteo,
            interval_min=15,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .order_by("ts_utc", "source_oper")
        .values(*values_fields)
    )

    rows = list(qs)
    if not rows:
        return _json_response_strict(
            _empty_payload(
                plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
                dt0_utc=dt0_utc, dt1_utc=dt1_utc,
                src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo, available_oper=avail_oper,
                message="Sem pontos no intervalo selecionado.",
            )
        )

    # ----------------------------
    # Pivot por timestamp
    # ----------------------------
    rec_by_ts: "OrderedDict[datetime, Dict[str, Dict[str, Any]]]" = OrderedDict()
    sources_set = set()

    for r in rows:
        ts_utc = r.get("ts_utc")
        if isinstance(ts_utc, str):
            ts_utc = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        if ts_utc is None:
            continue
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=UTC)

        src = (r.get("source_oper") or "").strip() or "unknown"
        sources_set.add(src)
        rec_by_ts.setdefault(ts_utc, {})[src] = r

    if not rec_by_ts:
        return _json_response_strict(
            _empty_payload(
                plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
                dt0_utc=dt0_utc, dt1_utc=dt1_utc,
                src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo, available_oper=avail_oper,
                message="Sem timestamps válidos no intervalo (ts_utc ausente/inválido).",
            )
        )

    sources = sorted(sources_set)
    mppt_sources = [s for s in sources if _is_mppt_source(s)]
    agg_sources = [s for s in sources if _is_agg_source(s)]

    # ----------------------------
    # Séries (TOTAL sem dupla contagem + AGG separado + por source)
    # ----------------------------
    x_iso: List[str] = []
    x_label: List[str] = []
    t_utc: List[datetime] = []

    # TOTAL (prefer ΣMPPT)
    p_ac: List[Optional[float]] = []
    p_dc: List[Optional[float]] = []
    e15_wh: List[Optional[float]] = []  # para debug (opcional)
    v_dc: List[Optional[float]] = []
    i_dc: List[Optional[float]] = []
    v_ac: List[Optional[float]] = []
    i_ac: List[Optional[float]] = []

    # AGG (medido) — comparação
    p_ac_agg: List[Optional[float]] = []
    p_dc_agg: List[Optional[float]] = []
    e15_wh_agg: List[Optional[float]] = []

    ghi: List[Optional[float]] = []
    gti: List[Optional[float]] = []
    dni: List[Optional[float]] = []
    dhi: List[Optional[float]] = []
    temp_air: List[Optional[float]] = []
    wind: List[Optional[float]] = []
    rh: List[Optional[float]] = []

    e_wh_total = 0.0
    p_ac_max = None
    ghi_max = None

    # KPIs — agora por timestamp (não por “ponto fonte”)
    inv_cov_ts: List[Optional[float]] = []
    met_missing_ts = 0
    inv_missing_ts_all = 0
    inv_missing_ts_partial = 0

    # series_by_source
    series_by_source: Dict[str, Dict[str, List[Any]]] = {}
    for src in sources:
        series_by_source[src] = {
            "p_ac_w": [],
            "p_dc_w": [],
            "e_ac_wh_15": [],
            "v_dc_v": [],
            "i_dc_a": [],
            "v_ac_v": [],
            "i_ac_a": [],
            "inv_coverage": [],
            "flag_inv_missing": [],

            # MPPT (se vierem no values)
            "mppt1_vdc_v": [], "mppt2_vdc_v": [], "mppt3_vdc_v": [], "mppt4_vdc_v": [],
            "mppt1_idc_a": [], "mppt2_idc_a": [], "mppt3_idc_a": [], "mppt4_idc_a": [],

            # alarmes
            "alarm_code": [],
            "alarm_sev": [],
        }

    for ts_utc, per_src in rec_by_ts.items():
        t_utc.append(ts_utc)

        ts_local = ts_utc.astimezone(tz)
        x_iso.append(ts_local.isoformat())
        x_label.append(ts_local.strftime("%d/%m %H:%M"))

        # meteo: usa qualquer row do timestamp (primeiro)
        first_row = None
        if per_src:
            # tenta ser determinístico: pega o primeiro pela ordem sources
            for s0 in sources:
                if s0 in per_src:
                    first_row = per_src[s0]
                    break
            if first_row is None:
                first_row = next(iter(per_src.values()))

        g_ghi = f(first_row.get("ghi")) if first_row else None
        g_gti = f(first_row.get("gti")) if first_row else None
        g_dni = f(first_row.get("dni")) if first_row else None
        g_dhi = f(first_row.get("dhi")) if first_row else None
        g_ta = f(first_row.get("temp_air")) if first_row else None
        g_ws = f(first_row.get("wind_speed")) if first_row else None
        g_rh = f(first_row.get("rh")) if first_row else None

        ghi.append(g_ghi)
        gti.append(g_gti)
        dni.append(g_dni)
        dhi.append(g_dhi)
        temp_air.append(g_ta)
        wind.append(g_ws)
        rh.append(g_rh)

        if g_ghi is not None:
            ghi_max = g_ghi if (ghi_max is None or g_ghi > ghi_max) else ghi_max

        if first_row and bool(first_row.get("flag_meteo_missing")):
            met_missing_ts += 1

        # -------- preenche series_by_source --------
        for src in sources:
            r = per_src.get(src)

            pac = f(r.get("p_ac_w")) if r else None
            pdc = f(r.get("p_dc_w")) if r else None
            e15 = f(r.get("e_ac_wh_15")) if r else None
            vdc = f(r.get("v_dc_v")) if r else None
            idc_ = f(r.get("i_dc_a")) if r else None
            vac = f(r.get("v_ac_v")) if r else None
            iac = f(r.get("i_ac_a")) if r else None
            cov = f(r.get("inv_coverage")) if r else None
            flag_inv_missing = bool(r.get("flag_inv_missing")) if r else True

            mppt1_v = f(r.get("mppt1_vdc_v")) if r else None
            mppt2_v = f(r.get("mppt2_vdc_v")) if r else None
            mppt3_v = f(r.get("mppt3_vdc_v")) if r else None
            mppt4_v = f(r.get("mppt4_vdc_v")) if r else None

            mppt1_i = f(r.get("mppt1_idc_a")) if r else None
            mppt2_i = f(r.get("mppt2_idc_a")) if r else None
            mppt3_i = f(r.get("mppt3_idc_a")) if r else None
            mppt4_i = f(r.get("mppt4_idc_a")) if r else None

            a_code = _safe_int(r.get("alarm_code")) if r else None
            a_sev = _safe_int(r.get("alarm_sev")) if r else None

            sb = series_by_source[src]
            sb["p_ac_w"].append(pac)
            sb["p_dc_w"].append(pdc)
            sb["e_ac_wh_15"].append(e15)
            sb["v_dc_v"].append(vdc)
            sb["i_dc_a"].append(idc_)
            sb["v_ac_v"].append(vac)
            sb["i_ac_a"].append(iac)
            sb["inv_coverage"].append(cov)
            sb["flag_inv_missing"].append(bool(flag_inv_missing))

            sb["mppt1_vdc_v"].append(mppt1_v)
            sb["mppt2_vdc_v"].append(mppt2_v)
            sb["mppt3_vdc_v"].append(mppt3_v)
            sb["mppt4_vdc_v"].append(mppt4_v)
            sb["mppt1_idc_a"].append(mppt1_i)
            sb["mppt2_idc_a"].append(mppt2_i)
            sb["mppt3_idc_a"].append(mppt3_i)
            sb["mppt4_idc_a"].append(mppt4_i)

            sb["alarm_code"].append(a_code)
            sb["alarm_sev"].append(a_sev)

        # -------- TOTAL (prefer ΣMPPT; senão AGG) --------
        mppt_keys_ts = [k for k in per_src.keys() if _is_mppt_source(k)]
        agg_keys_ts = [k for k in per_src.keys() if _is_agg_source(k)]

        if mppt_keys_ts:
            chosen_keys = mppt_keys_ts
            total_mode = "mppt_sum"
        else:
            chosen_keys = agg_keys_ts if agg_keys_ts else list(per_src.keys())
            total_mode = "agg_fallback" if agg_keys_ts else "any_fallback"

        # TOTAL
        pac_vals = [f(per_src[k].get("p_ac_w")) for k in chosen_keys if per_src.get(k)]
        pdc_vals = [f(per_src[k].get("p_dc_w")) for k in chosen_keys if per_src.get(k)]
        e15_vals = [f(per_src[k].get("e_ac_wh_15")) for k in chosen_keys if per_src.get(k)]
        vdc_vals = [f(per_src[k].get("v_dc_v")) for k in chosen_keys if per_src.get(k)]
        idc_vals = [f(per_src[k].get("i_dc_a")) for k in chosen_keys if per_src.get(k)]
        vac_vals = [f(per_src[k].get("v_ac_v")) for k in chosen_keys if per_src.get(k)]
        iac_vals = [f(per_src[k].get("i_ac_a")) for k in chosen_keys if per_src.get(k)]
        cov_vals = [f(per_src[k].get("inv_coverage")) for k in chosen_keys if per_src.get(k)]

        pac_total = _sum_none(pac_vals)
        pdc_total = _sum_none(pdc_vals)
        e15_total = _sum_none(e15_vals)
        vdc_agg = _mean_none(vdc_vals)
        idc_agg = _sum_none(idc_vals)
        vac_agg = _mean_none(vac_vals)
        iac_agg = _sum_none(iac_vals)
        cov_agg = _mean_none(cov_vals)

        p_ac.append(pac_total)
        p_dc.append(pdc_total)
        e15_wh.append(e15_total)
        v_dc.append(vdc_agg)
        i_dc.append(idc_agg)
        v_ac.append(vac_agg)
        i_ac.append(iac_agg)

        inv_cov_ts.append(cov_agg)

        if pac_total is not None:
            p_ac_max = pac_total if (p_ac_max is None or pac_total > p_ac_max) else p_ac_max

        if e15_total is not None:
            e_wh_total += float(e15_total)

        # AGG separado (comparação) — soma se houver mais de um agg no ts
        pac_agg_vals = [f(per_src[k].get("p_ac_w")) for k in agg_keys_ts if per_src.get(k)]
        pdc_agg_vals = [f(per_src[k].get("p_dc_w")) for k in agg_keys_ts if per_src.get(k)]
        e15_agg_vals = [f(per_src[k].get("e_ac_wh_15")) for k in agg_keys_ts if per_src.get(k)]

        p_ac_agg.append(_sum_none(pac_agg_vals) if agg_keys_ts else None)
        p_dc_agg.append(_sum_none(pdc_agg_vals) if agg_keys_ts else None)
        e15_wh_agg.append(_sum_none(e15_agg_vals) if agg_keys_ts else None)

        # Missing por timestamp (não por fonte)
        chosen_flags = []
        for k in chosen_keys:
            rr = per_src.get(k)
            if rr is None:
                continue
            chosen_flags.append(bool(rr.get("flag_inv_missing") or False))

        if not chosen_flags:
            inv_missing_ts_all += 1
        else:
            all_missing = all(chosen_flags)
            any_missing = any(chosen_flags)
            if all_missing:
                inv_missing_ts_all += 1
            elif any_missing and (len(chosen_flags) > 1):
                inv_missing_ts_partial += 1

    n = len(x_iso)
    if n == 0:
        return _json_response_strict(
            _empty_payload(
                plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
                dt0_utc=dt0_utc, dt1_utc=dt1_utc,
                src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo, available_oper=avail_oper,
                message="Sem pontos válidos após agregação por timestamp.",
            )
        )

    # ----------------------------
    # MODELO: power_model.py + charts
    # ----------------------------
    dt_minutes = 15.0
    persist_minutes = 60.0

    p_ac_model_w = None
    mismatch_rel = None
    tcell_c = None
    e_model_kwh = None

    p_ac_pu_model = None
    p_ac_pu_real = None
    pr_model_inst = None
    pr_real_inst = None
    g_std_60m = None
    g_cv_60m = None
    csi = None
    eta_inv = None

    pac_model_pu_stc = None
    pac_real_pu_stc = None
    k_cs = None
    g_poa_used = None

    rca_label = None
    valid_model = None

    charts = {
        "gauge": {"score": None, "gcv_stat": None, "label": "Indisponível"},
        "scatter": {"times": [], "x_gcv": [], "y_mismatch": [], "code": [], "code_name": []},
        "sankey": {"nodes": [], "links": [], "values_kwh": {}},
        "timeline": {"times": [], "code": [], "code_hyst": [], "dt_minutes": dt_minutes, "min_persist_minutes": persist_minutes},
    }

    audit = {"model_ok": False, "model_error": None, "model_meta": {}, "g_used": "—"}

    try:
        import numpy as _np
        from dataclasses import asdict

        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
            transpose_ghi_to_poa_isotropic,
        )

        def list_to_np_nan(xs):
            return _np.array([_np.nan if v is None else float(v) for v in (xs or [])], dtype=float)

        def np_to_list_none(a):
            a = _np.asarray(a, dtype=float)
            return [None if (not _np.isfinite(v)) else float(v) for v in a.tolist()]

        def np_to_list_str(a):
            a = _np.asarray(a, dtype=object)
            out = []
            for v in a.tolist():
                out.append(None if v is None else str(v))
            return out

        def np_to_list_bool(a):
            a = _np.asarray(a, dtype=bool)
            return [bool(v) for v in a.tolist()]

        # charts builder
        build_dashboard_payload = None
        try:
            from core.services.dashboard.dashboard_charts import build_dashboard_payload as _build
            build_dashboard_payload = _build
        except Exception:
            build_dashboard_payload = None

        details = getattr(plant, "details", None)

        if details and getattr(details, "module_id", None):
            n_mod = int(getattr(details, "modules_total", 0) or 0)
            if n_mod > 0:
                mod = module_from_pvmodule(details.module)
                inv = getattr(details, "inverter", None)
                pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

                pld = asdict(pl)
                if pld.get("lat_deg") is None:
                    pld["lat_deg"] = _safe_float(getattr(plant, "latitude", None) or getattr(plant, "latitude_deg", None) or getattr(plant, "lat_deg", None))
                if pld.get("lon_deg") is None:
                    pld["lon_deg"] = _safe_float(getattr(plant, "longitude", None) or getattr(plant, "longitude_deg", None) or getattr(plant, "lon_deg", None))
                if pld.get("tilt_deg") is None:
                    pld["tilt_deg"] = _safe_float(getattr(details, "tilt_deg", None) or getattr(plant, "tilt_deg", None))
                if pld.get("azimuth_deg") is None:
                    pld["azimuth_deg"] = _safe_float(getattr(details, "azimuth_deg", None) or getattr(plant, "azimuth_deg", None))
                pl = pl.__class__(**pld)

                gti_np = list_to_np_nan(gti)
                ghi_np = list_to_np_nan(ghi)
                dni_np = list_to_np_nan(dni)
                dhi_np = list_to_np_nan(dhi)

                gti_ok = _np.isfinite(gti_np)
                ghi_ok = _np.isfinite(ghi_np)

                gpoa_np = _np.where(gti_ok, gti_np, _np.nan)

                can_transpose = (
                    (pl.lat_deg is not None) and (pl.lon_deg is not None) and
                    (pl.tilt_deg is not None) and (pl.azimuth_deg is not None) and
                    (len(t_utc) == len(ghi)) and _np.any(ghi_ok)
                )

                if can_transpose and (not _np.all(gti_ok)):
                    trans = transpose_ghi_to_poa_isotropic(
                        ghi=ghi_np,
                        dhi=dhi_np if _np.any(_np.isfinite(dhi_np)) else None,
                        dni=dni_np if _np.any(_np.isfinite(dni_np)) else None,
                        times_utc=t_utc,
                        lat_deg=float(pl.lat_deg),
                        lon_deg=float(pl.lon_deg),
                        tilt_deg=float(pl.tilt_deg),
                        azimuth_deg=float(pl.azimuth_deg),
                        albedo=float(pl.albedo),
                    )
                    gpoa_tr = _np.asarray(trans.get("g_poa"), dtype=float)
                    gpoa_np = _np.where(gti_ok, gti_np, gpoa_tr)
                    audit["g_used"] = "GTI + POA(transposição) p/ faltas" if _np.any(gti_ok) else "POA(transposição)"
                elif _np.any(gti_ok):
                    audit["g_used"] = "GTI"
                else:
                    gpoa_np = ghi_np
                    audit["g_used"] = "GHI (sem POA/geo)"

                g_poa_used = np_to_list_none(gpoa_np)

                tamb_np = list_to_np_nan(temp_air)
                pac_real_np = list_to_np_nan(p_ac)
                vdc_np = list_to_np_nan(v_dc)
                idc_np = list_to_np_nan(i_dc)

                sig = inspect.signature(expected_and_mismatch)
                kwargs = dict(
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
                    kwargs["dt_minutes"] = dt_minutes
                if "window_minutes" in sig.parameters:
                    kwargs["window_minutes"] = 60.0
                if "times_utc" in sig.parameters:
                    kwargs["times_utc"] = t_utc

                if "v_dc_real_v" in sig.parameters:
                    kwargs["v_dc_real_v"] = vdc_np
                elif "vdc_meas_v" in sig.parameters:
                    kwargs["vdc_meas_v"] = vdc_np

                if "i_dc_real_a" in sig.parameters:
                    kwargs["i_dc_real_a"] = idc_np
                elif "idc_meas_a" in sig.parameters:
                    kwargs["idc_meas_a"] = idc_np

                out_model = expected_and_mismatch(**kwargs) or {}
                meta = out_model.get("meta", {}) if isinstance(out_model, dict) else {}

                audit["model_ok"] = True
                audit["model_meta"] = meta

                pac_exp = out_model.get("pac_expected_w")
                if pac_exp is not None:
                    p_ac_model_w = np_to_list_none(pac_exp)
                    dt_h = dt_minutes / 60.0
                    pac_exp_np = _np.asarray(pac_exp, dtype=float)
                    e_model_kwh = float(_np.nansum(_np.clip(pac_exp_np, 0.0, None)) * dt_h / 1000.0)

                if out_model.get("mismatch_rel") is not None:
                    mismatch_rel = np_to_list_none(out_model["mismatch_rel"])
                if out_model.get("tcell_c") is not None:
                    tcell_c = np_to_list_none(out_model["tcell_c"])

                if out_model.get("p_ac_pu_model") is not None:
                    p_ac_pu_model = np_to_list_none(out_model["p_ac_pu_model"])
                if out_model.get("p_ac_pu_real") is not None:
                    p_ac_pu_real = np_to_list_none(out_model["p_ac_pu_real"])
                if out_model.get("pr_model_inst") is not None:
                    pr_model_inst = np_to_list_none(out_model["pr_model_inst"])
                if out_model.get("pr_real_inst") is not None:
                    pr_real_inst = np_to_list_none(out_model["pr_real_inst"])
                if out_model.get("g_std_60m") is not None:
                    g_std_60m = np_to_list_none(out_model["g_std_60m"])
                if out_model.get("g_cv_60m") is not None:
                    g_cv_60m = np_to_list_none(out_model["g_cv_60m"])
                if out_model.get("csi") is not None:
                    csi = np_to_list_none(out_model["csi"])
                    k_cs = csi
                if out_model.get("eta_inv") is not None:
                    eta_inv = np_to_list_none(out_model["eta_inv"])

                pac_model_pu_stc = p_ac_pu_model
                pac_real_pu_stc = p_ac_pu_real

                if out_model.get("rca_label") is not None:
                    rca_label = np_to_list_str(out_model["rca_label"])
                if out_model.get("valid") is not None:
                    valid_model = np_to_list_bool(out_model["valid"])

                if callable(build_dashboard_payload):
                    charts = build_dashboard_payload(
                        times=x_iso,
                        out_model=out_model,
                        dt_minutes=dt_minutes,
                        min_persist_minutes=persist_minutes,
                    )

    except Exception as e:
        audit["model_ok"] = False
        audit["model_error"] = f"{type(e).__name__}: {e}"
        logger.exception("Falha ao calcular modelo físico no pv_dashboard_timeseries_api (plant_id=%s).", plant_id)

    # ----------------------------
    # KPIs + payload
    # ----------------------------
    e_kwh = e_wh_total / 1000.0

    inv_cov_mean = _mean_none(inv_cov_ts)
    met_missing_frac = round(met_missing_ts / n, 3) if n else 0.0
    inv_missing_frac = round(inv_missing_ts_all / n, 3) if n else 0.0
    inv_partial_missing_frac = round(inv_missing_ts_partial / n, 3) if n else 0.0

    payload: Dict[str, Any] = {
        "ok": True,
        "empty": False,
        "plant": {"id": plant.id, "nome": plant.nome, "tz": tz_name},
        "plant_info": plant_info,
        "range": {
            "start_local": start_s,
            "end_local": end_s,
            "dt0_utc": dt0_utc.isoformat() if hasattr(dt0_utc, "isoformat") else dt0_utc,
            "dt1_utc": dt1_utc.isoformat() if hasattr(dt1_utc, "isoformat") else dt1_utc,
        },
        "sources": {
            "source_oper": src_oper,
            "source_oper_list": src_oper_list,     # selecionadas
            "available_oper": avail_oper,          # disponíveis p/ dropdown
            "source_meteo": src_meteo,
            "total_policy": "prefer_mppt_sum",
            "mppt_sources": mppt_sources,
            "agg_sources": agg_sources,
        },
        "x": x_iso,
        "x_label": x_label,
        "charts": charts,
        "series": {
            # TOTAL (sem dupla contagem; prefer ΣMPPT)
            "p_ac_w": p_ac,
            "p_dc_w": p_dc,
            "v_dc_v": v_dc,
            "i_dc_a": i_dc,
            "v_ac_v": v_ac,
            "i_ac_a": i_ac,
            "e_ac_wh_15": e15_wh,  # opcional

            # AGG (medido) para comparar com ΣMPPT
            "p_ac_agg_w": p_ac_agg,
            "p_dc_agg_w": p_dc_agg,
            "e_ac_wh_15_agg": e15_wh_agg,

            # meteo
            "ghi": ghi,
            "gti": gti,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "wind_speed": wind,
            "rh": rh,

            # modelo
            "p_ac_model_w": p_ac_model_w,
            "mismatch_rel": mismatch_rel,
            "tcell_c": tcell_c,
            "p_ac_pu_model": p_ac_pu_model,
            "p_ac_pu_real": p_ac_pu_real,
            "pr_model_inst": pr_model_inst,
            "pr_real_inst": pr_real_inst,
            "g_std_60m": g_std_60m,
            "g_cv_60m": g_cv_60m,
            "csi": csi,
            "eta_inv": eta_inv,
            "pac_model_pu_stc": pac_model_pu_stc,
            "pac_real_pu_stc": pac_real_pu_stc,
            "k_cs": k_cs,
            "g_poa_used": g_poa_used,

            # repasse
            "valid_model": valid_model,
            "rca_label": rca_label,
        },
        "series_by_source": series_by_source,
        "kpis": {
            "energy_kwh": round(e_kwh, 3),
            "energy_model_kwh": None if e_model_kwh is None else round(e_model_kwh, 3),
            "p_ac_max_w": None if p_ac_max is None else round(p_ac_max, 1),
            "ghi_max_wm2": None if ghi_max is None else round(ghi_max, 1),
            "inv_coverage_mean": None if inv_cov_mean is None else round(inv_cov_mean, 3),
            "meteo_missing_frac": met_missing_frac,
            "inv_missing_frac": inv_missing_frac,
            "inv_partial_missing_frac": inv_partial_missing_frac,
            "points": n,
            "sources_oper_qty": len(src_oper_list),
            "mppt_sources_qty": len(mppt_sources),
            "agg_sources_qty": len(agg_sources),
            "meteo_reliability_score": (charts.get("gauge") or {}).get("score") if isinstance(charts, dict) else None,
            "meteo_reliability_label": (charts.get("gauge") or {}).get("label") if isinstance(charts, dict) else None,
        },
        "audit": audit,
        "debug": {
            "len_x": len(x_iso),
            "len_sources_selected": len(src_oper_list),
            "len_sources_in_payload": len(sources),
            "mppt_sources": mppt_sources,
            "agg_sources": agg_sources,
            "has_model": bool(audit.get("model_ok")) and (p_ac_model_w is not None),
            "model_error": audit.get("model_error"),
        },
    }

    return _json_response_strict(payload)