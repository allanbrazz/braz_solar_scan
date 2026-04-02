from __future__ import annotations

from datetime import datetime, timezone as dt_tz
from typing import Any, Dict, List, Tuple

from django.db.models import Count

from core.models import PVPlant, PVPlantMergedRecord15m
from core.services.fdd.dashboard_common import DashboardServiceError, pick_best_sources
from core.services.fdd.runtime_types import MismatchDashboardParams

BASE_VALUES = [
    "ts_utc",
    "source_oper",
    "p_ac_w",
    "p_dc_w",
    "e_ac_wh_15",
    "v_dc_v",
    "i_dc_a",
    "v_ac_v",
    "i_ac_a",
    "freq_hz",
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

OPTIONAL_VALUES = [
    "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
    "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
    "alarm_code", "alarm_sev",
]


def ensure_plant_configuration(plant: PVPlant) -> Tuple[Any, int]:
    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
        raise DashboardServiceError(
            "PVPlantDetails.module não configurado. Cadastre o módulo em 'Planta > Detalhes'.",
            status_code=400,
        )
    n_mod = int(getattr(details, "modules_total", 0) or 0)
    if n_mod <= 0:
        raise DashboardServiceError(
            "PVPlantDetails.modules_total inválido. Configure strings/módulos totais.",
            status_code=400,
        )
    return details, n_mod


def get_values_fields() -> List[str]:
    field_names = {ff.name for ff in PVPlantMergedRecord15m._meta.get_fields() if hasattr(ff, "name")}
    fields = list(BASE_VALUES)
    for k in OPTIONAL_VALUES:
        if k in field_names:
            fields.append(k)
    return fields


def query_runtime_rows(plant: PVPlant, params: MismatchDashboardParams) -> Tuple[str, List[str], List[str], List[Dict[str, Any]]]:
    src_meteo = params.source_meteo
    if not src_meteo:
        _, best_m = pick_best_sources(plant.id, params.dt0_utc, params.dt1_utc)
        src_meteo = best_m
    if not src_meteo:
        raise DashboardServiceError("Sem registros no range (PVPlantMergedRecord15m).", status_code=404)

    src_oper_rows = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant.id,
            source_meteo=src_meteo,
            ts_utc__gte=params.dt0_utc,
            ts_utc__lt=params.dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    source_oper_list = [r["source_oper"] for r in src_oper_rows if r.get("source_oper")]
    if not source_oper_list:
        raise DashboardServiceError("Sem source_oper para a fonte meteo selecionada no range.", status_code=404)

    want_all = (not params.source_oper_raw) or (params.source_oper_raw.upper() == "ALL")
    if want_all:
        selected_sources = list(source_oper_list)
    else:
        if params.source_oper_raw not in source_oper_list:
            raise DashboardServiceError(
                f"source_oper '{params.source_oper_raw}' não existe no range.",
                status_code=404,
            )
        selected_sources = [params.source_oper_raw]

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant.id,
            source_meteo=src_meteo,
            source_oper__in=selected_sources,
            ts_utc__gte=params.dt0_utc,
            ts_utc__lt=params.dt1_utc,
        )
        .order_by("ts_utc", "source_oper")
        .values(*get_values_fields())
    )
    if not rows:
        raise DashboardServiceError("Sem registros no range para as fontes selecionadas.", status_code=404)
    return str(src_meteo), source_oper_list, selected_sources, rows


def group_runtime_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[datetime, Dict[str, Dict[str, Any]]], List[datetime]]:
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
    if not times_utc:
        raise DashboardServiceError("Sem timestamps válidos no range.", status_code=404)
    return per_ts, times_utc
