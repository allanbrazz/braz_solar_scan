from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.db.models import Count

from core.models import FaultEvent, PVPlantMergedRecord15m
from core.services.mppt_gnn_fdd.features import WindowArrays
from core.services.mppt_gnn_fdd.window_loader import compute_pac_model_and_mismatch


def _pick_best_source_meteo(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(plant_id=plant_id, ts_utc__gte=dt0_utc, ts_utc__lte=dt1_utc)
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")


def _pick_best_source_oper(plant_id: int, source_meteo: str, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lte=dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_oper")


def _grid_range_utc(dt0_utc: datetime, dt1_utc: datetime, dt_min: int = 15) -> List[datetime]:
    out: List[datetime] = []
    cur = dt0_utc
    while cur <= dt1_utc:
        out.append(cur)
        cur += timedelta(minutes=dt_min)
    return out


def _fill_on_grid(ts_grid: List[datetime], rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    idx = {t: j for j, t in enumerate(ts_grid)}
    arr = np.full(len(ts_grid), np.nan, dtype=float)
    for r in rows:
        j = idx.get(r["ts_utc"])
        if j is None:
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            arr[j] = float(v)
        except Exception:
            pass
    return arr


def load_event_window(
    *,
    event_id: int,
    pre_bins: int = 8,
    post_bins: int = 8,
    n_mppt: int = 4,
) -> Tuple[WindowArrays, List[datetime], Dict[str, Any]]:
    event = FaultEvent.objects.select_related(
        "plant",
        "plant__details",
        "plant__details__module",
        "plant__details__inverter",
    ).filter(id=event_id).first()
    if event is None:
        raise ValueError("FaultEvent não encontrado")

    dt0_utc = event.ts_start_utc - timedelta(minutes=15 * int(pre_bins))
    dt1_utc = event.ts_end_utc + timedelta(minutes=15 * int(post_bins))

    source_meteo = event.source_meteo or _pick_best_source_meteo(event.plant_id, dt0_utc, dt1_utc) or "OPENMETEO"
    source_oper = event.source_oper or _pick_best_source_oper(event.plant_id, source_meteo, dt0_utc, dt1_utc) or "SHINEMONITOR"

    ts_grid = _grid_range_utc(dt0_utc, dt1_utc, dt_min=15)

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=event.plant_id,
            source_oper=source_oper,
            source_meteo=source_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lte=dt1_utc,
        )
        .order_by("ts_utc")
        .values(
            "ts_utc",
            "p_ac_w", "v_dc_v", "i_ac_a",
            "gti", "ghi", "dni", "dhi", "temp_air",
            "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
            "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
        )
    )

    pac = _fill_on_grid(ts_grid, rows, "p_ac_w")
    vdc_total = _fill_on_grid(ts_grid, rows, "v_dc_v")
    iac = _fill_on_grid(ts_grid, rows, "i_ac_a")
    gti = _fill_on_grid(ts_grid, rows, "gti")
    ghi = _fill_on_grid(ts_grid, rows, "ghi")
    dni = _fill_on_grid(ts_grid, rows, "dni")
    dhi = _fill_on_grid(ts_grid, rows, "dhi")
    tair = _fill_on_grid(ts_grid, rows, "temp_air")

    mppt_vdc = np.full((n_mppt, len(ts_grid)), np.nan, dtype=float)
    mppt_idc = np.full((n_mppt, len(ts_grid)), np.nan, dtype=float)
    for k in range(1, n_mppt + 1):
        mppt_vdc[k - 1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_vdc_v")
        mppt_idc[k - 1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_idc_a")

    pac_model, mismatch = compute_pac_model_and_mismatch(
        plant=event.plant,
        times_utc=ts_grid,
        gti=gti,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        temp_air=tair,
        pac_real=pac,
    )

    meta = {
        "event_id": event.id,
        "plant_id": event.plant_id,
        "source_oper": source_oper,
        "source_meteo": source_meteo,
        "event_start_utc": event.ts_start_utc,
        "event_end_utc": event.ts_end_utc,
        "event_start_idx": pre_bins,
        "event_end_idx": len(ts_grid) - post_bins - 1,
    }

    win = WindowArrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        mismatch=mismatch,
        g=np.where(np.isfinite(gti), gti, ghi),
        t=tair,
        mppt_vdc=mppt_vdc,
        mppt_idc=mppt_idc,
    )
    return win, ts_grid, meta