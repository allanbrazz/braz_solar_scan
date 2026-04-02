from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.db.models import Count

from core.models import FaultEvent, PVPlantMergedRecord15m
from core.services.mppt_gnn_fdd.features import WindowArrays
from core.services.power_model.runtime_residuals import compute_pac_model_and_mismatch
from core.services.residuals.facade import compute_residual_series_from_observations


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


def _merged_model_field_names() -> set[str]:
    names: set[str] = set()
    try:
        for f in PVPlantMergedRecord15m._meta.get_fields():
            if getattr(f, "attname", None):
                names.add(str(f.attname))
            if getattr(f, "name", None):
                names.add(str(f.name))
    except Exception:
        pass
    return names


def _candidate_mppt_indices(max_mppt: int = 16) -> List[int]:
    names = _merged_model_field_names()
    out: List[int] = []
    for i in range(1, max_mppt + 1):
        if f"mppt{i}_vdc_v" in names or f"mppt{i}_idc_a" in names:
            out.append(i)
    return out or [1, 2, 3, 4]


def _resolve_mppt_indices(rows: List[Dict[str, Any]], requested_n_mppt: Optional[int]) -> List[int]:
    candidates = _candidate_mppt_indices(max_mppt=max(int(requested_n_mppt or 16), 16))
    observed: List[int] = []

    for i in candidates:
        v_key = f"mppt{i}_vdc_v"
        i_key = f"mppt{i}_idc_a"
        has_any = any((r.get(v_key) is not None) or (r.get(i_key) is not None) for r in rows)
        if has_any:
            observed.append(i)

    base = observed or candidates
    if requested_n_mppt:
        base = [i for i in base if i <= int(requested_n_mppt)] or [i for i in candidates if i <= int(requested_n_mppt)]

    if not base:
        base = [1, 2, 3, 4][: max(1, int(requested_n_mppt or 4))]

    top = max(base)
    return list(range(1, top + 1))


def load_event_window(
    *,
    event_id: int,
    pre_bins: int = 8,
    post_bins: int = 8,
    n_mppt: Optional[int] = None,
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

    base_fields = [
        "ts_utc",
        "p_ac_w", "v_dc_v", "i_ac_a", "v_ac_v", "freq_hz",
        "gti", "ghi", "dni", "dhi", "temp_air",
    ]
    mppt_candidates = _candidate_mppt_indices(max_mppt=max(int(n_mppt or 16), 16))
    mppt_fields: List[str] = []
    for k in mppt_candidates:
        mppt_fields.extend([f"mppt{k}_vdc_v", f"mppt{k}_idc_a"])

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=event.plant_id,
            source_oper=source_oper,
            source_meteo=source_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lte=dt1_utc,
        )
        .order_by("ts_utc")
        .values(*(base_fields + mppt_fields))
    )

    pac = _fill_on_grid(ts_grid, rows, "p_ac_w")
    vdc_total = _fill_on_grid(ts_grid, rows, "v_dc_v")
    iac = _fill_on_grid(ts_grid, rows, "i_ac_a")
    vac = _fill_on_grid(ts_grid, rows, "v_ac_v")
    freq = _fill_on_grid(ts_grid, rows, "freq_hz")
    gti = _fill_on_grid(ts_grid, rows, "gti")
    ghi = _fill_on_grid(ts_grid, rows, "ghi")
    dni = _fill_on_grid(ts_grid, rows, "dni")
    dhi = _fill_on_grid(ts_grid, rows, "dhi")
    tair = _fill_on_grid(ts_grid, rows, "temp_air")

    resolved_mppts = _resolve_mppt_indices(rows, n_mppt)
    mppt_vdc = np.full((len(resolved_mppts), len(ts_grid)), np.nan, dtype=float)
    mppt_idc = np.full((len(resolved_mppts), len(ts_grid)), np.nan, dtype=float)
    for pos, k in enumerate(resolved_mppts):
        mppt_vdc[pos] = _fill_on_grid(ts_grid, rows, f"mppt{k}_vdc_v")
        mppt_idc[pos] = _fill_on_grid(ts_grid, rows, f"mppt{k}_idc_a")

    residual_out = compute_residual_series_from_observations(
        plant=event.plant,
        times_utc=ts_grid,
        gti=list(np.asarray(gti, dtype=float).tolist()),
        ghi=list(np.asarray(ghi, dtype=float).tolist()),
        dni=list(np.asarray(dni, dtype=float).tolist()),
        dhi=list(np.asarray(dhi, dtype=float).tolist()),
        temp_air=list(np.asarray(tair, dtype=float).tolist()),
        p_ac_w=list(np.asarray(pac, dtype=float).tolist()),
        p_dc_w=[None] * len(ts_grid),
        v_dc_v=list(np.asarray(vdc_total, dtype=float).tolist()),
        i_dc_a=[None] * len(ts_grid),
        v_ac_v=list(np.asarray(vac, dtype=float).tolist()),
        i_ac_a=list(np.asarray(iac, dtype=float).tolist()),
        freq_hz=list(np.asarray(freq, dtype=float).tolist()),
        source_oper=source_oper,
        source_meteo=source_meteo,
    )
    rs = residual_out.get("series") or {}
    pac_model = np.asarray([np.nan if v is None else float(v) for v in rs.get("pac_expected_w", [None] * len(ts_grid))], dtype=float)
    mismatch = np.asarray([np.nan if v is None else float(v) for v in rs.get("p_ac_residual_rel", [None] * len(ts_grid))], dtype=float)

    inv = getattr(getattr(event.plant, "details", None), "inverter", None)
    try:
        vac_nom_v = float(getattr(inv, "v_ac_nom_v", None)) if inv is not None and getattr(inv, "v_ac_nom_v", None) is not None else None
    except Exception:
        vac_nom_v = None

    meta = {
        "event_id": event.id,
        "plant_id": event.plant_id,
        "source_oper": source_oper,
        "source_meteo": source_meteo,
        "event_start_utc": event.ts_start_utc,
        "event_end_utc": event.ts_end_utc,
        "event_start_idx": pre_bins,
        "event_end_idx": len(ts_grid) - post_bins - 1,
        "mppt_indices": resolved_mppts,
        "n_mppt": len(resolved_mppts),
        "v_ac_nom_v": vac_nom_v,
        "freq_nom_hz": float(getattr(__import__("django.conf").conf.settings, "FDD_GRID_FREQ_NOM_HZ", 50.0) or 50.0),
        "residuals": {
            "p_ac_residual_rel": rs.get("p_ac_residual_rel"),
            "p_dc_residual_rel": rs.get("p_dc_residual_rel"),
            "v_dc_residual_rel": rs.get("v_dc_residual_rel"),
            "i_dc_residual_rel": rs.get("i_dc_residual_rel"),
            "channel_confidence": rs.get("channel_confidence"),
            "pac_expected_w": rs.get("pac_expected_w"),
            "v_dc_expected_v": rs.get("v_dc_expected_v"),
            "i_dc_expected_a": rs.get("i_dc_expected_a"),
            "global_confidence": rs.get("global_confidence"),
        },
    }

    win = WindowArrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        mismatch=mismatch,
        g=np.where(np.isfinite(gti), gti, ghi),
        t=tair,
        vac=vac,
        freq=freq,
        mppt_vdc=mppt_vdc,
        mppt_idc=mppt_idc,
    )
    return win, ts_grid, meta
