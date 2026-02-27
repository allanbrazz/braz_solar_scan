# core/services/mppt_gnn_fdd/window_loader.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from zoneinfo import ZoneInfo
from django.db.models import Count

from core.models import PVPlant, PVPlantMergedRecord15m
from core.services.mppt_gnn_fdd.features import WindowArrays
from core.services.mppt_gnn_fdd.constants import T_STEPS_DEFAULT, DT_MIN_DEFAULT, EPS


def _plant_tz(plant: PVPlant) -> ZoneInfo:
    tz_name = getattr(plant, "timezone", None) or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _pick_best_source_meteo(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(plant_id=plant_id, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc)
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")


def _pick_best_source_oper(plant_id: int, source_meteo: str, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id, source_meteo=source_meteo, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_oper")


def _grid_utc(dt0_utc: datetime, steps: int = T_STEPS_DEFAULT, dt_min: int = DT_MIN_DEFAULT) -> List[datetime]:
    out = []
    cur = dt0_utc
    for _ in range(int(steps)):
        out.append(cur)
        cur = cur + timedelta(minutes=int(dt_min))
    return out


def _fill_on_grid(ts_grid: List[datetime], rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    idx = {t: j for j, t in enumerate(ts_grid)}
    arr = np.full(len(ts_grid), np.nan, dtype=float)
    for r in rows:
        t = r["ts_utc"]
        j = idx.get(t)
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


def compute_pac_model_and_mismatch(
    *,
    plant: PVPlant,
    times_utc: List[datetime],
    gti: np.ndarray,
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    temp_air: np.ndarray,
    pac_real: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Usa o power_model do teu projeto para gerar:
      pac_model_w [T] e mismatch [T]
    """
    from dataclasses import asdict, is_dataclass
    import inspect
    from core.services.power_model.power_model import (
        expected_and_mismatch,
        module_from_pvmodule,
        plant_from_details,
        transpose_ghi_to_poa_isotropic,
    )

    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
        # fallback: modelo indisponível
        pac_model = np.full_like(pac_real, np.nan)
        mm = np.full_like(pac_real, np.nan)
        return pac_model, mm

    mod = module_from_pvmodule(details.module)
    inv = getattr(details, "inverter", None)
    pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

    pld = asdict(pl) if is_dataclass(pl) else dict(getattr(pl, "__dict__", {}))
    if pld.get("lat_deg") is None:
        pld["lat_deg"] = float(getattr(plant, "latitude", 0.0) or 0.0)
    if pld.get("lon_deg") is None:
        pld["lon_deg"] = float(getattr(plant, "longitude", 0.0) or 0.0)
    if pld.get("tilt_deg") is None:
        pld["tilt_deg"] = float(getattr(details, "tilt_deg", 0.0) or 0.0)
    if pld.get("azimuth_deg") is None:
        pld["azimuth_deg"] = float(getattr(details, "azimuth_deg", 0.0) or 0.0)
    pl = pl.__class__(**pld)

    times_np = np.asarray(times_utc, dtype="datetime64[ns]")

    # escolhe GPOA usado: gti se existe, senão transpo de ghi, senão ghi
    mask_gti = np.isfinite(gti)
    has_gti = bool(mask_gti.any())

    ghi_arg = ghi if np.isfinite(ghi).any() else None
    dni_arg = dni if np.isfinite(dni).any() else None
    dhi_arg = dhi if np.isfinite(dhi).any() else None

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
                times_utc=times_np,
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

    if has_gti:
        if g_poa_transpo is not None and g_poa_transpo.size == gti.size:
            g_poa_used = np.where(mask_gti, gti, g_poa_transpo)
        else:
            g_poa_used = gti
    else:
        if g_poa_transpo is not None and g_poa_transpo.size == gti.size:
            g_poa_used = g_poa_transpo
        else:
            g_poa_used = ghi_arg if ghi_arg is not None else np.full_like(gti, np.nan)

    sig = inspect.signature(expected_and_mismatch)
    kwargs: Dict[str, Any] = dict(
        g_poa=g_poa_used,
        tamb_c=temp_air,
        pac_real_w=pac_real,
        module=mod,
        plant=pl,
        g_min_valid=0.0,
        n_points=60,
        eps_w=50.0,
    )
    if "times_utc" in sig.parameters:
        kwargs["times_utc"] = times_np
    if "dt_minutes" in sig.parameters:
        kwargs["dt_minutes"] = float(DT_MIN_DEFAULT)
    if "window_minutes" in sig.parameters:
        kwargs["window_minutes"] = 60.0

    out = expected_and_mismatch(**kwargs) or {}
    pac_expected = np.asarray(out.get("pac_expected_w"), dtype=float) if out.get("pac_expected_w") is not None else None
    mismatch = np.asarray(out.get("mismatch_rel"), dtype=float) if out.get("mismatch_rel") is not None else None

    if pac_expected is None:
        pac_model = np.full_like(pac_real, np.nan)
    else:
        pac_model = pac_expected

    if mismatch is None:
        den = np.maximum(np.abs(pac_model), 50.0)
        mismatch = (pac_real - pac_model) / den

    return pac_model, mismatch


def load_daily_window(
    *,
    plant_id: int,
    day_local: date,
    n_mppt: int = 4,
) -> Tuple[WindowArrays, List[datetime], Dict[str, Any]]:
    """
    Retorna WindowArrays (globais+mppt), lista times_utc (grid) e meta.
    """
    plant = PVPlant.objects.filter(id=plant_id).first()
    if plant is None:
        raise ValueError("Plant not found")

    tz = _plant_tz(plant)

    dt0_local = datetime.combine(day_local, time.min, tzinfo=tz)
    dt1_local = dt0_local + timedelta(days=1)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    source_meteo = _pick_best_source_meteo(plant_id, dt0_utc, dt1_utc) or "OPENMETEO"
    source_oper = _pick_best_source_oper(plant_id, source_meteo, dt0_utc, dt1_utc) or "SHINEMONITOR"

    ts_grid = _grid_utc(dt0_utc)

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            source_oper=source_oper,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).values(
            "ts_utc",
            "p_ac_w", "v_dc_v", "i_ac_a",
            "gti", "ghi", "dni", "dhi",
            "temp_air",
            "mppt1_vdc_v","mppt2_vdc_v","mppt3_vdc_v","mppt4_vdc_v",
            "mppt1_idc_a","mppt2_idc_a","mppt3_idc_a","mppt4_idc_a",
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
        mppt_vdc[k-1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_vdc_v")
        mppt_idc[k-1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_idc_a")

    # power_model -> pac_model + mismatch
    pac_model, mm = compute_pac_model_and_mismatch(
        plant=plant,
        times_utc=ts_grid,
        gti=gti,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        temp_air=tair,
        pac_real=pac,
    )

    meta = {
        "plant_id": plant_id,
        "source_oper": source_oper,
        "source_meteo": source_meteo,
        "day_local": day_local.isoformat(),
        "tz": str(tz),
    }

    win = WindowArrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        mismatch=mm,
        g=(gti if np.isfinite(gti).any() else ghi),
        t=tair,
        mppt_vdc=mppt_vdc,
        mppt_idc=mppt_idc,
    )
    return win, ts_grid, meta