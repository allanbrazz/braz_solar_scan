from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple
import inspect

import numpy as np

from core.models import PVPlant
from core.services.mppt_gnn_fdd.constants import DT_MIN_DEFAULT
from core.services.power_model.power_model import (
    expected_and_mismatch,
    module_from_pvmodule,
    plant_from_details,
    transpose_ghi_to_poa_isotropic,
)


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
    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
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
            trans_kwargs: Dict[str, Any] = dict(
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
                trans_kwargs["times_shift_minutes"] = float(
                    getattr(pl, "meteo_time_shift_minutes", 0.0) or 0.0
                )
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
    pac_expected = (
        np.asarray(out.get("pac_expected_w"), dtype=float)
        if out.get("pac_expected_w") is not None
        else None
    )
    mismatch = (
        np.asarray(out.get("mismatch_rel"), dtype=float)
        if out.get("mismatch_rel") is not None
        else None
    )

    pac_model = pac_expected if pac_expected is not None else np.full_like(pac_real, np.nan)
    if mismatch is None:
        den = np.maximum(np.abs(pac_model), 50.0)
        mismatch = (pac_real - pac_model) / den

    return pac_model, mismatch
