from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import numpy as np

from core.services.power_model.power_model import expected_and_mismatch, module_from_pvmodule, plant_from_details


def compute_expected_state(*, plant: Any, times_utc: List[Any], g_poa_used: np.ndarray, ghi: np.ndarray, dni: np.ndarray, dhi: np.ndarray, temp_air: np.ndarray, p_ac_real: np.ndarray, p_dc_real: np.ndarray | None = None, v_dc_real: np.ndarray | None = None, i_dc_real: np.ndarray | None = None) -> Dict[str, Any]:
    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
        n = len(times_utc)
        nan = np.full(n, np.nan, dtype=float)
        return {
            "valid": np.zeros(n, dtype=bool),
            "pac_expected_w": nan,
            "pdc_expected_w": nan,
            "v_dc_expected_v": nan,
            "i_dc_expected_a": nan,
            "mismatch_rel": nan,
            "mismatch_abs_w": nan,
            "tcell_c": nan,
            "v_ratio": nan,
            "i_ratio": nan,
            "meta": {"reason": "missing_module"},
        }

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

    kwargs: Dict[str, Any] = dict(
        g_poa=g_poa_used,
        tamb_c=temp_air,
        pac_real_w=p_ac_real,
        module=mod,
        plant=pl,
        ghi=(ghi if np.isfinite(ghi).any() else None),
        dni=(dni if np.isfinite(dni).any() else None),
        dhi=(dhi if np.isfinite(dhi).any() else None),
        times_utc=np.asarray(times_utc, dtype="datetime64[ns]"),
        v_dc_real_v=v_dc_real,
        i_dc_real_a=i_dc_real,
        g_min_valid=0.0,
        n_points=60,
        eps_w=50.0,
        dt_minutes=15.0,
        window_minutes=60.0,
    )
    out = expected_and_mismatch(**kwargs) or {}
    return out
