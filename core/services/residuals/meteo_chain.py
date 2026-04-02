from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.services.power_model.power_model import tcell_noct, transpose_ghi_to_poa_isotropic


def choose_effective_poa(*, times_utc: Any, gti: np.ndarray, ghi: np.ndarray, dni: np.ndarray, dhi: np.ndarray, lat_deg: Optional[float], lon_deg: Optional[float], tilt_deg: Optional[float], azimuth_deg: Optional[float], albedo: float = 0.2) -> Dict[str, Any]:
    mask_gti = np.isfinite(gti)
    out: Dict[str, Any] = {"g_poa_used": None, "transposition_used": False, "g_poa_transposed": None}
    g_poa_transposed = None
    if np.isfinite(ghi).any() and None not in (lat_deg, lon_deg, tilt_deg, azimuth_deg):
        trans = transpose_ghi_to_poa_isotropic(
            ghi=ghi,
            dhi=(dhi if np.isfinite(dhi).any() else None),
            dni=(dni if np.isfinite(dni).any() else None),
            times_utc=times_utc,
            lat_deg=float(lat_deg),
            lon_deg=float(lon_deg),
            tilt_deg=float(tilt_deg),
            azimuth_deg=float(azimuth_deg),
            albedo=float(albedo or 0.2),
        )
        g_poa_transposed = np.asarray(trans.get("g_poa"), dtype=float)
        out["transposition_used"] = True
    if mask_gti.any():
        out["g_poa_used"] = np.where(mask_gti, gti, g_poa_transposed if g_poa_transposed is not None else np.nan)
    else:
        out["g_poa_used"] = g_poa_transposed if g_poa_transposed is not None else ghi
    out["g_poa_transposed"] = g_poa_transposed
    return out


def estimate_tcell(g_poa_used: np.ndarray, temp_air: np.ndarray, noct_c: float) -> np.ndarray:
    return np.asarray(tcell_noct(g_poa_used, temp_air, noct_c=float(noct_c or 45.0)), dtype=float)
