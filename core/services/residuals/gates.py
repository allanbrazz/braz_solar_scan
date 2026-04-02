from __future__ import annotations

from typing import Optional

from .contracts import ResidualConfig
from .enums import ResidualStatus
from .types import ResidualInputRow


def common_gate(row: ResidualInputRow, cfg: ResidualConfig) -> tuple[bool, str]:
    if row.flag_inv_missing:
        return False, ResidualStatus.INSUFFICIENT_INPUT.value
    if row.flag_meteo_missing:
        return False, ResidualStatus.INSUFFICIENT_INPUT.value
    if row.g_poa_wm2 is None or float(row.g_poa_wm2) < float(cfg.g_poa_min_wm2):
        return False, ResidualStatus.LOW_IRRADIANCE.value
    return True, ResidualStatus.VALID.value


def channel_gate(channel: str, row: ResidualInputRow, cfg: ResidualConfig) -> tuple[bool, str]:
    ok, status = common_gate(row, cfg)
    if not ok:
        return ok, status

    obs_map = {
        "p_ac": row.p_ac_w,
        "p_dc": row.p_dc_w,
        "v_dc": row.v_dc_v,
        "i_dc": row.i_dc_a,
    }
    obs = obs_map.get(channel)
    if obs is None:
        return False, ResidualStatus.INSUFFICIENT_INPUT.value
    if channel == "v_dc" and (row.g_poa_wm2 is None or float(row.g_poa_wm2) < float(cfg.g_poa_vdc_min_wm2)):
        return False, ResidualStatus.LOW_IRRADIANCE.value
    return True, ResidualStatus.VALID.value


def diagnostic_readiness(row: ResidualInputRow, cfg: ResidualConfig) -> bool:
    ok, _ = common_gate(row, cfg)
    if not ok:
        return False
    if row.g_poa_wm2 is None:
        return False
    return float(row.g_poa_wm2) >= float(cfg.g_poa_fine_diag_wm2)
