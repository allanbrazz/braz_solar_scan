from __future__ import annotations

from typing import Dict

from .contracts import ResidualConfig
from .types import ResidualInputRow


def channel_confidence(row: ResidualInputRow, *, has_expected: bool, channel: str, cfg: ResidualConfig) -> float:
    if not has_expected:
        return 0.0
    weights = cfg.confidence_weights
    score = 0.0

    meteo_qc = row.meteo_qc_score
    if meteo_qc is None:
        meteo_component = 0.55
        if row.flag_meteo_low_confidence:
            meteo_component = 0.35
        if row.flag_meteo_interpolated:
            meteo_component -= 0.10
        if row.flag_meteo_outlier or row.flag_meteo_artifact:
            meteo_component -= 0.20
        meteo_component = max(0.0, min(1.0, meteo_component))
    else:
        meteo_component = max(0.0, min(1.0, float(meteo_qc)))
        if row.flag_meteo_interpolated:
            meteo_component *= 0.90
        if row.flag_meteo_outlier or row.flag_meteo_artifact:
            meteo_component *= 0.70
    score += weights.get("meteo_qc", 0.35) * meteo_component

    cov = row.inv_coverage
    if cov is None:
        cov_component = 0.70 if not row.flag_inv_missing else 0.0
    else:
        cov_component = max(0.0, min(1.0, float(cov)))
    score += weights.get("coverage", 0.20) * cov_component

    inputs_component = 1.0
    obs_map = {"p_ac": row.p_ac_w, "p_dc": row.p_dc_w, "v_dc": row.v_dc_v, "i_dc": row.i_dc_a}
    if obs_map.get(channel) is None:
        inputs_component = 0.0
    if row.temp_air_c is None:
        inputs_component *= 0.70
    if row.g_poa_wm2 is None and row.ghi_wm2 is None:
        inputs_component *= 0.25
    score += weights.get("inputs", 0.25) * inputs_component

    irr = row.g_poa_wm2
    irr_component = 0.0
    if irr is not None:
        irr_f = float(irr)
        if irr_f >= float(cfg.g_poa_fine_diag_wm2):
            irr_component = 1.0
        elif irr_f >= float(cfg.g_poa_coarse_diag_wm2):
            irr_component = 0.85
        elif irr_f >= float(cfg.g_poa_vdc_min_wm2):
            irr_component = 0.70
        elif irr_f >= float(cfg.g_poa_min_wm2):
            irr_component = 0.50
    score += weights.get("irradiance", 0.20) * irr_component

    return max(0.0, min(1.0, float(score)))


def bundle_confidence(conf_map: Dict[str, float]) -> float:
    vals = [float(v) for v in conf_map.values() if v is not None]
    if not vals:
        return 0.0
    return max(0.0, min(1.0, sum(vals) / len(vals)))
