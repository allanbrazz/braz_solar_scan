from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class ResidualConfig:
    g_poa_min_wm2: float = 50.0
    g_poa_vdc_min_wm2: float = 150.0
    g_poa_coarse_diag_wm2: float = 700.0
    g_poa_fine_diag_wm2: float = 800.0
    eps_power_w: float = 50.0
    eps_current_a: float = 0.25
    eps_voltage_v: float = 5.0
    vdc_rel_clip: float = 2.0
    idc_rel_clip: float = 2.0
    pdc_rel_clip: float = 2.0
    pac_rel_clip: float = 2.0
    inv_cov_min: float = 0.30
    noct_default_c: float = 45.0
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "meteo_qc": 0.35,
        "coverage": 0.20,
        "inputs": 0.25,
        "irradiance": 0.20,
    })
