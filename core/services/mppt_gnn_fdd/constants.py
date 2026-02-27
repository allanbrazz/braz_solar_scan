# core/services/mppt_gnn_fdd/constants.py
from __future__ import annotations

FAULT_LABEL_BY_CODE: dict[int, str] = {
    0: "normal",
    1: "mppt_disconnected",
    2: "inverter_off_under_sun",
    3: "mppt_imbalance",
    4: "curtailment_clipping",
    5: "meteo_bias",
}

FAULT_CODE_BY_LABEL: dict[str, int] = {v: k for k, v in FAULT_LABEL_BY_CODE.items()}

N_MPPT_DEFAULT = 4
DT_MIN_DEFAULT = 15
T_STEPS_DEFAULT = 96

EPS = 1e-9