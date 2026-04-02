from __future__ import annotations

from enum import StrEnum


class ResidualChannel(StrEnum):
    P_AC = "p_ac"
    P_DC = "p_dc"
    V_DC = "v_dc"
    I_DC = "i_dc"


class ResidualStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    LOW_IRRADIANCE = "low_irradiance"
    INSUFFICIENT_INPUT = "insufficient_input"
    MODEL_UNAVAILABLE = "model_unavailable"
    CURTAILED = "curtailed"
    CLIPPED = "clipped"


class ResidualGranularity(StrEnum):
    PLANT = "plant"
    MPPT = "mppt"
    STRING = "string"
