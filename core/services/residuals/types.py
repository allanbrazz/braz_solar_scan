from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ResidualInputRow:
    ts_utc: datetime
    plant_id: int
    source_oper: str = ""
    source_meteo: str = ""
    p_ac_w: Optional[float] = None
    p_dc_w: Optional[float] = None
    v_dc_v: Optional[float] = None
    i_dc_a: Optional[float] = None
    v_ac_v: Optional[float] = None
    i_ac_a: Optional[float] = None
    freq_hz: Optional[float] = None
    g_poa_wm2: Optional[float] = None
    ghi_wm2: Optional[float] = None
    dni_wm2: Optional[float] = None
    dhi_wm2: Optional[float] = None
    temp_air_c: Optional[float] = None
    wind_ms: Optional[float] = None
    alarm_code: Optional[int] = None
    alarm_sev: Optional[int] = None
    inv_coverage: Optional[float] = None
    data_quality_score: Optional[float] = None
    meteo_qc_score: Optional[float] = None
    flag_meteo_missing: bool = False
    flag_meteo_low_confidence: bool = False
    flag_meteo_interpolated: bool = False
    flag_meteo_outlier: bool = False
    flag_meteo_artifact: bool = False
    flag_inv_missing: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpectedElectricalState:
    tcell_c: Optional[float]
    ee_poa_wm2: Optional[float]
    p_dc_exp_w: Optional[float]
    p_ac_exp_w: Optional[float]
    v_dc_exp_v: Optional[float]
    i_dc_exp_a: Optional[float]
    model_valid: bool
    model_notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ResidualValue:
    observed: Optional[float]
    expected: Optional[float]
    abs_residual: Optional[float]
    rel_residual: Optional[float]
    norm_residual: Optional[float]
    valid: bool
    status: str
    confidence: float
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ResidualBundle:
    ts_utc: datetime
    granularity: str
    scope_id: str
    p_ac: ResidualValue
    p_dc: ResidualValue
    v_dc: ResidualValue
    i_dc: ResidualValue
    tcell_c: Optional[float]
    ee_poa_wm2: Optional[float]
    global_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
