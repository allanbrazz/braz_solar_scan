from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

from core.services.fdd_mismatch import MismatchThresholds


@dataclass(frozen=True)
class MismatchDashboardParams:
    raw_data: Mapping[str, Any]
    start: date
    end: date
    tz_name: str
    tz: ZoneInfo
    dt0_utc: datetime
    dt1_utc: datetime
    source_oper_raw: str
    source_meteo: Optional[str]
    gpoa_gate: float
    pmin_w: float
    thr: MismatchThresholds
    use_legacy: bool
    persist: bool
    gpoa_plot_min: float
    pmodel_plot_min: float
    mismatch_clip_abs: float
    display_mode: str = "mismatch"

    def get_float(self, key: str, default: float) -> float:
        raw = self.raw_data.get(key)
        raw = raw.strip() if hasattr(raw, "strip") else raw
        if raw in (None, ""):
            return float(default)
        try:
            return float(str(raw).replace(",", "."))
        except Exception:
            return float(default)

    def get_int(self, key: str, default: int) -> int:
        raw = self.raw_data.get(key)
        raw = raw.strip() if hasattr(raw, "strip") else raw
        if raw in (None, ""):
            return int(default)
        try:
            return int(float(str(raw).replace(",", ".")))
        except Exception:
            return int(default)
