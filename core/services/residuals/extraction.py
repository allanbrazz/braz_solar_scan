from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .types import ResidualInputRow


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if x == x else None


def row_from_mapping(mapping: Dict[str, Any], *, plant_id: int, source_oper: str = "", source_meteo: str = "") -> ResidualInputRow:
    return ResidualInputRow(
        ts_utc=mapping["ts_utc"],
        plant_id=int(plant_id),
        source_oper=str(mapping.get("source_oper") or source_oper or ""),
        source_meteo=str(mapping.get("source_meteo") or source_meteo or ""),
        p_ac_w=_f(mapping.get("p_ac_w")),
        p_dc_w=_f(mapping.get("p_dc_w")),
        v_dc_v=_f(mapping.get("v_dc_v")),
        i_dc_a=_f(mapping.get("i_dc_a")),
        v_ac_v=_f(mapping.get("v_ac_v")),
        i_ac_a=_f(mapping.get("i_ac_a")),
        freq_hz=_f(mapping.get("freq_hz")),
        g_poa_wm2=_f(mapping.get("g_poa_wm2") if mapping.get("g_poa_wm2") is not None else mapping.get("gti")),
        ghi_wm2=_f(mapping.get("ghi")),
        dni_wm2=_f(mapping.get("dni")),
        dhi_wm2=_f(mapping.get("dhi")),
        temp_air_c=_f(mapping.get("temp_air")),
        wind_ms=_f(mapping.get("wind_speed")),
        alarm_code=int(mapping.get("alarm_code")) if mapping.get("alarm_code") is not None else None,
        alarm_sev=int(mapping.get("alarm_sev")) if mapping.get("alarm_sev") is not None else None,
        inv_coverage=_f(mapping.get("inv_coverage")),
        meteo_qc_score=_f(mapping.get("meteo_qc_score")),
        flag_meteo_missing=bool(mapping.get("flag_meteo_missing") or False),
        flag_meteo_low_confidence=bool(mapping.get("flag_meteo_low_confidence") or False),
        flag_meteo_interpolated=bool(mapping.get("flag_meteo_interpolated") or False),
        flag_meteo_outlier=bool(mapping.get("flag_meteo_outlier") or False),
        flag_meteo_artifact=bool(mapping.get("flag_meteo_artifact") or False),
        flag_inv_missing=bool(mapping.get("flag_inv_missing") or False),
    )


def rows_from_mappings(rows: Iterable[Dict[str, Any]], *, plant_id: int, source_oper: str = "", source_meteo: str = "") -> List[ResidualInputRow]:
    out: List[ResidualInputRow] = []
    for row in rows:
        if not row or not row.get("ts_utc"):
            continue
        out.append(row_from_mapping(row, plant_id=plant_id, source_oper=source_oper, source_meteo=source_meteo))
    return out
