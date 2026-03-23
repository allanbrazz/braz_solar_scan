# core/services/data_loaders.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from django.db.models import F

# >>> TROQUE pelos seus models reais <<<
from core.models import InverterOperationalData, MeteoRecord  # noqa: F401


def _model_field_names(model) -> set[str]:
    return {f.name for f in model._meta.get_fields() if hasattr(f, "name")}


def _pick_field(model, candidates: List[str]) -> Optional[str]:
    fields = _model_field_names(model)
    for c in candidates:
        if c in fields:
            return c
    return None


# candidatos -> tente cobrir variações comuns do seu schema
_INV_MAP: Dict[str, List[str]] = {
    "ts_utc":   ["ts_utc", "timestamp_utc", "ts", "timestamp"],
    "p_dc_w":   ["p_dc_w", "pdc_w", "p_dc", "pdc"],
    "p_ac_w":   ["p_ac_w", "pac_w", "p_ac", "pac"],
    "v_dc_v":   ["v_dc_v", "vdc_v", "v_dc", "vdc"],
    "i_dc_a":   ["i_dc_a", "idc_a", "i_dc", "idc"],
    "v_ac_v":   ["v_ac_v", "vac_v", "v_ac", "vac"],
    "i_ac_a":   ["i_ac_a", "iac_a", "i_ac", "iac"],
}

_MET_MAP: Dict[str, List[str]] = {
    "ts_utc":      ["ts_utc", "timestamp_utc", "ts", "timestamp"],
    "ghi":         ["ghi"],
    "dni":         ["dni"],
    "dhi":         ["dhi"],
    "gti":         ["gti", "poa", "g_poa", "g_poa_wm2"],
    "temp_air":    ["temp_air", "tamb_c", "t_amb", "temperature"],
    "wind_speed":  ["wind_speed", "vento", "wind"],
    "rh":          ["rh", "humidity"],
    "pressure":    ["pressure", "pressao"],
    "meteo_qc_score": ["meteo_qc_score"],
    "flag_meteo_low_confidence": ["flag_meteo_low_confidence"],
    "flag_meteo_interpolated": ["flag_meteo_interpolated"],
    "flag_meteo_outlier": ["flag_meteo_outlier"],
    "flag_meteo_artifact": ["flag_meteo_artifact"],
}


def load_inverter_df_5min(*, plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> pd.DataFrame:
    model = InverterOperationalData

    ts_field = _pick_field(model, _INV_MAP["ts_utc"])
    if not ts_field:
        raise RuntimeError("Não encontrei campo de timestamp no model de inversor.")

    # monta projection dinamicamente
    values_kwargs = {}
    for out_col, candidates in _INV_MAP.items():
        if out_col == "ts_utc":
            values_kwargs["ts_utc"] = F(ts_field)
            continue
        f = _pick_field(model, candidates)
        if f:
            values_kwargs[out_col] = F(f)

    qs = (
        model.objects
        .filter(plant_id=plant_id, **{f"{ts_field}__gte": dt0_utc, f"{ts_field}__lt": dt1_utc})
        .values(**values_kwargs)
        .order_by(ts_field)
    )

    df = pd.DataFrame.from_records(qs)
    if df.empty:
        return pd.DataFrame(columns=list(values_kwargs.keys()))

    return df


def load_meteo_df(*, plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> pd.DataFrame:
    model = MeteoRecord

    ts_field = _pick_field(model, _MET_MAP["ts_utc"])
    if not ts_field:
        raise RuntimeError("Não encontrei campo de timestamp no model meteo.")

    values_kwargs = {}
    for out_col, candidates in _MET_MAP.items():
        if out_col == "ts_utc":
            values_kwargs["ts_utc"] = F(ts_field)
            continue
        f = _pick_field(model, candidates)
        if f:
            values_kwargs[out_col] = F(f)

    qs = (
        model.objects
        .filter(plant_id=plant_id, **{f"{ts_field}__gte": dt0_utc, f"{ts_field}__lt": dt1_utc})
        .values(**values_kwargs)
        .order_by(ts_field)
    )

    df = pd.DataFrame.from_records(qs)
    if df.empty:
        return pd.DataFrame(columns=list(values_kwargs.keys()))

    return df
