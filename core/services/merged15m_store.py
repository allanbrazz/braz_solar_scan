# core/services/merged15m_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from django.db import transaction

from core.models import PVPlant, PVPlantMergedRecord15m


MERGED_COLS = (
    "p_dc_w", "p_ac_w", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a",
    "e_ac_wh_15",
    "inv_n", "inv_coverage", "flag_low_coverage",
    "ghi", "dni", "dhi", "gti",
    "temp_air", "wind_speed", "rh", "pressure",
    "flag_meteo_missing", "flag_inv_missing",
)


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _to_bool(v: Any) -> bool:
    try:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return bool(v)
    except Exception:
        return False


@transaction.atomic
def upsert_merged_15m_df(
    *,
    plant: PVPlant,
    df15: pd.DataFrame,
    source_oper: str,
    source_meteo: str,
    interval_min: int = 15,
    batch_size: int = 2000,
) -> int:
    """
    Persiste df15 (index ts_15) em PVPlantMergedRecord15m.
    df15 precisa estar com índice tz-aware (UTC recomendado).
    """
    if df15 is None or df15.empty:
        return 0

    if not isinstance(df15.index, pd.DatetimeIndex) or df15.index.tz is None:
        raise ValueError("df15 deve estar indexado por DatetimeIndex tz-aware (ex.: UTC).")

    # Converte índice para UTC e usa como ts_utc
    idx_utc = df15.index.tz_convert("UTC")

    objs: List[PVPlantMergedRecord15m] = []

    for i, ts in enumerate(idx_utc):
        row = df15.iloc[i]

        obj = PVPlantMergedRecord15m(
            plant=plant,
            source_oper=source_oper,
            source_meteo=source_meteo,
            interval_min=int(interval_min),
            ts_utc=ts.to_pydatetime(),

            p_dc_w=_to_float(row.get("p_dc_w")),
            p_ac_w=_to_float(row.get("p_ac_w")),
            v_dc_v=_to_float(row.get("v_dc_v")),
            i_dc_a=_to_float(row.get("i_dc_a")),
            v_ac_v=_to_float(row.get("v_ac_v")),
            i_ac_a=_to_float(row.get("i_ac_a")),

            e_ac_wh_15=_to_float(row.get("e_ac_wh_15")),

            inv_n=int(row.get("inv_n")) if pd.notna(row.get("inv_n")) else None,
            inv_coverage=_to_float(row.get("inv_coverage")),
            flag_low_coverage=_to_bool(row.get("flag_low_coverage")),

            ghi=_to_float(row.get("ghi")),
            dni=_to_float(row.get("dni")),
            dhi=_to_float(row.get("dhi")),
            gti=_to_float(row.get("gti")),

            temp_air=_to_float(row.get("temp_air")),
            wind_speed=_to_float(row.get("wind_speed")),
            rh=_to_float(row.get("rh")),
            pressure=_to_float(row.get("pressure")),

            flag_meteo_missing=_to_bool(row.get("flag_meteo_missing")),
            flag_inv_missing=_to_bool(row.get("flag_inv_missing")),
        )
        objs.append(obj)

    # Upsert
    try:
        PVPlantMergedRecord15m.objects.bulk_create(
            objs,
            batch_size=batch_size,
            update_conflicts=True,
            unique_fields=["plant", "source_oper", "source_meteo", "interval_min", "ts_utc"],
            update_fields=list(MERGED_COLS),
        )
        return len(objs)
    except TypeError:
        # fallback: sem update_conflicts
        PVPlantMergedRecord15m.objects.bulk_create(objs, batch_size=batch_size, ignore_conflicts=True)
        return len(objs)
