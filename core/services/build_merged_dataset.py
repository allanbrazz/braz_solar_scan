from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Literal

import pandas as pd

from core.models import PVPlant
from core.services.timeseries_io import fetch_meteo_df, fetch_inverter_df, FetchConfig
from core.services.timeseries_merge import (
    InverterAggregationConfig,
    MeteoPreparationConfig,
    aggregate_inverter_to_15min,
    prepare_meteo_15min,
    join_inverter_meteo_15min,
    densify_15min_grid,
    rollup_15min_to_hour,
)
from core.services.merged15m_store import upsert_merged_15m_df


JoinHow = Literal["left", "inner", "right", "outer"]


@dataclass(frozen=True)
class MergeRunResult:
    df15: pd.DataFrame
    df_hour: pd.DataFrame
    stats: Dict[str, Any]


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_iso(dt_like) -> str:
    try:
        if dt_like is None:
            return ""
        if hasattr(dt_like, "to_pydatetime"):
            dt_like = dt_like.to_pydatetime()
        if getattr(dt_like, "tzinfo", None) is not None:
            dt_like = dt_like.astimezone(timezone.utc)
        return dt_like.isoformat()
    except Exception:
        return str(dt_like)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or x is pd.NA:
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_plant_merged_dataset(
    *,
    plant: PVPlant,
    dt_start_utc: datetime,
    dt_end_utc: datetime,
    want_hourly: bool = True,
    fetch_cfg: FetchConfig = FetchConfig(),
    inv_cfg: InverterAggregationConfig = InverterAggregationConfig(ts_col="ts_utc", freq="15min"),
    met_cfg: MeteoPreparationConfig = MeteoPreparationConfig(ts_col="ts_utc", freq="15min"),
    how: JoinHow = "left",
    persist: bool = True,
    source_oper: str = "SHINEMONITOR",
    source_meteo: str = "OPENMETEO",
    interval_min: int = 15,
    densify: bool = True,
) -> MergeRunResult:
    if how not in ("left", "inner", "right", "outer"):
        raise ValueError("how inválido. Use: left|inner|right|outer")

    dt_start_utc = _ensure_utc(dt_start_utc)
    dt_end_utc = _ensure_utc(dt_end_utc)
    if dt_end_utc <= dt_start_utc:
        raise ValueError("dt_end_utc deve ser maior que dt_start_utc")

    # 1) Extrai do banco
    df_inv = fetch_inverter_df(
        plant=plant,
        dt_start_utc=dt_start_utc,
        dt_end_utc=dt_end_utc,
        cfg=fetch_cfg,
    )
    df_met = fetch_meteo_df(
        plant=plant,
        dt_start_utc=dt_start_utc,
        dt_end_utc=dt_end_utc,
        cfg=fetch_cfg,
    )

    inv_meta = (df_inv.attrs.get("meta") or {}) if hasattr(df_inv, "attrs") else {}

    # 2) Agrega/prepara
    inv15 = aggregate_inverter_to_15min(df_inv, cfg=inv_cfg, tz_work="UTC", assume_tz_if_naive="UTC")
    met15 = prepare_meteo_15min(df_met, cfg=met_cfg, tz_work="UTC", assume_tz_if_naive="UTC")

    # 3) Join e densify
    if densify:
        # join outer para não perder buckets meteo-only
        df15_raw = join_inverter_meteo_15min(inv15, met15, how="outer")
        df15 = densify_15min_grid(
            df15_raw,
            start_utc=dt_start_utc,
            end_utc=dt_end_utc,
            freq=inv_cfg.freq,
            coverage_threshold=inv_cfg.coverage_threshold,
        )
    else:
        df15 = join_inverter_meteo_15min(inv15, met15, how=how)

    expected_15 = int((dt_end_utc - dt_start_utc).total_seconds() // (15 * 60))

    stats: Dict[str, Any] = {
        "plant_id": getattr(plant, "id", None),
        "dt_start_utc": dt_start_utc.isoformat(),
        "dt_end_utc": dt_end_utc.isoformat(),
        "how": ("outer+dense" if densify else how),

        "inv_rows_raw": int(inv_meta.get("inv_rows_raw", len(df_inv))),
        "inv_rows_in_window": int(inv_meta.get("inv_rows_in_window", len(df_inv))),
        "plant_tz": str(inv_meta.get("plant_tz", "UTC")),

        "ts_shift_h_median": _safe_float(inv_meta.get("ts_shift_h_median", 0.0), 0.0),
        "ts_shift_h_min": _safe_float(inv_meta.get("ts_shift_h_min", 0.0), 0.0),
        "ts_shift_h_max": _safe_float(inv_meta.get("ts_shift_h_max", 0.0), 0.0),

        "meteo_rows_raw": int(len(df_met)),
        "inv15_rows": int(len(inv15)),
        "met15_rows": int(len(met15)),
        "merged_rows_15": int(len(df15)),
        "expected_rows_15": int(expected_15),
    }

    if not df15.empty:
        idx = df15.index
        stats["merged_min_ts_utc"] = _safe_iso(idx.min())
        stats["merged_max_ts_utc"] = _safe_iso(idx.max())

        if "flag_meteo_missing" in df15.columns:
            stats["meteo_missing_frac"] = _safe_float(df15["flag_meteo_missing"].mean(), 0.0)

        if "flag_inv_missing" in df15.columns:
            stats["inv_missing_frac"] = _safe_float(df15["flag_inv_missing"].mean(), 0.0)
        else:
            stats["inv_missing_frac"] = 0.0

        # cobertura média apenas onde há inversor (inv_n>0), pois inv_coverage é NA quando missing
        if "inv_coverage" in df15.columns:
            stats["inv_coverage_mean_present"] = _safe_float(pd.to_numeric(df15["inv_coverage"], errors="coerce").mean(), 0.0)
        else:
            stats["inv_coverage_mean_present"] = 0.0

        # low coverage apenas onde há inversor (por construção, missing => False)
        if "flag_low_coverage" in df15.columns:
            stats["inv_lowcov_frac_present"] = _safe_float(df15["flag_low_coverage"].mean(), 0.0)
        else:
            stats["inv_lowcov_frac_present"] = 0.0

        # buckets bons: tem inversor, tem meteo e não é low coverage
        good = pd.Series(True, index=df15.index)
        if "flag_inv_missing" in df15.columns:
            good &= ~df15["flag_inv_missing"].fillna(True)
        if "flag_meteo_missing" in df15.columns:
            good &= ~df15["flag_meteo_missing"].fillna(True)
        if "flag_low_coverage" in df15.columns:
            good &= ~df15["flag_low_coverage"].fillna(True)

        stats["good_bucket_frac"] = _safe_float(good.mean(), 0.0)

    else:
        stats["merged_min_ts_utc"] = ""
        stats["merged_max_ts_utc"] = ""
        stats["meteo_missing_frac"] = 1.0
        stats["inv_missing_frac"] = 1.0
        stats["inv_coverage_mean_present"] = 0.0
        stats["inv_lowcov_frac_present"] = 0.0
        stats["good_bucket_frac"] = 0.0

    # Rollup horário
    df_hour = pd.DataFrame()
    if want_hourly and not df15.empty:
        df_hour = rollup_15min_to_hour(df15)

    # Persistência
    if persist and not df15.empty:
        saved = upsert_merged_15m_df(
            plant=plant,
            df15=df15,
            source_oper=str(source_oper),
            source_meteo=str(source_meteo),
            interval_min=int(interval_min),
        )
        stats["saved_rows_15m"] = int(saved)
    else:
        stats["saved_rows_15m"] = 0

    return MergeRunResult(df15=df15, df_hour=df_hour, stats=stats)
