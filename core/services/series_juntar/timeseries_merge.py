from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Literal

import pandas as pd

DEFAULT_TZ = "UTC"

# =========================
# Colunas padrão
# =========================
DEFAULT_INV_MEAN_COLS = (
    "p_dc_w", "p_ac_w", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a", "freq_hz", "pf", "qac_var",
    "mppt1_v_dc_v","mppt2_v_dc_v","mppt3_v_dc_v","mppt4_v_dc_v",
    "mppt1_i_dc_a","mppt2_i_dc_a","mppt3_i_dc_a","mppt4_i_dc_a",
    "mppt1_p_dc_w","mppt2_p_dc_w","mppt3_p_dc_w","mppt4_p_dc_w",
)

DEFAULT_METEO_COLS = (
    "ghi", "dni", "dhi", "gti",
    "temp_air", "wind_speed", "rh", "pressure",
)


# =========================
# Configs
# =========================
@dataclass(frozen=True)
class InverterAggregationConfig:
    ts_col: str = "ts_utc"
    freq: str = "15min"

    # agregação por MÉDIA
    mean_cols: Sequence[str] = DEFAULT_INV_MEAN_COLS

    # energia (Wh) calculada a partir de p_ac_w
    pac_col: str = "p_ac_w"

    # se None, inferido por freq/sampling_minutes (ex.: 15/5 = 3)
    expected_samples_per_bucket: Optional[int] = None

    # energia: "fixed" = dt fixo (sampling_minutes)
    #         "next"  = dt baseado no delta para o próximo sample (clamp em max_dt_minutes)
    energy_dt_mode: Literal["fixed", "next"] = "fixed"
    sampling_minutes: int = 5
    max_dt_minutes: int = 15

    coverage_threshold: float = 0.7


@dataclass(frozen=True)
class MeteoPreparationConfig:
    ts_col: str = "ts_utc"
    freq: str = "15min"
    value_cols: Sequence[str] = DEFAULT_METEO_COLS

    # Open-Meteo costuma usar timestamps como "period_start".
    meteo_time_label: Literal["period_start", "period_end", "midpoint"] = "period_start"
    duplicate_agg: Literal["mean", "median", "first"] = "mean"


# =========================
# Helpers
# =========================
def _ensure_datetime_tz(
    s: pd.Series,
    tz_work: str = DEFAULT_TZ,
    assume_tz_if_naive: str = DEFAULT_TZ,
) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(assume_tz_if_naive)
    return dt.dt.tz_convert(tz_work)


def _floor_bucket(dt: pd.Series, freq: str) -> pd.Series:
    return dt.dt.floor(freq)


def _shift_meteo_label(dt: pd.Series, freq: str, label: str) -> pd.Series:
    """
    Ajusta timestamp meteo para representar o início do bucket ("period_start").
    """
    delta = pd.Timedelta(freq)
    if label == "period_end":
        return dt - delta
    if label == "midpoint":
        return dt - (delta / 2)
    return dt  # period_start


def _expected_samples(cfg: InverterAggregationConfig) -> int:
    if cfg.expected_samples_per_bucket is not None:
        return max(1, int(cfg.expected_samples_per_bucket))
    freq_min = int(pd.Timedelta(cfg.freq).total_seconds() // 60)
    return max(1, int(round(freq_min / cfg.sampling_minutes)))


# =========================
# PASSO 2: Inversor 5min -> 15min (média)
# =========================
def aggregate_inverter_to_15min(
    df_inv: pd.DataFrame,
    cfg: InverterAggregationConfig = InverterAggregationConfig(),
    tz_work: str = DEFAULT_TZ,
    assume_tz_if_naive: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """
    Agrega amostras do inversor (tipicamente 5 min) para buckets de 15 min:
      - mean_cols: média por bucket (inclui MPPTs se presentes)
      - e_ac_wh_15: soma da energia por amostra no bucket
      - inv_n: número de amostras no bucket
      - inv_coverage: inv_n/expected (clamp 1.0)
      - flags: flag_inv_missing (False aqui),
               flag_low_coverage (inv_n>0 e coverage<threshold)
    """
    if df_inv is None or df_inv.empty:
        return pd.DataFrame()

    df = df_inv.copy()
    if cfg.ts_col not in df.columns:
        raise ValueError(f"df_inv precisa conter a coluna de timestamp '{cfg.ts_col}'")

    df[cfg.ts_col] = _ensure_datetime_tz(df[cfg.ts_col], tz_work=tz_work, assume_tz_if_naive=assume_tz_if_naive)
    df = df.dropna(subset=[cfg.ts_col]).sort_values(cfg.ts_col)

    df["ts_15"] = _floor_bucket(df[cfg.ts_col], cfg.freq)

    mean_cols = [c for c in cfg.mean_cols if c in df.columns]
    for c in mean_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Energia (Wh) por amostra a partir de p_ac_w
    if cfg.pac_col in df.columns:
        pac = pd.to_numeric(df[cfg.pac_col], errors="coerce")

        if cfg.energy_dt_mode == "fixed":
            dt_h = float(cfg.sampling_minutes) / 60.0
            df["_e_ac_wh_sample"] = pac * dt_h
        else:
            df = df.sort_values(cfg.ts_col)
            idx = df[cfg.ts_col]
            dt_next_h = (idx.shift(-1) - idx).dt.total_seconds() / 3600.0
            dt_next_h = pd.to_numeric(dt_next_h, errors="coerce")

            med = dt_next_h.dropna()
            fill = float(med.median()) if not med.empty else (float(cfg.sampling_minutes) / 60.0)
            if len(dt_next_h) > 0:
                dt_next_h.iloc[-1] = fill

            max_h = float(cfg.max_dt_minutes) / 60.0
            dt_next_h = dt_next_h.clip(lower=0.0, upper=max_h)
            df["_e_ac_wh_sample"] = pac * dt_next_h
    else:
        df["_e_ac_wh_sample"] = pd.NA

    grouped = df.groupby("ts_15", observed=True)
    out = grouped.agg({c: "mean" for c in mean_cols})

    # Energia total do bucket (Wh)
    out["e_ac_wh_15"] = grouped["_e_ac_wh_sample"].sum(min_count=1)

    # Contagem e cobertura
    out["inv_n"] = grouped.size().astype("int64")
    expected = _expected_samples(cfg)
    out["inv_coverage"] = (out["inv_n"] / expected).clip(upper=1.0)

    # Flags
    out["flag_inv_missing"] = False
    out["flag_low_coverage"] = (
        (out["inv_n"] > 0)
        & (pd.to_numeric(out["inv_coverage"], errors="coerce") < float(cfg.coverage_threshold))
    )

    out = out.sort_index()
    out.index.name = "ts_15"
    return out


# =========================
# Meteo 15min (preparo)
# =========================
def prepare_meteo_15min(
    df_met: pd.DataFrame,
    cfg: MeteoPreparationConfig = MeteoPreparationConfig(),
    tz_work: str = DEFAULT_TZ,
    assume_tz_if_naive: str = DEFAULT_TZ,
) -> pd.DataFrame:
    if df_met is None or df_met.empty:
        return pd.DataFrame()

    df = df_met.copy()
    if cfg.ts_col not in df.columns:
        raise ValueError(f"df_met precisa conter a coluna de timestamp '{cfg.ts_col}'")

    df[cfg.ts_col] = _ensure_datetime_tz(df[cfg.ts_col], tz_work=tz_work, assume_tz_if_naive=assume_tz_if_naive)
    df = df.dropna(subset=[cfg.ts_col]).sort_values(cfg.ts_col)

    df["_ts_adj"] = _shift_meteo_label(df[cfg.ts_col], cfg.freq, cfg.meteo_time_label)
    df["ts_15"] = _floor_bucket(df["_ts_adj"], cfg.freq)

    cols = [c for c in cfg.value_cols if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if not cols:
        out = df[["ts_15"]].drop_duplicates(subset=["ts_15"]).set_index("ts_15").sort_index()
        out.index.name = "ts_15"
        return out

    grouped = df.groupby("ts_15", observed=True)[cols]
    if cfg.duplicate_agg == "median":
        out = grouped.median()
    elif cfg.duplicate_agg == "first":
        out = grouped.first()
    else:
        out = grouped.mean()

    out = out.sort_index()
    out.index.name = "ts_15"
    return out


# =========================
# Join inverter + meteo
# =========================
def join_inverter_meteo_15min(
    inv15: pd.DataFrame,
    met15: pd.DataFrame,
    how: Literal["left", "inner", "right", "outer"] = "left",
    meteo_missing_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    inv15 = inv15 if inv15 is not None else pd.DataFrame()
    met15 = met15 if met15 is not None else pd.DataFrame()

    if inv15.empty and met15.empty:
        return pd.DataFrame()

    if inv15.empty:
        df = met15.copy()
        df["inv_n"] = 0
        df["inv_coverage"] = pd.NA
        df["flag_inv_missing"] = True
        df["flag_low_coverage"] = False
    else:
        df = inv15.join(met15, how=how)

        if "inv_n" in df.columns:
            df["inv_n"] = pd.to_numeric(df["inv_n"], errors="coerce").fillna(0).astype("int64")
        else:
            df["inv_n"] = 0

        df["flag_inv_missing"] = df["inv_n"] == 0

        if "inv_coverage" in df.columns:
            df["inv_coverage"] = pd.to_numeric(df["inv_coverage"], errors="coerce")
            df.loc[df["flag_inv_missing"], "inv_coverage"] = pd.NA
        else:
            df["inv_coverage"] = pd.NA

        if "flag_low_coverage" in df.columns:
            df["flag_low_coverage"] = df["flag_low_coverage"].fillna(False).astype(bool)
            df.loc[df["flag_inv_missing"], "flag_low_coverage"] = False
        else:
            df["flag_low_coverage"] = False

    # flag meteo missing (todas as colunas meteo NaN)
    if meteo_missing_cols is None:
        meteo_missing_cols = [c for c in DEFAULT_METEO_COLS if c in df.columns]
    else:
        meteo_missing_cols = [c for c in meteo_missing_cols if c in df.columns]

    if meteo_missing_cols:
        df["flag_meteo_missing"] = df[meteo_missing_cols].isna().all(axis=1)
    else:
        df["flag_meteo_missing"] = True

    return df


# =========================
# Densificar grade 15min
# =========================
def densify_15min_grid(
    df15: pd.DataFrame,
    *,
    start_utc,
    end_utc,
    freq: str = "15min",
    coverage_threshold: float = 0.7,
    meteo_missing_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if df15 is None or df15.empty:
        start = pd.Timestamp(start_utc)
        end = pd.Timestamp(end_utc)

        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

        start = start.floor(freq)
        end = end.floor(freq)
        if end <= start:
            return pd.DataFrame()

        last = end - pd.Timedelta(freq)
        idx = pd.date_range(start=start, end=last, freq=freq, tz="UTC")

        out = pd.DataFrame(index=idx)
        out.index.name = "ts_15"
        out["inv_n"] = 0
        out["inv_coverage"] = pd.NA
        out["flag_inv_missing"] = True
        out["flag_low_coverage"] = False
        out["flag_meteo_missing"] = True
        return out

    out = df15.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("densify_15min_grid exige df indexado por DatetimeIndex.")
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")

    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    start = start.floor(freq)
    end = end.floor(freq)
    if end <= start:
        return out.sort_index()

    last = end - pd.Timedelta(freq)
    full_idx = pd.date_range(start=start, end=last, freq=freq, tz="UTC")

    out = out.reindex(full_idx)
    out.index.name = "ts_15"

    if "inv_n" in out.columns:
        out["inv_n"] = pd.to_numeric(out["inv_n"], errors="coerce").fillna(0).astype("int64")
    else:
        out["inv_n"] = 0

    out["flag_inv_missing"] = out["inv_n"] == 0

    if "inv_coverage" in out.columns:
        out["inv_coverage"] = pd.to_numeric(out["inv_coverage"], errors="coerce")
    else:
        out["inv_coverage"] = pd.NA

    out.loc[out["flag_inv_missing"], "inv_coverage"] = pd.NA

    cov = pd.to_numeric(out["inv_coverage"], errors="coerce")
    out["flag_low_coverage"] = (out["inv_n"] > 0) & (cov < float(coverage_threshold))
    out["flag_low_coverage"] = out["flag_low_coverage"].fillna(False).astype(bool)

    if meteo_missing_cols is None:
        meteo_missing_cols = [c for c in DEFAULT_METEO_COLS if c in out.columns]
    else:
        meteo_missing_cols = [c for c in meteo_missing_cols if c in out.columns]

    if meteo_missing_cols:
        out["flag_meteo_missing"] = out[meteo_missing_cols].isna().all(axis=1)
    else:
        out["flag_meteo_missing"] = True

    return out


# =========================
# Pipeline completo (inv + met)
# =========================
def build_merged_15min(
    df_inv: pd.DataFrame,
    df_met: pd.DataFrame,
    inv_cfg: InverterAggregationConfig = InverterAggregationConfig(),
    met_cfg: MeteoPreparationConfig = MeteoPreparationConfig(),
    tz_work: str = DEFAULT_TZ,
    assume_tz_if_naive: str = DEFAULT_TZ,
    how: Literal["left", "inner", "right", "outer"] = "left",
) -> pd.DataFrame:
    inv15 = aggregate_inverter_to_15min(df_inv, cfg=inv_cfg, tz_work=tz_work, assume_tz_if_naive=assume_tz_if_naive)
    met15 = prepare_meteo_15min(df_met, cfg=met_cfg, tz_work=tz_work, assume_tz_if_naive=assume_tz_if_naive)
    return join_inverter_meteo_15min(inv15, met15, how=how)


# =========================
# Rollup 15min -> 1h
# =========================
def rollup_15min_to_hour(
    df15: pd.DataFrame,
    label: Literal["left", "right"] = "left",
    closed: Literal["left", "right"] = "left",
) -> pd.DataFrame:
    if df15 is None or df15.empty:
        return pd.DataFrame()

    df = df15.copy()

    # garante tipos coerentes p/ agregação
    for c in DEFAULT_INV_MEAN_COLS + DEFAULT_METEO_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("flag_low_coverage", "flag_meteo_missing", "flag_inv_missing"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    agg: Dict[str, Any] = {}

    for c in DEFAULT_INV_MEAN_COLS:
        if c in df.columns:
            agg[c] = "mean"

    if "e_ac_wh_15" in df.columns:
        agg["e_ac_wh_15"] = "sum"

    if "inv_n" in df.columns:
        agg["inv_n"] = "sum"

    if "inv_coverage" in df.columns:
        agg["inv_coverage"] = "mean"

    for c in DEFAULT_METEO_COLS:
        if c in df.columns:
            agg[c] = "mean"

    for c in ("flag_low_coverage", "flag_meteo_missing", "flag_inv_missing"):
        if c in df.columns:
            agg[c] = "max"

    out = df.resample("H", label=label, closed=closed).agg(agg).sort_index()
    out.index.name = "ts_hour"
    return out