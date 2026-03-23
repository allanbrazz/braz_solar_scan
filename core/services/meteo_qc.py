from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SOLAR_COLS: Tuple[str, ...] = ("ghi", "dni", "dhi", "gti")
METEO_QC_SCORE_COLS: Tuple[str, ...] = ("meteo_qc_score",)
METEO_QC_BOOL_COLS: Tuple[str, ...] = (
    "flag_meteo_low_confidence",
    "flag_meteo_interpolated",
    "flag_meteo_outlier",
    "flag_meteo_artifact",
)
METEO_QC_COLS: Tuple[str, ...] = METEO_QC_SCORE_COLS + METEO_QC_BOOL_COLS

# Heurística deliberadamente conservadora para refletir a documentação do Open-Meteo:
# 15-min solar nativo apenas na América do Norte e Europa Central. Fora disso,
# considerar como "provavelmente interpolado".
CENTRAL_EUROPE_BOUNDS = {
    "lat_min": 41.0,
    "lat_max": 59.5,
    "lon_min": -10.5,
    "lon_max": 26.0,
}
NORTH_AMERICA_BOUNDS = {
    "lat_min": 14.0,
    "lat_max": 84.5,
    "lon_min": -170.0,
    "lon_max": -50.0,
}


@dataclass(frozen=True)
class MeteoQCConfig:
    interval_min: int = 15
    source: str = "OPENMETEO"

    solar_hampel_window: int = 9
    solar_hampel_sigma: float = 4.5
    solar_hampel_floor_wm2: float = 80.0

    temp_hampel_window: int = 9
    temp_hampel_sigma: float = 4.0
    temp_hampel_floor_c: float = 6.0

    wind_hampel_window: int = 9
    wind_hampel_sigma: float = 4.0
    wind_hampel_floor_ms: float = 6.0

    solar_day_threshold_wm2: float = 50.0
    interp_artifact_tol_wm2: float = 0.5
    low_confidence_score_threshold: float = 0.75


PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ghi": (0.0, 1500.0),
    "dni": (0.0, 1500.0),
    "dhi": (0.0, 1200.0),
    "gti": (0.0, 1700.0),
    "temp_air": (-40.0, 65.0),
    "wind_speed": (0.0, 75.0),
    "rh": (0.0, 100.0),
    "pressure": (70000.0, 110000.0),
}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rolling_median_mad(s: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    win = max(3, int(window))
    med = s.rolling(win, center=True, min_periods=max(3, win // 2)).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=max(3, win // 2)).median()
    return med, mad


def _hampel_mask(
    s: pd.Series,
    *,
    window: int,
    sigma: float,
    floor_abs: float,
    valid_mask: Optional[pd.Series] = None,
) -> pd.Series:
    x = _to_num(s)
    med, mad = _rolling_median_mad(x, window)
    scale = 1.4826 * mad
    thr = np.maximum(scale * float(sigma), float(floor_abs))
    bad = x.notna() & med.notna() & thr.notna() & ((x - med).abs() > thr)
    if valid_mask is not None:
        valid_mask = valid_mask.fillna(False).astype(bool)
        bad &= valid_mask
    return bad.fillna(False)


def _in_bounds(lat: float, lon: float, box: Dict[str, float]) -> bool:
    return (
        float(box["lat_min"]) <= float(lat) <= float(box["lat_max"])
        and float(box["lon_min"]) <= float(lon) <= float(box["lon_max"])
    )


def is_likely_native_openmeteo_15min_solar(lat: Any, lon: Any) -> bool:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except Exception:
        return False
    return _in_bounds(lat_f, lon_f, CENTRAL_EUROPE_BOUNDS) or _in_bounds(lat_f, lon_f, NORTH_AMERICA_BOUNDS)


def _pick_primary_solar(df: pd.DataFrame) -> pd.Series:
    for c in ("gti", "ghi", "dni", "dhi"):
        if c in df.columns:
            s = _to_num(df[c])
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def _detect_interp_artifact(
    s: pd.Series,
    *,
    daylight_mask: pd.Series,
    tol_wm2: float,
) -> pd.Series:
    x = _to_num(s)
    d1 = x.diff()
    d2 = d1.diff().abs()
    base = x.notna() & daylight_mask.fillna(False).astype(bool) & d1.notna() & d2.notna() & (d2 <= float(tol_wm2))
    # Exige persistência do padrão linear para evitar punir séries suaves legítimas.
    art = base & base.shift(1, fill_value=False) & base.shift(-1, fill_value=False)
    return art.astype(bool)


def apply_meteo_qc(
    df: pd.DataFrame,
    *,
    lat: Any,
    lon: Any,
    cfg: Optional[MeteoQCConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aplica uma camada explícita de QC para séries meteo usadas no FDD.

    Estratégia:
      1) validação física dura (bounds);
      2) limpeza robusta por Hampel/MAD para spikes;
      3) detecção de artefato de interpolação em 15 min do Open-Meteo;
      4) score consolidado de confiança por linha.

    A função preserva os campos físicos originais, mas substitui por NaN os pontos
    claramente espúrios. Flags de QC são adicionadas ao dataframe de saída.
    """
    if df is None or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        for col in METEO_QC_COLS:
            out[col] = pd.Series(dtype=float if col == "meteo_qc_score" else bool)
        return out, {
            "rows": 0,
            "flagged_outliers": 0,
            "flagged_artifacts": 0,
            "flagged_low_confidence": 0,
            "likely_interpolated_15min": False,
        }

    conf = cfg or MeteoQCConfig()
    out = df.copy()

    for col in set(PHYSICAL_BOUNDS.keys()).intersection(out.columns):
        out[col] = _to_num(out[col])

    likely_interpolated = (
        str(conf.source or "").upper() == "OPENMETEO"
        and int(conf.interval_min) == 15
        and not is_likely_native_openmeteo_15min_solar(lat, lon)
    )

    row_outlier = pd.Series(False, index=out.index, dtype=bool)

    # 1) Bounds físicos
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in out.columns:
            continue
        s = _to_num(out[col])
        bad = s.notna() & ((s < float(lo)) | (s > float(hi)))
        row_outlier |= bad
        out.loc[bad, col] = np.nan

    # 2) Consistência radiométrica simples
    if "ghi" in out.columns and "dhi" in out.columns:
        ghi = _to_num(out["ghi"])
        dhi = _to_num(out["dhi"])
        bad = ghi.notna() & dhi.notna() & (dhi > (ghi + 25.0))
        row_outlier |= bad
        out.loc[bad, "dhi"] = np.nan

    # Máscara de dia para filtros solares
    primary = _pick_primary_solar(out)
    daylight_mask = primary.notna() & (primary >= float(conf.solar_day_threshold_wm2))

    # 3) Hampel/MAD em radiação e meteo
    for col in SOLAR_COLS:
        if col not in out.columns:
            continue
        bad = _hampel_mask(
            out[col],
            window=conf.solar_hampel_window,
            sigma=conf.solar_hampel_sigma,
            floor_abs=conf.solar_hampel_floor_wm2,
            valid_mask=daylight_mask,
        )
        row_outlier |= bad
        out.loc[bad, col] = np.nan

    if "temp_air" in out.columns:
        bad = _hampel_mask(
            out["temp_air"],
            window=conf.temp_hampel_window,
            sigma=conf.temp_hampel_sigma,
            floor_abs=conf.temp_hampel_floor_c,
        )
        row_outlier |= bad
        out.loc[bad, "temp_air"] = np.nan

    if "wind_speed" in out.columns:
        bad = _hampel_mask(
            out["wind_speed"],
            window=conf.wind_hampel_window,
            sigma=conf.wind_hampel_sigma,
            floor_abs=conf.wind_hampel_floor_ms,
        )
        row_outlier |= bad
        out.loc[bad, "wind_speed"] = np.nan

    primary_after = _pick_primary_solar(out)
    daylight_mask_after = primary_after.notna() & (primary_after >= float(conf.solar_day_threshold_wm2))
    row_artifact = (
        _detect_interp_artifact(
            primary_after,
            daylight_mask=daylight_mask_after,
            tol_wm2=conf.interp_artifact_tol_wm2,
        )
        if likely_interpolated
        else pd.Series(False, index=out.index, dtype=bool)
    )

    # 4) Score consolidado
    score = pd.Series(1.0, index=out.index, dtype=float)
    score = score - row_outlier.astype(float) * 0.55
    score = score - row_artifact.astype(float) * 0.10
    if likely_interpolated:
        score = score - 0.20

    met_missing = pd.Series(False, index=out.index, dtype=bool)
    phys_cols = [c for c in ("gti", "ghi", "dni", "dhi", "temp_air", "wind_speed", "rh", "pressure") if c in out.columns]
    if phys_cols:
        met_missing = out[phys_cols].isna().all(axis=1)
        score = score - met_missing.astype(float) * 0.35

    score = score.clip(lower=0.0, upper=1.0)
    low_conf = score < float(conf.low_confidence_score_threshold)

    out["meteo_qc_score"] = score.astype(float)
    out["flag_meteo_low_confidence"] = low_conf.astype(bool)
    out["flag_meteo_interpolated"] = bool(likely_interpolated)
    out["flag_meteo_outlier"] = row_outlier.astype(bool)
    out["flag_meteo_artifact"] = row_artifact.astype(bool)

    meta = {
        "rows": int(len(out)),
        "likely_interpolated_15min": bool(likely_interpolated),
        "flagged_outliers": int(row_outlier.sum()),
        "flagged_artifacts": int(row_artifact.sum()),
        "flagged_low_confidence": int(low_conf.sum()),
        "qc_score_mean": float(score.mean()) if len(score) else None,
        "qc_score_min": float(score.min()) if len(score) else None,
    }
    return out, meta
