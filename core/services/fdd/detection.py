from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectionParams:
    # elegibilidade radiométrica
    sun_available_gpoa_wm2: float = 150.0
    coarse_diag_gpoa_wm2: float = 700.0
    fine_diag_gpoa_wm2: float = 800.0

    # qualidade meteorológica
    stable_cv_max: float = 0.08
    stable_ramp_max_wm2: float = 120.0
    stable_window_points: int = 6  # 6*15min = 90min

    # EWMA
    ewma_lambda: float = 0.20
    ewma_L: float = 3.0

    # CUSUM (em z-score)
    cusum_k: float = 0.50
    cusum_h: float = 8.0

    # baseline mínimo p/ estimar sigma
    min_baseline_points: int = 24

    # qualidade mínima de dados do inversor
    inv_cov_min: float = 0.30


def _to_np(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _rolling_cv(x: np.ndarray, window: int) -> np.ndarray:
    n = x.size
    cv = np.full(n, np.nan, dtype=float)
    if window <= 1:
        return cv

    for i in range(n):
        j0 = max(0, i - window + 1)
        w = x[j0 : i + 1]
        w = w[np.isfinite(w)]
        if w.size < max(3, window // 2):
            continue
        m = float(np.mean(w))
        if abs(m) < 1e-9:
            continue
        s = float(np.std(w, ddof=0))
        cv[i] = s / abs(m)
    return cv


def _rolling_abs_ramp(x: np.ndarray) -> np.ndarray:
    out = np.full(x.size, np.nan, dtype=float)
    if x.size == 0:
        return out
    out[0] = 0.0 if np.isfinite(x[0]) else np.nan
    for i in range(1, x.size):
        if np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            out[i] = abs(float(x[i]) - float(x[i - 1]))
    return out


def _robust_loc_scale(z: np.ndarray) -> Tuple[float, float]:
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    sigma = 1.4826 * mad
    if (not np.isfinite(sigma)) or sigma < 1e-6:
        sigma = float(np.std(z, ddof=0))
    if (not np.isfinite(sigma)) or sigma < 1e-6:
        sigma = 1.0
    return med, sigma


def _tier_for_g(g: float, p: DetectionParams) -> str:
    if not np.isfinite(g):
        return "N"
    if g >= float(p.fine_diag_gpoa_wm2):
        return "A"
    if g >= float(p.coarse_diag_gpoa_wm2):
        return "B"
    return "C"


def detect_anomalies(
    *,
    mismatch_rel: List[Optional[float]],
    g_poa_wm2: List[Optional[float]],
    valid_model: List[bool],
    flag_meteo_missing: Optional[List[bool]] = None,
    flag_meteo_low_confidence: Optional[List[bool]] = None,
    flag_meteo_interpolated: Optional[List[bool]] = None,
    flag_inv_missing: Optional[List[bool]] = None,
    inv_coverage: Optional[List[Optional[float]]] = None,
    params: Optional[DetectionParams] = None,
) -> Dict[str, Any]:
    """
    Saídas principais:
      - valid_period: há irradiância suficiente para avaliação operacional básica
      - coarse_period: residual pode apoiar diagnóstico grosseiro (>=700 W/m²)
      - fine_period: residual fino permitido (>=800 W/m² + meteo ok)
      - meteo_quality_ok: estabilidade meteorológica aprovada
      - anomaly: anomalia por residual (EWMA || CUSUM) somente onde residual é elegível
    """
    p = params or DetectionParams()

    mm = _to_np(mismatch_rel)
    g = _to_np(g_poa_wm2)

    vm = np.asarray(valid_model, dtype=bool)
    met_miss = np.asarray(flag_meteo_missing, dtype=bool) if flag_meteo_missing is not None else np.zeros_like(vm)
    met_low = np.asarray(flag_meteo_low_confidence, dtype=bool) if flag_meteo_low_confidence is not None else np.zeros_like(vm)
    met_interp = np.asarray(flag_meteo_interpolated, dtype=bool) if flag_meteo_interpolated is not None else np.zeros_like(vm)
    inv_miss = np.asarray(flag_inv_missing, dtype=bool) if flag_inv_missing is not None else np.zeros_like(vm)

    if inv_coverage is not None:
        cov = _to_np(inv_coverage)
        cov_ok = np.isfinite(cov) & (cov >= float(p.inv_cov_min))
    else:
        cov_ok = np.ones_like(vm, dtype=bool)

    cv = _rolling_cv(g, int(p.stable_window_points))
    ramp = _rolling_abs_ramp(g)
    stable_sky = np.isfinite(cv) & (cv <= float(p.stable_cv_max))
    meteo_quality_ok = stable_sky & np.isfinite(ramp) & (ramp <= float(p.stable_ramp_max_wm2)) & (~met_low)

    data_ok = np.isfinite(g) & (~met_miss) & (~inv_miss) & cov_ok
    valid_period = data_ok & (g >= float(p.sun_available_gpoa_wm2))
    coarse_period = valid_period & vm & np.isfinite(mm) & (g >= float(p.coarse_diag_gpoa_wm2))
    residual_ready = coarse_period & meteo_quality_ok
    fine_period = valid_period & vm & np.isfinite(mm) & (g >= float(p.fine_diag_gpoa_wm2)) & meteo_quality_ok & (~met_interp)

    base = mm[residual_ready]
    med, sig = _robust_loc_scale(base)

    if base.size < int(p.min_baseline_points):
        fallback_mask = coarse_period
        med, sig = _robust_loc_scale(mm[fallback_mask])

    z = (mm - med) / sig

    lam = float(p.ewma_lambda)
    ewma = np.full_like(z, np.nan, dtype=float)
    prev = 0.0
    has_prev = False
    for i in range(z.size):
        if not residual_ready[i] or (not np.isfinite(z[i])):
            ewma[i] = np.nan
            continue
        if not has_prev:
            prev = float(z[i])
            has_prev = True
        else:
            prev = lam * float(z[i]) + (1.0 - lam) * prev
        ewma[i] = prev

    ewma_sigma = np.sqrt(lam / (2.0 - lam))
    ewma_flag = residual_ready & np.isfinite(ewma) & (np.abs(ewma) > float(p.ewma_L) * ewma_sigma)

    k = float(p.cusum_k)
    h = float(p.cusum_h)
    s_pos = np.full_like(z, 0.0, dtype=float)
    s_neg = np.full_like(z, 0.0, dtype=float)
    cusum_score = np.full_like(z, np.nan, dtype=float)
    for i in range(z.size):
        if not residual_ready[i] or (not np.isfinite(z[i])):
            s_pos[i] = 0.0
            s_neg[i] = 0.0
            cusum_score[i] = np.nan
            continue
        sp = (s_pos[i - 1] if i > 0 else 0.0)
        sn = (s_neg[i - 1] if i > 0 else 0.0)
        sp = max(0.0, sp + (float(z[i]) - k))
        sn = max(0.0, sn + (-float(z[i]) - k))
        s_pos[i] = sp
        s_neg[i] = sn
        cusum_score[i] = max(sp, sn)

    cusum_flag = residual_ready & np.isfinite(cusum_score) & (cusum_score > h)
    anomaly = ewma_flag | cusum_flag

    irr_tier = [_tier_for_g(float(v), p) for v in g]

    return {
        "valid_period": valid_period.tolist(),
        "coarse_period": coarse_period.tolist(),
        "fine_period": fine_period.tolist(),
        "stable_sky": stable_sky.tolist(),
        "meteo_quality_ok": meteo_quality_ok.tolist(),
        "meteo_low_confidence": met_low.tolist(),
        "meteo_interpolated": met_interp.tolist(),
        "irradiance_tier": irr_tier,
        "gpoa_cv": [None if (not np.isfinite(v)) else float(v) for v in cv.tolist()],
        "gpoa_ramp_abs": [None if (not np.isfinite(v)) else float(v) for v in ramp.tolist()],
        "z": [None if (not np.isfinite(v)) else float(v) for v in z.tolist()],
        "ewma_z": [None if (not np.isfinite(v)) else float(v) for v in ewma.tolist()],
        "cusum": [None if (not np.isfinite(v)) else float(v) for v in cusum_score.tolist()],
        "anomaly": anomaly.tolist(),
        "baseline": {
            "median": med,
            "sigma": sig,
            "n_base": int(np.isfinite(base).sum()),
            "sun_available_gpoa_wm2": float(p.sun_available_gpoa_wm2),
            "coarse_diag_gpoa_wm2": float(p.coarse_diag_gpoa_wm2),
            "fine_diag_gpoa_wm2": float(p.fine_diag_gpoa_wm2),
        },
    }
