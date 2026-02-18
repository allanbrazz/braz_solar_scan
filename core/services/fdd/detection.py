# core/services/fdd_detection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectionParams:
    # gate de POA para “período válido”
    gpoa_gate_wm2: float = 300.0

    # estabilidade do céu medida pelo CV (std/mean) de GPOA numa janela
    stable_cv_max: float = 0.08
    stable_window_points: int = 6  # 6*15min = 90min

    # EWMA
    ewma_lambda: float = 0.20  # ~75min de constante de tempo (1/lambda pontos)
    ewma_L: float = 3.0        # limite em desvios (para EWMA z)

    # CUSUM (em z-score)
    cusum_k: float = 0.50      # slack
    cusum_h: float = 8.0       # limiar

    # baseline mínimo p/ estimar sigma
    min_baseline_points: int = 24  # 24*15min = 6h de pontos válidos

    # qualidade mínima de dados
    inv_cov_min: float = 0.30


def _to_np(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _rolling_cv(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling CV ignorando NaN. Retorna NaN quando não há dados suficientes."""
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


def _robust_loc_scale(z: np.ndarray) -> Tuple[float, float]:
    """(median, sigma) com sigma via MAD; fallback para std."""
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


def detect_anomalies(
    *,
    mismatch_rel: List[Optional[float]],
    g_poa_wm2: List[Optional[float]],
    valid_model: List[bool],
    flag_meteo_missing: Optional[List[bool]] = None,
    flag_inv_missing: Optional[List[bool]] = None,
    inv_coverage: Optional[List[Optional[float]]] = None,
    params: Optional[DetectionParams] = None,
) -> Dict[str, Any]:
    """
    Saídas principais:
      - valid_period: pontos em que podemos detectar (gate + stable + sem missing + valid_model)
      - stable_sky: estabilidade baseada em CV(GPOA)
      - anomaly: flag final (EWMA || CUSUM) em pontos valid_period
      - score_*: scores úteis para debug/plot
    """
    p = params or DetectionParams()

    mm = _to_np(mismatch_rel)
    g = _to_np(g_poa_wm2)

    vm = np.asarray(valid_model, dtype=bool)
    met_miss = np.asarray(flag_meteo_missing, dtype=bool) if flag_meteo_missing is not None else np.zeros_like(vm)
    inv_miss = np.asarray(flag_inv_missing, dtype=bool) if flag_inv_missing is not None else np.zeros_like(vm)

    if inv_coverage is not None:
        cov = _to_np(inv_coverage)
        cov_ok = np.isfinite(cov) & (cov >= float(p.inv_cov_min))
    else:
        cov_ok = np.ones_like(vm, dtype=bool)

    # Céu estável via CV(GPOA)
    cv = _rolling_cv(g, int(p.stable_window_points))
    stable_sky = np.isfinite(cv) & (cv <= float(p.stable_cv_max))

    # Período válido para DETECÇÃO
    valid_period = (
        vm
        & np.isfinite(mm)
        & np.isfinite(g)
        & (g >= float(p.gpoa_gate_wm2))
        & stable_sky
        & (~met_miss)
        & (~inv_miss)
        & cov_ok
    )

    # baseline: somente pontos "válidos"
    base = mm[valid_period]
    med, sig = _robust_loc_scale(base)

    # Se baseline insuficiente, relaxa (ainda retorna arrays coerentes)
    if base.size < int(p.min_baseline_points):
        # usa qualquer ponto com mm finito + g alto (sem stable) como fallback
        fallback_mask = vm & np.isfinite(mm) & np.isfinite(g) & (g >= float(p.gpoa_gate_wm2)) & (~met_miss) & (~inv_miss) & cov_ok
        med, sig = _robust_loc_scale(mm[fallback_mask])

    z = (mm - med) / sig  # z-score do mismatch

    # EWMA em z-score, apenas propagando nos pontos valid_period (senão mantém)
    lam = float(p.ewma_lambda)
    ewma = np.full_like(z, np.nan, dtype=float)
    prev = 0.0
    has_prev = False
    for i in range(z.size):
        if not valid_period[i] or (not np.isfinite(z[i])):
            ewma[i] = np.nan
            continue
        if not has_prev:
            prev = float(z[i])
            has_prev = True
        else:
            prev = lam * float(z[i]) + (1.0 - lam) * prev
        ewma[i] = prev

    # Limite de EWMA (desvio do EWMA)
    # var(EWMA) = lam/(2-lam) quando entrada é N(0,1)
    ewma_sigma = np.sqrt(lam / (2.0 - lam))
    ewma_flag = valid_period & np.isfinite(ewma) & (np.abs(ewma) > float(p.ewma_L) * ewma_sigma)

    # CUSUM (two-sided) em z-score
    k = float(p.cusum_k)
    h = float(p.cusum_h)
    s_pos = np.full_like(z, 0.0, dtype=float)
    s_neg = np.full_like(z, 0.0, dtype=float)
    cusum_score = np.full_like(z, np.nan, dtype=float)
    for i in range(z.size):
        if not valid_period[i] or (not np.isfinite(z[i])):
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

    cusum_flag = valid_period & np.isfinite(cusum_score) & (cusum_score > h)

    anomaly = ewma_flag | cusum_flag

    return {
        "valid_period": valid_period.tolist(),
        "stable_sky": stable_sky.tolist(),
        "z": [None if (not np.isfinite(v)) else float(v) for v in z.tolist()],
        "ewma_z": [None if (not np.isfinite(v)) else float(v) for v in ewma.tolist()],
        "cusum": [None if (not np.isfinite(v)) else float(v) for v in cusum_score.tolist()],
        "anomaly": anomaly.tolist(),
        "baseline": {"median": med, "sigma": sig, "n_base": int(np.isfinite(base).sum())},
    }
