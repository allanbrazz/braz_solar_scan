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
    contextual_min_points: int = 12
    sigma_floor: float = 0.08

    # fusão multicanal
    p_ac_weight: float = 0.45
    p_dc_weight: float = 0.20
    v_dc_weight: float = 0.15
    i_dc_weight: float = 0.20
    fusion_min_effective_weight: float = 0.20

    # qualidade mínima de dados do inversor
    inv_cov_min: float = 0.30

    # inflação de sigma por incerteza
    confidence_sigma_gain: float = 1.00
    low_conf_sigma_boost: float = 0.25
    interp_sigma_boost: float = 0.15


def _to_np(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _to_np_or_none(xs: Optional[List[Optional[float]]], n: int) -> np.ndarray:
    if xs is None:
        return np.full(n, np.nan, dtype=float)
    out = np.full(n, np.nan, dtype=float)
    m = min(n, len(xs))
    for i in range(m):
        v = xs[i]
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


def _robust_loc_scale(z: np.ndarray, *, sigma_floor: float = 0.08) -> Tuple[float, float]:
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, max(1.0, float(sigma_floor))
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    sigma = 1.4826 * mad
    if (not np.isfinite(sigma)) or sigma < 1e-6:
        sigma = float(np.std(z, ddof=0))
    if (not np.isfinite(sigma)) or sigma < 1e-6:
        sigma = 1.0
    sigma = max(float(sigma), float(sigma_floor))
    return med, sigma


def _tier_for_g(g: float, p: DetectionParams) -> str:
    if not np.isfinite(g):
        return "N"
    if g >= float(p.fine_diag_gpoa_wm2):
        return "A"
    if g >= float(p.coarse_diag_gpoa_wm2):
        return "B"
    return "C"


def _series_from_map(data: Optional[Dict[str, List[Optional[float]]]], key: str, n: int) -> np.ndarray:
    if not isinstance(data, dict):
        return np.full(n, np.nan, dtype=float)
    return _to_np_or_none(data.get(key), n)


def _fuse_detection_signal(
    *,
    base_signal: np.ndarray,
    residual_channels: Optional[Dict[str, List[Optional[float]]]],
    residual_channel_confidence: Optional[Dict[str, List[Optional[float]]]],
    valid_model: np.ndarray,
    params: DetectionParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[Optional[float]]]]:
    n = base_signal.size
    pdc = _series_from_map(residual_channels, "p_dc", n)
    vdc = _series_from_map(residual_channels, "v_dc", n)
    idc = _series_from_map(residual_channels, "i_dc", n)
    pac_conf = _series_from_map(residual_channel_confidence, "p_ac", n)
    pdc_conf = _series_from_map(residual_channel_confidence, "p_dc", n)
    vdc_conf = _series_from_map(residual_channel_confidence, "v_dc", n)
    idc_conf = _series_from_map(residual_channel_confidence, "i_dc", n)

    weights = {
        "p_ac": float(params.p_ac_weight),
        "p_dc": float(params.p_dc_weight),
        "v_dc": float(params.v_dc_weight),
        "i_dc": float(params.i_dc_weight),
    }
    nominal_weight_sum = max(1e-9, sum(weights.values()))

    fused = np.full(n, np.nan, dtype=float)
    support = np.zeros(n, dtype=float)
    mean_conf = np.full(n, np.nan, dtype=float)
    contrib = {
        "p_ac": [None] * n,
        "p_dc": [None] * n,
        "v_dc": [None] * n,
        "i_dc": [None] * n,
    }

    for i in range(n):
        if not bool(valid_model[i]):
            continue

        entries: List[Tuple[str, float, float, float]] = []

        if np.isfinite(base_signal[i]):
            ac_cf = float(pac_conf[i]) if np.isfinite(pac_conf[i]) else 1.0
            ac_cf = max(0.0, min(1.0, ac_cf))
            entries.append(("p_ac", float(base_signal[i]), weights["p_ac"], ac_cf))

        for name, arr, carr in (
            ("p_dc", pdc, pdc_conf),
            ("v_dc", vdc, vdc_conf),
            ("i_dc", idc, idc_conf),
        ):
            if not np.isfinite(arr[i]):
                continue
            cf = float(carr[i]) if np.isfinite(carr[i]) else 0.75
            cf = max(0.0, min(1.0, cf))
            entries.append((name, float(arr[i]), weights[name], cf))

        num = 0.0
        den = 0.0
        conf_num = 0.0
        for name, value, base_w, conf in entries:
            eff_w = base_w * conf
            if eff_w <= 0.0:
                continue
            num += eff_w * value
            den += eff_w
            conf_num += eff_w
            contrib[name][i] = eff_w

        if den <= 0.0:
            continue

        fused[i] = num / den
        support[i] = den / nominal_weight_sum
        mean_conf[i] = conf_num / max(1e-9, sum(v for _, _, v, _ in entries)) if entries else np.nan

        if support[i] < float(params.fusion_min_effective_weight) and np.isfinite(base_signal[i]):
            fused[i] = float(base_signal[i])

    return fused, support, mean_conf, contrib


def _contextual_baseline(
    signal: np.ndarray,
    g_poa: np.ndarray,
    base_mask: np.ndarray,
    *,
    fine_threshold: float,
    min_points: int,
    sigma_floor: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    n = signal.size
    med_arr = np.full(n, np.nan, dtype=float)
    sig_arr = np.full(n, np.nan, dtype=float)

    g_med, g_sig = _robust_loc_scale(signal[base_mask], sigma_floor=sigma_floor)
    g_n = int(np.isfinite(signal[base_mask]).sum())

    coarse_mask = base_mask & np.isfinite(g_poa) & (g_poa < float(fine_threshold))
    fine_mask = base_mask & np.isfinite(g_poa) & (g_poa >= float(fine_threshold))

    c_med, c_sig = _robust_loc_scale(signal[coarse_mask], sigma_floor=sigma_floor)
    c_n = int(np.isfinite(signal[coarse_mask]).sum())
    f_med, f_sig = _robust_loc_scale(signal[fine_mask], sigma_floor=sigma_floor)
    f_n = int(np.isfinite(signal[fine_mask]).sum())

    use_coarse = c_n >= int(min_points)
    use_fine = f_n >= int(min_points)

    for i in range(n):
        if np.isfinite(g_poa[i]) and g_poa[i] >= float(fine_threshold) and use_fine:
            med_arr[i] = f_med
            sig_arr[i] = f_sig
        elif np.isfinite(g_poa[i]) and use_coarse:
            med_arr[i] = c_med
            sig_arr[i] = c_sig
        else:
            med_arr[i] = g_med
            sig_arr[i] = g_sig

    details = {
        "global": {"median": float(g_med), "sigma": float(g_sig), "n": g_n},
        "coarse": {"median": float(c_med), "sigma": float(c_sig), "n": c_n, "active": bool(use_coarse)},
        "fine": {"median": float(f_med), "sigma": float(f_sig), "n": f_n, "active": bool(use_fine)},
    }
    return med_arr, sig_arr, details


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
    residual_channels: Optional[Dict[str, List[Optional[float]]]] = None,
    residual_channel_confidence: Optional[Dict[str, List[Optional[float]]]] = None,
    params: Optional[DetectionParams] = None,
) -> Dict[str, Any]:
    """
    Saídas principais:
      - valid_period: há irradiância suficiente para avaliação operacional básica
      - coarse_period: residual pode apoiar diagnóstico grosseiro (>=700 W/m²)
      - fine_period: residual fino permitido (>=800 W/m² + meteo ok + sem interpolação)
      - meteo_quality_ok: estabilidade meteorológica aprovada para avaliação residual
      - anomaly: anomalia por residual (EWMA || CUSUM) somente onde residual é elegível

    Compatibilidade:
      - se residual_channels não for fornecido, o detector permanece operando sobre mismatch_rel
      - mismatch_rel continua sendo o canal AC principal; os demais canais entram como confirmação ponderada
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
    meteo_quality_ok = stable_sky & np.isfinite(ramp) & (ramp <= float(p.stable_ramp_max_wm2))

    fused_signal, fused_support, fused_confidence, channel_weights = _fuse_detection_signal(
        base_signal=mm,
        residual_channels=residual_channels,
        residual_channel_confidence=residual_channel_confidence,
        valid_model=vm,
        params=p,
    )

    data_ok = np.isfinite(g) & (~met_miss) & (~inv_miss) & cov_ok
    valid_period = data_ok & (g >= float(p.sun_available_gpoa_wm2))
    coarse_period = valid_period & vm & np.isfinite(fused_signal) & (g >= float(p.coarse_diag_gpoa_wm2))
    residual_ready = coarse_period & meteo_quality_ok
    fine_period = (
        valid_period
        & vm
        & np.isfinite(fused_signal)
        & (g >= float(p.fine_diag_gpoa_wm2))
        & meteo_quality_ok
        & (~met_low)
        & (~met_interp)
    )

    base_mask = residual_ready
    if int(base_mask.sum()) < int(p.min_baseline_points):
        base_mask = coarse_period

    ctx_med, ctx_sig, ctx_details = _contextual_baseline(
        fused_signal,
        g,
        base_mask,
        fine_threshold=float(p.fine_diag_gpoa_wm2),
        min_points=int(p.contextual_min_points),
        sigma_floor=float(p.sigma_floor),
    )

    sigma_eff = np.asarray(ctx_sig, dtype=float)
    for i in range(sigma_eff.size):
        infl = 1.0
        conf_i = float(fused_confidence[i]) if np.isfinite(fused_confidence[i]) else 0.75
        infl += float(p.confidence_sigma_gain) * max(0.0, 1.0 - conf_i)
        if met_low[i]:
            infl += float(p.low_conf_sigma_boost)
        if met_interp[i]:
            infl += float(p.interp_sigma_boost)
        sigma_eff[i] = max(float(sigma_eff[i]) * infl, float(p.sigma_floor))

    z = (fused_signal - ctx_med) / sigma_eff
    z[~np.isfinite(fused_signal)] = np.nan

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
        "detection_signal_rel": [None if (not np.isfinite(v)) else float(v) for v in fused_signal.tolist()],
        "detection_signal_support": [None if (not np.isfinite(v)) else float(v) for v in fused_support.tolist()],
        "detection_signal_confidence": [None if (not np.isfinite(v)) else float(v) for v in fused_confidence.tolist()],
        "contextual_sigma": [None if (not np.isfinite(v)) else float(v) for v in sigma_eff.tolist()],
        "channel_weights": channel_weights,
        "baseline": {
            "median": float(ctx_details["global"]["median"]),
            "sigma": float(ctx_details["global"]["sigma"]),
            "n_base": int(ctx_details["global"]["n"]),
            "sun_available_gpoa_wm2": float(p.sun_available_gpoa_wm2),
            "coarse_diag_gpoa_wm2": float(p.coarse_diag_gpoa_wm2),
            "fine_diag_gpoa_wm2": float(p.fine_diag_gpoa_wm2),
            "contextual": ctx_details,
        },
    }
