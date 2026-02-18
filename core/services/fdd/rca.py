# core/services/fdd_rca.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# mantém compatível com o seu mapa de severidade:
CODE_OK = 0
CODE_LIMIT = 1   # clipping/limitação -> "ok"
CODE_ANOM = 2
CODE_CRIT = 3
CODE_CRIT2 = 4


@dataclass
class RCAParams:
    warn_abs: float = 0.35
    fault_abs: float = 0.80

    # ratios vs baseline
    low_i_ratio_warn: float = 0.35
    low_i_ratio_crit: float = 0.15
    low_v_ratio_warn: float = 0.80
    low_v_ratio_crit: float = 0.60

    # clipping
    clip_margin: float = 0.98

    # baseline mínimo
    min_baseline_points: int = 24


def _to_np(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _robust_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def diagnose_rca_series(
    *,
    anomaly: List[bool],
    valid_period: List[bool],
    mismatch_rel: List[Optional[float]],
    v_dc_v: List[Optional[float]],
    i_dc_a: List[Optional[float]],
    pac_real_w: List[Optional[float]],
    pac_model_w: List[Optional[float]],
    flag_inv_missing: Optional[List[bool]] = None,
    flag_meteo_missing: Optional[List[bool]] = None,
    inv_coverage: Optional[List[Optional[float]]] = None,
    pac_cap_w: Optional[float] = None,
    params: Optional[RCAParams] = None,
) -> Dict[str, Any]:
    """
    RCA só entra quando anomaly=True.
    Fora de valid_period: marca invalid.
    """
    p = params or RCAParams()

    an = np.asarray(anomaly, dtype=bool)
    vp = np.asarray(valid_period, dtype=bool)

    mm = _to_np(mismatch_rel)
    vdc = _to_np(v_dc_v)
    idc = _to_np(i_dc_a)
    pac = _to_np(pac_real_w)
    pm  = _to_np(pac_model_w)

    inv_miss = np.asarray(flag_inv_missing, dtype=bool) if flag_inv_missing is not None else np.zeros_like(an)
    met_miss = np.asarray(flag_meteo_missing, dtype=bool) if flag_meteo_missing is not None else np.zeros_like(an)

    if inv_coverage is not None:
        cov = _to_np(inv_coverage)
        cov_ok = np.isfinite(cov) & (cov >= 0.30)
    else:
        cov_ok = np.ones_like(an, dtype=bool)

    # baseline para Vdc/Idc: usa pontos válidos e NÃO-anômalos
    base_mask = vp & (~an) & np.isfinite(vdc) & np.isfinite(idc) & cov_ok & (~inv_miss) & (~met_miss)
    if int(base_mask.sum()) < int(p.min_baseline_points):
        base_mask = vp & np.isfinite(vdc) & np.isfinite(idc) & cov_ok & (~inv_miss) & (~met_miss)

    vdc_med = _robust_median(vdc[base_mask])
    idc_med = _robust_median(idc[base_mask])

    # clipping: precisa cap ou fallback por percentil do próprio pac
    if pac_cap_w is None:
        # fallback conservador
        pac_cap_w = float(np.nanpercentile(pac[np.isfinite(pac)], 99)) if np.isfinite(pac).any() else None

    codes: List[int] = []
    labels: List[str] = []

    for i in range(len(an)):
        # fora do período válido: não diagnostica
        if (not vp[i]) or (not np.isfinite(mm[i])) or inv_miss[i] or met_miss[i] or (not cov_ok[i]):
            codes.append(-1)
            labels.append("invalid")
            continue

        # sem anomalia: ok
        if not an[i]:
            codes.append(CODE_OK)
            labels.append("ok")
            continue

        # ---------- RCA (somente se anomalia) ----------
        m = float(mm[i])

        # 1) Clipping / limitação AC
        is_clip = False
        if pac_cap_w is not None and np.isfinite(pac[i]) and np.isfinite(pm[i]):
            if (pac[i] >= float(p.clip_margin) * float(pac_cap_w)) and (pm[i] > pac[i] * 1.02):
                is_clip = True

        if is_clip:
            codes.append(CODE_LIMIT)
            labels.append("clipping_limit")
            continue

        # 2) mismatch POSITIVO grande (real > modelo): provável viés meteo/modelo
        if m >= float(p.warn_abs):
            if m >= float(p.fault_abs):
                codes.append(CODE_ANOM)
                labels.append("meteo_bias_underestimate")
            else:
                codes.append(CODE_ANOM)
                labels.append("meteo_bias_small")
            continue

        # 3) mismatch NEGATIVO: queda real de potência
        rv = (float(vdc[i]) / float(vdc_med)) if (np.isfinite(vdc[i]) and np.isfinite(vdc_med) and abs(vdc_med) > 1e-9) else float("nan")
        ri = (float(idc[i]) / float(idc_med)) if (np.isfinite(idc[i]) and np.isfinite(idc_med) and abs(idc_med) > 1e-9) else float("nan")

        # heurística: corrente muito baixa -> “low_current” (string offline, shading forte, desconexão)
        if np.isfinite(ri) and (ri <= float(p.low_i_ratio_crit)):
            codes.append(CODE_CRIT2 if abs(m) >= float(p.fault_abs) else CODE_CRIT)
            labels.append("low_current_string_offline")
            continue
        if np.isfinite(ri) and (ri <= float(p.low_i_ratio_warn)):
            codes.append(CODE_CRIT if abs(m) >= float(p.fault_abs) else CODE_ANOM)
            labels.append("low_current_shading_soiling")
            continue

        # tensão muito baixa -> bypass/short/MPPT issue
        if np.isfinite(rv) and (rv <= float(p.low_v_ratio_crit)):
            codes.append(CODE_CRIT2 if abs(m) >= float(p.fault_abs) else CODE_CRIT)
            labels.append("low_voltage_bypass_short_mppt")
            continue
        if np.isfinite(rv) and (rv <= float(p.low_v_ratio_warn)):
            codes.append(CODE_CRIT if abs(m) >= float(p.fault_abs) else CODE_ANOM)
            labels.append("low_voltage_anomaly")
            continue

        # caso genérico: power_loss
        if abs(m) >= float(p.fault_abs):
            codes.append(CODE_CRIT)
            labels.append("power_loss")
        else:
            codes.append(CODE_ANOM)
            labels.append("anomaly_unspecified")

    return {
        "codes": codes,
        "labels": labels,
        "baseline": {"vdc_med": None if not np.isfinite(vdc_med) else float(vdc_med),
                     "idc_med": None if not np.isfinite(idc_med) else float(idc_med),
                     "pac_cap_w": pac_cap_w},
    }
