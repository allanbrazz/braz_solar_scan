from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


CODE_OK = 0
CODE_LIMIT = 1
CODE_ANOM = 2
CODE_CRIT = 3
CODE_CRIT2 = 4


@dataclass
class RCAParams:
    sun_available_gpoa_wm2: float = 180.0
    expected_power_min_w: float = 150.0
    zero_abs_w: float = 15.0
    zero_rel_model: float = 0.02
    degraded_rel: float = 0.25
    severe_rel: float = 0.65
    low_i_ratio_warn: float = 0.35
    low_i_ratio_crit: float = 0.15
    low_v_ratio_warn: float = 0.80
    low_v_ratio_crit: float = 0.60
    vac_low_ratio: float = 0.90
    vac_high_ratio: float = 1.10
    vac_abs_margin_v: float = 10.0
    freq_abs_tol_hz: float = 1.0
    clip_margin: float = 0.98
    clip_model_margin: float = 1.02
    min_baseline_points: int = 24
    dc_open_current_rel: float = -0.85
    dc_open_voltage_rel_low: float = -0.20
    dc_short_voltage_rel: float = -0.45
    dc_short_power_rel: float = -0.70
    dc_shading_current_rel: float = -0.25
    dc_shading_voltage_abs_rel: float = 0.18
    dc_degradation_current_rel: float = -0.12
    dc_degradation_power_rel: float = -0.15
    dc_mppt_voltage_rel: float = -0.18


_DC_ANOM_LABELS = {
    "dc_shading_soiling_probable",
    "dc_degradation_probable",
    "dc_mppt_tracking_anomaly_probable",
    "dc_partial_open_circuit_probable",
    "dc_side_partial_loss_probable",
    "dc_side_voltage_anomaly_probable",
    "partial_generation_loss_probable",
    "persistent_underperformance",
}

_DC_CRIT_LABELS = {
    "dc_open_circuit_probable",
    "dc_short_circuit_probable",
}


def _to_np(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _robust_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _encode_label(label: str) -> int:
    if label in {"ok", "invalid", "low_irradiance"}:
        return CODE_OK
    if label in {"curtailment_clipping", "operational_curtailment"}:
        return CODE_LIMIT
    if label.startswith("grid_"):
        return CODE_CRIT
    if label in {"inverter_off_under_sun", "unknown_shutdown_with_sun"}:
        return CODE_CRIT
    if label in _DC_CRIT_LABELS:
        return CODE_CRIT
    if label in _DC_ANOM_LABELS:
        return CODE_ANOM
    return CODE_CRIT2 if label not in {"ok", "invalid"} else CODE_OK


def _safe_round(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), ndigits)


def _confidence_from_components(*components: float) -> float:
    score = 0.0
    for c in components:
        score += float(c)
    return max(0.0, min(1.0, score))


def diagnose_rca_series(
    *,
    anomaly: List[bool],
    valid_period: List[bool],
    coarse_period: Optional[List[bool]] = None,
    fine_period: Optional[List[bool]] = None,
    meteo_quality_ok: Optional[List[bool]] = None,
    irradiance_tier: Optional[List[str]] = None,
    mismatch_rel: List[Optional[float]],
    g_poa_wm2: List[Optional[float]],
    v_dc_v: List[Optional[float]],
    i_dc_a: List[Optional[float]],
    v_ac_v: Optional[List[Optional[float]]] = None,
    i_ac_a: Optional[List[Optional[float]]] = None,
    freq_hz: Optional[List[Optional[float]]] = None,
    alarm_code: Optional[List[Optional[float]]] = None,
    alarm_sev: Optional[List[Optional[float]]] = None,
    pac_real_w: List[Optional[float]] = None,
    pac_model_w: List[Optional[float]] = None,
    flag_inv_missing: Optional[List[bool]] = None,
    flag_meteo_missing: Optional[List[bool]] = None,
    inv_coverage: Optional[List[Optional[float]]] = None,
    pac_cap_w: Optional[float] = None,
    p_ac_residual_rel: Optional[List[Optional[float]]] = None,
    p_dc_residual_rel: Optional[List[Optional[float]]] = None,
    v_dc_residual_rel: Optional[List[Optional[float]]] = None,
    i_dc_residual_rel: Optional[List[Optional[float]]] = None,
    residual_channel_confidence: Optional[Dict[str, List[Optional[float]]]] = None,
    params: Optional[RCAParams] = None,
) -> Dict[str, Any]:
    p = params or RCAParams()

    an = np.asarray(anomaly, dtype=bool)
    vp = np.asarray(valid_period, dtype=bool)
    cp = np.asarray(coarse_period if coarse_period is not None else valid_period, dtype=bool)
    fp = np.asarray(fine_period if fine_period is not None else np.zeros_like(vp), dtype=bool)
    mq = np.asarray(meteo_quality_ok if meteo_quality_ok is not None else np.zeros_like(vp), dtype=bool)
    tiers = list(irradiance_tier or ["N"] * len(vp))

    mm = _to_np(mismatch_rel)
    gpoa = _to_np(g_poa_wm2)
    vdc = _to_np(v_dc_v)
    idc = _to_np(i_dc_a)
    vac = _to_np(v_ac_v or [None] * len(vp))
    iac = _to_np(i_ac_a or [None] * len(vp))
    freq = _to_np(freq_hz or [None] * len(vp))
    acode = _to_np(alarm_code or [None] * len(vp))
    asev = _to_np(alarm_sev or [None] * len(vp))
    pac = _to_np(pac_real_w or [None] * len(vp))
    pm = _to_np(pac_model_w or [None] * len(vp))

    pac_res = _to_np(p_ac_residual_rel or mismatch_rel)
    pdc_res = _to_np(p_dc_residual_rel or [None] * len(vp))
    vdc_res = _to_np(v_dc_residual_rel or [None] * len(vp))
    idc_res = _to_np(i_dc_residual_rel or [None] * len(vp))

    ch_conf = residual_channel_confidence or {}
    pac_conf = _to_np(ch_conf.get("p_ac") or [None] * len(vp))
    vdc_conf = _to_np(ch_conf.get("v_dc") or [None] * len(vp))
    idc_conf = _to_np(ch_conf.get("i_dc") or [None] * len(vp))

    inv_miss = np.asarray(flag_inv_missing, dtype=bool) if flag_inv_missing is not None else np.zeros_like(an)
    met_miss = np.asarray(flag_meteo_missing, dtype=bool) if flag_meteo_missing is not None else np.zeros_like(an)

    if inv_coverage is not None:
        cov = _to_np(inv_coverage)
        cov_ok = np.isfinite(cov) & (cov >= 0.30)
    else:
        cov_ok = np.ones_like(an, dtype=bool)

    base_mask = cp & (~an) & cov_ok & (~inv_miss) & (~met_miss)
    if int(base_mask.sum()) < int(p.min_baseline_points):
        base_mask = cp & cov_ok & (~inv_miss) & (~met_miss)

    vdc_med = _robust_median(vdc[base_mask])
    idc_med = _robust_median(idc[base_mask])
    vac_med = _robust_median(vac[base_mask])
    freq_med = _robust_median(freq[base_mask])

    if pac_cap_w is None:
        pac_cap_w = float(np.nanpercentile(pac[np.isfinite(pac)], 99)) if np.isfinite(pac).any() else None

    codes: List[int] = []
    labels: List[str] = []
    state_labels: List[str] = []
    domain_labels: List[str] = []
    diagnosis_labels: List[str] = []
    diagnosis_conf: List[Optional[float]] = []
    direct_grid_evidence: List[bool] = []
    zero_injection_flags: List[bool] = []
    evidence_json: List[dict] = []

    for i in range(len(an)):
        g = float(gpoa[i]) if np.isfinite(gpoa[i]) else np.nan
        sun_available = np.isfinite(g) and g >= float(p.sun_available_gpoa_wm2)

        pac_i = float(pac[i]) if np.isfinite(pac[i]) else np.nan
        pm_i = float(pm[i]) if np.isfinite(pm[i]) else np.nan
        pac_res_i = float(pac_res[i]) if np.isfinite(pac_res[i]) else np.nan
        pdc_res_i = float(pdc_res[i]) if np.isfinite(pdc_res[i]) else np.nan
        vdc_res_i = float(vdc_res[i]) if np.isfinite(vdc_res[i]) else np.nan
        idc_res_i = float(idc_res[i]) if np.isfinite(idc_res[i]) else np.nan

        zero_thr = max(
            float(p.zero_abs_w),
            float(p.zero_rel_model) * max(pm_i, 0.0) if np.isfinite(pm_i) else float(p.zero_abs_w),
        )
        zero_inj = np.isfinite(pac_i) and (pac_i <= zero_thr)
        zero_injection_flags.append(bool(zero_inj))

        degraded = False
        severe_degraded = False
        if np.isfinite(pac_i) and np.isfinite(pm_i) and pm_i >= float(p.expected_power_min_w):
            degraded = pac_i < (1.0 - float(p.degraded_rel)) * pm_i
            severe_degraded = pac_i < (1.0 - float(p.severe_rel)) * pm_i

        if not sun_available:
            state = "low_irradiance"
        elif zero_inj:
            state = "sun_available_not_injecting"
        elif degraded:
            state = "injecting_degraded"
        else:
            state = "injecting_normal"

        ev: dict[str, Any] = {
            "irradiance_tier": tiers[i],
            "sun_available": bool(sun_available),
            "meteo_quality_ok": bool(mq[i]),
            "fine_diag_allowed": bool(fp[i]),
            "residual_anomaly": bool(an[i]),
            "mismatch_rel": _safe_round(mm[i]),
            "g_poa_wm2": _safe_round(gpoa[i]),
            "pac_real_w": _safe_round(pac[i]),
            "pac_model_w": _safe_round(pm[i]),
            "vdc_v": _safe_round(vdc[i]),
            "idc_a": _safe_round(idc[i]),
            "vac_v": _safe_round(vac[i]),
            "iac_a": _safe_round(iac[i]),
            "freq_hz": _safe_round(freq[i]),
            "alarm_code": int(acode[i]) if np.isfinite(acode[i]) else None,
            "alarm_sev": int(asev[i]) if np.isfinite(asev[i]) else None,
            "p_ac_residual_rel": _safe_round(pac_res_i),
            "p_dc_residual_rel": _safe_round(pdc_res_i),
            "v_dc_residual_rel": _safe_round(vdc_res_i),
            "i_dc_residual_rel": _safe_round(idc_res_i),
            "channel_confidence": {
                "p_ac": _safe_round(pac_conf[i]),
                "v_dc": _safe_round(vdc_conf[i]),
                "i_dc": _safe_round(idc_conf[i]),
            },
        }

        if (inv_miss[i] or met_miss[i] or (not cov_ok[i])):
            label = "invalid"
            domain = "unknown"
            conf = 0.0
            state = "telemetry_invalid"
            ev["reason"] = "missing_or_low_coverage"
            grid_evd = False
        elif not vp[i]:
            label = "invalid"
            domain = "unknown"
            conf = 0.0
            ev["reason"] = "outside_operational_gate"
            grid_evd = False
        else:
            vac_low = False
            vac_high = False
            if np.isfinite(vac[i]) and np.isfinite(vac_med) and vac_med > 0.0:
                vac_low = float(vac[i]) < min(vac_med * float(p.vac_low_ratio), vac_med - float(p.vac_abs_margin_v))
                vac_high = float(vac[i]) > max(vac_med * float(p.vac_high_ratio), vac_med + float(p.vac_abs_margin_v))

            freq_low = False
            freq_high = False
            if np.isfinite(freq[i]) and np.isfinite(freq_med) and freq_med > 0.0:
                freq_low = float(freq[i]) < (freq_med - float(p.freq_abs_tol_hz))
                freq_high = float(freq[i]) > (freq_med + float(p.freq_abs_tol_hz))

            grid_evd = bool(vac_low or vac_high or freq_low or freq_high)
            ev["vac_baseline_v"] = _safe_round(vac_med)
            ev["freq_baseline_hz"] = _safe_round(freq_med)
            ev["grid_undervoltage"] = vac_low
            ev["grid_overvoltage"] = vac_high
            ev["grid_underfrequency"] = freq_low
            ev["grid_overfrequency"] = freq_high

            clip = False
            if pac_cap_w is not None and np.isfinite(pac_i) and np.isfinite(pm_i):
                clip = (pac_i >= float(p.clip_margin) * float(pac_cap_w)) and (pm_i > pac_i * float(p.clip_model_margin))

            idc_ratio = None
            if np.isfinite(idc[i]) and np.isfinite(idc_med) and idc_med > 1e-6:
                idc_ratio = float(idc[i]) / float(idc_med)
            vdc_ratio = None
            if np.isfinite(vdc[i]) and np.isfinite(vdc_med) and abs(vdc_med) > 1e-6:
                vdc_ratio = float(vdc[i]) / float(vdc_med)
            ev["idc_ratio"] = _safe_round(idc_ratio)
            ev["vdc_ratio"] = _safe_round(vdc_ratio)

            domain = "operational"
            label = "ok"
            conf = 0.25

            dc_open = bool(np.isfinite(idc_res_i) and idc_res_i <= float(p.dc_open_current_rel) and ((not np.isfinite(vdc_res_i)) or vdc_res_i >= float(p.dc_open_voltage_rel_low)))
            dc_short = bool(np.isfinite(vdc_res_i) and vdc_res_i <= float(p.dc_short_voltage_rel) and np.isfinite(pac_res_i) and pac_res_i <= float(p.dc_short_power_rel))
            dc_shading = bool(np.isfinite(idc_res_i) and idc_res_i <= float(p.dc_shading_current_rel) and ((not np.isfinite(vdc_res_i)) or abs(vdc_res_i) <= float(p.dc_shading_voltage_abs_rel)))
            dc_deg = bool(np.isfinite(idc_res_i) and idc_res_i <= float(p.dc_degradation_current_rel) and np.isfinite(pac_res_i) and pac_res_i <= float(p.dc_degradation_power_rel))
            dc_mppt = bool(np.isfinite(vdc_res_i) and vdc_res_i <= float(p.dc_mppt_voltage_rel) and ((not np.isfinite(idc_res_i)) or idc_res_i > float(p.dc_shading_current_rel)))

            ev["dc_open_signature"] = dc_open
            ev["dc_short_signature"] = dc_short
            ev["dc_shading_signature"] = dc_shading
            ev["dc_degradation_signature"] = dc_deg
            ev["dc_mppt_signature"] = dc_mppt

            if clip:
                domain = "operational"
                label = "curtailment_clipping"
                conf = _confidence_from_components(0.55, 0.10 if mq[i] else 0.0, 0.10 if tiers[i] in {"A", "B"} else 0.0)
                ev["reason"] = "pac_near_cap"
            elif grid_evd and sun_available and (zero_inj or degraded or an[i]):
                domain = "grid"
                if vac_high:
                    label = "grid_overvoltage_trip" if zero_inj else "grid_overvoltage_derating"
                elif vac_low:
                    label = "grid_undervoltage_trip" if zero_inj else "grid_undervoltage_derating"
                elif freq_high:
                    label = "grid_overfrequency_trip"
                else:
                    label = "grid_underfrequency_trip"
                conf = _confidence_from_components(0.45, 0.35, 0.10 if sun_available else 0.0, 0.05 if (zero_inj or severe_degraded) else 0.0, 0.05 if np.isfinite(acode[i]) else 0.0)
                ev["reason"] = "direct_grid_evidence"
            elif sun_available and zero_inj and dc_open:
                domain = "dc_side"
                label = "dc_open_circuit_probable"
                conf = _confidence_from_components(0.32, 0.18 if mq[i] else 0.0, 0.12 if np.isfinite(idc_conf[i]) and idc_conf[i] >= 0.6 else 0.04, 0.08 if tiers[i] == "A" else 0.03)
                ev["reason"] = "zero_injection_with_open_signature"
            elif sun_available and zero_inj and dc_short:
                domain = "dc_side"
                label = "dc_short_circuit_probable"
                conf = _confidence_from_components(0.32, 0.18 if mq[i] else 0.0, 0.12 if np.isfinite(vdc_conf[i]) and vdc_conf[i] >= 0.6 else 0.04, 0.08 if tiers[i] == "A" else 0.03)
                ev["reason"] = "zero_injection_with_short_signature"
            elif sun_available and zero_inj:
                domain = "inverter"
                if np.isfinite(acode[i]) or (np.isfinite(asev[i]) and float(asev[i]) >= 2.0):
                    label = "inverter_off_under_sun"
                    conf = _confidence_from_components(0.35, 0.20, 0.10 if tiers[i] in {"A", "B"} else 0.0, 0.10 if mq[i] else 0.0, 0.10)
                    ev["reason"] = "zero_injection_with_alarm"
                else:
                    label = "unknown_shutdown_with_sun"
                    conf = _confidence_from_components(0.20, 0.10 if tiers[i] in {"A", "B"} else 0.0, 0.10 if mq[i] else 0.0)
                    ev["reason"] = "zero_injection_no_direct_cause"
            elif sun_available and degraded and dc_short:
                domain = "dc_side"
                label = "dc_short_circuit_probable"
                conf = _confidence_from_components(0.28, 0.16 if mq[i] else 0.0, 0.12 if np.isfinite(vdc_conf[i]) and vdc_conf[i] >= 0.6 else 0.05, 0.06 if tiers[i] in {"A", "B"} else 0.02)
                ev["reason"] = "vdc_residual_strong_negative"
            elif sun_available and degraded and dc_open and severe_degraded:
                domain = "dc_side"
                label = "dc_partial_open_circuit_probable"
                conf = _confidence_from_components(0.26, 0.16 if mq[i] else 0.0, 0.12 if np.isfinite(idc_conf[i]) and idc_conf[i] >= 0.6 else 0.05, 0.06 if tiers[i] in {"A", "B"} else 0.02)
                ev["reason"] = "idc_residual_extremely_negative"
            elif sun_available and degraded and dc_shading:
                domain = "dc_side"
                label = "dc_shading_soiling_probable"
                conf = _confidence_from_components(0.24, 0.14 if mq[i] else 0.0, 0.08 if np.isfinite(idc_conf[i]) and idc_conf[i] >= 0.6 else 0.03, 0.06 if tiers[i] in {"A", "B"} else 0.02)
                ev["reason"] = "idc_drop_with_vdc_preserved"
            elif sun_available and degraded and dc_deg:
                domain = "dc_side"
                label = "dc_degradation_probable"
                conf = _confidence_from_components(0.20, 0.12 if mq[i] else 0.0, 0.08 if np.isfinite(idc_conf[i]) and idc_conf[i] >= 0.6 else 0.03, 0.05 if bool(fp[i]) else 0.0)
                ev["reason"] = "moderate_persistent_current_loss"
            elif sun_available and degraded and dc_mppt:
                domain = "dc_side"
                label = "dc_mppt_tracking_anomaly_probable"
                conf = _confidence_from_components(0.22, 0.10 if mq[i] else 0.0, 0.10 if np.isfinite(vdc_conf[i]) and vdc_conf[i] >= 0.6 else 0.03, 0.05 if bool(fp[i]) else 0.0)
                ev["reason"] = "vdc_drop_without_matching_idc_drop"
            elif sun_available and degraded:
                if tiers[i] == "A" and bool(fp[i]) and idc_ratio is not None and idc_ratio <= float(p.low_i_ratio_warn):
                    domain = "dc_side"
                    label = "dc_side_partial_loss_probable"
                    conf = _confidence_from_components(0.25, 0.20, 0.15 if idc_ratio <= float(p.low_i_ratio_crit) else 0.05, 0.15 if mq[i] else 0.0, 0.10)
                    ev["reason"] = "low_idc_vs_baseline"
                elif tiers[i] == "A" and bool(fp[i]) and vdc_ratio is not None and vdc_ratio <= float(p.low_v_ratio_warn):
                    domain = "dc_side"
                    label = "dc_side_voltage_anomaly_probable"
                    conf = _confidence_from_components(0.25, 0.20, 0.15 if vdc_ratio <= float(p.low_v_ratio_crit) else 0.05, 0.15 if mq[i] else 0.0)
                    ev["reason"] = "low_vdc_vs_baseline"
                elif tiers[i] in {"A", "B"}:
                    domain = "dc_side"
                    label = "partial_generation_loss_probable"
                    conf = _confidence_from_components(0.20, 0.10 if tiers[i] == "A" else 0.05, 0.10 if mq[i] else 0.0, 0.10 if an[i] else 0.0)
                    ev["reason"] = "pac_below_expected"
                else:
                    domain = "unknown"
                    label = "persistent_underperformance"
                    conf = _confidence_from_components(0.12, 0.08 if an[i] else 0.0)
                    ev["reason"] = "underperformance_low_irradiance_confidence"
            elif an[i] and tiers[i] in {"A", "B"}:
                domain = "unknown"
                label = "persistent_underperformance"
                conf = _confidence_from_components(0.15, 0.10 if mq[i] else 0.0, 0.10 if tiers[i] == "A" else 0.05)
                ev["reason"] = "residual_anomaly_only"
            else:
                domain = "operational"
                label = "ok"
                conf = 0.20
                ev["reason"] = "no_strong_fault_evidence"

        conf = max(0.0, min(1.0, float(conf)))
        code = _encode_label(label)

        codes.append(code)
        labels.append(label)
        state_labels.append(state)
        domain_labels.append(domain)
        diagnosis_labels.append(label)
        diagnosis_conf.append(conf)
        direct_grid_evidence.append(bool(grid_evd))
        evidence_json.append(ev)

    return {
        "codes": codes,
        "labels": labels,
        "state_labels": state_labels,
        "domain_labels": domain_labels,
        "diagnosis_labels": diagnosis_labels,
        "diagnosis_confidence": diagnosis_conf,
        "direct_grid_evidence": direct_grid_evidence,
        "zero_injection_flag": zero_injection_flags,
        "evidence_json": evidence_json,
        "baseline": {
            "vdc_median": _safe_round(vdc_med),
            "idc_median": _safe_round(idc_med),
            "vac_median": _safe_round(vac_med),
            "freq_median": _safe_round(freq_med),
            "n_base": int(base_mask.sum()),
        },
    }
