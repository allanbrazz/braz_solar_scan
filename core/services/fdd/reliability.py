from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional


def _clamp01(value: float | int | None) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _is_finite_number(value: Any) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return v == v and v not in {float("inf"), float("-inf")}


def confidence_level(score: Optional[float]) -> str:
    if score is None:
        return ""
    s = _clamp01(score)
    if s >= 0.85:
        return "very_high"
    if s >= 0.70:
        return "high"
    if s >= 0.50:
        return "moderate"
    if s >= 0.30:
        return "low"
    return "very_low"


def _merge_top_reasons(reason_groups: Iterable[list[str] | tuple[str, ...] | None], *, top_k: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for group in reason_groups:
        if not group:
            continue
        for reason in group:
            reason_s = str(reason or "").strip()
            if reason_s:
                counter[reason_s] += 1
    return [reason for reason, _ in counter.most_common(max(1, int(top_k)))]


def compute_data_reliability(
    *,
    row: dict[str, Any],
    pac_real_w: Optional[float],
    pac_model_w: Optional[float],
    mismatch_rel: Optional[float],
) -> dict[str, Any]:
    reasons: list[str] = []

    inv_missing = bool(row.get("flag_inv_missing"))
    met_missing = bool(row.get("flag_meteo_missing"))
    low_cov = bool(row.get("flag_low_coverage"))
    met_low = bool(row.get("flag_meteo_low_confidence"))
    met_interp = bool(row.get("flag_meteo_interpolated"))
    met_outlier = bool(row.get("flag_meteo_outlier"))
    met_artifact = bool(row.get("flag_meteo_artifact"))

    inv_cov_raw = row.get("inv_coverage")
    inv_cov = _clamp01(inv_cov_raw) if _is_finite_number(inv_cov_raw) else (0.85 if not inv_missing else 0.0)
    if inv_missing:
        reasons.append("operational telemetry missing")
    elif low_cov:
        reasons.append("operational telemetry coverage reduced")

    met_qc_raw = row.get("meteo_qc_score")
    if _is_finite_number(met_qc_raw):
        met_qc = _clamp01(met_qc_raw)
    else:
        met_qc = 0.40 if met_low else 0.65

    if met_missing:
        reasons.append("meteorological data missing")
    if met_low:
        reasons.append("meteorological source low confidence")
    if met_interp:
        reasons.append("meteorological series interpolated")
    if met_outlier:
        reasons.append("meteorological outlier flag active")
    if met_artifact:
        reasons.append("meteorological artifact flag active")

    model_ready = all(
        [
            _is_finite_number(pac_real_w),
            _is_finite_number(pac_model_w),
            _is_finite_number(mismatch_rel),
        ]
    )
    if not model_ready:
        reasons.append("power model evidence incomplete")

    irr_ready = _is_finite_number(row.get("gti")) or _is_finite_number(row.get("ghi"))
    if not irr_ready:
        reasons.append("irradiance unavailable in merged record")

    integrity_component = 1.0
    if low_cov:
        integrity_component -= 0.20
    if met_low:
        integrity_component -= 0.20
    if met_interp:
        integrity_component -= 0.10
    if met_outlier:
        integrity_component -= 0.20
    if met_artifact:
        integrity_component -= 0.25
    integrity_component = _clamp01(integrity_component)

    score = 0.0
    score += 0.30 * inv_cov
    score += 0.20 * (0.0 if met_missing else 1.0)
    score += 0.25 * met_qc
    score += 0.15 * (1.0 if model_ready else (0.40 if (_is_finite_number(pac_real_w) and irr_ready) else 0.0))
    score += 0.10 * integrity_component
    score = _clamp01(score)

    return {
        "score": score,
        "level": confidence_level(score),
        "reasons": reasons,
        "components": {
            "inv_coverage_component": round(inv_cov, 4),
            "meteo_available_component": 0.0 if met_missing else 1.0,
            "meteo_qc_component": round(met_qc, 4),
            "model_ready_component": 1.0 if model_ready else (0.40 if (_is_finite_number(pac_real_w) and irr_ready) else 0.0),
            "integrity_component": round(integrity_component, 4),
        },
    }


def compute_detection_confidence(
    *,
    data_reliability_score: Optional[float],
    valid_period: bool,
    coarse_period: bool,
    fine_period: bool,
    meteo_quality_ok: bool,
    stable_sky: bool,
    anomaly_flag: bool,
    mismatch_rel: Optional[float],
    ewma_z: Optional[float],
    cusum_score: Optional[float],
) -> dict[str, Any]:
    reasons: list[str] = []
    data_score = _clamp01(data_reliability_score)

    if not valid_period:
        reasons.append("outside operational detection gate")
    if not coarse_period:
        reasons.append("insufficient irradiance for coarse residual diagnosis")
    if not meteo_quality_ok:
        reasons.append("meteorological quality not approved for residual inference")
    if not stable_sky:
        reasons.append("unstable irradiance conditions")

    mismatch_abs = abs(float(mismatch_rel)) if _is_finite_number(mismatch_rel) else 0.0
    ewma_abs = abs(float(ewma_z)) if _is_finite_number(ewma_z) else 0.0
    cusum_abs = abs(float(cusum_score)) if _is_finite_number(cusum_score) else 0.0

    severity_component = _clamp01(mismatch_abs / 0.50)
    ewma_component = _clamp01(ewma_abs / 3.0)
    cusum_component = _clamp01(cusum_abs / 8.0)

    gate_component = 0.0
    gate_component += 0.40 if valid_period else 0.0
    gate_component += 0.20 if coarse_period else 0.0
    gate_component += 0.15 if fine_period else 0.0
    gate_component += 0.15 if meteo_quality_ok else 0.0
    gate_component += 0.10 if stable_sky else 0.0
    gate_component = _clamp01(gate_component)

    evidence_component = 0.0
    if anomaly_flag:
        evidence_component += 0.45 * severity_component
        evidence_component += 0.20 * ewma_component
        evidence_component += 0.35 * cusum_component
        if severity_component < 0.15 and ewma_component < 0.20 and cusum_component < 0.20:
            reasons.append("weak detector evidence despite anomaly flag")
    else:
        calm_component = _clamp01(1.0 - max(severity_component, ewma_component, _clamp01(cusum_component / 1.2)))
        evidence_component = calm_component * 0.45
        if valid_period and calm_component >= 0.60:
            reasons.append("detector evidence remained below alert thresholds")

    score = 0.0
    score += 0.35 * data_score
    score += 0.35 * gate_component
    score += 0.30 * _clamp01(evidence_component)
    if anomaly_flag and not valid_period:
        score *= 0.35
    score = _clamp01(score)

    return {
        "score": score,
        "level": confidence_level(score),
        "reasons": reasons,
        "components": {
            "data_component": round(data_score, 4),
            "gate_component": round(gate_component, 4),
            "severity_component": round(severity_component, 4),
            "ewma_component": round(ewma_component, 4),
            "cusum_component": round(cusum_component, 4),
        },
    }


_AMBIGUOUS_LABEL_CAPS = {
    "unknown_shutdown_with_sun": 0.70,
    "persistent_underperformance": 0.65,
    "partial_generation_loss_probable": 0.75,
    "dc_side_partial_loss_probable": 0.80,
    "dc_side_voltage_anomaly_probable": 0.80,
}


_DIRECT_EVIDENCE_LABELS = {
    "grid_overvoltage_trip",
    "grid_overvoltage_derating",
    "grid_undervoltage_trip",
    "grid_undervoltage_derating",
    "grid_overfrequency_trip",
    "grid_underfrequency_trip",
    "inverter_off_under_sun",
    "curtailment_clipping",
}


def compute_diagnosis_confidence(
    *,
    diagnosis_label: str,
    base_diagnosis_confidence: Optional[float],
    data_reliability_score: Optional[float],
    detection_confidence_score: Optional[float],
    fine_diag_allowed: bool,
    meteo_quality_ok: bool,
    direct_grid_evidence: bool,
    zero_injection_flag: bool,
    irradiance_tier: str,
) -> dict[str, Any]:
    label = str(diagnosis_label or "invalid").strip() or "invalid"
    reasons: list[str] = []

    if label == "invalid":
        reasons.append("diagnosis intentionally invalid due to insufficient telemetry or gate")
        return {
            "score": 0.0,
            "level": confidence_level(0.0),
            "reasons": reasons,
            "components": {
                "base_component": 0.0,
                "data_component": _clamp01(data_reliability_score),
                "detection_component": _clamp01(detection_confidence_score),
                "specificity_component": 0.0,
            },
        }

    base_component = _clamp01(base_diagnosis_confidence)
    data_component = _clamp01(data_reliability_score)
    detection_component = _clamp01(detection_confidence_score)

    specificity_component = 0.0
    if direct_grid_evidence or label in _DIRECT_EVIDENCE_LABELS:
        specificity_component = 1.0
    elif fine_diag_allowed:
        specificity_component = 0.85
    elif irradiance_tier in {"A", "B"}:
        specificity_component = 0.55
    else:
        specificity_component = 0.30

    if not meteo_quality_ok:
        reasons.append("diagnosis built without fully approved meteorological quality")
    if not fine_diag_allowed and label not in _DIRECT_EVIDENCE_LABELS:
        reasons.append("diagnosis based on coarse evidence")
    if direct_grid_evidence:
        reasons.append("direct grid-side evidence available")
    if zero_injection_flag and label in {"unknown_shutdown_with_sun", "inverter_off_under_sun"}:
        reasons.append("zero injection observed under sun-available conditions")

    score = 0.0
    score += 0.45 * base_component
    score += 0.20 * data_component
    score += 0.20 * detection_component
    score += 0.15 * specificity_component
    score = _clamp01(score)

    cap = _AMBIGUOUS_LABEL_CAPS.get(label)
    if cap is not None:
        score = min(score, float(cap))
        reasons.append("diagnosis label is intentionally treated as probabilistic")

    if label in _DIRECT_EVIDENCE_LABELS and (direct_grid_evidence or fine_diag_allowed):
        score = max(score, 0.70)

    return {
        "score": score,
        "level": confidence_level(score),
        "reasons": reasons,
        "components": {
            "base_component": round(base_component, 4),
            "data_component": round(data_component, 4),
            "detection_component": round(detection_component, 4),
            "specificity_component": round(specificity_component, 4),
        },
    }


def aggregate_event_confidence(
    *,
    data_scores: Iterable[Optional[float]],
    detection_scores: Iterable[Optional[float]],
    diagnosis_scores: Iterable[Optional[float]],
    diagnosis_labels: Iterable[str],
    per_bin_notes: Iterable[dict[str, Any] | None],
    n_bins: int,
) -> dict[str, Any]:
    def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
        vals = []
        for x in xs:
            if x is None:
                continue
            try:
                vals.append(float(x))
            except Exception:
                continue
        if not vals:
            return None
        return sum(vals) / len(vals)

    data_score = _mean(data_scores)
    det_score = _mean(detection_scores)
    diag_score = _mean(diagnosis_scores)

    labels = [str(x or "unknown") for x in diagnosis_labels]
    dom_ratio = 0.0
    if labels:
        label_counter = Counter(labels)
        dom_ratio = max(label_counter.values()) / max(1, len(labels))
    persistence_bonus = 0.10 if n_bins >= 8 else (0.06 if n_bins >= 4 else (0.03 if n_bins >= 2 else 0.0))
    consistency_bonus = 0.08 * dom_ratio if dom_ratio > 0 else 0.0

    det_score_evt = None if det_score is None else _clamp01(det_score + persistence_bonus)
    diag_score_evt = None if diag_score is None else _clamp01(diag_score + persistence_bonus + consistency_bonus)

    notes = [n or {} for n in per_bin_notes]
    data_reasons = _merge_top_reasons((n.get("data_reliability", {}) or {}).get("reasons") for n in notes)
    detection_reasons = _merge_top_reasons((n.get("detection_confidence", {}) or {}).get("reasons") for n in notes)
    diagnosis_reasons = _merge_top_reasons((n.get("diagnosis_confidence", {}) or {}).get("reasons") for n in notes)

    return {
        "data_reliability_score": data_score,
        "data_reliability_level": confidence_level(data_score),
        "detection_confidence_score": det_score_evt,
        "detection_confidence_level": confidence_level(det_score_evt),
        "diagnosis_confidence_score": diag_score_evt,
        "diagnosis_confidence_level": confidence_level(diag_score_evt),
        "confidence_notes_json": {
            "event_summary": {
                "n_bins": int(n_bins),
                "persistence_bonus": round(persistence_bonus, 4),
                "dominant_label_ratio": round(dom_ratio, 4),
                "consistency_bonus": round(consistency_bonus, 4),
            },
            "data_reliability": {"reasons": data_reasons},
            "detection_confidence": {"reasons": detection_reasons},
            "diagnosis_confidence": {"reasons": diagnosis_reasons},
        },
    }
