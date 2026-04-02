from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.services.fdd.reliability import (
    compute_data_reliability,
    compute_detection_confidence,
    compute_diagnosis_confidence,
)
from core.services.fdd.runtime_detection import pick_diag_row_for_ts
from core.services.fdd.runtime_types import MismatchDashboardParams

def _pad_or_trim(seq: List[Any], n: int, fill: Any) -> List[Any]:
    seq = list(seq or [])
    if len(seq) < n:
        seq.extend([fill] * (n - len(seq)))
    elif len(seq) > n:
        seq = seq[:n]
    return seq

def build_runtime_confidence(
    *,
    times_utc: List[datetime],
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]],
    selected_sources: List[str],
    agg: Dict[str, Any],
    model: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(times_utc)
    rca = pipeline["rca"]
    labels = pipeline["labels"]
    anomaly = pipeline["anomaly"]
    valid_period = pipeline["valid_period"]
    coarse_period = pipeline["coarse_period"]
    fine_period = pipeline["fine_period"]
    meteo_quality_ok = pipeline["meteo_quality_ok"]
    stable_sky = pipeline["stable_sky"]
    irradiance_tier = pipeline["irradiance_tier"]
    det_dbg = pipeline["det_dbg"]

    diag_state_labels = [str(v) for v in _pad_or_trim(list(rca.get("state_labels") or []), n, "unknown")]
    diag_domain_labels = [str(v) for v in _pad_or_trim(list(rca.get("domain_labels") or []), n, "unknown")]
    diag_diagnosis_labels = [str(v) for v in _pad_or_trim(list(rca.get("diagnosis_labels") or labels), n, "invalid")]
    diag_base_conf = _pad_or_trim(list(rca.get("diagnosis_confidence") or []), n, None)
    diag_direct_grid = [bool(v) for v in _pad_or_trim(list(rca.get("direct_grid_evidence") or []), n, False)]
    diag_zero_inj = [bool(v) for v in _pad_or_trim(list(rca.get("zero_injection_flag") or []), n, False)]
    diag_evidence_json = _pad_or_trim(list(rca.get("evidence_json") or []), n, {})

    data_reliability_score: List[Optional[float]] = [None] * n
    data_reliability_level: List[str] = [""] * n
    detection_confidence_score: List[Optional[float]] = [None] * n
    detection_confidence_level: List[str] = [""] * n
    diagnosis_confidence_score: List[Optional[float]] = [None] * n
    diagnosis_confidence_level: List[str] = [""] * n
    confidence_notes: List[Dict[str, Any]] = [{} for _ in range(n)]

    for i, ts_utc in enumerate(times_utc):
        row_ref = pick_diag_row_for_ts(ts_utc, per_ts, selected_sources) or {}
        row_runtime = dict(row_ref)
        row_runtime.setdefault("flag_inv_missing", agg["flag_inv_missing_all"][i])
        row_runtime.setdefault("flag_low_coverage", agg["flag_inv_missing_partial"][i])
        row_runtime.setdefault("flag_meteo_missing", agg["flag_meteo_missing"][i])
        row_runtime.setdefault("flag_meteo_low_confidence", agg["flag_meteo_low_confidence"][i])
        row_runtime.setdefault("flag_meteo_interpolated", agg["flag_meteo_interpolated"][i])
        row_runtime.setdefault("flag_meteo_outlier", agg["flag_meteo_outlier"][i])
        row_runtime.setdefault("flag_meteo_artifact", agg["flag_meteo_artifact"][i])
        row_runtime.setdefault("inv_coverage", agg["inv_cov"][i])
        row_runtime.setdefault("meteo_qc_score", agg["meteo_qc_score"][i])
        row_runtime.setdefault("gti", agg["gti"][i])
        row_runtime.setdefault("ghi", agg["ghi"][i])

        ewma_i = None
        cusum_i = None
        ewma_seq = det_dbg.get("ewma_z") if isinstance(det_dbg, dict) else None
        if isinstance(ewma_seq, list) and i < len(ewma_seq):
            ewma_i = ewma_seq[i]
        cusum_seq = det_dbg.get("cusum") if isinstance(det_dbg, dict) else None
        if isinstance(cusum_seq, list) and i < len(cusum_seq):
            cusum_i = cusum_seq[i]

        diag_label = str(diag_diagnosis_labels[i] or labels[i] or "invalid")
        anomaly_final = bool(anomaly[i]) or bool(diag_direct_grid[i]) or (diag_label not in {"normal", "ok", "invalid", "low_irradiance"})

        data_rel = compute_data_reliability(
            row=row_runtime,
            pac_real_w=agg["p_ac_w"][i],
            pac_model_w=model["pac_model_w"][i],
            mismatch_rel=model["mismatch_rel"][i],
        )
        det_rel = compute_detection_confidence(
            data_reliability_score=data_rel["score"],
            valid_period=bool(valid_period[i]),
            coarse_period=bool(coarse_period[i]),
            fine_period=bool(fine_period[i]),
            meteo_quality_ok=bool(meteo_quality_ok[i]),
            stable_sky=bool(stable_sky[i]),
            anomaly_flag=bool(anomaly_final),
            mismatch_rel=model["mismatch_rel"][i],
            ewma_z=ewma_i,
            cusum_score=cusum_i,
        )
        diag_rel = compute_diagnosis_confidence(
            diagnosis_label=diag_label,
            base_diagnosis_confidence=(diag_base_conf[i] if i < len(diag_base_conf) else None),
            data_reliability_score=data_rel["score"],
            detection_confidence_score=det_rel["score"],
            fine_diag_allowed=bool(fine_period[i]),
            meteo_quality_ok=bool(meteo_quality_ok[i]),
            direct_grid_evidence=bool(diag_direct_grid[i]),
            zero_injection_flag=bool(diag_zero_inj[i]),
            irradiance_tier=str(irradiance_tier[i] or "N"),
        )

        data_reliability_score[i] = data_rel["score"]
        data_reliability_level[i] = str(data_rel["level"] or "")
        detection_confidence_score[i] = det_rel["score"]
        detection_confidence_level[i] = str(det_rel["level"] or "")
        diagnosis_confidence_score[i] = diag_rel["score"]
        diagnosis_confidence_level[i] = str(diag_rel["level"] or "")
        confidence_notes[i] = {
            "data_reliability": data_rel,
            "detection_confidence": det_rel,
            "diagnosis_confidence": diag_rel,
            "diagnostic_context": {
                "state_label": diag_state_labels[i],
                "domain_label": diag_domain_labels[i],
                "diagnosis_label": diag_diagnosis_labels[i],
                "direct_grid_evidence": bool(diag_direct_grid[i]),
                "zero_injection_flag": bool(diag_zero_inj[i]),
                "irradiance_tier": str(irradiance_tier[i] or "N"),
                "evidence_json": diag_evidence_json[i] if i < len(diag_evidence_json) else {},
            },
        }

    return {
        "diag_state_labels": diag_state_labels,
        "diag_domain_labels": diag_domain_labels,
        "diag_diagnosis_labels": diag_diagnosis_labels,
        "diag_direct_grid": diag_direct_grid,
        "diag_zero_inj": diag_zero_inj,
        "diag_evidence_json": diag_evidence_json,
        "data_reliability_score": data_reliability_score,
        "data_reliability_level": data_reliability_level,
        "detection_confidence_score": detection_confidence_score,
        "detection_confidence_level": detection_confidence_level,
        "diagnosis_confidence_score": diagnosis_confidence_score,
        "diagnosis_confidence_level": diagnosis_confidence_level,
        "confidence_notes": confidence_notes,
    }

def compute_plot_mismatch(params: MismatchDashboardParams, agg: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    np = model["np"]
    mismatch_rel_raw = model["mismatch_rel"][:]
    mismatch_rel_plot: List[Optional[float]] = []
    for i in range(len(model["mismatch_rel"])):
        mm = model["mismatch_rel"][i]
        gp = model["g_poa_used"][i]
        pm = model["pac_model_w"][i]
        pr = agg["p_ac_w"][i]
        ok_plot = (
            (mm is not None)
            and (gp is not None) and (float(gp) >= float(params.gpoa_plot_min))
            and (pm is not None) and (abs(float(pm)) >= float(params.pmodel_plot_min))
            and (pr is not None)
            and (not bool(agg["flag_meteo_missing"][i]))
            and (not bool(agg["flag_inv_missing_all"][i]))
        )
        if not ok_plot:
            mismatch_rel_plot.append(None)
            continue
        v = float(mm)
        if np.isfinite(v):
            v = max(-float(params.mismatch_clip_abs), min(float(params.mismatch_clip_abs), v))
            mismatch_rel_plot.append(v)
        else:
            mismatch_rel_plot.append(None)
    return {"mismatch_rel_raw": mismatch_rel_raw, "mismatch_rel_plot": mismatch_rel_plot}

