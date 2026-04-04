from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from math import ceil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from django.db.models import Q

from core.models import FaultEvent, GroundTruthEvent, PlantDiagnostic15m


OK_DIAG_LABELS = {"", "ok", "normal", "invalid", "low_irradiance"}
FAULT_GROUP_BY_LABEL = {
    "grid_overvoltage_trip": "grid_fault",
    "grid_overvoltage_derating": "grid_fault",
    "grid_undervoltage_trip": "grid_fault",
    "grid_undervoltage_derating": "grid_fault",
    "grid_overfrequency_trip": "grid_fault",
    "grid_underfrequency_trip": "grid_fault",
    "inverter_off_under_sun": "inverter_shutdown",
    "unknown_shutdown_with_sun": "inverter_shutdown",
    "dc_side_partial_loss_probable": "dc_side_partial_loss",
    "dc_side_voltage_anomaly_probable": "dc_side_partial_loss",
    "partial_generation_loss_probable": "dc_side_partial_loss",
    "persistent_underperformance": "persistent_underperformance",
    "curtailment_clipping": "curtailment_clipping",
    "normal_window": "normal",
    "normal": "normal",
}


def normalize_label(label: Any, default: str = "unknown") -> str:
    value = str(label or "").strip()
    return value or default


def infer_truth_group(label: Any, truth_state: str = GroundTruthEvent.STATE_CONFIRMED) -> str:
    if truth_state == GroundTruthEvent.STATE_NORMAL:
        return "normal"
    lab = normalize_label(label)
    return FAULT_GROUP_BY_LABEL.get(lab, lab if lab != "unknown" else "unknown")


def _safe_div(a: float, b: float) -> Optional[float]:
    try:
        if float(b) == 0.0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _overlap_seconds(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0.0, (end - start).total_seconds())


def _iou_time(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    ov = _overlap_seconds(a0, a1, b0, b1)
    if ov <= 0.0:
        return 0.0
    union = max((a1 - a0).total_seconds(), 0.0) + max((b1 - b0).total_seconds(), 0.0) - ov
    if union <= 0.0:
        return 0.0
    return float(ov / union)


def _event_display_label(ev: FaultEvent) -> str:
    return normalize_label(getattr(ev, "final_label", None) or getattr(ev, "event_label_prelim", None))


def _event_review_state(ev: FaultEvent) -> str:
    review = (((getattr(ev, "meta", None) or {}).get("review") or {}).get("review_state") or "").strip().lower()
    if review:
        return review
    status = str(getattr(ev, "status", "") or "").strip().lower()
    if status == FaultEvent.STATUS_DISMISSED:
        return "dismissed"
    if status == FaultEvent.STATUS_REVIEWED:
        return "confirmed"
    return status or "pending"


def _bucket_gpoa(value: Any) -> str:
    try:
        g = float(value)
    except Exception:
        return "unknown"
    if g < 180.0:
        return "<180"
    if g < 320.0:
        return "180-320"
    if g < 500.0:
        return "320-500"
    if g < 700.0:
        return "500-700"
    return ">=700"


def _metrics_dict(tp: int, fp: int, fn: int, tn: Optional[int] = None) -> Dict[str, Any]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    out = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if tn is not None:
        out["tn"] = int(tn)
    return out


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def query_predicted_events(*, plant_id: int, ts_start_utc: datetime, ts_end_utc: datetime, detector_version: str = "", source_oper: str = "", source_meteo: str = "") -> List[FaultEvent]:
    qs = FaultEvent.objects.filter(plant_id=plant_id, ts_start_utc__lt=ts_end_utc, ts_end_utc__gte=ts_start_utc).order_by("ts_start_utc")
    if detector_version:
        qs = qs.filter(detector_version=detector_version)
    if source_oper:
        qs = qs.filter(source_oper=source_oper)
    if source_meteo:
        qs = qs.filter(source_meteo=source_meteo)
    return list(qs)


def query_truth_events(*, plant_id: int, ts_start_utc: datetime, ts_end_utc: datetime, source_oper: str = "", source_meteo: str = "") -> List[GroundTruthEvent]:
    qs = GroundTruthEvent.objects.filter(plant_id=plant_id, ts_start_utc__lt=ts_end_utc, ts_end_utc__gte=ts_start_utc).order_by("ts_start_utc")
    if source_oper:
        qs = qs.filter(Q(source_oper="") | Q(source_oper=source_oper))
    if source_meteo:
        qs = qs.filter(Q(source_meteo="") | Q(source_meteo=source_meteo))
    return list(qs)


def _match_events(pred_events: Sequence[FaultEvent], truth_events: Sequence[GroundTruthEvent]) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int]]:
    candidates: List[Tuple[int, float, float, int, int]] = []
    for pi, pred in enumerate(pred_events):
        linked_truth_id = getattr(pred, "id", None)
        for ti, truth in enumerate(truth_events):
            if truth.truth_state != GroundTruthEvent.STATE_CONFIRMED:
                continue
            linked = bool(truth.linked_fault_event_id and truth.linked_fault_event_id == pred.id)
            overlap = _overlap_seconds(pred.ts_start_utc, pred.ts_end_utc, truth.ts_start_utc, truth.ts_end_utc)
            if overlap <= 0.0 and not linked:
                continue
            iou = _iou_time(pred.ts_start_utc, pred.ts_end_utc, truth.ts_start_utc, truth.ts_end_utc)
            candidates.append((1 if linked else 0, iou, overlap, pi, ti))
    candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))

    pred_used: Dict[int, int] = {}
    truth_used: Dict[int, int] = {}
    matches: List[Dict[str, Any]] = []
    for linked, iou, overlap, pi, ti in candidates:
        if pi in pred_used or ti in truth_used:
            continue
        pred = pred_events[pi]
        truth = truth_events[ti]
        delay_min = max(0.0, (pred.ts_start_utc - truth.ts_start_utc).total_seconds() / 60.0)
        match_type = "TP" if normalize_label(truth.truth_label) == _event_display_label(pred) else "PARTIAL"
        matches.append({
            "pred_event_id": pred.id,
            "truth_event_id": truth.id,
            "match_type": match_type,
            "iou_time": iou,
            "overlap_minutes": overlap / 60.0,
            "detection_delay_min": delay_min,
            "pred_label": _event_display_label(pred),
            "truth_label": normalize_label(truth.truth_label),
        })
        pred_used[pi] = ti
        truth_used[ti] = pi
    return matches, pred_used, truth_used


def derive_truth_for_timestamp(ts_utc: datetime, truth_events: Sequence[GroundTruthEvent]) -> Dict[str, Any]:
    active = [ev for ev in truth_events if ev.ts_start_utc <= ts_utc <= ev.ts_end_utc]
    if not active:
        return {"bin_truth_state": "unlabeled", "bin_truth_label": None, "truth_event_id": None}
    if any(ev.truth_state in {GroundTruthEvent.STATE_UNCERTAIN, GroundTruthEvent.STATE_DISMISSED} for ev in active):
        return {"bin_truth_state": "ignore", "bin_truth_label": None, "truth_event_id": None}
    fault_labels = {normalize_label(ev.truth_label) for ev in active if ev.truth_state == GroundTruthEvent.STATE_CONFIRMED}
    normal_events = [ev for ev in active if ev.truth_state == GroundTruthEvent.STATE_NORMAL]
    if fault_labels and normal_events:
        return {"bin_truth_state": "ignore", "bin_truth_label": None, "truth_event_id": None}
    if len(fault_labels) > 1:
        return {"bin_truth_state": "ignore", "bin_truth_label": None, "truth_event_id": None}
    if fault_labels:
        ev = next(ev for ev in active if ev.truth_state == GroundTruthEvent.STATE_CONFIRMED)
        label = next(iter(fault_labels))
        return {"bin_truth_state": "fault", "bin_truth_label": label, "truth_event_id": ev.id}
    if normal_events:
        ev = normal_events[0]
        return {"bin_truth_state": "normal", "bin_truth_label": "normal", "truth_event_id": ev.id}
    return {"bin_truth_state": "ignore", "bin_truth_label": None, "truth_event_id": None}


def build_dashboard_validation_context(
    *,
    plant_id: int,
    tz,
    times_utc: Sequence[datetime],
    pred_anomaly_flags: Sequence[bool],
    pred_labels: Sequence[Any],
    g_poa: Sequence[Any],
    meteo_quality_ok: Sequence[Any],
    detector_version: str = "",
    source_oper: str = "",
    source_meteo: str = "",
) -> Dict[str, Any]:
    if not times_utc:
        return {"overlay_by_tkey": {}, "events": [], "summary": {"coverage": {}, "event_metrics": {}, "sample_metrics": {}, "confusion_matrix": {}}}

    ts_start = min(times_utc)
    ts_end = max(times_utc) + timedelta(minutes=15)
    pred_events = query_predicted_events(plant_id=plant_id, ts_start_utc=ts_start, ts_end_utc=ts_end, detector_version=detector_version, source_oper=source_oper, source_meteo=source_meteo)
    truth_events = query_truth_events(plant_id=plant_id, ts_start_utc=ts_start, ts_end_utc=ts_end, source_oper=source_oper, source_meteo=source_meteo)
    return compute_validation_report(
        times_utc=list(times_utc),
        pred_anomaly_flags=list(pred_anomaly_flags),
        pred_labels=list(pred_labels),
        g_poa=list(g_poa),
        meteo_quality_ok=list(meteo_quality_ok),
        pred_events=pred_events,
        truth_events=truth_events,
        tz=tz,
    )


def compute_validation_report(
    *,
    times_utc: Sequence[datetime],
    pred_anomaly_flags: Sequence[bool],
    pred_labels: Sequence[Any],
    g_poa: Sequence[Any],
    meteo_quality_ok: Sequence[Any],
    pred_events: Sequence[FaultEvent],
    truth_events: Sequence[GroundTruthEvent],
    tz,
) -> Dict[str, Any]:
    overlay_by_tkey: Dict[str, Any] = {}
    event_matches, pred_used, truth_used = _match_events(pred_events, truth_events)

    pred_event_rows: List[Dict[str, Any]] = []
    for idx, ev in enumerate(pred_events):
        matched_truth = truth_events[pred_used[idx]] if idx in pred_used else None
        pred_event_rows.append({
            "id": ev.id,
            "ts_start_utc": ev.ts_start_utc.isoformat(),
            "ts_end_utc": ev.ts_end_utc.isoformat(),
            "status": ev.status,
            "review_state": _event_review_state(ev),
            "event_label_prelim": normalize_label(ev.event_label_prelim),
            "final_label": normalize_label(ev.final_label, default=""),
            "pred_label": _event_display_label(ev),
            "known_vs_unknown": normalize_label(ev.known_vs_unknown, default="pending"),
            "matched_truth_event_id": getattr(matched_truth, "id", None),
            "matched_truth_label": getattr(matched_truth, "truth_label", None),
            "match_type": "TP" if idx in pred_used else "FP",
        })

    # Event metrics
    tp_events = len(pred_used)
    fp_events = len(pred_events) - tp_events
    positive_truth_events = [ev for ev in truth_events if ev.truth_state == GroundTruthEvent.STATE_CONFIRMED]
    fn_events = len(positive_truth_events) - len(truth_used)
    event_metrics = _metrics_dict(tp_events, fp_events, fn_events)
    event_confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in event_matches:
        event_confusion[str(row.get("truth_label") or "unknown")][str(row.get("pred_label") or "unknown")] += 1

    delay_values = [row.get("detection_delay_min") for row in event_matches if row.get("detection_delay_min") is not None]
    event_metrics.update({
        "mean_detection_delay_min": _mean(delay_values),
        "median_detection_delay_min": _quantile(delay_values, 0.5),
        "p90_detection_delay_min": _quantile(delay_values, 0.9),
    })

    # Bin/sample metrics + overlays
    tp = fp = fn = tn = 0
    by_gpoa: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    by_meteo: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    labeled_days = set()
    for i, ts_utc in enumerate(times_utc):
        truth_info = derive_truth_for_timestamp(ts_utc, truth_events)
        pred_flag = bool(pred_anomaly_flags[i]) if i < len(pred_anomaly_flags) else False
        pred_label = normalize_label(pred_labels[i] if i < len(pred_labels) else None, default="normal")
        g_bucket = _bucket_gpoa(g_poa[i] if i < len(g_poa) else None)
        mq = bool(meteo_quality_ok[i]) if i < len(meteo_quality_ok) else False
        m_bucket = "meteo_ok" if mq else "meteo_low_quality"
        tloc = ts_utc.astimezone(tz)
        tkey = tloc.strftime("%Y-%m-%dT%H:%M")

        pred_event = next((ev for ev in pred_events if ev.ts_start_utc <= ts_utc <= ev.ts_end_utc), None)
        overlay_by_tkey[tkey] = {
            "pred_event_id": getattr(pred_event, "id", None),
            "pred_event_status": getattr(pred_event, "status", None),
            "pred_event_label_prelim": normalize_label(getattr(pred_event, "event_label_prelim", None), default="") if pred_event else "",
            "pred_event_final_label": normalize_label(getattr(pred_event, "final_label", None), default="") if pred_event else "",
            "pred_event_review_state": _event_review_state(pred_event) if pred_event else "",
            "truth_event_id": truth_info.get("truth_event_id"),
            "truth_label": truth_info.get("bin_truth_label"),
            "bin_truth_state": truth_info.get("bin_truth_state"),
            "bin_truth_label": truth_info.get("bin_truth_label"),
            "pred_positive": pred_flag,
            "pred_label": pred_label,
        }

        truth_state = truth_info.get("bin_truth_state")
        if truth_state in {"ignore", "unlabeled"}:
            continue
        labeled_days.add(tloc.date().isoformat())
        if truth_state == "fault":
            if pred_flag:
                tp += 1
                by_gpoa[g_bucket]["tp"] += 1
                by_meteo[m_bucket]["tp"] += 1
                confusion[str(truth_info.get("bin_truth_label") or "unknown")][pred_label] += 1
            else:
                fn += 1
                by_gpoa[g_bucket]["fn"] += 1
                by_meteo[m_bucket]["fn"] += 1
                confusion[str(truth_info.get("bin_truth_label") or "unknown")]["normal"] += 1
        elif truth_state == "normal":
            if pred_flag:
                fp += 1
                by_gpoa[g_bucket]["fp"] += 1
                by_meteo[m_bucket]["fp"] += 1
            else:
                tn += 1
                by_gpoa[g_bucket]["tn"] += 1
                by_meteo[m_bucket]["tn"] += 1

    sample_metrics = _metrics_dict(tp, fp, fn, tn=tn)
    monitored_days = max(1, len(labeled_days) or len({ts.astimezone(tz).date().isoformat() for ts in times_utc}))
    monitored_weeks = max(1, ceil(monitored_days / 7.0))
    sample_metrics.update({
        "false_alarm_bins_per_day": fp / monitored_days,
    })
    event_metrics.update({
        "false_alarm_events_per_day": fp_events / monitored_days,
        "false_alarm_events_per_week": fp_events / monitored_weeks,
    })

    by_gpoa_metrics = {k: _metrics_dict(v["tp"], v["fp"], v["fn"], tn=v["tn"]) for k, v in by_gpoa.items()}
    by_meteo_metrics = {k: _metrics_dict(v["tp"], v["fp"], v["fn"], tn=v["tn"]) for k, v in by_meteo.items()}
    confusion_matrix = {truth: dict(preds) for truth, preds in confusion.items()}

    coverage = {
        "predicted_events": len(pred_events),
        "truth_events_confirmed": len(positive_truth_events),
        "truth_windows_normal": len([ev for ev in truth_events if ev.truth_state == GroundTruthEvent.STATE_NORMAL]),
        "truth_events_uncertain": len([ev for ev in truth_events if ev.truth_state == GroundTruthEvent.STATE_UNCERTAIN]),
        "truth_events_dismissed": len([ev for ev in truth_events if ev.truth_state == GroundTruthEvent.STATE_DISMISSED]),
        "labeled_days": monitored_days,
        "labeled_bins": tp + fp + fn + tn,
    }

    return {
        "overlay_by_tkey": overlay_by_tkey,
        "events": pred_event_rows,
        "matches": event_matches,
        "summary": {
            "coverage": coverage,
            "event_metrics": event_metrics,
            "sample_metrics": sample_metrics,
            "by_irradiance_bucket": by_gpoa_metrics,
            "by_meteo_quality": by_meteo_metrics,
            "confusion_matrix": confusion_matrix,
            "event_confusion_matrix": {truth: dict(preds) for truth, preds in event_confusion.items()},
        },
    }


def compute_validation_report_from_db(*, plant_id: int, ts_start_utc: datetime, ts_end_utc: datetime, detector_version: str = "", source_oper: str = "", source_meteo: str = "") -> Dict[str, Any]:
    qs = PlantDiagnostic15m.objects.filter(plant_id=plant_id, ts_utc__gte=ts_start_utc, ts_utc__lt=ts_end_utc).order_by("ts_utc")
    if detector_version:
        qs = qs.filter(detector_version=detector_version)
    if source_oper:
        qs = qs.filter(source_oper=source_oper)
    if source_meteo:
        qs = qs.filter(source_meteo=source_meteo)
    rows = list(qs)
    pred_events = query_predicted_events(plant_id=plant_id, ts_start_utc=ts_start_utc, ts_end_utc=ts_end_utc, detector_version=detector_version, source_oper=source_oper, source_meteo=source_meteo)
    truth_events = query_truth_events(plant_id=plant_id, ts_start_utc=ts_start_utc, ts_end_utc=ts_end_utc, source_oper=source_oper, source_meteo=source_meteo)
    tz_name = "UTC"
    if rows:
        tz_name = getattr(rows[0].plant, "timezone", None) or "UTC"
    else:
        gt = (truth_events[0] if truth_events else None)
        if gt is not None:
            tz_name = getattr(gt.plant, "timezone", None) or "UTC"
    try:
        tz = ZoneInfo(str(tz_name or "UTC"))
    except Exception:
        tz = ZoneInfo("UTC")
    return compute_validation_report(
        times_utc=[r.ts_utc for r in rows],
        pred_anomaly_flags=[bool(r.anomaly_flag) for r in rows],
        pred_labels=[r.diagnosis_label for r in rows],
        g_poa=[r.g_poa for r in rows],
        meteo_quality_ok=[bool(r.meteo_quality_ok) for r in rows],
        pred_events=pred_events,
        truth_events=truth_events,
        tz=tz,
    )
