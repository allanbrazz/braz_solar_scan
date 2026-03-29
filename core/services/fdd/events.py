from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from django.db import transaction
from django.db.models import QuerySet

from core.models import FaultEvent, PlantDiagnostic15m
from core.services.fdd.reliability import aggregate_event_confidence


COARSE_EVENT_BY_DIAGNOSIS = {
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
}


@dataclass(slots=True)
class EventBuildParams:
    gap_bins: int = 1
    min_event_bins: int = 2
    detector_version: str = "hybrid_rules_v1"
    source_oper: str = ""
    source_meteo: str = ""
    replace_existing: bool = True


def _bucket_minutes(ts0: datetime, ts1: datetime) -> int:
    return max(1, int(round((ts1 - ts0).total_seconds() / 60.0)))


def _group_contiguous(rows: list[PlantDiagnostic15m], gap_bins: int = 1) -> list[list[PlantDiagnostic15m]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r.ts_utc)
    if len(rows) == 1:
        return [rows]

    dt_min = _bucket_minutes(rows[0].ts_utc, rows[1].ts_utc)
    max_gap = timedelta(minutes=dt_min * (gap_bins + 1))

    groups: list[list[PlantDiagnostic15m]] = []
    cur: list[PlantDiagnostic15m] = [rows[0]]
    for row in rows[1:]:
        if (row.ts_utc - cur[-1].ts_utc) <= max_gap:
            cur.append(row)
        else:
            groups.append(cur)
            cur = [row]
    groups.append(cur)
    return groups


def _safe_mean(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def _safe_max_abs(xs: Iterable[float | None]) -> float | None:
    vals = [abs(float(x)) for x in xs if x is not None]
    return max(vals) if vals else None


def _energy_loss_wh(rows: list[PlantDiagnostic15m], dt_hours: float = 0.25) -> float | None:
    acc = 0.0
    seen = False
    for r in rows:
        if r.pac_model_w is None or r.pac_real_w is None:
            continue
        acc += max(float(r.pac_model_w) - float(r.pac_real_w), 0.0) * dt_hours
        seen = True
    return acc if seen else None


def _dominant(counter: Counter[str], default: str) -> str:
    if not counter:
        return default
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _event_label_prelim(rows: list[PlantDiagnostic15m]) -> tuple[str, str, str]:
    diag_counter: Counter[str] = Counter()
    dom_counter: Counter[str] = Counter()
    state_counter: Counter[str] = Counter()

    for r in rows:
        diag = str((getattr(r, "diagnosis_label", None) or getattr(r, "rca_label", None) or "").strip())
        dom = str((getattr(r, "domain_label", None) or "unknown").strip())
        st = str((getattr(r, "state_label", None) or "unknown").strip())
        if diag and diag not in {"ok", "invalid", "low_irradiance"}:
            diag_counter[diag] += 1
        if dom:
            dom_counter[dom] += 1
        if st:
            state_counter[st] += 1

    diag_dominant = _dominant(diag_counter, "unknown")
    coarse = COARSE_EVENT_BY_DIAGNOSIS.get(diag_dominant, diag_dominant)
    return coarse, _dominant(dom_counter, "unknown"), _dominant(state_counter, "unknown")


def build_fault_events_for_range(
    *,
    plant_id: int,
    ts_start_utc: datetime,
    ts_end_utc: datetime,
    params: Optional[EventBuildParams] = None,
) -> dict:
    p = params or EventBuildParams()

    qs: QuerySet[PlantDiagnostic15m] = PlantDiagnostic15m.objects.filter(
        plant_id=plant_id,
        ts_utc__gte=ts_start_utc,
        ts_utc__lt=ts_end_utc,
        anomaly_flag=True,
        detector_version=p.detector_version,
    ).order_by("ts_utc")

    if p.source_oper:
        qs = qs.filter(source_oper=p.source_oper)
    if p.source_meteo:
        qs = qs.filter(source_meteo=p.source_meteo)

    rows = list(qs)
    groups = [g for g in _group_contiguous(rows, gap_bins=p.gap_bins) if len(g) >= p.min_event_bins]

    if p.replace_existing:
        qdel = FaultEvent.objects.filter(
            plant_id=plant_id,
            detector_version=p.detector_version,
            ts_start_utc__lt=ts_end_utc,
            ts_end_utc__gte=ts_start_utc,
        )
        if p.source_oper:
            qdel = qdel.filter(source_oper=p.source_oper)
        if p.source_meteo:
            qdel = qdel.filter(source_meteo=p.source_meteo)
        qdel.delete()

    created = 0
    updated = 0

    with transaction.atomic():
        for g in groups:
            event_label_prelim, domain_prelim, state_prelim = _event_label_prelim(g)
            diagnosis_counter = Counter(str((getattr(r, "diagnosis_label", None) or getattr(r, "rca_label", None) or "unknown")) for r in g)
            event_conf = aggregate_event_confidence(
                data_scores=(getattr(r, "data_reliability_score", None) for r in g),
                detection_scores=(getattr(r, "detection_confidence_score", None) for r in g),
                diagnosis_scores=(getattr(r, "diagnosis_confidence_score", getattr(r, "diagnosis_confidence", None)) for r in g),
                diagnosis_labels=(getattr(r, "diagnosis_label", None) for r in g),
                per_bin_notes=(getattr(r, "confidence_notes_json", None) for r in g),
                n_bins=len(g),
            )
            defaults = {
                "source_oper": p.source_oper,
                "source_meteo": p.source_meteo,
                "status": FaultEvent.STATUS_OPEN,
                "detector_score_max": _safe_max_abs(r.detector_score for r in g),
                "detector_score_mean": _safe_mean(r.detector_score for r in g),
                "severity_score": _safe_max_abs(r.mismatch_rel for r in g),
                "energy_loss_wh": _energy_loss_wh(g),
                "event_label_prelim": event_label_prelim,
                "known_vs_unknown": "pending",
                "final_label": "",
                "confidence": event_conf.get("diagnosis_confidence_score"),
                "data_reliability_score": event_conf.get("data_reliability_score"),
                "data_reliability_level": event_conf.get("data_reliability_level", ""),
                "detection_confidence_score": event_conf.get("detection_confidence_score"),
                "detection_confidence_level": event_conf.get("detection_confidence_level", ""),
                "diagnosis_confidence_score": event_conf.get("diagnosis_confidence_score"),
                "diagnosis_confidence_level": event_conf.get("diagnosis_confidence_level", ""),
                "confidence_notes_json": event_conf.get("confidence_notes_json"),
                "novelty_score": None,
                "meta": {
                    "n_bins": len(g),
                    "rca_labels": [r.rca_label for r in g],
                    "diagnosis_labels": [getattr(r, "diagnosis_label", None) for r in g],
                    "dominant_diagnosis": _dominant(diagnosis_counter, "unknown"),
                    "diagnosis_counts": dict(diagnosis_counter),
                    "domain_prelim": domain_prelim,
                    "state_prelim": state_prelim,
                    "ts_bins_utc": [r.ts_utc.isoformat() for r in g],
                    "source_oper": p.source_oper,
                    "source_meteo": p.source_meteo,
                    "detector_version": p.detector_version,
                    "irradiance_tier_counts": dict(Counter(str(getattr(r, "irradiance_tier", "N")) for r in g)),
                    "confidence_summary": event_conf,
                },
            }
            obj, was_created = FaultEvent.objects.update_or_create(
                plant_id=plant_id,
                source_oper=p.source_oper,
                ts_start_utc=g[0].ts_utc,
                ts_end_utc=g[-1].ts_utc,
                detector_version=p.detector_version,
                defaults=defaults,
            )
            created += int(was_created)
            updated += int(not was_created)

    return {
        "ok": True,
        "plant_id": plant_id,
        "detector_version": p.detector_version,
        "source_oper": p.source_oper,
        "source_meteo": p.source_meteo,
        "events": len(groups),
        "created": created,
        "updated": updated,
    }
