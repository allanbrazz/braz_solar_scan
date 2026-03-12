from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mode
from typing import Iterable, Optional

from django.db import transaction
from django.db.models import QuerySet

from core.models import FaultEvent, PlantDiagnostic15m


EVENT_LABEL_MAP = {
    "clipping_limit": "curtailment_clipping",
    "meteo_bias_underestimate": "meteo_bias",
    "meteo_bias_small": "meteo_bias",
    "low_current_string_offline": "localized_loss",
    "low_current_shading_soiling": "localized_loss",
    "low_voltage_bypass_short_mppt": "localized_loss",
    "low_voltage_anomaly": "localized_loss",
    "power_loss": "plant_wide_loss",
    "anomaly_unspecified": "plant_wide_loss",
}


@dataclass(slots=True)
class EventBuildParams:
    gap_bins: int = 1
    min_event_bins: int = 2
    detector_version: str = "residual_v1"
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


def _event_label_prelim(rows: list[PlantDiagnostic15m]) -> str:
    labels = [r.rca_label for r in rows if r.rca_label and r.rca_label not in {"ok", "invalid"}]
    if not labels:
        return "unknown"

    mapped = [EVENT_LABEL_MAP.get(lbl, lbl) for lbl in labels]
    try:
        return mode(mapped)
    except Exception:
        counts: dict[str, int] = {}
        for lbl in mapped:
            counts[lbl] = counts.get(lbl, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


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
        valid=True,
        anomaly_flag=True,
        detector_version=p.detector_version,
    ).order_by("ts_utc")

    rows = list(qs)
    groups = [g for g in _group_contiguous(rows, gap_bins=p.gap_bins) if len(g) >= p.min_event_bins]

    if p.replace_existing:
        FaultEvent.objects.filter(
            plant_id=plant_id,
            detector_version=p.detector_version,
            ts_start_utc__lt=ts_end_utc,
            ts_end_utc__gte=ts_start_utc,
        ).delete()

    created = 0
    updated = 0

    with transaction.atomic():
        for g in groups:
            defaults = {
                "source_oper": p.source_oper,
                "source_meteo": p.source_meteo,
                "status": FaultEvent.STATUS_OPEN,
                "detector_score_max": _safe_max_abs(r.detector_score for r in g),
                "detector_score_mean": _safe_mean(r.detector_score for r in g),
                "severity_score": _safe_max_abs(r.mismatch_rel for r in g),
                "energy_loss_wh": _energy_loss_wh(g),
                "event_label_prelim": _event_label_prelim(g),
                "known_vs_unknown": "pending",
                "final_label": "",
                "confidence": None,
                "novelty_score": None,
                "meta": {
                    "n_bins": len(g),
                    "rca_labels": [r.rca_label for r in g],
                    "ts_bins_utc": [r.ts_utc.isoformat() for r in g],
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