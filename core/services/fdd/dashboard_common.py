from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from django.db import transaction
from django.db.models import Count
from django.utils import timezone

from core.models import PVPlant, PVPlantMergedRecord15m, PlantDiagnostic15m


MISMATCH_VERSION_SUMMARY = {
    "detector_version": "mismatch_runtime_v1",
    "event_classifier_version": None,
    "trained_model_version": None,
    "detector_note": "Detector runtime desta tela: modelo físico de potência + limiares/heurísticas de mismatch configurados na UI.",
    "event_classifier_note": "Não aplicável no dashboard Mismatch.",
    "trained_model_note": "Não aplicável no dashboard Mismatch.",
}


class DashboardServiceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)


def is_mppt_source(src: str) -> bool:
    return "|MPPT" in str(src or "").upper()


def is_agg_source(src: str) -> bool:
    s = str(src or "").strip()
    if not s:
        return False
    u = s.upper()
    if "|" not in u:
        return True
    return u.endswith("|AGG")


def source_root(src: str) -> str:
    s = str(src or "").strip()
    if not s:
        return ""
    return s.split("|", 1)[0].strip()


def canonical_source_oper(selected_sources: List[str]) -> str:
    for s in selected_sources:
        if is_agg_source(s):
            return s
    if selected_sources:
        return source_root(selected_sources[0])
    return ""


def runtime_severity(
    *,
    state_label: str,
    diagnosis_label: str,
    direct_grid_evidence: bool,
    anomaly_flag: bool,
) -> str:
    state = str(state_label or "").strip().lower()
    diag = str(diagnosis_label or "").strip().lower()

    if diag in {"ok", "normal"} or state == "injecting_normal":
        return "ok"

    if diag in {
        "grid_overvoltage_trip",
        "grid_undervoltage_trip",
        "grid_overfrequency_trip",
        "grid_underfrequency_trip",
        "inverter_off_under_sun",
        "unknown_shutdown_with_sun",
    }:
        return "crit"

    if diag in {
        "grid_overvoltage_derating",
        "grid_undervoltage_derating",
        "partial_generation_loss_probable",
        "persistent_underperformance",
        "dc_side_partial_loss_probable",
        "dc_side_voltage_anomaly_probable",
        "curtailment_clipping",
    } or state == "injecting_degraded":
        return "warn"

    if direct_grid_evidence and state == "sun_available_not_injecting":
        return "crit"

    if diag == "invalid" or state in {"telemetry_invalid", "unknown", "low_irradiance"}:
        return "none"

    if anomaly_flag:
        return "warn"

    return "none"


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except Exception:
        return None


def sum_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    ok = False
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        ok = True
    return acc if ok else None


def mean_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    n = 0
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        n += 1
    return (acc / n) if n else None


def pick_best_sources(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Tuple[Optional[str], Optional[str]]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper", "source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    if not row:
        return None, None
    return row.get("source_oper"), row.get("source_meteo")


def upsert_diag15m(
    *,
    plant: PVPlant,
    source_oper: str,
    source_meteo: str,
    detector_version: str,
    times_utc: List[datetime],
    rca_codes: List[int],
    rca_labels: List[str],
    valid: List[bool],
    anomaly_flags: List[bool],
    detector_scores: List[Optional[float]],
    ewma_z: List[Optional[float]],
    cusum_scores: List[Optional[float]],
    stable_sky: List[bool],
    g_poa: List[Optional[float]],
    tcell_c: List[Optional[float]],
    pac_real_w: List[Optional[float]],
    pac_model_w: List[Optional[float]],
    mismatch_rel: List[Optional[float]],
    irradiance_tier: List[str],
    fine_diag_allowed: List[bool],
    meteo_quality_ok: List[bool],
    direct_grid_evidence: List[bool],
    zero_injection_flag: List[bool],
    state_labels: List[str],
    domain_labels: List[str],
    diagnosis_labels: List[str],
    diagnosis_confidence_score: List[Optional[float]],
    data_reliability_score: List[Optional[float]],
    data_reliability_level: List[str],
    detection_confidence_score: List[Optional[float]],
    detection_confidence_level: List[str],
    diagnosis_confidence_level: List[str],
    v_ac_v: List[Optional[float]],
    i_ac_a: List[Optional[float]],
    freq_hz: List[Optional[float]],
    alarm_code_oper: List[Optional[int]],
    alarm_sev_oper: List[Optional[int]],
    evidence_json: List[dict],
    confidence_notes_json: List[dict],
) -> Dict[str, Any]:
    n = len(times_utc)
    seqs = [
        rca_codes, rca_labels, valid, anomaly_flags, detector_scores, ewma_z,
        cusum_scores, stable_sky, g_poa, tcell_c, pac_real_w, pac_model_w,
        mismatch_rel, irradiance_tier, fine_diag_allowed, meteo_quality_ok,
        direct_grid_evidence, zero_injection_flag, state_labels, domain_labels,
        diagnosis_labels, diagnosis_confidence_score, data_reliability_score,
        data_reliability_level, detection_confidence_score, detection_confidence_level,
        diagnosis_confidence_level, v_ac_v, i_ac_a, freq_hz, alarm_code_oper,
        alarm_sev_oper, evidence_json, confidence_notes_json,
    ]
    if not all(len(seq) == n for seq in seqs):
        raise ValueError("upsert_diag15m: sequências com tamanhos inconsistentes")

    if not times_utc:
        return {
            "deleted": 0,
            "created": 0,
            "source_oper": source_oper,
            "source_meteo": source_meteo,
            "detector_version": detector_version,
        }

    root = source_root(source_oper)
    delete_qs = PlantDiagnostic15m.objects.filter(
        plant=plant,
        ts_utc__in=times_utc,
        source_meteo=source_meteo,
        detector_version=detector_version,
    )
    if root:
        delete_qs = delete_qs.filter(source_oper__startswith=root)
    else:
        delete_qs = delete_qs.filter(source_oper=source_oper)

    now = timezone.now()
    objs: List[PlantDiagnostic15m] = []
    for i, ts in enumerate(times_utc):
        objs.append(
            PlantDiagnostic15m(
                plant=plant,
                ts_utc=ts,
                source_oper=source_oper,
                source_meteo=source_meteo,
                detector_version=detector_version,
                rca_code=int(rca_codes[i]),
                rca_label=str(diagnosis_labels[i] or rca_labels[i] or "invalid"),
                valid=bool(valid[i]),
                anomaly_flag=bool(anomaly_flags[i]),
                detector_score=detector_scores[i],
                ewma_z=ewma_z[i],
                cusum_score=cusum_scores[i],
                stable_sky=bool(stable_sky[i]),
                g_poa_wm2=g_poa[i],
                tcell_c=tcell_c[i],
                pac_real_w=pac_real_w[i],
                pac_model_w=pac_model_w[i],
                mismatch_rel=mismatch_rel[i],
                irradiance_tier=str(irradiance_tier[i] or ""),
                fine_diag_allowed=bool(fine_diag_allowed[i]),
                meteo_quality_ok=bool(meteo_quality_ok[i]),
                direct_grid_evidence=bool(direct_grid_evidence[i]),
                zero_injection_flag=bool(zero_injection_flag[i]),
                state_label=str(state_labels[i] or ""),
                domain_label=str(domain_labels[i] or ""),
                diagnosis_label=str(diagnosis_labels[i] or ""),
                diagnosis_confidence_score=diagnosis_confidence_score[i],
                data_reliability_score=data_reliability_score[i],
                data_reliability_level=str(data_reliability_level[i] or ""),
                detection_confidence_score=detection_confidence_score[i],
                detection_confidence_level=str(detection_confidence_level[i] or ""),
                diagnosis_confidence_level=str(diagnosis_confidence_level[i] or ""),
                v_ac_v=v_ac_v[i],
                i_ac_a=i_ac_a[i],
                freq_hz=freq_hz[i],
                alarm_code_oper=alarm_code_oper[i],
                alarm_sev_oper=alarm_sev_oper[i],
                evidence_json=evidence_json[i],
                confidence_notes_json=confidence_notes_json[i],
                updated_at=now,
            )
        )

    with transaction.atomic():
        deleted, _ = delete_qs.delete()
        PlantDiagnostic15m.objects.bulk_create(objs, batch_size=1000)

    return {
        "deleted": int(deleted),
        "created": len(objs),
        "source_oper": source_oper,
        "source_meteo": source_meteo,
        "detector_version": detector_version,
    }
