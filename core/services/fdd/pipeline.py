from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
from django.db import transaction
from django.db.models import Count

from core.models import PVPlant, PVPlantMergedRecord15m, PlantDiagnostic15m
from core.services.fdd.detection import DetectionParams, detect_anomalies
from core.services.fdd.events import EventBuildParams, build_fault_events_for_range
from core.services.fdd.rca import RCAParams, diagnose_rca_series
from core.services.fdd.reliability import (
    compute_data_reliability,
    compute_detection_confidence,
    compute_diagnosis_confidence,
)
from core.services.residuals.facade import compute_residual_series_from_observations


MERGED_FIELDS = (
    "ts_utc",
    "source_oper",
    "p_ac_w",
    "p_dc_w",
    "v_dc_v",
    "i_dc_a",
    "v_ac_v",
    "i_ac_a",
    "freq_hz",
    "gti",
    "ghi",
    "dni",
    "dhi",
    "temp_air",
    "meteo_qc_score",
    "flag_meteo_low_confidence",
    "flag_meteo_interpolated",
    "flag_meteo_outlier",
    "flag_meteo_artifact",
    "alarm_code",
    "alarm_sev",
    "inv_coverage",
    "flag_low_coverage",
    "flag_meteo_missing",
    "flag_inv_missing",
)


def _pick_best_source_meteo(plant_id: int, ts_start_utc: datetime, ts_end_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")



def _source_root(src: str) -> str:
    s = str(src or "").strip()
    return s.split("|", 1)[0] if "|" in s else s



def _is_agg_source(src: str) -> bool:
    s = str(src or "").strip().upper()
    if not s:
        return False
    if "|" not in s:
        return True
    return s.endswith("|AGG")



def _is_mppt_source(src: str) -> bool:
    return "|MPPT" in str(src or "").upper()



def _pick_best_agg_source_oper(plant_id: int, source_meteo: str, ts_start_utc: datetime, ts_end_utc: datetime) -> Optional[str]:
    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    for row in rows:
        src = str((row or {}).get("source_oper") or "")
        if _is_agg_source(src):
            return src
    return None



def _load_exact_rows(
    *,
    plant_id: int,
    source_oper: str,
    source_meteo: str,
    ts_start_utc: datetime,
    ts_end_utc: datetime,
) -> list[dict[str, Any]]:
    return list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_oper=source_oper,
            source_meteo=source_meteo,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .order_by("ts_utc")
        .values(*MERGED_FIELDS)
    )



def _first_finite(values: list[Any]) -> Optional[float]:
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            return f
    return None



def _bool_any(values: list[Any]) -> bool:
    return any(bool(v) for v in values)



def _bool_all(values: list[Any], *, default: bool = False) -> bool:
    vals = [bool(v) for v in values]
    return all(vals) if vals else default



def _synthesize_agg_rows_from_mppt(
    *,
    plant_id: int,
    source_meteo: str,
    ts_start_utc: datetime,
    ts_end_utc: datetime,
) -> tuple[str, list[dict[str, Any]]]:
    raw = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .order_by("ts_utc", "source_oper")
        .values(*MERGED_FIELDS)
    )
    mppt_rows = [r for r in raw if _is_mppt_source(str(r.get("source_oper") or ""))]
    if not mppt_rows:
        return "", []

    roots = sorted({_source_root(str(r.get("source_oper") or "")) for r in mppt_rows if str(r.get("source_oper") or "").strip()})
    agg_source = f"{roots[0]}|AGG" if len(roots) == 1 else "PLANT|AGG"

    by_ts: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in mppt_rows:
        tsu = row.get("ts_utc")
        if tsu is not None:
            by_ts[tsu].append(row)

    out: list[dict[str, Any]] = []
    for tsu in sorted(by_ts.keys()):
        grp = by_ts[tsu]
        pac = sum(float(r.get("p_ac_w") or 0.0) for r in grp if r.get("p_ac_w") is not None)
        pdc = sum(float(r.get("p_dc_w") or 0.0) for r in grp if r.get("p_dc_w") is not None)
        idc = sum(float(r.get("i_dc_a") or 0.0) for r in grp if r.get("i_dc_a") is not None)
        vdc_vals = [r.get("v_dc_v") for r in grp if r.get("v_dc_v") is not None]
        vdc = (sum(float(v) for v in vdc_vals) / len(vdc_vals)) if vdc_vals else None

        out.append(
            {
                "ts_utc": tsu,
                "source_oper": agg_source,
                "p_ac_w": pac if pac != 0.0 or any(r.get("p_ac_w") is not None for r in grp) else None,
                "p_dc_w": pdc if pdc != 0.0 or any(r.get("p_dc_w") is not None for r in grp) else None,
                "v_dc_v": vdc,
                "i_dc_a": idc if idc != 0.0 or any(r.get("i_dc_a") is not None for r in grp) else None,
                "v_ac_v": _first_finite([r.get("v_ac_v") for r in grp]),
                "i_ac_a": _first_finite([r.get("i_ac_a") for r in grp]),
                "freq_hz": _first_finite([r.get("freq_hz") for r in grp]),
                "gti": _first_finite([r.get("gti") for r in grp]),
                "ghi": _first_finite([r.get("ghi") for r in grp]),
                "dni": _first_finite([r.get("dni") for r in grp]),
                "dhi": _first_finite([r.get("dhi") for r in grp]),
                "temp_air": _first_finite([r.get("temp_air") for r in grp]),
                "meteo_qc_score": _first_finite([r.get("meteo_qc_score") for r in grp]),
                "flag_meteo_low_confidence": _bool_any([r.get("flag_meteo_low_confidence") for r in grp]),
                "flag_meteo_interpolated": _bool_any([r.get("flag_meteo_interpolated") for r in grp]),
                "flag_meteo_outlier": _bool_any([r.get("flag_meteo_outlier") for r in grp]),
                "flag_meteo_artifact": _bool_any([r.get("flag_meteo_artifact") for r in grp]),
                "alarm_code": None,
                "alarm_sev": None,
                "inv_coverage": _first_finite([r.get("inv_coverage") for r in grp]),
                "flag_low_coverage": _bool_any([r.get("flag_low_coverage") for r in grp]),
                "flag_meteo_missing": _bool_all([r.get("flag_meteo_missing") for r in grp], default=False),
                "flag_inv_missing": _bool_all([r.get("flag_inv_missing") for r in grp], default=False),
            }
        )
    return agg_source, out



def _float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    out = np.full(len(rows), np.nan, dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            continue
        try:
            out[i] = float(v)
        except Exception:
            pass
    return out



def _bool_list(rows: list[dict[str, Any]], key: str) -> list[bool]:
    return [bool(r.get(key)) for r in rows]



def run_detection_pipeline(
    *,
    plant_id: int,
    ts_start_utc: datetime,
    ts_end_utc: datetime,
    source_oper: Optional[str] = None,
    source_meteo: Optional[str] = None,
    detector_version: str = "hybrid_rules_v1",
    detection_params: Optional[DetectionParams] = None,
    rca_params: Optional[RCAParams] = None,
    delete_existing: bool = True,
) -> dict:
    plant = (
        PVPlant.objects.filter(id=plant_id)
        .select_related("details", "details__module", "details__inverter")
        .first()
    )
    if plant is None:
        raise ValueError("Plant not found")

    src_meteo = source_meteo or _pick_best_source_meteo(plant_id, ts_start_utc, ts_end_utc)
    if not src_meteo:
        raise ValueError("Nenhuma source_meteo encontrada no período")

    synthesized_agg = False
    if source_oper:
        src_oper = source_oper
        rows = _load_exact_rows(
            plant_id=plant_id,
            source_oper=src_oper,
            source_meteo=src_meteo,
            ts_start_utc=ts_start_utc,
            ts_end_utc=ts_end_utc,
        )
    else:
        src_oper = _pick_best_agg_source_oper(plant_id, src_meteo, ts_start_utc, ts_end_utc) or ""
        rows = (
            _load_exact_rows(
                plant_id=plant_id,
                source_oper=src_oper,
                source_meteo=src_meteo,
                ts_start_utc=ts_start_utc,
                ts_end_utc=ts_end_utc,
            )
            if src_oper
            else []
        )
        if not rows:
            src_oper, rows = _synthesize_agg_rows_from_mppt(
                plant_id=plant_id,
                source_meteo=src_meteo,
                ts_start_utc=ts_start_utc,
                ts_end_utc=ts_end_utc,
            )
            synthesized_agg = bool(rows)

    if not src_oper:
        raise ValueError("Nenhuma source_oper agregada encontrada no período")

    if not rows:
        return {
            "ok": True,
            "plant_id": plant_id,
            "source_oper": src_oper,
            "source_meteo": src_meteo,
            "detector_version": detector_version,
            "written_diag": 0,
            "events": 0,
            "message": "Sem dados casados no período.",
        }

    times_utc = [r["ts_utc"] for r in rows]
    pac_real = _float_array(rows, "p_ac_w")
    vdc = _float_array(rows, "v_dc_v")
    idc = _float_array(rows, "i_dc_a")
    vac = _float_array(rows, "v_ac_v")
    iac = _float_array(rows, "i_ac_a")
    freq = _float_array(rows, "freq_hz")
    gti = _float_array(rows, "gti")
    ghi = _float_array(rows, "ghi")
    dni = _float_array(rows, "dni")
    dhi = _float_array(rows, "dhi")
    temp_air = _float_array(rows, "temp_air")
    inv_coverage = _float_array(rows, "inv_coverage")
    alarm_code = _float_array(rows, "alarm_code")
    alarm_sev = _float_array(rows, "alarm_sev")

    residuals = compute_residual_series_from_observations(
        plant=plant,
        times_utc=times_utc,
        gti=gti.tolist(),
        ghi=ghi.tolist(),
        dni=dni.tolist(),
        dhi=dhi.tolist(),
        temp_air=temp_air.tolist(),
        p_ac_w=pac_real.tolist(),
        p_dc_w=_float_array(rows, "p_dc_w").tolist(),
        v_dc_v=vdc.tolist(),
        i_dc_a=idc.tolist(),
        v_ac_v=vac.tolist(),
        i_ac_a=iac.tolist(),
        freq_hz=freq.tolist(),
        meteo_qc_score=_float_array(rows, "meteo_qc_score").tolist(),
        flag_meteo_missing=_bool_list(rows, "flag_meteo_missing"),
        flag_meteo_low_confidence=_bool_list(rows, "flag_meteo_low_confidence"),
        flag_meteo_interpolated=_bool_list(rows, "flag_meteo_interpolated"),
        flag_meteo_outlier=_bool_list(rows, "flag_meteo_outlier"),
        flag_meteo_artifact=_bool_list(rows, "flag_meteo_artifact"),
        flag_inv_missing=_bool_list(rows, "flag_inv_missing"),
        inv_coverage=[None if not np.isfinite(v) else float(v) for v in inv_coverage],
        source_oper=src_oper,
        source_meteo=src_meteo,
    )
    rs = residuals.get("series") or {}
    pac_model = _float_array([{"x": v} for v in rs.get("pac_expected_w", [])], "x")
    mismatch = _float_array([{"x": v} for v in rs.get("p_ac_residual_rel", [])], "x")
    g_used = _float_array([{"x": v} for v in rs.get("g_poa_used", [])], "x")
    tcell = _float_array([{"x": v} for v in rs.get("tcell_c", [])], "x")
    p_dc_residual_rel = rs.get("p_dc_residual_rel", [])
    v_dc_residual_rel = rs.get("v_dc_residual_rel", [])
    i_dc_residual_rel = rs.get("i_dc_residual_rel", [])
    channel_confidence = rs.get("channel_confidence", {})

    valid_model = [bool(v) for v in (rs.get("valid_model") or (np.isfinite(pac_real) & np.isfinite(pac_model) & np.isfinite(mismatch)).tolist())]

    residual_channel_map = {
        "p_ac": list(rs.get("p_ac_residual_rel") or [None] * len(rows)),
        "p_dc": list(p_dc_residual_rel or [None] * len(rows)),
        "v_dc": list(v_dc_residual_rel or [None] * len(rows)),
        "i_dc": list(i_dc_residual_rel or [None] * len(rows)),
    }

    det = detect_anomalies(
        mismatch_rel=[None if not np.isfinite(v) else float(v) for v in mismatch],
        g_poa_wm2=[None if not np.isfinite(v) else float(v) for v in g_used],
        valid_model=valid_model,
        flag_meteo_missing=_bool_list(rows, "flag_meteo_missing"),
        flag_meteo_low_confidence=_bool_list(rows, "flag_meteo_low_confidence"),
        flag_meteo_interpolated=_bool_list(rows, "flag_meteo_interpolated"),
        flag_inv_missing=_bool_list(rows, "flag_inv_missing"),
        inv_coverage=[None if not np.isfinite(v) else float(v) for v in inv_coverage],
        residual_channels=residual_channel_map,
        residual_channel_confidence=channel_confidence if isinstance(channel_confidence, dict) else None,
        params=detection_params,
    )

    pac_cap_w = None
    inv = getattr(getattr(plant, "details", None), "inverter", None)
    if inv is not None and getattr(inv, "p_ac_nom_w", None) is not None:
        try:
            pac_cap_w = float(inv.p_ac_nom_w)
        except Exception:
            pac_cap_w = None

    rca = diagnose_rca_series(
        anomaly=det["anomaly"],
        valid_period=det["valid_period"],
        coarse_period=det.get("coarse_period"),
        fine_period=det.get("fine_period"),
        meteo_quality_ok=det.get("meteo_quality_ok"),
        irradiance_tier=det.get("irradiance_tier"),
        mismatch_rel=[None if not np.isfinite(v) else float(v) for v in mismatch],
        g_poa_wm2=[None if not np.isfinite(v) else float(v) for v in g_used],
        v_dc_v=[None if not np.isfinite(v) else float(v) for v in vdc],
        i_dc_a=[None if not np.isfinite(v) else float(v) for v in idc],
        p_ac_residual_rel=list(rs.get("p_ac_residual_rel") or [None] * len(rows)),
        p_dc_residual_rel=list(p_dc_residual_rel or [None] * len(rows)),
        v_dc_residual_rel=list(v_dc_residual_rel or [None] * len(rows)),
        i_dc_residual_rel=list(i_dc_residual_rel or [None] * len(rows)),
        residual_channel_confidence=channel_confidence if isinstance(channel_confidence, dict) else None,
        v_ac_v=[None if not np.isfinite(v) else float(v) for v in vac],
        i_ac_a=[None if not np.isfinite(v) else float(v) for v in iac],
        freq_hz=[None if not np.isfinite(v) else float(v) for v in freq],
        alarm_code=[None if not np.isfinite(v) else float(v) for v in alarm_code],
        alarm_sev=[None if not np.isfinite(v) else float(v) for v in alarm_sev],
        pac_real_w=[None if not np.isfinite(v) else float(v) for v in pac_real],
        pac_model_w=[None if not np.isfinite(v) else float(v) for v in pac_model],
        flag_inv_missing=_bool_list(rows, "flag_inv_missing"),
        flag_meteo_missing=_bool_list(rows, "flag_meteo_missing"),
        inv_coverage=[None if not np.isfinite(v) else float(v) for v in inv_coverage],
        pac_cap_w=pac_cap_w,
        params=rca_params,
    )

    objs: list[PlantDiagnostic15m] = []
    detection_signal = list(det.get("detection_signal_rel") or [None] * len(rows))
    for i, row in enumerate(rows):
        ewma_z = det["ewma_z"][i]
        cusum = det["cusum"][i]
        mismatch_i = None if not np.isfinite(mismatch[i]) else float(mismatch[i])
        detection_signal_i = None
        try:
            dv = detection_signal[i]
            detection_signal_i = None if dv is None else float(dv)
        except Exception:
            detection_signal_i = mismatch_i
        pac_real_i = None if not np.isfinite(pac_real[i]) else float(pac_real[i])
        pac_model_i = None if not np.isfinite(pac_model[i]) else float(pac_model[i])
        detector_score = max(
            abs(float(ewma_z)) if ewma_z is not None else 0.0,
            float(cusum) if cusum is not None else 0.0,
            abs(float(detection_signal_i)) if detection_signal_i is not None and rca["labels"][i] not in {"ok", "invalid"} else 0.0,
        )
        diagnosis_label = str(rca["diagnosis_labels"][i])
        state_label = str(rca["state_labels"][i])
        domain_label = str(rca["domain_labels"][i])
        direct_grid = bool(rca["direct_grid_evidence"][i])
        zero_inj = bool(rca["zero_injection_flag"][i])
        anomaly_final = bool(det["anomaly"][i]) or direct_grid or diagnosis_label not in {"ok", "invalid"}

        data_rel = compute_data_reliability(
            row=row,
            pac_real_w=pac_real_i,
            pac_model_w=pac_model_i,
            mismatch_rel=mismatch_i,
        )
        detection_rel = compute_detection_confidence(
            data_reliability_score=data_rel["score"],
            valid_period=bool(det["valid_period"][i]),
            coarse_period=bool(det["coarse_period"][i]),
            fine_period=bool(det["fine_period"][i]),
            meteo_quality_ok=bool(det["meteo_quality_ok"][i]),
            stable_sky=bool(det["stable_sky"][i]),
            anomaly_flag=bool(anomaly_final),
            mismatch_rel=detection_signal_i if detection_signal_i is not None else mismatch_i,
            ewma_z=ewma_z,
            cusum_score=cusum,
        )
        diagnosis_rel = compute_diagnosis_confidence(
            diagnosis_label=diagnosis_label,
            base_diagnosis_confidence=rca["diagnosis_confidence"][i],
            data_reliability_score=data_rel["score"],
            detection_confidence_score=detection_rel["score"],
            fine_diag_allowed=bool(det["fine_period"][i]),
            meteo_quality_ok=bool(det["meteo_quality_ok"][i]),
            direct_grid_evidence=direct_grid,
            zero_injection_flag=zero_inj,
            irradiance_tier=str(det["irradiance_tier"][i]),
        )
        confidence_notes = {
            "data_reliability": data_rel,
            "detection_confidence": detection_rel,
            "diagnosis_confidence": diagnosis_rel,
        }

        objs.append(
            PlantDiagnostic15m(
                plant_id=plant_id,
                ts_utc=row["ts_utc"],
                source_oper=src_oper,
                source_meteo=src_meteo,
                rca_code=int(rca["codes"][i]),
                rca_label=diagnosis_label,
                valid=bool(det["valid_period"][i]),
                anomaly_flag=bool(anomaly_final),
                detector_score=float(detector_score),
                ewma_z=ewma_z,
                cusum_score=cusum,
                stable_sky=bool(det["stable_sky"][i]),
                detector_version=detector_version,
                g_poa=None if not np.isfinite(g_used[i]) else float(g_used[i]),
                tcell_c=None if not np.isfinite(tcell[i]) else float(tcell[i]),
                pac_real_w=pac_real_i,
                pac_model_w=pac_model_i,
                mismatch_rel=mismatch_i,
                irradiance_tier=str(det["irradiance_tier"][i]),
                fine_diag_allowed=bool(det["fine_period"][i]),
                meteo_quality_ok=bool(det["meteo_quality_ok"][i]),
                direct_grid_evidence=direct_grid,
                zero_injection_flag=zero_inj,
                state_label=state_label,
                domain_label=domain_label,
                diagnosis_label=diagnosis_label,
                diagnosis_confidence=diagnosis_rel["score"],
                data_reliability_score=data_rel["score"],
                data_reliability_level=data_rel["level"],
                detection_confidence_score=detection_rel["score"],
                detection_confidence_level=detection_rel["level"],
                diagnosis_confidence_score=diagnosis_rel["score"],
                diagnosis_confidence_level=diagnosis_rel["level"],
                v_ac_v=None if not np.isfinite(vac[i]) else float(vac[i]),
                i_ac_a=None if not np.isfinite(iac[i]) else float(iac[i]),
                freq_hz=None if not np.isfinite(freq[i]) else float(freq[i]),
                alarm_code_oper=None if not np.isfinite(alarm_code[i]) else int(alarm_code[i]),
                alarm_sev_oper=None if not np.isfinite(alarm_sev[i]) else int(alarm_sev[i]),
                evidence_json=rca["evidence_json"][i],
                confidence_notes_json=confidence_notes,
            )
        )

    with transaction.atomic():
        if delete_existing:
            PlantDiagnostic15m.objects.filter(
                plant_id=plant_id,
                ts_utc__gte=ts_start_utc,
                ts_utc__lt=ts_end_utc,
                detector_version=detector_version,
                source_oper=src_oper,
                source_meteo=src_meteo,
            ).delete()
        PlantDiagnostic15m.objects.bulk_create(objs, batch_size=1000)

    events_out = build_fault_events_for_range(
        plant_id=plant_id,
        ts_start_utc=ts_start_utc,
        ts_end_utc=ts_end_utc,
        params=EventBuildParams(
            detector_version=detector_version,
            source_oper=src_oper,
            source_meteo=src_meteo,
            replace_existing=delete_existing,
        ),
    )

    return {
        "ok": True,
        "plant_id": plant_id,
        "source_oper": src_oper,
        "source_meteo": src_meteo,
        "detector_version": detector_version,
        "ts_start_utc": ts_start_utc.isoformat(),
        "ts_end_utc": ts_end_utc.isoformat(),
        "written_diag": len(objs),
        "events": int(events_out.get("events", 0)),
        "event_summary": events_out,
        "baseline": det.get("baseline"),
        "rca_baseline": rca.get("baseline"),
        "synthesized_agg": synthesized_agg,
    }
