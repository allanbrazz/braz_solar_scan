from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
from django.db import transaction
from django.db.models import Count

from core.models import PVPlant, PVPlantMergedRecord15m, PlantDiagnostic15m
from core.services.fdd.detection import DetectionParams, detect_anomalies
from core.services.fdd.events import EventBuildParams, build_fault_events_for_range
from core.services.fdd.rca import RCAParams, diagnose_rca_series
from core.services.mppt_gnn_fdd.window_loader import compute_pac_model_and_mismatch


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


def _pick_best_source_oper(plant_id: int, source_meteo: str, ts_start_utc: datetime, ts_end_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_oper")


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
    detector_version: str = "residual_v1",
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

    src_oper = source_oper or _pick_best_source_oper(plant_id, src_meteo, ts_start_utc, ts_end_utc)
    if not src_oper:
        raise ValueError("Nenhuma source_oper encontrada no período")

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_oper=src_oper,
            source_meteo=src_meteo,
            ts_utc__gte=ts_start_utc,
            ts_utc__lt=ts_end_utc,
        )
        .order_by("ts_utc")
        .values(
            "ts_utc",
            "p_ac_w",
            "v_dc_v",
            "i_dc_a",
            "gti",
            "ghi",
            "dni",
            "dhi",
            "temp_air",
            "alarm_code",
            "alarm_sev",
            "inv_coverage",
            "flag_low_coverage",
            "flag_meteo_missing",
            "flag_inv_missing",
        )
    )
    if not rows:
        return {
            "ok": True,
            "plant_id": plant_id,
            "source_oper": src_oper,
            "source_meteo": src_meteo,
            "written_diag": 0,
            "events": 0,
            "message": "Sem dados casados no período.",
        }

    times_utc = [r["ts_utc"] for r in rows]
    pac_real = _float_array(rows, "p_ac_w")
    vdc = _float_array(rows, "v_dc_v")
    idc = _float_array(rows, "i_dc_a")
    gti = _float_array(rows, "gti")
    ghi = _float_array(rows, "ghi")
    dni = _float_array(rows, "dni")
    dhi = _float_array(rows, "dhi")
    temp_air = _float_array(rows, "temp_air")
    inv_coverage = _float_array(rows, "inv_coverage")

    pac_model, mismatch = compute_pac_model_and_mismatch(
        plant=plant,
        times_utc=times_utc,
        gti=gti,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        temp_air=temp_air,
        pac_real=pac_real,
    )

    g_used = np.where(np.isfinite(gti), gti, ghi)

    valid_model = (
        np.isfinite(pac_real)
        & np.isfinite(pac_model)
        & np.isfinite(mismatch)
    ).tolist()

    det = detect_anomalies(
        mismatch_rel=[None if not np.isfinite(v) else float(v) for v in mismatch],
        g_poa_wm2=[None if not np.isfinite(v) else float(v) for v in g_used],
        valid_model=valid_model,
        flag_meteo_missing=_bool_list(rows, "flag_meteo_missing"),
        flag_inv_missing=_bool_list(rows, "flag_inv_missing"),
        inv_coverage=[None if not np.isfinite(v) else float(v) for v in inv_coverage],
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
        mismatch_rel=[None if not np.isfinite(v) else float(v) for v in mismatch],
        v_dc_v=[None if not np.isfinite(v) else float(v) for v in vdc],
        i_dc_a=[None if not np.isfinite(v) else float(v) for v in idc],
        pac_real_w=[None if not np.isfinite(v) else float(v) for v in pac_real],
        pac_model_w=[None if not np.isfinite(v) else float(v) for v in pac_model],
        flag_inv_missing=_bool_list(rows, "flag_inv_missing"),
        flag_meteo_missing=_bool_list(rows, "flag_meteo_missing"),
        inv_coverage=[None if not np.isfinite(v) else float(v) for v in inv_coverage],
        pac_cap_w=pac_cap_w,
        params=rca_params,
    )

    objs: list[PlantDiagnostic15m] = []
    for i, row in enumerate(rows):
        ewma_z = det["ewma_z"][i]
        cusum = det["cusum"][i]
        detector_score = max(
            abs(float(ewma_z)) if ewma_z is not None else 0.0,
            float(cusum) if cusum is not None else 0.0,
        )
        objs.append(
            PlantDiagnostic15m(
                plant_id=plant_id,
                ts_utc=row["ts_utc"],
                rca_code=int(rca["codes"][i]),
                rca_label=str(rca["labels"][i]),
                valid=bool(det["valid_period"][i]),
                anomaly_flag=bool(det["anomaly"][i]),
                detector_score=float(detector_score),
                ewma_z=ewma_z,
                cusum_score=cusum,
                stable_sky=bool(det["stable_sky"][i]),
                detector_version=detector_version,
                g_poa=None if not np.isfinite(g_used[i]) else float(g_used[i]),
                tcell_c=None,
                pac_real_w=None if not np.isfinite(pac_real[i]) else float(pac_real[i]),
                pac_model_w=None if not np.isfinite(pac_model[i]) else float(pac_model[i]),
                mismatch_rel=None if not np.isfinite(mismatch[i]) else float(mismatch[i]),
            )
        )

    with transaction.atomic():
        if delete_existing:
            PlantDiagnostic15m.objects.filter(
                plant_id=plant_id,
                ts_utc__gte=ts_start_utc,
                ts_utc__lt=ts_end_utc,
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
            replace_existing=True,
        ),
    )

    return {
        "ok": True,
        "plant_id": plant_id,
        "source_oper": src_oper,
        "source_meteo": src_meteo,
        "ts_start_utc": ts_start_utc.isoformat(),
        "ts_end_utc": ts_end_utc.isoformat(),
        "written_diag": len(objs),
        "events": int(events_out.get("events", 0)),
        "event_summary": events_out,
        "baseline": det.get("baseline"),
        "rca_baseline": rca.get("baseline"),
    }