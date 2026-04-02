from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from core.models import PVPlant
from core.services.fdd.dashboard_common import (
    MISMATCH_VERSION_SUMMARY,
    DashboardServiceError,
    mean_none,
    parse_date,
    runtime_severity,
)
from core.services.fdd.aggregation import DUMP_FIELDS, RCA_CODE_TO_SEV, aggregate_runtime_series
from core.services.fdd.dump_builder import build_runtime_dump
from core.services.fdd.runtime_confidence import build_runtime_confidence, compute_plot_mismatch
from core.services.fdd.runtime_detection import compute_power_model, run_detection_and_rca
from core.services.fdd.runtime_persistence import persist_runtime_outputs
from core.services.fdd.source_selection import ensure_plant_configuration, group_runtime_rows, query_runtime_rows
from core.services.fdd_mismatch import MismatchThresholds


from core.services.fdd.runtime_types import MismatchDashboardParams

def parse_dashboard_params(data: Mapping[str, Any], tz_name: str) -> MismatchDashboardParams:
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
        tz_name = "UTC"

    d0 = parse_date(data.get("start") or "")
    d1 = parse_date(data.get("end") or "")
    if not d0 or not d1:
        raise DashboardServiceError("start/end (YYYY-MM-DD) são obrigatórios", status_code=400)
    if d1 < d0:
        raise DashboardServiceError("end < start", status_code=400)

    dt0_local = datetime.combine(d0, time.min, tzinfo=tz)
    dt1_local = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    def _gf(key: str, default: float) -> float:
        raw = data.get(key)
        if raw in (None, ""):
            return float(default)
        try:
            return float(str(raw).replace(",", "."))
        except Exception:
            return float(default)

    def _gi(key: str, default: int) -> int:
        raw = data.get(key)
        if raw in (None, ""):
            return int(default)
        try:
            return int(float(str(raw).replace(",", ".")))
        except Exception:
            return int(default)

    gpoa_gate = _gf("gpoa_gate", _gf("gpoa_min", 50.0))
    pmin_w = _gf("pmin_w", 0.0)
    thr = MismatchThresholds(
        gpoa_gate_wm2=gpoa_gate,
        warn_abs=_gf("warn_abs", 0.35),
        fault_abs=_gf("fault_abs", 0.90),
        meteo_pos_abs=_gf("meteo_pos_abs", 0.25),
        shading_std_abs=_gf("shading_std_abs", 0.22),
        shading_window_points=_gi("shading_window_points", 6),
        dt_minutes=15.0,
        max_gap_minutes=_gf("max_gap_minutes", 30.0),
    )

    return MismatchDashboardParams(
        raw_data=data,
        start=d0,
        end=d1,
        tz_name=tz_name,
        tz=tz,
        dt0_utc=dt0_utc,
        dt1_utc=dt1_utc,
        source_oper_raw=(data.get("source_oper") or data.get("src_oper") or "").strip(),
        source_meteo=((data.get("source_meteo") or data.get("src_meteo") or "").strip() or None),
        gpoa_gate=gpoa_gate,
        pmin_w=pmin_w,
        thr=thr,
        use_legacy=(str(data.get("legacy") or data.get("use_legacy") or "").strip().lower() in ("1", "true", "yes", "on")),
        persist=(str(data.get("persist") or data.get("save") or "").strip().lower() in ("1", "true", "yes", "on")),
        gpoa_plot_min=_gf("gpoa_plot_min", max(700.0, float(gpoa_gate))),
        pmodel_plot_min=_gf("pmodel_plot_min", max(200.0, float(pmin_w))),
        mismatch_clip_abs=_gf("mismatch_clip_abs", 2.0),
    )

def build_mismatch_dashboard_payload(plant: PVPlant, params: MismatchDashboardParams) -> Dict[str, Any]:
    details, _ = ensure_plant_configuration(plant)
    src_meteo, source_oper_list, selected_sources, rows = query_runtime_rows(plant, params)
    per_ts, times_utc = group_runtime_rows(rows)
    agg = aggregate_runtime_series(per_ts=per_ts, times_utc=times_utc, selected_sources=selected_sources)

    x_local_dt = [t.astimezone(params.tz) for t in times_utc]
    x_local = [t.isoformat() for t in x_local_dt]
    x_utc = [t.astimezone(dt_tz.utc).isoformat() for t in times_utc]
    hm_day_local = [t.date().isoformat() for t in x_local_dt]
    hm_minute_local = [t.hour * 60 + t.minute for t in x_local_dt]

    model = compute_power_model(plant, details, times_utc, agg)
    pipeline = run_detection_and_rca(
        plant=plant,
        details=details,
        params=params,
        times_utc=times_utc,
        per_ts=per_ts,
        selected_sources=selected_sources,
        agg=agg,
        model=model,
    )
    confidence = build_runtime_confidence(
        times_utc=times_utc,
        per_ts=per_ts,
        selected_sources=selected_sources,
        agg=agg,
        model=model,
        pipeline=pipeline,
    )
    plot_data = compute_plot_mismatch(params, agg, model)
    persist = persist_runtime_outputs(
        plant=plant,
        params=params,
        src_meteo=src_meteo,
        selected_sources=selected_sources,
        times_utc=times_utc,
        model=model,
        agg=agg,
        pipeline=pipeline,
        confidence=confidence,
    )
    dump_by_tkey = build_runtime_dump(
        tz=params.tz,
        src_meteo=src_meteo,
        selected_sources=selected_sources,
        times_utc=times_utc,
        per_ts=per_ts,
        agg=agg,
        confidence=confidence,
    )

    sev_runtime: List[str] = []
    sev_counts = {"none": 0, "ok": 0, "warn": 0, "crit": 0}
    for i in range(len(times_utc)):
        sev = runtime_severity(
            state_label=confidence["diag_state_labels"][i],
            diagnosis_label=confidence["diag_diagnosis_labels"][i],
            direct_grid_evidence=bool(confidence["diag_direct_grid"][i]),
            anomaly_flag=bool(pipeline["anomaly"][i]),
        )
        sev_runtime.append(sev)
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    return {
        "ok": True,
        "pipeline": pipeline["pipeline_name"],
        "plant": {"id": plant.id, "nome": plant.nome, "tz": params.tz_name},
        "range": {
            "start": params.start.isoformat(),
            "end": params.end.isoformat(),
            "start_utc": params.dt0_utc.isoformat(),
            "end_utc_excl": params.dt1_utc.isoformat(),
            "source_meteo": src_meteo,
            "selected_sources": selected_sources,
        },
        "versions": MISMATCH_VERSION_SUMMARY,
        "confidence_summary": {
            "data_reliability_mean": mean_none(confidence["data_reliability_score"]),
            "detection_confidence_mean": mean_none(confidence["detection_confidence_score"]),
            "diagnosis_confidence_mean": mean_none(confidence["diagnosis_confidence_score"]),
        },
        "sources": {
            "source_meteo": src_meteo,
            "source_oper_list": source_oper_list,
            "selected_sources": selected_sources,
            "total_policy": "prefer_mppt_sum",
        },
        "x_local": x_local,
        "x_utc": x_utc,
        "rca_code_to_sev": RCA_CODE_TO_SEV,
        "dump_fields": DUMP_FIELDS,
        "dump_by_tkey": dump_by_tkey,
        "series": {
            "t_local": x_local,
            "t_utc": x_utc,
            "g_poa": model["g_poa_used"],
            "g_poa_used": model["g_poa_used"],
            "gti": agg["gti"],
            "ghi": agg["ghi"],
            "dni": agg["dni"],
            "dhi": agg["dhi"],
            "temp_air": agg["temp_air"],
            "wind_speed": agg["wind_speed"],
            "rh": agg["rh"],
            "meteo_qc_score": agg["meteo_qc_score"],
            "flag_meteo_low_confidence": agg["flag_meteo_low_confidence"],
            "flag_meteo_interpolated": agg["flag_meteo_interpolated"],
            "flag_meteo_outlier": agg["flag_meteo_outlier"],
            "flag_meteo_artifact": agg["flag_meteo_artifact"],
            "flag_meteo_missing": agg["flag_meteo_missing"],
            "p_ac_w": agg["p_ac_w"],
            "p_ac_real_w": agg["p_ac_w"],
            "p_dc_w": agg["p_dc_w"],
            "e_ac_wh_15": agg["e_ac_wh_15"],
            "v_dc_v": agg["v_dc_v"],
            "i_dc_a": agg["i_dc_a"],
            "v_ac_v": agg["v_ac_v"],
            "i_ac_a": agg["i_ac_a"],
            "freq_hz": agg["freq_hz"],
            "inv_coverage": agg["inv_cov"],
            "flag_inv_missing": agg["flag_inv_missing_all"],
            "flag_inv_missing_all": agg["flag_inv_missing_all"],
            "flag_inv_missing_partial": agg["flag_inv_missing_partial"],
            "p_ac_mppt_sum_w": agg["p_ac_mppt_sum_w"],
            "p_ac_agg_w": agg["p_ac_agg_w"],
            "policy_used": agg["policy_used"],
            "p_ac_model_w": model["pac_model_w"],
            "tcell_c": model["tcell_c"],
            "mismatch_rel": plot_data["mismatch_rel_plot"],
            "mismatch_rel_raw": plot_data["mismatch_rel_raw"],
            "gpoa_plot_min": [float(params.gpoa_plot_min)] * len(times_utc),
            "data_reliability_score": confidence["data_reliability_score"],
            "data_reliability_level": confidence["data_reliability_level"],
            "detection_confidence_score": confidence["detection_confidence_score"],
            "detection_confidence_level": confidence["detection_confidence_level"],
            "diagnosis_confidence_score": confidence["diagnosis_confidence_score"],
            "diagnosis_confidence_level": confidence["diagnosis_confidence_level"],
            "state_label": confidence["diag_state_labels"],
            "domain_label": confidence["diag_domain_labels"],
            "diagnosis_label": confidence["diag_diagnosis_labels"],
            "direct_grid_evidence": confidence["diag_direct_grid"],
            "zero_injection_flag": confidence["diag_zero_inj"],
            "irradiance_tier": pipeline["irradiance_tier"],
            "pmodel_plot_min": [float(params.pmodel_plot_min)] * len(times_utc),
            "valid_model": model["valid_model"],
            "valid_period": pipeline["valid_period"],
            "valid": pipeline["valid_period"],
            "stable_sky": pipeline["stable_sky"],
            "anomaly": pipeline["anomaly"],
            "rca_code": pipeline["codes"],
            "rca_label": pipeline["labels"],
            "codes": pipeline["codes"],
            "labels": pipeline["labels"],
            "sev_runtime": sev_runtime,
            "hm_day_local": hm_day_local,
            "hm_minute_local": hm_minute_local,
        },
        "series_by_source": agg["series_by_source"],
        "summary": {
            "counts": sev_counts,
            "events": [],
            "n_points": len(times_utc),
            "state_label_counts": dict(Counter(str(v or "unknown") for v in confidence["diag_state_labels"])),
            "diagnosis_label_counts": dict(Counter(str(v or "invalid") for v in confidence["diag_diagnosis_labels"])),
        },
        "thresholds": {
            "gpoa_gate": params.gpoa_gate,
            "pmin_w": params.pmin_w,
            "warn_abs": float(params.thr.warn_abs),
            "fault_abs": float(params.thr.fault_abs),
        },
        "debug": {
            "det": pipeline["det_dbg"],
            "rca": pipeline["rca_dbg"],
        },
        "persist": persist,
    }

