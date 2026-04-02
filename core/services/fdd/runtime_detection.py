from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.models import PVPlant
from core.services.fdd.dashboard_common import DashboardServiceError, as_float
from core.services.fdd.runtime_types import MismatchDashboardParams
from core.services.fdd_mismatch import CODE_INVALID, classify_mismatch_series

logger = logging.getLogger(__name__)

def compute_power_model(plant: PVPlant, details: Any, times_utc: List[datetime], agg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
            transpose_ghi_to_poa_isotropic,
        )
    except Exception as exc:
        raise DashboardServiceError(f"ImportError power_model: {type(exc).__name__}: {exc}", status_code=500)

    try:
        mod = module_from_pvmodule(details.module)
        inv = getattr(details, "inverter", None)
        pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

        pld = asdict(pl) if is_dataclass(pl) else dict(getattr(pl, "__dict__", {}))
        if pld.get("lat_deg") is None:
            pld["lat_deg"] = as_float(getattr(plant, "latitude", None))
        if pld.get("lon_deg") is None:
            pld["lon_deg"] = as_float(getattr(plant, "longitude", None))
        if pld.get("tilt_deg") is None:
            pld["tilt_deg"] = as_float(getattr(details, "tilt_deg", None))
        if pld.get("azimuth_deg") is None:
            pld["azimuth_deg"] = as_float(getattr(details, "azimuth_deg", None))
        pl = pl.__class__(**pld)

        def list_to_np_nan(xs: List[Optional[float]]):
            out = np.empty(len(xs), dtype=np.float64)
            for j, v in enumerate(xs):
                try:
                    out[j] = np.nan if v is None else float(v)
                except Exception:
                    out[j] = np.nan
            return out

        gti_np = list_to_np_nan(agg["gti"])
        ghi_np = list_to_np_nan(agg["ghi"])
        dni_np = list_to_np_nan(agg["dni"])
        dhi_np = list_to_np_nan(agg["dhi"])

        mask_gti = np.isfinite(gti_np)
        has_any_gti = bool(mask_gti.any())
        ghi_arg = ghi_np if np.isfinite(ghi_np).any() else None
        dni_arg = dni_np if np.isfinite(dni_np).any() else None
        dhi_arg = dhi_np if np.isfinite(dhi_np).any() else None

        g_poa_transpo = None
        if ghi_arg is not None:
            lat = getattr(pl, "lat_deg", None)
            lon = getattr(pl, "lon_deg", None)
            tilt = getattr(pl, "tilt_deg", None)
            azs = getattr(pl, "azimuth_deg", None)
            if None not in (lat, lon, tilt, azs):
                trans_sig = inspect.signature(transpose_ghi_to_poa_isotropic)
                trans_kwargs = dict(
                    ghi=ghi_arg,
                    dhi=dhi_arg,
                    dni=dni_arg,
                    times_utc=times_utc,
                    lat_deg=float(lat),
                    lon_deg=float(lon),
                    tilt_deg=float(tilt),
                    azimuth_deg=float(azs),
                    albedo=float(getattr(pl, "albedo", 0.20) or 0.20),
                )
                if "times_shift_minutes" in trans_sig.parameters:
                    trans_kwargs["times_shift_minutes"] = float(getattr(pl, "meteo_time_shift_minutes", 0.0) or 0.0)
                trans = transpose_ghi_to_poa_isotropic(**trans_kwargs)
                g_poa_transpo = np.asarray(trans.get("g_poa"), dtype=float)

        if has_any_gti:
            if g_poa_transpo is not None and g_poa_transpo.size == gti_np.size:
                g_poa_used_np = np.where(mask_gti, gti_np, g_poa_transpo)
            else:
                g_poa_used_np = gti_np
        else:
            if g_poa_transpo is not None and g_poa_transpo.size == gti_np.size:
                g_poa_used_np = g_poa_transpo
            else:
                g_poa_used_np = ghi_arg if ghi_arg is not None else np.full_like(gti_np, np.nan)

        g_poa_used = [None if (not np.isfinite(v)) else float(v) for v in g_poa_used_np.tolist()]
        tamb_np = list_to_np_nan(agg["temp_air"])
        pac_real_np = list_to_np_nan(agg["p_ac_w"])
        sig = inspect.signature(expected_and_mismatch)
        kwargs: Dict[str, Any] = dict(
            g_poa=g_poa_used_np,
            tamb_c=tamb_np,
            pac_real_w=pac_real_np,
            module=mod,
            plant=pl,
            g_min_valid=0.0,
            n_points=60,
            eps_w=50.0,
        )
        if "times_utc" in sig.parameters:
            kwargs["times_utc"] = times_utc
        if "dt_minutes" in sig.parameters:
            kwargs["dt_minutes"] = 15.0
        if "window_minutes" in sig.parameters:
            kwargs["window_minutes"] = 60.0

        out_model = expected_and_mismatch(**kwargs) or {}
        pac_expected = out_model.get("pac_expected_w")
        mismatch = out_model.get("mismatch_rel")
        valid_model_np = out_model.get("valid")
        tcell_np = out_model.get("tcell_c")
        if pac_expected is None:
            raise DashboardServiceError("power_model não retornou pac_expected_w.", status_code=500)

        pac_model_w = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(pac_expected, dtype=float).tolist()]
        if mismatch is None:
            eps = 50.0
            mm: List[Optional[float]] = []
            for pr, pm in zip(agg["p_ac_w"], pac_model_w):
                if pr is None or pm is None:
                    mm.append(None)
                    continue
                den = max(abs(pm), eps)
                mm.append((float(pr) - float(pm)) / float(den))
            mismatch_rel = mm
        else:
            mismatch_rel = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(mismatch, dtype=float).tolist()]

        if valid_model_np is None:
            valid_model = [False if (m is None) else True for m in mismatch_rel]
        else:
            valid_model = [bool(v) for v in list(np.asarray(valid_model_np, dtype=bool).tolist())]

        if tcell_np is None:
            tcell_c = [None] * len(times_utc)
        else:
            tcell_c = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(tcell_np, dtype=float).tolist()]

        return {
            "g_poa_used": g_poa_used,
            "pac_model_w": pac_model_w,
            "mismatch_rel": mismatch_rel,
            "valid_model": valid_model,
            "tcell_c": tcell_c,
            "np": np,
        }
    except DashboardServiceError:
        raise
    except Exception as exc:
        logger.exception("Falha no power_model (dashboard_runtime) plant_id=%s", plant.id)
        raise DashboardServiceError(f"{type(exc).__name__}: {exc}", status_code=500)

def build_base_gate(params: MismatchDashboardParams, model: Dict[str, Any], agg: Dict[str, Any]) -> List[bool]:
    base_gate: List[bool] = []
    for i in range(len(model["mismatch_rel"])):
        gp = model["g_poa_used"][i]
        pr = agg["p_ac_w"][i]
        ok = bool(model["valid_model"][i])
        ok = ok and (gp is not None) and (float(gp) >= float(params.gpoa_gate))
        ok = ok and (pr is not None) and (float(pr) >= float(params.pmin_w))
        ok = ok and (not bool(agg["flag_meteo_missing"][i]))
        ok = ok and (not bool(agg["flag_inv_missing_all"][i]))
        base_gate.append(ok)
    return base_gate

def pick_diag_row_for_ts(ts_utc: datetime, per_ts: Dict[datetime, Dict[str, Dict[str, Any]]], selected_sources: List[str]) -> Optional[Dict[str, Any]]:
    by_src = per_ts.get(ts_utc, {})
    if not by_src:
        return None
    for s in selected_sources:
        rr = by_src.get(s)
        if rr is not None:
            return rr
    try:
        return next(iter(by_src.values()))
    except Exception:
        return None

def build_alarm_vectors(times_utc: List[datetime], per_ts: Dict[datetime, Dict[str, Dict[str, Any]]], selected_sources: List[str]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    alarm_code: List[Optional[int]] = []
    alarm_sev: List[Optional[int]] = []
    for ts_utc in times_utc:
        row = pick_diag_row_for_ts(ts_utc, per_ts, selected_sources)
        alarm_code.append(None if row is None or row.get("alarm_code") is None else int(row.get("alarm_code")))
        alarm_sev.append(None if row is None or row.get("alarm_sev") is None else int(row.get("alarm_sev")))
    return alarm_code, alarm_sev

def run_detection_and_rca(
    *,
    plant: PVPlant,
    details: Any,
    params: MismatchDashboardParams,
    times_utc: List[datetime],
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]],
    selected_sources: List[str],
    agg: Dict[str, Any],
    model: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(times_utc)
    base_gate = build_base_gate(params, model, agg)

    coarse_period = [False] * n
    fine_period = [False] * n
    meteo_quality_ok = [False] * n
    irradiance_tier = ["N"] * n
    rca: Dict[str, Any] = {}
    ewma_z: List[Optional[float]] = [None] * n
    cusum_score: List[Optional[float]] = [None] * n
    freq_hz: List[Optional[float]] = list(agg.get("freq_hz") or ([None] * n))
    det_dbg: Dict[str, Any] = {}
    rca_dbg: Dict[str, Any] = {}

    if params.use_legacy:
        out_cls = classify_mismatch_series(
            times_utc=times_utc,
            mismatch_rel=model["mismatch_rel"],
            g_poa_wm2=model["g_poa_used"],
            valid=base_gate,
            thresholds=params.thr,
        )
        codes = [int(c) for c in out_cls["codes"]]
        labels = [str(x) for x in out_cls["labels"]]
        valid_period = [bool(v) for v in base_gate]
        anomaly = [False] * n
        stable_sky = [False] * n
        coarse_period = valid_period[:]
        fine_period = valid_period[:]
        meteo_quality_ok = [not bool(v) for v in agg["flag_meteo_missing"]]
        irradiance_tier = ["C" if bool(v) else "N" for v in valid_period]
        pipeline_name = "legacy_mismatch_classifier"
    else:
        try:
            from core.services.fdd.detection import DetectionParams, detect_anomalies
            from core.services.fdd.rca import RCAParams, diagnose_rca_series
        except Exception as exc:
            raise DashboardServiceError(
                f"ImportError fdd/detection ou fdd/rca: {type(exc).__name__}: {exc}",
                status_code=500,
            )

        det_sig = inspect.signature(DetectionParams)
        det_param_names = set(det_sig.parameters.keys())
        if "gpoa_gate_wm2" in det_param_names:
            det_params = DetectionParams(
                gpoa_gate_wm2=float(params.gpoa_gate),
                stable_cv_max=params.get_float("stable_cv_max", 0.08),
                stable_window_points=params.get_int("stable_window_points", 6),
                ewma_lambda=params.get_float("ewma_lambda", 0.20),
                ewma_L=params.get_float("ewma_L", 3.0),
                cusum_k=params.get_float("cusum_k", 0.50),
                cusum_h=params.get_float("cusum_h", 8.0),
                min_baseline_points=params.get_int("min_baseline_points", 24),
                inv_cov_min=params.get_float("inv_cov_min", 0.30),
            )
        else:
            det_params = DetectionParams(
                sun_available_gpoa_wm2=params.get_float("sun_available_gpoa_wm2", max(150.0, float(params.gpoa_gate))),
                coarse_diag_gpoa_wm2=params.get_float("coarse_diag_gpoa_wm2", max(700.0, float(params.gpoa_gate))),
                fine_diag_gpoa_wm2=params.get_float("fine_diag_gpoa_wm2", max(800.0, float(params.gpoa_gate))),
                stable_cv_max=params.get_float("stable_cv_max", 0.08),
                stable_ramp_max_wm2=params.get_float("stable_ramp_max_wm2", 120.0),
                stable_window_points=params.get_int("stable_window_points", 6),
                ewma_lambda=params.get_float("ewma_lambda", 0.20),
                ewma_L=params.get_float("ewma_L", 3.0),
                cusum_k=params.get_float("cusum_k", 0.50),
                cusum_h=params.get_float("cusum_h", 8.0),
                min_baseline_points=params.get_int("min_baseline_points", 24),
                inv_cov_min=params.get_float("inv_cov_min", 0.30),
            )

        det = detect_anomalies(
            mismatch_rel=model["mismatch_rel"],
            g_poa_wm2=model["g_poa_used"],
            valid_model=base_gate,
            flag_meteo_missing=agg["flag_meteo_missing"],
            flag_meteo_low_confidence=agg["flag_meteo_low_confidence"],
            flag_meteo_interpolated=agg["flag_meteo_interpolated"],
            flag_inv_missing=agg["flag_inv_missing_all"],
            inv_coverage=agg["inv_cov"],
            params=det_params,
        ) or {}

        valid_period = [bool(v) for v in (det.get("valid_period") or base_gate)]
        anomaly = [bool(v) for v in (det.get("anomaly") or [False] * n)]
        stable_sky = [bool(v) for v in (det.get("stable_sky") or [False] * n)]
        coarse_period = [bool(v) for v in (det.get("coarse_period") or valid_period)]
        fine_period = [bool(v) for v in (det.get("fine_period") or [False] * n)]
        meteo_quality_ok = [bool(v) for v in (det.get("meteo_quality_ok") or stable_sky)]
        irradiance_tier = [str(v) for v in (det.get("irradiance_tier") or ["N"] * n)]

        ewma_z = list(det.get("ewma_z") or ([None] * n))
        cusum_score = list(det.get("cusum") or ([None] * n))
        if len(ewma_z) < n:
            ewma_z.extend([None] * (n - len(ewma_z)))
        else:
            ewma_z = ewma_z[:n]
        if len(cusum_score) < n:
            cusum_score.extend([None] * (n - len(cusum_score)))
        else:
            cusum_score = cusum_score[:n]

        det_dbg = {
            "z": det.get("z"),
            "ewma_z": ewma_z,
            "cusum": cusum_score,
            "baseline": det.get("baseline"),
            "coarse_period": coarse_period,
            "fine_period": fine_period,
            "meteo_quality_ok": meteo_quality_ok,
            "meteo_qc_score": agg["meteo_qc_score"],
            "flag_meteo_low_confidence": agg["flag_meteo_low_confidence"],
            "flag_meteo_interpolated": agg["flag_meteo_interpolated"],
            "flag_meteo_outlier": agg["flag_meteo_outlier"],
            "flag_meteo_artifact": agg["flag_meteo_artifact"],
            "irradiance_tier": irradiance_tier,
        }

        pac_cap_w = None
        try:
            inv_obj = getattr(details, "inverter", None)
            for attr in ("pac_nom_w", "p_ac_nom_w", "rated_power_w", "pnom_w", "pac_nom_kw", "rated_power_kw"):
                vv = getattr(inv_obj, attr, None) if inv_obj is not None else None
                if vv is None:
                    continue
                pac_cap_w = float(vv) * (1000.0 if str(attr).endswith("_kw") else 1.0)
                break
        except Exception:
            pac_cap_w = None

        rca_sig = inspect.signature(RCAParams)
        rca_param_names = set(rca_sig.parameters.keys())
        if "warn_abs" in rca_param_names and "fault_abs" in rca_param_names:
            rca_params = RCAParams(
                warn_abs=float(params.thr.warn_abs),
                fault_abs=float(params.thr.fault_abs),
                min_baseline_points=params.get_int("rca_min_baseline_points", 24),
            )
        else:
            rca_params = RCAParams(
                sun_available_gpoa_wm2=params.get_float("sun_available_gpoa_wm2", max(150.0, float(params.gpoa_gate))),
                expected_power_min_w=float(params.pmin_w),
                zero_abs_w=params.get_float("zero_abs_w", 100.0),
                zero_rel_model=params.get_float("zero_rel_model", 0.05),
                degraded_rel=params.get_float("degraded_rel", 0.25),
                severe_rel=params.get_float("severe_rel", 0.50),
                low_i_ratio_warn=params.get_float("low_i_ratio_warn", 0.35),
                low_i_ratio_crit=params.get_float("low_i_ratio_crit", 0.15),
                low_v_ratio_warn=params.get_float("low_v_ratio_warn", 0.80),
                low_v_ratio_crit=params.get_float("low_v_ratio_crit", 0.60),
                vac_low_ratio=params.get_float("vac_low_ratio", 0.90),
                vac_high_ratio=params.get_float("vac_high_ratio", 1.10),
                vac_abs_margin_v=params.get_float("vac_abs_margin_v", 10.0),
                freq_abs_tol_hz=params.get_float("freq_abs_tol_hz", 1.0),
                clip_margin=params.get_float("clip_margin", 0.98),
                clip_model_margin=params.get_float("clip_model_margin", 1.02),
                min_baseline_points=params.get_int("rca_min_baseline_points", 24),
            )

        diag_sig = inspect.signature(diagnose_rca_series)
        diag_param_names = set(diag_sig.parameters.keys())
        diag_kwargs = dict(
            anomaly=anomaly,
            valid_period=valid_period,
            mismatch_rel=model["mismatch_rel"],
            v_dc_v=agg["v_dc_v"],
            i_dc_a=agg["i_dc_a"],
            pac_real_w=agg["p_ac_w"],
            pac_model_w=model["pac_model_w"],
            flag_inv_missing=agg["flag_inv_missing_all"],
            flag_meteo_missing=agg["flag_meteo_missing"],
            inv_coverage=agg["inv_cov"],
            pac_cap_w=pac_cap_w,
            params=rca_params,
        )
        if "g_poa_wm2" in diag_param_names:
            diag_kwargs["g_poa_wm2"] = model["g_poa_used"]
        if "coarse_period" in diag_param_names:
            diag_kwargs["coarse_period"] = coarse_period
        if "fine_period" in diag_param_names:
            diag_kwargs["fine_period"] = fine_period
        if "meteo_quality_ok" in diag_param_names:
            diag_kwargs["meteo_quality_ok"] = meteo_quality_ok
        if "irradiance_tier" in diag_param_names:
            diag_kwargs["irradiance_tier"] = irradiance_tier
        if "v_ac_v" in diag_param_names:
            diag_kwargs["v_ac_v"] = agg["v_ac_v"]
        if "i_ac_a" in diag_param_names:
            diag_kwargs["i_ac_a"] = agg["i_ac_a"]
        if "freq_hz" in diag_param_names:
            diag_kwargs["freq_hz"] = freq_hz

        alarm_code, alarm_sev = build_alarm_vectors(times_utc, per_ts, selected_sources)
        if "alarm_code" in diag_param_names:
            diag_kwargs["alarm_code"] = alarm_code
        if "alarm_sev" in diag_param_names:
            diag_kwargs["alarm_sev"] = alarm_sev

        rca = diagnose_rca_series(**diag_kwargs) or {}
        rca_codes_raw = rca.get("codes") or [0] * n
        rca_labels_raw = rca.get("labels") or ["normal"] * n
        codes: List[int] = [CODE_INVALID] * n
        labels: List[str] = ["invalid"] * n
        for i in range(n):
            if not valid_period[i]:
                codes[i] = CODE_INVALID
                labels[i] = "invalid"
                continue
            if not anomaly[i]:
                codes[i] = 0
                labels[i] = "normal"
                continue
            try:
                c = int(rca_codes_raw[i])
            except Exception:
                c = 2
            codes[i] = c
            labels[i] = str(rca_labels_raw[i] or "anom")

        rca_dbg = {"baseline": rca.get("baseline"), "alarm_code": alarm_code, "alarm_sev": alarm_sev}
        pipeline_name = "ewma_cusum_detection + rca_patterns"
        return {
            "pipeline_name": pipeline_name,
            "valid_period": valid_period,
            "anomaly": anomaly,
            "stable_sky": stable_sky,
            "coarse_period": coarse_period,
            "fine_period": fine_period,
            "meteo_quality_ok": meteo_quality_ok,
            "irradiance_tier": irradiance_tier,
            "ewma_z": ewma_z,
            "cusum_score": cusum_score,
            "freq_hz": freq_hz,
            "rca": rca,
            "codes": codes,
            "labels": labels,
            "alarm_code": alarm_code,
            "alarm_sev": alarm_sev,
            "det_dbg": det_dbg,
            "rca_dbg": rca_dbg,
        }

    return {
        "pipeline_name": pipeline_name,
        "valid_period": valid_period,
        "anomaly": anomaly,
        "stable_sky": stable_sky,
        "coarse_period": coarse_period,
        "fine_period": fine_period,
        "meteo_quality_ok": meteo_quality_ok,
        "irradiance_tier": irradiance_tier,
        "ewma_z": ewma_z,
        "cusum_score": cusum_score,
        "freq_hz": freq_hz,
        "rca": rca,
        "codes": codes,
        "labels": labels,
        "alarm_code": [None] * n,
        "alarm_sev": [None] * n,
        "det_dbg": det_dbg,
        "rca_dbg": rca_dbg,
    }

