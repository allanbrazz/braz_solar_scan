from __future__ import annotations

import inspect
import logging
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.models import PVPlant
from core.services.fdd.dashboard_common import DashboardServiceError, as_float
from core.services.fdd.runtime_types import MismatchDashboardParams
from core.services.fdd_mismatch import CODE_INVALID, classify_mismatch_series

logger = logging.getLogger(__name__)


def _mppt_no_from_source(src: Any) -> Optional[int]:
    s = str(src or "")
    m = re.search(r"(?:^|\|)MPPT\s*([0-9]+)\b", s, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _float_list(xs: Any, *, np_mod) -> List[Optional[float]]:
    if xs is None:
        return []
    out: List[Optional[float]] = []
    for v in list(np_mod.asarray(xs, dtype=float).tolist()):
        out.append(None if (not np_mod.isfinite(v)) else float(v))
    return out


def _fill_aggregate_dc_from_mppt_model(*, agg: Dict[str, Any], mppt_model_by_source: Dict[str, Dict[str, Any]], v_dc_model_v: List[Optional[float]], i_dc_model_a: List[Optional[float]]) -> None:
    if not mppt_model_by_source:
        return
    sbs = agg.get("series_by_source") or {}
    n = len(v_dc_model_v)
    for i in range(n):
        vv = []
        ii = []
        for src, model_block in mppt_model_by_source.items():
            meas = sbs.get(src) or {}
            meas_pdc = None
            meas_idc = None
            try:
                meas_pdc = (meas.get("p_dc_w") or [None] * n)[i]
            except Exception:
                meas_pdc = None
            try:
                meas_idc = (meas.get("i_dc_a") or [None] * n)[i]
            except Exception:
                meas_idc = None
            active = False
            try:
                active = (meas_pdc is not None and float(meas_pdc) > 1.0) or (meas_idc is not None and float(meas_idc) > 0.2)
            except Exception:
                active = False
            if not active:
                continue
            mv = None
            mi = None
            try:
                mv = (model_block.get("v_dc_model_v") or [None] * n)[i]
            except Exception:
                mv = None
            try:
                mi = (model_block.get("i_dc_model_a") or [None] * n)[i]
            except Exception:
                mi = None
            if mv is not None:
                vv.append(float(mv))
            if mi is not None:
                ii.append(float(mi))
        if v_dc_model_v[i] is None and vv:
            v_dc_model_v[i] = float(sum(vv) / len(vv))
        if i_dc_model_a[i] is None and ii:
            i_dc_model_a[i] = float(sum(ii))



def compute_power_model(plant: PVPlant, details: Any, times_utc: List[datetime], agg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
            transpose_ghi_to_poa_isotropic,
            expected_dc_by_mppt_from_details,
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
        pdc_expected = out_model.get("pdc_expected_w")
        vdc_expected = out_model.get("v_dc_expected_v")
        idc_expected = out_model.get("i_dc_expected_a")
        v_ratio_np = out_model.get("v_ratio")
        i_ratio_np = out_model.get("i_ratio")
        mismatch = out_model.get("mismatch_rel")
        valid_model_np = out_model.get("valid")
        tcell_np = out_model.get("tcell_c")
        if pac_expected is None:
            raise DashboardServiceError("power_model não retornou pac_expected_w.", status_code=500)

        pac_model_w = [None if (not np.isfinite(v)) else float(v) for v in np.asarray(pac_expected, dtype=float).tolist()]
        pdc_model_w = [None] * len(times_utc) if pdc_expected is None else [None if (not np.isfinite(v)) else float(v) for v in np.asarray(pdc_expected, dtype=float).tolist()]
        v_dc_model_v = [None] * len(times_utc) if vdc_expected is None else [None if (not np.isfinite(v)) else float(v) for v in np.asarray(vdc_expected, dtype=float).tolist()]
        i_dc_model_a = [None] * len(times_utc) if idc_expected is None else [None if (not np.isfinite(v)) else float(v) for v in np.asarray(idc_expected, dtype=float).tolist()]
        v_ratio = [None] * len(times_utc) if v_ratio_np is None else [None if (not np.isfinite(v)) else float(v) for v in np.asarray(v_ratio_np, dtype=float).tolist()]
        i_ratio = [None] * len(times_utc) if i_ratio_np is None else [None if (not np.isfinite(v)) else float(v) for v in np.asarray(i_ratio_np, dtype=float).tolist()]
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

        mppt_model_by_source: Dict[str, Dict[str, Any]] = {}
        try:
            mppt_expected = expected_dc_by_mppt_from_details(
                details=details,
                module=mod,
                plant=pl,
                g_poa=g_poa_used_np,
                tamb_c=tamb_np,
                g_min_valid=0.0,
                n_points=60,
            )
        except Exception:
            logger.exception("Falha ao calcular modelo DC por MPPT plant_id=%s", plant.id)
            mppt_expected = {}

        sbs = agg.get("series_by_source") or {}
        has_any_mppt_config = bool(mppt_expected)
        for src in sbs.keys():
            mppt_no = _mppt_no_from_source(src)
            if mppt_no is None:
                continue
            block = mppt_expected.get(mppt_no)
            if block is not None:
                mppt_model_by_source[src] = {
                    "v_dc_model_v": _float_list(block.get("v_dc_expected_v"), np_mod=np),
                    "i_dc_model_a": _float_list(block.get("i_dc_expected_a"), np_mod=np),
                    "p_dc_model_w": _float_list(block.get("pdc_expected_w"), np_mod=np),
                    "topology_ok": [True] * len(times_utc),
                    "model_note": [None] * len(times_utc),
                }
            else:
                note = "Sem strings configuradas para este MPPT." if has_any_mppt_config else "Cadastre PVPlantStringConfig com campo MPPT preenchido para habilitar o modelo DC por MPPT."
                mppt_model_by_source[src] = {
                    "v_dc_model_v": [None] * len(times_utc),
                    "i_dc_model_a": [None] * len(times_utc),
                    "p_dc_model_w": [None] * len(times_utc),
                    "topology_ok": [False] * len(times_utc),
                    "model_note": [note] * len(times_utc),
                }

        _fill_aggregate_dc_from_mppt_model(
            agg=agg,
            mppt_model_by_source=mppt_model_by_source,
            v_dc_model_v=v_dc_model_v,
            i_dc_model_a=i_dc_model_a,
        )

        return {
            "g_poa_used": g_poa_used,
            "pac_model_w": pac_model_w,
            "pdc_model_w": pdc_model_w,
            "v_dc_model_v": v_dc_model_v,
            "i_dc_model_a": i_dc_model_a,
            "v_ratio": v_ratio,
            "i_ratio": i_ratio,
            "mismatch_rel": mismatch_rel,
            "valid_model": valid_model,
            "tcell_c": tcell_c,
            "mppt_model_by_source": mppt_model_by_source,
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


def build_base_gate_debug(params: MismatchDashboardParams, model: Dict[str, Any], agg: Dict[str, Any]) -> Dict[str, List[Any]]:
    gate_valid_model: List[bool] = []
    gate_gpoa_ok: List[bool] = []
    gate_pac_ok: List[bool] = []
    gate_meteo_ok: List[bool] = []
    gate_inverter_ok: List[bool] = []
    gate_reason: List[str] = []

    for i in range(len(model["mismatch_rel"])):
        gp = model["g_poa_used"][i]
        pr = agg["p_ac_w"][i]

        valid_model_ok = bool(model["valid_model"][i])
        gpoa_ok = (gp is not None) and (float(gp) >= float(params.gpoa_gate))
        pac_ok = (pr is not None) and (float(pr) >= float(params.pmin_w))
        meteo_ok = not bool(agg["flag_meteo_missing"][i])
        inverter_ok = not bool(agg["flag_inv_missing_all"][i])

        gate_valid_model.append(valid_model_ok)
        gate_gpoa_ok.append(bool(gpoa_ok))
        gate_pac_ok.append(bool(pac_ok))
        gate_meteo_ok.append(bool(meteo_ok))
        gate_inverter_ok.append(bool(inverter_ok))

        reasons: List[str] = []
        if not valid_model_ok:
            reasons.append("invalid_model")
        if not gpoa_ok:
            reasons.append("low_gpoa")
        if not pac_ok:
            reasons.append("low_pac")
        if not meteo_ok:
            reasons.append("meteo_missing")
        if not inverter_ok:
            reasons.append("inv_missing")
        gate_reason.append("ok" if not reasons else ";".join(reasons))

    return {
        "gate_valid_model": gate_valid_model,
        "gate_gpoa_ok": gate_gpoa_ok,
        "gate_pac_ok": gate_pac_ok,
        "gate_meteo_ok": gate_meteo_ok,
        "gate_inverter_ok": gate_inverter_ok,
        "gate_reason": gate_reason,
    }


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


def _safe_abs(v: Optional[float]) -> Optional[float]:
    try:
        if v is None:
            return None
        return abs(float(v))
    except Exception:
        return None


def _residual_score_from_channels(i: int, residual_series: Optional[Dict[str, Any]]) -> Tuple[Optional[float], bool]:
    if not residual_series:
        return None, False
    conf = residual_series.get("channel_confidence") or {}
    chans = [
        (residual_series.get("p_dc_residual_rel") or [], (conf.get("p_dc") or []), 0.40),
        (residual_series.get("v_dc_residual_rel") or [], (conf.get("v_dc") or []), 0.25),
        (residual_series.get("i_dc_residual_rel") or [], (conf.get("i_dc") or []), 0.35),
    ]
    num = 0.0
    den = 0.0
    for arr, carr, weight in chans:
        try:
            rv = arr[i]
        except Exception:
            rv = None
        try:
            cv = carr[i]
        except Exception:
            cv = None
        if rv is None or cv is None:
            continue
        try:
            av = min(abs(float(rv)), 2.0) / 2.0
            cf = max(0.0, min(1.0, float(cv)))
        except Exception:
            continue
        num += weight * av * cf
        den += weight
    if den <= 0:
        return None, False
    score = max(0.0, min(1.0, num / den))
    return score, bool(score >= 0.45)




def _combined_detection_signal(base_signal: List[Optional[float]], residual_series: Optional[Dict[str, Any]], base_gate: List[bool]) -> List[Optional[float]]:
    """Build the canonical runtime detection signal from available residual channels.

    Priority is given to the existing plant-level AC power mismatch so the dashboard
    remains backward compatible, but DC residual channels participate whenever they
    are available and sufficiently reliable.
    """
    if not residual_series:
        return list(base_signal)

    conf = residual_series.get("channel_confidence") or {}
    pdc = residual_series.get("p_dc_residual_rel") or []
    vdc = residual_series.get("v_dc_residual_rel") or []
    idc = residual_series.get("i_dc_residual_rel") or []
    c_pdc = conf.get("p_dc") or []
    c_vdc = conf.get("v_dc") or []
    c_idc = conf.get("i_dc") or []

    out: List[Optional[float]] = [None] * len(base_signal)
    for i in range(len(base_signal)):
        if i < len(base_gate) and not bool(base_gate[i]):
            out[i] = None
            continue

        num = 0.0
        den = 0.0

        # Preserve the existing AC-power mismatch contribution for continuity.
        try:
            bv = base_signal[i]
            if bv is not None:
                num += 0.40 * float(bv)
                den += 0.40
        except Exception:
            pass

        for arr, carr, weight in ((pdc, c_pdc, 0.25), (vdc, c_vdc, 0.15), (idc, c_idc, 0.20)):
            try:
                rv = arr[i]
            except Exception:
                rv = None
            try:
                cv = carr[i]
            except Exception:
                cv = None
            if rv is None:
                continue
            try:
                cf = 1.0 if cv is None else max(0.0, min(1.0, float(cv)))
                num += weight * float(rv) * cf
                den += weight * cf
            except Exception:
                continue

        out[i] = (num / den) if den > 0 else (base_signal[i] if i < len(base_signal) else None)
    return out

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
    residual_series: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = len(times_utc)
    base_gate = build_base_gate(params, model, agg)
    gate_debug = build_base_gate_debug(params, model, agg)
    detection_signal: List[Optional[float]] = list(model["mismatch_rel"])

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
    anomaly_power: List[bool] = [False] * n
    residual_trigger: List[bool] = [False] * n
    residual_event_score: List[Optional[float]] = [None] * n
    combined_event_score: List[Optional[float]] = [None] * n

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

        residual_channel_map = {
            "p_ac": list((residual_series or {}).get("p_ac_residual_rel") or model["mismatch_rel"]),
            "p_dc": list((residual_series or {}).get("p_dc_residual_rel") or [None] * n),
            "v_dc": list((residual_series or {}).get("v_dc_residual_rel") or [None] * n),
            "i_dc": list((residual_series or {}).get("i_dc_residual_rel") or [None] * n),
        }

        det = detect_anomalies(
            mismatch_rel=model["mismatch_rel"],
            g_poa_wm2=model["g_poa_used"],
            valid_model=base_gate,
            flag_meteo_missing=agg["flag_meteo_missing"],
            flag_meteo_low_confidence=agg["flag_meteo_low_confidence"],
            flag_meteo_interpolated=agg["flag_meteo_interpolated"],
            flag_inv_missing=agg["flag_inv_missing_all"],
            inv_coverage=agg["inv_cov"],
            residual_channels=residual_channel_map,
            residual_channel_confidence=(residual_series or {}).get("channel_confidence"),
            params=det_params,
        ) or {}

        detection_signal = list(det.get("detection_signal_rel") or model["mismatch_rel"])

        valid_period = [bool(v) for v in (det.get("valid_period") or base_gate)]
        anomaly_power = [bool(v) for v in (det.get("anomaly") or [False] * n)]
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

        for i in range(n):
            score_i, trig_i = _residual_score_from_channels(i, residual_series)
            residual_event_score[i] = score_i
            residual_trigger[i] = bool(base_gate[i] and trig_i)
            power_score = 0.0
            mm = _safe_abs(detection_signal[i])
            if mm is not None:
                power_score = min(mm, 2.0) / 2.0
            if residual_event_score[i] is None:
                combined_event_score[i] = power_score if base_gate[i] else None
            else:
                combined_event_score[i] = (0.60 * power_score + 0.40 * float(residual_event_score[i])) if base_gate[i] else None

        anomaly = [bool(base_gate[i] and (anomaly_power[i] or residual_trigger[i])) for i in range(n)]

        det_dbg = {
            "z": det.get("z"),
            "base_gate": base_gate,
            "gate_valid_model": gate_debug["gate_valid_model"],
            "gate_gpoa_ok": gate_debug["gate_gpoa_ok"],
            "gate_pac_ok": gate_debug["gate_pac_ok"],
            "gate_meteo_ok": gate_debug["gate_meteo_ok"],
            "gate_inverter_ok": gate_debug["gate_inverter_ok"],
            "gate_reason": gate_debug["gate_reason"],
            "ewma_z": ewma_z,
            "cusum": cusum_score,
            "baseline": det.get("baseline"),
            "detection_signal_rel": detection_signal,
            "coarse_period": coarse_period,
            "fine_period": fine_period,
            "meteo_quality_ok": meteo_quality_ok,
            "meteo_qc_score": agg["meteo_qc_score"],
            "flag_meteo_low_confidence": agg["flag_meteo_low_confidence"],
            "flag_meteo_interpolated": agg["flag_meteo_interpolated"],
            "flag_meteo_outlier": agg["flag_meteo_outlier"],
            "flag_meteo_artifact": agg["flag_meteo_artifact"],
            "irradiance_tier": irradiance_tier,
            "anomaly_power": anomaly_power,
            "residual_trigger": residual_trigger,
            "residual_event_score": residual_event_score,
            "combined_event_score": combined_event_score,
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

        alarm_code, alarm_sev = build_alarm_vectors(times_utc, per_ts, selected_sources)
        rca = diagnose_rca_series(
            anomaly=anomaly,
            valid_period=valid_period,
            mismatch_rel=model["mismatch_rel"],
            g_poa_wm2=model["g_poa_used"],
            coarse_period=coarse_period,
            fine_period=fine_period,
            meteo_quality_ok=meteo_quality_ok,
            irradiance_tier=irradiance_tier,
            v_dc_v=agg["v_dc_v"],
            i_dc_a=agg["i_dc_a"],
            v_ac_v=agg.get("v_ac_v"),
            i_ac_a=agg.get("i_ac_a"),
            freq_hz=freq_hz,
            alarm_code=alarm_code,
            alarm_sev=alarm_sev,
            pac_real_w=agg["p_ac_w"],
            pac_model_w=model["pac_model_w"],
            flag_inv_missing=agg["flag_inv_missing_all"],
            flag_meteo_missing=agg["flag_meteo_missing"],
            inv_coverage=agg["inv_cov"],
            pac_cap_w=pac_cap_w,
            p_ac_residual_rel=(residual_series or {}).get("p_ac_residual_rel"),
            p_dc_residual_rel=(residual_series or {}).get("p_dc_residual_rel"),
            v_dc_residual_rel=(residual_series or {}).get("v_dc_residual_rel"),
            i_dc_residual_rel=(residual_series or {}).get("i_dc_residual_rel"),
            residual_channel_confidence=(residual_series or {}).get("channel_confidence"),
            params=rca_params,
        ) or {}
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

        rca_dbg = {
            "baseline": rca.get("baseline"),
            "alarm_code": alarm_code,
            "alarm_sev": alarm_sev,
            "residual_channel_confidence": (residual_series or {}).get("channel_confidence"),
        }
        pipeline_name = "ewma_cusum_multichannel_residuals + rca_patterns"
        return {
            "pipeline_name": pipeline_name,
            "base_gate": base_gate,
            "gate_valid_model": gate_debug["gate_valid_model"],
            "gate_gpoa_ok": gate_debug["gate_gpoa_ok"],
            "gate_pac_ok": gate_debug["gate_pac_ok"],
            "gate_meteo_ok": gate_debug["gate_meteo_ok"],
            "gate_inverter_ok": gate_debug["gate_inverter_ok"],
            "gate_reason": gate_debug["gate_reason"],
            "detection_signal_rel": detection_signal,
            "valid_period": valid_period,
            "anomaly": anomaly,
            "anomaly_power": anomaly_power,
            "residual_trigger": residual_trigger,
            "residual_event_score": residual_event_score,
            "combined_event_score": combined_event_score,
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
        "base_gate": base_gate,
        "gate_valid_model": gate_debug["gate_valid_model"],
        "gate_gpoa_ok": gate_debug["gate_gpoa_ok"],
        "gate_pac_ok": gate_debug["gate_pac_ok"],
        "gate_meteo_ok": gate_debug["gate_meteo_ok"],
        "gate_inverter_ok": gate_debug["gate_inverter_ok"],
        "gate_reason": gate_debug["gate_reason"],
        "detection_signal_rel": detection_signal,
        "valid_period": valid_period,
        "anomaly": anomaly,
        "anomaly_power": anomaly_power,
        "residual_trigger": residual_trigger,
        "residual_event_score": residual_event_score,
        "combined_event_score": combined_event_score,
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
