from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Mapping, Optional, Tuple
import re
from zoneinfo import ZoneInfo

from core.models import PVPlant, PVPlantStringConfig
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
from core.services.fdd.runtime_types import MismatchDashboardParams
from core.services.fdd_mismatch import MismatchThresholds
from core.services.residuals.facade import compute_residual_series_from_observations


def _pick_first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _series_with_fallback(*series_list: List[Any]) -> List[Any]:
    if not series_list:
        return []
    n = max((len(s) for s in series_list if isinstance(s, list)), default=0)
    out: List[Any] = [None] * n
    for i in range(n):
        vals = []
        for s in series_list:
            try:
                vals.append(s[i])
            except Exception:
                vals.append(None)
        out[i] = _pick_first_not_none(*vals)
    return out




def _mppt_no_from_source(src: Any) -> Optional[int]:
    m = re.search(r"(?:\||\b)MPPT\s*([0-9]+)", str(src or ""), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _float_list(xs: Any, n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * n
    if not isinstance(xs, (list, tuple)):
        return out
    for i in range(min(n, len(xs))):
        try:
            v = xs[i]
            out[i] = None if v is None else float(v)
        except Exception:
            out[i] = None
    return out


def _build_mppt_model_by_source(plant: PVPlant, details: Any, times_utc: List[datetime], agg: Dict[str, Any], selected_sources: List[str], g_poa_used: List[Any]) -> Dict[str, Dict[str, List[Any]]]:
    n = len(times_utc)
    try:
        import numpy as np
        from core.services.power_model.power_model import (
            module_from_pvmodule,
            plant_from_details,
            StringGroup,
            tcell_noct,
            iph_irr_temp,
            i0_temp,
            rp_irr,
            _vt_cell,
            voc_guess,
            pmp_array_groups_vec,
        )
    except Exception:
        return {}

    cfgs = list(PVPlantStringConfig.objects.filter(details_id=getattr(details, "id", None)).order_by("mppt", "order", "id"))
    groups_by_mppt: Dict[int, List[Any]] = defaultdict(list)
    for c in cfgs:
        try:
            mppt = int(getattr(c, "mppt", None))
            sq = int(getattr(c, "strings_qty", None))
            ns = int(getattr(c, "modules_per_string", None))
        except Exception:
            continue
        if mppt < 1 or sq < 1 or ns < 1:
            continue
        groups_by_mppt[mppt].append(StringGroup(strings_qty=sq, modules_per_string=ns))

    out: Dict[str, Dict[str, List[Any]]] = {}
    if not groups_by_mppt:
        for src in selected_sources:
            if _mppt_no_from_source(src) is None:
                continue
            out[src] = {
                "v_dc_model_v": [None] * n,
                "i_dc_model_a": [None] * n,
                "p_dc_model_w": [None] * n,
                "topology_ok": [False] * n,
                "model_note": ["Cadastre PVPlantStringConfig com MPPT para habilitar o modelo DC por MPPT."] * n,
            }
        return out

    try:
        mod = module_from_pvmodule(details.module)
        inv = getattr(details, "inverter", None)
        pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

        G0 = np.asarray([np.nan if v is None else float(v) for v in g_poa_used], dtype=float)
        Tair = np.asarray([np.nan if v is None else float(v) for v in (agg.get("temp_air") or [None] * n)], dtype=float)
        valid = np.isfinite(G0) & np.isfinite(Tair) & (G0 >= 0.0)
        Tc = tcell_noct(G0, Tair, noct_c=float(getattr(pl, "noct_c", 45.0) or 45.0))
        iph = iph_irr_temp(mod, G0, Tc)
        i0 = i0_temp(mod, Tc)
        rp = rp_irr(mod, G0)
        Tk = Tc + 273.15
        aVt = float(mod.a) * (_vt_cell(Tk) * float(mod.ns))
        voc_g = voc_guess(mod, Tc, G0)

        for src in selected_sources:
            mppt_no = _mppt_no_from_source(src)
            if mppt_no is None:
                continue
            groups = groups_by_mppt.get(mppt_no)
            if not groups:
                out[src] = {
                    "v_dc_model_v": [None] * n,
                    "i_dc_model_a": [None] * n,
                    "p_dc_model_w": [None] * n,
                    "topology_ok": [False] * n,
                    "model_note": ["Sem strings configuradas para este MPPT."] * n,
                }
                continue

            grp = pmp_array_groups_vec(
                iph=iph,
                i0=i0,
                rs=float(mod.rs_ohm),
                rp=rp,
                aVt=aVt,
                voc_g=voc_g,
                groups=groups,
                n_points=60,
            )
            pdc_raw = np.asarray(grp.get("pmp"), dtype=float)
            vdc_exp = np.asarray(grp.get("vmp"), dtype=float)
            idc_exp = np.asarray(grp.get("imp"), dtype=float)
            pdc_exp = np.where(valid, pdc_raw * float(getattr(pl, "k_sys", 1.0) or 1.0), np.nan)
            vdc_exp = np.where(valid, vdc_exp, np.nan)
            idc_exp = np.where(valid, idc_exp, np.nan)

            out[src] = {
                "v_dc_model_v": [None if not np.isfinite(v) else float(v) for v in vdc_exp.tolist()],
                "i_dc_model_a": [None if not np.isfinite(v) else float(v) for v in idc_exp.tolist()],
                "p_dc_model_w": [None if not np.isfinite(v) else float(v) for v in pdc_exp.tolist()],
                "topology_ok": [True] * n,
                "model_note": [None] * n,
            }
    except Exception as exc:
        note = f"Falha no modelo DC por MPPT: {type(exc).__name__}: {exc}"
        for src in selected_sources:
            if _mppt_no_from_source(src) is None:
                continue
            out[src] = {
                "v_dc_model_v": [None] * n,
                "i_dc_model_a": [None] * n,
                "p_dc_model_w": [None] * n,
                "topology_ok": [False] * n,
                "model_note": [note] * n,
            }
    return out


def _fill_aggregate_dc_from_mppt(agg: Dict[str, Any], series_by_source: Dict[str, Dict[str, List[Any]]], model: Dict[str, Any]) -> None:
    n = len(model.get("v_dc_model_v") or [])
    for i in range(n):
        vv: List[float] = []
        ii: List[float] = []
        for src, sb in (series_by_source or {}).items():
            if _mppt_no_from_source(src) is None:
                continue
            try:
                mv = sb.get("v_dc_model_v")[i]
                mi = sb.get("i_dc_model_a")[i]
                meas_p = (sb.get("p_dc_w") or [None] * n)[i]
                meas_i = (sb.get("i_dc_a") or [None] * n)[i]
            except Exception:
                continue
            is_active = ((meas_p is not None and float(meas_p) > 1.0) or (meas_i is not None and float(meas_i) > 0.25))
            if not is_active:
                continue
            if mv is not None:
                vv.append(float(mv))
            if mi is not None:
                ii.append(float(mi))
        if model.get("v_dc_model_v") is not None and i < len(model["v_dc_model_v"]) and model["v_dc_model_v"][i] is None and vv:
            model["v_dc_model_v"][i] = float(sum(vv) / len(vv))
        if model.get("i_dc_model_a") is not None and i < len(model["i_dc_model_a"]) and model["i_dc_model_a"][i] is None and ii:
            model["i_dc_model_a"][i] = float(sum(ii))

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

    display_mode = str(data.get("display_mode") or "mismatch").strip().lower()
    if display_mode not in {"mismatch", "tipologia"}:
        display_mode = "mismatch"

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
        display_mode=display_mode,
    )


def _build_typology_classes(times_utc: List[datetime], confidence: Dict[str, Any], pipeline: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, int]]:
    sev_runtime: List[str] = []
    sev_reason: List[str] = []
    sev_counts = {"none": 0, "ok": 0, "warn": 0, "crit": 0}
    for i in range(len(times_utc)):
        sev = runtime_severity(
            state_label=confidence["diag_state_labels"][i],
            diagnosis_label=confidence["diag_diagnosis_labels"][i],
            direct_grid_evidence=bool(confidence["diag_direct_grid"][i]),
            anomaly_flag=bool(pipeline["anomaly"][i]),
        )
        reason = str(confidence["diag_diagnosis_labels"][i] or confidence["diag_state_labels"][i] or sev)
        sev_runtime.append(sev)
        sev_reason.append(reason)
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    return sev_runtime, sev_reason, sev_counts


def _build_mismatch_classes(times_utc: List[datetime], model: Dict[str, Any], pipeline: Dict[str, Any], params: MismatchDashboardParams) -> Tuple[List[str], List[str], Dict[str, int]]:
    classes: List[str] = []
    reasons: List[str] = []
    counts = {"none": 0, "ok": 0, "warn": 0, "crit": 0}
    for i in range(len(times_utc)):
        if not bool(pipeline["valid_period"][i]):
            cls = "none"
            reason = "invalid_or_insufficient_data"
        else:
            mm = model["mismatch_rel"][i]
            abs_mm = None
            try:
                abs_mm = abs(float(mm)) if mm is not None else None
            except Exception:
                abs_mm = None
            ces = pipeline.get("combined_event_score") or []
            try:
                combined = ces[i]
            except Exception:
                combined = None
            if mm is not None and float(mm) <= -float(params.thr.fault_abs):
                cls = "crit"
                reason = "mismatch<=-fault_abs"
            elif combined is not None and float(combined) >= 0.90:
                cls = "crit"
                reason = "combined_event_score>=0.90"
            elif abs_mm is not None and abs_mm >= float(params.thr.warn_abs):
                cls = "warn"
                reason = "|mismatch|>=warn_abs"
            elif combined is not None and float(combined) >= 0.45:
                cls = "warn"
                reason = "combined_event_score>=0.45"
            elif bool((pipeline.get("anomaly_power") or [False] * len(times_utc))[i]):
                cls = "warn"
                reason = "ewma_cusum_power"
            elif bool((pipeline.get("residual_trigger") or [False] * len(times_utc))[i]):
                cls = "warn"
                reason = "residual_multichannel"
            else:
                cls = "ok"
                reason = "within_runtime_thresholds"
        classes.append(cls)
        reasons.append(reason)
        counts[cls] = counts.get(cls, 0) + 1
    return classes, reasons, counts


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
    p_ac_real_series = _series_with_fallback(agg.get("p_ac_w") or [], agg.get("p_ac_mppt_sum_w") or [], agg.get("p_ac_agg_w") or [])
    residual_kwargs = dict(
        plant=plant,
        times_utc=times_utc,
        gti=agg["gti"],
        ghi=agg["ghi"],
        dni=agg["dni"],
        dhi=agg["dhi"],
        temp_air=agg["temp_air"],
        p_ac_w=p_ac_real_series,
        p_dc_w=agg["p_dc_w"],
        v_dc_v=agg["v_dc_v"],
        i_dc_a=agg["i_dc_a"],
        v_ac_v=agg["v_ac_v"],
        i_ac_a=agg["i_ac_a"],
        freq_hz=agg["freq_hz"],
        meteo_qc_score=agg["meteo_qc_score"],
        flag_meteo_missing=agg["flag_meteo_missing"],
        flag_meteo_low_confidence=agg["flag_meteo_low_confidence"],
        flag_meteo_interpolated=agg["flag_meteo_interpolated"],
        flag_meteo_outlier=agg["flag_meteo_outlier"],
        flag_meteo_artifact=agg["flag_meteo_artifact"],
        flag_inv_missing=agg["flag_inv_missing_all"],
        inv_coverage=agg["inv_cov"],
        source_oper=(source_oper_list[0] if source_oper_list else ""),
        source_meteo=src_meteo,
    )
    try:
        import inspect as _inspect
        if "g_poa_wm2" in _inspect.signature(compute_residual_series_from_observations).parameters:
            residual_kwargs["g_poa_wm2"] = model["g_poa_used"]
    except Exception:
        pass
    residual_out = compute_residual_series_from_observations(**residual_kwargs)
    residual_series = residual_out.get("series") or {}

    mppt_model_by_source = _build_mppt_model_by_source(
        plant=plant,
        details=details,
        times_utc=times_utc,
        agg=agg,
        selected_sources=selected_sources,
        g_poa_used=model.get("g_poa_used") or [None] * len(times_utc),
    )
    for src, sb in (agg.get("series_by_source") or {}).items():
        block = mppt_model_by_source.get(src) or {}
        sb["v_dc_model_v"] = list(block.get("v_dc_model_v") or ([None] * len(times_utc)))
        sb["i_dc_model_a"] = list(block.get("i_dc_model_a") or ([None] * len(times_utc)))
        sb["p_dc_model_w"] = list(block.get("p_dc_model_w") or ([None] * len(times_utc)))
        sb["topology_ok"] = list(block.get("topology_ok") or ([False] * len(times_utc)))
        sb["model_note"] = list(block.get("model_note") or ([None] * len(times_utc)))

    # Sincroniza expectativas do módulo canônico com o payload principal.
    if residual_series:
        model["g_poa_used"] = residual_series.get("g_poa_used") or model["g_poa_used"]
        model["tcell_c"] = residual_series.get("tcell_c") or model["tcell_c"]
        model["pac_model_w"] = residual_series.get("pac_expected_w") or model["pac_model_w"]
        model["mismatch_rel"] = residual_series.get("p_ac_residual_rel") or model["mismatch_rel"]
        model["pdc_model_w"] = residual_series.get("pdc_expected_w") or model.get("pdc_model_w") or [None] * len(times_utc)
        model["v_dc_model_v"] = residual_series.get("v_dc_expected_v") or model.get("v_dc_model_v") or [None] * len(times_utc)
        model["i_dc_model_a"] = residual_series.get("i_dc_expected_a") or model.get("i_dc_model_a") or [None] * len(times_utc)

    _fill_aggregate_dc_from_mppt(agg, agg.get("series_by_source") or {}, model)

    pipeline = run_detection_and_rca(
        plant=plant,
        details=details,
        params=params,
        times_utc=times_utc,
        per_ts=per_ts,
        selected_sources=selected_sources,
        agg=agg,
        model=model,
        residual_series=residual_series,
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
        model=model,
        pipeline=pipeline,
        confidence=confidence,
        residual_series=residual_series,
    )

    sev_typology, reason_typology, counts_typology = _build_typology_classes(times_utc, confidence, pipeline)
    sev_mismatch, reason_mismatch, counts_mismatch = _build_mismatch_classes(times_utc, model, pipeline, params)

    freq_series = _series_with_fallback(pipeline.get("freq_hz") or [], agg.get("freq_hz") or [])

    if params.display_mode == "tipologia":
        sev_selected = sev_typology
        reason_selected = reason_typology
        counts_selected = counts_typology
        heatmap_label = "Tipologia de falha"
        heatmap_note = "Classes do heatmap derivadas do diagnóstico/RCa explicável do pipeline."
    else:
        sev_selected = sev_mismatch
        reason_selected = reason_mismatch
        counts_selected = counts_mismatch
        heatmap_label = "Mismatch"
        heatmap_note = "Classes do heatmap derivadas da severidade do mismatch/potência no backend."

    return {
        "ok": True,
        "pipeline": pipeline["pipeline_name"],
        "display_mode": params.display_mode,
        "heatmap_mode": {
            "selected": params.display_mode,
            "selected_label": heatmap_label,
            "selected_note": heatmap_note,
        },
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
            "residual_global_confidence_mean": mean_none(residual_series.get("global_confidence") or []),
        },
        "sources": {
            "source_meteo": src_meteo,
            "source_oper_list": source_oper_list,
            "selected_sources": selected_sources,
            "total_policy": "prefer_mppt_sum",
        },
        "thresholds": {
            "warn_abs": float(params.thr.warn_abs),
            "fault_abs": float(params.thr.fault_abs),
            "gpoa_gate": float(params.gpoa_gate),
            "pmin_w": float(params.pmin_w),
            "gpoa_plot_min": float(params.gpoa_plot_min),
            "pmodel_plot_min": float(params.pmodel_plot_min),
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
            "p_ac_w": p_ac_real_series,
            "p_ac_real_w": p_ac_real_series,
            "p_dc_w": agg["p_dc_w"],
            "e_ac_wh_15": agg["e_ac_wh_15"],
            "v_dc_v": agg["v_dc_v"],
            "i_dc_a": agg["i_dc_a"],
            "v_ac_v": agg["v_ac_v"],
            "i_ac_a": agg["i_ac_a"],
            "freq_hz": freq_series,
            "inv_coverage": agg["inv_cov"],
            "flag_inv_missing": agg["flag_inv_missing_all"],
            "flag_inv_missing_all": agg["flag_inv_missing_all"],
            "flag_inv_missing_partial": agg["flag_inv_missing_partial"],
            "p_ac_mppt_sum_w": agg["p_ac_mppt_sum_w"],
            "p_ac_agg_w": agg["p_ac_agg_w"],
            "policy_used": agg["policy_used"],
            "p_ac_model_w": model["pac_model_w"],
            "p_dc_model_w": model.get("pdc_model_w") or [None] * len(times_utc),
            "v_dc_model_v": model.get("v_dc_model_v") or [None] * len(times_utc),
            "i_dc_model_a": model.get("i_dc_model_a") or [None] * len(times_utc),
            "tcell_c": model["tcell_c"],
            "mismatch_rel": plot_data["mismatch_rel_plot"],
            "mismatch_rel_raw": plot_data["mismatch_rel_raw"],
            "p_ac_residual_abs": residual_series.get("p_ac_residual_abs") or [None] * len(times_utc),
            "p_ac_residual_rel": residual_series.get("p_ac_residual_rel") or [None] * len(times_utc),
            "p_dc_residual_abs": residual_series.get("p_dc_residual_abs") or [None] * len(times_utc),
            "p_dc_residual_rel": residual_series.get("p_dc_residual_rel") or [None] * len(times_utc),
            "v_dc_residual_abs": residual_series.get("v_dc_residual_abs") or [None] * len(times_utc),
            "v_dc_residual_rel": residual_series.get("v_dc_residual_rel") or [None] * len(times_utc),
            "i_dc_residual_abs": residual_series.get("i_dc_residual_abs") or [None] * len(times_utc),
            "i_dc_residual_rel": residual_series.get("i_dc_residual_rel") or [None] * len(times_utc),
            "residual_global_confidence": residual_series.get("global_confidence") or [None] * len(times_utc),
            "ewma_z": pipeline.get("ewma_z") or [None] * len(times_utc),
            "cusum_score": pipeline.get("cusum_score") or [None] * len(times_utc),
            "alarm_code": pipeline.get("alarm_code") or [None] * len(times_utc),
            "alarm_sev": pipeline.get("alarm_sev") or [None] * len(times_utc),
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
            "anomaly_power": pipeline.get("anomaly_power") or [False] * len(times_utc),
            "residual_trigger": pipeline.get("residual_trigger") or [False] * len(times_utc),
            "residual_event_score": pipeline.get("residual_event_score") or [None] * len(times_utc),
            "combined_event_score": pipeline.get("combined_event_score") or [None] * len(times_utc),
            "rca_code": pipeline["codes"],
            "rca_label": pipeline["labels"],
            "codes": pipeline["codes"],
            "labels": pipeline["labels"],
            "sev_runtime": sev_selected,
            "hm_class": sev_selected,
            "hm_reason": reason_selected,
            "hm_class_selected": sev_selected,
            "hm_reason_selected": reason_selected,
            "heatmap_class": sev_selected,
            "heatmap_reason": reason_selected,
            "hm_class_typology": sev_typology,
            "hm_reason_typology": reason_typology,
            "hm_class_mismatch": sev_mismatch,
            "hm_reason_mismatch": reason_mismatch,
            "cls": sev_selected,
            "severity": sev_selected,
            "hm_day_local": hm_day_local,
            "hm_minute_local": hm_minute_local,
        },
        "series_by_source": agg["series_by_source"],
        "summary": {
            "counts": counts_selected,
            "counts_by_mode": {
                "tipologia": counts_typology,
                "mismatch": counts_mismatch,
            },
            "events": [],
            "n_points": len(times_utc),
            "n_oper_sources": len(selected_sources),
            "persist": persist,
        },
    }
