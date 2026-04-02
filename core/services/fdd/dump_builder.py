from __future__ import annotations

from datetime import datetime, timezone as dt_tz
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.services.fdd.dashboard_common import is_agg_source


def _at(seq: Any, i: int, default: Any = None) -> Any:
    try:
        return seq[i]
    except Exception:
        return default


def _pick_first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def build_runtime_dump(
    *,
    tz: ZoneInfo,
    src_meteo: str,
    selected_sources: List[str],
    times_utc: List[datetime],
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]],
    agg: Dict[str, Any],
    model: Dict[str, Any],
    pipeline: Dict[str, Any],
    confidence: Dict[str, Any],
    residual_series: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dump_by_tkey: Dict[str, Any] = {}
    residual_series = residual_series or {}
    ch_conf = residual_series.get("channel_confidence") or {}

    for i, ts_utc in enumerate(times_utc):
        tloc = ts_utc.astimezone(tz)
        tkey = tloc.strftime("%Y-%m-%dT%H:%M")
        by_src = per_ts.get(ts_utc, {})
        meteo_dump: Dict[str, Any] = {}
        src_dump: Dict[str, Any] = {}

        any_row = None
        for sname in selected_sources:
            if not is_agg_source(sname):
                continue
            rr = by_src.get(sname)
            if rr is not None:
                any_row = rr
                break
        if any_row is None:
            for sname in selected_sources:
                rr = by_src.get(sname)
                if rr is not None:
                    any_row = rr
                    break
        if any_row is not None:
            meteo_dump["g_poa_used"] = _at(model.get("g_poa_used"), i)
            meteo_dump["gti"] = any_row.get("gti")
            meteo_dump["ghi"] = any_row.get("ghi")
            meteo_dump["dni"] = any_row.get("dni")
            meteo_dump["dhi"] = any_row.get("dhi")
            meteo_dump["temp_air"] = any_row.get("temp_air")
            meteo_dump["wind_speed"] = any_row.get("wind_speed")
            meteo_dump["rh"] = any_row.get("rh")
            meteo_dump["flag_meteo_missing"] = any_row.get("flag_meteo_missing")
            meteo_dump["flag_meteo_low_confidence"] = any_row.get("flag_meteo_low_confidence")
            meteo_dump["flag_meteo_interpolated"] = any_row.get("flag_meteo_interpolated")
            meteo_dump["flag_meteo_outlier"] = any_row.get("flag_meteo_outlier")
            meteo_dump["flag_meteo_artifact"] = any_row.get("flag_meteo_artifact")
            meteo_dump["meteo_qc_score"] = any_row.get("meteo_qc_score")

        for sname in selected_sources:
            rr = by_src.get(sname)
            if rr is None:
                continue
            sb = (agg.get("series_by_source") or {}).get(sname) or {}
            src_dump[sname] = {
                "p_ac_w": rr.get("p_ac_w"),
                "p_dc_w": rr.get("p_dc_w"),
                "e_ac_wh_15": rr.get("e_ac_wh_15"),
                "v_dc_v": rr.get("v_dc_v"),
                "i_dc_a": rr.get("i_dc_a"),
                "v_ac_v": rr.get("v_ac_v"),
                "i_ac_a": rr.get("i_ac_a"),
                "freq_hz": rr.get("freq_hz"),
                "v_dc_model_v": _at(sb.get("v_dc_model_v"), i),
                "i_dc_model_a": _at(sb.get("i_dc_model_a"), i),
                "p_dc_model_w": _at(sb.get("p_dc_model_w"), i),
                "topology_ok": _at(sb.get("topology_ok"), i),
                "model_note": _at(sb.get("model_note"), i),
                "mppt1_vdc_v": rr.get("mppt1_vdc_v"),
                "mppt2_vdc_v": rr.get("mppt2_vdc_v"),
                "mppt3_vdc_v": rr.get("mppt3_vdc_v"),
                "mppt4_vdc_v": rr.get("mppt4_vdc_v"),
                "mppt1_idc_a": rr.get("mppt1_idc_a"),
                "mppt2_idc_a": rr.get("mppt2_idc_a"),
                "mppt3_idc_a": rr.get("mppt3_idc_a"),
                "mppt4_idc_a": rr.get("mppt4_idc_a"),
                "alarm_code": rr.get("alarm_code"),
                "alarm_sev": rr.get("alarm_sev"),
                "inv_coverage": rr.get("inv_coverage"),
                "flag_inv_missing": rr.get("flag_inv_missing"),
            }

        dump_by_tkey[tkey] = {
            "ts_local": tloc.isoformat(),
            "ts_utc": ts_utc.astimezone(dt_tz.utc).isoformat(),
            "source_meteo": src_meteo,
            "policy": agg["policy_used"][i],
            "confidence": {
                "data_reliability_score": confidence["data_reliability_score"][i],
                "data_reliability_level": confidence["data_reliability_level"][i],
                "detection_confidence_score": confidence["detection_confidence_score"][i],
                "detection_confidence_level": confidence["detection_confidence_level"][i],
                "diagnosis_confidence_score": confidence["diagnosis_confidence_score"][i],
                "diagnosis_confidence_level": confidence["diagnosis_confidence_level"][i],
                "state_label": _at(confidence.get("diag_state_labels"), i),
                "domain_label": _at(confidence.get("diag_domain_labels"), i),
                "diagnosis_label": _at(confidence.get("diag_diagnosis_labels"), i),
                "direct_grid_evidence": _at(confidence.get("diag_direct_grid"), i),
                "zero_injection_flag": _at(confidence.get("diag_zero_inj"), i),
                "irradiance_tier": _at(pipeline.get("irradiance_tier"), i),
                "notes": confidence["confidence_notes"][i],
            },
            "detection": {
                "base_gate": _at(pipeline.get("base_gate"), i),
                "gate_reason": _at(pipeline.get("gate_reason"), i),
                "gate_valid_model": _at(pipeline.get("gate_valid_model"), i),
                "gate_gpoa_ok": _at(pipeline.get("gate_gpoa_ok"), i),
                "gate_pac_ok": _at(pipeline.get("gate_pac_ok"), i),
                "gate_meteo_ok": _at(pipeline.get("gate_meteo_ok"), i),
                "gate_inverter_ok": _at(pipeline.get("gate_inverter_ok"), i),
                "valid_period": _at(pipeline.get("valid_period"), i),
                "coarse_period": _at(pipeline.get("coarse_period"), i),
                "fine_period": _at(pipeline.get("fine_period"), i),
                "meteo_quality_ok": _at(pipeline.get("meteo_quality_ok"), i),
                "stable_sky": _at(pipeline.get("stable_sky"), i),
                "anomaly": _at(pipeline.get("anomaly"), i),
                "anomaly_power": _at(pipeline.get("anomaly_power"), i),
                "residual_trigger": _at(pipeline.get("residual_trigger"), i),
                "residual_event_score": _at(pipeline.get("residual_event_score"), i),
                "combined_event_score": _at(pipeline.get("combined_event_score"), i),
                "detection_signal_rel": _at(pipeline.get("detection_signal_rel"), i),
                "ewma_z": _at(pipeline.get("ewma_z"), i),
                "cusum_score": _at(pipeline.get("cusum_score"), i),
                "irradiance_tier": _at(pipeline.get("irradiance_tier"), i),
                "rca_code": _at(pipeline.get("codes"), i),
                "rca_label": _at(pipeline.get("labels"), i),
                "state_label": _at(confidence.get("diag_state_labels"), i),
                "domain_label": _at(confidence.get("diag_domain_labels"), i),
                "diagnosis_label": _at(confidence.get("diag_diagnosis_labels"), i),
                "direct_grid_evidence": _at(confidence.get("diag_direct_grid"), i),
                "zero_injection_flag": _at(confidence.get("diag_zero_inj"), i),
            },
            "chosen_total": {
                "p_ac_w": _pick_first_not_none(agg["p_ac_w"][i], agg["p_ac_mppt_sum_w"][i], agg["p_ac_agg_w"][i], (any_row.get("p_ac_w") if any_row is not None else None)),
                "p_dc_w": agg["p_dc_w"][i],
                "p_ac_mppt_sum_w": agg["p_ac_mppt_sum_w"][i],
                "p_ac_agg_w": agg["p_ac_agg_w"][i],
                "v_dc_v": agg["v_dc_v"][i],
                "i_dc_a": agg["i_dc_a"][i],
                "v_ac_v": _pick_first_not_none(agg["v_ac_v"][i], (any_row.get("v_ac_v") if any_row is not None else None)),
                "i_ac_a": _pick_first_not_none(agg["i_ac_a"][i], (any_row.get("i_ac_a") if any_row is not None else None)),
                "freq_hz": _pick_first_not_none(agg["freq_hz"][i], _at(pipeline.get("freq_hz"), i), (any_row.get("freq_hz") if any_row is not None else None)),
                "inv_coverage": agg["inv_cov"][i],
                "flag_inv_missing_all": agg["flag_inv_missing_all"][i],
                "flag_inv_missing_partial": agg["flag_inv_missing_partial"][i],
            },
            "model": {
                "g_poa_used": _pick_first_not_none(_at(model.get("g_poa_used"), i), _at(residual_series.get("g_poa_used"), i)),
                "tcell_c": _pick_first_not_none(_at(model.get("tcell_c"), i), _at(residual_series.get("tcell_c"), i)),
                "p_ac_model_w": _pick_first_not_none(_at(model.get("pac_model_w"), i), _at(residual_series.get("pac_expected_w"), i)),
                "p_dc_model_w": _pick_first_not_none(_at(model.get("pdc_model_w"), i), _at(residual_series.get("pdc_expected_w"), i)),
                "v_dc_model_v": _pick_first_not_none(_at(model.get("v_dc_model_v"), i), _at(residual_series.get("v_dc_expected_v"), i)),
                "i_dc_model_a": _pick_first_not_none(_at(model.get("i_dc_model_a"), i), _at(residual_series.get("i_dc_expected_a"), i)),
                "mismatch_rel": _at(model.get("mismatch_rel"), i),
                "valid_model": _at(model.get("valid_model"), i),
            },
            "residuals": {
                "p_ac": {
                    "measured": agg["p_ac_w"][i],
                    "expected": _at(residual_series.get("pac_expected_w"), i),
                    "abs": _at(residual_series.get("p_ac_residual_abs"), i),
                    "rel": _at(residual_series.get("p_ac_residual_rel"), i),
                    "confidence": _at(ch_conf.get("p_ac"), i),
                },
                "p_dc": {
                    "measured": agg["p_dc_w"][i],
                    "expected": _at(residual_series.get("pdc_expected_w"), i),
                    "abs": _at(residual_series.get("p_dc_residual_abs"), i),
                    "rel": _at(residual_series.get("p_dc_residual_rel"), i),
                    "confidence": _at(ch_conf.get("p_dc"), i),
                },
                "v_dc": {
                    "measured": agg["v_dc_v"][i],
                    "expected": _at(residual_series.get("v_dc_expected_v"), i),
                    "abs": _at(residual_series.get("v_dc_residual_abs"), i),
                    "rel": _at(residual_series.get("v_dc_residual_rel"), i),
                    "confidence": _at(ch_conf.get("v_dc"), i),
                },
                "i_dc": {
                    "measured": agg["i_dc_a"][i],
                    "expected": _at(residual_series.get("i_dc_expected_a"), i),
                    "abs": _at(residual_series.get("i_dc_residual_abs"), i),
                    "rel": _at(residual_series.get("i_dc_residual_rel"), i),
                    "confidence": _at(ch_conf.get("i_dc"), i),
                },
                "global_confidence": _at(residual_series.get("global_confidence"), i),
            },
            "sources": src_dump,
            "meteo": meteo_dump,
            # atalhos flat para facilitar compatibilidade do drawer/template
            "base_gate": _at(pipeline.get("base_gate"), i),
            "gate_reason": _at(pipeline.get("gate_reason"), i),
            "gate_valid_model": _at(pipeline.get("gate_valid_model"), i),
            "gate_gpoa_ok": _at(pipeline.get("gate_gpoa_ok"), i),
            "gate_pac_ok": _at(pipeline.get("gate_pac_ok"), i),
            "gate_meteo_ok": _at(pipeline.get("gate_meteo_ok"), i),
            "gate_inverter_ok": _at(pipeline.get("gate_inverter_ok"), i),
            "detection_signal_rel": _at(pipeline.get("detection_signal_rel"), i),
            "ewma_z": _at(pipeline.get("ewma_z"), i),
            "cusum_score": _at(pipeline.get("cusum_score"), i),
            "residual_event_score": _at(pipeline.get("residual_event_score"), i),
            "combined_event_score": _at(pipeline.get("combined_event_score"), i),
            "rca_code": _at(pipeline.get("codes"), i),
            "rca_label": _at(pipeline.get("labels"), i),
            "code": _at(pipeline.get("codes"), i),
            "label": _at(pipeline.get("labels"), i),
            "state_label": _at(confidence.get("diag_state_labels"), i),
            "domain_label": _at(confidence.get("diag_domain_labels"), i),
            "diagnosis_label": _at(confidence.get("diag_diagnosis_labels"), i),
            "direct_grid_evidence": _at(confidence.get("diag_direct_grid"), i),
            "zero_injection_flag": _at(confidence.get("diag_zero_inj"), i),
            "p_ac_w": _pick_first_not_none(agg["p_ac_w"][i], agg["p_ac_mppt_sum_w"][i], agg["p_ac_agg_w"][i], (any_row.get("p_ac_w") if any_row is not None else None)),
            "v_ac_v": _pick_first_not_none(agg["v_ac_v"][i], (any_row.get("v_ac_v") if any_row is not None else None)),
            "i_ac_a": _pick_first_not_none(agg["i_ac_a"][i], (any_row.get("i_ac_a") if any_row is not None else None)),
            "freq_hz": _pick_first_not_none(agg["freq_hz"][i], _at(pipeline.get("freq_hz"), i), (any_row.get("freq_hz") if any_row is not None else None)),
            "p_ac_model_w": _pick_first_not_none(_at(model.get("pac_model_w"), i), _at(residual_series.get("pac_expected_w"), i)),
            "p_dc_model_w": _pick_first_not_none(_at(model.get("pdc_model_w"), i), _at(residual_series.get("pdc_expected_w"), i)),
            "v_dc_model_v": _pick_first_not_none(_at(model.get("v_dc_model_v"), i), _at(residual_series.get("v_dc_expected_v"), i)),
            "i_dc_model_a": _pick_first_not_none(_at(model.get("i_dc_model_a"), i), _at(residual_series.get("i_dc_expected_a"), i)),
            "g_poa_used": _pick_first_not_none(_at(model.get("g_poa_used"), i), _at(residual_series.get("g_poa_used"), i)),
            "tcell_c": _pick_first_not_none(_at(model.get("tcell_c"), i), _at(residual_series.get("tcell_c"), i)),
        }
    return dump_by_tkey
