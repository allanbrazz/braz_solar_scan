from __future__ import annotations

from datetime import datetime, timezone as dt_tz
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

def build_runtime_dump(
    *,
    tz: ZoneInfo,
    src_meteo: str,
    selected_sources: List[str],
    times_utc: List[datetime],
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]],
    agg: Dict[str, Any],
    confidence: Dict[str, Any],
) -> Dict[str, Any]:
    dump_by_tkey: Dict[str, Any] = {}
    for i, ts_utc in enumerate(times_utc):
        tloc = ts_utc.astimezone(tz)
        tkey = tloc.strftime("%Y-%m-%dT%H:%M")
        by_src = per_ts.get(ts_utc, {})
        meteo_dump: Dict[str, Any] = {}
        src_dump: Dict[str, Any] = {}

        any_row = None
        for sname in selected_sources:
            rr = by_src.get(sname)
            if rr is not None:
                any_row = rr
                break
        if any_row is not None:
            for k in ["gti", "ghi", "dni", "dhi", "temp_air", "wind_speed", "rh", "flag_meteo_missing", "meteo_qc_score"]:
                meteo_dump[k] = any_row.get(k)

        for sname in selected_sources:
            rr = by_src.get(sname)
            if rr is None:
                continue
            src_dump[sname] = {
                "p_ac_w": rr.get("p_ac_w"),
                "p_dc_w": rr.get("p_dc_w"),
                "e_ac_wh_15": rr.get("e_ac_wh_15"),
                "v_dc_v": rr.get("v_dc_v"),
                "i_dc_a": rr.get("i_dc_a"),
                "v_ac_v": rr.get("v_ac_v"),
                "i_ac_a": rr.get("i_ac_a"),
                "freq_hz": rr.get("freq_hz"),
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
                "notes": confidence["confidence_notes"][i],
            },
            "chosen_total": {
                "p_ac_w": agg["p_ac_w"][i],
                "p_ac_mppt_sum_w": agg["p_ac_mppt_sum_w"][i],
                "p_ac_agg_w": agg["p_ac_agg_w"][i],
                "inv_coverage": agg["inv_cov"][i],
                "flag_inv_missing_all": agg["flag_inv_missing_all"][i],
                "flag_inv_missing_partial": agg["flag_inv_missing_partial"][i],
                "freq_hz": agg["freq_hz"][i],
            },
            "sources": src_dump,
            "meteo": meteo_dump,
        }
    return dump_by_tkey

