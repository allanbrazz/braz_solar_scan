from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.services.fdd_mismatch import CODE_INVALID
from core.services.fdd.dashboard_common import as_float, is_agg_source, is_mppt_source, mean_none, sum_none

DUMP_FIELDS = [
    "p_ac_w", "p_dc_w", "e_ac_wh_15", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a", "freq_hz",
    "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
    "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
    "alarm_code", "alarm_sev",
    "inv_coverage", "flag_inv_missing",
    "gti", "ghi", "dni", "dhi", "temp_air", "wind_speed", "rh", "flag_meteo_missing",
]

RCA_CODE_TO_SEV = {
    str(CODE_INVALID): "none",
    "0": "ok",
    "1": "warn",
    "2": "warn",
    "3": "crit",
    "4": "crit",
}


def aggregate_runtime_series(
    *,
    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]],
    times_utc: List[datetime],
    selected_sources: List[str],
) -> Dict[str, Any]:
    n = len(times_utc)
    p_ac_w: List[Optional[float]] = [None] * n
    p_dc_w: List[Optional[float]] = [None] * n
    e_ac_wh_15: List[Optional[float]] = [None] * n
    v_dc_v: List[Optional[float]] = [None] * n
    i_dc_a: List[Optional[float]] = [None] * n
    v_ac_v: List[Optional[float]] = [None] * n
    i_ac_a: List[Optional[float]] = [None] * n
    freq_hz: List[Optional[float]] = [None] * n
    inv_cov: List[Optional[float]] = [None] * n
    flag_inv_missing_all: List[bool] = [False] * n
    flag_inv_missing_partial: List[bool] = [False] * n

    gti: List[Optional[float]] = [None] * n
    ghi: List[Optional[float]] = [None] * n
    dni: List[Optional[float]] = [None] * n
    dhi: List[Optional[float]] = [None] * n
    temp_air: List[Optional[float]] = [None] * n
    wind_speed: List[Optional[float]] = [None] * n
    rh: List[Optional[float]] = [None] * n
    meteo_qc_score: List[Optional[float]] = [None] * n
    flag_meteo_low_confidence: List[bool] = [False] * n
    flag_meteo_interpolated: List[bool] = [False] * n
    flag_meteo_outlier: List[bool] = [False] * n
    flag_meteo_artifact: List[bool] = [False] * n
    flag_meteo_missing: List[bool] = [False] * n

    p_ac_mppt_sum_w: List[Optional[float]] = [None] * n
    p_ac_agg_w: List[Optional[float]] = [None] * n
    policy_used: List[str] = [""] * n

    series_by_source: Dict[str, Dict[str, List[Any]]] = {
        src: {
            "p_ac_w": [None] * n,
            "p_dc_w": [None] * n,
            "e_ac_wh_15": [None] * n,
            "v_dc_v": [None] * n,
            "i_dc_a": [None] * n,
            "v_ac_v": [None] * n,
            "i_ac_a": [None] * n,
            "freq_hz": [None] * n,
            "mppt1_vdc_v": [None] * n, "mppt2_vdc_v": [None] * n, "mppt3_vdc_v": [None] * n, "mppt4_vdc_v": [None] * n,
            "mppt1_idc_a": [None] * n, "mppt2_idc_a": [None] * n, "mppt3_idc_a": [None] * n, "mppt4_idc_a": [None] * n,
            "alarm_code": [None] * n,
            "alarm_sev": [None] * n,
            "inv_coverage": [None] * n,
            "flag_inv_missing": [None] * n,
        }
        for src in selected_sources
    }

    for i, ts in enumerate(times_utc):
        by_src = per_ts.get(ts, {})
        present = [s for s in selected_sources if s in by_src]
        present_mppt = [s for s in present if is_mppt_source(s)]
        present_agg = [s for s in present if is_agg_source(s)]

        if present_mppt:
            chosen = present_mppt
            policy = "mppt_sum"
        elif present_agg:
            chosen = present_agg
            policy = "agg_fallback"
        else:
            chosen = present
            policy = "any_fallback"
        policy_used[i] = policy

        first_row: Optional[Dict[str, Any]] = None
        for s0 in present:
            rr = by_src.get(s0)
            if rr is not None:
                first_row = rr
                break

        if first_row is not None:
            gti[i] = as_float(first_row.get("gti"))
            ghi[i] = as_float(first_row.get("ghi"))
            dni[i] = as_float(first_row.get("dni"))
            dhi[i] = as_float(first_row.get("dhi"))
            temp_air[i] = as_float(first_row.get("temp_air"))
            wind_speed[i] = as_float(first_row.get("wind_speed"))
            rh[i] = as_float(first_row.get("rh"))
            meteo_qc_score[i] = as_float(first_row.get("meteo_qc_score"))
            flag_meteo_low_confidence[i] = bool(first_row.get("flag_meteo_low_confidence") or False)
            flag_meteo_interpolated[i] = bool(first_row.get("flag_meteo_interpolated") or False)
            flag_meteo_outlier[i] = bool(first_row.get("flag_meteo_outlier") or False)
            flag_meteo_artifact[i] = bool(first_row.get("flag_meteo_artifact") or False)
            flag_meteo_missing[i] = bool(first_row.get("flag_meteo_missing") or False)

        for src in present:
            r = by_src.get(src)
            if r is None:
                continue
            sb = series_by_source[src]
            sb["p_ac_w"][i] = as_float(r.get("p_ac_w"))
            sb["p_dc_w"][i] = as_float(r.get("p_dc_w"))
            sb["e_ac_wh_15"][i] = as_float(r.get("e_ac_wh_15"))
            sb["v_dc_v"][i] = as_float(r.get("v_dc_v"))
            sb["i_dc_a"][i] = as_float(r.get("i_dc_a"))
            sb["v_ac_v"][i] = as_float(r.get("v_ac_v"))
            sb["i_ac_a"][i] = as_float(r.get("i_ac_a"))
            sb["freq_hz"][i] = as_float(r.get("freq_hz"))
            sb["inv_coverage"][i] = as_float(r.get("inv_coverage"))
            sb["flag_inv_missing"][i] = bool(r.get("flag_inv_missing") or False)
            for key in ("mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v", "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a"):
                sb[key][i] = as_float(r.get(key))
            sb["alarm_code"][i] = r.get("alarm_code")
            sb["alarm_sev"][i] = r.get("alarm_sev")

        pac_mppt = sum_none([as_float(by_src[s].get("p_ac_w")) for s in present_mppt]) if present_mppt else None
        pac_agg = sum_none([as_float(by_src[s].get("p_ac_w")) for s in present_agg]) if present_agg else None
        p_ac_mppt_sum_w[i] = pac_mppt
        p_ac_agg_w[i] = pac_agg

        pac_l = [as_float(by_src[s].get("p_ac_w")) for s in chosen] if chosen else []
        pdc_l = [as_float(by_src[s].get("p_dc_w")) for s in chosen] if chosen else []
        e15_l = [as_float(by_src[s].get("e_ac_wh_15")) for s in chosen] if chosen else []
        vdc_l = [as_float(by_src[s].get("v_dc_v")) for s in chosen] if chosen else []
        idc_l = [as_float(by_src[s].get("i_dc_a")) for s in chosen] if chosen else []
        cov_l = [as_float(by_src[s].get("inv_coverage")) for s in chosen] if chosen else []

        agg_ref_src = present_agg[0] if present_agg else None
        agg_ref_row = by_src.get(agg_ref_src) if agg_ref_src else None

        vac_candidates = (
            [as_float(by_src[s].get("v_ac_v")) for s in present_agg]
            if present_agg else
            [as_float(by_src[s].get("v_ac_v")) for s in chosen]
        )
        fac_candidates = (
            [as_float(by_src[s].get("freq_hz")) for s in present_agg]
            if present_agg else
            [as_float(by_src[s].get("freq_hz")) for s in chosen]
        )

        agg_iac = as_float(agg_ref_row.get("i_ac_a")) if agg_ref_row is not None else None
        if agg_iac is not None:
            iac_value = agg_iac
        else:
            iac_candidates = [as_float(by_src[s].get("i_ac_a")) for s in chosen] if chosen else []
            iac_value = next((x for x in iac_candidates if x is not None), None)

        pac_total = pac_mppt if pac_mppt is not None else pac_agg
        if pac_total is None:
            pac_total = sum_none(pac_l)

        active_vdc = []
        if chosen:
            for s in chosen:
                vv = as_float(by_src[s].get("v_dc_v"))
                ii = as_float(by_src[s].get("i_dc_a"))
                pp = as_float(by_src[s].get("p_dc_w"))
                is_active = (
                    (pp is not None and pp > 1.0)
                    or (ii is not None and ii > 0.25)
                )
                if is_active and vv is not None and vv > 0.0:
                    active_vdc.append(vv)
        vdc_value = mean_none(active_vdc) if active_vdc else mean_none(vdc_l)

        miss_flags = [bool(by_src[s].get("flag_inv_missing") or False) for s in chosen] if chosen else []
        p_ac_w[i] = pac_total
        p_dc_w[i] = sum_none(pdc_l)
        e_ac_wh_15[i] = sum_none(e15_l)
        v_dc_v[i] = vdc_value
        i_dc_a[i] = sum_none(idc_l)
        v_ac_v[i] = mean_none(vac_candidates)
        i_ac_a[i] = iac_value
        freq_hz[i] = mean_none(fac_candidates)
        inv_cov[i] = mean_none(cov_l)

        if not miss_flags:
            flag_inv_missing_all[i] = True
            flag_inv_missing_partial[i] = False
        else:
            all_miss = all(miss_flags)
            any_miss = any(miss_flags)
            flag_inv_missing_all[i] = bool(all_miss)
            flag_inv_missing_partial[i] = bool(any_miss and (not all_miss))

    return {
        "p_ac_w": p_ac_w,
        "p_dc_w": p_dc_w,
        "e_ac_wh_15": e_ac_wh_15,
        "v_dc_v": v_dc_v,
        "i_dc_a": i_dc_a,
        "v_ac_v": v_ac_v,
        "i_ac_a": i_ac_a,
        "freq_hz": freq_hz,
        "inv_cov": inv_cov,
        "flag_inv_missing_all": flag_inv_missing_all,
        "flag_inv_missing_partial": flag_inv_missing_partial,
        "gti": gti,
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "temp_air": temp_air,
        "wind_speed": wind_speed,
        "rh": rh,
        "meteo_qc_score": meteo_qc_score,
        "flag_meteo_low_confidence": flag_meteo_low_confidence,
        "flag_meteo_interpolated": flag_meteo_interpolated,
        "flag_meteo_outlier": flag_meteo_outlier,
        "flag_meteo_artifact": flag_meteo_artifact,
        "flag_meteo_missing": flag_meteo_missing,
        "p_ac_mppt_sum_w": p_ac_mppt_sum_w,
        "p_ac_agg_w": p_ac_agg_w,
        "policy_used": policy_used,
        "series_by_source": series_by_source,
    }
