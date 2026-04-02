from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .channels import build_residual_value
from .confidence import bundle_confidence
from .contracts import ResidualConfig
from .electrical_expectation import compute_expected_state
from .enums import ResidualGranularity
from .extraction import rows_from_mappings
from .meteo_chain import choose_effective_poa
from .types import ExpectedElectricalState, ResidualBundle, ResidualInputRow


def _list_to_np_nan(xs: List[Optional[float]]) -> np.ndarray:
    out = np.empty(len(xs), dtype=float)
    for i, v in enumerate(xs):
        out[i] = np.nan if v is None else float(v)
    return out


def _np_to_opt_list(arr: np.ndarray) -> List[Optional[float]]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(arr, dtype=float).tolist()]


def compute_residual_series_from_rows(*, plant: Any, rows: Iterable[Dict[str, Any]], source_oper: str = "", source_meteo: str = "", config: Optional[ResidualConfig] = None) -> Dict[str, Any]:
    cfg = config or ResidualConfig()
    row_objs = rows_from_mappings(rows, plant_id=int(plant.id), source_oper=source_oper, source_meteo=source_meteo)
    if not row_objs:
        return {"ok": True, "rows": [], "bundles": [], "series": {}}

    times_utc = [r.ts_utc for r in row_objs]
    gti = _list_to_np_nan([r.g_poa_wm2 for r in row_objs])
    ghi = _list_to_np_nan([r.ghi_wm2 for r in row_objs])
    dni = _list_to_np_nan([r.dni_wm2 for r in row_objs])
    dhi = _list_to_np_nan([r.dhi_wm2 for r in row_objs])
    temp_air = _list_to_np_nan([r.temp_air_c for r in row_objs])
    p_ac_real = _list_to_np_nan([r.p_ac_w for r in row_objs])
    p_dc_real = _list_to_np_nan([r.p_dc_w for r in row_objs])
    v_dc_real = _list_to_np_nan([r.v_dc_v for r in row_objs])
    i_dc_real = _list_to_np_nan([r.i_dc_a for r in row_objs])

    details = getattr(plant, "details", None)
    lat_deg = float(getattr(plant, "latitude", 0.0) or 0.0)
    lon_deg = float(getattr(plant, "longitude", 0.0) or 0.0)
    tilt_deg = float(getattr(details, "tilt_deg", 0.0) or 0.0) if details is not None else None
    azimuth_deg = float(getattr(details, "azimuth_deg", 0.0) or 0.0) if details is not None else None
    albedo = float(getattr(getattr(plant, "details", None), "albedo", 0.20) or 0.20)

    meteo = choose_effective_poa(
        times_utc=np.asarray(times_utc, dtype="datetime64[ns]"),
        gti=gti,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        albedo=albedo,
    )
    g_poa_used = np.asarray(meteo.get("g_poa_used"), dtype=float)
    model = compute_expected_state(
        plant=plant,
        times_utc=times_utc,
        g_poa_used=g_poa_used,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        temp_air=temp_air,
        p_ac_real=p_ac_real,
        p_dc_real=p_dc_real,
        v_dc_real=v_dc_real,
        i_dc_real=i_dc_real,
    )

    pac_expected = np.asarray(model.get("pac_expected_w"), dtype=float)
    pdc_expected = np.asarray(model.get("pdc_expected_w"), dtype=float)
    vdc_expected = np.asarray(model.get("v_dc_expected_v"), dtype=float)
    idc_expected = np.asarray(model.get("i_dc_expected_a"), dtype=float)
    valid_model = np.asarray(model.get("valid"), dtype=bool)
    tcell = np.asarray(model.get("tcell_c"), dtype=float)

    bundles: List[ResidualBundle] = []
    ch_conf = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}
    ch_status = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}
    ch_valid = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}
    ch_abs = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}
    ch_rel = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}
    ch_norm = {"p_ac": [], "p_dc": [], "v_dc": [], "i_dc": []}

    for i, row in enumerate(row_objs):
        exp_state = ExpectedElectricalState(
            tcell_c=None if not np.isfinite(tcell[i]) else float(tcell[i]),
            ee_poa_wm2=None if not np.isfinite(g_poa_used[i]) else float(g_poa_used[i]),
            p_dc_exp_w=None if not np.isfinite(pdc_expected[i]) else float(pdc_expected[i]),
            p_ac_exp_w=None if not np.isfinite(pac_expected[i]) else float(pac_expected[i]),
            v_dc_exp_v=None if not np.isfinite(vdc_expected[i]) else float(vdc_expected[i]),
            i_dc_exp_a=None if not np.isfinite(idc_expected[i]) else float(idc_expected[i]),
            model_valid=bool(valid_model[i]),
            model_notes=[],
        )
        v_pac = build_residual_value(channel="p_ac", observed=row.p_ac_w, expected=exp_state.p_ac_exp_w, row=row, expected_state=exp_state, cfg=cfg)
        v_pdc = build_residual_value(channel="p_dc", observed=row.p_dc_w, expected=exp_state.p_dc_exp_w, row=row, expected_state=exp_state, cfg=cfg)
        v_vdc = build_residual_value(channel="v_dc", observed=row.v_dc_v, expected=exp_state.v_dc_exp_v, row=row, expected_state=exp_state, cfg=cfg)
        v_idc = build_residual_value(channel="i_dc", observed=row.i_dc_a, expected=exp_state.i_dc_exp_a, row=row, expected_state=exp_state, cfg=cfg)
        confs = {"p_ac": v_pac.confidence, "p_dc": v_pdc.confidence, "v_dc": v_vdc.confidence, "i_dc": v_idc.confidence}
        bundle = ResidualBundle(
            ts_utc=row.ts_utc,
            granularity=ResidualGranularity.PLANT.value,
            scope_id=str(row.plant_id),
            p_ac=v_pac,
            p_dc=v_pdc,
            v_dc=v_vdc,
            i_dc=v_idc,
            tcell_c=exp_state.tcell_c,
            ee_poa_wm2=exp_state.ee_poa_wm2,
            global_confidence=bundle_confidence(confs),
            metadata={"source_oper": row.source_oper, "source_meteo": row.source_meteo},
        )
        bundles.append(bundle)
        for key, obj in (("p_ac", v_pac), ("p_dc", v_pdc), ("v_dc", v_vdc), ("i_dc", v_idc)):
            ch_conf[key].append(obj.confidence)
            ch_status[key].append(obj.status)
            ch_valid[key].append(bool(obj.valid))
            ch_abs[key].append(obj.abs_residual)
            ch_rel[key].append(obj.rel_residual)
            ch_norm[key].append(obj.norm_residual)

    return {
        "ok": True,
        "rows": row_objs,
        "bundles": bundles,
        "series": {
            "g_poa_used": _np_to_opt_list(g_poa_used),
            "tcell_c": _np_to_opt_list(tcell),
            "pac_expected_w": _np_to_opt_list(pac_expected),
            "pdc_expected_w": _np_to_opt_list(pdc_expected),
            "v_dc_expected_v": _np_to_opt_list(vdc_expected),
            "i_dc_expected_a": _np_to_opt_list(idc_expected),
            "p_ac_residual_abs": ch_abs["p_ac"],
            "p_ac_residual_rel": ch_rel["p_ac"],
            "p_ac_residual_norm": ch_norm["p_ac"],
            "p_dc_residual_abs": ch_abs["p_dc"],
            "p_dc_residual_rel": ch_rel["p_dc"],
            "p_dc_residual_norm": ch_norm["p_dc"],
            "v_dc_residual_abs": ch_abs["v_dc"],
            "v_dc_residual_rel": ch_rel["v_dc"],
            "v_dc_residual_norm": ch_norm["v_dc"],
            "i_dc_residual_abs": ch_abs["i_dc"],
            "i_dc_residual_rel": ch_rel["i_dc"],
            "i_dc_residual_norm": ch_norm["i_dc"],
            "valid_model": [bool(v) for v in valid_model.tolist()],
            "channel_confidence": ch_conf,
            "channel_status": ch_status,
            "channel_valid": ch_valid,
            "global_confidence": [float(b.global_confidence) for b in bundles],
            "v_ratio": _np_to_opt_list(np.asarray(model.get("v_ratio"), dtype=float)),
            "i_ratio": _np_to_opt_list(np.asarray(model.get("i_ratio"), dtype=float)),
            "meta": model.get("meta") or {},
        },
    }


def compute_residual_series_from_observations(*, plant: Any, times_utc: List[Any], gti: List[Optional[float]] | np.ndarray, ghi: List[Optional[float]] | np.ndarray, dni: List[Optional[float]] | np.ndarray, dhi: List[Optional[float]] | np.ndarray, temp_air: List[Optional[float]] | np.ndarray, p_ac_w: List[Optional[float]] | np.ndarray, p_dc_w: List[Optional[float]] | np.ndarray, v_dc_v: List[Optional[float]] | np.ndarray, i_dc_a: List[Optional[float]] | np.ndarray, v_ac_v: Optional[List[Optional[float]] | np.ndarray] = None, i_ac_a: Optional[List[Optional[float]] | np.ndarray] = None, freq_hz: Optional[List[Optional[float]] | np.ndarray] = None, meteo_qc_score: Optional[List[Optional[float]] | np.ndarray] = None, flag_meteo_missing: Optional[List[bool]] = None, flag_meteo_low_confidence: Optional[List[bool]] = None, flag_meteo_interpolated: Optional[List[bool]] = None, flag_meteo_outlier: Optional[List[bool]] = None, flag_meteo_artifact: Optional[List[bool]] = None, flag_inv_missing: Optional[List[bool]] = None, inv_coverage: Optional[List[Optional[float]] | np.ndarray] = None, source_oper: str = "", source_meteo: str = "", config: Optional[ResidualConfig] = None, g_poa_wm2: Optional[List[Optional[float]] | np.ndarray] = None) -> Dict[str, Any]:
    def at(seq, i):
        if seq is None:
            return None
        try:
            return seq[i]
        except Exception:
            return None
    rows = []
    n = len(times_utc)
    for i in range(n):
        rows.append({
            "ts_utc": times_utc[i],
            "source_oper": source_oper,
            "source_meteo": source_meteo,
            "p_ac_w": at(p_ac_w, i),
            "p_dc_w": at(p_dc_w, i),
            "v_dc_v": at(v_dc_v, i),
            "i_dc_a": at(i_dc_a, i),
            "v_ac_v": at(v_ac_v, i),
            "i_ac_a": at(i_ac_a, i),
            "freq_hz": at(freq_hz, i),
            "gti": at(gti, i),
            "g_poa_wm2": at(g_poa_wm2, i) if g_poa_wm2 is not None else at(gti, i),
            "ghi": at(ghi, i),
            "dni": at(dni, i),
            "dhi": at(dhi, i),
            "temp_air": at(temp_air, i),
            "meteo_qc_score": at(meteo_qc_score, i),
            "flag_meteo_missing": bool(at(flag_meteo_missing, i) or False),
            "flag_meteo_low_confidence": bool(at(flag_meteo_low_confidence, i) or False),
            "flag_meteo_interpolated": bool(at(flag_meteo_interpolated, i) or False),
            "flag_meteo_outlier": bool(at(flag_meteo_outlier, i) or False),
            "flag_meteo_artifact": bool(at(flag_meteo_artifact, i) or False),
            "flag_inv_missing": bool(at(flag_inv_missing, i) or False),
            "inv_coverage": at(inv_coverage, i),
        })
    return compute_residual_series_from_rows(plant=plant, rows=rows, source_oper=source_oper, source_meteo=source_meteo, config=config)


def compute_pac_model_and_mismatch(*, plant: Any, times_utc: List[Any], gti: np.ndarray, ghi: np.ndarray, dni: np.ndarray, dhi: np.ndarray, temp_air: np.ndarray, pac_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(times_utc)
    residuals = compute_residual_series_from_observations(
        plant=plant,
        times_utc=times_utc,
        gti=list(np.asarray(gti, dtype=float).tolist()),
        ghi=list(np.asarray(ghi, dtype=float).tolist()),
        dni=list(np.asarray(dni, dtype=float).tolist()),
        dhi=list(np.asarray(dhi, dtype=float).tolist()),
        temp_air=list(np.asarray(temp_air, dtype=float).tolist()),
        p_ac_w=list(np.asarray(pac_real, dtype=float).tolist()),
        p_dc_w=[None] * n,
        v_dc_v=[None] * n,
        i_dc_a=[None] * n,
    )
    series = residuals.get("series") or {}
    pac_model = np.asarray([np.nan if v is None else float(v) for v in series.get("pac_expected_w", [None] * n)], dtype=float)
    mismatch = np.asarray([np.nan if v is None else float(v) for v in series.get("p_ac_residual_rel", [None] * n)], dtype=float)
    return pac_model, mismatch
