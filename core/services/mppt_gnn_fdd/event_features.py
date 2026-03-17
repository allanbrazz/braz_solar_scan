from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from core.services.mppt_gnn_fdd.constants import EPS
from core.services.mppt_gnn_fdd.features import WindowArrays


def _safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else 0.0


def _safe_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else 0.0


def _safe_min(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else 0.0


def _safe_frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(mask)) if mask.size else 0.0


def build_event_mppt_features(
    *,
    win: WindowArrays,
    ts_grid: List[datetime],
    event_start_utc: datetime,
    event_end_utc: datetime,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ts_arr = np.asarray(ts_grid, dtype="datetime64[ns]")
    event_mask = (ts_arr >= np.datetime64(event_start_utc)) & (ts_arr <= np.datetime64(event_end_utc))
    if not bool(np.any(event_mask)):
        raise ValueError("Máscara do evento vazia")

    pac_evt = np.asarray(win.pac[event_mask], float)
    pac_model_evt = np.asarray(win.pac_model[event_mask], float)
    mismatch_evt = np.asarray(win.mismatch[event_mask], float)
    g_evt = np.asarray(win.g[event_mask], float)

    pdc = np.asarray(win.mppt_vdc * win.mppt_idc, float)
    pdc_evt = pdc[:, event_mask]
    i_evt = np.asarray(win.mppt_idc[:, event_mask], float)
    v_evt = np.asarray(win.mppt_vdc[:, event_mask], float)

    p_sum = np.nansum(pdc_evt, axis=0)
    i_med = np.nanmedian(i_evt, axis=0)
    v_med = np.nanmedian(v_evt, axis=0)

    pac_cap = float(np.nanpercentile(win.pac[np.isfinite(win.pac)], 99)) if np.isfinite(win.pac).any() else 0.0
    clip_mask = (
        np.isfinite(pac_evt)
        & np.isfinite(pac_model_evt)
        & (pac_cap > 0.0)
        & (pac_evt >= 0.98 * pac_cap)
        & (pac_model_evt > pac_evt * 1.02)
    )
    pac_ratio = np.divide(
        pac_evt,
        np.maximum(np.abs(pac_model_evt), 50.0),
        out=np.zeros_like(pac_evt, dtype=float),
        where=np.isfinite(pac_evt) & np.isfinite(pac_model_evt),
    )

    plant_summary: Dict[str, Any] = {
        "duration_bins": int(np.sum(event_mask)),
        "mismatch_mean": _safe_mean(mismatch_evt),
        "mismatch_min": _safe_min(mismatch_evt),
        "mismatch_abs_max": float(np.max(np.abs(mismatch_evt[np.isfinite(mismatch_evt)]))) if np.isfinite(mismatch_evt).any() else 0.0,
        "pac_ratio_mean": _safe_mean(pac_ratio),
        "zero_power_frac": _safe_frac(np.nan_to_num(pac_evt, nan=0.0) <= max(100.0, 0.05 * pac_cap)),
        "clip_frac": _safe_frac(clip_mask),
        "g_mean": _safe_mean(g_evt),
        "g_high_frac": _safe_frac(np.nan_to_num(g_evt, nan=0.0) >= 800.0),
        "g_mid_frac": _safe_frac((np.nan_to_num(g_evt, nan=0.0) >= 700.0) & (np.nan_to_num(g_evt, nan=0.0) < 800.0)),
        "pac_cap_w": pac_cap,
        "n_mppt": int(pdc_evt.shape[0]),
    }

    out: List[Dict[str, Any]] = []
    expected_share = 1.0 / max(int(pdc_evt.shape[0]), 1)

    for k in range(pdc_evt.shape[0]):
        i_rel = np.divide(
            i_evt[k],
            np.maximum(i_med, EPS),
            out=np.zeros_like(i_evt[k], dtype=float),
            where=np.isfinite(i_evt[k]),
        )
        v_rel_ratio = np.divide(
            v_evt[k],
            np.maximum(np.abs(v_med), EPS),
            out=np.zeros_like(v_evt[k], dtype=float),
            where=np.isfinite(v_evt[k]),
        )
        share_p = np.divide(
            pdc_evt[k],
            np.maximum(p_sum, EPS),
            out=np.zeros_like(pdc_evt[k], dtype=float),
            where=np.isfinite(pdc_evt[k]),
        )

        out.append(
            {
                "mppt": k + 1,
                "i_rel_med": _safe_median(i_rel),
                "i_rel_min": _safe_min(i_rel),
                "v_rel_med": _safe_median(v_rel_ratio),
                "share_p_med": _safe_median(share_p),
                "share_p_min": _safe_min(share_p),
                "outage_frac": _safe_frac(i_rel <= 0.15),
                "low_i_frac": _safe_frac(i_rel <= 0.50),
                "low_v_frac": _safe_frac(v_rel_ratio <= 0.80),
                "share_low_frac": _safe_frac(share_p <= 0.50 * expected_share),
                "pdc_mean_w": _safe_mean(pdc_evt[k]),
            }
        )

    return out, plant_summary
