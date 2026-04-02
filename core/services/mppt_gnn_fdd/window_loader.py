# core/services/mppt_gnn_fdd/window_loader.py
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from zoneinfo import ZoneInfo
from django.db.models import Count, Q

from core.models import PVPlant, PVPlantMergedRecord15m
from core.services.mppt_gnn_fdd.features import WindowArrays
from core.services.mppt_gnn_fdd.constants import T_STEPS_DEFAULT, DT_MIN_DEFAULT
from core.services.meteo_qc import METEO_QC_BOOL_COLS, METEO_QC_SCORE_COLS
from core.services.residuals.facade import compute_residual_series_from_observations


def _plant_tz(plant: PVPlant) -> ZoneInfo:
    tz_name = getattr(plant, "timezone", None) or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")




def _is_mppt_source(src: str) -> bool:
    return "|MPPT" in str(src or "").upper()


def _source_base(src: str) -> str:
    s = str(src or "").strip()
    if not s:
        return ""
    u = s.upper()
    pos = u.find("|MPPT")
    if pos >= 0:
        return s[:pos].strip()
    if u.endswith("|AGG"):
        return s[:-4].strip()
    return s


def _is_agg_source(src: str) -> bool:
    s = str(src or "").strip()
    if not s:
        return False
    u = s.upper()
    return ("|" not in u) or u.endswith("|AGG")

def _pick_best_source_meteo(
    plant_id: int,
    dt0_utc: datetime,
    dt1_utc: datetime,
) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")


def _pick_best_source_oper(
    plant_id: int,
    source_meteo: str,
    dt0_utc: datetime,
    dt1_utc: datetime,
) -> Optional[str]:
    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    if not rows:
        return None

    agg_rows = [r for r in rows if _is_agg_source((r or {}).get("source_oper"))]
    if agg_rows:
        return (agg_rows[0] or {}).get("source_oper")

    # legado: só existem rows ...|MPPTk no intervalo; colapsa para o source base.
    collapsed: Dict[str, int] = {}
    for r in rows:
        base = _source_base((r or {}).get("source_oper"))
        if not base:
            continue
        collapsed[base] = collapsed.get(base, 0) + int((r or {}).get("n") or 0)
    if not collapsed:
        return None
    return max(collapsed.items(), key=lambda kv: kv[1])[0]


def _grid_utc(
    dt0_utc: datetime,
    steps: int = T_STEPS_DEFAULT,
    dt_min: int = DT_MIN_DEFAULT,
) -> List[datetime]:
    out: List[datetime] = []
    cur = dt0_utc
    for _ in range(int(steps)):
        out.append(cur)
        cur = cur + timedelta(minutes=int(dt_min))
    return out


def _fill_on_grid(
    ts_grid: List[datetime],
    rows: List[Dict[str, Any]],
    key: str,
) -> np.ndarray:
    idx = {t: j for j, t in enumerate(ts_grid)}
    arr = np.full(len(ts_grid), np.nan, dtype=float)
    for r in rows:
        t = r["ts_utc"]
        j = idx.get(t)
        if j is None:
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            arr[j] = float(v)
        except Exception:
            pass
    return arr


def compute_pac_model_and_mismatch(
    *,
    plant: PVPlant,
    times_utc: List[datetime],
    gti: np.ndarray,
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    temp_air: np.ndarray,
    pac_real: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compat shim: a modelagem/resíduos canônicos agora vivem em
    core.services.residuals. Esta função mantém a assinatura legada
    consumida pelo pipeline MPPT.
    """
    n = len(times_utc)
    out = compute_residual_series_from_observations(
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
    s = out.get("series") or {}
    pac_model = np.asarray([np.nan if v is None else float(v) for v in s.get("pac_expected_w", [None] * n)], dtype=float)
    mismatch = np.asarray([np.nan if v is None else float(v) for v in s.get("p_ac_residual_rel", [None] * n)], dtype=float)
    return pac_model, mismatch

def load_daily_window(
    *,
    plant_id: int,
    day_local: date,
    n_mppt: int = 4,
) -> Tuple[WindowArrays, List[datetime], Dict[str, Any]]:
    """
    Retorna WindowArrays (globais + mppt), lista times_utc (grid) e meta.
    """
    plant = PVPlant.objects.filter(id=plant_id).first()
    if plant is None:
        raise ValueError("Plant not found")

    tz = _plant_tz(plant)

    dt0_local = datetime.combine(day_local, time.min, tzinfo=tz)
    dt1_local = dt0_local + timedelta(days=1)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    source_meteo = _pick_best_source_meteo(plant_id, dt0_utc, dt1_utc) or "OPENMETEO"
    source_oper = _pick_best_source_oper(plant_id, source_meteo, dt0_utc, dt1_utc) or "SHINEMONITOR"

    ts_grid = _grid_utc(dt0_utc)

    rows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            source_oper=source_oper,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).values(
            "ts_utc",
            "p_ac_w",
            "v_dc_v",
            "i_ac_a",
            "v_ac_v",
            "freq_hz",
            "gti",
            "ghi",
            "dni",
            "dhi",
            "temp_air",
            "meteo_qc_score",
            "flag_meteo_low_confidence",
            "flag_meteo_interpolated",
            "flag_meteo_outlier",
            "flag_meteo_artifact",
            "mppt1_vdc_v",
            "mppt2_vdc_v",
            "mppt3_vdc_v",
            "mppt4_vdc_v",
            "mppt1_idc_a",
            "mppt2_idc_a",
            "mppt3_idc_a",
            "mppt4_idc_a",
        )
    )

    pac = _fill_on_grid(ts_grid, rows, "p_ac_w")
    vdc_total = _fill_on_grid(ts_grid, rows, "v_dc_v")
    iac = _fill_on_grid(ts_grid, rows, "i_ac_a")
    vac = _fill_on_grid(ts_grid, rows, "v_ac_v")
    freq = _fill_on_grid(ts_grid, rows, "freq_hz")

    gti = _fill_on_grid(ts_grid, rows, "gti")
    ghi = _fill_on_grid(ts_grid, rows, "ghi")
    dni = _fill_on_grid(ts_grid, rows, "dni")
    dhi = _fill_on_grid(ts_grid, rows, "dhi")
    tair = _fill_on_grid(ts_grid, rows, "temp_air")

    mppt_vdc = np.full((n_mppt, len(ts_grid)), np.nan, dtype=float)
    mppt_idc = np.full((n_mppt, len(ts_grid)), np.nan, dtype=float)
    for k in range(1, n_mppt + 1):
        mppt_vdc[k - 1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_vdc_v")
        mppt_idc[k - 1] = _fill_on_grid(ts_grid, rows, f"mppt{k}_idc_a")

    residual_out = compute_residual_series_from_observations(
        plant=plant,
        times_utc=ts_grid,
        gti=list(np.asarray(gti, dtype=float).tolist()),
        ghi=list(np.asarray(ghi, dtype=float).tolist()),
        dni=list(np.asarray(dni, dtype=float).tolist()),
        dhi=list(np.asarray(dhi, dtype=float).tolist()),
        temp_air=list(np.asarray(tair, dtype=float).tolist()),
        p_ac_w=list(np.asarray(pac, dtype=float).tolist()),
        p_dc_w=[None] * len(ts_grid),
        v_dc_v=list(np.asarray(vdc_total, dtype=float).tolist()),
        i_dc_a=[None] * len(ts_grid),
        v_ac_v=list(np.asarray(vac, dtype=float).tolist()),
        i_ac_a=list(np.asarray(iac, dtype=float).tolist()),
        freq_hz=list(np.asarray(freq, dtype=float).tolist()),
        source_oper=source_oper,
        source_meteo=source_meteo,
    )
    rs = residual_out.get("series") or {}
    pac_model = np.asarray([np.nan if v is None else float(v) for v in rs.get("pac_expected_w", [None] * len(ts_grid))], dtype=float)
    mm = np.asarray([np.nan if v is None else float(v) for v in rs.get("p_ac_residual_rel", [None] * len(ts_grid))], dtype=float)

    meteo_qc = {}
    if rows:
        first = rows[0]
        for c in METEO_QC_SCORE_COLS:
            if c in first:
                arr = _fill_on_grid(ts_grid, rows, c)
                finite = arr[np.isfinite(arr)]
                meteo_qc[c] = {
                    "mean": None if finite.size == 0 else float(np.mean(finite)),
                    "min": None if finite.size == 0 else float(np.min(finite)),
                }
        for c in METEO_QC_BOOL_COLS:
            if c in first:
                meteo_qc[c] = bool(any(bool(r.get(c, False)) for r in rows))

    meta = {
        "plant_id": plant_id,
        "source_oper": source_oper,
        "source_meteo": source_meteo,
        "day_local": day_local.isoformat(),
        "tz": str(tz),
        "meteo_qc": meteo_qc,
        "residuals": {
            "p_ac_residual_rel": rs.get("p_ac_residual_rel"),
            "p_dc_residual_rel": rs.get("p_dc_residual_rel"),
            "v_dc_residual_rel": rs.get("v_dc_residual_rel"),
            "i_dc_residual_rel": rs.get("i_dc_residual_rel"),
            "channel_confidence": rs.get("channel_confidence"),
            "pac_expected_w": rs.get("pac_expected_w"),
            "v_dc_expected_v": rs.get("v_dc_expected_v"),
            "i_dc_expected_a": rs.get("i_dc_expected_a"),
            "global_confidence": rs.get("global_confidence"),
        },
    }

    win = WindowArrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        mismatch=mm,
        g=(gti if np.isfinite(gti).any() else ghi),
        t=tair,
        vac=vac,
        freq=freq,
        mppt_vdc=mppt_vdc,
        mppt_idc=mppt_idc,
    )
    return win, ts_grid, meta