from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from django.db.models import Count

from core.models import PVPlant, PVPlantMergedRecord15m
from core.services.mppt_gnn_fdd.config import TrainConfig, load_mppt_nominals
from core.services.mppt_gnn_fdd.graph import build_edge_attr
from core.services.mppt_gnn_fdd.labeling import weak_label_disconnected, IGNORE


def pick_best_source_meteo(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(plant_id=plant_id, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc)
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")


def list_source_oper(plant_id: int, source_meteo: str, dt0_utc: datetime, dt1_utc: datetime) -> list[str]:
    rows = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id, source_meteo=source_meteo, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    return [r["source_oper"] for r in rows if r.get("source_oper")]


def _to_local_range(plant: PVPlant, d0: date, d1: date) -> tuple[datetime, datetime, ZoneInfo]:
    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    tz = ZoneInfo(tz_name)
    dt0_local = datetime.combine(d0, time.min, tzinfo=tz)
    dt1_local = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz)  # exclusivo
    return dt0_local, dt1_local, tz


def _normalize_series(v: np.ndarray, i: np.ndarray, *, n_series: int, n_parallel: int, voc: float, isc: float):
    v = v.astype(np.float32, copy=False)
    i = i.astype(np.float32, copy=False)
    den_v = max(float(n_series) * float(voc), 1e-6)
    den_i = max(float(n_parallel) * float(isc), 1e-6)
    v_norm = v / den_v
    i_norm = i / den_i
    p_norm = v_norm * i_norm
    return v_norm, i_norm, p_norm


@dataclass
class GraphWindow:
    X_ts: np.ndarray     # [Nmax,T,F]
    E_attr: np.ndarray   # [Nmax,Nmax,Fe]
    mask: np.ndarray     # [Nmax]
    y: np.ndarray        # [Nmax] (0/1 ou IGNORE)
    ts_end_utc: datetime
    source_oper: str


def iter_graph_windows_from_db(
    *,
    plant_id: int,
    start: date,
    end: date,
    cfg: TrainConfig,
    source_oper: str | None = None,
    source_meteo: str | None = None,
    stride_steps: int | None = None,
    mode: str = "train",  # "train" ou "infer"
) -> Iterable[GraphWindow]:
    """
    Cria janelas [T=96] diretamente do DB (sem export).
    - train: stride default cfg.stride_steps (ex 4)
    - infer: tipicamente stride 1 (15 min)
    """
    plant = PVPlant.objects.filter(id=plant_id).select_related("details", "details__module").first()
    if plant is None:
        raise ValueError("Plant not found")

    dt0_local, dt1_local, tz = _to_local_range(plant, start, end)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    if source_meteo is None:
        source_meteo = pick_best_source_meteo(plant_id, dt0_utc, dt1_utc)
    if not source_meteo:
        return

    stride = int(stride_steps or (1 if mode == "infer" else cfg.stride_steps))
    T = int(cfg.T)
    Nmax = 4

    nom = load_mppt_nominals(plant, nmax=Nmax)
    E_attr = build_edge_attr(nom, nmax=Nmax)

    sources = [source_oper] if source_oper else list_source_oper(plant_id, source_meteo, dt0_utc, dt1_utc)
    if not sources:
        return

    # margem para inferência (janela lookback)
    margin = timedelta(minutes=15 * (T - 1))
    q0 = dt0_utc - margin if mode == "infer" else dt0_utc
    q1 = dt1_utc

    fields = [
        "ts_utc", "source_oper", "source_meteo",
        "gti", "ghi", "temp_air",
        "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
        "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
        "flag_inv_missing", "inv_coverage",
    ]

    for src in sources:
        qs = (
            PVPlantMergedRecord15m.objects.filter(
                plant_id=plant_id,
                source_meteo=source_meteo,
                source_oper=src,
                ts_utc__gte=q0,
                ts_utc__lt=q1,
            )
            .order_by("ts_utc")
            .values(*fields)
        )
        rows = list(qs)
        if not rows:
            continue

        df = pd.DataFrame.from_records(rows)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df["ts_local"] = df["ts_utc"].dt.tz_convert(tz)
        df = df.set_index("ts_local").sort_index()

        # reindex 15min
        full_idx = pd.date_range(df.index.min().floor("15min"), df.index.max().ceil("15min"), freq="15min", tz=tz)
        df = df.reindex(full_idx)

        gpoa = df["gti"].astype("float32").fillna(df["ghi"].astype("float32"))
        tair = df["temp_air"].astype("float32")

        # broadcast meteo (normalizado)
        gpoa_norm = (gpoa / 1000.0).clip(lower=0.0, upper=2.0).to_numpy(dtype=np.float32)
        tair_norm = ((tair + 10.0) / 60.0).clip(lower=-1.0, upper=2.0).to_numpy(dtype=np.float32)

        # monta arrays por MPPT
        v_raw = np.full((Nmax, len(df)), np.nan, dtype=np.float32)
        i_raw = np.full((Nmax, len(df)), np.nan, dtype=np.float32)

        for k in range(1, Nmax + 1):
            vcol = f"mppt{k}_vdc_v"
            icol = f"mppt{k}_idc_a"
            if vcol in df.columns:
                v_raw[k - 1] = df[vcol].to_numpy(dtype=np.float32)
            if icol in df.columns:
                i_raw[k - 1] = df[icol].to_numpy(dtype=np.float32)

        # normaliza por nominais; se não tiver nominal, máscara false
        mask = np.zeros((Nmax,), dtype=bool)
        v_norm = np.full_like(v_raw, np.nan, dtype=np.float32)
        i_norm = np.full_like(i_raw, np.nan, dtype=np.float32)
        p_norm = np.full_like(v_raw, np.nan, dtype=np.float32)

        for k in range(1, Nmax + 1):
            nk = nom.get(k)
            if nk is None:
                continue
            mask[k - 1] = True
            vn, in_, pn = _normalize_series(
                v_raw[k - 1], i_raw[k - 1],
                n_series=nk.n_series, n_parallel=nk.n_parallel, voc=nk.voc_v, isc=nk.isc_a,
            )
            v_norm[k - 1] = vn
            i_norm[k - 1] = in_
            p_norm[k - 1] = pn

        # features: [v_norm, i_norm, p_norm, gpoa_norm, tair_norm] => F=5
        F = 5 if cfg.use_meteo else 3
        X_all = np.zeros((Nmax, len(df), F), dtype=np.float32)
        X_all[:, :, 0] = np.nan_to_num(v_norm, nan=0.0)
        X_all[:, :, 1] = np.nan_to_num(i_norm, nan=0.0)
        X_all[:, :, 2] = np.nan_to_num(p_norm, nan=0.0)
        if cfg.use_meteo:
            X_all[:, :, 3] = np.nan_to_num(gpoa_norm, nan=0.0)[None, :]
            X_all[:, :, 4] = np.nan_to_num(tair_norm, nan=0.0)[None, :]

        # sliding windows
        # define alvos (infer: apenas janelas cujo fim está no range de interesse)
        idx = df.index
        for end_pos in range(T - 1, len(df), stride):
            ts_end_local = idx[end_pos]
            if mode == "infer":
                if not (dt0_local <= ts_end_local < dt1_local):
                    continue

            start_pos = end_pos - (T - 1)
            X_ts = X_all[:, start_pos : end_pos + 1, :]  # [N,T,F]
            # gate de qualidade mínima: precisa ter ao menos 2 MPPTs com "atividade" em parte da janela
            active_nodes = 0
            for k in range(Nmax):
                if not mask[k]:
                    continue
                # usa p_norm (col 2) como proxy de presença
                if float((X_ts[k, :, 2] > 0.001).mean()) >= 0.10:
                    active_nodes += 1
            if active_nodes < 2:
                continue

            # labels (train) ou IGNORE (infer não precisa label)
            if mode == "train":
                y = weak_label_disconnected(
                    gpoa_wm2=np.nan_to_num(gpoa.to_numpy(dtype=np.float32)[start_pos : end_pos + 1], nan=0.0),
                    i_norm_by_mppt=i_norm[:, start_pos : end_pos + 1],
                    mask=mask,
                    g_gate=float(cfg.gpoa_gate_wm2),
                    i_disc=float(cfg.i_disc_thresh),
                    i_peer=float(cfg.i_peer_active),
                    min_sun_points=int(cfg.min_sun_points),
                    min_peer_points=int(cfg.min_peer_points),
                )
            else:
                y = np.full((Nmax,), IGNORE, dtype=np.int64)

            ts_end_utc = ts_end_local.to_pydatetime().astimezone(dt_tz.utc)
            yield GraphWindow(
                X_ts=X_ts.astype(np.float32, copy=False),
                E_attr=E_attr,
                mask=mask,
                y=y,
                ts_end_utc=ts_end_utc,
                source_oper=src,
            )