# core/services/mppt_gnn_fdd/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

from core.services.mppt_gnn_fdd.window_loader import load_daily_window
from core.services.mppt_gnn_fdd.normalization import PlantScaler
from core.services.mppt_gnn_fdd.features import build_node_features, WindowArrays
from core.services.mppt_gnn_fdd.fault_injector import inject_fault, InjectConfig


@dataclass
class DatasetConfig:
    n_mppt: int = 4

    # dia "utilizável" (mínimo de sol e mínimo de geração)
    g_day_gate: float = 120.0
    pac_gate_w: float = 80.0
    min_day_points: int = 12  # pontos 15-min sob sol

    # se pac_model existir, usa pac_model como gate; senão, cai no pac_gate_w
    pac_model_gate: float = 80.0

    # critério de "normal" (se mismatch existir)
    mm_abs_p90_max_for_normal: float = 0.50  # relaxado (antes 0.20)

    # composição dataset
    n_days_max: int = 180
    base_days_target: int = 60  # quantos dias "base" queremos no máximo
    aug_per_day: int = 6
    seed: int = 42
    keep_most_recent_days: bool = True

    # mix de faults no augmentation
    p_disc: float = 0.25
    p_off: float = 0.20
    p_imb: float = 0.25
    p_curt: float = 0.20
    p_meteo: float = 0.10


def _sun_mask(win: WindowArrays, g_gate: float) -> np.ndarray:
    g = np.asarray(win.g, float)
    return np.isfinite(g) & (g >= float(g_gate))


def _day_is_usable(win: WindowArrays, cfg: DatasetConfig) -> bool:
    pac = np.asarray(win.pac, float)
    mask = _sun_mask(win, cfg.g_day_gate) & np.isfinite(pac)

    if int(mask.sum()) < int(cfg.min_day_points):
        return False

    pacv = pac[mask]
    pacv = pacv[np.isfinite(pacv)]
    if pacv.size == 0:
        return False

    p90 = float(np.nanpercentile(pacv, 90))
    return p90 >= float(cfg.pac_gate_w)


def _day_is_normal(win: WindowArrays, cfg: DatasetConfig) -> bool:
    if not _day_is_usable(win, cfg):
        return False

    pac = np.asarray(win.pac, float)
    pm = np.asarray(win.pac_model, float) if win.pac_model is not None else None
    mm = np.asarray(win.mismatch, float) if win.mismatch is not None else None

    mask = _sun_mask(win, cfg.g_day_gate) & np.isfinite(pac)

    if pm is not None and np.isfinite(pm[mask]).sum() >= cfg.min_day_points:
        mask = mask & np.isfinite(pm) & (pm >= float(cfg.pac_model_gate))

    if mm is not None and np.isfinite(mm[mask]).sum() >= cfg.min_day_points:
        mmv = np.abs(mm[mask])
        mmv = mmv[np.isfinite(mmv)]
        if mmv.size == 0:
            return False
        p90 = float(np.nanpercentile(mmv, 90))
        return p90 <= float(cfg.mm_abs_p90_max_for_normal)

    return True


def _flatten_node_sequence(X_node: np.ndarray) -> np.ndarray:
    N, T, F = X_node.shape
    return X_node.reshape(N, T * F)


def build_training_dataset(
    *,
    plant_id: int,
    start: date,
    end: date,
    cfg: DatasetConfig = DatasetConfig(),
) -> Tuple[np.ndarray, np.ndarray, PlantScaler, Dict[str, int], Dict[str, int], Dict[str, List[Any]]]:
    rng = np.random.default_rng(int(cfg.seed))

    d0, d1 = (start, end) if start <= end else (end, start)
    days: List[date] = []
    cur = d0
    while cur <= d1:
        days.append(cur)
        cur = cur + timedelta(days=1)

    if int(cfg.n_days_max) > 0 and len(days) > int(cfg.n_days_max):
        if bool(cfg.keep_most_recent_days):
            days = days[-int(cfg.n_days_max):]
        else:
            days = days[: int(cfg.n_days_max)]

    usable_windows: List[Tuple[date, WindowArrays]] = []
    normal_windows: List[Tuple[date, WindowArrays]] = []

    stats = {
        "days_scanned": 0,
        "days_usable": 0,
        "days_normal": 0,
        "fallback_used": 0,
    }

    for d in days:
        stats["days_scanned"] += 1
        try:
            win, _grid, _meta = load_daily_window(plant_id=plant_id, day_local=d, n_mppt=cfg.n_mppt)
        except Exception:
            continue

        if _day_is_usable(win, cfg):
            usable_windows.append((d, win))
        if _day_is_normal(win, cfg):
            normal_windows.append((d, win))

    stats["days_usable"] = len(usable_windows)
    stats["days_normal"] = len(normal_windows)

    if normal_windows:
        base_windows = normal_windows[: int(cfg.base_days_target)]
    else:
        if not usable_windows:
            raise RuntimeError(
                "Não encontrei dias utilizáveis no range (sem sol/geração suficiente). "
                "Ajuste start/end ou reduza g_day_gate/pac_gate_w."
            )
        base_windows = usable_windows[: int(cfg.base_days_target)]
        stats["fallback_used"] = 1

    base_day_list = [d for d, _ in base_windows]

    pac = np.concatenate([np.asarray(w.pac, float) for _d, w in base_windows])
    vdc_total = np.concatenate([np.asarray(w.vdc_total, float) for _d, w in base_windows])
    iac = np.concatenate([np.asarray(w.iac, float) for _d, w in base_windows])
    pac_model = np.concatenate([np.asarray(w.pac_model, float) for _d, w in base_windows])
    g = np.concatenate([np.asarray(w.g, float) for _d, w in base_windows])
    t = np.concatenate([np.asarray(w.t, float) for _d, w in base_windows])

    mv = np.concatenate([np.asarray(w.mppt_vdc, float) for _d, w in base_windows], axis=1)
    mi = np.concatenate([np.asarray(w.mppt_idc, float) for _d, w in base_windows], axis=1)

    scaler = PlantScaler.fit_from_arrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        g=g,
        t=t,
        mppt_vdc=mv,
        mppt_idc=mi,
    )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    sample_day_local: List[str] = []
    sample_kind: List[str] = []
    sample_fault_code: List[int] = []
    fmap: Dict[str, int] = {}

    inj_cfg = InjectConfig(g_gate=float(cfg.g_day_gate))

    for day_local, win in base_windows:
        X_node, fmap = build_node_features(win, scaler)
        Xf = _flatten_node_sequence(X_node)
        X_list.append(Xf)
        y_list.append(np.zeros(cfg.n_mppt, dtype=int))
        sample_day_local.extend([day_local.isoformat()] * int(Xf.shape[0]))
        sample_kind.extend(["real_normal"] * int(Xf.shape[0]))
        sample_fault_code.extend([0] * int(Xf.shape[0]))

    probs = np.array([cfg.p_disc, cfg.p_off, cfg.p_imb, cfg.p_curt, cfg.p_meteo], dtype=float)
    probs = probs / probs.sum()

    for day_local, win in base_windows:
        for _ in range(int(cfg.aug_per_day)):
            fault_choice = int(rng.choice([1, 2, 3, 4, 5], p=probs))
            mppt_k = None
            if fault_choice in (1, 3):
                mppt_k = int(rng.integers(1, cfg.n_mppt + 1))
            win2, y = inject_fault(win, fault_code=fault_choice, mppt_k=mppt_k, rng=rng, cfg=inj_cfg)
            X_node2, fmap = build_node_features(win2, scaler)
            Xf2 = _flatten_node_sequence(X_node2)
            X_list.append(Xf2)
            y_list.append(y.astype(int))
            sample_day_local.extend([day_local.isoformat()] * int(Xf2.shape[0]))
            sample_kind.extend(["synthetic_fault"] * int(Xf2.shape[0]))
            sample_fault_code.extend([int(fault_choice)] * int(Xf2.shape[0]))

    X_block = np.concatenate(X_list, axis=0)
    y_block = np.concatenate(y_list, axis=0)

    stats_out = {
        **stats,
        "base_days_used": len(base_windows),
        "base_day_start": base_day_list[0].isoformat() if base_day_list else None,
        "base_day_end": base_day_list[-1].isoformat() if base_day_list else None,
        "samples_total": int(X_block.shape[0]),
        "dim": int(X_block.shape[1]),
    }
    sample_meta = {
        "day_local": sample_day_local,
        "sample_kind": sample_kind,
        "fault_code": sample_fault_code,
    }
    return X_block.astype(np.float32), y_block.astype(int), scaler, fmap, stats_out, sample_meta
