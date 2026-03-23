# core/services/mppt_gnn_fdd/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.services.mppt_gnn_fdd.constants import EPS
from core.services.mppt_gnn_fdd.normalization import PlantScaler


@dataclass
class WindowArrays:
    # globais [T]
    pac: np.ndarray
    vdc_total: np.ndarray
    iac: np.ndarray
    pac_model: np.ndarray
    mismatch: np.ndarray
    g: np.ndarray
    t: np.ndarray

    # mppt [N,T] -> campos obrigatórios antes dos opcionais
    mppt_vdc: np.ndarray
    mppt_idc: np.ndarray

    # opcionais [T]
    vac: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None


def _diff(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    if x.size > 1:
        y[1:] = x[1:] - x[:-1]
    return y


def _nan_to_0(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)


def build_node_features(
    win: WindowArrays,
    scaler: PlantScaler,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Retorna:
      X_node: [N,T,F]
      fmap: nome_feature -> idx

    NAN-SAFE:
      - nanmedian/nansum
      - nan_to_num antes de divisões e stack final
    """
    pac = np.asarray(win.pac, float)
    vdc_total = np.asarray(win.vdc_total, float)
    iac = np.asarray(win.iac, float)
    pac_model = np.asarray(win.pac_model, float)
    mm = np.asarray(win.mismatch, float)
    g = np.asarray(win.g, float)
    t = np.asarray(win.t, float)

    v_raw = np.asarray(win.mppt_vdc, float)  # [N,T]
    i_raw = np.asarray(win.mppt_idc, float)

    N, T = v_raw.shape

    # opcionais
    vac_raw = None if win.vac is None else np.asarray(win.vac, float)
    freq_raw = None if win.freq is None else np.asarray(win.freq, float)

    # pdc estimado (nan-safe)
    pdc_est_raw = v_raw * i_raw

    # normalizações (nan-safe)
    pac_n = scaler.n_pos(_nan_to_0(pac), scaler.pac_p99)
    vdc_total_n = scaler.n_pos(_nan_to_0(vdc_total), scaler.vdc_total_p99)
    iac_n = scaler.n_pos(_nan_to_0(iac), scaler.iac_p99)
    pac_model_n = scaler.n_pos(_nan_to_0(pac_model), scaler.pac_model_p99)
    g_n = scaler.n_pos(_nan_to_0(g), scaler.g_p99)
    t_n = scaler.n_pos(_nan_to_0(t), scaler.t_p99)

    mm0 = np.asarray(mm, float)
    mm0 = np.nan_to_num(mm0, nan=0.0, posinf=0.0, neginf=0.0)
    mm_n = scaler.mismatch_clip(mm0)  # [-1..1]

    v0 = _nan_to_0(v_raw)
    i0 = _nan_to_0(i_raw)
    pdc0 = _nan_to_0(pdc_est_raw)

    v_n = scaler.n_pos(v0, scaler.mppt_vdc_p99)
    i_n = scaler.n_pos(i0, scaler.mppt_idc_p99)
    pdc_n = scaler.n_pos(pdc0, scaler.mppt_pdc_p99)

    # IMPORTANTE: manter dimensionalidade fixa entre janelas.
    # Mesmo quando VAC/frequência não existirem para um dia/fonte,
    # as features precisam continuar presentes para evitar erro de concatenação
    # no dataset de treino (ex.: 17 vs 19 features por passo temporal).
    if vac_raw is None:
        vac_n = np.zeros(T, dtype=float)
    else:
        vac_n = scaler.n_pos(_nan_to_0(vac_raw), getattr(scaler, "vac_p99", 1000.0))

    if freq_raw is None:
        freq_n = np.zeros(T, dtype=float)
    else:
        f0 = np.asarray(freq_raw, float)
        finite_f = f0[np.isfinite(f0)]
        if finite_f.size == 0:
            freq_n = np.zeros(T, dtype=float)
        else:
            # normalização suave em torno de 50/60 Hz
            base = 60.0 if np.nanmedian(finite_f) > 55.0 else 50.0
            freq_n = np.clip((np.nan_to_num(f0, nan=base) - base) / 5.0, -1.0, 1.0)

    # peers (nanmedian/nansum)
    i_med = np.nanmedian(i_raw, axis=0)  # [T]
    v_med = np.nanmedian(v_raw, axis=0)
    p_sum = np.nansum(pdc_est_raw, axis=0)

    i_med = np.nan_to_num(i_med, nan=0.0)
    v_med = np.nan_to_num(v_med, nan=0.0)
    p_sum = np.nan_to_num(p_sum, nan=0.0)

    i_rel = i0 / (i_med[None, :] + EPS)
    v_rel = (v0 - v_med[None, :]) / (scaler.mppt_vdc_p99 + EPS)
    share_p = pdc0 / (p_sum[None, :] + EPS)

    # dinâmicas
    dpac = _diff(pac_n)
    dmm = _diff(mm_n)
    dv = np.stack([_diff(v_n[k]) for k in range(N)], axis=0)
    di = np.stack([_diff(i_n[k]) for k in range(N)], axis=0)

    feats = []
    fmap: Dict[str, int] = {}

    def add(name: str, arr: np.ndarray):
        fmap[name] = len(feats)
        feats.append(arr)

    # por nó
    add("v_mppt_n", v_n)
    add("i_mppt_n", i_n)
    add("pdc_est_n", pdc_n)
    add("i_rel", np.clip(i_rel, 0.0, 5.0))
    add("v_rel", np.clip(v_rel, -3.0, 3.0))
    add("share_p", np.clip(share_p, 0.0, 1.0))
    add("dv_mppt", np.clip(dv, -1.0, 1.0))
    add("di_mppt", np.clip(di, -1.0, 1.0))

    # globais repetidas
    add("pac_n", np.tile(pac_n[None, :], (N, 1)))
    add("vdc_total_n", np.tile(vdc_total_n[None, :], (N, 1)))
    add("iac_n", np.tile(iac_n[None, :], (N, 1)))
    add("pac_model_n", np.tile(pac_model_n[None, :], (N, 1)))
    add("mismatch_n", np.tile(mm_n[None, :], (N, 1)))
    add("g_n", np.tile(g_n[None, :], (N, 1)))
    add("t_n", np.tile(t_n[None, :], (N, 1)))
    add("dpac", np.tile(dpac[None, :], (N, 1)))
    add("dmm", np.tile(dmm[None, :], (N, 1)))

    # Sempre presentes para manter schema fixo entre todas as janelas.
    add("vac_n", np.tile(vac_n[None, :], (N, 1)))
    add("freq_n", np.tile(freq_n[None, :], (N, 1)))

    X = np.stack(feats, axis=0)     # [F,N,T]
    X = np.transpose(X, (1, 2, 0))  # [N,T,F]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), fmap
