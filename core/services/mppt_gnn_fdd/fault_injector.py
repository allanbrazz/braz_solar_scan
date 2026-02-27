# core/services/mppt_gnn_fdd/fault_injector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from core.services.mppt_gnn_fdd.features import WindowArrays
from core.services.mppt_gnn_fdd.constants import EPS


@dataclass
class InjectConfig:
    g_gate: float = 300.0  # atua apenas sob "sol"
    # mppt disconnected
    disc_i_mult: float = 0.02
    disc_v_mult: float = 1.10

    # mppt imbalance
    imb_i_mult: float = 0.35

    # inverter off
    off_pac_mult: float = 0.01
    off_iac_mult: float = 0.01

    # curtailment
    curtail_p_limit_pu: float = 0.65

    # meteo bias
    meteo_g_mult: float = 1.35


def _day_mask(g: np.ndarray, gate: float) -> np.ndarray:
    g = np.asarray(g, float)
    return np.isfinite(g) & (g >= float(gate))


def inject_fault(
    win: WindowArrays,
    *,
    fault_code: int,
    mppt_k: Optional[int],
    rng: np.random.Generator,
    cfg: InjectConfig = InjectConfig(),
) -> Tuple[WindowArrays, np.ndarray]:
    """
    Retorna (win_injected, y_codes_per_mppt[N])
    """
    N, T = win.mppt_vdc.shape
    y = np.zeros(N, dtype=int)

    # copia
    pac = win.pac.copy()
    vdc_total = win.vdc_total.copy()
    iac = win.iac.copy()
    pac_model = win.pac_model.copy()
    mm = win.mismatch.copy()
    g = win.g.copy()
    t = win.t.copy()
    v = win.mppt_vdc.copy()
    i = win.mppt_idc.copy()

    mask = _day_mask(g, cfg.g_gate)

    # helper mismatch
    def recompute_mismatch():
        den = np.maximum(np.abs(pac_model), 50.0)
        return (pac - pac_model) / den

    if fault_code == 2:
        # inverter_off_under_sun (afeta todos)
        pac[mask] = pac[mask] * cfg.off_pac_mult
        iac[mask] = iac[mask] * cfg.off_iac_mult
        # opcional: vdc_total pode cair um pouco (standby)
        vdc_total[mask] = vdc_total[mask] * (0.85 + 0.10 * rng.random())
        # mppts: corrente cai
        i[:, mask] = i[:, mask] * cfg.off_iac_mult
        y[:] = 2

    elif fault_code == 4:
        # curtailment/clipping-like (afeta todos): pac saturado
        # usa limite em pu do p99 de pac_model na janela
        pm = pac_model[mask]
        p_ref = float(np.nanpercentile(pm[np.isfinite(pm)], 95)) if np.isfinite(pm).any() else float(np.nanmax(pac_model))
        if not np.isfinite(p_ref) or p_ref <= 0:
            p_ref = 1000.0
        p_lim = cfg.curtail_p_limit_pu * p_ref
        pac[mask] = np.minimum(pac[mask], p_lim)
        y[:] = 4

    elif fault_code == 5:
        # meteo_bias (afeta todos): G inflado/deflacionado sem alterar pac
        g[mask] = g[mask] * cfg.meteo_g_mult
        # pac_model "aprox" escala com G
        pac_model[mask] = pac_model[mask] * cfg.meteo_g_mult
        y[:] = 5

    elif fault_code == 1:
        # mppt_disconnected em um MPPT
        k = int(mppt_k or 1)
        k = max(1, min(N, k)) - 1
        i[k, mask] = i[k, mask] * cfg.disc_i_mult
        v[k, mask] = v[k, mask] * cfg.disc_v_mult
        y[k] = 1

        # pac reduz proporcionalmente (aprox por pdc_est)
        pdc = v * i
        p_sum = np.sum(pdc, axis=0)
        pac[mask] = np.minimum(pac[mask], p_sum[mask])  # coerência fraca
        y[np.arange(N) != k] = 0

    elif fault_code == 3:
        # mppt_imbalance em um MPPT (parcial)
        k = int(mppt_k or 1)
        k = max(1, min(N, k)) - 1
        # aplica num "bloco" central do dia para simular sombra parcial
        idx = np.where(mask)[0]
        if idx.size > 10:
            a = int(idx.size * (0.35 + 0.10 * rng.random()))
            b = int(idx.size * (0.70 + 0.10 * rng.random()))
            sel = idx[a:b]
        else:
            sel = idx
        i[k, sel] = i[k, sel] * cfg.imb_i_mult
        y[k] = 3
        y[np.arange(N) != k] = 0

        pdc = v * i
        p_sum = np.sum(pdc, axis=0)
        pac[mask] = np.minimum(pac[mask], p_sum[mask])

    else:
        # normal
        y[:] = 0

    mm = recompute_mismatch()

    win2 = WindowArrays(
        pac=pac,
        vdc_total=vdc_total,
        iac=iac,
        pac_model=pac_model,
        mismatch=mm,
        g=g,
        t=t,
        mppt_vdc=v,
        mppt_idc=i,
    )
    return win2, y