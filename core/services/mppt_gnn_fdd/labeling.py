from __future__ import annotations

import numpy as np
from typing import Dict, Optional

IGNORE = -100


def weak_label_disconnected(
    *,
    gpoa_wm2: np.ndarray,          # [T]
    i_norm_by_mppt: np.ndarray,    # [N,T]
    mask: np.ndarray,              # [N]
    g_gate: float,
    i_disc: float,
    i_peer: float,
    min_sun_points: int,
    min_peer_points: int,
) -> np.ndarray:
    """
    y: [N] com {0 normal, 1 disconnected} ou IGNORE.
    Regra conservadora:
      - precisa ter "sol" suficiente na janela
      - disconnected se Idc_norm ~ 0 com sol, e existir peer ativo com sol
    """
    N, T = i_norm_by_mppt.shape
    y = np.full((N,), IGNORE, dtype=np.int64)

    in_sun = np.isfinite(gpoa_wm2) & (gpoa_wm2 >= g_gate)
    if int(in_sun.sum()) < int(min_sun_points):
        return y  # janela noturna -> não treina

    for i in range(N):
        if not bool(mask[i]):
            continue
        ii = i_norm_by_mppt[i]
        ok_i = np.isfinite(ii) & in_sun
        if ok_i.sum() < min_sun_points:
            continue

        frac_disc = float((ii[ok_i] <= i_disc).mean())

        # peer ativo?
        peer_active = False
        for j in range(N):
            if j == i or (not bool(mask[j])):
                continue
            jj = i_norm_by_mppt[j]
            ok_j = np.isfinite(jj) & in_sun
            if ok_j.sum() < min_peer_points:
                continue
            if float((jj[ok_j] >= i_peer).mean()) >= 0.5:
                peer_active = True
                break

        if peer_active and frac_disc >= 0.7:
            y[i] = 1
        else:
            y[i] = 0

    return y