from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from core.services.mppt_gnn_fdd.config import MpptNominal


def build_edge_attr(nom: Dict[int, MpptNominal], *, nmax: int = 4) -> np.ndarray:
    """
    E_attr: [Nmax,Nmax,Fe]
    Fe=4: [dtilt, daz, dns, dnp]  (por enquanto tilt/az = 0 intra-inversor)
    """
    Fe = 4
    E = np.zeros((nmax, nmax, Fe), dtype=np.float32)

    def _safe_norm_delta(a: float, b: float) -> float:
        den = max(abs(a), abs(b), 1.0)
        return float(a - b) / den

    for i in range(1, nmax + 1):
        for j in range(1, nmax + 1):
            if i == j:
                continue
            ni = nom.get(i)
            nj = nom.get(j)
            if ni is None or nj is None:
                continue

            dtilt = 0.0
            daz = 0.0
            dns = _safe_norm_delta(float(ni.n_series), float(nj.n_series))
            dnp = _safe_norm_delta(float(ni.n_parallel), float(nj.n_parallel))

            E[i - 1, j - 1, :] = (dtilt, daz, dns, dnp)

    return E