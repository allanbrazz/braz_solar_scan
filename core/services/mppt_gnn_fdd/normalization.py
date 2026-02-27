# core/services/mppt_gnn_fdd/normalization.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

from core.services.mppt_gnn_fdd.constants import EPS


def _p99(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return float(np.nanpercentile(x, 99))


@dataclass
class PlantScaler:
    """
    Escalas robustas por planta (p99) para invariância por potência/tensão.
    - globais: pac, vdc_total, iac, pac_model, G, T
    - por MPPT: vdc_mppt, idc_mppt, pdc_mppt_est (opcional)
    """
    pac_p99: float = 1.0
    vdc_total_p99: float = 1.0
    iac_p99: float = 1.0
    pac_model_p99: float = 1.0
    g_p99: float = 1000.0
    t_p99: float = 50.0

    mppt_vdc_p99: float = 1.0
    mppt_idc_p99: float = 1.0
    mppt_pdc_p99: float = 1.0  # v*i

    @classmethod
    def fit_from_arrays(
        cls,
        *,
        pac: np.ndarray,
        vdc_total: np.ndarray,
        iac: np.ndarray,
        pac_model: np.ndarray,
        g: np.ndarray,
        t: np.ndarray,
        mppt_vdc: np.ndarray,   # shape [N,T]
        mppt_idc: np.ndarray,   # shape [N,T]
    ) -> "PlantScaler":
        pac = np.asarray(pac, float)
        vdc_total = np.asarray(vdc_total, float)
        iac = np.asarray(iac, float)
        pac_model = np.asarray(pac_model, float)
        g = np.asarray(g, float)
        t = np.asarray(t, float)
        mv = np.asarray(mppt_vdc, float)
        mi = np.asarray(mppt_idc, float)

        pdc = mv * mi

        return cls(
            pac_p99=max(_p99(pac), 1.0),
            vdc_total_p99=max(_p99(vdc_total), 1.0),
            iac_p99=max(_p99(iac), 0.1),
            pac_model_p99=max(_p99(pac_model), 1.0),
            g_p99=max(_p99(g), 50.0),
            t_p99=max(_p99(t), 10.0),
            mppt_vdc_p99=max(_p99(mv.ravel()), 1.0),
            mppt_idc_p99=max(_p99(mi.ravel()), 0.1),
            mppt_pdc_p99=max(_p99(pdc.ravel()), 1.0),
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "PlantScaler":
        return cls(**{k: float(v) for k, v in d.items()})

    def n(self, x: np.ndarray, s: float, clip: float = 3.0) -> np.ndarray:
        x = np.asarray(x, float)
        y = x / max(float(s), EPS)
        return np.clip(y, -clip, clip)

    def n_pos(self, x: np.ndarray, s: float, clip: float = 3.0) -> np.ndarray:
        x = np.asarray(x, float)
        y = x / max(float(s), EPS)
        return np.clip(y, 0.0, clip)

    def mismatch_clip(self, mm: np.ndarray, clip: float = 2.0) -> np.ndarray:
        mm = np.asarray(mm, float)
        return np.clip(mm, -clip, clip) / clip  # [-1..1]