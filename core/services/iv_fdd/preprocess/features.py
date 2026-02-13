from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    # Whether to normalize V to [0,1] (V/Voc) and I to (I/Isc).
    normalize: bool = True

    # Small epsilon for divisions
    eps: float = 1e-12


def build_input_channels(
    V: np.ndarray,
    I: np.ndarray,
    *,
    cfg: FeatureConfig | None = None,
) -> np.ndarray:
    """Build 4 input channels used in Hopwood-style 1D-CNN.

    Inputs:
      V: (N,M) voltages (monotonic increasing)
      I: (N,M) currents

    Outputs:
      X: (N,M,4) channels-last layout
        [:,:,0] = iota = I/Isc
        [:,:,1] = P = V*I (optionally normalized by Voc*Isc)
        [:,:,2] = dI = forward difference of iota (padded with 0 at first)
        [:,:,3] = dIdV = finite difference of iota w.r.t normalized V (padded)

    This matches your summary:
      - Corrente normalizada (ι)
      - Potência (P)
      - Diferencial de corrente (ΔI)
      - Diferença finita (δI)

    Note: Hopwood do STC translation first; here we assume the dataset already
    has consistent V grid (fixed points). You can add IEC 60891 stage later.
    """
    cfg = cfg or FeatureConfig()
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    if V.shape != I.shape or V.ndim != 2:
        raise ValueError("V and I must be 2D arrays with same shape (N,M).")

    N, M = V.shape

    # Normalize
    if cfg.normalize:
        Isc = np.maximum(I[:, 0], cfg.eps)
        Voc = np.maximum(V[:, -1], cfg.eps)
        iota = I / Isc[:, None]
        Vn = V / Voc[:, None]
        P = (V * I) / (Voc * Isc)[:, None]
    else:
        iota = I.copy()
        Vn = V.copy()
        P = V * I

    # dI: forward difference of iota
    dI = np.zeros_like(iota)
    dI[:, 1:] = iota[:, 1:] - iota[:, :-1]

    # dIdV: finite difference w.r.t. Vn
    dV = np.zeros_like(Vn)
    dV[:, 1:] = Vn[:, 1:] - Vn[:, :-1]
    dIdV = np.zeros_like(iota)
    dIdV[:, 1:] = dI[:, 1:] / np.maximum(dV[:, 1:], cfg.eps)

    X = np.stack([iota, P, dI, dIdV], axis=-1)  # (N,M,4)
    return X.astype(np.float32)


def save_features_npz(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=np.asarray(y, dtype=np.int64),
        class_names=np.array(class_names, dtype=object),
    )
    return path


def load_features_npz(path: Path | str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        X = z["X"]
        y = z["y"]
        class_names = list(z["class_names"].tolist())
    return X, y, class_names
