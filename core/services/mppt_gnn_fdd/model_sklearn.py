# core/services/mppt_gnn_fdd/model_sklearn.py
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, Optional

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class SklearnModelConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 128)
    alpha: float = 1e-4
    max_iter: int = 60
    early_stopping: bool = True
    validation_fraction: float = 0.15
    n_iter_no_change: int = 8
    random_state: int = 42
    temporal_validation_fraction: float = 0.20
    min_temporal_validation_days: int = 7
    use_balanced_sample_weight: bool = True


def build_balanced_sample_weight(y: np.ndarray) -> tuple[np.ndarray, Dict[int, float]]:
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    sample_weight = np.asarray([class_weight.get(int(v), 1.0) for v in y], dtype=float)
    return sample_weight, class_weight


def train_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cfg: SklearnModelConfig = SklearnModelConfig(),
    sample_weight: Optional[np.ndarray] = None,
):
    classes = np.unique(y)
    fit_supports_sample_weight = "sample_weight" in inspect.signature(MLPClassifier.fit).parameters

    clf = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=float(cfg.alpha),
        max_iter=int(cfg.max_iter),
        early_stopping=bool(cfg.early_stopping),
        validation_fraction=float(cfg.validation_fraction),
        n_iter_no_change=int(cfg.n_iter_no_change),
        random_state=int(cfg.random_state),
        verbose=False,
    )

    used_sample_weight = bool(
        sample_weight is not None
        and bool(getattr(sample_weight, "size", 0))
        and fit_supports_sample_weight
    )

    if used_sample_weight:
        clf.fit(X, y, sample_weight=sample_weight)
    else:
        clf.fit(X, y)

    return clf, {
        "classes": classes.tolist(),
        "fit_supports_sample_weight": bool(fit_supports_sample_weight),
        "sample_weight_used_in_fit": bool(used_sample_weight),
    }
