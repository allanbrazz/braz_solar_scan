# core/services/mppt_gnn_fdd/model_sklearn.py
from __future__ import annotations

from dataclasses import dataclass
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


def train_mlp_classifier(X: np.ndarray, y: np.ndarray, cfg: SklearnModelConfig = SklearnModelConfig()):
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

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
    clf.fit(X, y)
    return clf, {"class_weight": class_weight, "classes": classes.tolist()}