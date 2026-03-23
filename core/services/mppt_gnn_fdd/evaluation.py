from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize_multiclass_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[int],
    target_names: Optional[Iterable[str]] = None,
    proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    labels_list = [int(v) for v in labels]
    names_list = [str(v) for v in (target_names or labels_list)]

    if y_true.size == 0:
        return {
            "n_samples": 0,
            "labels": labels_list,
            "target_names": names_list,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_macro": None,
            "f1_weighted": None,
            "confusion_matrix": None,
            "confusion_matrix_normalized_true": None,
            "classification_report": None,
            "log_loss": None,
            "mean_confidence": None,
        }

    metrics: Dict[str, Any] = {
        "n_samples": int(y_true.size),
        "labels": labels_list,
        "target_names": names_list,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels_list, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_list).tolist(),
        "confusion_matrix_normalized_true": confusion_matrix(
            y_true, y_pred, labels=labels_list, normalize="true"
        ).tolist(),
        "classification_report": _to_python(
            classification_report(
                y_true,
                y_pred,
                labels=labels_list,
                target_names=names_list,
                zero_division=0,
                output_dict=True,
            )
        ),
    }

    if proba is not None and getattr(proba, "size", 0):
        try:
            metrics["log_loss"] = float(log_loss(y_true, proba, labels=labels_list))
        except Exception:
            metrics["log_loss"] = None
        try:
            metrics["mean_confidence"] = float(np.nanmean(np.nanmax(np.asarray(proba, dtype=float), axis=1)))
        except Exception:
            metrics["mean_confidence"] = None
    else:
        metrics["log_loss"] = None
        metrics["mean_confidence"] = None

    return _to_python(metrics)
