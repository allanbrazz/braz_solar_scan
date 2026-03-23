# core/services/mppt_gnn_fdd/train_pipeline.py
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict

import numpy as np

from core.services.mppt_gnn_fdd.dataset import build_training_dataset, DatasetConfig
from core.services.mppt_gnn_fdd.evaluation import summarize_multiclass_metrics
from core.services.mppt_gnn_fdd.model_sklearn import (
    build_balanced_sample_weight,
    train_mlp_classifier,
    SklearnModelConfig,
)
from core.services.mppt_gnn_fdd.storage import save_artifacts


def _build_temporal_split(day_local: list[str], holdout_fraction: float, min_holdout_days: int) -> Dict[str, Any]:
    unique_days = sorted({str(d) for d in day_local if str(d).strip()})
    if len(unique_days) < 2:
        return {
            "strategy": "resubstitution_only",
            "train_days": unique_days,
            "validation_days": [],
            "train_mask": np.ones((len(day_local),), dtype=bool),
            "validation_mask": np.zeros((len(day_local),), dtype=bool),
        }

    holdout_n = max(int(round(len(unique_days) * float(holdout_fraction))), int(min_holdout_days))
    holdout_n = max(1, min(holdout_n, len(unique_days) - 1))

    val_days = unique_days[-holdout_n:]
    train_days = unique_days[:-holdout_n]
    val_set = set(val_days)

    train_mask = np.asarray([str(d) not in val_set for d in day_local], dtype=bool)
    val_mask = ~train_mask

    if not train_mask.any() or not val_mask.any():
        return {
            "strategy": "resubstitution_only",
            "train_days": unique_days,
            "validation_days": [],
            "train_mask": np.ones((len(day_local),), dtype=bool),
            "validation_mask": np.zeros((len(day_local),), dtype=bool),
        }

    return {
        "strategy": "temporal_holdout_by_day",
        "train_days": train_days,
        "validation_days": val_days,
        "train_mask": train_mask,
        "validation_mask": val_mask,
    }


def train_mppt_gnn_sklearn(
    *,
    plant_id: int,
    start: date,
    end: date,
    model_version: str,
    ds_cfg: DatasetConfig = DatasetConfig(),
    mlp_cfg: SklearnModelConfig = SklearnModelConfig(),
) -> Dict[str, Any]:
    X, y, scaler, fmap, stats, sample_meta = build_training_dataset(
        plant_id=plant_id,
        start=start,
        end=end,
        cfg=ds_cfg,
    )

    day_local = [str(v) for v in sample_meta.get("day_local", [])]
    split = _build_temporal_split(
        day_local,
        holdout_fraction=float(mlp_cfg.temporal_validation_fraction),
        min_holdout_days=int(mlp_cfg.min_temporal_validation_days),
    )

    train_mask = split["train_mask"]
    validation_mask = split["validation_mask"]

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[validation_mask] if validation_mask.any() else np.empty((0, X.shape[1]), dtype=X.dtype)
    y_val = y[validation_mask] if validation_mask.any() else np.empty((0,), dtype=y.dtype)

    train_sample_weight = None
    class_weight = None
    if bool(mlp_cfg.use_balanced_sample_weight):
        train_sample_weight, class_weight = build_balanced_sample_weight(y_train)

    clf, info = train_mlp_classifier(X_train, y_train, cfg=mlp_cfg, sample_weight=train_sample_weight)

    labels = [int(v) for v in info.get("classes", [])]
    target_names = [str(v) for v in labels]

    yhat_train = clf.predict(X_train)
    proba_train = clf.predict_proba(X_train) if hasattr(clf, "predict_proba") else None
    eval_train = summarize_multiclass_metrics(
        y_true=y_train,
        y_pred=yhat_train,
        labels=labels,
        target_names=target_names,
        proba=proba_train,
    )

    if X_val.shape[0] > 0:
        yhat_val = clf.predict(X_val)
        proba_val = clf.predict_proba(X_val) if hasattr(clf, "predict_proba") else None
        eval_val = summarize_multiclass_metrics(
            y_true=y_val,
            y_pred=yhat_val,
            labels=labels,
            target_names=target_names,
            proba=proba_val,
        )
    else:
        eval_val = summarize_multiclass_metrics(
            y_true=np.empty((0,), dtype=int),
            y_pred=np.empty((0,), dtype=int),
            labels=labels,
            target_names=target_names,
            proba=None,
        )

    split_meta = {
        "strategy": split["strategy"],
        "train_days": split["train_days"],
        "validation_days": split["validation_days"],
        "train_day_start": split["train_days"][0] if split["train_days"] else None,
        "train_day_end": split["train_days"][-1] if split["train_days"] else None,
        "validation_day_start": split["validation_days"][0] if split["validation_days"] else None,
        "validation_day_end": split["validation_days"][-1] if split["validation_days"] else None,
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "temporal_validation_fraction": float(mlp_cfg.temporal_validation_fraction),
        "min_temporal_validation_days": int(mlp_cfg.min_temporal_validation_days),
    }

    trained_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    meta: Dict[str, Any] = {
        "plant_id": plant_id,
        "trained_at_utc": trained_at_utc,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "model_version": model_version,
        "dataset": ds_cfg.__dict__,
        "dataset_stats": stats,
        "mlp": mlp_cfg.__dict__,
        "feature_map": fmap,
        "train_shape": {"X": list(X.shape), "y": list(y.shape)},
        "train_split_shape": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_validation": list(X_val.shape),
            "y_validation": list(y_val.shape),
        },
        "classes": info.get("classes"),
        "class_weight": class_weight,
        "dataset_split": split_meta,
        "evaluation": {
            "train": eval_train,
            "validation": eval_val,
        },
        "fit_runtime": {
            "fit_supports_sample_weight": bool(info.get("fit_supports_sample_weight")),
            "sample_weight_used_in_fit": bool(info.get("sample_weight_used_in_fit")),
        },
        "note": (
            "Modelo sklearn baseline (flatten 24h) + fault injection com holdout temporal por dia. "
            "A avaliação salva matriz de confusão, balanced accuracy, macro-F1, weighted-F1 e relatório por classe."
        ),
    }

    art = save_artifacts(model=clf, scaler=scaler, meta=meta, model_version=model_version)

    return {
        "ok": True,
        "plant_id": plant_id,
        "model_version": model_version,
        "saved": {"model": art.model_path, "scaler": art.scaler_path, "meta": art.meta_path},
        "train_shape": meta["train_shape"],
        "train_split_shape": meta["train_split_shape"],
        "classes": meta["classes"],
        "dataset_stats": stats,
        "dataset_split": split_meta,
        "evaluation": {
            "train": {
                "accuracy": eval_train.get("accuracy"),
                "balanced_accuracy": eval_train.get("balanced_accuracy"),
                "f1_macro": eval_train.get("f1_macro"),
                "f1_weighted": eval_train.get("f1_weighted"),
            },
            "validation": {
                "accuracy": eval_val.get("accuracy"),
                "balanced_accuracy": eval_val.get("balanced_accuracy"),
                "f1_macro": eval_val.get("f1_macro"),
                "f1_weighted": eval_val.get("f1_weighted"),
            },
        },
    }
