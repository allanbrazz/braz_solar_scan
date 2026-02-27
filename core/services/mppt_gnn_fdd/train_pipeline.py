# core/services/mppt_gnn_fdd/train_pipeline.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict

from core.services.mppt_gnn_fdd.dataset import build_training_dataset, DatasetConfig
from core.services.mppt_gnn_fdd.model_sklearn import train_mlp_classifier, SklearnModelConfig
from core.services.mppt_gnn_fdd.storage import save_artifacts


def train_mppt_gnn_sklearn(
    *,
    plant_id: int,
    start: date,
    end: date,
    model_version: str,
    ds_cfg: DatasetConfig = DatasetConfig(),
    mlp_cfg: SklearnModelConfig = SklearnModelConfig(),
) -> Dict[str, Any]:
    """
    Treino offline (deployável) do baseline sklearn:
      - seleciona dias base (normal/usable) com fallback
      - fit scaler robusto (p99) por planta
      - augmentation via fault injection
      - treina MLPClassifier
      - salva artifacts em MEDIA_ROOT/mppt_gnn_models/<model_version>/
    """
    # build_training_dataset agora retorna também stats_out
    X, y, scaler, fmap, stats = build_training_dataset(
        plant_id=plant_id,
        start=start,
        end=end,
        cfg=ds_cfg,
    )

    clf, info = train_mlp_classifier(X, y, cfg=mlp_cfg)

    meta: Dict[str, Any] = {
        "plant_id": plant_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "model_version": model_version,
        "dataset": ds_cfg.__dict__,
        "dataset_stats": stats,  # ✅ importante para debug no UI/log
        "mlp": mlp_cfg.__dict__,
        "feature_map": fmap,
        "train_shape": {"X": list(X.shape), "y": list(y.shape)},
        "classes": info.get("classes"),
        "note": (
            "Modelo sklearn baseline (flatten 24h) + fault injection. "
            "Dataset builder é robusto: se não houver dias 'normais', usa dias 'utilizáveis' como base."
        ),
    }

    art = save_artifacts(model=clf, scaler=scaler, meta=meta, model_version=model_version)

    return {
        "ok": True,
        "plant_id": plant_id,
        "model_version": model_version,
        "saved": {"model": art.model_path, "scaler": art.scaler_path, "meta": art.meta_path},
        "train_shape": meta["train_shape"],
        "classes": meta["classes"],
        "dataset_stats": stats,
    }