# core/services/mppt_gnn_fdd/storage.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from django.conf import settings
import joblib

from core.services.mppt_gnn_fdd.normalization import PlantScaler


@dataclass
class ModelArtifacts:
    model_path: str
    scaler_path: str
    meta_path: str


def _base_dir() -> str:
    root = getattr(settings, "MEDIA_ROOT", None) or os.path.join(getattr(settings, "BASE_DIR", "."), "media")
    return os.path.join(root, "mppt_gnn_models")


def artifacts_for(model_version: str) -> ModelArtifacts:
    d = os.path.join(_base_dir(), str(model_version))
    os.makedirs(d, exist_ok=True)
    return ModelArtifacts(
        model_path=os.path.join(d, "model.joblib"),
        scaler_path=os.path.join(d, "scaler.json"),
        meta_path=os.path.join(d, "meta.json"),
    )


def save_artifacts(*, model: Any, scaler: PlantScaler, meta: Dict[str, Any], model_version: str) -> ModelArtifacts:
    art = artifacts_for(model_version)
    joblib.dump(model, art.model_path)

    with open(art.scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_json(), f, ensure_ascii=False, indent=2)

    with open(art.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return art


def load_artifacts(*, model_version: str):
    art = artifacts_for(model_version)

    if not os.path.exists(art.model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {art.model_path}")

    model = joblib.load(art.model_path)

    with open(art.scaler_path, "r", encoding="utf-8") as f:
        scaler = PlantScaler.from_json(json.load(f))

    meta: Dict[str, Any] = {}
    if os.path.exists(art.meta_path):
        with open(art.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, scaler, meta, art