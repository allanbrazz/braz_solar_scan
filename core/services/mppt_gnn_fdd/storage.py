# core/services/mppt_gnn_fdd/storage.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _infer_trained_at_utc(meta: Dict[str, Any], art: Optional[ModelArtifacts] = None) -> Optional[str]:
    for key in ("trained_at_utc", "trained_at", "created_at_utc", "created_at"):
        raw = meta.get(key)
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")
            except Exception:
                return str(raw)

    if art is not None and os.path.exists(art.meta_path):
        try:
            ts = os.path.getmtime(art.meta_path)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return None
    return None


def summarize_model_health(*, model_version: str, meta: Dict[str, Any], art: Optional[ModelArtifacts] = None) -> Dict[str, Any]:
    evaluation = meta.get("evaluation") or {}
    train_eval = evaluation.get("train") or {}
    val_eval = evaluation.get("validation") or {}
    split = meta.get("dataset_split") or {}

    total_samples = None
    try:
        total_samples = int((((meta.get("train_shape") or {}).get("X") or [None])[0]))
    except Exception:
        total_samples = None
    if total_samples in (None, 0):
        try:
            total_samples = int(((meta.get("dataset_stats") or {}).get("samples_total")))
        except Exception:
            total_samples = None

    validation_samples = 0
    try:
        validation_samples = int(split.get("validation_samples") or 0)
    except Exception:
        validation_samples = 0
    if validation_samples <= 0:
        try:
            validation_samples = int(val_eval.get("n_samples") or 0)
        except Exception:
            validation_samples = 0

    train_samples = 0
    try:
        train_samples = int(split.get("train_samples") or 0)
    except Exception:
        train_samples = 0
    if train_samples <= 0:
        try:
            train_samples = int(train_eval.get("n_samples") or 0)
        except Exception:
            train_samples = 0

    validation_days = split.get("validation_days") or []
    train_days = split.get("train_days") or []
    has_temporal_validation = bool(validation_samples > 0 and validation_days)

    metrics_source = "validation" if has_temporal_validation else ("train" if train_eval else "none")
    chosen = val_eval if metrics_source == "validation" else train_eval

    f1_macro = _safe_float(chosen.get("f1_macro"))
    balanced_accuracy = _safe_float(chosen.get("balanced_accuracy"))
    coverage_fraction = None
    if total_samples and validation_samples > 0:
        coverage_fraction = max(0.0, min(1.0, float(validation_samples) / float(total_samples)))

    trained_at_utc = _infer_trained_at_utc(meta, art)

    status = "recalibrar"
    status_label = "Recalibrar"
    status_tone = "crit"
    status_note = "Sem validação temporal confiável para sustentar o uso operacional do classificador."

    if metrics_source == "validation" and f1_macro is not None and balanced_accuracy is not None:
        if validation_samples < 50 or len(validation_days) < 5 or (coverage_fraction is not None and coverage_fraction < 0.10):
            status = "atencao"
            status_label = "Atenção"
            status_tone = "warn"
            status_note = "Existe validação temporal, mas a cobertura do holdout ainda é curta para alta confiança operacional."
        elif f1_macro >= 0.75 and balanced_accuracy >= 0.70 and validation_samples >= 500 and len(validation_days) >= 14:
            status = "confiavel"
            status_label = "Confiável"
            status_tone = "ok"
            status_note = "As métricas validadas estão em faixa adequada para uso operacional assistido."
        elif f1_macro >= 0.55 and balanced_accuracy >= 0.60 and validation_samples >= 300 and len(validation_days) >= 10:
            status = "atencao"
            status_label = "Atenção"
            status_tone = "warn"
            status_note = "O modelo é utilizável, mas ainda requer leitura prudente e revisão periódica das classes confundidas."
        else:
            status_note = "As métricas validadas indicam necessidade de recalibração antes de confiar no diagnóstico automático."
    elif metrics_source == "train" and train_samples > 0:
        status = "atencao"
        status_label = "Atenção"
        status_tone = "warn"
        status_note = "Só há métricas de treino/resubstituição. Falta validação temporal para confiança operacional."

    source_label = {
        "validation": "Validação temporal",
        "train": "Treino/resubstituição",
        "none": "Sem métricas",
    }.get(metrics_source, metrics_source)

    coverage_label = "—"
    if validation_samples > 0:
        parts = []
        if coverage_fraction is not None:
            parts.append(f"{coverage_fraction * 100:.1f}%")
        parts.append(f"{validation_samples} amostras")
        if validation_days:
            parts.append(f"{len(validation_days)} dias")
        coverage_label = " · ".join(parts)

    advanced = {
        "metrics_source": metrics_source,
        "metrics_source_label": source_label,
        "trained_at_utc": trained_at_utc,
        "accuracy": _safe_float(chosen.get("accuracy")),
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": _safe_float(chosen.get("f1_weighted")),
        "log_loss": _safe_float(chosen.get("log_loss")),
        "mean_confidence": _safe_float(chosen.get("mean_confidence")),
        "n_samples": int(chosen.get("n_samples") or 0) if chosen else 0,
        "labels": chosen.get("labels"),
        "target_names": chosen.get("target_names"),
        "confusion_matrix": chosen.get("confusion_matrix"),
        "confusion_matrix_normalized_true": chosen.get("confusion_matrix_normalized_true"),
        "classification_report": chosen.get("classification_report"),
        "dataset_split": {
            "strategy": split.get("strategy"),
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "train_days": train_days,
            "validation_days": validation_days,
            "train_day_start": split.get("train_day_start"),
            "train_day_end": split.get("train_day_end"),
            "validation_day_start": split.get("validation_day_start"),
            "validation_day_end": split.get("validation_day_end"),
            "coverage_fraction": coverage_fraction,
        },
    }

    return {
        "model_version": str(model_version),
        "trained_at_utc": trained_at_utc,
        "validation_f1_macro": _safe_float(val_eval.get("f1_macro")) if val_eval else None,
        "validation_balanced_accuracy": _safe_float(val_eval.get("balanced_accuracy")) if val_eval else None,
        "validation_samples": validation_samples,
        "validation_days": len(validation_days),
        "validation_coverage_fraction": coverage_fraction,
        "validation_coverage_label": coverage_label,
        "metrics_source": metrics_source,
        "metrics_source_label": source_label,
        "status": status,
        "status_label": status_label,
        "status_tone": status_tone,
        "status_note": status_note,
        "advanced_available": bool(chosen),
        "advanced": advanced,
    }




def list_available_model_versions() -> list[str]:
    base = _base_dir()
    if not os.path.isdir(base):
        return []
    out: list[str] = []
    try:
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if not os.path.isdir(d):
                continue
            if any(os.path.exists(os.path.join(d, fn)) for fn in ("meta.json", "model.joblib", "scaler.json")):
                out.append(str(name))
    except Exception:
        return []
    return out

def load_model_health(*, model_version: str) -> Optional[Dict[str, Any]]:
    if not model_version:
        return None
    art = artifacts_for(model_version)
    if not os.path.exists(art.meta_path):
        return None
    try:
        with open(art.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return summarize_model_health(model_version=model_version, meta=meta, art=art)
