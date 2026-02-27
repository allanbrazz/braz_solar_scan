from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from django.conf import settings


@dataclass(frozen=True)
class MpptGnnPaths:
    base_dir: Path
    models_dir: Path


def get_mppt_gnn_paths() -> MpptGnnPaths:
    base = getattr(settings, "MPPT_GNN_FDD_DIR", None)
    models = getattr(settings, "MPPT_GNN_FDD_MODELS_DIR", None)

    if base is None or models is None:
        base = Path.cwd() / "mppt_gnn_fdd"
        models = base / "models"

    base = Path(base)
    models = Path(models)
    base.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return MpptGnnPaths(base_dir=base, models_dir=models)


def model_path(name: str, *, suffix: str = ".pt") -> Path:
    p = get_mppt_gnn_paths().models_dir / f"{name}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p