from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from django.conf import settings


@dataclass(frozen=True)
class IvFddPaths:
    base_dir: Path
    datasets_dir: Path
    models_dir: Path


def get_iv_fdd_paths() -> IvFddPaths:
    """Resolve writable paths for IV-FDD artifacts.

    Important for your project:
      - When frozen via PyInstaller, settings.BASE_DIR may point to _MEIPASS (read-only).
      - The repository already stores the SQLite DB in a user-writable data dir.
      - We mirror that strategy here via settings.IV_FDD_* constants.
    """
    base = getattr(settings, "IV_FDD_DIR", None)
    datasets = getattr(settings, "IV_FDD_DATASETS_DIR", None)
    models = getattr(settings, "IV_FDD_MODELS_DIR", None)

    if base is None or datasets is None or models is None:
        # fallback: relative to current working directory (development only)
        base = Path.cwd() / "iv_fdd"
        datasets = base / "datasets"
        models = base / "models"

    base = Path(base)
    datasets = Path(datasets)
    models = Path(models)

    base.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    return IvFddPaths(base_dir=base, datasets_dir=datasets, models_dir=models)


def dataset_path(name: str, *, suffix: str = ".npz") -> Path:
    p = get_iv_fdd_paths().datasets_dir / f"{name}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def model_path(name: str, *, suffix: str = ".pt") -> Path:
    p = get_iv_fdd_paths().models_dir / f"{name}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
