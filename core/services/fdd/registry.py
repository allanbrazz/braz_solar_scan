# =================================
# core/services/registry.py
# Central registry para bundles RF (cache por mtime) + helper de inferência
# =================================
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .features import RFFeatureSpec
from .rf_inference import InferenceConfig, load_rf_bundle, run_rf_inference


# -----------------------------
# Model spec
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    """
    Especificação mínima de um modelo RF "bundle".

    - name: identificador lógico (ex.: "default", "plant_12", "v1_2026_02")
    - bundle_path: caminho do .joblib gerado pelo rf_train.py
    - enabled: permite desativar sem remover
    - meta: metadados adicionais (plant_id, inverter_id, versão, etc.)
    - feature_spec: (opcional) especificação de features usada no treino (dt, janelas rolling, flags etc.)
      OBS: se você treina com um spec e infere com outro, pode degradar performance.
    """
    name: str
    bundle_path: Union[str, Path]
    enabled: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)
    feature_spec: Optional[RFFeatureSpec] = None

    def resolved_path(self, base_dir: Optional[Union[str, Path]] = None) -> Path:
        p = Path(self.bundle_path)
        if p.is_absolute():
            return p
        if base_dir is not None:
            return Path(base_dir) / p
        return p


@dataclass
class LoadedBundle:
    bundle: Dict[str, Any]
    mtime_ns: int


# -----------------------------
# Registry
# -----------------------------
class RfModelRegistry:
    """
    Registry com:
      - registro de specs
      - cache de bundle por mtime (recarrega se o arquivo mudar)
      - helpers para rodar inferência com out_model do power_model.expected_and_mismatch()
    """

    def __init__(self, *, base_dir: Optional[Union[str, Path]] = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else None
        self._specs: Dict[str, ModelSpec] = {}
        self._cache: Dict[str, LoadedBundle] = {}
        self._lock = threading.RLock()

    # ---------
    # specs
    # ---------
    @property
    def base_dir(self) -> Optional[Path]:
        return self._base_dir

    def set_base_dir(self, base_dir: Union[str, Path, None]) -> None:
        with self._lock:
            self._base_dir = Path(base_dir) if base_dir is not None else None
            # paths relativos mudam => invalida cache
            self._cache.clear()

    def register(self, spec: ModelSpec, *, overwrite: bool = True) -> None:
        name = (spec.name or "").strip()
        if not name:
            raise ValueError("ModelSpec.name vazio.")
        with self._lock:
            if (not overwrite) and (name in self._specs):
                raise ValueError(f"Modelo '{name}' já registrado.")
            self._specs[name] = spec
            # se você sobrescrever, invalida cache desse modelo
            self._cache.pop(name, None)

    def unregister(self, name: str) -> None:
        key = (name or "").strip()
        with self._lock:
            self._specs.pop(key, None)
            self._cache.pop(key, None)

    def get_spec(self, name: str) -> ModelSpec:
        key = (name or "").strip()
        if not key:
            raise ValueError("get_spec: name vazio.")
        with self._lock:
            spec = self._specs.get(key)
            if spec is None:
                raise KeyError(f"Modelo '{key}' não está registrado.")
            return spec

    def list_models(self) -> Dict[str, ModelSpec]:
        with self._lock:
            return dict(self._specs)

    # ---------
    # loading / cache
    # ---------
    @staticmethod
    def _file_mtime_ns(p: Path) -> int:
        try:
            return int(p.stat().st_mtime_ns)
        except Exception:
            return -1

    def load_bundle(self, name: str, *, force_reload: bool = False) -> Dict[str, Any]:
        """
        Carrega bundle (.joblib) com cache por mtime.
        Se arquivo mudou, recarrega automaticamente.

        IMPORTANTE:
        - Evita deadlock chamando get_spec() dentro do mesmo lock (RLock ajuda,
          mas mantemos consistente e simples).
        """
        key = (name or "").strip()
        if not key:
            raise ValueError("load_bundle: name vazio.")

        with self._lock:
            spec = self._specs.get(key)
            if spec is None:
                raise KeyError(f"Modelo '{key}' não está registrado.")
            if not spec.enabled:
                raise ValueError(f"Modelo '{key}' está desabilitado.")

            path = spec.resolved_path(self._base_dir)
            if not path.exists():
                raise FileNotFoundError(f"Bundle não encontrado: {path}")

            mtime = self._file_mtime_ns(path)

            if (not force_reload) and (key in self._cache):
                cached = self._cache[key]
                if cached.mtime_ns == mtime and cached.mtime_ns != -1:
                    return cached.bundle

            bundle = load_rf_bundle(path)
            self._cache[key] = LoadedBundle(bundle=bundle, mtime_ns=mtime)
            return bundle

    def invalidate_cache(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop((name or "").strip(), None)

    # ---------
    # inference
    # ---------
    def infer(
        self,
        *,
        model_name: str,
        out_model: Dict[str, Any],
        times_utc: Optional[Any] = None,
        cfg: Optional[InferenceConfig] = None,
        feature_names: Optional[Any] = None,
        feature_spec: Optional[RFFeatureSpec] = None,
        force_reload_bundle: bool = False,
    ) -> Dict[str, Any]:
        """
        Roda inferência RF + pós-processamento.
        out_model: saída do expected_and_mismatch()

        feature_spec: se None, tenta usar spec.feature_spec (registrado), senão default do rf_inference.
        """
        key = (model_name or "").strip()
        if not key:
            raise ValueError("infer: model_name vazio.")

        spec = self.get_spec(key)
        bundle = self.load_bundle(key, force_reload=force_reload_bundle)

        eff_feature_spec = feature_spec or spec.feature_spec

        return run_rf_inference(
            bundle=bundle,
            out_model=out_model,
            times_utc=times_utc,
            feature_names=feature_names,
            feature_spec=eff_feature_spec,
            cfg=cfg,
        )


# -----------------------------
# Default singleton
# -----------------------------
_DEFAULT_REGISTRY: Optional[RfModelRegistry] = None
_DEFAULT_LOCK = threading.RLock()


def get_default_registry() -> RfModelRegistry:
    """
    Registry singleton:
      - base_dir vem de env RF_MODELS_DIR (opcional)
    """
    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            base_dir = os.getenv("RF_MODELS_DIR", "").strip() or None
            _DEFAULT_REGISTRY = RfModelRegistry(base_dir=base_dir)
        return _DEFAULT_REGISTRY


# -----------------------------
# Helpers de bootstrap
# -----------------------------
def bootstrap_default_model(
    *,
    model_name: str = "default",
    bundle_path: Union[str, Path] = "rf_bundle.joblib",
    enabled: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    feature_spec: Optional[RFFeatureSpec] = None,
) -> RfModelRegistry:
    """
    Atalho para registrar um modelo "default" rapidamente.
    - bundle_path pode ser relativo ao RF_MODELS_DIR.
    - feature_spec: espec da engenharia de features usada no treino.
    """
    reg = get_default_registry()
    reg.register(
        ModelSpec(
            name=model_name,
            bundle_path=bundle_path,
            enabled=enabled,
            meta=meta or {},
            feature_spec=feature_spec,
        ),
        overwrite=True,
    )
    return reg
