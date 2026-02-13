from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from ..configs.hopwood_defaults import DEFAULT_FAILURE_CLASSES
from .sampling import TruncNormSpec, parse_truncnorm_spec


@dataclass(frozen=True)
class FailureClass:
    name: str
    weight: float
    # specs for multipliers
    specs: Dict[str, TruncNormSpec]

    def get_spec(self, key: str) -> TruncNormSpec | None:
        return self.specs.get(key)


def _extract_specs(cfg: Mapping[str, Any]) -> Dict[str, TruncNormSpec]:
    specs: Dict[str, TruncNormSpec] = {}
    for k, v in cfg.items():
        if not k.endswith("_mult"):
            continue
        if not isinstance(v, Mapping):
            continue
        dist = str(v.get("dist", "truncnorm")).lower()
        if dist != "truncnorm":
            continue
        specs[k] = parse_truncnorm_spec(dict(v))
    return specs


def load_failure_classes(config: Mapping[str, Any] | None = None) -> List[FailureClass]:
    """Load failure classes from config mapping or defaults."""
    config = config or DEFAULT_FAILURE_CLASSES
    out: List[FailureClass] = []
    for name, cfg in config.items():
        if not isinstance(cfg, Mapping):
            continue
        weight = float(cfg.get("weight", 1.0))
        specs = _extract_specs(cfg)
        out.append(FailureClass(name=str(name), weight=weight, specs=specs))
    if not out:
        raise ValueError("No failure classes defined.")
    return out


class FailureClassIndex:
    """Utility for name<->index mapping."""
    def __init__(self, classes: List[FailureClass]):
        self.classes = classes
        self.name_to_idx = {c.name: i for i, c in enumerate(classes)}
        self.idx_to_name = {i: c.name for i, c in enumerate(classes)}

    def idx(self, name: str) -> int:
        return self.name_to_idx[name]

    def name(self, idx: int) -> str:
        return self.idx_to_name[idx]


def class_multiplier_keys() -> Tuple[str, ...]:
    """Canonical multiplier keys."""
    return ("iph_mult", "i0_mult", "rs_mult", "rsh_mult", "n_mult")
