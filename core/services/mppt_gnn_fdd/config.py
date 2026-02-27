from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from collections import Counter

from core.models import PVPlant, PVPlantStringConfig


@dataclass(frozen=True)
class MpptNominal:
    mppt: int
    n_series: int
    n_parallel: int
    voc_v: float
    isc_a: float


def load_mppt_nominals(plant: PVPlant, *, nmax: int = 4) -> Dict[int, MpptNominal]:
    """
    Deriva (n_series, n_parallel) por MPPT a partir de PVPlantStringConfig(mppt=...).
    Usa Voc/Isc do módulo associado em PVPlantDetails.module.
    """
    details = getattr(plant, "details", None)
    module = getattr(details, "module", None) if details else None
    if module is None:
        raise ValueError("PVPlantDetails.module está vazio. Necessário para Voc/Isc (normalização).")

    voc_v = float(module.voc_v)
    isc_a = float(module.isc_a)

    cfgs = list(
        PVPlantStringConfig.objects.filter(details=details, mppt__isnull=False).only(
            "mppt", "strings_qty", "modules_per_string"
        )
    )

    out: Dict[int, MpptNominal] = {}
    for k in range(1, nmax + 1):
        ck = [c for c in cfgs if int(c.mppt or 0) == k]
        if not ck:
            continue
        n_parallel = sum(int(c.strings_qty or 0) for c in ck)
        mps = [int(c.modules_per_string or 0) for c in ck if c.modules_per_string]
        if not mps or n_parallel <= 0:
            continue
        # modo (mais comum)
        n_series = Counter(mps).most_common(1)[0][0]
        if n_series <= 0:
            continue
        out[k] = MpptNominal(mppt=k, n_series=n_series, n_parallel=n_parallel, voc_v=voc_v, isc_a=isc_a)

    return out


@dataclass(frozen=True)
class TrainConfig:
    # janela
    T: int = 96
    stride_steps: int = 4  # 1h (treino) por padrão

    # features
    use_meteo: bool = True
    gpoa_gate_wm2: float = 300.0  # para pseudo-label disconnected

    # pseudo-label disconnected
    i_disc_thresh: float = 0.02
    i_peer_active: float = 0.05
    min_sun_points: int = 12       # >=3h (12*15min)
    min_peer_points: int = 12

    # treino
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42

    # classes (fixo por agora)
    class_names: tuple[str, ...] = ("normal", "disconnected")

    # device
    device: str = "cpu"