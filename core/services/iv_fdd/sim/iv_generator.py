from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.services.power_model.power_model import (
    ModuleOneDiode,
    ref_params_stc,
    iph_irr_temp,
    i0_temp,
    rp_irr,
    voc_guess,
    voc_newton_vec,
    iv_current_mat,
    _vt_cell,
)

from .failure_defs import FailureClass, load_failure_classes
from .sampling import sample_multipliers
from ..configs.hopwood_defaults import DEFAULT_ENV_RANGES, DEFAULT_N_POINTS


# --- local helper: allow per-sample Rs (vector) ---
def iv_current_mat_rsvec(
    Vmat: np.ndarray,
    iph: np.ndarray,
    i0: np.ndarray,
    rs: np.ndarray,
    rp: np.ndarray,
    aVt: np.ndarray,
    *,
    max_iter: int = 30,
) -> np.ndarray:
    """Resolve I(V) for a matrix Vmat (n,m) with vector Rs (n,)."""
    Vmat = np.asarray(Vmat, dtype=float)
    iph = np.asarray(iph, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    rp = np.asarray(rp, dtype=float)
    aVt = np.asarray(aVt, dtype=float)
    rs = np.asarray(rs, dtype=float)

    if Vmat.ndim != 2:
        raise ValueError('Vmat must be 2D (n, m).')
    n, _ = Vmat.shape
    if iph.shape != (n,) or i0.shape != (n,) or rp.shape != (n,) or aVt.shape != (n,) or rs.shape != (n,):
        raise ValueError("iph/i0/rp/aVt/rs must be shape (n,)")

    # Use existing solver by looping rows (still fast for moderate n; optimize later if needed)
    I = np.empty_like(Vmat)
    for i in range(n):
        I[i, :] = iv_current_mat(
            Vmat[i : i + 1, :],
            iph[i : i + 1],
            i0[i : i + 1],
            float(rs[i]),
            rp[i : i + 1],
            aVt[i : i + 1],
            max_iter=max_iter,
        )[0, :]
    return I


@dataclass(frozen=True)
class SynthConfig:
    n_samples: int = 10_000
    n_points: int = DEFAULT_N_POINTS
    seed: Optional[int] = 42

    # environment ranges (Gpoa in W/m2, Tcell in °C)
    g_poa_lo: float = DEFAULT_ENV_RANGES["g_poa_lo"]
    g_poa_hi: float = DEFAULT_ENV_RANGES["g_poa_hi"]
    tcell_lo: float = DEFAULT_ENV_RANGES["tcell_lo"]
    tcell_hi: float = DEFAULT_ENV_RANGES["tcell_hi"]

    # sampling method for multipliers
    use_lhs: bool = True

    # voltage grid type
    # If True: use normalized Vgrid in [0,1] and scale by Voc per sample -> fixed points, Hopwood-style.
    normalize_voltage: bool = True


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_env(cfg: SynthConfig) -> Tuple[np.ndarray, np.ndarray]:
    gen = _rng(cfg.seed)
    g = gen.uniform(cfg.g_poa_lo, cfg.g_poa_hi, size=cfg.n_samples)
    t = gen.uniform(cfg.tcell_lo, cfg.tcell_hi, size=cfg.n_samples)
    return g, t


def _sample_failure_classes(cfg: SynthConfig, *, classes: List[FailureClass]) -> Tuple[np.ndarray, List[str]]:
    """Return y indices and class name list (index -> name)."""
    weights = np.array([c.weight for c in classes], dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    gen = _rng(cfg.seed)
    y = gen.choice(len(classes), size=cfg.n_samples, replace=True, p=weights)
    idx_to_name = [c.name for c in classes]
    return y, idx_to_name


def _apply_multipliers(
    base: Dict[str, np.ndarray],
    mult: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    out = dict(base)
    for k, v in mult.items():
        if k not in out:
            continue
        out[k] = out[k] * v
    return out


def _safe_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def generate_synth_dataset(
    module: ModuleOneDiode,
    *,
    cfg: SynthConfig | None = None,
    failure_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate synthetic IV dataset using your existing 1-diode model.

    This is a pragmatic "Hopwood-style" dataset builder:
      - sample environment (Gpoa, Tcell)
      - sample fault class
      - sample multipliers over single-diode parameters
      - generate I(V) curve over fixed-point V grid (normalized [0,1] scaled by Voc)

    Note:
      - Hopwood do STC translation (IEC 60891) later; here we just generate multiple conditions.
      - Bishop avalanche / bypass steps are NOT included yet (next milestone).
    """
    cfg = cfg or SynthConfig()
    classes = load_failure_classes(failure_cfg)

    # 1) env sampling
    g_poa, tcell = _sample_env(cfg)

    # 2) base params at STC
    # your power_model provides ref params helper
    ref = ref_params_stc(module)

    # 3) compute baseline params for each sample (from env)
    # iph, i0, rp are arrays per sample; rs and n usually scalars in your model
    iph0 = iph_irr_temp(ref, g_poa, tcell)
    i00 = i0_temp(ref, tcell)
    rp0 = rp_irr(ref, g_poa)

    # these are base scalar params from module/ref
    rs0 = float(ref.Rs)
    n0 = float(ref.a)

    # We'll keep rs and n as vectors for per-sample multipliers
    rs0_vec = np.full(cfg.n_samples, rs0, dtype=float)
    n0_vec = np.full(cfg.n_samples, n0, dtype=float)

    # rsh is rp in your notation (parallel/shunt)
    base_params = {
        "iph_mult": np.ones(cfg.n_samples, dtype=float),
        "i0_mult": np.ones(cfg.n_samples, dtype=float),
        "rs_mult": np.ones(cfg.n_samples, dtype=float),
        "rsh_mult": np.ones(cfg.n_samples, dtype=float),
        "n_mult": np.ones(cfg.n_samples, dtype=float),
    }

    # 4) sample fault class per sample
    y_idx, idx_to_name = _sample_failure_classes(cfg, classes=classes)

    # 5) sample multipliers for each class and assign to samples
    mult_iph = np.ones(cfg.n_samples, dtype=float)
    mult_i0 = np.ones(cfg.n_samples, dtype=float)
    mult_rs = np.ones(cfg.n_samples, dtype=float)
    mult_rsh = np.ones(cfg.n_samples, dtype=float)
    mult_n = np.ones(cfg.n_samples, dtype=float)

    for c_idx, c in enumerate(classes):
        mask = y_idx == c_idx
        n_c = int(mask.sum())
        if n_c <= 0:
            continue

        samples = sample_multipliers(c.specs, n=n_c, seed=cfg.seed, use_lhs=cfg.use_lhs)
        if "iph_mult" in samples:
            mult_iph[mask] = samples["iph_mult"]
        if "i0_mult" in samples:
            mult_i0[mask] = samples["i0_mult"]
        if "rs_mult" in samples:
            mult_rs[mask] = samples["rs_mult"]
        if "rsh_mult" in samples:
            mult_rsh[mask] = samples["rsh_mult"]
        if "n_mult" in samples:
            mult_n[mask] = samples["n_mult"]

    # Apply multipliers to baseline per-sample params
    iph = iph0 * mult_iph
    i0 = i00 * mult_i0
    rp = rp0 * mult_rsh
    rs = rs0_vec * mult_rs
    a = n0_vec * mult_n  # diode factor (a), consistent with your model naming

    # Safety: keep params within plausible ranges to avoid divergence
    iph = _safe_clip(iph, 1e-6, np.inf)
    i0 = _safe_clip(i0, 1e-12, np.inf)
    rp = _safe_clip(rp, 1e-3, np.inf)
    rs = _safe_clip(rs, 1e-6, 10.0)
    a = _safe_clip(a, 0.6, 3.0)

    # 6) build V grid (fixed points)
    # Estimate Voc for each sample (uses your existing voc_guess + newton)
    vt = _vt_cell(tcell)  # thermal voltage per cell in your model
    # aVt in your solver is a * Ns * Vt? your power_model uses aVt directly for matrix solver
    aVt = a * module.Ns * vt

    voc0 = voc_guess(iph, i0, rp, aVt)
    voc = voc_newton_vec(voc0, iph, i0, rs, rp, aVt)

    # normalized fixed V grid (0..1) then scale by Voc per sample
    if cfg.normalize_voltage:
        v_norm = np.linspace(0.0, 1.0, cfg.n_points, dtype=float)
        V = voc[:, None] * v_norm[None, :]
    else:
        # absolute fixed grid up to median Voc (less recommended)
        vmax = float(np.median(voc))
        V = np.linspace(0.0, vmax, cfg.n_points, dtype=float)[None, :] * np.ones((cfg.n_samples, 1))

    # 7) solve I(V)
    # your existing iv_current_mat expects scalar Rs; we need per-sample -> use helper loop
    I = iv_current_mat_rsvec(V, iph, i0, rs, rp, aVt, max_iter=30)

    # 8) package dataset
    dataset: Dict[str, Any] = {
        "V": V.astype(np.float32),
        "I": I.astype(np.float32),
        "y": y_idx.astype(np.int64),
        "class_names": idx_to_name,
        "g_poa": g_poa.astype(np.float32),
        "tcell": tcell.astype(np.float32),
        "voc": voc.astype(np.float32),
    }
    return dataset


def save_dataset_npz(dataset: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        V=dataset["V"],
        I=dataset["I"],
        y=dataset["y"],
        g_poa=dataset["g_poa"],
        tcell=dataset["tcell"],
        voc=dataset["voc"],
        class_names=np.array(dataset["class_names"], dtype=object),
    )
    return path


def load_dataset_npz(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        return {
            "V": z["V"],
            "I": z["I"],
            "y": z["y"],
            "g_poa": z["g_poa"],
            "tcell": z["tcell"],
            "voc": z["voc"],
            "class_names": list(z["class_names"].tolist()),
        }


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    y = np.asarray(dataset["y"])
    class_names = dataset["class_names"]
    counts = {class_names[i]: int((y == i).sum()) for i in range(len(class_names))}
    return {
        "n": int(y.size),
        "classes": counts,
        "n_points": int(np.asarray(dataset["V"]).shape[1]),
    }
