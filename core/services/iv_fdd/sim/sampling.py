from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class TruncNormSpec:
    mu: float
    sigma: float
    lo: float
    hi: float


def _try_scipy_truncnorm():
    try:
        from scipy.stats import truncnorm  # type: ignore
        return truncnorm
    except Exception:
        return None


def _try_scipy_lhs():
    try:
        from scipy.stats import qmc  # type: ignore
        return qmc
    except Exception:
        return None


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_truncnorm(
    spec: TruncNormSpec,
    n: int,
    *,
    seed: Optional[int] = None,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample from a truncated normal.

    If SciPy is installed, uses scipy.stats.truncnorm with inverse CDF.
    Otherwise falls back to rejection sampling.

    Args:
        spec: distribution parameters.
        n: number of samples.
        seed: RNG seed for fallback.
        u: optional uniform(0,1) samples for inverse transform (used with LHS).

    Returns:
        samples shape (n,)
    """
    spec = TruncNormSpec(float(spec.mu), float(spec.sigma), float(spec.lo), float(spec.hi))
    if spec.sigma <= 0:
        return np.full(n, spec.mu, dtype=float)

    truncnorm = _try_scipy_truncnorm()
    if truncnorm is not None:
        a = (spec.lo - spec.mu) / (spec.sigma + EPS)
        b = (spec.hi - spec.mu) / (spec.sigma + EPS)
        dist = truncnorm(a=a, b=b, loc=spec.mu, scale=spec.sigma)
        if u is None:
            return dist.rvs(size=n, random_state=seed)  # type: ignore[arg-type]
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != n:
            raise ValueError("u must have length n.")
        return dist.ppf(np.clip(u, EPS, 1.0 - EPS))

    # Fallback: rejection sampling
    gen = _rng(seed)
    out = np.empty(n, dtype=float)
    k = 0
    # Conservative cap to avoid infinite loops (should not happen with sane specs)
    max_draws = max(10_000, 50 * n)
    draws = 0
    while k < n and draws < max_draws:
        x = gen.normal(loc=spec.mu, scale=spec.sigma, size=(n - k) * 2)
        x = x[(x >= spec.lo) & (x <= spec.hi)]
        take = min(x.size, n - k)
        if take > 0:
            out[k : k + take] = x[:take]
            k += take
        draws += x.size

    if k < n:
        # Final fallback: clip normals
        x = gen.normal(loc=spec.mu, scale=spec.sigma, size=(n - k))
        out[k:] = np.clip(x, spec.lo, spec.hi)

    return out


def latin_hypercube_u(n: int, d: int, *, seed: Optional[int] = None) -> np.ndarray:
    """Generate an LHS design U in (0,1)^(n,d).

    Uses SciPy qmc.LatinHypercube if available; else uses stratified sampling.

    Returns:
        U shape (n,d)
    """
    qmc = _try_scipy_lhs()
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        return sampler.random(n=n)

    gen = _rng(seed)
    # stratified: each dim independently stratified
    U = np.empty((n, d), dtype=float)
    for j in range(d):
        # permuted strata
        perm = gen.permutation(n)
        U[:, j] = (perm + gen.random(n)) / n
    return U


def sample_multipliers(
    specs: Dict[str, TruncNormSpec],
    n: int,
    *,
    seed: Optional[int] = None,
    use_lhs: bool = True,
) -> Dict[str, np.ndarray]:
    """Sample multiple multiplier variables using truncnorm specs.

    If use_lhs=True, generates an LHS U design and applies inverse transform per variable.

    Args:
        specs: mapping name -> TruncNormSpec
        n: number of samples
        seed: RNG seed
        use_lhs: whether to use LHS

    Returns:
        mapping name -> array (n,)
    """
    keys = list(specs.keys())
    d = len(keys)
    if d == 0:
        return {}

    if use_lhs:
        U = latin_hypercube_u(n=n, d=d, seed=seed)
    else:
        U = None

    out: Dict[str, np.ndarray] = {}
    for j, k in enumerate(keys):
        u = None if U is None else U[:, j]
        out[k] = sample_truncnorm(specs[k], n=n, seed=seed, u=u)

    return out


def parse_truncnorm_spec(d: Dict[str, float]) -> TruncNormSpec:
    """Parse dict -> TruncNormSpec."""
    return TruncNormSpec(
        mu=float(d.get("mu", 1.0)),
        sigma=float(d.get("sigma", 0.1)),
        lo=float(d.get("lo", 0.0)),
        hi=float(d.get("hi", 10.0)),
    )


def normalize_class_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / max(1, w.size)
    return w / s


def sample_class_indices(
    n: int,
    weights: Sequence[float],
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample class indices 0..K-1 according to weights."""
    gen = _rng(seed)
    p = normalize_class_weights(weights)
    return gen.choice(len(p), size=n, replace=True, p=p)
