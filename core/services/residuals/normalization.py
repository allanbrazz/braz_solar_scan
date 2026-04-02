from __future__ import annotations

from typing import Optional


def safe_rel(observed: Optional[float], expected: Optional[float], eps: float) -> Optional[float]:
    if observed is None or expected is None:
        return None
    den = max(abs(float(expected)), float(eps))
    return (float(observed) - float(expected)) / den


def safe_abs(observed: Optional[float], expected: Optional[float]) -> Optional[float]:
    if observed is None or expected is None:
        return None
    return float(observed) - float(expected)


def clip_or_none(value: Optional[float], clip_abs: float) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if clip_abs <= 0:
        return v
    return max(-float(clip_abs), min(float(clip_abs), v))
