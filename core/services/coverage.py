# core/services/coverage.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_tz
from typing import Iterable, List, Optional, Tuple

UTC = dt_tz.utc


@dataclass(frozen=True)
class CoverageResult:
    start_utc: datetime
    end_utc: datetime
    interval_min: int

    expected_count: int
    existing_count: int
    missing_count: int
    coverage_pct: float

    # intervalos faltantes em UTC: lista de (start, end) inclusivos
    missing_ranges_utc: List[Tuple[datetime, datetime]]


def _floor_to_interval_utc(dt: datetime, interval_min: int) -> datetime:
    """
    Arredonda dt (UTC aware) para baixo para múltiplos de interval_min.
    """
    if dt.tzinfo is None:
        raise ValueError("dt precisa ser timezone-aware (UTC).")
    seconds = interval_min * 60
    ts = int(dt.timestamp())
    floored = ts - (ts % seconds)
    return datetime.fromtimestamp(floored, tz=UTC)


def _expected_timestamps_utc(start_utc: datetime, end_utc: datetime, interval_min: int) -> List[datetime]:
    """
    Gera timestamps esperados inclusivos: [start_aligned, ..., end_aligned]
    """
    start_aligned = _floor_to_interval_utc(start_utc, interval_min)
    end_aligned = _floor_to_interval_utc(end_utc, interval_min)

    if end_aligned < start_aligned:
        return []

    step = timedelta(minutes=interval_min)
    out: List[datetime] = []
    cur = start_aligned
    while cur <= end_aligned:
        out.append(cur)
        cur += step
    return out


def compute_time_coverage(
    *,
    queryset,
    start_utc: datetime,
    end_utc: datetime,
    interval_min: int,
    ts_field: str = "ts_utc",
    max_missing_ranges: int = 200,
) -> CoverageResult:
    """
    Calcula cobertura de uma série temporal armazenada em DB.

    - queryset: já filtrado para plant/source e período (ou apenas plant).
    - start_utc/end_utc: timezone-aware em UTC.
    - interval_min: resolução de referência.
    - ts_field: nome do campo datetime (padrão: 'ts_utc').

    Retorna missing_ranges comprimidos em blocos contíguos.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc e end_utc devem ser timezone-aware.")
    if start_utc.tzinfo != UTC or end_utc.tzinfo != UTC:
        raise ValueError("start_utc/end_utc devem estar em UTC.")

    expected_ts = _expected_timestamps_utc(start_utc, end_utc, interval_min)
    expected_count = len(expected_ts)

    if expected_count == 0:
        return CoverageResult(
            start_utc=start_utc,
            end_utc=end_utc,
            interval_min=interval_min,
            expected_count=0,
            existing_count=0,
            missing_count=0,
            coverage_pct=100.0,
            missing_ranges_utc=[],
        )

    # Buscar timestamps existentes dentro do período alinhado
    start_aligned = expected_ts[0]
    end_aligned = expected_ts[-1]

    values = queryset.filter(**{
        f"{ts_field}__gte": start_aligned,
        f"{ts_field}__lte": end_aligned,
    }).values_list(ts_field, flat=True)

    existing_set = set(values)
    existing_count = len(existing_set)

    missing_ranges: List[Tuple[datetime, datetime]] = []
    current_start: Optional[datetime] = None
    prev: Optional[datetime] = None

    for ts in expected_ts:
        if ts not in existing_set:
            if current_start is None:
                current_start = ts
            prev = ts
        else:
            if current_start is not None and prev is not None:
                missing_ranges.append((current_start, prev))
                current_start = None
                prev = None

        # limite para não explodir a resposta
        if len(missing_ranges) >= max_missing_ranges:
            break

    if current_start is not None and prev is not None and len(missing_ranges) < max_missing_ranges:
        missing_ranges.append((current_start, prev))

    missing_count = expected_count - existing_count
    coverage_pct = (existing_count / expected_count) * 100.0

    return CoverageResult(
        start_utc=start_utc,
        end_utc=end_utc,
        interval_min=interval_min,
        expected_count=expected_count,
        existing_count=existing_count,
        missing_count=missing_count,
        coverage_pct=coverage_pct,
        missing_ranges_utc=missing_ranges,
    )
