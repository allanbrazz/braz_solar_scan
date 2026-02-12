# =========================================
# core/services/health_heatmap.py
# Daily health summary (DB -> day buckets)
# =========================================
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Optional, Tuple

from django.db.models import Count
from django.db.models.functions import TruncDate
from django.utils import timezone

from core.models import PlantDiagnostic15m


# -----------------------------
# Diagnostic codes (alinhado ao seu FuzzyDiagnosticService)
# -----------------------------
@dataclass(frozen=True)
class DiagnosticCodes:
    INVALID: int = 0
    NORMAL: int = 1
    METEO_ERROR: int = 2
    SOILING: int = 3
    DEGRADATION: int = 4
    SHORT_BYPASS: int = 5
    STRING_DISCONNECTED: int = 6
    PARTIAL_SHADING: int = 7


def _get_tz(tz_name: Optional[str]):
    """
    Resolve timezone. Prefer:
      1) tz_name (ZoneInfo)
      2) Django current timezone
    """
    tz = timezone.get_current_timezone()
    if tz_name:
        try:
            from zoneinfo import ZoneInfo  # py3.9+

            tz = ZoneInfo(tz_name)
        except Exception:
            pass
    return tz


def _local_day_range_to_utc(
    start_d: date,
    end_d: date,
    tz,
) -> Tuple[datetime, datetime]:
    """
    Converte [start_date, end_date] (dias locais) para intervalo UTC:
      start_dt_utc = start_date 00:00 local -> UTC
      end_dt_utc   = (end_date+1) 00:00 local -> UTC

    Importante:
      - usa make_aware para lidar melhor com DST (evita replace(tzinfo=...)).
    """
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    start_local = timezone.make_aware(datetime.combine(start_d, time.min), timezone=tz)
    end_local = timezone.make_aware(datetime.combine(end_d + timedelta(days=1), time.min), timezone=tz)

    return start_local.astimezone(dt_tz.utc), end_local.astimezone(dt_tz.utc)


def _expected_bins_for_day(day_local: date, tz, *, sample_minutes: int) -> int:
    """
    Expected bins por dia, DST-aware (dia pode ter 23h/25h).
    """
    sm = max(int(sample_minutes), 1)

    d0 = timezone.make_aware(datetime.combine(day_local, time.min), timezone=tz)
    d1 = timezone.make_aware(datetime.combine(day_local + timedelta(days=1), time.min), timezone=tz)

    mins = (d1 - d0).total_seconds() / 60.0
    exp = int(round(mins / sm))
    return max(exp, 1)


def get_daily_health_summary_from_db(
    *,
    plant_id: int,
    start_date: date,
    end_date: date,
    tz_name: Optional[str] = None,
    sample_minutes: int = 15,
    min_coverage: float = 0.20,
    normal_min_ratio: float = 0.90,
    meteo_alert_min_ratio: float = 0.20,
    # opcional: permite usar "valid" como filtro (recomendado)
    require_valid: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retorna uma lista (um item por dia) com:
      - date (ISO)
      - status: ok | warn | fault | nodata
      - coverage: n_samples / expected (DST-aware)
      - ratios (normal/meteo)
      - counts por rca_code
      - n_total / expected

    Observações:
      - PlantDiagnostic15m.ts_utc é UTC (aware).
      - bucket diário é no fuso local (tz_name).
      - por padrão, considera apenas registros valid=True (senão polui a sanidade).
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    tz = _get_tz(tz_name)
    start_dt_utc, end_dt_utc = _local_day_range_to_utc(start_date, end_date, tz)

    qs = PlantDiagnostic15m.objects.filter(
        plant_id=int(plant_id),
        ts_utc__gte=start_dt_utc,
        ts_utc__lt=end_dt_utc,
    )
    if require_valid:
        qs = qs.filter(valid=True)

    # counts por dia/código (dia no fuso local)
    by_code = (
        qs.annotate(day=TruncDate("ts_utc", tzinfo=tz))
        .values("day", "rca_code")
        .annotate(n=Count("id"))
    )

    # total por dia
    by_day = (
        qs.annotate(day=TruncDate("ts_utc", tzinfo=tz))
        .values("day")
        .annotate(n=Count("id"))
    )

    counts_map: Dict[date, Dict[int, int]] = {}
    for r in by_code:
        d = r.get("day")
        if not isinstance(d, date):
            continue
        c_raw = r.get("rca_code")
        try:
            c = int(c_raw) if c_raw is not None else DiagnosticCodes.INVALID
        except Exception:
            c = DiagnosticCodes.INVALID
        counts_map.setdefault(d, {})[c] = int(r.get("n") or 0)

    total_map: Dict[date, int] = {}
    for r in by_day:
        d = r.get("day")
        if not isinstance(d, date):
            continue
        total_map[d] = int(r.get("n") or 0)

    def _ratio(counts: Dict[int, int], code: int, n_total: int) -> float:
        return float(counts.get(int(code), 0)) / float(max(int(n_total), 1))

    out: List[Dict[str, Any]] = []
    cur = start_date

    while cur <= end_date:
        counts = counts_map.get(cur, {})
        n_total = int(total_map.get(cur, 0))
        expected = _expected_bins_for_day(cur, tz, sample_minutes=sample_minutes)
        coverage = (float(n_total) / float(expected)) if expected > 0 else 0.0

        if n_total <= 0 or coverage < float(min_coverage):
            out.append(
                {
                    "date": cur.isoformat(),
                    "status": "nodata",
                    "coverage": float(coverage),
                    "ratios": {"normal": 0.0, "meteo": 0.0},
                    "counts": counts,
                    "n_total": n_total,
                    "expected": expected,
                }
            )
            cur += timedelta(days=1)
            continue

        r_normal = _ratio(counts, DiagnosticCodes.NORMAL, n_total)
        r_meteo = _ratio(counts, DiagnosticCodes.METEO_ERROR, n_total)

        # fault se houver qualquer evento "hard" no dia (você pode tornar isso mais sofisticado depois)
        has_fault = (
            (counts.get(DiagnosticCodes.STRING_DISCONNECTED, 0) > 0)
            or (counts.get(DiagnosticCodes.SHORT_BYPASS, 0) > 0)
            or (counts.get(DiagnosticCodes.PARTIAL_SHADING, 0) > 0)
        )

        if has_fault:
            status = "fault"
        elif r_meteo >= float(meteo_alert_min_ratio):
            status = "warn"
        elif r_normal >= float(normal_min_ratio):
            status = "ok"
        else:
            status = "warn"

        out.append(
            {
                "date": cur.isoformat(),
                "status": status,
                "coverage": float(coverage),
                "ratios": {"normal": float(r_normal), "meteo": float(r_meteo)},
                "counts": counts,
                "n_total": n_total,
                "expected": expected,
            }
        )

        cur += timedelta(days=1)

    return out
