from __future__ import annotations
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from django.conf import settings
from django.http import HttpRequest


LOCAL_TZ = ZoneInfo(getattr(settings, "TIME_ZONE", "America/Montevideo"))


def _get_float(request: HttpRequest, key: str, default: float) -> float:
    """
    Lê um float de request.GET[key], aceitando vírgula como decimal.
    Se vier vazio/ausente/ inválido -> retorna default.
    """
    raw = request.GET.get(key, None)
    if raw is None:
        return float(default)

    raw = str(raw).strip()
    if raw == "":
        return float(default)

    # aceita "34,9" -> "34.9"
    raw = raw.replace(",", ".")

    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)

def _to_decimal(value):
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        raise ValueError(f"Valor numérico inválido: '{value}'")


