#core/views/meteo
from __future__ import annotations
from core.views._imports import *
from datetime import datetime, timedelta
from core.services.dados_satelite.openmeteo import ingest_openmeteo_range
from django.utils.timezone import make_aware
from core.services.coverage import compute_time_coverage
from zoneinfo import ZoneInfo
# Forms
from core.forms import (
    MeteoRequestForm,

)

# Models
from core.models import (
    MeteoRecord,
    MeteoSource,

)

#---------------------------
#---------------------------  M E T E O
#---------------------------

UTC = ZoneInfo("UTC")


def _safe_zoneinfo(tzname: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tzname) if tzname else ZoneInfo("UTC")
    except Exception:
        return ZoneInfo("UTC")


def _local_dates_to_utc_range(*, plant_tz: str | None, start_date, end_date):
    """
    Converte datas (date) em intervalo UTC semiaberto:
      [start_local 00:00, (end_date+1) 00:00) em tz local
      -> retorna (start_utc, end_utc_exclusive, tz_local)
    Isso evita off-by-one e funciona bem com __gte / __lt e com grades (15min/5min).
    """
    tz_local = _safe_zoneinfo(plant_tz)

    start_local = make_aware(datetime.combine(start_date, time.min), timezone=tz_local)

    # end EXCLUSIVO: começo do dia seguinte
    end_local_excl = make_aware(
        datetime.combine(end_date + timedelta(days=1), time.min),
        timezone=tz_local,
    )

    return start_local.astimezone(UTC), end_local_excl.astimezone(UTC), tz_local


def _align_utc_range_to_interval(*, start_utc, end_utc, interval_min: int):
    """
    Opcional: alinha start/end para a grade do intervalo.
    - start: floor
    - end: ceil (mantém end exclusivo)
    """
    import pandas as pd

    freq = f"{int(interval_min)}min"
    s = pd.Timestamp(start_utc).floor(freq)
    e = pd.Timestamp(end_utc).ceil(freq)

    # garante tz
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")

    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")

    return s.to_pydatetime(), e.to_pydatetime()


@require_http_methods(["GET", "POST"])
def open_meteo_view(request):
    """
    Mantive o nome da view/URL para não quebrar a rota.
    Opera com Open-Meteo (ingest).
    """
    form = MeteoRequestForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        plant = form.cleaned_data["plant"]
        start_date = form.cleaned_data["start_date"]
        end_date = form.cleaned_data["end_date"]
        include_gti = form.cleaned_data["include_gti"]
        model = (form.cleaned_data.get("model") or "").strip() or None

        try:
            count, meta = ingest_openmeteo_range(
                plant=plant,
                start_date=start_date,
                end_date=end_date,
                include_gti=include_gti,
                model=model,
            )
            model_label = model or "best_match"
            messages.success(
                request,
                (
                    f"Open-Meteo: {count} registros ingeridos/atualizados. "
                    f"Modelo: {model_label}. "
                    f"Intervalo: {start_date} a {end_date}."
                )
            )
        except Exception as e:
            messages.error(request, f"Falha ao ingerir dados meteorológicos da Open-Meteo: {e}")

    return render(request, "meteo/open_meteo_request.html", {"form": form})

def open_meteo_view_api_json(request):
    """
    Endpoint reutilizável para cobertura/consistência no banco.
    GET params via MeteoRequestForm: plant, start_date, end_date, interval_min
    """
    form = MeteoRequestForm(request.GET)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    plant = form.cleaned_data["plant"]
    start_date = form.cleaned_data["start_date"]
    end_date = form.cleaned_data["end_date"]
    interval_min = int(form.cleaned_data["interval_min"])

    start_utc, end_utc, tz_local = _local_dates_to_utc_range(
        plant_tz=getattr(plant, "timezone", None),
        start_date=start_date,
        end_date=end_date,
    )

    # Opcional, mas ajuda muito a bater contagens com “expected_count”
    start_utc, end_utc = _align_utc_range_to_interval(
        start_utc=start_utc, end_utc=end_utc, interval_min=interval_min
    )

    # >>> CORREÇÃO PRINCIPAL AQUI: source do Open-Meteo <<<
    qs = MeteoRecord.objects.filter(plant=plant, source=MeteoSource.OPENMETEO)

    cov = compute_time_coverage(
        queryset=qs,
        start_utc=start_utc,
        end_utc=end_utc,
        interval_min=interval_min,
    )

    ranges_local = [
        {"start": a.astimezone(tz_local).isoformat(), "end": b.astimezone(tz_local).isoformat()}
        for (a, b) in cov.missing_ranges_utc[:50]
    ]

    return JsonResponse({
        "ok": True,
        "plant_id": plant.id,
        "plant_tz": str(tz_local),
        "interval_min": cov.interval_min,
        "start_utc": cov.start_utc.isoformat(),
        "end_utc": cov.end_utc.isoformat(),
        "expected_count": cov.expected_count,
        "existing_count": cov.existing_count,
        "missing_count": cov.missing_count,
        "coverage_pct": round(cov.coverage_pct, 2),
        "missing_ranges_local": ranges_local,
        "missing_ranges_truncated": len(cov.missing_ranges_utc) > 50,
    })

