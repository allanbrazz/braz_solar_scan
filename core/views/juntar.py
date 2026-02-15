# core/views/juntar.py
from __future__ import annotations

from core.views._imports import *  # mantém seu padrão (HttpRequest, HttpResponse, pd, messages, login_required, etc.)
from datetime import datetime, timedelta, time, timezone as dt_timezone
from zoneinfo import ZoneInfo

from core.services.series_juntar.build_merged_dataset import build_plant_merged_dataset

# Forms
from core.forms import MergeRunForm


# ---------------------------
#  JUNTAR BASES
# ---------------------------

def _local_dates_to_utc_range(start_date, end_date, tz_name: str) -> tuple[datetime, datetime]:
    """
    Converte [start_date, end_date] (datas locais da planta) em intervalo UTC [start, end).

    Exemplo (America/Maceio, UTC-03):
      start_date=2025-12-31 -> start_utc=2025-12-31T03:00Z
      end_date=2025-12-31   -> end_utc  =2026-01-01T03:00Z
    """
    tz = ZoneInfo(tz_name or "UTC")

    # Crie datetimes locais tz-aware corretamente
    start_local = datetime.combine(start_date, time.min, tzinfo=tz)
    end_local_excl = datetime.combine(end_date, time.min, tzinfo=tz) + timedelta(days=1)

    # Django 5.x não tem django.utils.timezone.utc -> use o UTC do Python
    return (
        start_local.astimezone(dt_timezone.utc),
        end_local_excl.astimezone(dt_timezone.utc),
    )


def _df_preview(df: pd.DataFrame, tz_name: str, n: int = 60) -> tuple[list[str], list[list]]:
    """
    Converte o índice (assumindo que o índice representa UTC) para o timezone da planta
    e retorna:
      - cols: lista de nomes de colunas
      - rows: lista de listas (valores alinhados com cols)

    Regras:
    - Se o índice for tz-naive: assume UTC e tz_localize("UTC")
    - Se for tz-aware: tz_convert("UTC") antes, e depois tz_convert(tz da planta)
    - Formata ts_local como string para a UI (evita exibir offset -03:00)
    """
    if df is None or df.empty:
        return [], []

    tz = ZoneInfo(tz_name or "UTC")
    d = df.copy()

    # --- Garantir índice datetime tz-aware (assumindo que o índice representa UTC) ---
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")

    if not isinstance(d.index, pd.DatetimeIndex):
        return [], []

    # Se index tz-naive, assume UTC
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        # normaliza para UTC antes de converter para tz local
        d.index = d.index.tz_convert("UTC")

    # Converte para tz local da planta
    d.index = d.index.tz_convert(tz)

    # Recorta e leva índice para coluna
    d = d.head(n).reset_index()

    # Normaliza nome da coluna de tempo
    if "ts_15" in d.columns:
        d = d.rename(columns={"ts_15": "ts_local"})
    elif "ts_utc" in d.columns:
        d = d.rename(columns={"ts_utc": "ts_local"})
    elif "index" in d.columns:
        d = d.rename(columns={"index": "ts_local"})
    else:
        d = d.rename(columns={d.columns[0]: "ts_local"})

    # Formata timestamp para UI (sem offset/timezone)
    if "ts_local" in d.columns:
        d["ts_local"] = pd.to_datetime(d["ts_local"], errors="coerce")
        d["ts_local"] = d["ts_local"].dt.strftime("%d/%m/%Y %H:%M")

    # Arredonda numéricos para preview
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            if pd.api.types.is_bool_dtype(d[c]):
                continue
            d[c] = pd.to_numeric(d[c], errors="coerce").round(3)

    cols = list(d.columns)
    rows = d.to_numpy(dtype=object).tolist()
    return cols, rows


# -----------------------------------------------------------------------------
# View
# -----------------------------------------------------------------------------

@login_required
def merge_run_view(request: HttpRequest) -> HttpResponse:
    """
    Tela para executar merge e (opcionalmente) persistir base casada 15 min.
    """
    stats = None
    cols15, rows15 = [], []
    colsh, rowsh = [], []

    if request.method == "POST":
        form = MergeRunForm(request.POST, user=request.user)
        if form.is_valid():
            plant = form.cleaned_data["plant"]
            start_date = form.cleaned_data["start_date"]
            end_date = form.cleaned_data["end_date"]
            persist = bool(form.cleaned_data.get("persist"))
            want_hourly = bool(form.cleaned_data.get("want_hourly"))
            source_oper = (form.cleaned_data.get("source_oper") or "SHINEMONITOR").strip()
            source_meteo = (form.cleaned_data.get("source_meteo") or "OPENMETEO").strip()

            tz_name = getattr(plant, "timezone", None) or "UTC"

            # intervalo UTC [start,end)
            dt_start_utc, dt_end_utc = _local_dates_to_utc_range(start_date, end_date, tz_name)

            # executa merge
            run = build_plant_merged_dataset(
                plant=plant,
                dt_start_utc=dt_start_utc,
                dt_end_utc=dt_end_utc,
                want_hourly=want_hourly,
                persist=persist,
                source_oper=source_oper,
                source_meteo=source_meteo,
                interval_min=15,
            )

            stats = run.stats

            # Preview com conversão correta para timezone local
            cols15, rows15 = _df_preview(run.df15, tz_name=tz_name, n=120)

            if want_hourly:
                colsh, rowsh = _df_preview(run.df_hour, tz_name=tz_name, n=72)

            if persist:
                messages.success(
                    request,
                    f"Merge executado e persistido: {stats.get('saved_rows_15m', 0)} linhas (15 min).",
                )
            else:
                messages.info(
                    request,
                    f"Merge executado (sem persistir). Linhas 15 min: {stats.get('merged_rows_15', 0)}.",
                )
    else:
        form = MergeRunForm(user=request.user)

    return render(
        request,
        "merge/merge_run.html",
        {
            "form": form,
            "stats": stats,
            "cols15": cols15,
            "rows15": rows15,
            "colsh": colsh,
            "rowsh": rowsh,
        },
    )
