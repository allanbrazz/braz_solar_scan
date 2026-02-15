from __future__ import annotations
from core.views._imports import *
from core.services.dados_satelite.nsrdb import fetch_nsrdb_goes_full_disc_csv

# Forms
from core.forms import (
    NSRDBForm,
)


# -------------------------
# NSRDB
# -------------------------


def _nsrdb_env() -> dict:
    api_key = os.environ.get("NREL_API_KEY") or os.environ.get("NSRDB_API_KEY") or getattr(settings, "NREL_API_KEY", None)
    full_name = os.environ.get("NREL_FULL_NAME") or os.environ.get("NSRDB_FULL_NAME") or getattr(settings, "NREL_FULL_NAME", None)
    email = os.environ.get("NREL_EMAIL") or os.environ.get("NSRDB_EMAIL") or getattr(settings, "NREL_EMAIL", None)

    affiliation = os.environ.get("NREL_AFFILIATION") or os.environ.get("NSRDB_AFFILIATION") or getattr(settings, "NREL_AFFILIATION", "UTEC")
    reason = os.environ.get("NREL_REASON") or os.environ.get("NSRDB_REASON") or getattr(settings, "NREL_REASON", "research")

    missing = [k for k, v in {"API_KEY": api_key, "FULL_NAME": full_name, "EMAIL": email}.items() if not v]
    if missing:
        raise ValueError(f"Credenciais NSRDB ausentes em env: {', '.join(missing)}")

    return dict(api_key=api_key, full_name=full_name, email=email, affiliation=affiliation, reason=reason)


def _nsrdb_make_datetime_index(df: pd.DataFrame, utc_flag: bool) -> pd.DataFrame:
    needed = {"Year", "Month", "Day", "Hour", "Minute"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV NSRDB sem colunas de tempo esperadas. Achei: {list(df.columns)[:25]} ...")

    ts = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]], errors="coerce")
    out = df.copy()
    out["datetime"] = ts
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # NSRDB retorna timestamps "naive":
    # - utc=true: representa UTC
    # - utc=false: representa Local Standard Time (LST)
    if utc_flag:
        out.index = out.index.tz_localize("UTC")

    return out


def _nsrdb_normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes compatíveis com o front anterior:
      - air_temperature -> temperature_2m
      - wind_speed -> wind_speed_10m
    """
    out = df.copy()
    if "air_temperature" in out.columns and "temperature_2m" not in out.columns:
        out = out.rename(columns={"air_temperature": "temperature_2m"})
    if "wind_speed" in out.columns and "wind_speed_10m" not in out.columns:
        out = out.rename(columns={"wind_speed": "wind_speed_10m"})
    return out


def _slice_by_dates(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Recorta [start, end], incluindo todo o dia final.
    Funciona para índice tz-naive e tz-aware.
    """
    if df.empty:
        return df

    tz = df.index.tz
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end) + pd.Timedelta(days=1)  # exclusivo

    if tz is not None:
        t0 = t0.tz_localize(tz)
        t1 = t1.tz_localize(tz)

    return df.loc[(df.index >= t0) & (df.index < t1)]


def _nsrdb_cache_key(lat: float, lon: float, year: int, interval_min: int, utc_flag: bool, attributes: str) -> str:
    # normaliza attributes para evitar variações por espaços
    attrs = ",".join([a.strip() for a in (attributes or "").split(",") if a.strip()])
    return f"nsrdb:psm3:{lat:.5f}:{lon:.5f}:{year}:{interval_min}:{int(utc_flag)}:{attrs}"


def _nsrdb_fetch_year_cached(
    *, lat: float, lon: float, year: int,
    interval_min: int, utc_flag: bool, attributes: str,
    timeout_s: int = 120,
) -> tuple[dict, pd.DataFrame]:
    """
    Busca e faz cache de um ano do NSRDB.
    Cache é importante porque a API devolve CSV anual (pesado).
    """
    key = _nsrdb_cache_key(lat, lon, year, interval_min, utc_flag, attributes)
    cached = cache.get(key)
    if cached is not None:
        meta, df = cached
        return meta, df

    creds = _nsrdb_env()
    info, df_raw = fetch_nsrdb_goes_full_disc_csv(
        lat=lat,
        lon=lon,
        year=year,
        api_key=creds["api_key"],
        full_name=creds["full_name"],
        email=creds["email"],
        affiliation=creds["affiliation"],
        reason=creds["reason"],
        interval_min=interval_min,
        utc=utc_flag,
        attributes=attributes,
        timeout_s=timeout_s,
    )

    meta = info.iloc[0].to_dict() if info is not None and len(info) else {}
    df = _nsrdb_make_datetime_index(df_raw, utc_flag=utc_flag)
    df = _nsrdb_normalize_cols(df)

    # Cache por 6h (ajuste conforme uso)
    cache.set(key, (meta, df), timeout=6 * 3600)
    return meta, df


def _nsrdb_fetch_range(
    *, lat: float, lon: float, start: dt.date, end: dt.date,
    interval_min: int, utc_flag: bool, attributes: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Busca 1 ou 2 anos (se o período cruza virada de ano), concatena e recorta.
    Retorna df_all (index datetime) e meta (dict).
    """
    years = sorted(set([start.year, end.year]))
    frames: list[pd.DataFrame] = []
    meta_out: dict = {}

    for y in years:
        meta_y, df_y = _nsrdb_fetch_year_cached(
            lat=lat, lon=lon, year=y,
            interval_min=interval_min,
            utc_flag=utc_flag,
            attributes=attributes,
        )
        # mantém o primeiro meta não-vazio
        if not meta_out and meta_y:
            meta_out = meta_y
        frames.append(df_y)

    df_all = pd.concat(frames).sort_index()
    df_all = _slice_by_dates(df_all, start, end)
    return df_all, meta_out


# -------------------------
# NSRDB: JSON (API/debug)
# -------------------------

def nsrdb_api_json(request: HttpRequest) -> JsonResponse:
    """
    JSON com ghi/dni/dhi + temperature_2m + wind_speed_10m.
    Querystring:
      - lat, lon
      - start, end (YYYY-MM-DD)
      - interval (30|60), utc (0|1)
      - attributes (default: ghi,dhi,dni,wind_speed,air_temperature)
    """
    lat = _get_float(request, "lat", float(os.environ.get("PV_LAT", -34.9)))
    lon = _get_float(request, "lon", float(os.environ.get("PV_LON", -56.2)))

    today = dt.date.today()
    end_str = request.GET.get("end")
    start_str = request.GET.get("start")
    end = dtparser.parse(end_str).date() if end_str else today
    start = dtparser.parse(start_str).date() if start_str else (end - dt.timedelta(days=3))

    interval = int(request.GET.get("interval", 60))
    if interval not in (30, 60):
        interval = 60

    utc_flag = request.GET.get("utc", "0") in ("1", "true", "True", "yes", "on")
    attributes = request.GET.get("attributes", "ghi,dhi,dni,wind_speed,air_temperature")

    try:
        df_all, meta = _nsrdb_fetch_range(
            lat=lat, lon=lon, start=start, end=end,
            interval_min=interval, utc_flag=utc_flag, attributes=attributes
        )
    except Exception as exc:
        return JsonResponse({"error": f"Falha ao consultar NSRDB: {exc}"}, status=502, safe=False)

    if df_all.empty:
        return JsonResponse({"records": [], "meta": {"count": 0, "msg": "sem dados"}}, safe=False)

    out = df_all.reset_index().rename(columns={"datetime": "datetime_ref"})

    # serialização ISO
    if utc_flag:
        # index tz-aware UTC -> ISO Z
        out["datetime_ref"] = out["datetime_ref"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # local standard time (naive)
        out["datetime_ref"] = out["datetime_ref"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    keep = ["datetime_ref"]
    for c in ["ghi", "dni", "dhi", "temperature_2m", "wind_speed_10m"]:
        if c in out.columns:
            keep.append(c)

    out = out[keep].where(pd.notna(out[keep]), None)

    return JsonResponse(
        {
            "records": out.to_dict(orient="records"),
            "meta": {
                "count": len(out),
                "lat": lat,
                "lon": lon,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "interval_min": interval,
                "utc": utc_flag,
                "nsrdb_meta": meta,
            },
        },
        safe=False,
    )


# -------------------------
# Página HTML (usa radiation_view.html)
# -------------------------

@login_required
def nsrdb_view(request: HttpRequest) -> HttpResponse:
    """
    View HTML que reutiliza radiation_view.html.
    Mantive OpenMeteoForm para compatibilidade imediata com o template,
    mas aqui tilt/azimuth NÃO são usados (NSRDB entrega GHI/DNI/DHI).
    """
    today = dt.date.today()
    initial = dict(
        lat=float(os.environ.get("PV_LAT", -34.9)),
        lon=float(os.environ.get("PV_LON", -56.2)),
        tilt=int(os.environ.get("PV_TILT", 25)),      # mantido no form/template
        azimuth=int(os.environ.get("PV_AZIMUTH", 0)), # mantido no form/template
        start=today - dt.timedelta(days=3),
        end=today,
    )
    form = NSRDBForm(request.GET or None, initial=initial)

    df_show = None
    meta: dict = {}

    if form.is_valid():
        cd = form.cleaned_data
        start = cd["start"]
        end = cd["end"]

        interval = int(request.GET.get("interval", 60))
        if interval not in (30, 60):
            interval = 60

        utc_flag = request.GET.get("utc", "0") in ("1", "true", "True", "yes", "on")
        attributes = request.GET.get("attributes", "ghi,dhi,dni,wind_speed,air_temperature")

        try:
            df_all, meta_nsrdb = _nsrdb_fetch_range(
                lat=cd["lat"], lon=cd["lon"],
                start=start, end=end,
                interval_min=interval,
                utc_flag=utc_flag,
                attributes=attributes
            )

            if df_all.empty:
                messages.warning(request, "Sem dados NSRDB para o intervalo escolhido.")
            else:
                # Para exibir em horário local:
                # - se utc_flag=True, converte UTC -> LOCAL_TZ
                # - se utc_flag=False, é LST (naive) e não tem tz; exibimos “como está”
                if utc_flag:
                    df_all = df_all.tz_convert(LOCAL_TZ)

                # RECOMENDADO: manter cadência nativa (evita NaN e é bem mais rápido)
                limit = int(request.GET.get("limit", 500))
                df_show = df_all.tail(limit).reset_index()
                df_show = df_show.where(pd.notna(df_show), None)

                meta = {
                    "count_total": len(df_all),
                    "count_shown": len(df_show),
                    "cadence": f"{interval} min (NSRDB nativo)",
                    "interval_in": interval,
                    "utc": utc_flag,
                    "timezone_display": str(LOCAL_TZ) if utc_flag else "Local Standard Time (NSRDB)",
                    "nsrdb_meta": meta_nsrdb,
                }

        except Exception as exc:
            messages.error(request, f"Falha ao buscar NSRDB: {exc}")

    ctx = {
        "form": form,
        "cols": list(df_show.columns) if df_show is not None else [],
        "rows": list(df_show.itertuples(index=False, name=None)) if df_show is not None else [],
        "meta": meta,
    }
    return render(request, "meteo/radiation_view.html", ctx)

