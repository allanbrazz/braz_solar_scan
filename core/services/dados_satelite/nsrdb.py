# core/services/nsrdb.py
import io
from datetime import datetime, timezone as dt_tz
from zoneinfo import ZoneInfo
from typing import Tuple, Dict, Any, Optional, List

import requests
import pandas as pd
from django.conf import settings
from django.db import transaction

from core.models import MeteoRecord, MeteoSource

# Dataset recomendado para fora de CONUS e para cobertura “full disc”
NSRDB_GOES_FULL_DISC_V4_CSV_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-full-disc-v4-0-0-download.csv"
)

# Conforme documentação do endpoint Full Disc v4 (ajuste se sua fonte suportar novos anos)
GOES_FULL_DISC_SUPPORTED_YEARS = set(range(2018, 2025))  # 2018..2024
GOES_FULL_DISC_SUPPORTED_INTERVALS = {10, 30, 60}

UTC = ZoneInfo("UTC")


def fetch_nsrdb_goes_full_disc_csv(
    *,
    lat: float,
    lon: float,
    year: int,
    api_key: str,
    email: str,
    full_name: str | None = None,
    affiliation: str | None = None,
    reason: str | None = None,
    interval_min: int = 60,
    utc: bool = False,
    attributes: str = "ghi,dhi,dni,wind_speed,air_temperature",
    leap_day: bool = False,
    mailing_list: bool = False,
    timeout_s: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa 1 ano (single site) do NSRDB GOES Full Disc v4 via CSV e retorna:
      - info: DataFrame (1 linha) com metadados (linha 2 do CSV)
      - df: DataFrame com colunas (Year/Month/Day/Hour/Minute + atributos)
    """
    if year not in GOES_FULL_DISC_SUPPORTED_YEARS:
        raise ValueError(
            f"Ano {year} não suportado pelo NSRDB GOES Full Disc v4. "
            f"Use um destes: {sorted(GOES_FULL_DISC_SUPPORTED_YEARS)}"
        )

    if int(interval_min) not in GOES_FULL_DISC_SUPPORTED_INTERVALS:
        raise ValueError(
            f"interval_min={interval_min} inválido. "
            f"Use {sorted(GOES_FULL_DISC_SUPPORTED_INTERVALS)}."
        )

    params = {
        "wkt": f"POINT({lon} {lat})",  # ordem: lon lat
        "names": str(year),
        "interval": str(int(interval_min)),
        "utc": "true" if utc else "false",
        "email": email,
        "api_key": api_key,
        "attributes": attributes,
        "leap_day": "true" if leap_day else "false",
        "mailing_list": "true" if mailing_list else "false",
    }

    # opcionais
    if full_name:
        params["full_name"] = full_name
    if affiliation:
        params["affiliation"] = affiliation
    if reason:
        params["reason"] = reason

    r = requests.get(NSRDB_GOES_FULL_DISC_V4_CSV_URL, params=params, timeout=timeout_s)

    # Se der erro, inclua o corpo para diagnosticar
    if not r.ok:
        body = (r.text or "")[:1500]
        raise RuntimeError(f"NSRDB HTTP {r.status_code}: {body}")

    text = r.text

    # CSV padrão NSRDB: 1ª linha header metadata, 2ª linha metadata, 3ª linha header dados
    info = pd.read_csv(io.StringIO(text), nrows=1)
    df = pd.read_csv(io.StringIO(text), skiprows=2)

    return info, df


def _get_required_setting(name: str) -> str:
    val = getattr(settings, name, None)
    if not val:
        raise RuntimeError(
            f"Setting '{name}' não configurada. Defina em settings.py ou via env var."
        )
    return val


def _normalize_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Cria um mapa: nome_normalizado -> nome_original.
    Ex.: "air_temperature" -> "air_temperature" (ou "Air Temperature", etc.)
    """
    m = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        m[key] = c
    return m


def _build_ts_utc(df: pd.DataFrame) -> pd.Series:
    """
    Constrói ts_utc timezone-aware a partir de Year/Month/Day/Hour/Minute.
    Assumimos que o request foi feito com utc=true.
    """
    required = ["Year", "Month", "Day", "Hour", "Minute"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV NSRDB sem colunas de tempo esperadas: faltando {missing}")

    ts = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]],
        errors="coerce",
        utc=True,  # gera tz-aware em UTC
    )
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise RuntimeError(f"Falha ao parsear timestamps do NSRDB (linhas inválidas: {bad}).")
    return ts


@transaction.atomic
def ingest_nsrdb_range(
    *,
    plant,
    start_utc: datetime,
    end_utc: datetime,
    interval_min: int = 60,
    # credenciais/config (preferência: settings)
    api_key: Optional[str] = None,
    email: Optional[str] = None,
    full_name: Optional[str] = None,
    affiliation: Optional[str] = None,
    reason: Optional[str] = None,
    attributes: str = "ghi,dhi,dni,wind_speed,air_temperature",
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """
    Faz ingestão do NSRDB GOES Full Disc v4 para o intervalo [start_utc, end_utc] (UTC),
    baixando por ano (pois o endpoint é anual).

    Salva em core.models.MeteoRecord com ts_utc canônico (UTC).
    Retorna estatísticas: inserted/updated/total_rows.
    """
    # garantir UTC aware
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc e end_utc devem ser timezone-aware (UTC).")

    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    if end_utc < start_utc:
        raise ValueError("end_utc deve ser >= start_utc.")

    if int(interval_min) not in GOES_FULL_DISC_SUPPORTED_INTERVALS:
        raise ValueError(
            f"interval_min={interval_min} inválido para GOES Full Disc v4. "
            f"Use {sorted(GOES_FULL_DISC_SUPPORTED_INTERVALS)}."
        )

    # credenciais
    api_key = api_key or getattr(settings, "NSRDB_API_KEY", None) or _get_required_setting("NSRDB_API_KEY")
    email = email or getattr(settings, "NSRDB_EMAIL", None) or _get_required_setting("NSRDB_EMAIL")
    full_name = full_name or getattr(settings, "NSRDB_FULL_NAME", None)
    affiliation = affiliation or getattr(settings, "NSRDB_AFFILIATION", None)
    reason = reason or getattr(settings, "NSRDB_REASON", None)

    lat = float(plant.latitude)
    lon = float(plant.longitude)

    years = list(range(start_utc.year, end_utc.year + 1))

    inserted = 0
    updated = 0
    total_rows = 0

    # Para performance, vamos preparar um queryset base
    base_qs = MeteoRecord.objects.filter(plant=plant, source=MeteoSource.NSRDB)

    for year in years:
        if year not in GOES_FULL_DISC_SUPPORTED_YEARS:
            raise ValueError(
                f"O range inclui o ano {year}, mas o GOES Full Disc v4 (config atual) "
                f"suporta apenas {sorted(GOES_FULL_DISC_SUPPORTED_YEARS)}."
            )

        _, df = fetch_nsrdb_goes_full_disc_csv(
            lat=lat,
            lon=lon,
            year=year,
            api_key=api_key,
            email=email,
            full_name=full_name,
            affiliation=affiliation,
            reason=reason,
            interval_min=int(interval_min),
            utc=True,  # IMPORTANTÍSSIMO: traz timestamps em UTC
            attributes=attributes,
            timeout_s=timeout_s,
        )

        if df.empty:
            continue

        # timestamps
        ts_utc = _build_ts_utc(df)
        df = df.copy()
        df["ts_utc"] = ts_utc

        # filtrar pelo range solicitado
        df = df[(df["ts_utc"] >= pd.Timestamp(start_utc)) & (df["ts_utc"] <= pd.Timestamp(end_utc))]
        if df.empty:
            continue

        total_rows += len(df)

        # mapear colunas (robusto a variações de header)
        colmap = _normalize_cols(df)
        def pick(col_key: str):
            return colmap.get(col_key)

        c_ghi = pick("ghi")
        c_dni = pick("dni")
        c_dhi = pick("dhi")
        c_wind = pick("wind_speed")
        c_tair = pick("air_temperature")
        c_rh = pick("relative_humidity") or pick("rh")  # caso exista
        c_pres = pick("pressure")  # caso exista

        # buscar existentes no período do dataframe para separar insert/update
        ts_list = [t.to_pydatetime().astimezone(UTC) for t in df["ts_utc"].dt.to_pydatetime()]
        existing = base_qs.filter(ts_utc__in=ts_list).only("id", "ts_utc")
        existing_map = {e.ts_utc: e for e in existing}

        to_create: List[MeteoRecord] = []
        to_update: List[MeteoRecord] = []

        for _, row in df.iterrows():
            ts = row["ts_utc"].to_pydatetime().astimezone(UTC)

            payload = {
                "interval_min": int(interval_min),
                "ghi": float(row[c_ghi]) if c_ghi and pd.notna(row[c_ghi]) else None,
                "dni": float(row[c_dni]) if c_dni and pd.notna(row[c_dni]) else None,
                "dhi": float(row[c_dhi]) if c_dhi and pd.notna(row[c_dhi]) else None,
                "wind_speed": float(row[c_wind]) if c_wind and pd.notna(row[c_wind]) else None,
                "temp_air": float(row[c_tair]) if c_tair and pd.notna(row[c_tair]) else None,
                "rh": float(row[c_rh]) if c_rh and pd.notna(row[c_rh]) else None,
                "pressure": float(row[c_pres]) if c_pres and pd.notna(row[c_pres]) else None,
            }

            obj = existing_map.get(ts)
            if obj is None:
                to_create.append(MeteoRecord(
                    plant=plant,
                    source=MeteoSource.NSRDB,
                    ts_utc=ts,
                    **payload
                ))
            else:
                # atualizar campos no objeto existente
                obj.interval_min = payload["interval_min"]
                obj.ghi = payload["ghi"]
                obj.dni = payload["dni"]
                obj.dhi = payload["dhi"]
                obj.wind_speed = payload["wind_speed"]
                obj.temp_air = payload["temp_air"]
                obj.rh = payload["rh"]
                obj.pressure = payload["pressure"]
                to_update.append(obj)

        # bulk ops em chunks
        if to_create:
            MeteoRecord.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=5000)
            inserted += len(to_create)

        if to_update:
            MeteoRecord.objects.bulk_update(
                to_update,
                fields=["interval_min", "ghi", "dni", "dhi", "wind_speed", "temp_air", "rh", "pressure"],
                batch_size=5000,
            )
            updated += len(to_update)

    return {"inserted": inserted, "updated": updated, "total": total_rows}
