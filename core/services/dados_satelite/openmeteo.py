# core/services/dados_satelite/openmeteo.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable
from urllib.parse import urlencode

import requests
import pandas as pd
from django.db import transaction

from core.models import (
    MeteoRecord,
    MeteoSource,
    MeteoImportBatch,
    MeteoDataTypology,
    PVPlant,
)
from core.services.meteo_qc import MeteoQCConfig, apply_meteo_qc

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_MIN_DATE = date(2022, 1, 1)

SUPPORTED_INTERVALS_MIN = {60, 15}

DEFAULT_TIME_VARS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
]

GTI_VAR = "global_tilted_irradiance"


@dataclass(frozen=True)
class OpenMeteoFetchResult:
    df: pd.DataFrame
    meta: Dict[str, Any]


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _daterange_chunks(start_d: date, end_d: date, chunk_days: int) -> Iterable[Tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days deve ser >= 1")

    cur = start_d
    while cur <= end_d:
        e = min(cur + timedelta(days=chunk_days - 1), end_d)
        yield cur, e
        cur = e + timedelta(days=1)


def _choose_endpoint(end_d: date, interval_min: int, start_d: date) -> str:
    if interval_min == 15:
        if start_d < HISTORICAL_FORECAST_MIN_DATE:
            raise ValueError(
                f"interval_min=15 exige dados a partir de {HISTORICAL_FORECAST_MIN_DATE.isoformat()} "
                f"(Historical Forecast API)."
            )
        if end_d >= (_utc_today() - timedelta(days=3)):
            return FORECAST_URL
        return HISTORICAL_FORECAST_URL

    if end_d >= (_utc_today() - timedelta(days=3)):
        return FORECAST_URL
    return ARCHIVE_URL


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        x = float(v)
        if not pd.notna(x):
            return None
        return x
    except Exception:
        return None


def _clamp_tilt_deg(v: Optional[float]) -> Optional[float]:
    x = _to_float_or_none(v)
    if x is None:
        return None
    if x < 0.0:
        x = 0.0
    if x > 90.0:
        x = 90.0
    return float(x)


def azimuth_0N_to_openmeteo(az_deg_0_359: Any) -> Optional[float]:
    x = _to_float_or_none(az_deg_0_359)
    if x is None:
        return None

    az = x % 360.0
    om = ((az - 180.0 + 540.0) % 360.0) - 180.0
    if abs(om + 180.0) < 1e-12:
        om = 180.0
    return float(om)


def _normalize_dataset_model(model: Optional[str]) -> str:
    raw = (model or "").strip()
    return raw if raw else "best_match"


def _build_request_url(endpoint: str, params: Dict[str, Any]) -> str:
    try:
        return f"{endpoint}?{urlencode(params, doseq=True)}"
    except Exception:
        return ""


def fetch_openmeteo_hourly(
    *,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    interval_min: int = 60,
    tilt_deg: Optional[float] = None,
    azimuth_deg: Optional[float] = None,
    include_gti: bool = True,
    model: Optional[str] = None,
    timeout_s: int = 60,
) -> OpenMeteoFetchResult:
    if interval_min not in SUPPORTED_INTERVALS_MIN:
        raise ValueError(f"interval_min={interval_min} inválido. Use {sorted(SUPPORTED_INTERVALS_MIN)}.")

    endpoint = _choose_endpoint(end_date, interval_min, start_date)
    vars_ = list(DEFAULT_TIME_VARS)

    tilt_val = _clamp_tilt_deg(tilt_deg)
    az_in_0n = _to_float_or_none(azimuth_deg)
    az_val = azimuth_0N_to_openmeteo(az_in_0n) if az_in_0n is not None else None

    add_gti = bool(include_gti and (tilt_val is not None) and (az_val is not None))
    if add_gti and GTI_VAR not in vars_:
        vars_.append(GTI_VAR)

    if interval_min == 60:
        block_key = "hourly"
        series_param_key = "hourly"
    else:
        block_key = "minutely_15"
        series_param_key = "minutely_15"

    params: Dict[str, Any] = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        series_param_key: ",".join(vars_),
        "timezone": "UTC",
        "timeformat": "iso8601",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }

    if add_gti:
        params["tilt"] = float(tilt_val)
        params["azimuth"] = float(az_val)

    if model:
        params["models"] = model

    r = requests.get(endpoint, params=params, timeout=timeout_s)
    if not r.ok:
        body = (r.text or "")[:1500]
        raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {body}")

    payload = r.json()
    block = payload.get(block_key) or {}
    times = block.get("time") or []

    if not times:
        return OpenMeteoFetchResult(
            df=pd.DataFrame(),
            meta={
                "endpoint": endpoint,
                "params": params,
                "request_url": _build_request_url(endpoint, params),
                "payload": payload,
                "block_key": block_key,
                "gti_requested": add_gti,
                "tilt_deg_used": tilt_val,
                "azimuth_input_0N_deg": az_in_0n,
                "azimuth_openmeteo_deg": az_val,
            },
        )

    df = pd.DataFrame({"ts_utc": pd.to_datetime(times, utc=True)})

    def _col(name: str) -> List[Any]:
        return block.get(name) or [None] * len(df)

    df["ghi"] = _col("shortwave_radiation")
    df["dni"] = _col("direct_normal_irradiance")
    df["dhi"] = _col("diffuse_radiation")
    df["gti"] = _col(GTI_VAR) if add_gti else None

    df["temp_air"] = _col("temperature_2m")
    df["wind_speed"] = _col("wind_speed_10m")
    df["rh"] = _col("relative_humidity_2m")

    sp_hpa = pd.to_numeric(pd.Series(_col("surface_pressure")), errors="coerce")
    df["pressure"] = (sp_hpa * 100.0).where(sp_hpa.notna(), other=None)

    df["interval_min"] = int(interval_min)

    meta = {
        "endpoint": endpoint,
        "params": params,
        "request_url": _build_request_url(endpoint, params),
        "block_key": block_key,
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "timezone": payload.get("timezone"),
        "timezone_abbreviation": payload.get("timezone_abbreviation"),
        "generationtime_ms": payload.get("generationtime_ms"),
        "gti_requested": add_gti,
        "tilt_deg_used": tilt_val,
        "azimuth_input_0N_deg": az_in_0n,
        "azimuth_openmeteo_deg": az_val,
    }
    return OpenMeteoFetchResult(df=df, meta=meta)


def ingest_openmeteo_range(
    *,
    plant: PVPlant,
    start_date: date,
    end_date: date,
    interval_min: int = 15,
    include_gti: bool = True,
    model: Optional[str] = None,
    timeout_s: int = 60,
) -> Tuple[int, Dict[str, Any]]:
    if interval_min not in SUPPORTED_INTERVALS_MIN:
        raise ValueError(f"interval_min={interval_min} inválido. Use {sorted(SUPPORTED_INTERVALS_MIN)}.")

    details = getattr(plant, "details", None)
    tilt = float(details.tilt_deg) if details and getattr(details, "tilt_deg", None) is not None else None
    azim_0n = float(details.azimuth_deg) if details and getattr(details, "azimuth_deg", None) is not None else None

    chunk_days = 31 if interval_min == 15 else 370

    dataset_model = _normalize_dataset_model(model)

    batch = MeteoImportBatch.objects.create(
        plant=plant,
        source=MeteoSource.OPENMETEO,
        source_endpoint="",
        dataset_model=dataset_model,
        data_typology=MeteoDataTypology.REANALYSIS_MODELED,
        interval_min=interval_min,
        start_date=start_date,
        end_date=end_date,
        request_url="",
        request_params={},
        response_meta={},
        imported_rows=0,
    )

    total_count = 0
    metas: List[Dict[str, Any]] = []

    for s, e in _daterange_chunks(start_date, end_date, chunk_days=chunk_days):
        res = fetch_openmeteo_hourly(
            lat=float(plant.latitude),
            lon=float(plant.longitude),
            start_date=s,
            end_date=e,
            interval_min=interval_min,
            tilt_deg=tilt,
            azimuth_deg=azim_0n,
            include_gti=include_gti,
            model=model,
            timeout_s=timeout_s,
        )

        df = res.df
        qc_cfg = MeteoQCConfig(interval_min=int(interval_min), source=MeteoSource.OPENMETEO)
        df, qc_meta = apply_meteo_qc(
            df,
            lat=float(plant.latitude),
            lon=float(plant.longitude),
            cfg=qc_cfg,
        )
        res.meta["qc"] = qc_meta
        metas.append(res.meta)

        if df.empty:
            continue

        objs: List[MeteoRecord] = []
        for row in df.itertuples(index=False):
            objs.append(
                MeteoRecord(
                    plant=plant,
                    source=MeteoSource.OPENMETEO,
                    import_batch=batch,
                    source_endpoint=str(res.meta.get("endpoint") or ""),
                    dataset_model=dataset_model,
                    data_typology=MeteoDataTypology.REANALYSIS_MODELED,
                    ts_utc=row.ts_utc.to_pydatetime(),
                    interval_min=int(row.interval_min),
                    ghi=_to_float_or_none(row.ghi),
                    dni=_to_float_or_none(row.dni),
                    dhi=_to_float_or_none(row.dhi),
                    gti=_to_float_or_none(getattr(row, "gti", None)),
                    temp_air=_to_float_or_none(row.temp_air),
                    wind_speed=_to_float_or_none(row.wind_speed),
                    rh=_to_float_or_none(row.rh),
                    pressure=_to_float_or_none(row.pressure),
                    meteo_qc_score=_to_float_or_none(getattr(row, "meteo_qc_score", None)),
                    flag_meteo_low_confidence=bool(getattr(row, "flag_meteo_low_confidence", False)),
                    flag_meteo_interpolated=bool(getattr(row, "flag_meteo_interpolated", False)),
                    flag_meteo_outlier=bool(getattr(row, "flag_meteo_outlier", False)),
                    flag_meteo_artifact=bool(getattr(row, "flag_meteo_artifact", False)),
                )
            )

        with transaction.atomic():
            try:
                MeteoRecord.objects.bulk_create(
                    objs,
                    batch_size=2000,
                    update_conflicts=True,
                    unique_fields=["plant", "source", "ts_utc"],
                    update_fields=[
                        "import_batch",
                        "source_endpoint",
                        "dataset_model",
                        "data_typology",
                        "interval_min",
                        "ghi",
                        "dni",
                        "dhi",
                        "gti",
                        "temp_air",
                        "wind_speed",
                        "rh",
                        "pressure",
                        "meteo_qc_score",
                        "flag_meteo_low_confidence",
                        "flag_meteo_interpolated",
                        "flag_meteo_outlier",
                        "flag_meteo_artifact",
                    ],
                )
                total_count += len(objs)
            except TypeError:
                MeteoRecord.objects.bulk_create(objs, batch_size=2000, ignore_conflicts=True)
                total_count += len(objs)

    meta_out: Dict[str, Any] = {
        "chunks": len(metas),
        "interval_min": interval_min,
        "dataset_model": dataset_model,
        "endpoints_used": sorted({m.get("endpoint") for m in metas if m.get("endpoint")}),
        "block_keys": sorted({m.get("block_key") for m in metas if m.get("block_key")}),
        "first_chunk": metas[0] if metas else None,
        "last_chunk": metas[-1] if metas else None,
    }

    first_meta = metas[0] if metas else {}
    batch.source_endpoint = str(first_meta.get("endpoint") or "")
    batch.request_params = dict(first_meta.get("params") or {})
    batch.request_url = str(first_meta.get("request_url") or "")
    batch.response_meta = meta_out
    batch.imported_rows = total_count
    batch.save(
        update_fields=[
            "source_endpoint",
            "request_params",
            "request_url",
            "response_meta",
            "imported_rows",
        ]
    )

    return total_count, meta_out