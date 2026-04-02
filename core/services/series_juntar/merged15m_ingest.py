# core/services/merged15m_ingest.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone as dt_timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from django.apps import apps
from django.conf import settings
from django.db import transaction
from django.utils import timezone as dj_timezone

from core.services.series_juntar.timeseries_merge import (
    DEFAULT_INV_MEAN_COLS,
    DEFAULT_METEO_COLS,
    InverterAggregationConfig,
    MeteoPreparationConfig,
    build_merged_15min,
    densify_15min_grid,
)

UTC = dt_timezone.utc

# =========================
# Defaults / labels de models
# =========================
DEFAULT_INVERTER_SAMPLE_MODEL = "core.InverterSample"
DEFAULT_METEO_15M_MODEL = "core.PVPlantMeteoRecord15m"
DEFAULT_MERGED_15M_MODEL = "core.PVPlantMergedRecord15m"

# Tentamos mapear chaves comuns no JSON do inversor -> colunas normalizadas
DEFAULT_INV_JSON_KEY_CANDIDATES: Dict[str, Sequence[str]] = {
    "p_dc_w": ("p_dc_w", "pdc", "p_dc", "Pdc", "power_dc", "dc_power"),
    "p_ac_w": ("p_ac_w", "pac", "p_ac", "Pac", "power_ac", "ac_power"),
    "v_dc_v": ("v_dc_v", "udc", "vdc", "v_dc", "Vdc", "dc_voltage", "pv_voltage"),
    "i_dc_a": ("i_dc_a", "idc", "i_dc", "Idc", "dc_current", "pv_current"),
    "v_ac_v": ("v_ac_v", "uac", "vac", "v_ac", "Vac", "ac_voltage", "grid_voltage"),
    "i_ac_a": ("i_ac_a", "iac", "i_ac", "Iac", "ac_current", "grid_current"),
    "freq_hz": ("freq_hz", "freq", "fac", "fac_hz", "frequency", "grid_frequency", "grid_freq", "frq", "f_ac", "ac_freq", "frequencia", "frequencia_rede"),
}

DEFAULT_METEO_JSON_KEY_CANDIDATES: Dict[str, Sequence[str]] = {
    "ghi": ("ghi", "GHI"),
    "dni": ("dni", "DNI"),
    "dhi": ("dhi", "DHI"),
    "gti": ("gti", "poa", "g_poa", "GTI", "GPOA"),
    "temp_air": ("temp_air", "t2m", "temperature_2m"),
    "wind_speed": ("wind_speed", "wspd", "windspeed_10m"),
    "rh": ("rh", "relative_humidity_2m"),
    "pressure": ("pressure", "surface_pressure"),
}


# =========================
# Helpers
# =========================
def _get_model(label: str):
    try:
        app_label, model_name = label.split(".", 1)
    except ValueError:
        raise RuntimeError(f"Model label inválido: {label}. Use 'app.Model'.")
    m = apps.get_model(app_label, model_name)
    if m is None:
        raise RuntimeError(f"Não foi possível resolver o model: {label}")
    return m


def _model_field_names(model) -> set:
    return {f.name for f in model._meta.get_fields() if hasattr(f, "name")}


def _ensure_aware_utc(dt: datetime) -> datetime:
    """
    Garante datetime timezone-aware em UTC.
    Compatível com Django 5+ (não usa django.utils.timezone.utc).
    """
    if dt.tzinfo is None:
        return dj_timezone.make_aware(dt, timezone=UTC)
    return dt.astimezone(UTC)


def _as_utc_range(start_utc: datetime, end_utc: datetime) -> Tuple[datetime, datetime]:
    s = _ensure_aware_utc(start_utc)
    e = _ensure_aware_utc(end_utc)
    if e <= s:
        raise ValueError("end_utc precisa ser > start_utc")
    return s, e


def _pick_first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _json_rows_to_df(
    rows: Sequence[Dict[str, Any]],
    *,
    ts_col: str,
    json_col: str,
    out_ts_col: str = "ts_utc",
    key_candidates: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Converte lista de dicts (com coluna timestamp + coluna json dict) em DataFrame
    com colunas normalizadas + ts_utc.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if ts_col not in df.columns:
        raise ValueError(f"Rows precisam conter '{ts_col}'")
    if json_col not in df.columns:
        raise ValueError(f"Rows precisam conter '{json_col}'")

    # ts
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df[out_ts_col] = ts
    df = df.dropna(subset=[out_ts_col]).sort_values(out_ts_col)

    # json expand
    payloads = df[json_col].tolist()
    payloads = [p if isinstance(p, dict) else {} for p in payloads]

    if key_candidates is None:
        key_candidates = {}

    out: Dict[str, List[Any]] = {k: [] for k in key_candidates.keys()}
    for p in payloads:
        for col, cands in key_candidates.items():
            out[col].append(_pick_first_present(p, cands))

    for col, vals in out.items():
        df[col] = pd.to_numeric(pd.Series(vals, index=df.index), errors="coerce")

    keep = [out_ts_col] + list(key_candidates.keys())
    return df[keep].rename(columns={out_ts_col: "ts_utc"})


# =========================
# Extração: Inversor (InverterSample)
# =========================
def load_inverter_samples_df(
    *,
    plant_id: int,
    start_utc: datetime,
    end_utc: datetime,
    device_key: Optional[str] = None,
    inverter_sample_model_label: Optional[str] = None,
    ts_field: str = "ts",
    json_field: str = "data",
    key_candidates: Dict[str, Sequence[str]] = DEFAULT_INV_JSON_KEY_CANDIDATES,
) -> pd.DataFrame:
    """
    Lê InverterSample (raw) e devolve df com:
      ts_utc, p_dc_w, p_ac_w, v_dc_v, i_dc_a, v_ac_v, i_ac_a
    """
    s, e = _as_utc_range(start_utc, end_utc)

    label = inverter_sample_model_label or getattr(settings, "INVERTER_SAMPLE_MODEL", DEFAULT_INVERTER_SAMPLE_MODEL)
    M = _get_model(label)

    qs = M.objects.filter(plant_id=plant_id, **{f"{ts_field}__gte": s, f"{ts_field}__lt": e})
    if device_key:
        qs = qs.filter(device_key=device_key)

    rows = list(qs.values(ts_field, json_field))
    df = _json_rows_to_df(rows, ts_col=ts_field, json_col=json_field, key_candidates=key_candidates)
    if df.empty:
        return df

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    return df


# =========================
# Extração: Meteo 15min
# =========================
def load_meteo_15m_df(
    *,
    plant_id: int,
    start_utc: datetime,
    end_utc: datetime,
    source_meteo: Optional[str] = None,
    meteo_model_label: Optional[str] = None,
    ts_field: str = "ts_utc",
    source_field_candidates: Sequence[str] = ("source", "source_meteo", "provider", "provedor"),
    json_field_candidates: Sequence[str] = ("payload", "data"),
    key_candidates: Dict[str, Sequence[str]] = DEFAULT_METEO_JSON_KEY_CANDIDATES,
) -> pd.DataFrame:
    """
    Lê meteo 15min e devolve df com:
      ts_utc + DEFAULT_METEO_COLS (se existirem)
    """
    s, e = _as_utc_range(start_utc, end_utc)

    label = meteo_model_label or getattr(settings, "PV_METEO_15M_MODEL", DEFAULT_METEO_15M_MODEL)
    M = _get_model(label)
    fields = _model_field_names(M)

    qs = M.objects.filter(plant_id=plant_id, **{f"{ts_field}__gte": s, f"{ts_field}__lt": e})

    src_field = next((f for f in source_field_candidates if f in fields), None)
    if source_meteo and src_field:
        qs = qs.filter(**{src_field: source_meteo})

    explicit_cols = [c for c in DEFAULT_METEO_COLS if c in fields]
    if ts_field in fields and explicit_cols:
        rows = list(qs.values(ts_field, *explicit_cols))
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()
        df["ts_utc"] = pd.to_datetime(df[ts_field], errors="coerce", utc=True)
        for c in explicit_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
        keep = ["ts_utc"] + explicit_cols
        return df[keep]

    json_field = next((f for f in json_field_candidates if f in fields), None)
    if json_field is None:
        return pd.DataFrame()

    rows = list(qs.values(ts_field, json_field))
    df = _json_rows_to_df(rows, ts_col=ts_field, json_col=json_field, key_candidates=key_candidates)
    if df.empty:
        return df

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    return df


# =========================
# Persistência: PVPlantMergedRecord15m (upsert)
# =========================
@dataclass(frozen=True)
class Merged15mPersistConfig:
    merged_model_label: str = DEFAULT_MERGED_15M_MODEL
    ts_field: str = "ts_utc"
    interval_min_field: str = "interval_min"
    interval_min_value: int = 15
    source_oper_field: str = "source_oper"
    source_meteo_field: str = "source_meteo"
    plant_field: str = "plant"
    batch_size: int = 2000


def persist_merged_15m(
    *,
    plant_id: int,
    df_15m: pd.DataFrame,
    source_oper: str,
    source_meteo: str,
    cfg: Merged15mPersistConfig = Merged15mPersistConfig(),
) -> Dict[str, Any]:
    """
    Upsert do dataframe 15min no model PVPlantMergedRecord15m.
    Espera df indexado por ts_15 (DatetimeIndex UTC) OU coluna ts_utc.
    """
    if df_15m is None or df_15m.empty:
        return {"ok": True, "created": 0, "updated": 0, "model": cfg.merged_model_label}

    M = _get_model(cfg.merged_model_label)
    fields = _model_field_names(M)

    df = df_15m.copy()

    # ts
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"ts_15": cfg.ts_field, "index": cfg.ts_field})

    if cfg.ts_field not in df.columns:
        if "ts_15" in df.columns:
            df = df.rename(columns={"ts_15": cfg.ts_field})
        else:
            raise ValueError("df_15m precisa ter DatetimeIndex ou coluna 'ts_utc'/'ts_15'.")

    df[cfg.ts_field] = pd.to_datetime(df[cfg.ts_field], errors="coerce", utc=True)
    df = df.dropna(subset=[cfg.ts_field]).sort_values(cfg.ts_field)

    candidate_cols = (
        list(DEFAULT_INV_MEAN_COLS)
        + ["freq_hz", "e_ac_wh_15", "inv_n", "inv_coverage", "flag_inv_missing", "flag_low_coverage"]
        + list(DEFAULT_METEO_COLS)
        + ["flag_meteo_missing"]
    )
    write_cols = [c for c in candidate_cols if c in df.columns and c in fields]

    for c in write_cols:
        if c.startswith("flag_"):
            df[c] = df[c].fillna(False).astype(bool)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if cfg.interval_min_field in fields:
        df[cfg.interval_min_field] = int(cfg.interval_min_value)
    if cfg.source_oper_field in fields:
        df[cfg.source_oper_field] = str(source_oper)
    if cfg.source_meteo_field in fields:
        df[cfg.source_meteo_field] = str(source_meteo)

    objs = []
    for r in df.itertuples(index=False):
        d = r._asdict()

        kwargs: Dict[str, Any] = {}
        kwargs[f"{cfg.plant_field}_id"] = plant_id

        ts = d.get(cfg.ts_field)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()

        # garante UTC aware
        if ts.tzinfo is None:
            ts = dj_timezone.make_aware(ts, timezone=UTC)
        else:
            ts = ts.astimezone(UTC)
        kwargs[cfg.ts_field] = ts

        if cfg.interval_min_field in fields:
            kwargs[cfg.interval_min_field] = int(d.get(cfg.interval_min_field) or cfg.interval_min_value)
        if cfg.source_oper_field in fields:
            kwargs[cfg.source_oper_field] = d.get(cfg.source_oper_field)
        if cfg.source_meteo_field in fields:
            kwargs[cfg.source_meteo_field] = d.get(cfg.source_meteo_field)

        for c in write_cols:
            v = d.get(c)
            try:
                if pd.isna(v):
                    v = None
            except Exception:
                pass
            kwargs[c] = v

        objs.append(M(**kwargs))

    unique_fields = []
    for k in (cfg.plant_field, cfg.ts_field, cfg.interval_min_field, cfg.source_oper_field, cfg.source_meteo_field):
        if k in fields:
            unique_fields.append(k)

    update_fields = [c for c in write_cols if c in fields]
    for k in (cfg.plant_field, cfg.ts_field, cfg.interval_min_field, cfg.source_oper_field, cfg.source_meteo_field):
        if k in update_fields:
            update_fields.remove(k)

    created = 0
    updated = 0

    with transaction.atomic():
        try:
            for i in range(0, len(objs), cfg.batch_size):
                chunk = objs[i : i + cfg.batch_size]
                res = M.objects.bulk_create(
                    chunk,
                    batch_size=cfg.batch_size,
                    update_conflicts=True,
                    unique_fields=unique_fields,
                    update_fields=update_fields,
                )
                created += len(res)
        except TypeError:
            # fallback: filtra corretamente por (plant, ts, interval, sources)
            ts_list = [getattr(o, cfg.ts_field) for o in objs]
            base_qs = M.objects.filter(
                plant_id=plant_id,
                **{f"{cfg.ts_field}__in": ts_list},
            )
            if cfg.interval_min_field in fields:
                base_qs = base_qs.filter(**{cfg.interval_min_field: cfg.interval_min_value})
            if cfg.source_oper_field in fields:
                base_qs = base_qs.filter(**{cfg.source_oper_field: str(source_oper)})
            if cfg.source_meteo_field in fields:
                base_qs = base_qs.filter(**{cfg.source_meteo_field: str(source_meteo)})

            exist = set(base_qs.values_list(cfg.ts_field, flat=True))

            to_create = [o for o in objs if getattr(o, cfg.ts_field) not in exist]
            if to_create:
                M.objects.bulk_create(to_create, batch_size=cfg.batch_size)
                created += len(to_create)

            to_update = [o for o in objs if getattr(o, cfg.ts_field) in exist]
            if to_update and update_fields:
                M.objects.bulk_update(to_update, fields=update_fields, batch_size=cfg.batch_size)
                updated += len(to_update)

    return {"ok": True, "created": created, "updated": updated, "model": cfg.merged_model_label, "write_cols": write_cols}


# =========================
# PIPELINE 3b: build + densify + persist
# =========================
def build_and_persist_merged15m_for_range(
    *,
    plant_id: int,
    start_utc: datetime,
    end_utc: datetime,
    source_oper: str,
    source_meteo: str,
    device_key: Optional[str] = None,
    tz_work: str = "UTC",
    inv_cfg: InverterAggregationConfig = InverterAggregationConfig(),
    met_cfg: MeteoPreparationConfig = MeteoPreparationConfig(),
) -> Dict[str, Any]:
    s, e = _as_utc_range(start_utc, end_utc)

    df_inv = load_inverter_samples_df(
        plant_id=plant_id,
        start_utc=s,
        end_utc=e,
        device_key=device_key,
    )

    df_met = load_meteo_15m_df(
        plant_id=plant_id,
        start_utc=s,
        end_utc=e,
        source_meteo=source_meteo,
    )

    merged = build_merged_15min(
        df_inv=df_inv,
        df_met=df_met,
        inv_cfg=inv_cfg,
        met_cfg=met_cfg,
        tz_work=tz_work,
        assume_tz_if_naive="UTC",
        how="left",
    )

    if merged is None or merged.empty:
        return {"ok": True, "empty": True, "message": "Sem dados para montar merged_15m.", "created": 0, "updated": 0}

    merged = densify_15min_grid(
        merged,
        start_utc=s,
        end_utc=e,
        freq=inv_cfg.freq,
        coverage_threshold=float(inv_cfg.coverage_threshold),
    )

    res = persist_merged_15m(
        plant_id=plant_id,
        df_15m=merged.reset_index().rename(columns={"ts_15": "ts_utc"}),
        source_oper=source_oper,
        source_meteo=source_meteo,
    )
    res["empty"] = False
    res["rows_df"] = int(len(merged))
    return res


def build_and_persist_merged15m_by_day(
    *,
    plant_id: int,
    start_day: date,
    end_day: date,
    source_oper: str,
    source_meteo: str,
    device_key: Optional[str] = None,
    tz_work: str = "UTC",
    inv_cfg: InverterAggregationConfig = InverterAggregationConfig(),
    met_cfg: MeteoPreparationConfig = MeteoPreparationConfig(),
) -> Dict[str, Any]:
    """
    Processa dia-a-dia em UTC [00:00, 00:00).
    """
    if end_day < start_day:
        raise ValueError("end_day < start_day")

    created = 0
    updated = 0
    days = 0
    errors: List[str] = []

    d = start_day
    while d <= end_day:
        s = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=UTC)
        e = s + timedelta(days=1)
        try:
            res = build_and_persist_merged15m_for_range(
                plant_id=plant_id,
                start_utc=s,
                end_utc=e,
                source_oper=source_oper,
                source_meteo=source_meteo,
                device_key=device_key,
                tz_work=tz_work,
                inv_cfg=inv_cfg,
                met_cfg=met_cfg,
            )
            created += int(res.get("created", 0) or 0)
            updated += int(res.get("updated", 0) or 0)
            days += 1
        except Exception as ex:
            errors.append(f"{d.isoformat()}: {type(ex).__name__}: {ex}")
        d = d + timedelta(days=1)

    return {"ok": not errors, "days": days, "created": created, "updated": updated, "errors": errors}
