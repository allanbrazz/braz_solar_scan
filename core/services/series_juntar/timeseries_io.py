from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence, Dict, Any, Literal, Tuple, List

import math
import pandas as pd
from django.apps import apps

from core.models import PVPlant, MeteoRecord, MeteoSource


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class FetchConfig:
    """
    Extração do banco -> DataFrames canônicos.

    Meteo:
      - sai com colunas conforme MeteoRecord

    Inversor:
      - lê InverterOperationalData (ts_utc + payload)
      - extrai métricas canônicas do payload
      - opcional: corrige possível ts_utc errado comparando com Data E Hora do payload

    IMPORTANTE (Renovigi/ShineMonitor):
      - payload["Data E Hora"] costuma vir UTC-naive em muitos cenários (sem offset).
      - Portanto, o modo correto geralmente é "utc" (ou "auto" para detectar).
    """

    # ---- MeteoRecord -> DataFrame
    meteo_cols: Sequence[str] = (
        "ts_utc",
        "ghi", "dni", "dhi", "gti",
        "temp_air", "wind_speed", "rh", "pressure",
        "meteo_qc_score", "flag_meteo_low_confidence", "flag_meteo_interpolated",
        "flag_meteo_outlier", "flag_meteo_artifact",
        "interval_min",
    )
    meteo_source: str = MeteoSource.OPENMETEO

    # ---- Inversor: Model real
    inverter_model_label: str = "core.InverterOperationalData"
    inv_plant_fk: str = "plant"

    # Campos do model operacional
    inv_ts_field: str = "ts_utc"
    inv_payload_field: str = "payload"
    inv_provider_field: str = "provedor"

    # Chave de tempo dentro do payload
    inv_payload_time_key: str = "Data E Hora"

    # Como interpretar "Data E Hora" quando vier SEM TZ no payload:
    # - "utc": tratar como UTC-naive
    # - "plant_local": tratar como local-naive da planta
    # - "auto": tenta ambos e escolhe o que minimiza |shift| mediano
    inv_payload_time_mode: Literal["auto", "utc", "plant_local"] = "auto"

    # Buffer para busca no banco antes de corrigir timestamp (em horas)
    inv_query_buffer_hours: int = 6

    # Se a diferença mediana for maior que isso, aplicamos correção (minutos)
    inv_min_shift_to_apply_minutes: int = 30


# -----------------------------
# Helpers gerais
# -----------------------------
def _to_utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _get_plant_tz(plant: PVPlant) -> str:
    v = getattr(plant, "timezone", None)
    if isinstance(v, str) and v:
        return v
    return "UTC"


def _parse_float(v: Any) -> float:
    """
    Converte strings numéricas pt/en para float.
    Aceita: "1.234,56" / "1234.56" / "  20 " / 20 / None.
    """
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return float("nan")

        allowed = "0123456789-.,"  # remove unidades
        s = "".join(ch for ch in s if ch in allowed)
        if not s or s in ("-", ".", ","):
            return float("nan")

        # pt_BR: 1.234,56  | en_US: 1234.56
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            if "," in s and "." not in s:
                s = s.replace(",", ".")

        try:
            return float(s)
        except Exception:
            return float("nan")

    return float("nan")


def _mean_nonzero(vals: List[float], min_abs: float = 1e-6) -> float:
    xs = [x for x in vals if x is not None and isinstance(x, (int, float)) and math.isfinite(x) and abs(x) > min_abs]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _weighted_vdc(mppt_v: List[float], mppt_i: List[float]) -> Tuple[float, float]:
    """
    Retorna (v_dc_v, i_dc_a) a partir de MPPTs:
      - i_dc_a = soma correntes positivas
      - v_dc_v = média ponderada por corrente (ΣViIi / ΣIi)
    """
    pairs = []
    for v, i in zip(mppt_v, mppt_i):
        if v is None or i is None:
            continue
        if not (isinstance(v, (int, float)) and isinstance(i, (int, float))):
            continue
        if not (math.isfinite(v) and math.isfinite(i)):
            continue
        if i <= 0:
            continue
        pairs.append((v, i))

    if not pairs:
        return float("nan"), float("nan")

    i_sum = sum(i for _, i in pairs)
    if i_sum <= 0:
        return float("nan"), float("nan")

    v_w = sum(v * i for v, i in pairs) / i_sum
    return float(v_w), float(i_sum)


# -----------------------------
# Extratores por provedor (payload)
# -----------------------------
def _extract_renovigi(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrai métricas canônicas do payload RENOVIGI (chaves PT-BR).
    Inclui MPPT1..MPPT4: V/I e Pdc estimado (V*I) por MPPT.
    """
    pac = _parse_float(payload.get("potência ativa total") or payload.get("Potência ativa total"))
    pdc_total = _parse_float(payload.get("Potência de saída CC") or payload.get("Potência de saída CC "))

    mppt_v = [
        _parse_float(payload.get("Tensão CC MPPT1")),
        _parse_float(payload.get("Tensão CC MPPT2")),
        _parse_float(payload.get("Tensão CC MPPT3")),
        _parse_float(payload.get("Tensão CC MPPT4")),
    ]
    mppt_i = [
        _parse_float(payload.get("Corrente CC MPPT1")),
        _parse_float(payload.get("Corrente CC MPPT2")),
        _parse_float(payload.get("Corrente CC MPPT3")),
        _parse_float(payload.get("Corrente CC MPPT4")),
    ]

    # Agregado total
    vdc, idc = _weighted_vdc(mppt_v, mppt_i)

    # Pdc por MPPT = V*I (se ambos finitos e I>0)
    mppt_p = []
    for v, i in zip(mppt_v, mppt_i):
        if isinstance(v, (int, float)) and isinstance(i, (int, float)) and math.isfinite(v) and math.isfinite(i) and i > 0:
            mppt_p.append(float(v * i))
        else:
            mppt_p.append(float("nan"))

    v_ph = [
        _parse_float(payload.get("Tensão de fase R")),
        _parse_float(payload.get("Tensão de fase S")),
        _parse_float(payload.get("Tensão de fase T")),
    ]
    i_ph = [
        _parse_float(payload.get("Corrente de fase A")),
        _parse_float(payload.get("Corrente de fase B")),
        _parse_float(payload.get("Corrente de fase C")),
    ]
    vac = _mean_nonzero(v_ph, min_abs=5.0)
    iac = _mean_nonzero(i_ph, min_abs=0.05)

    return {
        "p_ac_w": pac,
        "p_dc_w": pdc_total,  # total do payload (se existir)

        "v_dc_v": vdc,
        "i_dc_a": idc,
        "v_ac_v": vac,
        "i_ac_a": iac,

        # MPPTs (V/I) + Pdc estimado
        "mppt1_v_dc_v": mppt_v[0],
        "mppt2_v_dc_v": mppt_v[1],
        "mppt3_v_dc_v": mppt_v[2],
        "mppt4_v_dc_v": mppt_v[3],

        "mppt1_i_dc_a": mppt_i[0],
        "mppt2_i_dc_a": mppt_i[1],
        "mppt3_i_dc_a": mppt_i[2],
        "mppt4_i_dc_a": mppt_i[3],

        "mppt1_p_dc_w": mppt_p[0],
        "mppt2_p_dc_w": mppt_p[1],
        "mppt3_p_dc_w": mppt_p[2],
        "mppt4_p_dc_w": mppt_p[3],
    }


def _extract_generic(payload: Dict[str, Any]) -> Dict[str, float]:
    nan = float("nan")
    return {
        "p_ac_w": nan,
        "p_dc_w": nan,
        "v_dc_v": nan,
        "i_dc_a": nan,
        "v_ac_v": nan,
        "i_ac_a": nan,
        "mppt1_v_dc_v": nan, "mppt2_v_dc_v": nan, "mppt3_v_dc_v": nan, "mppt4_v_dc_v": nan,
        "mppt1_i_dc_a": nan, "mppt2_i_dc_a": nan, "mppt3_i_dc_a": nan, "mppt4_i_dc_a": nan,
        "mppt1_p_dc_w": nan, "mppt2_p_dc_w": nan, "mppt3_p_dc_w": nan, "mppt4_p_dc_w": nan,
    }


def _extract_payload(provider: str, payload: Dict[str, Any]) -> Dict[str, float]:
    prov = (provider or "").upper()
    if prov == "RENOVIGI":
        return _extract_renovigi(payload or {})
    return _extract_generic(payload or {})


def _localize_payload_time_to_utc(
    s_payload: pd.Series,
    *,
    plant_tz: str,
    mode: Literal["auto", "utc", "plant_local"],
) -> tuple[pd.Series, str]:
    """
    Recebe série datetime (pandas) possivelmente naive e devolve:
      - série tz-aware em UTC (Timestamp UTC)
      - modo efetivo usado ("utc" | "plant_local" | "tzaware")

    Regras:
      - se vier tz-aware: converte direto pra UTC e retorna "tzaware"
      - se naive:
          * mode="utc": localiza como UTC
          * mode="plant_local": localiza como plant_tz e converte pra UTC
          * mode="auto": retorna naive; a escolha será feita no fetch (comparando shifts)
    """
    dt = pd.to_datetime(s_payload, errors="coerce")

    # tz-aware?
    try:
        if dt.dt.tz is not None:
            return dt.dt.tz_convert("UTC"), "tzaware"
    except Exception:
        pass

    if mode == "utc":
        out = dt.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
        return out, "utc"

    if mode == "plant_local":
        out = dt.dt.tz_localize(plant_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        return out, "plant_local"

    # auto
    return dt, "auto"


# -----------------------------
# Meteo
# -----------------------------
def fetch_meteo_df(
    *,
    plant: PVPlant,
    dt_start_utc: datetime,
    dt_end_utc: datetime,
    cfg: FetchConfig = FetchConfig(),
) -> pd.DataFrame:
    if dt_start_utc.tzinfo is None or dt_end_utc.tzinfo is None:
        raise ValueError("dt_start_utc e dt_end_utc devem ser timezone-aware (UTC).")

    qs = (
        MeteoRecord.objects
        .filter(
            plant=plant,
            source=cfg.meteo_source,
            ts_utc__gte=dt_start_utc,
            ts_utc__lt=dt_end_utc,
        )
        .values(*cfg.meteo_cols)
        .order_by("ts_utc")
    )

    rows = list(qs)
    if not rows:
        return pd.DataFrame(columns=list(cfg.meteo_cols))

    df = pd.DataFrame.from_records(rows)
    df["ts_utc"] = _to_utc_series(df["ts_utc"])

    for c in df.columns:
        if c != "ts_utc":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# Operativo (InverterOperationalData)
# -----------------------------
def fetch_inverter_df(
    *,
    plant: PVPlant,
    dt_start_utc: datetime,
    dt_end_utc: datetime,
    cfg: FetchConfig = FetchConfig(),
) -> pd.DataFrame:
    if dt_start_utc.tzinfo is None or dt_end_utc.tzinfo is None:
        raise ValueError("dt_start_utc e dt_end_utc devem ser timezone-aware (UTC).")

    Model = apps.get_model(cfg.inverter_model_label)
    if Model is None:
        raise LookupError(
            f"Não encontrei o model '{cfg.inverter_model_label}'. "
            "Confirme que ele existe e está em INSTALLED_APPS."
        )

    plant_tz = _get_plant_tz(plant)

    # 1) busca com buffer
    buf_h = int(cfg.inv_query_buffer_hours)
    dt0 = dt_start_utc - timedelta(hours=buf_h)
    dt1 = dt_end_utc + timedelta(hours=buf_h)

    filters = {
        cfg.inv_plant_fk: plant,
        f"{cfg.inv_ts_field}__gte": dt0,
        f"{cfg.inv_ts_field}__lt": dt1,
    }

    raw_fields = [cfg.inv_ts_field, cfg.inv_provider_field, cfg.inv_payload_field]
    qs = Model.objects.filter(**filters).values(*raw_fields).order_by(cfg.inv_ts_field)
    rows = list(qs)

    base_cols = [
        "ts_utc",
        "p_dc_w", "p_ac_w", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a",
        "mppt1_v_dc_v","mppt2_v_dc_v","mppt3_v_dc_v","mppt4_v_dc_v",
        "mppt1_i_dc_a","mppt2_i_dc_a","mppt3_i_dc_a","mppt4_i_dc_a",
        "mppt1_p_dc_w","mppt2_p_dc_w","mppt3_p_dc_w","mppt4_p_dc_w",
    ]

    if not rows:
        out = pd.DataFrame(columns=base_cols)
        out.attrs["meta"] = {
            "plant_tz": plant_tz,
            "inv_rows_raw": 0,
            "inv_rows_in_window": 0,
            "payload_time_mode_used": "",
            "ts_shift_h_median": 0.0,
            "ts_shift_h_min": 0.0,
            "ts_shift_h_max": 0.0,
            "ts_shift_applied_minutes": 0.0,
        }
        return out

    df_raw = pd.DataFrame.from_records(rows)
    df_raw = df_raw.rename(columns={cfg.inv_ts_field: "ts_utc"})
    df_raw["ts_utc"] = _to_utc_series(df_raw["ts_utc"])

    providers = df_raw[cfg.inv_provider_field].astype(str).tolist()
    payloads = df_raw[cfg.inv_payload_field].tolist()

    metrics = [_extract_payload(p, pl or {}) for p, pl in zip(providers, payloads)]
    df_m = pd.DataFrame.from_records(metrics)

    df = pd.concat([df_raw[["ts_utc"]], df_m], axis=1)

    # 2) tentativa de correção de timestamp comparando com payload["Data E Hora"]
    payload_time = []
    for pl in payloads:
        if isinstance(pl, dict):
            payload_time.append(pl.get(cfg.inv_payload_time_key))
        else:
            payload_time.append(None)

    s_payload = pd.to_datetime(pd.Series(payload_time), errors="coerce")

    shift_td = pd.Timedelta(0)
    shift_stats = {"median": 0.0, "min": 0.0, "max": 0.0}
    payload_time_mode_used = ""

    if s_payload.notna().any():
        s_payload_utc, mode0 = _localize_payload_time_to_utc(
            s_payload, plant_tz=plant_tz, mode=cfg.inv_payload_time_mode
        )

        if mode0 == "tzaware":
            payload_time_mode_used = "tzaware"
            delta = (s_payload_utc - df["ts_utc"])
            dh = delta.dt.total_seconds() / 3600.0

        elif cfg.inv_payload_time_mode in ("utc", "plant_local"):
            payload_time_mode_used = mode0
            delta = (s_payload_utc - df["ts_utc"])
            dh = delta.dt.total_seconds() / 3600.0

        else:
            # AUTO: testa UTC-naive vs LOCAL-naive e escolhe o que minimiza |shift| mediano.
            cand_utc = s_payload.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            cand_local = s_payload.dt.tz_localize(plant_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

            dh_utc = (cand_utc - df["ts_utc"]).dt.total_seconds() / 3600.0
            dh_local = (cand_local - df["ts_utc"]).dt.total_seconds() / 3600.0

            dh_utc_valid = dh_utc.dropna()
            dh_local_valid = dh_local.dropna()

            if dh_utc_valid.empty and dh_local_valid.empty:
                dh = pd.Series([], dtype="float64")
                payload_time_mode_used = "auto:none"
            elif dh_local_valid.empty:
                dh = dh_utc
                payload_time_mode_used = "utc"
            elif dh_utc_valid.empty:
                dh = dh_local
                payload_time_mode_used = "plant_local"
            else:
                med_abs_utc = float(dh_utc_valid.abs().median())
                med_abs_local = float(dh_local_valid.abs().median())
                if med_abs_utc <= med_abs_local:
                    dh = dh_utc
                    payload_time_mode_used = "utc"
                else:
                    dh = dh_local
                    payload_time_mode_used = "plant_local"

        dh_valid = dh.dropna()
        if not dh_valid.empty:
            med_h = float(dh_valid.median())
            mn_h = float(dh_valid.min())
            mx_h = float(dh_valid.max())

            # arredonda para múltiplo de 15 min mais próximo
            shift_min = int(round((med_h * 60.0) / 15.0) * 15)

            # aplica só se relevante
            if abs(shift_min) >= int(cfg.inv_min_shift_to_apply_minutes):
                shift_td = pd.Timedelta(minutes=shift_min)
                df["ts_utc"] = df["ts_utc"] + shift_td

            shift_stats = {"median": med_h, "min": mn_h, "max": mx_h}

    # 3) recorta no intervalo correto (após eventual shift)
    inv_rows_raw = int(len(df))
    df = df[(df["ts_utc"] >= dt_start_utc) & (df["ts_utc"] < dt_end_utc)].copy()
    inv_rows_in_window = int(len(df))

    # numéricos (inclui MPPTs)
    for c in base_cols:
        if c == "ts_utc":
            continue
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("ts_utc").reset_index(drop=True)

    df.attrs["meta"] = {
        "plant_tz": plant_tz,
        "inv_rows_raw": inv_rows_raw,
        "inv_rows_in_window": inv_rows_in_window,
        "payload_time_mode_used": payload_time_mode_used,
        "ts_shift_h_median": float(shift_stats["median"]),
        "ts_shift_h_min": float(shift_stats["min"]),
        "ts_shift_h_max": float(shift_stats["max"]),
        "ts_shift_applied_minutes": float(shift_td.total_seconds() / 60.0),
    }

    return df