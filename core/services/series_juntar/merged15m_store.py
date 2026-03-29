from __future__ import annotations

from typing import Any, List, Optional
import math
import pandas as pd
from django.db import transaction

from core.models import PVPlant, PVPlantMergedRecord15m


MERGED_COLS = (
    "p_dc_w", "p_ac_w", "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a", "freq_hz",
    "mppt1_vdc_v", "mppt2_vdc_v", "mppt3_vdc_v", "mppt4_vdc_v",
    "mppt1_idc_a", "mppt2_idc_a", "mppt3_idc_a", "mppt4_idc_a",
    "e_ac_wh_15",
    "inv_n", "inv_coverage", "flag_low_coverage",
    "ghi", "dni", "dhi", "gti",
    "temp_air", "wind_speed", "rh", "pressure",
    "meteo_qc_score", "flag_meteo_low_confidence", "flag_meteo_interpolated",
    "flag_meteo_outlier", "flag_meteo_artifact",
    "flag_meteo_missing", "flag_inv_missing",
)

MPPT_KS = (1, 2, 3, 4)


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if pd.isna(v):
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def _to_bool(v: Any) -> bool:
    try:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return bool(v)
    except Exception:
        return False


def _row_getf(row, key: str) -> Optional[float]:
    try:
        return _to_float(row.get(key))
    except Exception:
        return None


def _has_any_mppt_cols(df: pd.DataFrame) -> bool:
    want = []
    for k in MPPT_KS:
        want += [
            f"mppt{k}_p_dc_w",
            f"mppt{k}_v_dc_v", f"mppt{k}_vdc_v",
            f"mppt{k}_i_dc_a", f"mppt{k}_idc_a",
        ]
    return any(c in df.columns for c in want)


def _row_alias_float(row, *keys: str) -> Optional[float]:
    for key in keys:
        try:
            val = row.get(key)
        except Exception:
            val = None
        out = _to_float(val)
        if out is not None:
            return out
    return None


@transaction.atomic
def upsert_merged_15m_df(
    *,
    plant: PVPlant,
    df15: pd.DataFrame,
    source_oper: str,
    source_meteo: str,
    interval_min: int = 15,
    batch_size: int = 2000,
) -> int:
    """
    Persiste df15 (index ts_15) em PVPlantMergedRecord15m.

    NOVO:
      - Se existirem colunas MPPT (mppt1..4_*), grava 4 linhas por timestamp:
          source_oper = f"{source_oper}|MPPT{k}"
        e aloca p_ac_w / e_ac_wh_15 por share DC (Pdc_mppt / sum Pdc_mppt).
      - Remove rows antigos "TOTAL" (source_oper puro) no intervalo gravado
        para evitar dupla contagem no dashboard.

    df15 precisa estar indexado por DatetimeIndex tz-aware (UTC recomendado).
    """
    if df15 is None or df15.empty:
        return 0

    if not isinstance(df15.index, pd.DatetimeIndex) or df15.index.tz is None:
        raise ValueError("df15 deve estar indexado por DatetimeIndex tz-aware (ex.: UTC).")

    idx_utc = df15.index.tz_convert("UTC")
    has_mppt = _has_any_mppt_cols(df15)

    # Se vamos gravar MPPT, remova rows "TOTAL" existentes no intervalo (evita soma duplicada)
    if has_mppt:
        ts_min = idx_utc.min().to_pydatetime()
        ts_max = idx_utc.max().to_pydatetime()
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_oper=str(source_oper),
            source_meteo=str(source_meteo),
            interval_min=int(interval_min),
            ts_utc__gte=ts_min,
            ts_utc__lte=ts_max,
        ).delete()

    objs: List[PVPlantMergedRecord15m] = []

    for i, ts in enumerate(idx_utc):
        row = df15.iloc[i]

        # Meteo (replicado)
        met = {
            "ghi": _to_float(row.get("ghi")),
            "dni": _to_float(row.get("dni")),
            "dhi": _to_float(row.get("dhi")),
            "gti": _to_float(row.get("gti")),
            "temp_air": _to_float(row.get("temp_air")),
            "wind_speed": _to_float(row.get("wind_speed")),
            "rh": _to_float(row.get("rh")),
            "pressure": _to_float(row.get("pressure")),
            "meteo_qc_score": _to_float(row.get("meteo_qc_score")),
            "flag_meteo_low_confidence": _to_bool(row.get("flag_meteo_low_confidence")),
            "flag_meteo_interpolated": _to_bool(row.get("flag_meteo_interpolated")),
            "flag_meteo_outlier": _to_bool(row.get("flag_meteo_outlier")),
            "flag_meteo_artifact": _to_bool(row.get("flag_meteo_artifact")),
            "flag_meteo_missing": _to_bool(row.get("flag_meteo_missing")),
        }

        # Qualidade inversor (replicado)
        inv_n = _to_int(row.get("inv_n"))
        inv_cov = _to_float(row.get("inv_coverage"))
        flags = {
            "inv_n": inv_n,
            "inv_coverage": inv_cov,
            "flag_low_coverage": _to_bool(row.get("flag_low_coverage")),
            "flag_inv_missing": _to_bool(row.get("flag_inv_missing")),
        }

        pac_total = _to_float(row.get("p_ac_w"))
        e_total = _to_float(row.get("e_ac_wh_15"))
        vac = _to_float(row.get("v_ac_v"))
        iac = _to_float(row.get("i_ac_a"))
        freq = _to_float(row.get("freq_hz"))

        if has_mppt:
            pdc_k = []
            vdc_k = []
            idc_k = []
            for k in MPPT_KS:
                pdc_k.append(_row_getf(row, f"mppt{k}_p_dc_w"))
                vdc_k.append(_row_getf(row, f"mppt{k}_v_dc_v"))
                idc_k.append(_row_getf(row, f"mppt{k}_i_dc_a"))

            pdc_sum = 0.0
            valid_any = False
            for p in pdc_k:
                if p is not None and p > 0:
                    pdc_sum += float(p)
                    valid_any = True

            if valid_any and pdc_sum > 0:
                for idx_k, k in enumerate(MPPT_KS):
                    pdc = pdc_k[idx_k]
                    vdc = vdc_k[idx_k]
                    idc = idc_k[idx_k]

                    share = (float(pdc) / pdc_sum) if (pdc is not None and pdc > 0) else 0.0

                    pac = (float(pac_total) * share) if (pac_total is not None and share > 0) else None
                    e15 = (float(e_total) * share) if (e_total is not None and share > 0) else None

                    objs.append(
                        PVPlantMergedRecord15m(
                            plant=plant,
                            source_oper=f"{str(source_oper)}|MPPT{k}",
                            source_meteo=str(source_meteo),
                            interval_min=int(interval_min),
                            ts_utc=ts.to_pydatetime(),

                            # por MPPT
                            p_dc_w=pdc,
                            p_ac_w=pac,
                            v_dc_v=vdc,
                            i_dc_a=idc,

                            # AC (não é por MPPT, mas útil)
                            v_ac_v=vac,
                            i_ac_a=iac,
                            freq_hz=freq,

                            e_ac_wh_15=e15,

                            **flags,
                            **met,
                        )
                    )
                continue  # não grava TOTAL

        # Fallback: grava 1 linha total (como antes)
        objs.append(
            PVPlantMergedRecord15m(
                plant=plant,
                source_oper=str(source_oper),
                source_meteo=str(source_meteo),
                interval_min=int(interval_min),
                ts_utc=ts.to_pydatetime(),

                p_dc_w=_to_float(row.get("p_dc_w")),
                p_ac_w=_to_float(row.get("p_ac_w")),
                v_dc_v=_to_float(row.get("v_dc_v")),
                i_dc_a=_to_float(row.get("i_dc_a")),
                v_ac_v=vac,
                i_ac_a=iac,
                freq_hz=freq,

                e_ac_wh_15=e_total,

                **flags,
                **met,
            )
        )

    # Upsert
    try:
        PVPlantMergedRecord15m.objects.bulk_create(
            objs,
            batch_size=batch_size,
            update_conflicts=True,
            unique_fields=["plant", "source_oper", "source_meteo", "interval_min", "ts_utc"],
            update_fields=list(MERGED_COLS),
        )
        return len(objs)
    except TypeError:
        PVPlantMergedRecord15m.objects.bulk_create(objs, batch_size=batch_size, ignore_conflicts=True)
        return len(objs)