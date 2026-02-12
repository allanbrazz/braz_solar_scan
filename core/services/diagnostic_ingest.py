# =================================
# core/services/diagnostic_ingest.py
# Diagnostic pipeline (Random Forest) + DB upsert (PlantDiagnostic15m)
# =================================
from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from django.apps import apps
from django.db import transaction
from django.utils import timezone as dj_timezone

from core.services.power_model import (
    expected_and_mismatch,
    module_from_pvmodule,
    plant_from_details,
)

# RF
from core.services.fdd.registry import get_default_registry
from core.services.fdd.rf_inference import InferenceConfig


# ============================================================
# Inputs
# ============================================================
@dataclass(frozen=True)
class DiagnosticInputs15m:
    """
    Estrutura mínima de entrada (tudo com mesmo tamanho n):
      - ts_utc: datetime64 / datetime / strings ISO (UTC)
      - g_poa_wm2, tamb_c
      - pac_real_w
      - opcionais: vdc/idc/vac, g_clear
    """
    ts_utc: Any
    g_poa_wm2: Any
    tamb_c: Any
    pac_real_w: Any
    vdc_v: Optional[Any] = None
    idc_a: Optional[Any] = None
    vac_v: Optional[Any] = None
    g_clear_wm2: Optional[Any] = None


# ============================================================
# Small helpers
# ============================================================
def _to_np_1d(x: Any, *, n: Optional[int] = None, fill: float = np.nan) -> np.ndarray:
    """
    Converte x para np.ndarray float 1D.
    Se n for informado:
      - x vazio/None -> preenche n com fill
      - tamanho != n -> preenche n com fill (fail-safe)
    """
    if x is None:
        return np.full(int(n or 0), float(fill), dtype=float)

    if hasattr(x, "to_numpy"):
        try:
            x = x.to_numpy()
        except Exception:
            pass

    a = np.asarray(x, dtype=float).reshape(-1)
    if n is None:
        return a

    n0 = int(n)
    if a.size == 0:
        return np.full(n0, float(fill), dtype=float)
    if a.size != n0:
        return np.full(n0, float(fill), dtype=float)
    return a


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO8601 tolerante a 'Z' e retorna datetime aware em UTC."""
    txt = (s or "").strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)


def _normalize_ts_to_python_utc(ts_any: Any) -> List[datetime]:
    """
    Converte timestamps para lista de datetime aware em UTC.
    Aceita:
      - numpy datetime64
      - pandas DatetimeIndex/Series (via to_numpy)
      - lista/array de datetime
      - strings ISO (ex.: '2026-01-05T10:00:00Z')
    """
    if ts_any is None:
        return []

    if hasattr(ts_any, "to_numpy"):
        try:
            ts_any = ts_any.to_numpy()
        except Exception:
            pass

    t = np.asarray(ts_any).reshape(-1)

    # datetime64
    if np.issubdtype(t.dtype, np.datetime64):
        out: List[datetime] = []
        for v in t:
            try:
                if np.isnat(v):
                    out.append(datetime(1970, 1, 1, tzinfo=dt_timezone.utc))
                    continue
                ns = v.astype("datetime64[ns]").astype("int64")
                out.append(datetime.fromtimestamp(float(ns) / 1e9, tz=dt_timezone.utc))
            except Exception:
                out.append(datetime(1970, 1, 1, tzinfo=dt_timezone.utc))
        return out

    # objetos (datetime / str / etc.)
    out: List[datetime] = []
    for v in t:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                out.append(v.replace(tzinfo=dt_timezone.utc))
            else:
                out.append(v.astimezone(dt_timezone.utc))
            continue

        try:
            out.append(_parse_iso_utc(str(v)))
        except Exception:
            out.append(datetime(1970, 1, 1, tzinfo=dt_timezone.utc))
    return out


def _call_expected_and_mismatch(**kwargs) -> Dict[str, Any]:
    """
    Chama expected_and_mismatch de forma robusta (só passa kwargs suportados).
    Isso evita quebra se o power_model mudar nomes de parâmetros.
    """
    sig = inspect.signature(expected_and_mismatch)
    call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    out = expected_and_mismatch(**call_kwargs)  # type: ignore[arg-type]
    return out or {}


def _pick_first_array(res: Dict[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        v = res.get(k, None)
        if v is None:
            continue
        a = np.asarray(v)
        if a.size == 0:
            continue
        return a
    return None


# ============================================================
# Pipeline: expected_and_mismatch -> RF (registry.infer)
# ============================================================
def run_rf_pipeline(
    inputs: DiagnosticInputs15m,
    *,
    pv_module_obj: Any,
    plant_details_obj: Any,
    inverter_obj: Optional[Any] = None,
    use_inverter_eff: bool = True,
    rf_model_name: str = "default",
    rf_cfg: Optional[InferenceConfig] = None,
    g_min_valid: float = 200.0,
    dt_minutes: float = 15.0,
) -> Dict[str, Any]:
    """
    1) module/plant
    2) expected_and_mismatch
    3) registry.infer (RF) -> code/label
    """
    module = module_from_pvmodule(pv_module_obj)
    plant = plant_from_details(
        plant_details_obj,
        inverter=inverter_obj,
        use_inverter_eff=use_inverter_eff,
    )

    # séries base (tamanho é ditado por G)
    G = _to_np_1d(inputs.g_poa_wm2)
    Ta = _to_np_1d(inputs.tamb_c, n=G.size)
    y = _to_np_1d(inputs.pac_real_w, n=G.size)

    vdc = _to_np_1d(inputs.vdc_v, n=G.size) if inputs.vdc_v is not None else None
    idc = _to_np_1d(inputs.idc_a, n=G.size) if inputs.idc_a is not None else None
    vac = _to_np_1d(inputs.vac_v, n=G.size) if inputs.vac_v is not None else None
    gclear = _to_np_1d(inputs.g_clear_wm2, n=G.size) if inputs.g_clear_wm2 is not None else None

    ts_list = _normalize_ts_to_python_utc(inputs.ts_utc)
    # (opcional) alinha por mínimo entre ts e G (evita inconsistência quando ts vem curto)
    if ts_list and len(ts_list) != int(G.size):
        n_al = int(min(len(ts_list), int(G.size)))
        ts_list = ts_list[:n_al]
        G = G[:n_al]
        Ta = Ta[:n_al]
        y = y[:n_al]
        if vdc is not None:
            vdc = vdc[:n_al]
        if idc is not None:
            idc = idc[:n_al]
        if vac is not None:
            vac = vac[:n_al]
        if gclear is not None:
            gclear = gclear[:n_al]

    out_model: Dict[str, Any] = _call_expected_and_mismatch(
        g_poa=G,
        tamb_c=Ta,
        pac_real_w=y,
        module=module,
        plant=plant,
        v_dc_real_v=vdc,
        i_dc_real_a=idc,
        v_ac_real_v=vac,
        g_clear=gclear,
        times_utc=ts_list if ts_list else None,
        g_min_valid=float(g_min_valid),
        dt_minutes=float(dt_minutes),
        window_minutes=60.0,
        compute_norm=True,
        compute_rca=False,
    )

    # RF inference
    reg = get_default_registry()
    try:
        rf_res = reg.infer(
            model_name=rf_model_name,
            out_model=out_model,
            times_utc=ts_list if ts_list else None,
            v_ac_real_v=vac,
            cfg=rf_cfg,
        ) or {}
    except Exception as e:
        # Você quer ver claramente quando RF não está pronto/registrado
        out_model["rf_error"] = f"{type(e).__name__}: {e}"
        raise

    # Extrai code/label de forma tolerante a chaves
    code_arr = _pick_first_array(
        rf_res,
        keys=("code_post", "rf_code", "code", "y_pred", "pred_code", "class_id"),
    )
    label_arr = _pick_first_array(
        rf_res,
        keys=("label_post", "rf_label", "label", "y_label", "pred_label", "class_label"),
    )

    if code_arr is None or label_arr is None:
        # mantém debug da resposta bruta
        out_model["rf_result"] = rf_res
        raise KeyError("RF inference não retornou arrays reconhecíveis (code/label).")

    out_model["rf_code"] = np.asarray(code_arr, dtype=int).reshape(-1)
    out_model["rf_label"] = np.asarray(label_arr, dtype=object).reshape(-1)
    out_model["rf_result"] = rf_res
    return out_model


# ============================================================
# DB Upsert
# ============================================================
def upsert_diagnostics_15m(
    *,
    plant_id: int,
    inputs: DiagnosticInputs15m,
    pv_module_obj: Any,
    plant_details_obj: Any,
    inverter_obj: Optional[Any] = None,
    DiagnosticModelLabel: str = "core.PlantDiagnostic15m",
    method: str = "rf",
    rf_model_name: str = "default",
    rf_cfg: Optional[InferenceConfig] = None,
) -> int:
    """
    Persiste/atualiza PlantDiagnostic15m (UPSERT por plant+ts_utc).
    Retorna quantidade de registros processados.
    """
    Diagnostic = apps.get_model(DiagnosticModelLabel)

    method_norm = (method or "").strip().lower()
    if method_norm and method_norm != "rf":
        raise ValueError("method deve ser 'rf'.")

    out = run_rf_pipeline(
        inputs,
        pv_module_obj=pv_module_obj,
        plant_details_obj=plant_details_obj,
        inverter_obj=inverter_obj,
        use_inverter_eff=True,
        rf_model_name=rf_model_name,
        rf_cfg=rf_cfg,
    )
    codes_raw = out.get("rf_code")
    labels_raw = out.get("rf_label")

    ts_list = _normalize_ts_to_python_utc(inputs.ts_utc)
    if not ts_list:
        return 0

    # Arrays principais
    codes = np.asarray(codes_raw if codes_raw is not None else [], dtype=int).reshape(-1)
    labels = np.asarray(labels_raw if labels_raw is not None else [], dtype=object).reshape(-1)
    valid = np.asarray(out.get("valid", []), dtype=bool).reshape(-1)

    g_poa = np.asarray(out.get("g_poa_used", out.get("g_poa", [])), dtype=float).reshape(-1)
    tcell = np.asarray(out.get("tcell_c", []), dtype=float).reshape(-1)
    pac_real = np.asarray(out.get("pac_real_w", []), dtype=float).reshape(-1)
    pac_model = np.asarray(out.get("pac_expected_w", []), dtype=float).reshape(-1)
    mismatch = np.asarray(out.get("mismatch_rel", []), dtype=float).reshape(-1)

    # Tamanho-alvo robusto: inclui TODOS os vetores usados no loop
    sizes = [
        len(ts_list),
        int(codes.size),
        int(labels.size),
        int(valid.size) if valid.size else len(ts_list),
        int(g_poa.size) if g_poa.size else len(ts_list),
        int(tcell.size) if tcell.size else len(ts_list),
        int(pac_real.size) if pac_real.size else len(ts_list),
        int(pac_model.size) if pac_model.size else len(ts_list),
        int(mismatch.size) if mismatch.size else len(ts_list),
    ]
    n = int(min(s for s in sizes if s is not None and s > 0) or 0)
    if n <= 0:
        return 0

    # corta tudo para n
    ts_list = ts_list[:n]
    codes = codes[:n]
    labels = labels[:n]

    # para vetores que podem vir vazios, preenche com default e corta
    def _ensure_float(arr: np.ndarray, default: float = np.nan) -> np.ndarray:
        if arr.size >= n:
            return arr[:n]
        if arr.size == 0:
            return np.full(n, float(default), dtype=float)
        # arr menor: completa com NaN
        out2 = np.full(n, float(default), dtype=float)
        out2[: arr.size] = arr
        return out2

    def _ensure_bool(arr: np.ndarray, default: bool = False) -> np.ndarray:
        if arr.size >= n:
            return arr[:n]
        if arr.size == 0:
            return np.full(n, bool(default), dtype=bool)
        out2 = np.full(n, bool(default), dtype=bool)
        out2[: arr.size] = arr
        return out2

    valid = _ensure_bool(valid, default=False)
    g_poa = _ensure_float(g_poa, default=np.nan)
    tcell = _ensure_float(tcell, default=np.nan)
    pac_real = _ensure_float(pac_real, default=np.nan)
    pac_model = _ensure_float(pac_model, default=np.nan)
    mismatch = _ensure_float(mismatch, default=np.nan)

    now = dj_timezone.now()

    rows = []
    for i in range(n):
        rows.append(
            Diagnostic(
                plant_id=int(plant_id),
                ts_utc=ts_list[i],
                rca_code=int(codes[i]),
                rca_label=str(labels[i]),
                valid=bool(valid[i]),
                g_poa=(float(g_poa[i]) if np.isfinite(g_poa[i]) else None),
                tcell_c=(float(tcell[i]) if np.isfinite(tcell[i]) else None),
                pac_real_w=(float(pac_real[i]) if np.isfinite(pac_real[i]) else None),
                pac_model_w=(float(pac_model[i]) if np.isfinite(pac_model[i]) else None),
                mismatch_rel=(float(mismatch[i]) if np.isfinite(mismatch[i]) else None),
                created_at=now,   # bulk_create não dispara auto_now_add
                updated_at=now,   # bulk_create não dispara auto_now
            )
        )

    update_fields = [
        "rca_code",
        "rca_label",
        "valid",
        "g_poa",
        "tcell_c",
        "pac_real_w",
        "pac_model_w",
        "mismatch_rel",
        "updated_at",
    ]

    with transaction.atomic():
        # Django >= 4.1: bulk_create(update_conflicts=True)
        try:
            Diagnostic.objects.bulk_create(
                rows,
                update_conflicts=True,
                unique_fields=["plant", "ts_utc"],
                update_fields=update_fields,
            )
        except TypeError:
            # Fallback (Django < 4.1): update_or_create por linha (mais lento, mas compatível)
            for r in rows:
                Diagnostic.objects.update_or_create(
                    plant_id=r.plant_id,
                    ts_utc=r.ts_utc,
                    defaults={
                        "rca_code": r.rca_code,
                        "rca_label": r.rca_label,
                        "valid": r.valid,
                        "g_poa": r.g_poa,
                        "tcell_c": r.tcell_c,
                        "pac_real_w": r.pac_real_w,
                        "pac_model_w": r.pac_model_w,
                        "mismatch_rel": r.mismatch_rel,
                        # NÃO force created_at aqui (senão você “reescreve” created_at em updates)
                    },
                )

    return int(n)
