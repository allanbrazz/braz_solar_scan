# core/services/dashboard_charts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# =========================
# Diagnostic codes (independente do skfuzzy)
# =========================
@dataclass(frozen=True)
class DiagnosticCodes:
    INVALID: int = 0
    NORMAL: int = 1
    METEO_ERROR: int = 2
    SOILING: int = 3
    DEGRADATION: int = 4
    SHORT_BYPASS: int = 5
    STRING_DISCONNECTED: int = 6
    PARTIAL_SHADING: int = 7


RCA_TO_CODE = {
    "invalid": DiagnosticCodes.INVALID,
    "normal": DiagnosticCodes.NORMAL,
    "soiling": DiagnosticCodes.SOILING,
    "degradation_like": DiagnosticCodes.DEGRADATION,
    "short_or_bypass": DiagnosticCodes.SHORT_BYPASS,
    "string_disconnected": DiagnosticCodes.STRING_DISCONNECTED,
    "partial_shading": DiagnosticCodes.PARTIAL_SHADING,
    "unknown": DiagnosticCodes.METEO_ERROR,
    "n/a": DiagnosticCodes.METEO_ERROR,
}


CODE_TO_NAME_PT = {
    DiagnosticCodes.INVALID: "Inválido/sem avaliação",
    DiagnosticCodes.NORMAL: "Normal",
    DiagnosticCodes.METEO_ERROR: "Incerteza meteorológica",
    DiagnosticCodes.SOILING: "Sujidade (soiling)",
    DiagnosticCodes.DEGRADATION: "Degradação (like)",
    DiagnosticCodes.SHORT_BYPASS: "Bypass/curto (like)",
    DiagnosticCodes.STRING_DISCONNECTED: "String desconectada",
    DiagnosticCodes.PARTIAL_SHADING: "Sombreamento parcial",
}


# =========================
# JSON safe helpers
# =========================
def _iso_list(times: Any) -> list[str]:
    if times is None:
        return []
    if hasattr(times, "to_numpy"):
        t = times.to_numpy()
    else:
        t = np.asarray(times)
    if t.size == 0:
        return []
    # datetime64 -> ISO
    if np.issubdtype(t.dtype, np.datetime64):
        # garante ms
        t_ns = t.astype("datetime64[ms]")
        return [str(x).replace(" ", "T") for x in t_ns.reshape(-1)]
    return [str(x) for x in t.reshape(-1)]


def _np_list(x: Any) -> list:
    a = np.asarray(x)
    if a.size == 0:
        return []
    if a.dtype == bool:
        return [bool(v) for v in a.reshape(-1)]
    if np.issubdtype(a.dtype, np.number):
        out = []
        for v in a.reshape(-1):
            fv = float(v) if np.isfinite(v) else None
            out.append(fv)
        return out
    return [None if v is None else str(v) for v in a.reshape(-1)]


# =========================
# 2) Gauge de Confiabilidade Meteorológica (G_cv)
# =========================
def meteorological_reliability_from_gcv(
    g_cv_60m: np.ndarray,
    *,
    thr_high: float = 0.10,
    thr_low: float = 0.40,
) -> Dict[str, Any]:
    """
    score: 0..100 (quanto maior, mais confiável)
    gcv_stat: estatística robusta (mediana) para representar o período
    """
    gcv = np.asarray(g_cv_60m, dtype=float)
    gcv = gcv[np.isfinite(gcv)]
    if gcv.size == 0:
        return {"score": None, "gcv_stat": None, "label": "Indisponível"}

    gcv_stat = float(np.nanmedian(gcv))

    # Mapeamento linear por faixas:
    # gcv <= 0.10 => 100
    # gcv >= 0.40 => 0
    x = (gcv_stat - thr_high) / max(thr_low - thr_high, 1e-9)
    x = float(np.clip(x, 0.0, 1.0))
    score = int(round(100.0 * (1.0 - x)))

    if gcv_stat <= thr_high:
        label = "Alta"
    elif gcv_stat >= thr_low:
        label = "Baixa"
    else:
        label = "Média"

    return {"score": score, "gcv_stat": gcv_stat, "label": label, "thr_high": thr_high, "thr_low": thr_low}


# =========================
# 3) Scatter: mismatch_rel vs g_cv_60m, colorido por diagnóstico
# =========================
def _diagnostic_code_series(out_model: Dict[str, Any]) -> np.ndarray:
    # Se você futuramente adicionar out_model["diag_code"], ele entra aqui.
    if "diag_code" in out_model:
        c = np.asarray(out_model["diag_code"])
        return c.astype(int, copy=False)

    valid = np.asarray(out_model.get("valid", []), dtype=bool)
    rca = np.asarray(out_model.get("rca_label", []), dtype=object)
    n = int(max(valid.size, rca.size))
    if n == 0:
        return np.array([], dtype=int)

    if valid.size == 0:
        valid = np.zeros(n, dtype=bool)
    if rca.size == 0:
        rca = np.full(n, "unknown", dtype=object)

    codes = np.full(n, DiagnosticCodes.METEO_ERROR, dtype=int)
    for k, v in RCA_TO_CODE.items():
        codes[np.asarray(rca, dtype=object) == k] = int(v)
    codes[~valid] = DiagnosticCodes.INVALID
    return codes


def scatter_payload(
    *,
    times: Any,
    out_model: Dict[str, Any],
) -> Dict[str, Any]:
    gcv = np.asarray(out_model.get("g_cv_60m", []), dtype=float)
    mr = np.asarray(out_model.get("mismatch_rel", []), dtype=float)
    valid = np.asarray(out_model.get("valid", []), dtype=bool)
    codes = _diagnostic_code_series(out_model)

    n = int(max(gcv.size, mr.size, valid.size, codes.size))
    if n == 0:
        return {"times": [], "x_gcv": [], "y_mismatch": [], "code": [], "code_name": []}

    # pad
    if gcv.size != n:
        gcv = np.resize(gcv, n)
    if mr.size != n:
        mr = np.resize(mr, n)
    if valid.size != n:
        valid = np.resize(valid, n)
    if codes.size != n:
        codes = np.resize(codes, n)

    m = valid & np.isfinite(gcv) & np.isfinite(mr)
    x = gcv[m]
    y = mr[m]
    c = codes[m]

    names = [CODE_TO_NAME_PT.get(int(ci), "n/a") for ci in c.tolist()]

    return {
        "times": _iso_list(np.asarray(times)[m] if times is not None else None),
        "x_gcv": _np_list(x),
        "y_mismatch": _np_list(y),
        "code": [int(v) for v in c.tolist()],
        "code_name": names,
    }


# =========================
# 4) Sankey: perdas energéticas (sequencial)
# =========================
def sankey_energy_payload(
    *,
    out_model: Dict[str, Any],
    dt_minutes: float = 15.0,
    g_ref_stc: float = 1000.0,
) -> Dict[str, Any]:
    """
    Fluxo:
    P_STC -> (escala por G) -> (efeito temperatura) -> (k_sys) -> (eta_inv) -> P_AC_exp

    Observação: aqui a decomposição é "contábil" e coerente com seu pipeline:
    - P_stc_total_w (constante)
    - irradiance scaling: P_stc * (G/1000)
    - DC esperado do 1-diodo (já tem G e T): pdc_expected_w
      => temperatura efetiva = P_dc / P_G  (onde fizer sentido)
    - k_sys aplicado antes do inversor
    - eta_inv aplicado no inversor (já está em pac_expected_w)
    """
    G = np.asarray(out_model.get("g_poa_used", out_model.get("g_poa", [])), dtype=float)
    p_stc = np.asarray(out_model.get("p_stc_w", []), dtype=float)
    p_dc = np.asarray(out_model.get("pdc_expected_w", []), dtype=float)
    p_ac = np.asarray(out_model.get("pac_expected_w", []), dtype=float)
    eta = np.asarray(out_model.get("eta_inv", []), dtype=float)
    valid = np.asarray(out_model.get("valid", []), dtype=bool)

    n = int(max(G.size, p_stc.size, p_dc.size, p_ac.size, eta.size, valid.size))
    if n == 0:
        return {"nodes": [], "links": [], "values_kwh": {}}

    # pad
    def _pad(a, fill=np.nan):
        a = np.asarray(a)
        if a.size == n:
            return a
        if a.size == 0:
            return np.full(n, fill)
        return np.resize(a, n)

    G = _pad(G)
    p_stc = _pad(p_stc)
    p_dc = _pad(p_dc)
    p_ac = _pad(p_ac)
    eta = _pad(eta, fill=0.0)
    valid = _pad(valid, fill=False).astype(bool)

    # máscara útil
    m = valid & np.isfinite(G) & (G > 0)
    if not np.any(m):
        return {"nodes": [], "links": [], "values_kwh": {}}

    dt_h = float(dt_minutes) / 60.0

    P_stc = np.where(m, np.maximum(p_stc, 0.0), 0.0)
    P_G = np.where(m, np.maximum(p_stc, 0.0) * (np.clip(G, 0.0, None) / g_ref_stc), 0.0)

    # DC do modelo (já inclui G+T). Evita negativos/NaN
    P_dc = np.where(m, np.where(np.isfinite(p_dc), np.clip(p_dc, 0.0, None), 0.0), 0.0)

    # k_sys "efetivo": como seu pac_expected_w já inclui k_sys e eta_inv,
    # estimamos P_after_sys antes de eta: P_sys = P_ac / max(eta,eps)
    eps = 1e-9
    P_sys = np.where(m, np.where(np.isfinite(p_ac), np.clip(p_ac, 0.0, None) / np.maximum(eta, eps), 0.0), 0.0)
    P_ac = np.where(m, np.where(np.isfinite(p_ac), np.clip(p_ac, 0.0, None), 0.0), 0.0)

    # energias (kWh)
    E_stc = float(np.sum(P_stc) * dt_h / 1000.0)
    E_G = float(np.sum(P_G) * dt_h / 1000.0)
    E_dc = float(np.sum(P_dc) * dt_h / 1000.0)
    E_sys = float(np.sum(P_sys) * dt_h / 1000.0)
    E_ac = float(np.sum(P_ac) * dt_h / 1000.0)

    # perdas (kWh) – truncadas para não ficar negativas por arredondamento
    L_irr = max(E_stc - E_G, 0.0)
    L_temp = max(E_G - E_dc, 0.0)
    L_sys = max(E_dc - E_sys, 0.0)
    L_inv = max(E_sys - E_ac, 0.0)

    nodes = [
        "P_STC (base)",
        "Após irradiância (G/1000)",
        "Após temperatura (1-diodo)",
        "Após perdas do sistema (k_sys)",
        "Após inversor (η)",
        "P_AC_exp",
        "Perda irradiância",
        "Perda temperatura",
        "Perda sistema",
        "Perda inversor",
    ]

    # links (source, target, value)
    links = [
        (0, 1, E_G),
        (1, 2, E_dc),
        (2, 3, E_sys),
        (3, 4, E_ac),
        (4, 5, E_ac),

        (0, 6, L_irr),
        (1, 7, L_temp),
        (2, 8, L_sys),
        (3, 9, L_inv),
    ]

    return {
        "nodes": nodes,
        "links": [{"source": s, "target": t, "value": float(v)} for (s, t, v) in links],
        "values_kwh": {
            "E_stc": E_stc,
            "E_after_G": E_G,
            "E_after_T": E_dc,
            "E_after_k_sys": E_sys,
            "E_ac_exp": E_ac,
            "L_irradiance": L_irr,
            "L_temperature": L_temp,
            "L_system": L_sys,
            "L_inverter": L_inv,
        },
    }


# =========================
# 5) Timeline de persistência + histerese
# =========================
def apply_hysteresis(
    codes: np.ndarray,
    *,
    dt_minutes: float = 15.0,
    min_persist_minutes: float = 60.0,
    normal_code: int = DiagnosticCodes.NORMAL,
    keep_invalid: bool = True,
) -> np.ndarray:
    """
    Remove flickering: só mantém estados de falha que persistem >= min_persist_minutes.
    O resto vira NORMAL.
    """
    c = np.asarray(codes, dtype=int).copy()
    n = c.size
    if n == 0:
        return c

    min_len = int(np.ceil(float(min_persist_minutes) / max(float(dt_minutes), 1e-9)))
    if min_len <= 1:
        return c

    out = c.copy()

    i = 0
    while i < n:
        code = int(c[i])
        j = i + 1
        while j < n and int(c[j]) == code:
            j += 1
        seg_len = j - i

        is_fault = (code != normal_code)
        if keep_invalid and code == DiagnosticCodes.INVALID:
            is_fault = False  # não “normaliza” inválido

        if is_fault and seg_len < min_len:
            out[i:j] = normal_code

        i = j

    return out


def timeline_payload(
    *,
    times: Any,
    out_model: Dict[str, Any],
    dt_minutes: float = 15.0,
    min_persist_minutes: float = 60.0,
) -> Dict[str, Any]:
    codes = _diagnostic_code_series(out_model)
    t_iso = _iso_list(times)

    # payload base + payload com histerese (para toggle na UI sem refazer backend)
    codes_h = apply_hysteresis(codes, dt_minutes=dt_minutes, min_persist_minutes=min_persist_minutes)

    return {
        "times": t_iso,
        "code": [int(v) for v in codes.tolist()],
        "code_hyst": [int(v) for v in codes_h.tolist()],
        "dt_minutes": float(dt_minutes),
        "min_persist_minutes": float(min_persist_minutes),
        "code_name_pt": {str(k): v for k, v in CODE_TO_NAME_PT.items()},
        "normal_code": int(DiagnosticCodes.NORMAL),
        "invalid_code": int(DiagnosticCodes.INVALID),
    }


# =========================
# Master builder
# =========================
def build_dashboard_payload(
    *,
    times: Any,
    out_model: Dict[str, Any],
    dt_minutes: float = 15.0,
    min_persist_minutes: float = 60.0,
) -> Dict[str, Any]:
    gcv = np.asarray(out_model.get("g_cv_60m", []), dtype=float)

    return {
        "gauge": meteorological_reliability_from_gcv(gcv),
        "scatter": scatter_payload(times=times, out_model=out_model),
        "sankey": sankey_energy_payload(out_model=out_model, dt_minutes=dt_minutes),
        "timeline": timeline_payload(times=times, out_model=out_model, dt_minutes=dt_minutes, min_persist_minutes=min_persist_minutes),
    }
