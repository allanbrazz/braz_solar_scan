# ================================
# core/services/rules.py
# (Rule-based diagnosis + label hygiene for RF training)
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# ------------------------------------------------------------
# Try to reuse your existing codebook (preferred)
# ------------------------------------------------------------
try:
    from .fuzzy_diagnostic import DiagnosticCodes, CODE_TO_LABEL  # type: ignore
except Exception:  # pragma: no cover
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

    CODE_TO_LABEL: Dict[int, str] = {
        DiagnosticCodes.INVALID: "invalid",
        DiagnosticCodes.NORMAL: "normal",
        DiagnosticCodes.METEO_ERROR: "meteo_error",
        DiagnosticCodes.SOILING: "soiling",
        DiagnosticCodes.DEGRADATION: "degradation",
        DiagnosticCodes.SHORT_BYPASS: "short_bypass",
        DiagnosticCodes.STRING_DISCONNECTED: "string_disconnected",
        DiagnosticCodes.PARTIAL_SHADING: "partial_shading",
    }


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass(frozen=True)
class RuleConfig:
    # Gate diurno / validade
    g_day_min: float = 50.0          # abaixo disso: noite / transiente
    g_gate_train: float = 700.0      # gate típico p/ treinar/inferir falhas
    g_min_valid: float = 0.0         # reserva (não usado diretamente; útil se quiser endurecer valid)

    # Limiares de mismatch (mismatch_rel = (P_real - P_model)/max(P_model, eps))
    thr_ok: float = 0.05             # >= -thr_ok => "normal"
    thr_fault: float = 0.08          # <= -thr_fault => "falha" (não usado explicitamente; disponível)
    thr_meteo_pos: float = 0.25      # mismatch_rel muito positivo => provável erro meteo/modelo

    # Ratios DC (precisa v_ratio / i_ratio plausíveis)
    thr_ratio_band: float = 0.05     # faixa "perto de 1"
    thr_drop_i: float = 0.90         # soiling: i_ratio < thr_drop_i com v~1 e céu estável
    thr_drop_v: float = 0.90         # short/bypass: v_ratio < thr_drop_v com i~1

    # Estabilidade do céu
    cv_max_stable: float = 0.20      # coerente com power_model (se você recalcular sky_stable aqui)

    # Saneamento temporal (reduz ruído nos rótulos)
    fill_gaps_samples: int = 1            # preencher buracos curtos dentro do mesmo evento
    min_fault_duration_samples: int = 2   # eventos mais curtos viram "normal"

    # Para “string_disconnected” (se strings_count >= 2)
    string_drop_tol: float = 0.06         # tolerância em torno do alvo (Ns-1)/Ns

    # EPS para divisões
    eps_w: float = 50.0


# ------------------------------------------------------------
# Helpers (alinhamento estrito)
# ------------------------------------------------------------
def _to_np_1d(x: Any, *, dtype=float) -> np.ndarray:
    if x is None:
        return np.array([], dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False).reshape(-1)
    if hasattr(x, "to_numpy"):
        try:
            return np.asarray(x.to_numpy(), dtype=dtype).reshape(-1)
        except Exception:
            pass
    return np.asarray(x, dtype=dtype).reshape(-1)


def _as_float(x: Any, T: int, *, fill: float = np.nan) -> np.ndarray:
    a = _to_np_1d(x, dtype=float)
    if a.size == 0:
        return np.full(T, float(fill), dtype=float)
    if a.size != T:
        raise ValueError(f"_as_float: tamanho {a.size} != T={T}. Alinhamento temporal inconsistente.")
    return a


def _as_bool(x: Any, T: int, *, fill: bool = False) -> np.ndarray:
    a = _to_np_1d(x, dtype=bool)
    if a.size == 0:
        return np.full(T, bool(fill), dtype=bool)
    if a.size != T:
        raise ValueError(f"_as_bool: tamanho {a.size} != T={T}. Alinhamento temporal inconsistente.")
    return a


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    d = np.maximum(np.where(np.isfinite(den), den, np.nan), float(eps))
    out = num / d
    out[~np.isfinite(out)] = np.nan
    return out


def _fill_small_gaps(lbl: np.ndarray, *, gap: int) -> np.ndarray:
    """
    Preenche buracos curtos (gap amostras) dentro do mesmo label.
    Ex.: A A A B A A (gap=1) => A A A A A A
    """
    if gap <= 0:
        return lbl
    out = np.asarray(lbl, dtype=object).reshape(-1).copy()
    n = out.size
    if n == 0:
        return out

    i = 1
    while i < n - 1:
        if out[i] != out[i - 1] and out[i - 1] == out[i + 1]:
            # mede o tamanho do buraco
            j = i
            while j < n and out[j] != out[i - 1]:
                j += 1
            if (j - i) <= gap:
                out[i:j] = out[i - 1]
                i = j
                continue
        i += 1
    return out


def _enforce_min_duration(lbl: np.ndarray, *, min_len: int, normal_label: str = "normal") -> np.ndarray:
    """
    Eventos com duração < min_len viram normal (reduz labels espúrios).
    """
    if min_len <= 1:
        return lbl
    out = np.asarray(lbl, dtype=object).reshape(-1).copy()
    n = out.size
    if n == 0:
        return out

    start = 0
    while start < n:
        cur = out[start]
        end = start + 1
        while end < n and out[end] == cur:
            end += 1
        run = end - start
        if cur not in ("invalid", normal_label) and run < min_len:
            out[start:end] = normal_label
        start = end
    return out


# ------------------------------------------------------------
# Core RCA (recomputes if not present)
# ------------------------------------------------------------
def _rca_from_signatures(
    *,
    valid: np.ndarray,
    mismatch_rel: np.ndarray,
    v_ratio: np.ndarray,
    i_ratio: np.ndarray,
    sky_stable_mask: np.ndarray,
    strings_count: Optional[int],
    cfg: RuleConfig,
) -> np.ndarray:
    n = int(valid.size)
    out = np.full(n, "invalid", dtype=object)

    m1 = np.isfinite(mismatch_rel)
    v1 = np.isfinite(v_ratio)
    i1 = np.isfinite(i_ratio)

    ok = valid & m1
    out[ok] = "unknown"

    # normal (sem perda relevante)
    out[ok & (mismatch_rel >= -float(cfg.thr_ok))] = "normal"

    both = ok & v1 & i1

    v_near1 = both & (np.abs(v_ratio - 1.0) <= float(cfg.thr_ratio_band))
    i_near1 = both & (np.abs(i_ratio - 1.0) <= float(cfg.thr_ratio_band))

    # string desconectada (corrente cai para ~ (Ns-1)/Ns se strings em paralelo)
    if strings_count is not None and int(strings_count) >= 2:
        Ns = int(strings_count)
        target = (Ns - 1) / float(Ns)
        string_drop = (
            both
            & v_near1
            & (np.abs(i_ratio - target) <= float(cfg.string_drop_tol))
            & (mismatch_rel < -float(cfg.thr_ok))
        )
        out[string_drop] = "string_disconnected"

    # soiling: V ~ nominal, I cai, céu estável
    soiling = (
        both
        & v_near1
        & (i_ratio < float(cfg.thr_drop_i))
        & sky_stable_mask
        & (mismatch_rel < -float(cfg.thr_ok))
    )
    out[soiling] = "soiling"

    # short/bypass: I ~ nominal, V cai
    short_bypass = (
        both
        & i_near1
        & (v_ratio < float(cfg.thr_drop_v))
        & (mismatch_rel < -float(cfg.thr_ok))
    )
    out[short_bypass] = "short_or_bypass"

    # shading: V e I caem com céu instável
    shading = (
        both
        & (v_ratio < (1.0 - float(cfg.thr_ratio_band)))
        & (i_ratio < (1.0 - float(cfg.thr_ratio_band)))
        & (~sky_stable_mask)
        & (mismatch_rel < -float(cfg.thr_ok))
    )
    out[shading] = "partial_shading"

    # degradação-like: V e I caem com céu estável
    degr = (
        both
        & (v_ratio < (1.0 - float(cfg.thr_ratio_band)))
        & (i_ratio < (1.0 - float(cfg.thr_ratio_band)))
        & sky_stable_mask
        & (mismatch_rel < -float(cfg.thr_ok))
    )
    out[degr] = "degradation_like"

    return out


def _map_rca_to_code(rca: np.ndarray) -> np.ndarray:
    """
    Converte rca_label em DiagnosticCodes (exceto meteo_error, tratado à parte).
    """
    r = np.asarray(rca, dtype=object).reshape(-1)
    n = r.size
    code = np.full(n, DiagnosticCodes.INVALID, dtype=int)

    code[r == "normal"] = DiagnosticCodes.NORMAL
    code[r == "soiling"] = DiagnosticCodes.SOILING
    code[r == "degradation_like"] = DiagnosticCodes.DEGRADATION
    code[r == "short_or_bypass"] = DiagnosticCodes.SHORT_BYPASS
    code[r == "string_disconnected"] = DiagnosticCodes.STRING_DISCONNECTED
    code[r == "partial_shading"] = DiagnosticCodes.PARTIAL_SHADING

    # unknown permanece INVALID (normalmente excluído do treino)
    return code


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def run_rules(
    out_model: Dict[str, Any],
    *,
    cfg: Optional[RuleConfig] = None,
    prefer_model_rca: bool = True,
) -> Dict[str, Any]:
    """
    Diagnóstico determinístico (rótulos “professor” p/ RF + baseline de produção).

    Saída:
      - code: (T,) int
      - label: (T,) str  (CODE_TO_LABEL)
      - rca_label: (T,) str (explicativo)
      - anomaly: (T,) float
      - train_mask: (T,) bool
      - meta: dict
    """
    cfg = cfg or RuleConfig()

    # infer T (contrato: expected_and_mismatch sempre tem tcell_c)
    tc = _to_np_1d(out_model.get("tcell_c"), dtype=float)
    T = int(tc.size)
    if T == 0:
        g_try = _to_np_1d(out_model.get("g_poa_used", out_model.get("g_poa")), dtype=float)
        T = int(g_try.size)
        if T == 0:
            raise ValueError("run_rules: out_model sem séries (T=0).")

        tc = _as_float(out_model.get("tcell_c"), T, fill=np.nan)

    # séries alinhadas
    valid = _as_bool(out_model.get("valid"), T, fill=False)
    g = _as_float(out_model.get("g_poa_used", out_model.get("g_poa")), T, fill=np.nan)
    sky_stable = _as_bool(out_model.get("sky_stable_mask"), T, fill=False)

    mismatch_rel = _as_float(out_model.get("mismatch_rel"), T, fill=np.nan)
    mismatch_abs = _as_float(out_model.get("mismatch_abs_w"), T, fill=np.nan)

    pac_real = _as_float(out_model.get("pac_real_w"), T, fill=np.nan)
    pac_exp = _as_float(out_model.get("pac_expected_w"), T, fill=np.nan)

    v_ratio = _as_float(out_model.get("v_ratio"), T, fill=np.nan)
    i_ratio = _as_float(out_model.get("i_ratio"), T, fill=np.nan)

    csi = _as_float(out_model.get("csi"), T, fill=np.nan)

    meta_in = out_model.get("meta", {}) if isinstance(out_model.get("meta", {}), dict) else {}
    strings_count = meta_in.get("strings_count", None)
    try:
        strings_count = None if strings_count is None else int(strings_count)
    except Exception:
        strings_count = None

    # masks
    day = np.isfinite(g) & (g >= float(cfg.g_day_min))
    gate = np.isfinite(g) & (g >= float(cfg.g_gate_train))

    # RCA label (preferir modelo, mas validar tamanho)
    rca_in = out_model.get("rca_label", None)
    rca: np.ndarray
    if prefer_model_rca and rca_in is not None:
        r0 = np.asarray(rca_in, dtype=object).reshape(-1)
        if r0.size == T:
            rca = np.array([str(s).strip().lower() for s in r0], dtype=object)
        else:
            rca = _rca_from_signatures(
                valid=valid,
                mismatch_rel=mismatch_rel,
                v_ratio=v_ratio,
                i_ratio=i_ratio,
                sky_stable_mask=sky_stable,
                strings_count=strings_count,
                cfg=cfg,
            )
    else:
        rca = _rca_from_signatures(
            valid=valid,
            mismatch_rel=mismatch_rel,
            v_ratio=v_ratio,
            i_ratio=i_ratio,
            sky_stable_mask=sky_stable,
            strings_count=strings_count,
            cfg=cfg,
        )

    # base code from rca
    code = _map_rca_to_code(rca)

    # -------- meteo_error override (mismatch muito positivo) --------
    pac_ratio = _safe_div(pac_real, pac_exp, eps=float(cfg.eps_w))
    meteo_pos = (
        valid
        & day
        & np.isfinite(mismatch_rel)
        & (mismatch_rel > float(cfg.thr_meteo_pos))
    )

    # reforço: céu estável OU csi alto + pac_ratio plausível
    meteo_pos = meteo_pos & (sky_stable | (np.isfinite(csi) & (csi > 1.2))) & (pac_ratio < 1.30)

    code[meteo_pos] = DiagnosticCodes.METEO_ERROR
    rca[meteo_pos] = "meteo_error"

    # -------- invalid override (noite / sem validade) --------
    invalid = (~valid) | (~day) | (~np.isfinite(g)) | (~np.isfinite(tc))
    code[invalid] = DiagnosticCodes.INVALID
    rca[invalid] = "invalid"

    # -------- label hygiene (gap fill + min duration) --------
    # (suaviza apenas no RCA; depois remapeia p/ code)
    rca2 = _fill_small_gaps(rca, gap=int(cfg.fill_gaps_samples))
    rca2 = _enforce_min_duration(rca2, min_len=int(cfg.min_fault_duration_samples), normal_label="normal")

    # re-aplica prioridades máximas
    rca2[meteo_pos] = "meteo_error"
    rca2[invalid] = "invalid"

    code2 = _map_rca_to_code(rca2)
    code2[rca2 == "meteo_error"] = DiagnosticCodes.METEO_ERROR
    code2[rca2 == "invalid"] = DiagnosticCodes.INVALID

    # -------- anomaly score (ranking/eventos) --------
    loss = np.where(np.isfinite(mismatch_rel), np.maximum(-mismatch_rel, 0.0), 0.0)
    absw = np.where(np.isfinite(mismatch_abs), np.abs(mismatch_abs), 0.0)
    absw_norm = np.where(np.isfinite(pac_exp), absw / np.maximum(pac_exp, float(cfg.eps_w)), 0.0)

    anomaly = 0.7 * loss + 0.3 * absw_norm
    anomaly = np.where(invalid, 0.0, anomaly)

    # -------- train mask (rótulos “bons” p/ RF) --------
    train_mask = (
        valid
        & gate
        & np.isfinite(mismatch_rel)
        & (rca2 != "invalid")
        & (rca2 != "unknown")
        & (rca2 != "meteo_error")
    )

    # label final (string do code)
    label = np.array([CODE_TO_LABEL.get(int(c), "invalid") for c in code2], dtype=object)

    meta = {
        "cfg": cfg.__dict__,
        "strings_count": strings_count,
        "used_model_rca": bool(prefer_model_rca and rca_in is not None and np.asarray(rca_in).reshape(-1).size == T),
        "notes": {
            "meteo_error": "ativado por mismatch_rel positivo alto (P_real > P_model).",
            "invalid": "noite / NaN / valid=False.",
            "train_mask": "gate>=g_gate_train e sem invalid/unknown/meteo_error.",
        },
    }

    return {
        "code": code2.astype(int),
        "label": label.astype(str),
        "rca_label": np.array([str(s).strip().lower() for s in rca2], dtype=object),
        "anomaly": anomaly.astype(float),
        "train_mask": train_mask.astype(bool),
        "meta": meta,
    }


# ------------------------------------------------------------
# Convenience: label <-> code
# ------------------------------------------------------------
LABEL_TO_CODE: Dict[str, int] = {str(v).strip().lower(): int(k) for k, v in CODE_TO_LABEL.items()}


def labels_to_codes(labels: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels, dtype=object).reshape(-1)
    out = np.full(labs.size, DiagnosticCodes.INVALID, dtype=int)
    for i, s in enumerate(labs):
        out[i] = int(LABEL_TO_CODE.get(str(s).strip().lower(), DiagnosticCodes.INVALID))
    return out


def codes_to_labels(codes: np.ndarray) -> np.ndarray:
    cc = np.asarray(codes, dtype=int).reshape(-1)
    out = np.empty(cc.size, dtype=object)
    for i, c in enumerate(cc):
        out[i] = CODE_TO_LABEL.get(int(c), "invalid")
    return out.astype(str)
