# =================================
# core/services/labeling.py
# Weak-supervision / pseudo-labels para treinar RF
# - Usa rca_label do power_model (heurística elétrica) + checagens meteo
# - Permite sobrepor rótulos manuais por intervalos de tempo
# =================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Dicionário de códigos canônico para treinamento/inferência RF.
@dataclass(frozen=True)
class _DiagnosticCodes:
    INVALID: int = 0
    NORMAL: int = 1
    METEO_ERROR: int = 2
    SOILING: int = 3
    DEGRADATION: int = 4
    SHORT_BYPASS: int = 5
    STRING_DISCONNECTED: int = 6
    PARTIAL_SHADING: int = 7


_CODE_TO_LABEL = {
    _DiagnosticCodes.INVALID: "invalid",
    _DiagnosticCodes.NORMAL: "normal",
    _DiagnosticCodes.METEO_ERROR: "meteo_error",
    _DiagnosticCodes.SOILING: "soiling",
    _DiagnosticCodes.DEGRADATION: "degradation",
    _DiagnosticCodes.SHORT_BYPASS: "short_bypass",
    _DiagnosticCodes.STRING_DISCONNECTED: "string_disconnected",
    _DiagnosticCodes.PARTIAL_SHADING: "partial_shading",
}

DiagnosticCodes = _DiagnosticCodes
CODE_TO_LABEL: Dict[int, str] = dict(_CODE_TO_LABEL)

# normaliza labels
LABEL_TO_CODE: Dict[str, int] = {str(v).strip().lower(): int(k) for k, v in CODE_TO_LABEL.items()}

# ordem canônica útil para stats
ALL_LABELS = [
    "invalid",
    "normal",
    "meteo_error",
    "soiling",
    "degradation",
    "short_bypass",
    "string_disconnected",
    "partial_shading",
]


# ============================================================
# Helpers: arrays com alinhamento ESTRITO
# ============================================================
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


def _as_bool_1d(x: Any, n: int, *, fill: bool = False) -> np.ndarray:
    a = _to_np_1d(x, dtype=bool)
    if a.size == 0:
        return np.full(n, bool(fill), dtype=bool)
    if a.size != n:
        raise ValueError(f"_as_bool_1d: tamanho {a.size} != n={n}. Alinhamento temporal inconsistente.")
    return a


def _as_float_1d(x: Any, n: int, *, fill: float = np.nan) -> np.ndarray:
    a = _to_np_1d(x, dtype=float)
    if a.size == 0:
        return np.full(n, float(fill), dtype=float)
    if a.size != n:
        raise ValueError(f"_as_float_1d: tamanho {a.size} != n={n}. Alinhamento temporal inconsistente.")
    return a


def _as_obj_1d(x: Any, n: int, *, fill: str = "n/a") -> np.ndarray:
    if x is None:
        return np.full(n, fill, dtype=object)
    a = np.asarray(x, dtype=object).reshape(-1)
    if a.size != n:
        raise ValueError(f"_as_obj_1d: tamanho {a.size} != n={n}. Alinhamento temporal inconsistente.")
    return a


def _get_first(out_model: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in out_model and out_model[k] is not None:
            return out_model[k]
    return None


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class LabelingConfig:
    """
    g_min_train:
      gate para usar amostras no treino (ex.: 200 ou 700 W/m²)
    allow_unknown:
      se True, permite a classe "unknown" (só se existir em CODE_TO_LABEL).
      caso contrário, "unknown" é tratado como invalid.
    meteo_error:
      regras para marcar METEO_ERROR com base em CSI/PR fora do físico.
    """
    g_min_train: float = 200.0
    allow_unknown: bool = False

    # --- METEO_ERROR heurística ---
    g_min_csi_check: float = 150.0
    csi_min: float = 0.03
    csi_max: float = 1.35
    pr_max: float = 1.20

    # --- fallback para “normal” quando rca_label não existir ---
    thr_ok_mismatch_rel: float = 0.05  # mismatch_rel >= -thr_ok => normal


# ============================================================
# Mapeamento do RCA -> labels canônicos
# ============================================================
_RCA_TO_LABEL: Dict[str, str] = {
    "normal": "normal",
    "soiling": "soiling",
    "degradation_like": "degradation",
    "degradation": "degradation",
    "short_or_bypass": "short_bypass",
    "short_bypass": "short_bypass",
    "string_disconnected": "string_disconnected",
    "partial_shading": "partial_shading",
    # valores que viram "não rotulado"
    "unknown": "invalid",
    "n/a": "invalid",
    "invalid": "invalid",
}


def _label_from_rca(rca: str, *, allow_unknown: bool) -> str:
    r = (rca or "").strip().lower()
    if r == "unknown" and allow_unknown and ("unknown" in LABEL_TO_CODE):
        return "unknown"
    return _RCA_TO_LABEL.get(r, "invalid")


# ============================================================
# METEO_ERROR: detecção rápida (CSI/PR fora do plausível)
# ============================================================
def _meteo_error_mask(
    *,
    valid_ok: np.ndarray,
    g_poa: np.ndarray,
    csi: np.ndarray,
    pr_real: np.ndarray,
    cfg: LabelingConfig,
) -> np.ndarray:
    g = np.asarray(g_poa, dtype=float)
    csi = np.asarray(csi, dtype=float)
    pr = np.asarray(pr_real, dtype=float)

    sun = valid_ok & np.isfinite(g) & (g >= float(cfg.g_min_csi_check))
    bad_csi = np.isfinite(csi) & ((csi < float(cfg.csi_min)) | (csi > float(cfg.csi_max)))
    bad_pr = np.isfinite(pr) & (pr > float(cfg.pr_max))
    return sun & (bad_csi | bad_pr)


# ============================================================
# API principal: gerar y (code/label) + máscara de treino
# ============================================================
def make_labels_from_out_model(
    out_model: Dict[str, Any],
    *,
    cfg: Optional[LabelingConfig] = None,
) -> Dict[str, Any]:
    """
    Retorna:
      - y_code: (T,) int
      - y_label: (T,) str
      - train_mask: (T,) bool (True => usar no treino)
      - stats: contagens por classe
    """
    cfg = cfg or LabelingConfig()

    # infer T via tcell (contrato do expected_and_mismatch)
    tc = _to_np_1d(_get_first(out_model, "tcell_c"), dtype=float)
    T = int(tc.size)
    if T == 0:
        raise ValueError("make_labels_from_out_model: out_model sem 'tcell_c' (T=0).")

    g = _as_float_1d(_get_first(out_model, "g_poa_used", "g_poa"), T, fill=np.nan)
    valid = _as_bool_1d(out_model.get("valid"), T, fill=False)

    mismatch_rel = _as_float_1d(out_model.get("mismatch_rel"), T, fill=np.nan)
    csi = _as_float_1d(out_model.get("csi"), T, fill=np.nan)
    pr_real = _as_float_1d(out_model.get("pr_real_inst"), T, fill=np.nan)
    rca = _as_obj_1d(out_model.get("rca_label"), T, fill="n/a")

    # máscara base de “dado usável”
    base_ok = valid & np.isfinite(g) & np.isfinite(tc)

    # gate de treino (ex.: GPOA>=700)
    gate = base_ok & (g >= float(cfg.g_min_train))

    # labels (object p/ atribuição rápida)
    y_label = np.full(T, "invalid", dtype=object)

    # 1) METEO_ERROR (override)
    m_meteo = _meteo_error_mask(valid_ok=base_ok, g_poa=g, csi=csi, pr_real=pr_real, cfg=cfg)
    y_label[m_meteo] = "meteo_error"

    # 2) RCA (quando não é meteo_error)
    # vetoriza em vez de loop python (mais rápido e menos propenso a bug)
    m_rca = gate & (~m_meteo)
    if m_rca.any():
        rca_norm = np.array([str(s).strip().lower() for s in rca], dtype=object)
        mapped = np.array([_label_from_rca(str(s), allow_unknown=bool(cfg.allow_unknown)) for s in rca_norm], dtype=object)
        y_label[m_rca] = mapped[m_rca]

    # 3) fallback “normal” quando RCA não ajudou, mas mismatch está ok
    fallback_normal = (
        gate
        & (~m_meteo)
        & (y_label == "invalid")
        & np.isfinite(mismatch_rel)
        & (mismatch_rel >= -float(cfg.thr_ok_mismatch_rel))
    )
    y_label[fallback_normal] = "normal"

    # codifica (garantindo lower/strip)
    y_label_str = np.array([str(s).strip().lower() for s in y_label], dtype=object)
    y_code = np.array([LABEL_TO_CODE.get(str(s), DiagnosticCodes.INVALID) for s in y_label_str], dtype=int)

    # máscara final de treino: gate AND label != invalid (e != unknown se não permitido)
    train_mask = gate & (y_label_str != "invalid")
    if not cfg.allow_unknown:
        train_mask = train_mask & (y_label_str != "unknown")

    stats = _label_stats(y_label_str)

    return {
        "y_code": y_code,
        "y_label": y_label_str.astype(str),
        "train_mask": train_mask.astype(bool),
        "stats": stats,
        "meta": {
            "g_min_train": float(cfg.g_min_train),
            "allow_unknown": bool(cfg.allow_unknown),
        },
    }


def _label_stats(y_label: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y_label, dtype=object).reshape(-1)
    out: Dict[str, int] = {}
    for lab in ALL_LABELS:
        out[lab] = int(np.sum(y == lab))
    out["total"] = int(y.size)
    return out


# ============================================================
# Opcional: sobrepor labels manuais por intervalos
# ============================================================
# interval: (t_start, t_end, label_str)
ManualInterval = Tuple[Any, Any, str]


def _to_datetime64ns(times: Any) -> np.ndarray:
    """
    Wrapper para converter times -> datetime64[ns].
    Preferimos reutilizar a função do power_model se existir.
    """
    try:
        from .power_model import _to_datetime64ns as _pm_to_dt  # type: ignore

        return _pm_to_dt(times)
    except Exception:
        t = np.asarray(times)
        if t.size == 0:
            return np.array([], dtype="datetime64[ns]")
        if np.issubdtype(t.dtype, np.datetime64):
            out = t.astype("datetime64[ns]").reshape(-1)
        else:
            flat = t.reshape(-1)
            out = np.empty(flat.size, dtype="datetime64[ns]")
            for i, v in enumerate(flat):
                out[i] = np.datetime64(v).astype("datetime64[ns]")
        if np.isnat(out).any():
            raise ValueError("times_utc contém NaT após conversão")
        return out.reshape(t.shape)


def apply_manual_intervals(
    y_label: np.ndarray,
    *,
    times_utc: Any,
    intervals: Sequence[ManualInterval],
    overwrite: bool = True,
) -> np.ndarray:
    """
    Sobrepõe labels por intervalos [start, end) (start inclusivo, end exclusivo).
    """
    y = np.asarray(y_label, dtype=object).reshape(-1).copy()
    t = _to_datetime64ns(times_utc).reshape(-1)
    if y.size != t.size:
        raise ValueError(f"apply_manual_intervals: y_label e times_utc tamanhos diferentes ({y.size} vs {t.size}).")

    for (t0, t1, lab) in intervals:
        lab0 = (lab or "").strip().lower()
        if lab0 not in LABEL_TO_CODE:
            raise ValueError(f"Label manual inválido '{lab}'. Válidos: {sorted(LABEL_TO_CODE.keys())}")

        a = np.datetime64(t0).astype("datetime64[ns]")
        b = np.datetime64(t1).astype("datetime64[ns]")
        if b < a:
            a, b = b, a

        m = (t >= a) & (t < b)
        if not m.any():
            continue

        if overwrite:
            y[m] = lab0
        else:
            y[m & (y == "invalid")] = lab0

    return np.array([str(s).strip().lower() for s in y], dtype=str)


# ============================================================
# Opcional: sample weights (para RF)
# ============================================================
def compute_sample_weight(
    y_code: np.ndarray,
    *,
    train_mask: Optional[np.ndarray] = None,
    method: str = "balanced",
) -> np.ndarray:
    """
    method:
      - "none": pesos = 1
      - "balanced": inverso da frequência (estilo sklearn)
    """
    y = np.asarray(y_code, dtype=int).reshape(-1)
    n = int(y.size)
    w = np.ones(n, dtype=float)

    m = np.ones(n, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool).reshape(-1)
    if m.size != n:
        raise ValueError(f"compute_sample_weight: train_mask tamanho {m.size} != n={n}.")

    if method.lower().strip() == "none":
        return w

    y_tr = y[m]
    if y_tr.size == 0:
        return w

    classes, counts = np.unique(y_tr, return_counts=True)
    counts = np.maximum(counts.astype(float), 1.0)
    inv = (y_tr.size / (classes.size * counts))  # sklearn-like

    map_w = {int(c): float(v) for c, v in zip(classes, inv)}
    # aplica na mesma ordem das amostras do treino
    w[m] = np.array([map_w.get(int(c), 1.0) for c in y_tr], dtype=float)
    return w
