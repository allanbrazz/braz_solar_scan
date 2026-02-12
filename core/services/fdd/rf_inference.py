# =================================
# core/services/rf_inference.py
# RandomForest inference + postprocess + event extraction
# (aligned with rf_train.py + features.py)
# =================================
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("joblib não está instalado. Use: pip install joblib") from e

try:
    from sklearn.pipeline import Pipeline  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-learn não está instalado. Use: pip install scikit-learn") from e

from .features import RFFeatureSpec, build_rf_features
from .rules import CODE_TO_LABEL, codes_to_labels, run_rules

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True)
class InferenceConfig:
    # Gate de inferência
    use_rules_gate: bool = True           # usa run_rules() para base_code (físico) e bloquear meteo_error no RF
    g_min_infer: float = 50.0             # W/m² (evita inferência em baixa irradiância)
    require_valid_mask: bool = True       # exige FeatureBuildResult.valid_mask (valid & G finito & Tc finito)
    require_finite_mismatch: bool = True  # exige mismatch finito (se existir) para inferir

    # Confiança / fallback
    proba_threshold: float = 0.55         # se max_proba < thr => unknown ou mantém rule-code
    fallback_to_rules_when_low_conf: bool = True

    # Pós-processamento temporal
    smooth_window: int = 3                # moda trailing (>=1)
    min_event_len: int = 3                # mínimo de pontos para evento (ex.: 3*15min=45min)
    merge_gap: int = 1                    # une eventos com gap <= merge_gap (mesma classe)

    # "unknown" interno
    unknown_code: int = -1                # não precisa estar no CODE_TO_LABEL


# Ordem default das features (tem que existir em build_rf_features().series)
DEFAULT_FEATURES: Tuple[str, ...] = (
    "g_poa",
    "tcell_c",
    "mismatch_rel",
    "v_ratio",
    "i_ratio",
    "g_cv_60m",
    "sky_stable",  # <- não é sky_stable_mask
    "csi",
    "v_ac_v",
)


# -----------------------------
# Helpers
# -----------------------------
def _find_code_by_label(code_to_label: Dict[int, str], label: str, default: int) -> int:
    tgt = (label or "").strip().lower()
    for c, s in code_to_label.items():
        if str(s).strip().lower() == tgt:
            try:
                return int(c)
            except Exception:
                continue
    return int(default)


def _infer_dt_minutes(times_utc: Optional[Any], fallback: float = 15.0) -> float:
    if times_utc is None:
        return float(fallback)
    t = np.asarray(times_utc).reshape(-1)
    if t.size < 2:
        return float(fallback)
    try:
        tn = t.astype("datetime64[s]").astype("int64").astype(float)
        d = np.diff(tn)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            return float(fallback)
        return float(np.nanmedian(d) / 60.0)
    except Exception:
        return float(fallback)


def _subset_X_from_feature_result(feat_res: Any, feature_names: Iterable[str]) -> np.ndarray:
    names = list(feature_names)
    T = int(feat_res.X.shape[0])
    cols: List[np.ndarray] = []
    for k in names:
        if k in feat_res.series:
            v = np.asarray(feat_res.series[k], dtype=float).reshape(-1)
        else:
            v = np.full(T, np.nan, dtype=float)
        if v.size != T:
            raise ValueError(f"Feature '{k}' size {v.size} != T={T}")
        cols.append(v)
    X = np.column_stack(cols).astype(np.float32, copy=False)
    # sklearn: SimpleImputer lida com NaN, mas não com inf
    X[~np.isfinite(X)] = np.nan
    return X


def _codes_to_labels_extended(
    codes: np.ndarray,
    *,
    code_to_label: Dict[int, str],
    unknown_code: int,
) -> np.ndarray:
    x = np.asarray(codes, dtype=int).reshape(-1)
    out = np.empty(x.size, dtype=object)
    for i, c in enumerate(x):
        if int(c) == int(unknown_code):
            out[i] = "unknown"
        else:
            out[i] = str(code_to_label.get(int(c), "invalid"))
    return out.astype(str)


def _rolling_mode_int(x: np.ndarray, w: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Suaviza por moda (janela trailing). Mantém x quando mask=False.
    """
    x = np.asarray(x, dtype=int).reshape(-1)
    n = x.size
    w = int(max(1, w))
    out = x.copy()
    if w <= 1 or n == 0:
        return out

    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != n:
            mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        j0 = max(0, i - w + 1)
        win = x[j0 : i + 1]
        win_mask = mask[j0 : i + 1]
        win = win[win_mask]
        if win.size == 0:
            continue
        vals, cnt = np.unique(win, return_counts=True)
        out[i] = int(vals[np.argmax(cnt)])
    return out


def _suppress_short_runs(
    codes: np.ndarray,
    *,
    normal_code: int,
    min_len: int,
    protected_codes: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Remove "picos" curtos: qualquer run != normal com len < min_len vira normal,
    exceto para códigos protegidos (ex.: invalid, meteo_error, unknown).
    """
    x = np.asarray(codes, dtype=int).reshape(-1)
    n = x.size
    out = x.copy()
    if n == 0 or int(min_len) <= 1:
        return out

    prot = set(int(c) for c in (protected_codes or []))

    i = 0
    while i < n:
        c = int(out[i])
        j = i + 1
        while j < n and int(out[j]) == c:
            j += 1

        run_len = j - i
        if c != int(normal_code) and c not in prot and run_len < int(min_len):
            out[i:j] = int(normal_code)

        i = j
    return out


def _extract_events(
    codes: np.ndarray,
    *,
    times_utc: Optional[Any],
    code_to_label: Dict[int, str],
    normal_code: int,
    min_len: int,
    merge_gap: int,
    dt_minutes: float,
    unknown_code: int,
    pac_expected_w: Optional[np.ndarray] = None,
    pac_real_w: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Extrai eventos como runs de classes != normal com len >= min_len.
    Une eventos adjacentes com gap <= merge_gap se a classe for a mesma.
    Calcula perdas (kWh) se pac_expected_w e pac_real_w existirem.
    """
    x = np.asarray(codes, dtype=int).reshape(-1)
    n = x.size
    if n == 0:
        return []

    t = None
    if times_utc is not None:
        try:
            t = np.asarray(times_utc).reshape(-1)
            if t.size != n:
                t = None
        except Exception:
            t = None

    def _label(c: int) -> str:
        if int(c) == int(unknown_code):
            return "unknown"
        return str(code_to_label.get(int(c), "invalid"))

    runs: List[Tuple[int, int, int]] = []  # (start, end_exclusive, code)
    i = 0
    while i < n:
        c = int(x[i])
        j = i + 1
        while j < n and int(x[j]) == c:
            j += 1
        if c != int(normal_code) and (j - i) >= int(min_len):
            runs.append((i, j, c))
        i = j

    if not runs:
        return []

    # merge gaps
    merged: List[Tuple[int, int, int]] = []
    for (s, e, c) in runs:
        if not merged:
            merged.append((s, e, c))
            continue
        ps, pe, pc = merged[-1]
        gap = s - pe
        if int(c) == int(pc) and gap <= int(merge_gap):
            merged[-1] = (ps, e, pc)
        else:
            merged.append((s, e, c))

    # loss integration
    loss_wh = None
    if pac_expected_w is not None and pac_real_w is not None:
        pe = np.asarray(pac_expected_w, dtype=float).reshape(-1)
        pr = np.asarray(pac_real_w, dtype=float).reshape(-1)
        if pe.size == n and pr.size == n:
            loss_w = np.clip(pe - pr, 0.0, None)
            loss_wh = loss_w * (float(dt_minutes) / 60.0)  # Wh por passo

    events: List[Dict[str, Any]] = []
    for (s, e, c) in merged:
        dur_steps = int(e - s)
        dur_min = float(dur_steps) * float(dt_minutes)
        item: Dict[str, Any] = {
            "code": int(c),
            "label": _label(int(c)),
            "start_idx": int(s),
            "end_idx": int(e - 1),
            "n_steps": int(dur_steps),
            "duration_minutes": dur_min,
        }
        if t is not None:
            item["start_time_utc"] = t[s]
            item["end_time_utc"] = t[e - 1]
        if loss_wh is not None:
            item["energy_loss_kwh"] = float(np.nansum(loss_wh[s:e]) / 1000.0)
        events.append(item)

    return events


# -----------------------------
# Bundle IO
# -----------------------------
def load_rf_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rf_inference.load_rf_bundle: arquivo não encontrado: {p}")
    bundle = joblib.load(p)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("rf_inference.load_rf_bundle: bundle inválido (esperado dict com 'model').")
    return bundle


# -----------------------------
# Inference core
# -----------------------------
def predict_rf(bundle: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    model: Pipeline = bundle["model"]
    X = np.asarray(X, dtype=float)
    X2 = X.copy()
    X2[~np.isfinite(X2)] = np.nan

    code_pred = model.predict(X2).astype(int)

    proba = None
    classes_ = None
    proba_max = None
    try:
        proba = model.predict_proba(X2)
        classes_ = getattr(model.named_steps.get("rf", None), "classes_", None)
        if classes_ is None:
            classes_ = np.arange(proba.shape[1], dtype=int)
        classes_ = np.asarray(classes_, dtype=int)
        proba_max = np.max(proba, axis=1).astype(float)
    except Exception:
        proba = None
        classes_ = None
        proba_max = None

    return {"code_pred": code_pred, "proba": proba, "classes_": classes_, "proba_max": proba_max}


def run_rf_inference(
    *,
    bundle: Dict[str, Any],
    out_model: Dict[str, Any],
    times_utc: Optional[Any] = None,
    feature_names: Optional[Iterable[str]] = None,
    feature_spec: Optional[RFFeatureSpec] = None,
    cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """
    Pipeline completo:
      1) features -> X
      2) rules gate (opcional) -> base_code
      3) RF predict em infer_mask
      4) threshold de confiança -> unknown / fallback
      5) pós-processamento temporal (mode + remove picos)
      6) extração de eventos
    """
    cfg = cfg or InferenceConfig()

    # label map (preferir o do bundle)
    code_to_label = dict(bundle.get("code_to_label", CODE_TO_LABEL))
    normal_code = _find_code_by_label(code_to_label, "normal", default=1)
    invalid_code = _find_code_by_label(code_to_label, "invalid", default=0)
    meteo_code = _find_code_by_label(code_to_label, "meteo_error", default=2)

    # features / X (timeline completo)
    feature_spec = feature_spec or RFFeatureSpec()
    feat_names = tuple(feature_names or bundle.get("feature_names", DEFAULT_FEATURES))

    feat_res = build_rf_features(out_model, times_utc=times_utc, spec=feature_spec)
    T = int(feat_res.X.shape[0])
    X_all = _subset_X_from_feature_result(feat_res, feat_names)

    # base_code e infer_mask
    base_code = np.full(T, int(invalid_code), dtype=int)
    infer_mask = np.ones(T, dtype=bool)

    # gate por features (sempre disponível)
    g = np.asarray(feat_res.series.get("g_poa", np.full(T, np.nan)), dtype=float).reshape(-1)
    valid_mask = np.asarray(getattr(feat_res, "valid_mask", np.ones(T, dtype=bool)), dtype=bool).reshape(-1)

    if cfg.require_valid_mask:
        infer_mask &= valid_mask

    infer_mask &= np.isfinite(g) & (g >= float(cfg.g_min_infer))

    if cfg.require_finite_mismatch:
        mm = np.asarray(feat_res.series.get("mismatch_rel", np.full(T, np.nan)), dtype=float).reshape(-1)
        infer_mask &= np.isfinite(mm)

    # rules gate (base_code e bloqueio meteo_error no RF)
    if cfg.use_rules_gate:
        rules_out = run_rules(out_model)
        rc = np.asarray(rules_out.get("code", base_code), dtype=int).reshape(-1)
        if rc.size == T:
            base_code = rc
        # por padrão, não roda RF em meteo_error nem invalid
        infer_mask &= (base_code != int(meteo_code)) & (base_code != int(invalid_code))

    # RF predict em infer_mask
    code_rf = np.full(T, int(cfg.unknown_code), dtype=int)
    proba_max = np.full(T, np.nan, dtype=float)
    proba_mask = None
    classes_ = None

    if infer_mask.any():
        pred = predict_rf(bundle, X_all[infer_mask])
        code_rf[infer_mask] = np.asarray(pred["code_pred"], dtype=int)
        if pred["proba_max"] is not None:
            proba_max[infer_mask] = np.asarray(pred["proba_max"], dtype=float)
        proba_mask = pred["proba"]
        classes_ = pred["classes_"]

    # threshold de confiança + fallback
    code_final = base_code.copy()
    used_rf = np.zeros(T, dtype=bool)

    if infer_mask.any():
        if np.isfinite(proba_max).any():
            ok_conf = infer_mask & np.isfinite(proba_max) & (proba_max >= float(cfg.proba_threshold))
        else:
            ok_conf = infer_mask.copy()

        used_rf[ok_conf] = True
        code_final[ok_conf] = code_rf[ok_conf]

        low_conf = infer_mask & (~ok_conf)
        if low_conf.any() and (not cfg.fallback_to_rules_when_low_conf):
            code_final[low_conf] = int(cfg.unknown_code)

    # pós-processamento temporal (suaviza só onde RF foi usado)
    code_smooth = code_final.copy()
    w = int(max(1, cfg.smooth_window))
    if w > 1 and T > 0:
        code_smooth = _rolling_mode_int(code_smooth, w=w, mask=used_rf)

    # remove picos curtos (não mexe em invalid/meteo/unknown)
    code_post = _suppress_short_runs(
        code_smooth,
        normal_code=int(normal_code),
        min_len=int(max(1, cfg.min_event_len)),
        protected_codes=[int(invalid_code), int(meteo_code), int(cfg.unknown_code)],
    )

    # labels (com unknown)
    label_final = _codes_to_labels_extended(code_final, code_to_label=code_to_label, unknown_code=int(cfg.unknown_code))
    label_post = _codes_to_labels_extended(code_post, code_to_label=code_to_label, unknown_code=int(cfg.unknown_code))

    # eventos
    meta_in = out_model.get("meta", {}) if isinstance(out_model.get("meta", {}), dict) else {}
    dt_min = float(meta_in.get("dt_minutes", _infer_dt_minutes(times_utc, fallback=15.0)) or 15.0)

    pac_expected = out_model.get("pac_expected_w", None)
    pac_real = out_model.get("pac_real_w", None)

    events = _extract_events(
        code_post,
        times_utc=times_utc,
        code_to_label=code_to_label,
        normal_code=int(normal_code),
        min_len=int(max(1, cfg.min_event_len)),
        merge_gap=int(max(0, cfg.merge_gap)),
        dt_minutes=float(dt_min),
        unknown_code=int(cfg.unknown_code),
        pac_expected_w=np.asarray(pac_expected, dtype=float).reshape(-1) if pac_expected is not None else None,
        pac_real_w=np.asarray(pac_real, dtype=float).reshape(-1) if pac_real is not None else None,
    )

    return {
        # por ponto
        "code_base": base_code.astype(int),
        "code_rf": code_rf.astype(int),
        "code_final": code_final.astype(int),
        "code_post": code_post.astype(int),
        "label_final": label_final,
        "label_post": label_post,
        "used_rf": used_rf.astype(bool),
        "infer_mask": infer_mask.astype(bool),
        "proba_max": proba_max.astype(float),

        # probabilidades do RF (apenas para os pontos inferidos)
        "proba_mask": proba_mask,   # shape (n_infer, n_classes) ou None
        "classes_": classes_,       # classes do RF (ordem das colunas) ou None

        # info
        "feature_names": list(feat_names),
        "dt_minutes": float(dt_min),
        "normal_code": int(normal_code),
        "invalid_code": int(invalid_code),
        "meteo_code": int(meteo_code),
        "unknown_code": int(cfg.unknown_code),
        "code_to_label": code_to_label,
        "cfg": asdict(cfg),

        # eventos
        "events": events,
    }


# -----------------------------
# Convenience: load + run
# -----------------------------
def load_and_run(
    *,
    bundle_path: Union[str, Path],
    out_model: Dict[str, Any],
    times_utc: Optional[Any] = None,
    feature_names: Optional[Iterable[str]] = None,
    feature_spec: Optional[RFFeatureSpec] = None,
    cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    bundle = load_rf_bundle(bundle_path)
    return run_rf_inference(
        bundle=bundle,
        out_model=out_model,
        times_utc=times_utc,
        feature_names=feature_names,
        feature_spec=feature_spec,
        cfg=cfg,
    )
