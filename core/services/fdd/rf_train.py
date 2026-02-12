# =================================
# core/services/rf_train.py
# RandomForest training + evaluation + persistence bundle
# (uses features.py + rules.py)
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
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-learn não está instalado. Use: pip install scikit-learn") from e

from .features import RFFeatureSpec, build_rf_features
from .rules import CODE_TO_LABEL, RuleConfig, codes_to_labels, run_rules

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 600
    max_depth: Optional[int] = None
    min_samples_split: int = 6
    min_samples_leaf: int = 3
    max_features: Union[str, float, int, None] = "sqrt"
    bootstrap: bool = True
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class TrainConfig:
    # split
    split_mode: str = "time"  # "time" | "random"
    test_size: float = 0.20
    random_state: int = 42

    # filtros adicionais (além do train_mask do rules)
    require_finite_mismatch: bool = True
    drop_unknown_like: bool = True  # reforço (já vem no train_mask do rules)
    min_samples_per_class: int = 30


# Subset default (tem que existir em build_rf_features().series)
DEFAULT_FEATURES: Tuple[str, ...] = (
    "g_poa",
    "tcell_c",
    "mismatch_rel",
    "v_ratio",
    "i_ratio",
    "g_cv_60m",
    "sky_stable",  # <-- (não é sky_stable_mask)
    "csi",
    "v_ac_v",
)


# -----------------------------
# Utils
# -----------------------------
def _time_split_indices(n: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    ts = float(np.clip(float(test_size), 0.05, 0.50))
    cut = int(round((1.0 - ts) * n))
    cut = int(np.clip(cut, 1, n - 1))
    idx = np.arange(n, dtype=int)
    return idx[:cut], idx[cut:]


def _subset_X_from_feature_result(
    feat_res: Any,
    feature_names: Iterable[str],
) -> np.ndarray:
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
    X[~np.isfinite(X)] = np.nan  # garante: inf -> nan (SimpleImputer lida)
    return X


# -----------------------------
# Dataset builder (labels via rules.py)
# -----------------------------
def build_training_set(
    out_model: Dict[str, Any],
    *,
    times_utc: Optional[Any] = None,
    feature_names: Optional[Iterable[str]] = None,
    feature_spec: Optional[RFFeatureSpec] = None,
    rules_cfg: Optional[RuleConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    keep_times_in_output: bool = True,
) -> Dict[str, Any]:
    """
    Constrói X,y a partir do out_model (expected_and_mismatch),
    com:
      - features: build_rf_features()
      - labels:   run_rules()

    Retorna:
      {
        "X": (n_train,p),
        "y": (n_train,),
        "mask": (T,) bool (máscara aplicada sobre o timeline original),
        "feature_names": [...],
        "times_utc_train": (n_train,) datetime64[ns] | None,
        "rules": {...} (opcional: code/label/rca_label do timeline completo),
        "meta": {...}
      }
    """
    feature_names = tuple(feature_names or DEFAULT_FEATURES)
    feature_spec = feature_spec or RFFeatureSpec()
    rules_cfg = rules_cfg or RuleConfig()
    train_cfg = train_cfg or TrainConfig()

    # 1) Features (timeline completo)
    feat_res = build_rf_features(out_model, times_utc=times_utc, spec=feature_spec)
    T = int(feat_res.X.shape[0])

    X_all = _subset_X_from_feature_result(feat_res, feature_names)  # (T,p)

    # 2) Rules -> labels + train_mask (timeline completo)
    rules_out = run_rules(out_model, cfg=rules_cfg)
    code = np.asarray(rules_out["code"], dtype=int).reshape(-1)
    rca_label = np.asarray(rules_out["rca_label"], dtype=object).reshape(-1)
    label = np.asarray(rules_out["label"], dtype=object).reshape(-1)
    train_mask = np.asarray(rules_out["train_mask"], dtype=bool).reshape(-1)

    if code.size != T:
        raise ValueError(f"build_training_set: code size {code.size} != T {T}")
    if train_mask.size != T:
        raise ValueError(f"build_training_set: train_mask size {train_mask.size} != T {T}")

    # 3) Filtros extras
    if train_cfg.require_finite_mismatch:
        mm = np.asarray(feat_res.series.get("mismatch_rel", np.full(T, np.nan)), dtype=float).reshape(-1)
        train_mask = train_mask & np.isfinite(mm)

    if train_cfg.drop_unknown_like:
        train_mask = train_mask & (rca_label != "unknown") & (rca_label != "invalid") & (rca_label != "meteo_error")

    # 4) Aplica máscara (treino)
    X = X_all[train_mask]
    y = code[train_mask]

    # sanity: classes e contagens
    uniq, cnt = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(uniq, cnt)}
    too_small = {k: v for k, v in class_counts.items() if v < int(train_cfg.min_samples_per_class)}

    # times no dataset (opcional)
    times_train = None
    if keep_times_in_output and times_utc is not None:
        t = np.asarray(times_utc).reshape(-1)
        if t.size == T:
            times_train = t[train_mask]
        else:
            # não falha; apenas não carrega
            times_train = None

    meta = {
        "T_total": int(T),
        "n_train": int(X.shape[0]),
        "feature_names": list(feature_names),
        "class_counts": class_counts,
        "classes_below_min": too_small,
        "rules_cfg": asdict(rules_cfg),
        "train_cfg": asdict(train_cfg),
        "feature_spec": asdict(feature_spec),
    }

    return {
        "X": X.astype(np.float32, copy=False),
        "y": y.astype(int, copy=False),
        "mask": train_mask.astype(bool, copy=False),
        "feature_names": list(feature_names),
        "times_utc_train": times_train,
        "rules": {
            "code": code,
            "label": label.astype(str),
            "rca_label": rca_label,
        },
        "meta": meta,
    }


# -----------------------------
# Training / Evaluation
# -----------------------------
def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rf_cfg: Optional[RFConfig] = None,
) -> Pipeline:
    """
    Retorna um Pipeline:
      SimpleImputer(median) -> RandomForestClassifier(...)
    """
    rf_cfg = rf_cfg or RFConfig()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    if X.ndim != 2:
        raise ValueError("fit_random_forest: X deve ser 2D (n,p).")
    if y.size != X.shape[0]:
        raise ValueError(f"fit_random_forest: y size {y.size} != X rows {X.shape[0]}")
    if np.unique(y).size < 2:
        raise ValueError("fit_random_forest: y possui <2 classes. Verifique rules/labels/gates.")

    # garante robustez: inf -> nan (imputer lida)
    X2 = X.copy()
    X2[~np.isfinite(X2)] = np.nan

    clf = RandomForestClassifier(
        n_estimators=int(rf_cfg.n_estimators),
        max_depth=rf_cfg.max_depth,
        min_samples_split=int(rf_cfg.min_samples_split),
        min_samples_leaf=int(rf_cfg.min_samples_leaf),
        max_features=rf_cfg.max_features,
        bootstrap=bool(rf_cfg.bootstrap),
        class_weight=rf_cfg.class_weight,
        random_state=int(rf_cfg.random_state),
        n_jobs=int(rf_cfg.n_jobs),
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", clf),
        ]
    )
    pipe.fit(X2, y)
    return pipe


def evaluate_classifier(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int).reshape(-1)

    X2 = X_test.copy()
    X2[~np.isfinite(X2)] = np.nan

    y_pred = model.predict(X2)
    acc = float(accuracy_score(y_test, y_pred))
    bacc = float(balanced_accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels_sorted = np.unique(np.concatenate([np.unique(y_test), np.unique(y_pred)])).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        target_names=[CODE_TO_LABEL.get(int(c), str(int(c))) for c in labels_sorted],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1m,
        "labels": labels_sorted,
        "confusion_matrix": cm.astype(int),
        "classification_report": report,
    }


def train_rf(
    dataset: Dict[str, Any],
    *,
    rf_cfg: Optional[RFConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """
    Treina e avalia RF.

    split_mode:
      - "time": mantém ordem temporal (recomendado p/ séries temporais).
      - "random": train_test_split estratificado.
    """
    rf_cfg = rf_cfg or RFConfig()
    train_cfg = train_cfg or TrainConfig()

    X = np.asarray(dataset["X"], dtype=float)
    y = np.asarray(dataset["y"], dtype=int).reshape(-1)
    feat_names = list(dataset.get("feature_names", list(DEFAULT_FEATURES)))
    times_train = dataset.get("times_utc_train", None)

    n = int(X.shape[0])
    if n < 50:
        raise ValueError(f"train_rf: dataset muito pequeno (n={n}).")
    if np.unique(y).size < 2:
        raise ValueError("train_rf: y tem <2 classes após máscara. Ajuste rules/gates/dados.")

    mode = (train_cfg.split_mode or "time").lower().strip()

    # ordena (se houver times e split temporal)
    if mode != "random" and times_train is not None:
        t = np.asarray(times_train).reshape(-1)
        if t.size == n:
            try:
                order = np.argsort(t.astype("datetime64[ns]"))
                X = X[order]
                y = y[order]
            except Exception:
                pass

    if mode == "random":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=float(train_cfg.test_size),
            random_state=int(train_cfg.random_state),
            stratify=y if np.unique(y).size > 1 else None,
        )
    else:
        idx_tr, idx_te = _time_split_indices(n, float(train_cfg.test_size))
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_te, y_te = X[idx_te], y[idx_te]

    model = fit_random_forest(X_tr, y_tr, rf_cfg=rf_cfg)
    metrics = evaluate_classifier(model, X_te, y_te)

    # feature importances
    try:
        rf = model.named_steps["rf"]
        importances = np.asarray(getattr(rf, "feature_importances_", np.array([])), dtype=float)
    except Exception:
        importances = np.array([], dtype=float)

    classes_ = np.unique(y).astype(int)
    return {
        "model": model,
        "feature_names": feat_names,
        "rf_cfg": asdict(rf_cfg),
        "train_cfg": asdict(train_cfg),
        "metrics": metrics,
        "feature_importances": importances,
        "classes_": classes_,
        "class_labels_": codes_to_labels(classes_),
        "dataset_meta": dataset.get("meta", {}),
    }


# -----------------------------
# Persistence bundle
# -----------------------------
def save_rf_bundle(
    path: Union[str, Path],
    train_out: Dict[str, Any],
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Salva um bundle joblib com:
      - model (Pipeline)
      - feature_names
      - code_to_label
      - configs
      - metrics
      - feature_importances
      - classes_
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": train_out["model"],
        "feature_names": list(train_out.get("feature_names", list(DEFAULT_FEATURES))),
        "code_to_label": dict(CODE_TO_LABEL),
        "rf_cfg": train_out.get("rf_cfg", {}),
        "train_cfg": train_out.get("train_cfg", {}),
        "metrics": train_out.get("metrics", {}),
        "feature_importances": np.asarray(train_out.get("feature_importances", np.array([])), dtype=float),
        "classes_": np.asarray(train_out.get("classes_", np.array([])), dtype=int),
        "dataset_meta": train_out.get("dataset_meta", {}),
        "extra_meta": extra_meta or {},
    }

    joblib.dump(bundle, p)
    return p


def load_rf_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"load_rf_bundle: arquivo não encontrado: {p}")
    bundle = joblib.load(p)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("load_rf_bundle: bundle inválido.")
    return bundle


# -----------------------------
# Inference helpers (produção)
# -----------------------------
def predict_from_out_model(
    bundle: Dict[str, Any],
    out_model: Dict[str, Any],
    *,
    times_utc: Optional[Any] = None,
    feature_names: Optional[Iterable[str]] = None,
    feature_spec: Optional[RFFeatureSpec] = None,
) -> Dict[str, Any]:
    """
    Gera predições RF a partir do out_model.
    Retorna code_pred, label_pred, proba (se disponível), e X.
    """
    model: Pipeline = bundle["model"]
    feat_names = list(feature_names or bundle.get("feature_names", list(DEFAULT_FEATURES)))
    feature_spec = feature_spec or RFFeatureSpec()

    feat_res = build_rf_features(out_model, times_utc=times_utc, spec=feature_spec)
    X = _subset_X_from_feature_result(feat_res, feat_names)

    code_pred = model.predict(X).astype(int)
    label_pred = codes_to_labels(code_pred)

    proba = None
    try:
        proba = model.predict_proba(X)
    except Exception:
        proba = None

    return {
        "X": X,
        "code_pred": code_pred,
        "label_pred": label_pred,
        "proba": proba,
        "feature_names": feat_names,
        "meta": {"feature_spec": asdict(feature_spec)},
    }
