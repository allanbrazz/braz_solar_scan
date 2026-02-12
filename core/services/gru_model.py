# ================================
# core/services/gru_model.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# sklearn / joblib
try:
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover
    raise ImportError("scikit-learn não está instalado. Use: pip install scikit-learn") from e

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise ImportError("joblib não está instalado. Use: pip install joblib") from e

# tensorflow
try:
    import tensorflow as tf
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow não está instalado (necessário para GRU). "
        "Use: pip install tensorflow (ou tensorflow-cpu)."
    ) from e


# ----------------------------
# Defaults / Artefatos
# ----------------------------
DEFAULT_WINDOW = 96          # 24h @ 15-min
DEFAULT_STRIDE = 4           # 1h (barato p/ heatmap) — você pode usar 1
DEFAULT_BATCH = 512

DEFAULT_LABELS = np.array(
    [
        "normal",
        "meteo_error",
        "soiling",
        "degradation_like",
        "short_or_bypass",
        "string_disconnected",
        "partial_shading",
        "unknown",
    ],
    dtype=object,
)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "ai_models"
DEFAULT_MODEL_FILE = "pv_gru_v1.h5"
DEFAULT_SCALER_FILE = "pv_gru_v1_scaler.pkl"
DEFAULT_LABELS_FILE = "pv_gru_v1_labels.pkl"
DEFAULT_META_FILE = "pv_gru_v1_meta.pkl"


# ----------------------------
# Feature contract (a GRU "vê")
# ----------------------------
FEATURE_NAMES = [
    "g_poa",
    "tcell_c",
    "mismatch_rel",
    "g_cv_60m",
    "csi",
    "v_ratio",
    "i_ratio",
    "isfinite_vr",
    "isfinite_ir",
    "sun_up",
]


# ----------------------------
# Utils
# ----------------------------
def _nan_to_zero(x: np.ndarray) -> np.ndarray:
    """Converte NaN/inf para 0 (mantém shape)."""
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _finite01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.isfinite(x).astype(float)


def _series_len(x: Any) -> int:
    if x is None:
        return 0
    try:
        a = np.asarray(x)
        return int(a.size)
    except Exception:
        return 0


def _as_float_aligned(x: Any, n: int, *, fill: float = np.nan) -> np.ndarray:
    """
    Converte para float e ALINHA tamanho:
      - se menor: pad à direita com fill
      - se maior: trunca à esquerda? (aqui: trunca no início -> pega primeiros n)
    """
    if n <= 0:
        return np.asarray([], dtype=float)

    if x is None:
        return np.full(n, fill, dtype=float)

    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        return np.full(n, fill, dtype=float)

    if a.size == n:
        return a

    if a.size < n:
        out = np.full(n, fill, dtype=float)
        out[: a.size] = a
        return out

    # a.size > n
    return a[:n]


def forward_fill_obj(a: np.ndarray) -> np.ndarray:
    """Forward-fill para labels (object). Trata None e string vazia como "faltante"."""
    a = np.asarray(a, dtype=object).copy()
    last = None
    for i in range(a.size):
        v = a[i]
        if v is None:
            a[i] = last
            continue
        s = str(v).strip()
        if (s == "") or (s.lower() == "none"):
            a[i] = last
            continue
        last = v
    return a


def _artifact_paths(artifact_dir: Path) -> Dict[str, Path]:
    ad = Path(artifact_dir)
    return {
        "dir": ad,
        "model": ad / DEFAULT_MODEL_FILE,
        "scaler": ad / DEFAULT_SCALER_FILE,
        "labels": ad / DEFAULT_LABELS_FILE,
        "meta": ad / DEFAULT_META_FILE,
    }


def _load_labels_any(obj: Any) -> np.ndarray:
    """
    Suporta labels salvos como:
      - np.ndarray/list[str]
      - dict[int,str]
    Retorna np.ndarray dtype=object onde idx -> label string.
    """
    if obj is None:
        return np.asarray(DEFAULT_LABELS, dtype=object)

    # dict (idx -> label)
    if isinstance(obj, dict):
        try:
            keys = sorted(int(k) for k in obj.keys())
            labels = [obj[int(k)] for k in keys]
            return np.asarray(labels, dtype=object)
        except Exception:
            # fallback: valores
            return np.asarray(list(obj.values()), dtype=object)

    # lista/np.array
    arr = np.asarray(obj, dtype=object).reshape(-1)
    if arr.size == 0:
        return np.asarray(DEFAULT_LABELS, dtype=object)
    return arr


def _validate_feature_contract(X: np.ndarray, expected_names: List[str], scaler: Optional[StandardScaler] = None) -> None:
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (T,F).")
    F = int(X.shape[1])
    expF = int(len(expected_names))
    if F != expF:
        raise ValueError(
            f"Feature mismatch: X tem F={F}, esperado F={expF}.\n"
            f"Esperado (ordem): {expected_names}"
        )
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        sf = int(getattr(scaler, "n_features_in_", expF))
        if sf != F:
            raise ValueError(
                f"Scaler foi treinado com n_features={sf}, mas X tem F={F}.\n"
                f"Provável divergência de contrato/ordem de features."
            )


# ----------------------------
# Feature builder (power_model -> X)
# ----------------------------
def build_feature_matrix_from_power_model(out_model: Dict[str, Any]) -> np.ndarray:
    """
    Monta X(t) = (T, 10) com contrato FIXO (FEATURE_NAMES).

    Correções:
    - T não depende só de tcell_c (pode estar ausente). Usa o maior T disponível.
    - Alinha tamanhos (pad/trunc) sem "zerar tudo" por mismatch de comprimento.
    - Usa g_poa_used/g_poa/g_poa_wm2 como fallback.
    """
    g = out_model.get("g_poa_used")
    if g is None:
        g = out_model.get("g_poa")
    if g is None:
        g = out_model.get("g_poa_wm2")

    tc = out_model.get("tcell_c")
    mis = out_model.get("mismatch_rel")
    gcv = out_model.get("g_cv_60m")
    csi = out_model.get("csi")
    vr = out_model.get("v_ratio")
    ir = out_model.get("i_ratio")

    # Determina T (o maior comprimento disponível)
    T = max(
        _series_len(g),
        _series_len(tc),
        _series_len(mis),
        _series_len(gcv),
        _series_len(csi),
        _series_len(vr),
        _series_len(ir),
    )
    if T <= 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)

    g = _as_float_aligned(g, T)
    tc = _as_float_aligned(tc, T)
    mis = _as_float_aligned(mis, T)
    gcv = _as_float_aligned(gcv, T)
    csi = _as_float_aligned(csi, T)
    vr = _as_float_aligned(vr, T)
    ir = _as_float_aligned(ir, T)

    is_vr = _finite01(vr)
    is_ir = _finite01(ir)

    # sol acima do horizonte (robusto)
    g0 = np.where(np.isfinite(g), g, 0.0)
    sun_up = (g0 >= 20.0).astype(float)

    X = np.column_stack([g, tc, mis, gcv, csi, vr, ir, is_vr, is_ir, sun_up]).astype(float)
    return X


# ----------------------------
# Windowing
# ----------------------------
def make_windows(
    X: np.ndarray,
    y_t: Optional[np.ndarray] = None,
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    label_mode: str = "last",   # "last" | "majority"
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    X: (T, F)
    y_t: (T,) labels int (ou None)

    Retorna:
      Xw: (N, window, F)
      yw: (N,) ou None
      ends: (N,) índices do timestep final de cada janela
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("make_windows: X deve ser 2D (T,F).")
    T, F = X.shape

    window = int(window)
    stride = int(max(1, stride))
    if T < window:
        return (
            np.empty((0, window, F), dtype=float),
            (None if y_t is None else np.empty((0,), dtype=int)),
            np.empty((0,), dtype=int),
        )

    ends = np.arange(window - 1, T, stride, dtype=int)
    N = ends.size
    Xw = np.empty((N, window, F), dtype=float)

    yw = None
    if y_t is not None:
        y_t = np.asarray(y_t, dtype=int)
        if y_t.size != T:
            raise ValueError("make_windows: y_t deve ter tamanho T.")
        yw = np.empty((N,), dtype=int)

    for k, e in enumerate(ends):
        sl = slice(e - window + 1, e + 1)
        Xw[k] = X[sl, :]
        if yw is not None:
            if label_mode == "majority":
                yy = y_t[sl]
                yy = yy[yy >= 0]
                if yy.size == 0:
                    yw[k] = -1
                else:
                    vals, cnt = np.unique(yy, return_counts=True)
                    yw[k] = int(vals[np.argmax(cnt)])
            else:
                yw[k] = int(y_t[e])

    return Xw, yw, ends


# ----------------------------
# Modelo GRU
# ----------------------------
@dataclass(frozen=True)
class GRUConfig:
    window: int = DEFAULT_WINDOW
    stride: int = DEFAULT_STRIDE
    batch_size: int = DEFAULT_BATCH
    epochs: int = 40
    lr: float = 1e-3
    dropout: float = 0.20
    gru1: int = 64
    gru2: int = 32
    l2: float = 0.0
    patience: int = 6
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5


def build_gru_classifier(
    *,
    n_features: int,
    n_classes: int,
    config: GRUConfig,
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(config.l2) if config.l2 and config.l2 > 0 else None

    inp = tf.keras.Input(shape=(config.window, n_features), name="x")
    x = tf.keras.layers.GRU(config.gru1, return_sequences=True, kernel_regularizer=reg)(inp)
    x = tf.keras.layers.Dropout(config.dropout)(x)
    x = tf.keras.layers.GRU(config.gru2, return_sequences=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(config.dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="y")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="pv_gru_classifier")
    opt = tf.keras.optimizers.Adam(learning_rate=float(config.lr))
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------
# Artefatos
# ----------------------------
@dataclass
class GRUArtifacts:
    model: tf.keras.Model
    scaler: StandardScaler
    labels: np.ndarray  # idx -> label string
    feature_names: List[str]
    window: int
    stride: int


def save_artifacts(art: GRUArtifacts, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
    p = _artifact_paths(artifact_dir)
    p["dir"].mkdir(parents=True, exist_ok=True)

    art.model.save(str(p["model"]))

    joblib.dump(art.scaler, p["scaler"])
    joblib.dump(np.asarray(art.labels, dtype=object), p["labels"])
    joblib.dump(
        {
            "feature_names": list(art.feature_names),
            "window": int(art.window),
            "stride": int(art.stride),
            "n_features": int(len(art.feature_names)),
            "n_classes": int(np.asarray(art.labels).size),
        },
        p["meta"],
    )


def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> GRUArtifacts:
    p = _artifact_paths(artifact_dir)
    if not p["model"].exists():
        raise FileNotFoundError(f"Modelo GRU não encontrado: {p['model']}")
    if not p["scaler"].exists():
        raise FileNotFoundError(f"Scaler não encontrado: {p['scaler']}")
    if not p["labels"].exists():
        raise FileNotFoundError(f"Labels não encontrados: {p['labels']}")

    model = tf.keras.models.load_model(str(p["model"]), compile=False)
    scaler = joblib.load(p["scaler"])
    raw_labels = joblib.load(p["labels"])
    labels = _load_labels_any(raw_labels)

    meta = joblib.load(p["meta"]) if p["meta"].exists() else {}
    feature_names = list(meta.get("feature_names", FEATURE_NAMES))
    window = int(meta.get("window", DEFAULT_WINDOW))
    stride = int(meta.get("stride", DEFAULT_STRIDE))

    return GRUArtifacts(
        model=model,
        scaler=scaler,
        labels=labels,
        feature_names=feature_names,
        window=window,
        stride=stride,
    )


# ----------------------------
# Treino
# ----------------------------
def train_gru_from_series(
    *,
    X_series_list: Sequence[np.ndarray],
    y_series_list: Sequence[np.ndarray],
    labels: np.ndarray = DEFAULT_LABELS,
    config: GRUConfig = GRUConfig(),
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    class_weight: Optional[Dict[int, float]] = None,
) -> GRUArtifacts:
    if len(X_series_list) != len(y_series_list):
        raise ValueError("X_series_list e y_series_list devem ter o mesmo tamanho.")

    Xw_all: List[np.ndarray] = []
    yw_all: List[np.ndarray] = []

    for X, y_t in zip(X_series_list, y_series_list):
        X = np.asarray(X, dtype=float)
        y_t = np.asarray(y_t, dtype=int)
        if X.ndim != 2:
            raise ValueError("Cada X deve ser 2D (T,F).")
        if y_t.ndim != 1 or y_t.size != X.shape[0]:
            raise ValueError("Cada y_t deve ser 1D e ter tamanho T.")
        _validate_feature_contract(X, FEATURE_NAMES, scaler=None)

        Xw, yw, _ = make_windows(X, y_t, window=config.window, stride=config.stride, label_mode="last")
        if yw is None:
            continue
        m = yw >= 0
        if m.any():
            Xw_all.append(Xw[m])
            yw_all.append(yw[m])

    if not Xw_all:
        raise ValueError("Nenhuma janela válida para treino (Xw_all vazio).")

    Xw = np.concatenate(Xw_all, axis=0)
    yw = np.concatenate(yw_all, axis=0)

    N, W, F = Xw.shape

    # escala: fit em (amostras*tempo, features)
    X2 = Xw.reshape(N * W, F)
    X2_0 = _nan_to_zero(X2)

    scaler = StandardScaler()
    scaler.fit(X2_0)
    X2s = scaler.transform(X2_0)
    Xws = X2s.reshape(N, W, F)

    n_classes = int(np.asarray(labels).size)
    model = build_gru_classifier(n_features=F, n_classes=n_classes, config=config)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            min_lr=config.min_lr,
        ),
    ]

    idx = np.arange(N)
    np.random.shuffle(idx)
    cut = int(round(0.85 * N))
    tr = idx[:cut]
    va = idx[cut:]

    model.fit(
        Xws[tr],
        yw[tr],
        validation_data=(Xws[va], yw[va]),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        verbose=1,
        class_weight=class_weight,
        callbacks=cbs,
    )

    art = GRUArtifacts(
        model=model,
        scaler=scaler,
        labels=np.asarray(labels, dtype=object),
        feature_names=list(FEATURE_NAMES),
        window=int(config.window),
        stride=int(config.stride),
    )
    save_artifacts(art, artifact_dir=artifact_dir)
    return art


# ----------------------------
# Inferência (singleton)
# ----------------------------
class GRUInferenceService:
    _artifacts: Optional[GRUArtifacts] = None

    @classmethod
    def get(cls, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> GRUArtifacts:
        if cls._artifacts is None:
            cls._artifacts = load_artifacts(artifact_dir=artifact_dir)
        return cls._artifacts

    @staticmethod
    def _predict_windows(model: tf.keras.Model, Xw: np.ndarray, batch_size: int) -> np.ndarray:
        probs = model.predict(Xw, batch_size=int(batch_size), verbose=0)
        probs = np.asarray(probs, dtype=float)
        # protege contra NaN/inf no output do modelo (raro, mas possível)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return probs

    @classmethod
    def predict_series(
        cls,
        X: np.ndarray,
        *,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH,
        forward_fill: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        X: (T,F) float (pode ter NaN/inf)
        Retorna:
          label_t: (T,) object (strings ou None)
          prob_t : (T,) float
        """
        art = cls.get(artifact_dir=artifact_dir)

        X = np.asarray(X, dtype=float)
        _validate_feature_contract(X, art.feature_names, scaler=art.scaler)

        W = int(window if window is not None else art.window)
        S = int(stride if stride is not None else art.stride)

        # prepara + escala (sempre sem NaN/inf)
        X0 = _nan_to_zero(X)
        Xs = art.scaler.transform(X0)

        Xw, _, ends = make_windows(Xs, None, window=W, stride=S)
        T = int(X.shape[0])

        label_t = np.full(T, None, dtype=object)
        prob_t = np.full(T, np.nan, dtype=float)

        if Xw.shape[0] == 0:
            return {"label_t": label_t, "prob_t": prob_t}

        probs = cls._predict_windows(art.model, Xw, batch_size=batch_size)

        idx = np.argmax(probs, axis=1).astype(int)
        pmax = probs[np.arange(probs.shape[0]), idx]

        # idx -> label string (garante dtype object e indexação segura)
        labels_arr = np.asarray(art.labels, dtype=object).reshape(-1)
        idx_clip = np.clip(idx, 0, max(labels_arr.size - 1, 0))
        labels = labels_arr[idx_clip]

        label_t[ends] = labels
        prob_t[ends] = pmax

        if forward_fill:
            label_t = forward_fill_obj(label_t)
            # prob: forward-fill do último pmax conhecido
            lastp = np.nan
            for i in range(prob_t.size):
                if np.isfinite(prob_t[i]):
                    lastp = prob_t[i]
                else:
                    prob_t[i] = lastp

        return {"label_t": label_t, "prob_t": prob_t}


# ----------------------------
# Smoke test
# ----------------------------
def smoke_test_train_and_infer(tmp_dir: Optional[str] = None) -> None:
    out_dir = Path(tmp_dir) if tmp_dir else (DEFAULT_ARTIFACT_DIR / "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)

    T = 10 * 96
    g = np.clip(800 * np.sin(np.linspace(0, 10 * np.pi, T)) + 50 * rng.standard_normal(T), 0, None)
    tc = 25 + (g / 800) * 20 + 1.0 * rng.standard_normal(T)
    mis = 0.02 * rng.standard_normal(T)

    gcv = np.clip(0.08 + 0.03 * rng.standard_normal(T), 0, 1.2)
    meteo_mask = np.zeros(T, dtype=bool)
    meteo_mask[3 * 96: 4 * 96] = True
    gcv[meteo_mask] = np.clip(0.6 + 0.1 * rng.standard_normal(meteo_mask.sum()), 0, 1.2)

    csi = np.clip(1.0 + 0.05 * rng.standard_normal(T), 0, 1.5)
    vr = np.clip(1.0 + 0.01 * rng.standard_normal(T), 0, 1.3)
    ir = np.clip(1.0 + 0.01 * rng.standard_normal(T), 0, 1.3)
    sun_up = (g >= 20).astype(float)

    X = np.column_stack([g, tc, mis, gcv, csi, vr, ir, _finite01(vr), _finite01(ir), sun_up])

    labels = np.array(["normal", "meteo_error"], dtype=object)
    y_t = np.zeros(T, dtype=int)
    y_t[meteo_mask] = 1

    cfg = GRUConfig(epochs=6, batch_size=256, window=96, stride=4)
    train_gru_from_series(
        X_series_list=[X],
        y_series_list=[y_t],
        labels=labels,
        config=cfg,
        artifact_dir=out_dir,
    )

    GRUInferenceService._artifacts = None
    pred = GRUInferenceService.predict_series(X, artifact_dir=out_dir, window=96, stride=4, forward_fill=True)

    nonnull = pred["label_t"][(pred["label_t"] != None)]  # noqa: E711
    unique, cnt = np.unique(nonnull, return_counts=True)
    print("Smoke test OK.")
    print("Distribuição labels:", dict(zip(unique.tolist(), cnt.tolist())))


if __name__ == "__main__":
    smoke_test_train_and_infer()
