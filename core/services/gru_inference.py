# core/services/gru_inference.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import joblib
import tensorflow as tf


@dataclass(frozen=True)
class GRUArtifacts:
    model: tf.keras.Model
    scaler: object
    labels: np.ndarray  # array de strings ou mapping idx->label


class GRUInferenceService:
    _artifacts: Optional[GRUArtifacts] = None

    @classmethod
    def load(cls) -> GRUArtifacts:
        if cls._artifacts is not None:
            return cls._artifacts

        model = tf.keras.models.load_model("core/ai_models/pv_gru_v1.h5", compile=False)
        scaler = joblib.load("core/ai_models/pv_gru_v1_scaler.pkl")
        labels = joblib.load("core/ai_models/pv_gru_v1_labels.pkl")  # ex: np.array([...])

        cls._artifacts = GRUArtifacts(model=model, scaler=scaler, labels=np.asarray(labels, dtype=object))
        return cls._artifacts

    @staticmethod
    def _build_windows(X: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (T, F)
        Retorna:
          Xw: (N, window, F)
          idx_end: índices no tempo correspondentes ao fim de cada janela
        """
        T, F = X.shape
        if T < window:
            return np.empty((0, window, F), dtype=float), np.empty((0,), dtype=int)

        ends = np.arange(window - 1, T, stride, dtype=int)
        N = ends.size
        Xw = np.empty((N, window, F), dtype=float)
        for k, e in enumerate(ends):
            Xw[k] = X[e - window + 1 : e + 1, :]
        return Xw, ends

    @classmethod
    def predict_series_labels(
        cls,
        X: np.ndarray,
        *,
        window: int = 96,
        stride: int = 4,
        batch_size: int = 512,
    ) -> Dict[str, np.ndarray]:
        """
        X: (T, F) features por timestep (float, pode ter NaN).
        Retorna arrays alinhados ao tempo original:
          - label_t: (T,) label (None nos primeiros window-1)
          - prob_t : (T,) prob max (NaN nos primeiros window-1)
        """
        art = cls.load()
        X = np.asarray(X, dtype=float)

        # substitui NaN por 0 antes do scaler (mas preserve flags de missing nas features!)
        X0 = np.where(np.isfinite(X), X, 0.0)

        # escala por timestep
        Xs = art.scaler.transform(X0)

        Xw, ends = cls._build_windows(Xs, window=window, stride=stride)
        T = X.shape[0]

        label_t = np.full(T, None, dtype=object)
        prob_t = np.full(T, np.nan, dtype=float)

        if Xw.shape[0] == 0:
            return {"label_t": label_t, "prob_t": prob_t}

        probs = art.model.predict(Xw, batch_size=batch_size, verbose=0)
        idx = np.argmax(probs, axis=1)
        pmax = probs[np.arange(probs.shape[0]), idx]

        labels = art.labels[idx]

        # escreve no índice do fim da janela (alinhamento p/ heatmap)
        label_t[ends] = labels
        prob_t[ends] = pmax

        # opcional: propaga label/prob entre ends (forward-fill) se você quiser label em todo timestep
        # aqui deixo "sparse" para você decidir.

        return {"label_t": label_t, "prob_t": prob_t}


