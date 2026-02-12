# ================================
# core/services/fuzzy_diagnostic.py
# ================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np

try:
    import skfuzzy as fuzz
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-fuzzy não está instalado. Use: pip install scikit-fuzzy"
    ) from e


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


CODE_TO_LABEL = {
    DiagnosticCodes.INVALID: "invalid",
    DiagnosticCodes.NORMAL: "normal",
    DiagnosticCodes.METEO_ERROR: "meteo_error",
    DiagnosticCodes.SOILING: "soiling",
    DiagnosticCodes.DEGRADATION: "degradation_like",
    DiagnosticCodes.SHORT_BYPASS: "short_or_bypass",
    DiagnosticCodes.STRING_DISCONNECTED: "string_disconnected",
    DiagnosticCodes.PARTIAL_SHADING: "partial_shading",
}


def _to_float_vec(x, n: Optional[int] = None, fill: float = np.nan) -> np.ndarray:
    if x is None:
        return np.full(n or 0, fill, dtype=float)
    a = np.asarray(x, dtype=float)
    if n is not None and a.size not in (0, n):
        return np.full(n, fill, dtype=float)
    if n is not None and a.size == 0:
        return np.full(n, fill, dtype=float)
    return a


def _to_bool_vec(x, n: Optional[int] = None) -> np.ndarray:
    if x is None:
        return np.zeros(n or 0, dtype=bool)
    a = np.asarray(x, dtype=bool)
    if n is not None and a.size not in (0, n):
        return np.zeros(n, dtype=bool)
    if n is not None and a.size == 0:
        return np.zeros(n, dtype=bool)
    return a


def _and(*mus: np.ndarray) -> np.ndarray:
    return np.minimum.reduce(mus)


def _or(*mus: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(mus)


def _safe_mu(mu: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zera pertinência onde a entrada não é finita."""
    return np.where(np.isfinite(x), mu, 0.0)


class FuzzyDiagnosticService:
    """
    Diagnóstico fuzzy (Sugeno categórico).

    Entradas esperadas:
      - mismatch_rel : (P_real - P_model)/P_model
      - g_cv_60m     : coeficiente de variação irradiância (instabilidade)
      - csi          : clear sky index (G / G_clear) [pode ser NaN]
      - v_ratio      : Vdc_meas / Vdc_expected
      - i_ratio      : Idc_meas / Idc_expected
      - valid_ml     : máscara booleana de validade

    Saída:
      - np.ndarray[int] com códigos DiagnosticCodes (0..7)
    """

    def __init__(
        self,
        *,
        strings_count: Optional[int] = None,
        mismatch_max: float = 2.0,
        gcv_max: float = 1.2,
        csi_max: float = 1.5,
        ratio_max: float = 1.30,
        sigma_string_drop: float = 0.04,
    ) -> None:
        self.codes = DiagnosticCodes()
        self.strings_count = int(strings_count) if strings_count is not None else None
        self.sigma_string_drop = float(max(sigma_string_drop, 1e-6))

        # Universos
        self.uni_mis = np.linspace(0.0, float(mismatch_max), 401)  # |mismatch|
        self.uni_gcv = np.linspace(0.0, float(gcv_max), 301)
        self.uni_csi = np.linspace(0.0, float(csi_max), 301)
        self.uni_r   = np.linspace(0.0, float(ratio_max), 301)
        self.uni_dv  = np.linspace(0.0, 0.60, 301)               # |v_ratio - i_ratio|

        # =========================
        # Membership Functions (tuning)
        # =========================

        # mismatch magnitude (menos permissivo em 5%)
        self.mis_low  = fuzz.trapmf(self.uni_mis, [0.00, 0.00, 0.02, 0.05])
        self.mis_med  = fuzz.trimf(self.uni_mis, [0.03, 0.10, 0.25])
        self.mis_high = fuzz.trapmf(self.uni_mis, [0.18, 0.30, self.uni_mis[-1], self.uni_mis[-1]])

        # g_cv
        self.gcv_low  = fuzz.trapmf(self.uni_gcv, [0.00, 0.00, 0.08, 0.15])
        self.gcv_med  = fuzz.trimf(self.uni_gcv, [0.10, 0.25, 0.45])
        self.gcv_high = fuzz.trapmf(self.uni_gcv, [0.35, 0.55, self.uni_gcv[-1], self.uni_gcv[-1]])

        # CSI
        self.csi_vlow = fuzz.trapmf(self.uni_csi, [0.00, 0.00, 0.05, 0.15])
        self.csi_low  = fuzz.trimf(self.uni_csi, [0.10, 0.25, 0.45])
        self.csi_mid  = fuzz.trimf(self.uni_csi, [0.35, 0.70, 1.05])
        self.csi_high = fuzz.trapmf(self.uni_csi, [0.90, 1.05, 1.20, 1.30])
        self.csi_over = fuzz.trapmf(self.uni_csi, [1.15, 1.25, self.uni_csi[-1], self.uni_csi[-1]])

        # ratios
        self.r_vlow  = fuzz.trapmf(self.uni_r, [0.00, 0.00, 0.55, 0.75])
        self.r_low   = fuzz.trimf(self.uni_r, [0.70, 0.85, 0.97])
        self.r_near1 = fuzz.trimf(self.uni_r, [0.92, 1.00, 1.08])

        # delta V/I
        self.dv_low  = fuzz.trapmf(self.uni_dv, [0.00, 0.00, 0.04, 0.08])
        self.dv_high = fuzz.trapmf(self.uni_dv, [0.06, 0.12, 0.60, 0.60])

    def _mu(self, uni: np.ndarray, mf: np.ndarray, x: np.ndarray) -> np.ndarray:
        mu = fuzz.interp_membership(uni, mf, x)
        return _safe_mu(mu, x)

    def _mu_string_drop_target(self, i_ratio: np.ndarray) -> np.ndarray:
        sc = self.strings_count
        if sc is None or sc < 2:
            return np.zeros_like(i_ratio, dtype=float)
        target = (sc - 1.0) / sc
        mu = np.exp(-0.5 * ((i_ratio - target) / self.sigma_string_drop) ** 2)
        return _safe_mu(mu, i_ratio)

    def debug_scores(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Retorna scores, pertinências principais e winner por ponto (para debug)."""
        out = {}

        codes = self.predict(features)
        out["pred_code"] = codes
        out["pred_label"] = self.to_labels(codes)
        return out

    def predict(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        mis = _to_float_vec(features.get("mismatch_rel", None))
        n = mis.size
        if n == 0:
            return np.array([], dtype=int)

        gcv = _to_float_vec(features.get("g_cv_60m", None), n=n)
        csi = _to_float_vec(features.get("csi", None), n=n)
        vr  = _to_float_vec(features.get("v_ratio", None), n=n)
        ir  = _to_float_vec(features.get("i_ratio", None), n=n)
        valid = _to_bool_vec(features.get("valid_ml", None), n=n)

        # flags de ausência (crítico para não "jogar tudo" em METEO)
        ratio_missing = (~np.isfinite(vr)) | (~np.isfinite(ir))
        mu_ratio_missing = ratio_missing.astype(float)

        csi_missing = ~np.isfinite(csi)
        mu_csi_missing = csi_missing.astype(float)

        # magnitudes
        mis_mag   = np.clip(np.abs(mis), 0.0, self.uni_mis[-1])
        mis_under = np.clip(-mis,        0.0, self.uni_mis[-1])  # underperformance
        dv        = np.clip(np.abs(vr - ir), 0.0, self.uni_dv[-1])

        # pertinências (mismatch)
        mu_mis_low  = self._mu(self.uni_mis, self.mis_low,  mis_mag)
        mu_mis_med  = self._mu(self.uni_mis, self.mis_med,  mis_mag)
        mu_mis_high = self._mu(self.uni_mis, self.mis_high, mis_mag)

        mu_u_med   = self._mu(self.uni_mis, self.mis_med,  mis_under)
        mu_u_high  = self._mu(self.uni_mis, self.mis_high, mis_under)

        # gcv
        mu_gcv_low  = self._mu(self.uni_gcv, self.gcv_low,  gcv)
        mu_gcv_med  = self._mu(self.uni_gcv, self.gcv_med,  gcv)
        mu_gcv_high = self._mu(self.uni_gcv, self.gcv_high, gcv)

        # csi
        mu_csi_vlow = self._mu(self.uni_csi, self.csi_vlow, csi)
        mu_csi_low  = self._mu(self.uni_csi, self.csi_low,  csi)
        mu_csi_mid  = self._mu(self.uni_csi, self.csi_mid,  csi)
        mu_csi_high = self._mu(self.uni_csi, self.csi_high, csi)
        mu_csi_over = self._mu(self.uni_csi, self.csi_over, csi)

        # ratios
        mu_v_vlow   = self._mu(self.uni_r, self.r_vlow,  vr)
        mu_v_low    = self._mu(self.uni_r, self.r_low,   vr)
        mu_v_near1  = self._mu(self.uni_r, self.r_near1, vr)

        mu_i_vlow   = self._mu(self.uni_r, self.r_vlow,  ir)
        mu_i_low    = self._mu(self.uni_r, self.r_low,   ir)
        mu_i_near1  = self._mu(self.uni_r, self.r_near1, ir)

        # delta v/i
        mu_dv_low   = self._mu(self.uni_dv, self.dv_low,  dv)
        mu_dv_high  = self._mu(self.uni_dv, self.dv_high, dv)

        mu_i_drop_target = self._mu_string_drop_target(ir)

        # =========================
        # Regras Sugeno (categórico)
        # =========================
        scores = np.zeros((n, 8), dtype=float)

        def add_rule(code: int, w: np.ndarray) -> None:
            scores[:, code] += np.where(valid, w, 0.0)

        # NORMAL
        add_rule(self.codes.NORMAL, mu_mis_low)
        add_rule(
            self.codes.NORMAL,
            _and(mu_mis_med, mu_gcv_low, mu_v_near1, mu_i_near1, _or(mu_csi_mid, mu_csi_high, 1.0 - mu_csi_over, mu_csi_missing)),
        )

        # METEO_ERROR
        add_rule(self.codes.METEO_ERROR, _and(_or(mu_mis_med, mu_mis_high), mu_gcv_high))
        add_rule(self.codes.METEO_ERROR, _and(_or(mu_mis_med, mu_mis_high), _or(mu_csi_vlow, mu_csi_over)))
        add_rule(
            self.codes.METEO_ERROR,
            _and(mu_mis_high, mu_v_near1, mu_i_near1, _or(mu_gcv_med, mu_gcv_high, mu_csi_low, mu_csi_over)),
        )

        # SOILING (V ~1, I baixo, céu estável)
        add_rule(
            self.codes.SOILING,
            _and(_or(mu_u_med, mu_u_high), mu_gcv_low, _or(mu_csi_mid, mu_csi_high, mu_csi_missing), mu_v_near1, mu_i_low),
        )

        # DEGRADATION (assinatura elétrica: V e I baixos e coerentes)
        add_rule(
            self.codes.DEGRADATION,
            _and(_or(mu_u_med, mu_u_high), mu_gcv_low, _or(mu_csi_mid, mu_csi_high, mu_csi_missing), mu_v_low, mu_i_low, mu_dv_low),
        )

        # >>> NOVO: underperformance forte + céu estável, mas sem V/I confiáveis -> hardware_like
        add_rule(
            self.codes.DEGRADATION,
            _and(mu_u_high, mu_gcv_low, _or(mu_csi_mid, mu_csi_high, mu_csi_missing), mu_ratio_missing),
        )

        # SHORT/BYPASS (V baixo com I ~1)
        add_rule(
            self.codes.SHORT_BYPASS,
            _and(_or(mu_u_med, mu_u_high), _or(mu_v_vlow, mu_v_low), mu_i_near1, _or(mu_csi_mid, mu_csi_high, mu_csi_missing)),
        )

        # STRING DESCONECTADA
        add_rule(self.codes.STRING_DISCONNECTED, _and(mu_u_high, mu_v_near1, mu_i_vlow))
        add_rule(self.codes.STRING_DISCONNECTED, _and(mu_u_high, mu_v_near1, mu_i_drop_target))

        # SOMBREAMENTO PARCIAL
        add_rule(
            self.codes.PARTIAL_SHADING,
            _and(mu_u_high, _or(mu_gcv_low, mu_gcv_med), _or(mu_csi_mid, mu_csi_high, mu_csi_missing), mu_v_low, mu_i_low, mu_dv_high),
        )

        # decisão
        out = np.full(n, self.codes.INVALID, dtype=int)
        total = scores.sum(axis=1)
        any_rule = total > 1e-9

        # ignora INVALID no argmax
        winner = np.argmax(scores[:, 1:], axis=1) + 1

        # >>> NOVO fallback:
        # - mismatch baixo => NORMAL
        # - underperformance forte + céu estável => DEGRADATION (hardware_like)
        # - senão => METEO_ERROR
        hw_fallback = _and(mu_u_high, mu_gcv_low, _or(mu_csi_mid, mu_csi_high, mu_csi_missing)) > 0.20

        fallback = np.where(
            mu_mis_low >= np.maximum(mu_mis_med, mu_mis_high),
            self.codes.NORMAL,
            np.where(hw_fallback, self.codes.DEGRADATION, self.codes.METEO_ERROR),
        )

        pred = np.where(any_rule, winner, fallback).astype(int)

        ok = valid & np.isfinite(mis_mag)
        out[ok] = pred[ok]
        return out

    def predict_labels(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        codes = self.predict(features)
        return self.to_labels(codes)

    @staticmethod
    def to_labels(codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes)
        return np.array([CODE_TO_LABEL.get(int(c), "invalid") for c in codes], dtype=object)


def run_fuzzy_diagnostic(
    *,
    mismatch_rel,
    g_cv_60m,
    csi=None,
    v_ratio=None,
    i_ratio=None,
    valid=None,
    strings_count: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    svc = FuzzyDiagnosticService(strings_count=strings_count)

    n = np.asarray(mismatch_rel, dtype=float).size
    if n == 0:
        return {"rca_code": np.array([], dtype=int), "rca_label": np.array([], dtype=object)}

    feats = {
        "mismatch_rel": mismatch_rel,
        "g_cv_60m": g_cv_60m,
        "csi": csi,
        "v_ratio": v_ratio,
        "i_ratio": i_ratio,
        "valid_ml": valid,
    }

    codes = svc.predict(feats)
    labels = svc.to_labels(codes)
    return {"rca_code": codes, "rca_label": labels}


def diagnose(**kwargs):
    return run_fuzzy_diagnostic(**kwargs)

def predict(**kwargs):
    return run_fuzzy_diagnostic(**kwargs)
