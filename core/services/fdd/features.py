# ================================
# core/services/features.py
# (RF-only feature engineering)  <-- (sugestão: mover p/ core/services/fdd/features.py)
# ================================
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------
# Result + simple imputer (RF needs no NaN)
# ---------------------------------
@dataclass(frozen=True)
class RFImputer:
    """
    RandomForest (sklearn) NÃO aceita NaN.
    Estratégia: preencher NaN/inf com mediana por feature (fit no treino).
    """
    feature_names: List[str]
    medians: np.ndarray  # shape (n_features,)
    fill_if_all_nan: float = 0.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("RFImputer.transform: X deve ser 2D (n_samples, n_features).")
        if X.shape[1] != self.medians.size:
            raise ValueError("RFImputer.transform: n_features incompatível com medians.")

        X2 = X.copy()
        # inf -> nan
        X2[~np.isfinite(X2)] = np.nan

        med = self.medians.copy()
        med[~np.isfinite(med)] = float(self.fill_if_all_nan)

        nan_mask = np.isnan(X2)
        if nan_mask.any():
            X2[nan_mask] = np.take(med, np.where(nan_mask)[1])
        return X2.astype(np.float32, copy=False)


@dataclass(frozen=True)
class FeatureBuildResult:
    """
    X: (T x n_features) float32 (pode conter NaN antes do imputador)
    feature_names: ordem das features
    valid_mask: máscara recomendada p/ ML
    series: séries úteis para debug/auditoria
    meta: metadados usados
    """
    X: np.ndarray
    feature_names: List[str]
    valid_mask: np.ndarray
    series: Dict[str, np.ndarray]
    meta: Dict[str, Any]


# ---------------------------------
# Numerics: safe arrays + rolling
# ---------------------------------
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


def _align_len(x: Any, T: int, *, fill: float = np.nan) -> np.ndarray:
    a = _to_np_1d(x, dtype=float)
    if a.size == 0:
        return np.full(T, fill, dtype=float)
    if a.size == T:
        return a.astype(float, copy=False)
    # NÃO fazer pad/trunc por padrão: isso mascara bug de alinhamento temporal
    raise ValueError(f"_align_len: série tamanho {a.size} != T={T}. Verifique o pipeline de merge/alinhamento.")


def _align_bool(x: Any, T: int, *, fill: bool = False) -> np.ndarray:
    a = _to_np_1d(x, dtype=bool)
    if a.size == 0:
        return np.full(T, bool(fill), dtype=bool)
    if a.size == T:
        return a
    raise ValueError(f"_align_bool: série tamanho {a.size} != T={T}. Verifique o pipeline de merge/alinhamento.")


def _rolling_nanmean(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    w = int(max(1, w))
    if w == 1:
        return x.copy()
    if w > n:
        return out

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)

    csum = np.cumsum(x0, dtype=float)
    ccount = np.cumsum(valid.astype(float), dtype=float)

    csum0 = np.concatenate(([0.0], csum))
    ccount0 = np.concatenate(([0.0], ccount))

    end_idx = np.arange(w, n + 1)
    start_idx = end_idx - w

    sum_w = csum0[end_idx] - csum0[start_idx]
    cnt_w = ccount0[end_idx] - ccount0[start_idx]
    mean_w = sum_w / np.maximum(cnt_w, 1.0)

    out[w - 1 :] = np.where(cnt_w > 0, mean_w, np.nan)
    return out


def _rolling_nanstd(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    w = int(max(1, w))
    if w == 1:
        return np.zeros_like(x, dtype=float)
    if w > n:
        return out

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)
    x02 = np.where(valid, x * x, 0.0)

    csum = np.cumsum(x0, dtype=float)
    csum2 = np.cumsum(x02, dtype=float)
    ccount = np.cumsum(valid.astype(float), dtype=float)

    csum0 = np.concatenate(([0.0], csum))
    csum20 = np.concatenate(([0.0], csum2))
    ccount0 = np.concatenate(([0.0], ccount))

    end_idx = np.arange(w, n + 1)
    start_idx = end_idx - w

    sum_w = csum0[end_idx] - csum0[start_idx]
    sum2_w = csum20[end_idx] - csum20[start_idx]
    cnt_w = ccount0[end_idx] - ccount0[start_idx]

    mean_w = sum_w / np.maximum(cnt_w, 1.0)
    var_w = (sum2_w / np.maximum(cnt_w, 1.0)) - mean_w * mean_w
    std_w = np.sqrt(np.maximum(var_w, 0.0))

    out[w - 1 :] = np.where(cnt_w > 0, std_w, np.nan)
    return out


def _diff(x: np.ndarray, lag: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if lag <= 0 or x.size == 0:
        return out
    out[lag:] = x[lag:] - x[:-lag]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray, *, eps: float) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    d = np.maximum(np.where(np.isfinite(den), den, np.nan), float(eps))
    out = num / d
    out[~np.isfinite(out)] = np.nan
    return out


def _time_cyc_features(times_utc: Optional[Sequence[Union[datetime, np.datetime64]]], T: int) -> Dict[str, np.ndarray]:
    # correção: "if not times_utc" falha com arrays numpy e pode ser ambíguo
    if times_utc is None:
        nan = np.full(T, np.nan, dtype=float)
        return {"tod_sin": nan.copy(), "tod_cos": nan.copy(), "doy_sin": nan.copy(), "doy_cos": nan.copy()}

    tt = list(times_utc)
    if len(tt) == 0:
        nan = np.full(T, np.nan, dtype=float)
        return {"tod_sin": nan.copy(), "tod_cos": nan.copy(), "doy_sin": nan.copy(), "doy_cos": nan.copy()}

    # normaliza len T (repete último)
    if len(tt) < T:
        tt = tt + [tt[-1]] * (T - len(tt))
    elif len(tt) > T:
        tt = tt[:T]

    mins = np.empty(T, dtype=float)
    doy = np.empty(T, dtype=float)

    for i, t in enumerate(tt):
        if isinstance(t, np.datetime64):
            ts = t.astype("datetime64[s]").astype(int)
            dt = datetime.utcfromtimestamp(int(ts))
        else:
            dt = t
        mins[i] = dt.hour * 60 + dt.minute
        doy[i] = float(dt.timetuple().tm_yday)

    tod = (mins / (24.0 * 60.0)) * (2.0 * np.pi)
    doy_ang = (doy / 366.0) * (2.0 * np.pi)

    return {
        "tod_sin": np.sin(tod),
        "tod_cos": np.cos(tod),
        "doy_sin": np.sin(doy_ang),
        "doy_cos": np.cos(doy_ang),
    }


# ---------------------------------
# Main builder (from expected_and_mismatch output)
# ---------------------------------
@dataclass(frozen=True)
class RFFeatureSpec:
    """
    Config padrão p/ RF em FV (15-min).
    """
    dt_minutes: float = 15.0

    g_day_min: float = 50.0
    g_gate_wm2: float = 700.0

    # janelas (min)
    roll_windows_minutes: Tuple[float, ...] = (60.0, 180.0)

    include_time_features: bool = True
    include_roll: bool = True
    include_deltas: bool = True

    eps_w: float = 50.0


def build_rf_features(
    out_model: Dict[str, Any],
    *,
    times_utc: Optional[Sequence[Union[datetime, np.datetime64]]] = None,
    spec: Optional[RFFeatureSpec] = None,
) -> FeatureBuildResult:
    """
    Entrada: dict retornado por power_model.expected_and_mismatch()
    Saída: X (T x n_features) + feature_names + valid_mask.

    RF (sklearn) não aceita NaN => use `fit_imputer(...)` e `imputer.transform(X)`.
    """
    spec = spec or RFFeatureSpec()

    # infer T
    tc = _to_np_1d(out_model.get("tcell_c"))
    if tc.size == 0:
        g_try = _to_np_1d(out_model.get("g_poa_used", out_model.get("g_poa")))
        T = int(g_try.size)
        if T == 0:
            raise ValueError("build_rf_features: out_model não contém séries (T=0).")
        tc = _align_len(out_model.get("tcell_c"), T)
    else:
        T = int(tc.size)

    valid = _align_bool(out_model.get("valid"), T, fill=False)

    # dt (preferir meta)
    meta_in = out_model.get("meta", {}) if isinstance(out_model.get("meta", {}), dict) else {}
    dt = float(meta_in.get("dt_minutes", spec.dt_minutes) or spec.dt_minutes)
    if not np.isfinite(dt) or dt <= 0:
        dt = float(spec.dt_minutes)

    # séries base
    g = _align_len(out_model.get("g_poa_used", out_model.get("g_poa")), T)
    pac_real = _align_len(out_model.get("pac_real_w"), T)
    pac_exp = _align_len(out_model.get("pac_expected_w"), T)
    pdc_exp = _align_len(out_model.get("pdc_expected_w"), T)
    eta_inv = _align_len(out_model.get("eta_inv"), T)

    mismatch_rel = _align_len(out_model.get("mismatch_rel"), T)
    mismatch_abs = _align_len(out_model.get("mismatch_abs_w"), T)

    pr_real = _align_len(out_model.get("pr_real_inst"), T)
    pr_model = _align_len(out_model.get("pr_model_inst"), T)
    pu_real = _align_len(out_model.get("p_ac_pu_real"), T)
    pu_model = _align_len(out_model.get("p_ac_pu_model"), T)

    v_ratio = _align_len(out_model.get("v_ratio"), T)
    i_ratio = _align_len(out_model.get("i_ratio"), T)

    g_mean_60 = _align_len(out_model.get("g_mean_60m"), T)
    g_std_60 = _align_len(out_model.get("g_std_60m"), T)
    g_cv_60 = _align_len(out_model.get("g_cv_60m"), T)

    sky_stable = _align_bool(out_model.get("sky_stable_mask"), T, fill=False).astype(float)
    csi = _align_len(out_model.get("csi"), T)

    # correção: seu power_model expõe v_ac_real_v (não "v_ac_real_v"? ele cria out["v_ac_real_v"])
    vac = _align_len(out_model.get("v_ac_real_v", out_model.get("v_ac_real_v")), T)

    # máscaras úteis
    day = np.isfinite(g) & (g >= float(spec.g_day_min))
    gate700 = np.isfinite(g) & (g >= float(spec.g_gate_wm2))

    # máscara recomendada p/ ML (consistente com power_model.feature_extraction valid_ml)
    valid_mask = valid & np.isfinite(g) & np.isfinite(tc)

    # derivados
    pac_ratio = _safe_div(pac_real, pac_exp, eps=float(spec.eps_w))
    pdc_per_pac = _safe_div(pdc_exp, pac_exp, eps=float(spec.eps_w))
    mismatch_sign = np.where(np.isfinite(mismatch_rel), np.sign(mismatch_rel), np.nan)
    abs_mismatch_rel = np.abs(mismatch_rel)

    mismatch_x_g = mismatch_rel * g
    mismatch_x_sky = mismatch_rel * sky_stable
    pr_gap = pr_real - pr_model
    pu_gap = pu_real - pu_model

    series: Dict[str, np.ndarray] = {
        "g_poa": g,
        "tcell_c": tc,
        "pac_real_w": pac_real,
        "pac_expected_w": pac_exp,
        "pdc_expected_w": pdc_exp,
        "eta_inv": eta_inv,
        "mismatch_rel": mismatch_rel,
        "abs_mismatch_rel": abs_mismatch_rel,
        "mismatch_abs_w": mismatch_abs,
        "mismatch_sign": mismatch_sign,
        "pac_ratio": pac_ratio,
        "pdc_per_pac": pdc_per_pac,
        "pr_real_inst": pr_real,
        "pr_model_inst": pr_model,
        "pr_gap": pr_gap,
        "p_ac_pu_real": pu_real,
        "p_ac_pu_model": pu_model,
        "pu_gap": pu_gap,
        "v_ratio": v_ratio,
        "i_ratio": i_ratio,
        "g_mean_60m": g_mean_60,
        "g_std_60m": g_std_60,
        "g_cv_60m": g_cv_60,
        "sky_stable": sky_stable,
        "csi": csi,
        "v_ac_v": vac,
        "is_day": day.astype(float),
        "gate700": gate700.astype(float),
        "valid_model": valid.astype(float),
        "mismatch_x_g": mismatch_x_g,
        "mismatch_x_sky": mismatch_x_sky,
    }

    # rolling
    if spec.include_roll:
        for wmin in spec.roll_windows_minutes:
            w = int(max(1, round(float(wmin) / dt)))
            k = int(wmin)

            series[f"g_mu_{k}m"] = _rolling_nanmean(g, w)
            series[f"g_sd_{k}m"] = _rolling_nanstd(g, w)

            series[f"pac_mu_{k}m"] = _rolling_nanmean(pac_real, w)
            series[f"pac_sd_{k}m"] = _rolling_nanstd(pac_real, w)

            series[f"mis_mu_{k}m"] = _rolling_nanmean(mismatch_rel, w)
            series[f"mis_sd_{k}m"] = _rolling_nanstd(mismatch_rel, w)

            series[f"pr_mu_{k}m"] = _rolling_nanmean(pr_real, w)
            series[f"pr_sd_{k}m"] = _rolling_nanstd(pr_real, w)

            series[f"vr_mu_{k}m"] = _rolling_nanmean(v_ratio, w)
            series[f"ir_mu_{k}m"] = _rolling_nanmean(i_ratio, w)

    # deltas
    if spec.include_deltas:
        series["d_g"] = _diff(g, 1)
        series["d_pac"] = _diff(pac_real, 1)
        series["d_mis"] = _diff(mismatch_rel, 1)
        series["d_pr"] = _diff(pr_real, 1)
        series["d_vr"] = _diff(v_ratio, 1)
        series["d_ir"] = _diff(i_ratio, 1)
        series["d_csi"] = _diff(csi, 1)

    # tempo
    if spec.include_time_features:
        series.update(_time_cyc_features(times_utc, T))

    # -------- feature order --------
    feature_names: List[str] = [
        "valid_model",
        "is_day",
        "gate700",
        "g_poa",
        "g_mean_60m",
        "g_std_60m",
        "g_cv_60m",
        "csi",
        "tcell_c",
        "sky_stable",
        "v_ac_v",
        "pac_real_w",
        "pac_expected_w",
        "pdc_expected_w",
        "eta_inv",
        "pac_ratio",
        "pdc_per_pac",
        "pr_real_inst",
        "pr_model_inst",
        "pr_gap",
        "p_ac_pu_real",
        "p_ac_pu_model",
        "pu_gap",
        "mismatch_rel",
        "abs_mismatch_rel",
        "mismatch_abs_w",
        "mismatch_sign",
        "v_ratio",
        "i_ratio",
        "mismatch_x_g",
        "mismatch_x_sky",
    ]

    if spec.include_roll:
        for wmin in spec.roll_windows_minutes:
            k = int(wmin)
            feature_names += [
                f"g_mu_{k}m",
                f"g_sd_{k}m",
                f"pac_mu_{k}m",
                f"pac_sd_{k}m",
                f"mis_mu_{k}m",
                f"mis_sd_{k}m",
                f"pr_mu_{k}m",
                f"pr_sd_{k}m",
                f"vr_mu_{k}m",
                f"ir_mu_{k}m",
            ]

    if spec.include_deltas:
        feature_names += ["d_g", "d_pac", "d_mis", "d_pr", "d_vr", "d_ir", "d_csi"]

    if spec.include_time_features:
        feature_names += ["tod_sin", "tod_cos", "doy_sin", "doy_cos"]

    # garante presença
    for fn in feature_names:
        if fn not in series:
            series[fn] = np.full(T, np.nan, dtype=float)

    X = np.column_stack([series[n] for n in feature_names]).astype(np.float32, copy=False)

    meta = {
        "T": int(T),
        "dt_minutes": float(dt),
        "g_day_min": float(spec.g_day_min),
        "g_gate_wm2": float(spec.g_gate_wm2),
        "roll_windows_minutes": tuple(float(x) for x in spec.roll_windows_minutes),
        "include_roll": bool(spec.include_roll),
        "include_deltas": bool(spec.include_deltas),
        "include_time_features": bool(spec.include_time_features),
        "source_meta": meta_in,
    }

    return FeatureBuildResult(
        X=X,
        feature_names=feature_names,
        valid_mask=valid_mask,
        series=series,
        meta=meta,
    )


# ---------------------------------
# Training helpers
# ---------------------------------
def fit_imputer(X_train: np.ndarray, feature_names: List[str], *, fill_if_all_nan: float = 0.0) -> RFImputer:
    """
    Ajusta mediana por feature ignorando NaN/inf.
    """
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2:
        raise ValueError("fit_imputer: X_train deve ser 2D.")

    X2 = X.copy()
    X2[~np.isfinite(X2)] = np.nan

    # np.nanmedian em colunas totalmente NaN -> NaN (ok; tratamos abaixo)
    med = np.nanmedian(X2, axis=0)
    med[~np.isfinite(med)] = float(fill_if_all_nan)

    return RFImputer(
        feature_names=list(feature_names),
        medians=med.astype(float),
        fill_if_all_nan=float(fill_if_all_nan),
    )


def select_valid_rows(
    X: np.ndarray,
    y: Optional[np.ndarray],
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Seleciona apenas linhas valid_mask==True.
    """
    vm = np.asarray(valid_mask, dtype=bool).reshape(-1)
    Xs = np.asarray(X)[vm]
    ys = None if y is None else np.asarray(y)[vm]
    return Xs, ys
