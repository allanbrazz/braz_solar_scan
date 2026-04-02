# core/services/power_model.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import warnings

ArrayLike = Union[np.ndarray, Sequence[float]]

# =========================
# Constantes
# =========================
K_B = 1.380649e-23          # Boltzmann (J/K)
Q_E = 1.602176634e-19       # carga do elétron (C)
EG_SI_EV = 1.121            # bandgap Si ~1.121 eV (aprox.)
EPS = 1e-12
G_SC = 1367.0               # constante solar (W/m²), p/ Erbs (aprox.)


# =========================
# Helpers numéricos
# =========================
def _to_np(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    if hasattr(x, "to_numpy"):  # pandas Series/Index
        try:
            return np.asarray(x.to_numpy(), dtype=float)
        except Exception:
            pass
    return np.asarray(x, dtype=float)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _float_or(v: Any, default: float) -> float:
    x = _safe_float(v)
    return float(default if x is None else x)


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _nanmean_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    if not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmean(x))


def _nanpercentile_safe(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    if not np.isfinite(x).any():
        return float("nan")
    try:
        return float(np.nanpercentile(x, q))
    except Exception:
        return float("nan")


def _clip_exp_arg(a: np.ndarray, max_arg: float = 80.0) -> np.ndarray:
    return np.clip(a, -max_arg, max_arg)


def _vt_cell(Tk: np.ndarray) -> np.ndarray:
    """Tensão térmica por célula: Vt = kT/q."""
    return (K_B * Tk) / Q_E


@lru_cache(maxsize=64)
def _vhat01(n_points: int) -> np.ndarray:
    """Grade normalizada [0..1]."""
    n = int(max(30, n_points))
    return np.linspace(0.0, 1.0, n, dtype=float)


@lru_cache(maxsize=64)
def _vhat_eps(n_points: int) -> np.ndarray:
    """Grade (0..1) evitando endpoints exatos."""
    n = int(max(30, n_points))
    e = 1e-4
    return np.linspace(e, 1.0 - e, n, dtype=float)


def _rolling_nanmean(x: np.ndarray, w: int) -> np.ndarray:
    """Média móvel trailing ignorando NaN (NumPy puro)."""
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
    """Std móvel trailing ignorando NaN (ddof=0)."""
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


def irradiance_stability(
    g: ArrayLike,
    *,
    dt_minutes: float = 15.0,
    window_minutes: float = 60.0,
    eps_mean: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Estatísticas móveis da irradiância (sempre retorna g_mean/g_std/g_cv)."""
    G = _to_np(g)
    n = G.size

    out = {
        "g_mean": np.full(n, np.nan, dtype=float),
        "g_std": np.full(n, np.nan, dtype=float),
        "g_cv": np.full(n, np.nan, dtype=float),
    }
    if n == 0:
        return out

    dt = float(dt_minutes) if dt_minutes and dt_minutes > 0 else 15.0
    w = int(max(1, round(float(window_minutes) / dt)))

    mean = _rolling_nanmean(G, w)
    std = _rolling_nanstd(G, w)
    cv = std / np.maximum(mean, float(eps_mean))

    out["g_mean"] = mean
    out["g_std"] = std
    out["g_cv"] = cv
    return out


def clear_sky_index(g: np.ndarray, g_clear: Optional[np.ndarray], *, eps: float = 1.0) -> np.ndarray:
    """CSI = G / G_clear (opcional)."""
    g = np.asarray(g, dtype=float)
    if g_clear is None:
        return np.full_like(g, np.nan, dtype=float)

    gc = np.asarray(g_clear, dtype=float)
    if gc.size != g.size:
        return np.full_like(g, np.nan, dtype=float)

    den = np.maximum(gc, float(eps))
    csi = g / den
    csi = np.where(np.isfinite(csi), np.clip(csi, 0.0, 2.0), np.nan)
    return csi


# =========================
# Dataclasses do modelo
# =========================
@dataclass(frozen=True)
class ModuleOneDiode:
    # STC
    isc_n: float
    voc_n: float
    vmp_n: float
    imp_n: float
    ns: int

    # Coefs (V/°C e A/°C)
    kv_v_per_c: float
    ki_a_per_c: float

    # Parâmetros elétricos já extraídos (DB)
    rs_ohm: float
    rp_ohm: float
    a: float = 1.3

    # Referências
    gn: float = 1000.0
    tn_c: float = 25.0
    eg_ev: float = EG_SI_EV


@dataclass(frozen=True)
class PlantModel:
    n_modules_total: int

    # Topologia (crítica para Vdc/Idc esperados)
    strings_count: Optional[int] = None
    modules_per_string: Optional[int] = None

    k_sys: float = 0.900
    pac_rated_w: Optional[float] = None
    noct_c: float = 45.0

    # --- Geometria / transposição (conv.: azim 0=N, 90=E, 180=S, 270=W) ---
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    azimuth_deg: Optional[float] = None
    albedo: float = 0.20

    # --- Inversor: eficiência variável ---
    inv_model: str = "constant"  # "constant" | "load_curve"
    inv_eff: float = 1.0         # usado quando inv_model="constant"
    inv_eta_max: float = 0.985   # usado quando inv_model="load_curve"
    inv_eta_min: float = 0.92    # usado quando inv_model="load_curve"
    inv_alpha: float = 6.0       # controla subida
    inv_pso_w: float = 20.0      # perdas em vazio (W)
    inv_pdc_nom_w: Optional[float] = None  # base do load (W); se None, tenta inferir


# =========================
# Temperatura de célula (NOCT)
# =========================
def tcell_noct(g_poa: ArrayLike, tamb_c: ArrayLike, noct_c: float = 45.0) -> np.ndarray:
    G = _to_np(g_poa)
    Ta = _to_np(tamb_c)

    if G.size == 0 and Ta.size == 0:
        return np.array([], dtype=float)

    n = int(max(G.size, Ta.size))
    if G.size == 0:
        G = np.full(n, np.nan, dtype=float)
    if Ta.size == 0:
        Ta = np.full(n, np.nan, dtype=float)
    if G.size != Ta.size:
        raise ValueError(f"tcell_noct: g_poa e tamb_c tamanhos diferentes ({G.size} vs {Ta.size}).")

    G = np.clip(G, 0.0, None)
    Tc = Ta + (G / 800.0) * (noct_c - 20.0)
    Tc = np.maximum(Tc, Ta)
    return np.clip(Tc, -30.0, 95.0)


# =========================
# Datetime handling robusto
# =========================
def _unwrap_singleton(x: Any) -> Any:
    """Desembrulha casos como [DatetimeIndex] ou array size=1 contendo um Index."""
    try:
        while True:
            if isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
                continue
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x.reshape(-1)[0]
                continue
            break
    except Exception:
        pass
    return x


def _to_datetime64ns(times: Any) -> np.ndarray:
    """
    Retorna np.datetime64[ns] (UTC implícito). Se houver NaT, lança ValueError.
    Corrige casos com pandas.DatetimeIndex / Timestamp e listas embrulhadas.
    """
    times = _unwrap_singleton(times)

    if times is None:
        return np.array([], dtype="datetime64[ns]")

    # pandas path (robusto para tz-aware)
    try:
        import pandas as pd  # type: ignore
        if isinstance(times, (pd.DatetimeIndex, pd.Index, pd.Series)):
            tt = pd.to_datetime(times)
            # se tz-aware, converte para UTC; se naive, assume UTC (condizente com times_utc)
            if getattr(tt, "tz", None) is not None:
                tt = tt.tz_convert("UTC")
                # remove tz mantendo UTC
                tt = tt.tz_localize(None)
            out = tt.to_numpy(dtype="datetime64[ns]")
            if np.isnat(out).any():
                raise ValueError("times_utc contém NaT")
            return out

        if isinstance(times, pd.Timestamp):
            ts = times
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            out = np.array([np.datetime64(ts.to_datetime64())], dtype="datetime64[ns]")
            if np.isnat(out).any():
                raise ValueError("times_utc contém NaT")
            return out
    except Exception:
        pass

    # numpy datetime64 path
    t = np.asarray(times)
    if t.size == 0:
        return np.array([], dtype="datetime64[ns]")

    if np.issubdtype(t.dtype, np.datetime64):
        out = t.astype("datetime64[ns]")
        if np.isnat(out).any():
            raise ValueError("times_utc contém NaT")
        return out

    # objeto -> tentativa de conversão elemento-a-elemento
    out = np.empty(t.size, dtype="datetime64[ns]")
    flat = t.reshape(-1)

    for i, v in enumerate(flat):
        v = _unwrap_singleton(v)

        if v is None:
            out[i] = np.datetime64("NaT")
            continue

        # pandas Timestamp
        try:
            import pandas as pd  # type: ignore
            if isinstance(v, pd.Timestamp):
                ts = v
                if ts.tz is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                out[i] = np.datetime64(ts.to_datetime64()).astype("datetime64[ns]")
                continue

            if isinstance(v, (pd.DatetimeIndex, pd.Index, pd.Series)):
                # se caiu aqui, era um "Index dentro do array": tenta extrair 1º
                vv = pd.to_datetime(v)
                if vv.size > 0:
                    ts0 = vv[0]
                    if getattr(ts0, "tz", None) is not None:
                        ts0 = ts0.tz_convert("UTC").tz_localize(None)
                    out[i] = np.datetime64(ts0.to_datetime64()).astype("datetime64[ns]")
                    continue
        except Exception:
            pass

        # datetime python
        try:
            import datetime as pydt
            if isinstance(v, pydt.datetime):
                if v.tzinfo is not None:
                    v = v.astimezone(pydt.timezone.utc).replace(tzinfo=None)
                out[i] = np.datetime64(v).astype("datetime64[ns]")
                continue
        except Exception:
            pass

        # string / numpy datetime-like
        try:
            out[i] = np.datetime64(v).astype("datetime64[ns]")
        except Exception:
            out[i] = np.datetime64("NaT")

    if np.isnat(out).any():
        raise ValueError("times_utc contém valores inválidos/NaT após conversão")
    return out.reshape(t.shape)


def _shift_times_minutes(t: np.ndarray, shift_minutes: float) -> np.ndarray:
    if t.size == 0:
        return t
    m = float(shift_minutes)
    if abs(m) < 1e-12:
        return t
    sec = int(round(m * 60.0))
    return t + np.timedelta64(sec, "s")


def _infer_dt_minutes_from_times(t: np.ndarray, fallback: float = 15.0) -> float:
    """Estimativa de dt em minutos a partir de times_utc (mediana das diffs)."""
    if t.size < 2:
        return float(fallback)
    dt_s = np.diff(t.astype("datetime64[s]").astype("int64")).astype(float)
    dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
    if dt_s.size == 0:
        return float(fallback)
    return float(np.nanmedian(dt_s) / 60.0)


def _shift_series_by_minutes(values: np.ndarray, times_utc: np.ndarray, shift_minutes: float) -> np.ndarray:
    """
    Shifta a série no eixo do tempo preservando timestamps.
    Convenção: shift_minutes > 0 => ATRASA a curva (move para a direita).
      new(t) = old(t - shift)
    Interp linear no tempo (apenas pontos finitos).
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(times_utc, dtype="datetime64[ns]")
    if y.size == 0 or t.size == 0 or y.size != t.size:
        return y

    m = float(shift_minutes)
    if abs(m) < 1e-12:
        return y

    # eixo tempo em segundos (relativo)
    tn = t.astype("datetime64[ns]").astype("int64").astype(np.float64)
    tn0 = float(tn[0])
    x = (tn - tn0) / 1e9  # s

    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 3:
        return y

    xp = x[mask]
    fp = y[mask]
    xq = x - m * 60.0

    # np.interp extrapola pelos extremos; vamos NAN fora do suporte
    yq = np.interp(xq, xp, fp)
    yq = np.asarray(yq, dtype=float)

    xmin = float(xp.min())
    xmax = float(xp.max())
    oob = (xq < xmin) | (xq > xmax) | (~np.isfinite(xq))
    yq[oob] = np.nan
    return yq


def _best_lag_steps_xcorr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag_steps: int,
    min_samples: int = 40,
) -> int:
    """
    Retorna lag_steps que maximiza correlação entre x e y shiftado.
    Convenção: lag_steps > 0 => ATRASA y (move y para a direita).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < max(min_samples, 10):
        return 0

    # normaliza (ignora NaN)
    def _norm(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a)
        if m.sum() < 3:
            return a
        mu = float(np.nanmean(a[m]))
        sd = float(np.nanstd(a[m]))
        if sd < 1e-9:
            sd = 1.0
        return (a - mu) / sd

    xn = _norm(x[:n])
    yn = _norm(y[:n])

    best_s = 0
    best_r = -1e9

    L = int(max(0, max_lag_steps))
    for s in range(-L, L + 1):
        if s == 0:
            xa = xn
            ya = yn
        elif s > 0:
            # y atrasado: compara x[s:] com y[:-s]
            xa = xn[s:]
            ya = yn[:-s]
        else:
            # y adiantado: compara x[:s] com y[-s:]
            xa = xn[:s]
            ya = yn[-s:]

        m = np.isfinite(xa) & np.isfinite(ya)
        if m.sum() < min_samples:
            continue

        # correlação via dot (já normalizado)
        r = float(np.nanmean(xa[m] * ya[m]))
        if r > best_r:
            best_r = r
            best_s = s

    return int(best_s)


# =========================
# Solar position (NOAA simplificado) - vetorizado
# =========================
def solar_position_noaa_utc(
    times_utc: Any,
    *,
    lat_deg: float,
    lon_deg: float,
) -> Dict[str, np.ndarray]:
    """Retorna zenith_deg e azimuth_deg (0=N, 90=E) para timestamps UTC."""
    t = _to_datetime64ns(times_utc)
    n = t.size
    if n == 0:
        return {"zenith_deg": np.array([], dtype=float), "azimuth_deg": np.array([], dtype=float)}

    date = t.astype("datetime64[D]")
    doy = (date - date.astype("datetime64[Y]")).astype(int) + 1

    seconds = (t - date).astype("timedelta64[s]").astype(float)
    hours = seconds / 3600.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hours - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # NOAA: tempo solar verdadeiro depende de longitude (graus, leste positivo)
    time_offset = eqtime + 4.0 * float(lon_deg)
    tst = (hours * 60.0 + time_offset) % 1440.0

    ha = np.where(tst / 4.0 < 0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    ha_rad = np.deg2rad(ha)

    lat = np.deg2rad(float(lat_deg))

    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha_rad)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zen = np.arccos(cos_zen)

    sin_az = -np.sin(ha_rad) * np.cos(decl) / np.maximum(np.sin(zen), EPS)
    cos_az = (np.sin(decl) - np.sin(lat) * np.cos(zen)) / (np.cos(lat) * np.maximum(np.sin(zen), EPS))
    az = np.arctan2(sin_az, cos_az)
    az_deg = (np.rad2deg(az) + 360.0) % 360.0
    zen_deg = np.rad2deg(zen)

    return {"zenith_deg": zen_deg, "azimuth_deg": az_deg}


def _extraterrestrial_horizontal_irradiance(
    times_utc: np.ndarray,
    zenith_deg: np.ndarray,
) -> np.ndarray:
    t = times_utc.astype("datetime64[ns]")
    date = t.astype("datetime64[D]")
    doy = (date - date.astype("datetime64[Y]")).astype(int) + 1

    E0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    cosz = np.cos(np.deg2rad(zenith_deg))
    g0h = G_SC * E0 * np.clip(cosz, 0.0, None)
    return g0h


def _erbs_diffuse_fraction_kt(kt: np.ndarray) -> np.ndarray:
    kt = np.asarray(kt, dtype=float)
    kd = np.full_like(kt, np.nan, dtype=float)

    m1 = kt <= 0.22
    m2 = (kt > 0.22) & (kt <= 0.80)
    m3 = kt > 0.80

    kd[m1] = 1.0 - 0.09 * kt[m1]
    k = kt[m2]
    kd[m2] = 0.9511 - 0.1604 * k + 4.388 * k**2 - 16.638 * k**3 + 12.336 * k**4
    kd[m3] = 0.165

    return np.clip(kd, 0.0, 1.0)


def transpose_ghi_to_poa_isotropic(
    *,
    ghi: ArrayLike,
    times_utc: Any,
    lat_deg: float,
    lon_deg: float,
    tilt_deg: float,
    azimuth_deg: float,
    albedo: float = 0.20,
    dhi: Optional[ArrayLike] = None,
    dni: Optional[ArrayLike] = None,
    times_shift_minutes: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Transposição isotrópica (Liu-Jordan): POA = B_poa + D_poa + G_ref"""
    GHI = _to_np(ghi)
    t0 = _to_datetime64ns(times_utc)
    if t0.size == 0 and GHI.size == 0:
        return {
            "g_poa": np.array([], dtype=float),
            "ghi": np.array([], dtype=float),
            "dhi": np.array([], dtype=float),
            "dni": np.array([], dtype=float),
            "zenith_deg": np.array([], dtype=float),
            "azimuth_deg": np.array([], dtype=float),
            "cos_inc": np.array([], dtype=float),
            "b_poa": np.array([], dtype=float),
            "d_poa": np.array([], dtype=float),
            "g_ref": np.array([], dtype=float),
        }

    if t0.size != GHI.size:
        raise ValueError(f"times_utc e ghi devem ter mesmo tamanho. got {t0.size} vs {GHI.size}")

    # aplica shift de tempo (para corrigir rótulo de janela 15-min vs instante)
    t = _shift_times_minutes(t0, float(times_shift_minutes))

    GHI = np.clip(GHI, 0.0, None)

    sp = solar_position_noaa_utc(t, lat_deg=float(lat_deg), lon_deg=float(lon_deg))
    zen = sp["zenith_deg"]
    az = sp["azimuth_deg"]

    cosz = np.cos(np.deg2rad(zen))
    sun_up = cosz > 0.0

    beta = np.deg2rad(float(tilt_deg))
    az_surf = np.deg2rad(float(azimuth_deg))
    az_sun = np.deg2rad(az)
    sinz = np.sin(np.deg2rad(zen))

    cos_inc = sinz * np.sin(beta) * np.cos(az_sun - az_surf) + cosz * np.cos(beta)
    cos_inc = np.clip(cos_inc, 0.0, None)

    DHI = _to_np(dhi) if dhi is not None else None
    DNI = _to_np(dni) if dni is not None else None

    if DHI is None or DHI.size != GHI.size:
        g0h = _extraterrestrial_horizontal_irradiance(t, zen)
        kt = GHI / np.maximum(g0h, 1.0)
        kt = np.clip(kt, 0.0, 2.0)
        kd = _erbs_diffuse_fraction_kt(kt)
        DHI = kd * GHI

    if DNI is None or DNI.size != GHI.size:
        DNI = (GHI - DHI) / np.maximum(cosz, 0.065)
        DNI = np.clip(DNI, 0.0, None)

    B_poa = DNI * cos_inc
    D_poa = DHI * (1.0 + np.cos(beta)) / 2.0
    G_ref = GHI * float(albedo) * (1.0 - np.cos(beta)) / 2.0
    G_poa = B_poa + D_poa + G_ref

    G_poa = np.where(sun_up, G_poa, 0.0)
    B_poa = np.where(sun_up, B_poa, 0.0)
    D_poa = np.where(sun_up, D_poa, 0.0)
    G_ref = np.where(sun_up, G_ref, 0.0)

    return {
        "g_poa": np.clip(G_poa, 0.0, None),
        "ghi": GHI,
        "dhi": np.clip(DHI, 0.0, None),
        "dni": np.clip(DNI, 0.0, None),
        "zenith_deg": zen,
        "azimuth_deg": az,
        "cos_inc": cos_inc,
        "b_poa": B_poa,
        "d_poa": D_poa,
        "g_ref": G_ref,
    }


# =========================
# 1-diodo (Villalva-like)
# =========================
@dataclass(frozen=True)
class RefParams:
    iph_n: float
    i0_n: float


@lru_cache(maxsize=256)
def ref_params_stc(module: ModuleOneDiode) -> RefParams:
    """Fecha (I0_n, Iph_n) em STC com Isc e Voc."""
    TnK = module.tn_c + 273.15
    Vt_mod = float(_vt_cell(np.array([TnK]))[0]) * float(module.ns)
    aVt = float(module.a) * Vt_mod

    Isc = float(module.isc_n)
    Voc = float(module.voc_n)
    Rs = float(module.rs_ohm)
    Rp = float(module.rp_ohm)

    exp_oc = np.exp(np.clip(Voc / max(aVt, EPS), -80, 80))
    exp_sc = np.exp(np.clip((Isc * Rs) / max(aVt, EPS), -80, 80))

    denom = (exp_oc - exp_sc)
    num = (Isc - (Voc / max(Rp, EPS)) + (Isc * Rs) / max(Rp, EPS))

    i0 = float(num / max(denom, EPS))
    i0 = max(i0, 1e-16)

    iph = i0 * (float(exp_oc) - 1.0) + Voc / max(Rp, EPS)
    iph = max(float(iph), 0.0)

    return RefParams(iph_n=iph, i0_n=i0)


def i0_temp(module: ModuleOneDiode, tc_c: np.ndarray) -> np.ndarray:
    """Ajuste térmico de I0."""
    T = tc_c + 273.15
    Tn = module.tn_c + 273.15
    EgJ = float(module.eg_ev) * Q_E

    i0n = ref_params_stc(module).i0_n
    expo = (EgJ / (float(module.a) * K_B)) * ((1.0 / Tn) - (1.0 / T))
    return np.clip(i0n * (T / Tn) ** 3 * np.exp(_clip_exp_arg(expo)), 1e-16, None)


def iph_irr_temp(module: ModuleOneDiode, g: np.ndarray, tc_c: np.ndarray) -> np.ndarray:
    """Iph(G,T) ≈ (Iph_n + Ki*(T-Tn))*(G/Gn)"""
    iph_n = ref_params_stc(module).iph_n
    g = np.asarray(g, dtype=float)
    g = np.clip(g, 0.0, None)
    return np.clip(
        (iph_n + float(module.ki_a_per_c) * (tc_c - float(module.tn_c))) * (g / float(module.gn)),
        0.0,
        None,
    )


def rp_irr(module: ModuleOneDiode, g: np.ndarray) -> np.ndarray:
    """Rp cresce em baixa irradiância (aprox.)."""
    g = np.asarray(g, dtype=float)
    g = np.clip(g, 0.0, None)
    return np.clip(float(module.rp_ohm) * (float(module.gn) / np.maximum(g, 50.0)), 0.1, 1e8)


def voc_guess(module: ModuleOneDiode, tc_c: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Chute Vetorizado de Voc."""
    tc_c = np.asarray(tc_c, dtype=float)
    g = np.asarray(g, dtype=float)
    Tk = tc_c + 273.15
    Vt_mod = _vt_cell(Tk) * float(module.ns)
    aVt = float(module.a) * Vt_mod

    base = float(module.voc_n) + float(module.kv_v_per_c) * (tc_c - float(module.tn_c))
    ratio = np.maximum(g, 1.0) / float(module.gn)
    return np.maximum(base + aVt * np.log(ratio), 0.1)


def voc_newton_vec(iph: np.ndarray, i0: np.ndarray, rp: np.ndarray, aVt: np.ndarray, guess: np.ndarray) -> np.ndarray:
    """Resolve Voc (I=0) vetorizado."""
    V = np.asarray(guess, dtype=float).copy()
    iph = np.asarray(iph, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    rp = np.asarray(rp, dtype=float)
    aVt = np.asarray(aVt, dtype=float)

    V = np.maximum(V, 0.1)
    mask = np.isfinite(V) & np.isfinite(iph) & np.isfinite(i0) & np.isfinite(rp) & np.isfinite(aVt) & (rp > 0)

    for _ in range(25):
        if not np.any(mask):
            break
        Vm = V[mask]
        iph_m = iph[mask]
        i0_m = i0[mask]
        rp_m = rp[mask]
        aVt_m = aVt[mask]

        arg = np.clip(Vm / np.maximum(aVt_m, EPS), -80.0, 80.0)
        e = np.exp(arg)
        f = iph_m - i0_m * (e - 1.0) - Vm / np.maximum(rp_m, EPS)
        df = -i0_m * e * (1.0 / np.maximum(aVt_m, EPS)) - 1.0 / np.maximum(rp_m, EPS)

        step = f / np.where(np.abs(df) > EPS, df, -EPS)
        Vn = np.maximum(Vm - step, 0.0)

        conv = np.abs(Vn - Vm) < 1e-7
        V[mask] = Vn
        if np.any(conv):
            m_idx = np.where(mask)[0]
            mask[m_idx[conv]] = False

    return V


def iv_current_mat(
    Vmat: np.ndarray,
    iph: np.ndarray,
    i0: np.ndarray,
    rs: float,
    rp: np.ndarray,
    aVt: np.ndarray,
    *,
    max_iter: int = 30,
) -> np.ndarray:
    """Resolve I(V) para uma matriz Vmat (n,m)."""
    Vmat = np.asarray(Vmat, dtype=float)
    iph = np.asarray(iph, dtype=float)
    i0 = np.asarray(i0, dtype=float)
    rp = np.asarray(rp, dtype=float)
    aVt = np.asarray(aVt, dtype=float)

    if Vmat.ndim != 2:
        raise ValueError("Vmat deve ser 2D (n, m).")
    n, _ = Vmat.shape
    if iph.size != n:
        raise ValueError(f"iv_current_mat: iph size {iph.size} != Vmat n {n}")

    I = np.clip(iph[:, None] * 0.90, 0.0, None)

    rs = float(rs)
    inv_rp = 1.0 / np.maximum(rp, EPS)
    inv_aVt = 1.0 / np.maximum(aVt, EPS)

    for _ in range(int(max_iter)):
        arg = (Vmat + I * rs) * inv_aVt[:, None]
        e = np.exp(_clip_exp_arg(arg))

        f = I - iph[:, None] + i0[:, None] * (e - 1.0) + (Vmat + I * rs) * inv_rp[:, None]
        df = 1.0 + i0[:, None] * e * (rs * inv_aVt[:, None]) + rs * inv_rp[:, None]

        step = f / np.maximum(df, EPS)
        In = np.clip(I - step, 0.0, np.maximum(iph[:, None], 0.0))

        if np.nanmax(np.abs(In - I)) < 1e-7:
            return In
        I = In

    return I


def pmp_module_vec(
    iph: np.ndarray,
    i0: np.ndarray,
    rs: float,
    rp: np.ndarray,
    aVt: np.ndarray,
    voc_g: np.ndarray,
    *,
    n_points: int = 60,
) -> Dict[str, np.ndarray]:
    """Retorna Potência, Tensão e Corrente no MPP (por módulo)."""
    voc = np.maximum(voc_newton_vec(iph, i0, rp, aVt, voc_g), 0.1)

    vhat = _vhat01(int(n_points))
    Vmat = voc[:, None] * vhat[None, :]
    Imat = iv_current_mat(Vmat, iph, i0, float(rs), rp, aVt, max_iter=28)

    Pmat = Vmat * Imat
    idx_max = np.argmax(Pmat, axis=1)
    rows = np.arange(Pmat.shape[0])

    pmp = Pmat[rows, idx_max]
    vmp = Vmat[rows, idx_max]
    imp = Imat[rows, idx_max]

    return {"pmp": pmp, "vmp": vmp, "imp": imp}


# =========================
# Inversor - eficiência variável
# =========================
def inverter_efficiency(pdc_w: np.ndarray, plant: PlantModel) -> np.ndarray:
    """
    Retorna eta_inv(t) (0..1).

    - Trata pdc NaN como 0 para não propagar NaN para eta.
    - Zera eta quando pdc_net == 0 (inversor "off").
    """
    pdc = np.asarray(pdc_w, dtype=float)
    pdc0 = np.where(np.isfinite(pdc), pdc, 0.0)

    model = (getattr(plant, "inv_model", "constant") or "constant").lower().strip()
    if model == "constant":
        e = float(getattr(plant, "inv_eff", 1.0) or 1.0)
        eta = np.clip(np.full_like(pdc0, e, dtype=float), 0.0, 1.0)
        eta = np.where(pdc0 <= 0.0, 0.0, eta)
        return eta

    eta_max = float(getattr(plant, "inv_eta_max", 0.985) or 0.985)
    eta_min = float(getattr(plant, "inv_eta_min", 0.92) or 0.92)
    alpha = float(getattr(plant, "inv_alpha", 6.0) or 6.0)
    pso = float(getattr(plant, "inv_pso_w", 20.0) or 0.0)

    pdc_nom = getattr(plant, "inv_pdc_nom_w", None)
    if pdc_nom is None:
        if plant.pac_rated_w is not None and eta_max > 0:
            pdc_nom = float(plant.pac_rated_w) / max(eta_max, 0.80)
        else:
            pdc_net_tmp = np.clip(pdc0 - pso, 0.0, None)
            p = _nanpercentile_safe(np.where(np.isfinite(pdc_net_tmp), pdc_net_tmp, np.nan), 99)
            pdc_nom = 1.0 if (not np.isfinite(p)) or p <= 0 else float(p)

    pdc_nom = float(pdc_nom)
    if (not np.isfinite(pdc_nom)) or pdc_nom <= 0:
        pdc_nom = 1.0

    pdc_net = np.clip(pdc0 - pso, 0.0, None)
    load = pdc_net / max(pdc_nom, 1.0)

    eta_curve = eta_min + (eta_max - eta_min) * (1.0 - np.exp(-alpha * load))
    eta_curve = np.clip(eta_curve, 0.0, eta_max)
    eta_curve = np.where(pdc_net <= 0.0, 0.0, eta_curve)
    return eta_curve


# =========================
# RCA heurística (vetorizada)
# =========================
def classify_root_cause_vec(
    *,
    valid: np.ndarray,
    mismatch_rel: np.ndarray,
    v_ratio: np.ndarray,
    i_ratio: np.ndarray,
    sky_stable_mask: np.ndarray,
    strings_count: Optional[int] = None,
    thr_ok: float = 0.05,
    thr_ratio_band: float = 0.05,
    thr_drop_i: float = 0.90,
    thr_drop_v: float = 0.90,
) -> np.ndarray:
    """Heurística simples. Retorna array de strings (labels)."""
    n = valid.size
    out = np.full(n, "invalid", dtype=object)

    v1 = np.isfinite(v_ratio)
    i1 = np.isfinite(i_ratio)
    m1 = np.isfinite(mismatch_rel)

    ok = valid & m1
    out[ok] = "unknown"

    normal = ok & (mismatch_rel >= -thr_ok)
    out[normal] = "normal"

    both = ok & v1 & i1

    v_near1 = both & (np.abs(v_ratio - 1.0) <= thr_ratio_band)
    i_near1 = both & (np.abs(i_ratio - 1.0) <= thr_ratio_band)

    if strings_count is not None and int(strings_count) >= 2:
        target = (int(strings_count) - 1) / float(int(strings_count))
        string_drop = both & v_near1 & (np.abs(i_ratio - target) <= 0.06) & (mismatch_rel < -thr_ok)
        out[string_drop] = "string_disconnected"

    soiling = both & v_near1 & (i_ratio < thr_drop_i) & sky_stable_mask & (mismatch_rel < -thr_ok)
    out[soiling] = "soiling"

    short_bypass = both & i_near1 & (v_ratio < thr_drop_v) & (mismatch_rel < -thr_ok)
    out[short_bypass] = "short_or_bypass"

    shading = (
        both
        & (v_ratio < (1.0 - thr_ratio_band))
        & (i_ratio < (1.0 - thr_ratio_band))
        & (~sky_stable_mask)
        & (mismatch_rel < -thr_ok)
    )
    out[shading] = "partial_shading"

    degr = (
        both
        & (v_ratio < (1.0 - thr_ratio_band))
        & (i_ratio < (1.0 - thr_ratio_band))
        & sky_stable_mask
        & (mismatch_rel < -thr_ok)
    )
    out[degr] = "degradation_like"

    return out


def feature_extraction(
    out_model: Dict[str, np.ndarray],
    *,
    v_ac_real_v: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """Padroniza o vetor de features p/ ML."""
    g = out_model.get("g_poa_used", None)
    if g is None:
        g = out_model.get("g_poa", None)
    g = np.asarray(g, dtype=float) if g is not None else np.full_like(out_model["tcell_c"], np.nan, dtype=float)

    tc = np.asarray(out_model.get("tcell_c", np.array([], dtype=float)), dtype=float)
    mismatch = np.asarray(out_model.get("mismatch_rel", np.full_like(tc, np.nan, dtype=float)), dtype=float)

    v_ratio = out_model.get("v_ratio", None)
    i_ratio = out_model.get("i_ratio", None)
    v_ratio = np.asarray(v_ratio, dtype=float) if v_ratio is not None else np.full_like(tc, np.nan, dtype=float)
    i_ratio = np.asarray(i_ratio, dtype=float) if i_ratio is not None else np.full_like(tc, np.nan, dtype=float)

    gcv = out_model.get("g_cv_60m", None)
    gcv = np.asarray(gcv, dtype=float) if gcv is not None else np.full_like(tc, np.nan, dtype=float)

    sky = out_model.get("sky_stable_mask", None)
    sky = np.asarray(sky, dtype=bool) if sky is not None else np.zeros_like(tc, dtype=bool)

    csi = out_model.get("csi", None)
    csi = np.asarray(csi, dtype=float) if csi is not None else np.full_like(tc, np.nan, dtype=float)

    vac = _to_np(v_ac_real_v) if v_ac_real_v is not None else np.full_like(tc, np.nan, dtype=float)
    if vac.size not in (0, tc.size):
        vac = np.full_like(tc, np.nan, dtype=float)

    valid = np.asarray(out_model.get("valid", np.zeros_like(tc, dtype=bool)), dtype=bool)
    valid_ml = valid & np.isfinite(g) & np.isfinite(tc)

    return {
        "g_poa": g,
        "tcell_c": tc,
        "mismatch_rel": mismatch,
        "v_ratio": v_ratio,
        "i_ratio": i_ratio,
        "g_cv_60m": gcv,
        "sky_stable_mask": sky.astype(float),
        "csi": csi,
        "v_ac_v": vac,
        "valid_ml": valid_ml,
    }


# =========================
# Modelo esperado + mismatch + normalizações
# =========================
def expected_and_mismatch(
    g_poa: Optional[ArrayLike],
    tamb_c: ArrayLike,
    pac_real_w: Optional[ArrayLike],
    module: ModuleOneDiode,
    plant: PlantModel,
    *,
    v_dc_real_v: Optional[ArrayLike] = None,
    i_dc_real_a: Optional[ArrayLike] = None,
    v_ac_real_v: Optional[ArrayLike] = None,
    ghi: Optional[ArrayLike] = None,
    dhi: Optional[ArrayLike] = None,
    dni: Optional[ArrayLike] = None,
    times_utc: Optional[Any] = None,
    use_transposition_if_needed: bool = True,
    g_clear: Optional[ArrayLike] = None,
    g_clear_sky: Optional[ArrayLike] = None,
    compute_norm: bool = True,
    compute_rca: bool = True,
    g_min_valid: float = 0.0,
    n_points: int = 60,
    eps_w: float = 50.0,
    dt_minutes: float = 15.0,
    window_minutes: float = 60.0,
    cv_max_stable: float = 0.20,
    force_zero_when_invalid: bool = True,
    # ---- CORREÇÃO de desencontro temporal (principal) ----
    meteo_time_shift_minutes: float = 0.0,          # ajuste manual (ex.: +7.5 para dados 15-min "no início da janela")
    auto_time_shift: bool = True,                   # estima lag automaticamente (pac_real vs G) e corrige
    max_auto_shift_minutes: float = 90.0,           # limite do auto-shift
    auto_shift_smooth_minutes: float = 30.0,         # suavização antes de correlacionar
) -> Dict[str, np.ndarray]:
    """
    Correções relevantes para desencontro das curvas:
      - _to_datetime64ns robusto (corrige AttributeError com DatetimeIndex).
      - meteo_time_shift_minutes: shift manual aplicado (irradiância e tamb).
      - auto_time_shift: estima lag por correlação (pac_real vs G) e aplica shift.
        Isso corrige o caso típico de séries 15-min rotuladas no início/fim da janela
        vs potência do inversor em timestamps diferentes.
    """
    if g_clear is None and g_clear_sky is not None:
        g_clear = g_clear_sky

    Ta = _to_np(tamb_c)
    G = _to_np(g_poa)

    # tenta converter times (se disponível) para uso no shift/interp e transposição
    t0 = None
    if times_utc is not None:
        try:
            t0 = _to_datetime64ns(times_utc)
        except Exception as e:
            warnings.warn(f"Aviso (modelo físico): times_utc inválido para alinhamento temporal: {e}")
            t0 = None

    transpo_used = False
    transpo_info: Dict[str, Any] = {}

    # ----------------------------
    # Transposição se necessário
    # ----------------------------
    if (g_poa is None or G.size == 0) and use_transposition_if_needed:
        if ghi is not None and times_utc is not None:
            lat = plant.lat_deg
            lon = plant.lon_deg
            tilt = plant.tilt_deg
            azs = plant.azimuth_deg
            if None not in (lat, lon, tilt, azs):
                # aplica shift de tempo também dentro da transposição
                trans = transpose_ghi_to_poa_isotropic(
                    ghi=ghi,
                    dhi=dhi,
                    dni=dni,
                    times_utc=times_utc,
                    lat_deg=float(lat),
                    lon_deg=float(lon),
                    tilt_deg=float(tilt),
                    azimuth_deg=float(azs),
                    albedo=float(getattr(plant, "albedo", 0.20) or 0.20),
                    times_shift_minutes=float(meteo_time_shift_minutes),
                )
                G = trans["g_poa"]
                transpo_used = True
                transpo_info = {
                    "method": "liu_jordan_isotropic",
                    "has_dhi": dhi is not None,
                    "has_dni": dni is not None,
                    "times_shift_minutes": float(meteo_time_shift_minutes),
                }
            else:
                G = _to_np(ghi)
                transpo_used = True
                transpo_info = {"method": "fallback_poa_equals_ghi", "reason": "missing_geo_fields"}
        else:
            G = np.array([], dtype=float)

    # ----------------------------
    # Harmoniza tamanhos
    # ----------------------------
    if Ta.size == 0 and G.size == 0:
        n = 0
    else:
        n = int(max(Ta.size, G.size))
        if Ta.size == 0:
            Ta = np.full(n, np.nan, dtype=float)
        if G.size == 0:
            G = np.full(n, np.nan, dtype=float)
        if Ta.size != G.size:
            raise ValueError(f"g_poa e tamb_c devem ter o mesmo tamanho. got {G.size} vs {Ta.size}")

    if n == 0:
        empty = np.array([], dtype=float)
        empty_b = np.array([], dtype=bool)
        return {
            "tcell_c": empty,
            "pdc_raw_w": empty,
            "pdc_expected_w": empty,
            "pac_expected_w": empty,
            "eta_inv": empty,
            "valid": empty_b,
            "v_dc_expected_v": empty,
            "i_dc_expected_a": empty,
            "v_ratio": empty,
            "i_ratio": empty,
            "g_mean_60m": empty,
            "g_std_60m": empty,
            "g_cv_60m": empty,
            "sky_stable_mask": empty_b,
            "csi": empty,
            "p_stc_w": empty,
            "p_ac_pu_real": empty,
            "p_ac_pu_model": empty,
            "pr_real_inst": empty,
            "pr_model_inst": empty,
            "mismatch_abs_w": empty,
            "mismatch_rel": empty,
            "rca_label": np.array([], dtype=object),
            "meta": {
                "k_sys": float(getattr(plant, "k_sys", np.nan) or np.nan),
                "inv_model": str(getattr(plant, "inv_model", "constant")),
                "noct_c": float(getattr(plant, "noct_c", np.nan) or np.nan),
                "pac_rated_w": getattr(plant, "pac_rated_w", None),
                "n_modules_total": int(getattr(plant, "n_modules_total", 0) or 0),
                "strings_count": getattr(plant, "strings_count", None),
                "modules_per_string": getattr(plant, "modules_per_string", None),
                "dt_minutes": float(dt_minutes),
                "window_minutes": float(window_minutes),
                "transposition_used": bool(transpo_used),
                "transposition": transpo_info,
                "topology_ok": False,
                "force_zero_when_invalid": bool(force_zero_when_invalid),
                "meteo_time_shift_minutes": float(meteo_time_shift_minutes),
                "auto_time_shift": bool(auto_time_shift),
            },
        }

    # irradiância negativa -> 0 (NaN permanece NaN)
    G0 = np.where(np.isfinite(G), np.clip(G, 0.0, None), np.nan)

    # =========================
    # APLICAR SHIFT (manual)
    # =========================
    total_shift_min = float(meteo_time_shift_minutes)

    # =========================
    # APLICAR SHIFT (automático) para casar curvas
    # =========================
    auto_shift_min = 0.0
    if auto_time_shift and pac_real_w is not None and t0 is not None and t0.size == n:
        y = _to_np(pac_real_w)
        if y.size == n and np.isfinite(y).any() and np.isfinite(G0).any():
            dt_est = _infer_dt_minutes_from_times(t0, fallback=float(dt_minutes))
            dt_est = float(dt_est if dt_est > 0 else float(dt_minutes))

            max_steps = int(max(1, round(float(max_auto_shift_minutes) / dt_est)))
            w_smooth = int(max(1, round(float(auto_shift_smooth_minutes) / dt_est)))

            # suaviza para correlação (remove efeito de nuvens rápidas no pac_real)
            y_s = _rolling_nanmean(y, w_smooth)
            g_s = _rolling_nanmean(G0, w_smooth)

            # foca em período diurno (evita dominar por zeros noturnos)
            day = (np.isfinite(g_s) & (g_s >= 50.0)) | (np.isfinite(y_s) & (y_s >= 0.02 * np.nanmax(y_s)))
            if day.sum() >= 40:
                lag_steps = _best_lag_steps_xcorr(
                    x=np.where(day, y_s, np.nan),
                    y=np.where(day, g_s, np.nan),
                    max_lag_steps=max_steps,
                    min_samples=40,
                )
                auto_shift_min = float(lag_steps) * dt_est
                # clip do auto-shift
                auto_shift_min = float(np.clip(auto_shift_min, -float(max_auto_shift_minutes), float(max_auto_shift_minutes)))

                # aplica auto-shift atrasando/adiantando meteo para casar com pac_real
                # (shift>0 => atrasa meteo/modelo)
                if abs(auto_shift_min) >= 1e-6:
                    total_shift_min += auto_shift_min
                    warnings.warn(
                        f"Aviso (alinhamento temporal): aplicado auto_shift={auto_shift_min:.2f} min (dt~{dt_est:.2f} min). "
                        "Se não fizer sentido, desative auto_time_shift."
                    )

    # aplica shift total (manual + auto) em G0 e Ta (se tiver time base)
    if t0 is not None and t0.size == n and abs(total_shift_min) >= 1e-6:
        G0 = _shift_series_by_minutes(G0, t0, total_shift_min)
        Ta = _shift_series_by_minutes(Ta, t0, total_shift_min)

    # Temperatura de célula
    Tc = tcell_noct(G0, Ta, noct_c=float(plant.noct_c))

    # parâmetros por ponto
    Tk = Tc + 273.15
    Vt_mod = _vt_cell(Tk) * float(module.ns)
    aVt = float(module.a) * Vt_mod

    iph = iph_irr_temp(module, G0, Tc)
    i0 = i0_temp(module, Tc)
    rp = rp_irr(module, G0)
    rs = float(module.rs_ohm)

    n_mod_total = int(getattr(plant, "n_modules_total", 0) or 0)
    valid = (G0 >= float(g_min_valid)) & np.isfinite(G0) & np.isfinite(Tc) & (n_mod_total > 0)

    # MPP por módulo (somente válidos)
    pmp_mod = np.full(n, np.nan, dtype=float)
    vmp_mod = np.full(n, np.nan, dtype=float)
    imp_mod = np.full(n, np.nan, dtype=float)

    idx = np.where(valid)[0]
    if idx.size > 0:
        voc_g = voc_guess(module, Tc[idx], G0[idx])
        mpp = pmp_module_vec(
            iph=iph[idx],
            i0=i0[idx],
            rs=rs,
            rp=rp[idx],
            aVt=aVt[idx],
            voc_g=voc_g,
            n_points=n_points,
        )
        pmp_mod[idx] = mpp["pmp"]
        vmp_mod[idx] = mpp["vmp"]
        imp_mod[idx] = mpp["imp"]

    # Potência DC "ideal" do arranjo (sem k_sys)
    pdc_raw_w_raw = pmp_mod * float(n_mod_total)

    if force_zero_when_invalid:
        pdc_raw_w = np.where(valid, np.where(np.isfinite(pdc_raw_w_raw), pdc_raw_w_raw, 0.0), 0.0)
    else:
        pdc_raw_w = np.where(valid, pdc_raw_w_raw, np.nan)

    # ==== k_sys aplicado no lado DC (entrada do inversor) ====
    k_sys = float(getattr(plant, "k_sys", 1.0) or 1.0)
    if force_zero_when_invalid:
        pdc_expected_w = np.where(valid, np.where(np.isfinite(pdc_raw_w), pdc_raw_w * k_sys, 0.0), 0.0)
    else:
        pdc_expected_w = np.where(valid, pdc_raw_w * k_sys, np.nan)

    # Topologia p/ Vdc/Idc esperados
    strings_count = getattr(plant, "strings_count", None)
    modules_per_string = getattr(plant, "modules_per_string", None)

    topology_ok = (
        strings_count is not None
        and modules_per_string is not None
        and int(strings_count) > 0
        and int(modules_per_string) > 0
    )

    v_dc_expected_v = np.full(n, np.nan, dtype=float)
    i_dc_expected_a = np.full(n, np.nan, dtype=float)
    if topology_ok:
        v_dc_expected_v = np.where(valid, vmp_mod * float(modules_per_string), np.nan)
        i_dc_expected_a = np.where(valid, imp_mod * float(strings_count), np.nan)

    # Inversor
    eta_inv = inverter_efficiency(pdc_expected_w, plant)
    inv_model = (getattr(plant, "inv_model", "constant") or "constant").lower().strip()
    pso = float(getattr(plant, "inv_pso_w", 0.0) or 0.0)

    if inv_model == "load_curve":
        pdc_net = np.clip(pdc_expected_w - pso, 0.0, None)
        pac_expected_w = pdc_net * eta_inv
    else:
        pac_expected_w = pdc_expected_w * eta_inv

    if plant.pac_rated_w is not None:
        pac_expected_w = np.minimum(pac_expected_w, float(plant.pac_rated_w))

    if force_zero_when_invalid:
        pac_expected_w = np.where(valid, np.where(np.isfinite(pac_expected_w), pac_expected_w, 0.0), 0.0)

    # Estabilidade de irradiância
    stab = irradiance_stability(G0, dt_minutes=dt_minutes, window_minutes=window_minutes)
    g_mean = stab["g_mean"]
    g_std = stab["g_std"]
    g_cv = stab["g_cv"]
    sky_stable_mask = (g_cv <= float(cv_max_stable)) & (G0 >= float(g_min_valid)) & np.isfinite(g_cv)

    # CSI
    g_clear_np = None
    if g_clear is not None:
        gc = _to_np(g_clear)
        if gc.size == n:
            g_clear_np = np.clip(gc, 0.0, None)
    csi = clear_sky_index(G0, g_clear_np)

    # Normalizações
    pmp_stc_mod = float(module.vmp_n) * float(module.imp_n)
    p_stc_scalar = float(n_mod_total) * pmp_stc_mod
    p_stc_w = np.full(n, p_stc_scalar, dtype=float)

    denom_pr = p_stc_w * (G0 / 1000.0)
    denom_pr = np.maximum(denom_pr, 1.0)

    if compute_norm:
        pr_model = np.where(valid, pac_expected_w / denom_pr, np.nan)
        p_ac_pu_model = np.where(valid, pac_expected_w / np.maximum(p_stc_w, 1.0), np.nan)
    else:
        pr_model = np.full(n, np.nan, dtype=float)
        p_ac_pu_model = np.full(n, np.nan, dtype=float)

    # Ratios DC
    v_ratio = np.full(n, np.nan, dtype=float)
    i_ratio = np.full(n, np.nan, dtype=float)

    if v_dc_real_v is not None and topology_ok:
        vdc = _to_np(v_dc_real_v)
        if vdc.size == n:
            v_ratio = np.where(valid, vdc / np.maximum(v_dc_expected_v, 1.0), np.nan)

    if i_dc_real_a is not None and topology_ok:
        idc = _to_np(i_dc_real_a)
        if idc.size == n:
            i_ratio = np.where(valid, idc / np.maximum(i_dc_expected_a, 0.1), np.nan)

    meta: Dict[str, Any] = {
        "k_sys": k_sys,
        "inv_model": str(getattr(plant, "inv_model", "constant")),
        "inv_eta_min": float(getattr(plant, "inv_eta_min", np.nan) or np.nan),
        "inv_eta_max": float(getattr(plant, "inv_eta_max", np.nan) or np.nan),
        "inv_alpha": float(getattr(plant, "inv_alpha", np.nan) or np.nan),
        "inv_pso_w": float(getattr(plant, "inv_pso_w", np.nan) or np.nan),
        "inv_pdc_nom_w": float(getattr(plant, "inv_pdc_nom_w", np.nan) or np.nan),
        "eta_inv_mean": _nanmean_safe(eta_inv),
        "eta_inv_p10": _nanpercentile_safe(eta_inv, 10),
        "eta_inv_p90": _nanpercentile_safe(eta_inv, 90),
        "noct_c": float(getattr(plant, "noct_c", np.nan) or np.nan),
        "pac_rated_w": getattr(plant, "pac_rated_w", None),
        "n_modules_total": int(n_mod_total),
        "strings_count": int(strings_count) if strings_count is not None else None,
        "modules_per_string": int(modules_per_string) if modules_per_string is not None else None,
        "topology_ok": bool(topology_ok),
        "topology_note": None if topology_ok else "Para V/I esperados e RCA: informe strings_count e modules_per_string em PVPlantDetails.",
        "g_min_valid": float(g_min_valid),
        "dt_minutes": float(dt_minutes),
        "window_minutes": float(window_minutes),
        "cv_max_stable": float(cv_max_stable),
        "pmp_stc_mod_w": float(pmp_stc_mod),
        "p_stc_total_w": float(p_stc_scalar),
        "transposition_used": bool(transpo_used),
        "transposition": transpo_info,
        "force_zero_when_invalid": bool(force_zero_when_invalid),
        "time_shift_minutes_manual": float(meteo_time_shift_minutes),
        "time_shift_minutes_auto": float(auto_shift_min),
        "time_shift_minutes_total": float(total_shift_min),
        "geo": {
            "lat_deg": getattr(plant, "lat_deg", None),
            "lon_deg": getattr(plant, "lon_deg", None),
            "tilt_deg": getattr(plant, "tilt_deg", None),
            "azimuth_deg": getattr(plant, "azimuth_deg", None),
            "albedo": float(getattr(plant, "albedo", 0.20) or 0.20),
        },
    }

    out: Dict[str, np.ndarray] = {
        "tcell_c": Tc,
        "pdc_raw_w": pdc_raw_w,
        "pdc_expected_w": pdc_expected_w,
        "pac_expected_w": pac_expected_w,
        "eta_inv": eta_inv,
        "valid": valid,
        "g_mean_60m": g_mean,
        "g_std_60m": g_std,
        "g_cv_60m": g_cv,
        "sky_stable_mask": sky_stable_mask,
        "csi": csi,
        "p_stc_w": p_stc_w,
        "p_ac_pu_model": p_ac_pu_model,
        "pr_model_inst": pr_model,
        "pmp_mod_w": pmp_mod,
        "vmp_mod_v": vmp_mod,
        "imp_mod_a": imp_mod,
        "v_dc_expected_v": v_dc_expected_v,
        "i_dc_expected_a": i_dc_expected_a,
        "v_ratio": v_ratio,
        "i_ratio": i_ratio,
        "meta": meta,
    }

    # mismatch + PR real + pu real
    if pac_real_w is not None:
        y = _to_np(pac_real_w)
        if y.size != n:
            raise ValueError(f"pac_real_w deve ter mesmo tamanho de g_poa. got {y.size} vs {n}")

        abs_err = y - pac_expected_w
        den_rel = np.maximum(pac_expected_w, float(eps_w))
        den_rel = np.where(np.isfinite(den_rel), den_rel, float(eps_w))
        rel_err = abs_err / den_rel

        out["mismatch_abs_w"] = np.where(valid, abs_err, np.nan)
        out["mismatch_rel"] = np.where(valid, rel_err, np.nan)

        if compute_norm:
            out["p_ac_pu_real"] = np.where(valid, y / np.maximum(p_stc_w, 1.0), np.nan)
            out["pr_real_inst"] = np.where(valid, y / denom_pr, np.nan)
        else:
            out["p_ac_pu_real"] = np.full(n, np.nan, dtype=float)
            out["pr_real_inst"] = np.full(n, np.nan, dtype=float)
    else:
        out["mismatch_abs_w"] = np.full(n, np.nan, dtype=float)
        out["mismatch_rel"] = np.full(n, np.nan, dtype=float)
        out["p_ac_pu_real"] = np.full(n, np.nan, dtype=float)
        out["pr_real_inst"] = np.full(n, np.nan, dtype=float)

    # RCA label
    out["rca_label"] = np.full(n, "n/a", dtype=object)
    if compute_rca:
        mr = out.get("mismatch_rel", None)
        if mr is not None and topology_ok and np.isfinite(v_ratio).any() and np.isfinite(i_ratio).any():
            out["rca_label"] = classify_root_cause_vec(
                valid=valid,
                mismatch_rel=np.asarray(mr, dtype=float),
                v_ratio=v_ratio,
                i_ratio=i_ratio,
                sky_stable_mask=sky_stable_mask,
                strings_count=int(strings_count) if strings_count is not None else None,
            )

    # campos auxiliares p/ feature_extraction
    out["g_poa_used"] = G0
    out["g_poa"] = out["g_poa_used"]  # alias p/ UI
    if pac_real_w is not None:
        y = _to_np(pac_real_w)
        out["pac_real_w"] = y if y.size == n else np.full(n, np.nan, dtype=float)
    else:
        out["pac_real_w"] = np.full(n, np.nan, dtype=float)

    if v_ac_real_v is not None:
        vac = _to_np(v_ac_real_v)
        out["v_ac_real_v"] = vac if vac.size == n else np.full(n, np.nan, dtype=float)
    else:
        out["v_ac_real_v"] = np.full(n, np.nan, dtype=float)

    return out


# =========================
# Helpers: mapear Django models -> dataclasses
# =========================
def module_from_pvmodule(pv_module: Any) -> ModuleOneDiode:
    """Converte PVModule (Django) -> ModuleOneDiode."""
    voc = float(pv_module.voc_v)
    isc = float(pv_module.isc_a)

    kv = voc * (float(pv_module.temp_coeff_voc_pct_c) / 100.0)
    ki = isc * (float(pv_module.temp_coeff_isc_pct_c) / 100.0)

    a = float(pv_module.diode_a) if getattr(pv_module, "diode_a", None) is not None else 1.3

    return ModuleOneDiode(
        isc_n=isc,
        voc_n=voc,
        vmp_n=float(pv_module.vmp_v),
        imp_n=float(pv_module.imp_a),
        ns=int(pv_module.num_celulas),
        kv_v_per_c=kv,
        ki_a_per_c=ki,
        rs_ohm=float(pv_module.rs_ohm),
        rp_ohm=float(pv_module.rp_ohm),
        a=a,
    )


def plant_from_details(
    details: Any,
    *,
    k_sys_default: float = 0.900,
    noct_default: float = 45.0,
    pac_rated_w: Optional[float] = None,
    inverter: Optional[Any] = None,
    use_inverter_eff: bool = False,
) -> PlantModel:
    """Converte PVPlantDetails (Django) -> PlantModel."""
    sc = _safe_int(getattr(details, "strings_count", None) or getattr(details, "num_strings", None))
    mps = _safe_int(getattr(details, "modules_per_string", None) or getattr(details, "modulos_por_string", None))

    n_mod = getattr(details, "modules_total", None)
    if n_mod is None and sc is not None and mps is not None:
        n_mod = int(sc) * int(mps)
    n_mod = int(n_mod or 0)

    k_sys = _float_or(getattr(details, "k_sys", None), k_sys_default)
    noct_c = _float_or(getattr(details, "noct_c", None), noct_default)

    lat = _safe_float(getattr(details, "latitude_deg", None) or getattr(details, "lat_deg", None))
    lon = _safe_float(getattr(details, "longitude_deg", None) or getattr(details, "lon_deg", None))
    tilt = _safe_float(getattr(details, "tilt_deg", None) or getattr(details, "inclinacao_deg", None))
    azs = _safe_float(getattr(details, "azimuth_deg", None) or getattr(details, "azimute_deg", None))
    alb = _safe_float(getattr(details, "albedo", None))
    if alb is None:
        alb = 0.20

    if inverter is None:
        inverter = getattr(details, "inverter", None)

    inv_model = "constant"
    inv_eff = 1.0
    eta_max = 0.985
    eta_min = 0.92
    alpha = 6.0
    pso = 20.0
    pdc_nom = None

    if inverter is not None:
        if pac_rated_w is None:
            pac_nom = _safe_float(getattr(inverter, "p_ac_nom_w", None))
            if pac_nom is not None:
                pac_rated_w = float(pac_nom)

        eff_pct = _safe_float(getattr(inverter, "eficiencia_max_pct", None))
        if eff_pct is not None:
            eta_max = float(np.clip(eff_pct / 100.0, 0.0, 1.0))

        eff_euro = _safe_float(getattr(inverter, "eficiencia_euro_pct", None))
        if eff_euro is not None:
            eta_min = float(np.clip((eff_euro / 100.0) - 0.03, 0.85, eta_max))

        pso_db = _safe_float(getattr(inverter, "consumo_vazio_w", None))
        if pso_db is not None:
            pso = float(np.clip(pso_db, 0.0, 500.0))

    if use_inverter_eff:
        inv_model = "load_curve"
        if pac_rated_w is not None and eta_max > 0:
            pdc_nom = float(pac_rated_w) / max(eta_max, 0.80)

    return PlantModel(
        n_modules_total=n_mod,
        strings_count=sc,
        modules_per_string=mps,
        k_sys=float(k_sys),
        pac_rated_w=pac_rated_w,
        noct_c=float(noct_c),
        lat_deg=lat,
        lon_deg=lon,
        tilt_deg=tilt,
        azimuth_deg=azs,
        albedo=float(alb),
        inv_model=str(inv_model),
        inv_eff=float(inv_eff),
        inv_eta_max=float(eta_max),
        inv_eta_min=float(eta_min),
        inv_alpha=float(alpha),
        inv_pso_w=float(pso),
        inv_pdc_nom_w=pdc_nom,
    )


# =========================
# Strings heterogêneas (Ns diferentes em paralelo)
# =========================
@dataclass(frozen=True)
class StringGroup:
    strings_qty: int
    modules_per_string: int




def _string_groups_by_mppt_from_details(details: Any) -> Dict[int, List[StringGroup]]:
    out: Dict[int, List[StringGroup]] = {}
    qs = getattr(details, "string_configs", None)
    if qs is None:
        return out
    try:
        cfgs = list(qs.all())
    except Exception:
        return out
    for c in cfgs:
        mppt = _safe_int(getattr(c, "mppt", None))
        sq = _safe_int(getattr(c, "strings_qty", None))
        ns = _safe_int(getattr(c, "modules_per_string", None))
        if mppt is None or sq is None or ns is None or sq < 1 or ns < 1:
            continue
        out.setdefault(int(mppt), []).append(StringGroup(strings_qty=int(sq), modules_per_string=int(ns)))
    return out


def expected_dc_from_string_groups(
    *,
    g_poa: ArrayLike,
    tamb_c: ArrayLike,
    module: ModuleOneDiode,
    plant: PlantModel,
    groups: Union[Sequence[Tuple[int, int]], Sequence[StringGroup]],
    g_min_valid: float = 0.0,
    n_points: int = 60,
    force_zero_when_invalid: bool = False,
) -> Dict[str, np.ndarray]:
    G0 = np.asarray(g_poa, dtype=float).ravel()
    Tair = np.asarray(tamb_c, dtype=float).ravel()
    if G0.size != Tair.size:
        raise ValueError("g_poa e tamb_c devem ter o mesmo tamanho.")

    n = G0.size
    valid = np.isfinite(G0) & np.isfinite(Tair) & (G0 >= float(g_min_valid))
    Tc = tcell_noct(G0, Tair, noct_c=float(getattr(plant, "noct_c", 45.0) or 45.0))

    iph = iph_irr_temp(module, G0, Tc)
    i0 = i0_temp(module, Tc)
    rp = rp_irr(module, G0)
    Tk = Tc + 273.15
    aVt = float(module.a) * (_vt_cell(Tk) * float(module.ns))
    voc_g = voc_guess(module, Tc, G0)

    grp = pmp_array_groups_vec(
        iph=iph,
        i0=i0,
        rs=float(module.rs_ohm),
        rp=rp,
        aVt=aVt,
        voc_g=voc_g,
        groups=groups,
        n_points=int(max(30, n_points)),
    )

    pdc_raw = np.asarray(grp.get("pmp"), dtype=float)
    vdc_exp = np.asarray(grp.get("vmp"), dtype=float)
    idc_exp = np.asarray(grp.get("imp"), dtype=float)

    if force_zero_when_invalid:
        pdc_raw = np.where(valid, np.where(np.isfinite(pdc_raw), pdc_raw, 0.0), 0.0)
        vdc_exp = np.where(valid, np.where(np.isfinite(vdc_exp), vdc_exp, 0.0), 0.0)
        idc_exp = np.where(valid, np.where(np.isfinite(idc_exp), idc_exp, 0.0), 0.0)
    else:
        pdc_raw = np.where(valid, pdc_raw, np.nan)
        vdc_exp = np.where(valid, vdc_exp, np.nan)
        idc_exp = np.where(valid, idc_exp, np.nan)

    k_sys = float(getattr(plant, "k_sys", 1.0) or 1.0)
    pdc_expected = np.where(valid, pdc_raw * k_sys, np.nan)

    return {
        "valid": valid,
        "tcell_c": Tc,
        "g_poa_used": G0,
        "pdc_expected_w": pdc_expected,
        "v_dc_expected_v": vdc_exp,
        "i_dc_expected_a": idc_exp,
    }


def expected_dc_by_mppt_from_details(
    *,
    details: Any,
    module: ModuleOneDiode,
    plant: PlantModel,
    g_poa: ArrayLike,
    tamb_c: ArrayLike,
    g_min_valid: float = 0.0,
    n_points: int = 60,
) -> Dict[int, Dict[str, np.ndarray]]:
    groups_by_mppt = _string_groups_by_mppt_from_details(details)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for mppt, groups in groups_by_mppt.items():
        if not groups:
            continue
        out[int(mppt)] = expected_dc_from_string_groups(
            g_poa=g_poa,
            tamb_c=tamb_c,
            module=module,
            plant=plant,
            groups=groups,
            g_min_valid=g_min_valid,
            n_points=n_points,
            force_zero_when_invalid=False,
        )
    return out


def _as_groups(groups: Union[Sequence[Tuple[int, int]], Sequence[StringGroup]]) -> List[StringGroup]:
    out: List[StringGroup] = []
    for g in groups:
        if isinstance(g, StringGroup):
            sq, ns = int(g.strings_qty), int(g.modules_per_string)
        else:
            sq, ns = int(g[0]), int(g[1])
        if sq < 1 or ns < 1:
            continue
        out.append(StringGroup(sq, ns))
    if not out:
        raise ValueError("string_groups vazio/ inválido. Informe ao menos um grupo (strings_qty, modules_per_string).")
    return out


def pmp_array_groups_vec(
    iph: np.ndarray,
    i0: np.ndarray,
    rs: float,
    rp: np.ndarray,
    aVt: np.ndarray,
    voc_g: np.ndarray,
    *,
    groups: Union[Sequence[Tuple[int, int]], Sequence[StringGroup]],
    n_points: int = 60,
    max_iter: int = 28,
    chunk_size: int = 4000,
) -> Dict[str, np.ndarray]:
    """Retorna (Pmp, Vmp, Imp) do arranjo, considerando grupos em paralelo com diferentes Ns."""
    groups_ = _as_groups(groups)
    maxNs = max(g.modules_per_string for g in groups_)

    iph = np.asarray(iph, dtype=float).ravel()
    i0 = np.asarray(i0, dtype=float).ravel()
    rp = np.asarray(rp, dtype=float).ravel()
    aVt = np.asarray(aVt, dtype=float).ravel()
    voc_g = np.asarray(voc_g, dtype=float).ravel()

    n = iph.size
    if not (i0.size == rp.size == aVt.size == voc_g.size == n):
        raise ValueError("iph, i0, rp, aVt, voc_g devem ter o mesmo tamanho.")

    vhat = _vhat_eps(int(n_points))[None, :]  # (1, M)

    pmp = np.full(n, np.nan, dtype=float)
    vmp = np.full(n, np.nan, dtype=float)
    imp = np.full(n, np.nan, dtype=float)

    for k0 in range(0, n, int(max(1, chunk_size))):
        k1 = min(n, k0 + int(max(1, chunk_size)))
        sl = slice(k0, k1)

        iph_c = iph[sl]
        i0_c = i0[sl]
        rp_c = rp[sl]
        aVt_c = aVt[sl]
        voc_g_c = voc_g[sl]

        voc_mod = np.maximum(voc_newton_vec(iph_c, i0_c, rp_c, aVt_c, voc_g_c), 0.1)
        Vmax = voc_mod * float(maxNs)

        Vmat = Vmax[:, None] * vhat
        Itot = np.zeros_like(Vmat, dtype=float)

        for g in groups_:
            Ns = float(g.modules_per_string)
            Np = float(g.strings_qty)
            Vmod = Vmat / Ns
            Imod = iv_current_mat(Vmod, iph_c, i0_c, float(rs), rp_c, aVt_c, max_iter=int(max_iter))
            Imod = np.maximum(Imod, 0.0)
            Itot += Np * Imod

        Pmat = Vmat * Itot
        idx = np.argmax(Pmat, axis=1)
        rows = np.arange(Pmat.shape[0])

        pmp[sl] = Pmat[rows, idx]
        vmp[sl] = Vmat[rows, idx]
        imp[sl] = Itot[rows, idx]

    return {"pmp": pmp, "vmp": vmp, "imp": imp}


def voc_isc_array_groups_vec(
    iph: np.ndarray,
    i0: np.ndarray,
    rs: float,
    rp: np.ndarray,
    aVt: np.ndarray,
    voc_g: np.ndarray,
    *,
    groups: Union[Sequence[Tuple[int, int]], Sequence[StringGroup]],
    max_iter: int = 28,
    chunk_size: int = 4000,
) -> Dict[str, np.ndarray]:
    """Voc_array ~ max(Ns)*Voc_mod; Isc_array ~ sum(strings_qty)*Isc_mod."""
    groups_ = _as_groups(groups)
    maxNs = max(g.modules_per_string for g in groups_)
    sumNp = sum(g.strings_qty for g in groups_)

    iph = np.asarray(iph, dtype=float).ravel()
    i0 = np.asarray(i0, dtype=float).ravel()
    rp = np.asarray(rp, dtype=float).ravel()
    aVt = np.asarray(aVt, dtype=float).ravel()
    voc_g = np.asarray(voc_g, dtype=float).ravel()

    n = iph.size
    if not (i0.size == rp.size == aVt.size == voc_g.size == n):
        raise ValueError("iph, i0, rp, aVt, voc_g devem ter o mesmo tamanho.")

    voc_arr = np.full(n, np.nan, dtype=float)
    isc_arr = np.full(n, np.nan, dtype=float)

    for k0 in range(0, n, int(max(1, chunk_size))):
        k1 = min(n, k0 + int(max(1, chunk_size)))
        sl = slice(k0, k1)

        iph_c = iph[sl]
        i0_c = i0[sl]
        rp_c = rp[sl]
        aVt_c = aVt[sl]
        voc_g_c = voc_g[sl]

        voc_mod = np.maximum(voc_newton_vec(iph_c, i0_c, rp_c, aVt_c, voc_g_c), 0.1)
        voc_arr[sl] = voc_mod * float(maxNs)

        V0 = np.zeros((iph_c.size, 1), dtype=float)
        Isc_mod = iv_current_mat(V0, iph_c, i0_c, float(rs), rp_c, aVt_c, max_iter=int(max_iter)).ravel()
        Isc_mod = np.maximum(Isc_mod, 0.0)
        isc_arr[sl] = float(sumNp) * Isc_mod

    return {"voc_array": voc_arr, "isc_array": isc_arr}


def signature_features_from_operating_point(
    v_dc_meas: np.ndarray,
    i_dc_meas: np.ndarray,
    *,
    v_dc_exp: np.ndarray,
    i_dc_exp: np.ndarray,
    voc_array: Optional[np.ndarray] = None,
    isc_array: Optional[np.ndarray] = None,
    ff_mpp: Optional[np.ndarray] = None,
    eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """Features básicos para assinaturas elétricas."""
    v_dc_meas = np.asarray(v_dc_meas, dtype=float)
    i_dc_meas = np.asarray(i_dc_meas, dtype=float)
    v_dc_exp = np.asarray(v_dc_exp, dtype=float)
    i_dc_exp = np.asarray(i_dc_exp, dtype=float)

    p_meas = v_dc_meas * i_dc_meas
    p_exp = v_dc_exp * i_dc_exp

    v_ratio = v_dc_meas / np.maximum(v_dc_exp, eps)
    i_ratio = i_dc_meas / np.maximum(i_dc_exp, eps)
    p_ratio = p_meas / np.maximum(p_exp, eps)

    out = {"v_ratio": v_ratio, "i_ratio": i_ratio, "p_ratio": p_ratio}

    if voc_array is not None and isc_array is not None and ff_mpp is not None:
        voc_array = np.asarray(voc_array, dtype=float)
        isc_array = np.asarray(isc_array, dtype=float)
        ff_mpp = np.asarray(ff_mpp, dtype=float)

        v_norm = v_dc_meas / np.maximum(voc_array, eps)
        i_norm = i_dc_meas / np.maximum(isc_array, eps)
        ff_op = v_norm * i_norm
        ff_ratio = ff_op / np.maximum(ff_mpp, eps)
        out["ff_ratio"] = ff_ratio

    return out



