from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Any

BASIC_PARAM_DEFAULTS: Dict[str, float] = {
    "warn_abs": 0.35,
    "fault_abs": 0.90,
    "gpoa_min": 50.0,
    "pmin_w": 0.0,
}

ADVANCED_PARAM_META: List[Dict[str, Any]] = [
    {"key": "meteo_pos_abs", "label": "Erro meteo + abs.", "default": 0.25, "auto_text": "0.25", "step": "0.01", "min": "0", "group": "Modelo e mismatch"},
    {"key": "shading_std_abs", "label": "Sombreamento std abs.", "default": 0.22, "auto_text": "0.22", "step": "0.01", "min": "0", "group": "Modelo e mismatch"},
    {"key": "shading_window_points", "label": "Janela sombreamento [pts]", "default": 6, "auto_text": "6", "step": "1", "min": "1", "group": "Modelo e mismatch"},
    {"key": "max_gap_minutes", "label": "Gap máx. evento [min]", "default": 30.0, "auto_text": "30", "step": "1", "min": "0", "group": "Modelo e mismatch"},
    {"key": "gpoa_plot_min", "label": "GPOA mín. gráfico [W/m²]", "default": 700.0, "auto_text": "max(700, GPOA gate)", "step": "1", "min": "0", "group": "Modelo e mismatch"},
    {"key": "pmodel_plot_min", "label": "Pmodelo mín. gráfico [W]", "default": 200.0, "auto_text": "max(200, Pmin)", "step": "1", "min": "0", "group": "Modelo e mismatch"},
    {"key": "mismatch_clip_abs", "label": "Clip |mismatch|", "default": 2.0, "auto_text": "2.0", "step": "0.1", "min": "0", "group": "Modelo e mismatch"},

    {"key": "sun_available_gpoa_wm2", "label": "Sol disponível [W/m²]", "default": 150.0, "auto_text": "150", "step": "1", "min": "0", "group": "Detecção"},
    {"key": "coarse_diag_gpoa_wm2", "label": "Diag. coarse [W/m²]", "default": 700.0, "auto_text": "max(700, GPOA gate)", "step": "1", "min": "0", "group": "Detecção"},
    {"key": "fine_diag_gpoa_wm2", "label": "Diag. fino [W/m²]", "default": 800.0, "auto_text": "max(800, GPOA gate)", "step": "1", "min": "0", "group": "Detecção"},
    {"key": "stable_cv_max", "label": "CV máx. estabilidade", "default": 0.08, "auto_text": "0.08", "step": "0.01", "min": "0", "group": "Detecção"},
    {"key": "stable_ramp_max_wm2", "label": "Rampa máx. [W/m²]", "default": 120.0, "auto_text": "120", "step": "1", "min": "0", "group": "Detecção"},
    {"key": "stable_window_points", "label": "Janela estabilidade [pts]", "default": 6, "auto_text": "6", "step": "1", "min": "2", "group": "Detecção"},
    {"key": "min_baseline_points", "label": "Baseline mín. [pts]", "default": 24, "auto_text": "24", "step": "1", "min": "4", "group": "Detecção"},
    {"key": "inv_cov_min", "label": "Cobertura mín. inversor", "default": 0.30, "auto_text": "0.30", "step": "0.01", "min": "0", "max": "1", "group": "Detecção"},
    {"key": "ewma_lambda", "label": "EWMA λ", "default": 0.20, "auto_text": "0.20", "step": "0.01", "min": "0.01", "max": "1", "group": "Detecção"},
    {"key": "ewma_L", "label": "EWMA L", "default": 3.0, "auto_text": "3.0", "step": "0.1", "min": "0", "group": "Detecção"},
    {"key": "cusum_k", "label": "CUSUM k", "default": 0.50, "auto_text": "0.50", "step": "0.01", "min": "0", "group": "Detecção"},
    {"key": "cusum_h", "label": "CUSUM h", "default": 8.0, "auto_text": "8.0", "step": "0.1", "min": "0", "group": "Detecção"},

    {"key": "zero_abs_w", "label": "Zero injeção abs. [W]", "default": 100.0, "auto_text": "100", "step": "1", "min": "0", "group": "RCA"},
    {"key": "zero_rel_model", "label": "Zero injeção rel. modelo", "default": 0.05, "auto_text": "0.05", "step": "0.01", "min": "0", "group": "RCA"},
    {"key": "degraded_rel", "label": "Perda moderada rel.", "default": 0.25, "auto_text": "0.25", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "severe_rel", "label": "Perda severa rel.", "default": 0.50, "auto_text": "0.50", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "low_i_ratio_warn", "label": "I ratio warn", "default": 0.35, "auto_text": "0.35", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "low_i_ratio_crit", "label": "I ratio crit", "default": 0.15, "auto_text": "0.15", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "low_v_ratio_warn", "label": "V ratio warn", "default": 0.80, "auto_text": "0.80", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "low_v_ratio_crit", "label": "V ratio crit", "default": 0.60, "auto_text": "0.60", "step": "0.01", "min": "0", "max": "1", "group": "RCA"},
    {"key": "vac_low_ratio", "label": "Vac ratio min.", "default": 0.90, "auto_text": "0.90", "step": "0.01", "min": "0", "max": "2", "group": "RCA"},
    {"key": "vac_high_ratio", "label": "Vac ratio max.", "default": 1.10, "auto_text": "1.10", "step": "0.01", "min": "0", "max": "2", "group": "RCA"},
    {"key": "vac_abs_margin_v", "label": "Margem abs. Vac [V]", "default": 10.0, "auto_text": "10", "step": "1", "min": "0", "group": "RCA"},
    {"key": "freq_abs_tol_hz", "label": "Tol. freq. abs. [Hz]", "default": 1.0, "auto_text": "1.0", "step": "0.1", "min": "0", "group": "RCA"},
    {"key": "clip_margin", "label": "Margem clipping", "default": 0.98, "auto_text": "0.98", "step": "0.01", "min": "0", "max": "2", "group": "RCA"},
    {"key": "clip_model_margin", "label": "Margem modelo clipping", "default": 1.02, "auto_text": "1.02", "step": "0.01", "min": "0", "max": "2", "group": "RCA"},
    {"key": "rca_min_baseline_points", "label": "Baseline mín. RCA [pts]", "default": 24, "auto_text": "24", "step": "1", "min": "4", "group": "RCA"},
]

ADVANCED_PARAM_DEFAULTS: Dict[str, Any] = {item["key"]: item["default"] for item in ADVANCED_PARAM_META}
ADVANCED_PARAM_KEYS: List[str] = [item["key"] for item in ADVANCED_PARAM_META]

TIPOLOGY_RANDOM_SEARCH_SPACE = OrderedDict([
    ("gpoa_min", [120.0, 150.0, 180.0, 200.0]),
    ("coarse_diag_gpoa_wm2", [250.0, 320.0, 400.0, 500.0, 700.0]),
    ("fine_diag_gpoa_wm2", [450.0, 500.0, 600.0, 800.0]),
    ("stable_cv_max", [0.06, 0.08, 0.10]),
    ("stable_ramp_max_wm2", [90.0, 120.0, 150.0]),
    ("zero_abs_w", [15.0, 30.0, 50.0, 80.0, 100.0]),
    ("zero_rel_model", [0.02, 0.03, 0.05, 0.08]),
    ("degraded_rel", [0.20, 0.25, 0.30]),
    ("severe_rel", [0.50, 0.60, 0.65, 0.70]),
])

RANDOM_SEARCH_DEFAULT_TRIALS = 24
RANDOM_SEARCH_MAX_TRIALS = 120
RANDOM_SEARCH_DEFAULT_SEED = 42


def advanced_groups() -> List[Dict[str, Any]]:
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in ADVANCED_PARAM_META:
        groups.setdefault(str(item["group"]), []).append(item)
    return [{"title": title, "controls": controls} for title, controls in groups.items()]
