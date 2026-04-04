from __future__ import annotations

from collections import Counter
from datetime import datetime
import random
from typing import Any, Dict, List, Mapping, Optional

from core.models import PVPlant
from core.services.fdd.dashboard_runtime import build_mismatch_dashboard_payload, parse_dashboard_params
from core.services.fdd.dashboard_common import DashboardServiceError
from core.services.fdd.param_catalog import (
    BASIC_PARAM_DEFAULTS,
    RANDOM_SEARCH_DEFAULT_SEED,
    RANDOM_SEARCH_DEFAULT_TRIALS,
    RANDOM_SEARCH_MAX_TRIALS,
    TIPOLOGY_RANDOM_SEARCH_SPACE,
)


def _coerce_float(raw: Any, default: float) -> float:
    if raw in (None, ""):
        return float(default)
    try:
        return float(str(raw).replace(",", "."))
    except Exception:
        return float(default)


def _coerce_int(raw: Any, default: int) -> int:
    if raw in (None, ""):
        return int(default)
    try:
        return int(float(str(raw).replace(",", ".")))
    except Exception:
        return int(default)


def _sample_candidate(rng: random.Random, base_data: Mapping[str, Any]) -> Dict[str, Any]:
    cand: Dict[str, Any] = {
        "warn_abs": _coerce_float(base_data.get("warn_abs"), BASIC_PARAM_DEFAULTS["warn_abs"]),
        "fault_abs": _coerce_float(base_data.get("fault_abs"), BASIC_PARAM_DEFAULTS["fault_abs"]),
        "pmin_w": _coerce_float(base_data.get("pmin_w"), BASIC_PARAM_DEFAULTS["pmin_w"]),
    }
    gpoa_min = float(rng.choice(TIPOLOGY_RANDOM_SEARCH_SPACE["gpoa_min"]))
    cand["gpoa_min"] = gpoa_min
    cand["sun_available_gpoa_wm2"] = gpoa_min

    coarse_choices = [v for v in TIPOLOGY_RANDOM_SEARCH_SPACE["coarse_diag_gpoa_wm2"] if v >= gpoa_min]
    coarse = float(rng.choice(coarse_choices or [max(250.0, gpoa_min)]))
    cand["coarse_diag_gpoa_wm2"] = coarse

    fine_choices = [v for v in TIPOLOGY_RANDOM_SEARCH_SPACE["fine_diag_gpoa_wm2"] if v >= coarse]
    fine = float(rng.choice(fine_choices or [max(coarse, 450.0)]))
    cand["fine_diag_gpoa_wm2"] = fine

    for key in ("stable_cv_max", "stable_ramp_max_wm2", "zero_abs_w", "zero_rel_model", "degraded_rel", "severe_rel"):
        cand[key] = float(rng.choice(TIPOLOGY_RANDOM_SEARCH_SPACE[key]))

    if cand["severe_rel"] < cand["degraded_rel"]:
        cand["severe_rel"] = max(float(cand["degraded_rel"]), 0.60)
    return cand


def _late_day_unknown_rate(payload: Dict[str, Any], gate: float) -> float:
    series = payload.get("series") or {}
    ts_local = series.get("t_local") or []
    labels = series.get("diagnosis_label") or []
    g_poa = series.get("g_poa") or []
    valid_n = max(1, int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("ok", 0)) + int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("warn", 0)) + int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("crit", 0)))
    n_bad = 0
    for t, lab, g in zip(ts_local, labels, g_poa):
        if str(lab or "") != "unknown_shutdown_with_sun":
            continue
        try:
            dt = datetime.fromisoformat(str(t))
        except Exception:
            continue
        if (dt.hour > 16 or (dt.hour == 16 and dt.minute >= 30)) and (g is not None and float(g) >= float(gate)):
            n_bad += 1
    return n_bad / valid_n


def _model_zero_under_sun_rate(payload: Dict[str, Any], gate: float) -> float:
    series = payload.get("series") or {}
    g_poa = series.get("g_poa") or []
    pac_model = series.get("p_ac_model_w") or []
    pac_real = series.get("p_ac_real_w") or []
    valid_n = max(1, int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("ok", 0)) + int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("warn", 0)) + int(payload.get("summary", {}).get("counts_by_mode", {}).get("tipologia", {}).get("crit", 0)))
    n_bad = 0
    for g, pm, pr in zip(g_poa, pac_model, pac_real):
        try:
            g_ok = g is not None and float(g) >= float(gate)
            pm_zero = pm is not None and abs(float(pm)) <= 1.0
            pr_pos = pr is not None and float(pr) > 1.0
        except Exception:
            continue
        if g_ok and pm_zero and pr_pos:
            n_bad += 1
    return n_bad / valid_n


def _score_payload(payload: Dict[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") or {}
    by_mode = summary.get("counts_by_mode") or {}
    counts = by_mode.get("tipologia") or summary.get("counts") or {}
    ok = int(counts.get("ok", 0))
    warn = int(counts.get("warn", 0))
    crit = int(counts.get("crit", 0))
    valid_n = max(1, ok + warn + crit)
    total_n = max(1, int(summary.get("n_points", 0) or 1))

    conf = payload.get("confidence_summary") or {}
    data_rel = float(conf.get("data_reliability_mean") or 0.0)
    det_conf = float(conf.get("detection_confidence_mean") or 0.0)
    diag_conf = float(conf.get("diagnosis_confidence_mean") or 0.0)

    series = payload.get("series") or {}
    labels = Counter(str(x or "") for x in (series.get("diagnosis_label") or []))
    direct_grid = sum(1 for x in (series.get("direct_grid_evidence") or []) if bool(x))
    hm_classes = series.get("hm_class_typology") or []
    diag_scores = series.get("diagnosis_confidence_score") or []
    low_conf_crit = sum(
        1
        for cls, sc in zip(hm_classes, diag_scores)
        if str(cls) == "crit" and sc is not None and float(sc) < 0.60
    )

    ok_rate = ok / valid_n
    warn_rate = warn / valid_n
    crit_rate = crit / valid_n
    valid_rate = valid_n / total_n
    unknown_rate = labels.get("unknown_shutdown_with_sun", 0) / valid_n
    persistent_rate = labels.get("persistent_underperformance", 0) / valid_n
    late_unknown_rate = _late_day_unknown_rate(payload, float(candidate.get("gpoa_min", 0.0)))
    model_zero_rate = _model_zero_under_sun_rate(payload, float(candidate.get("gpoa_min", 0.0)))
    grid_reward = min(direct_grid, 20) / 20.0
    low_conf_crit_rate = low_conf_crit / max(1, crit)

    score = (
        100.0 * (0.45 * diag_conf + 0.25 * det_conf + 0.15 * data_rel + 0.15 * ok_rate)
        + 20.0 * valid_rate
        + 8.0 * grid_reward
        - 100.0 * (1.10 * crit_rate + 0.55 * warn_rate + 0.90 * unknown_rate + 0.30 * persistent_rate)
        - 100.0 * (1.75 * model_zero_rate + 1.25 * late_unknown_rate)
        - 10.0 * low_conf_crit_rate
    )

    return {
        "score": float(score),
        "ok": ok,
        "warn": warn,
        "crit": crit,
        "valid_n": valid_n,
        "valid_rate": valid_rate,
        "data_reliability_mean": data_rel,
        "detection_confidence_mean": det_conf,
        "diagnosis_confidence_mean": diag_conf,
        "unknown_shutdown_with_sun": int(labels.get("unknown_shutdown_with_sun", 0)),
        "persistent_underperformance": int(labels.get("persistent_underperformance", 0)),
        "partial_generation_loss_probable": int(labels.get("partial_generation_loss_probable", 0)),
        "grid_evidence": int(direct_grid),
        "late_day_unknown_rate": float(late_unknown_rate),
        "model_zero_under_sun_rate": float(model_zero_rate),
        "low_conf_crit_rate": float(low_conf_crit_rate),
    }


def run_typology_random_search(*, plant: PVPlant, base_data: Mapping[str, Any], tz_name: str) -> Dict[str, Any]:
    trials = max(1, min(_coerce_int(base_data.get("rs_trials"), RANDOM_SEARCH_DEFAULT_TRIALS), RANDOM_SEARCH_MAX_TRIALS))
    seed = _coerce_int(base_data.get("rs_seed"), RANDOM_SEARCH_DEFAULT_SEED)
    rng = random.Random(seed)

    required = {
        "plant_id": str(plant.id),
        "start": str(base_data.get("start") or ""),
        "end": str(base_data.get("end") or ""),
        "display_mode": "tipologia",
    }
    if not required["start"] or not required["end"]:
        raise DashboardServiceError("start/end são obrigatórios para o random search.", status_code=400)

    trial_results: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None
    errors: List[str] = []

    for idx in range(trials):
        cand = _sample_candidate(rng, base_data)
        merged: Dict[str, Any] = dict(base_data)
        merged.update(required)
        merged.update({k: str(v) for k, v in cand.items()})
        merged["persist"] = "0"
        merged["save"] = "0"

        try:
            params = parse_dashboard_params(merged, tz_name)
            payload = build_mismatch_dashboard_payload(plant, params)
            if not payload.get("ok"):
                raise DashboardServiceError(str(payload.get("error") or "payload inválido"), status_code=500)
            metrics = _score_payload(payload, cand)
            trial = {
                "trial": idx + 1,
                "params": cand,
                "metrics": metrics,
            }
            trial_results.append(trial)
            if best_trial is None or float(metrics["score"]) > float(best_trial["metrics"]["score"]):
                best_trial = trial
        except Exception as exc:
            errors.append(f"trial {idx + 1}: {type(exc).__name__}: {exc}")

    if best_trial is None:
        raise DashboardServiceError("Nenhum trial válido foi concluído no random search.", status_code=500)

    trial_results_sorted = sorted(trial_results, key=lambda x: float(x["metrics"]["score"]), reverse=True)

    best_params = {
        "display_mode": "tipologia",
        "warn_abs": _coerce_float(base_data.get("warn_abs"), BASIC_PARAM_DEFAULTS["warn_abs"]),
        "fault_abs": _coerce_float(base_data.get("fault_abs"), BASIC_PARAM_DEFAULTS["fault_abs"]),
        "pmin_w": _coerce_float(base_data.get("pmin_w"), BASIC_PARAM_DEFAULTS["pmin_w"]),
    }
    best_params.update(best_trial["params"])

    return {
        "ok": True,
        "objective_mode": "tipologia",
        "trials_requested": trials,
        "trials_succeeded": len(trial_results),
        "seed": seed,
        "best_params": best_params,
        "best_metrics": best_trial["metrics"],
        "top_trials": trial_results_sorted[:10],
        "search_space": {k: list(v) for k, v in TIPOLOGY_RANDOM_SEARCH_SPACE.items()},
        "errors": errors[:20],
    }
