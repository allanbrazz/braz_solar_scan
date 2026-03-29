#views/mppt_gnn_fdd
from __future__ import annotations

import ast
import json
import logging
import traceback
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Max, Min
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from zoneinfo import ZoneInfo

from core.views._imports import *  # mantém compatibilidade com o projeto
from core.models import PVPlant, PVPlantMergedRecord15m

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Imports opcionais
# ---------------------------------------------------------------------
try:
    from core.models import FaultEvent, FaultEventMPPT, PlantDiagnostic15m  # type: ignore
except Exception:
    FaultEvent = None  # type: ignore
    FaultEventMPPT = None  # type: ignore
    PlantDiagnostic15m = None  # type: ignore

try:
    from core.services.fdd.pipeline import run_detection_pipeline  # type: ignore
except Exception:
    run_detection_pipeline = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.event_infer_pipeline import infer_events_and_persist  # type: ignore
except Exception:
    infer_events_and_persist = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.event_loader import load_event_window  # type: ignore
except Exception:
    load_event_window = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.storage import load_model_health, list_available_model_versions  # type: ignore
except Exception:
    load_model_health = None  # type: ignore
    list_available_model_versions = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.report_pdf import build_mppt_gnn_pdf_report  # type: ignore
except Exception:
    build_mppt_gnn_pdf_report = None  # type: ignore


# ============================================================
# Configuração
# ============================================================
LABEL_BY_CODE: dict[int, str] = {
    0: "normal",
    1: "mppt_disconnected",
    2: "inverter_off_under_sun",
    3: "mppt_imbalance",
    4: "curtailment_clipping",
    5: "meteo_bias",
    99: "unknown_fault",
}

SEV_BY_LABEL: dict[str, int] = {
    "normal": 0,
    "curtailment_clipping": 1,
    "meteo_bias": 1,
    "mppt_imbalance": 2,
    "mppt_disconnected": 3,
    "inverter_off_under_sun": 3,
    "localized_loss": 3,
    "plant_wide_loss": 3,
    "unknown_fault": 2,
    "invalid": 0,
    "anomaly": 2,
    "no_oper_data": 0,
}

BENIGN_LABELS = {"normal", "curtailment_clipping", "meteo_bias"}

HEATMAP_STATE_NONE = 0
HEATMAP_STATE_OK = 1
HEATMAP_STATE_WARN = 2
HEATMAP_STATE_CRIT = 3

EVENT_GREEN_CONFIDENCE_MIN = 0.70
EVENT_WARN_CONFIDENCE_MIN = 0.45
WARN_MISMATCH_REL = 0.10
WARN_MISMATCH_REL_SINGLE = 0.20
WARN_MIN_GPOA_WM2 = 700.0
WARN_MIN_QC_SCORE = 0.60
WARN_LABEL_MISMATCH = "warn_mismatch"
WARN_LABEL_MISMATCH_INTERP = "warn_mismatch_interp"
WARN_LABEL_GUARDED = "warn_refined_guard"

DIAG_BASE_CANDIDATES = [
    "ts_utc",
    "valid",
    "anomaly_flag",
    "rca_code",
    "rca_label",
    "detector_score",
    "detector_version",
    "ewma_z",
    "cusum_score",
    "stable_sky",
    "mismatch_rel",
    "pac_real_w",
    "pac_model_w",
    "p_ac_real_w",
    "p_ac_model_w",
    "g_poa",
    "gpoa",
    "ghi",
    "dni",
    "dhi",
    "gti",
    "tcell_c",
    "temp_air_c",
    "temp_air",
    "t_air_c",
    "wind_speed",
    "rh",
    "vdc_total_v",
    "v_dc_v",
    "vac_v",
    "v_ac_v",
    "iac_a",
    "i_ac_a",
    "fac_hz",
    "freq_hz",
    "pf",
    "qac_var",
    "status",
    "mode",
    "warning",
    "warnings",
    "alarm",
    "alarms",
    "inv_status",
    "inv_mode",
    "inv_warning",
    "inv_warnings",
    "inv_alarm",
    "inv_alarms",
    "inv_temp_c",
    "temp_inv_c",
    "flag_inv_missing",
    "flag_meteo_missing",
    "meteo_qc_score",
    "data_reliability_score",
    "data_reliability_level",
    "detection_confidence_score",
    "detection_confidence_level",
    "diagnosis_confidence",
    "diagnosis_confidence_score",
    "diagnosis_confidence_level",
    "confidence_notes_json",
    "state_label",
    "domain_label",
    "diagnosis_label",
    "irradiance_tier",
    "fine_diag_allowed",
    "meteo_quality_ok",
    "direct_grid_evidence",
    "zero_injection_flag",
    "meteo_qc_score",
    "flag_meteo_low_confidence",
    "flag_meteo_interpolated",
    "flag_meteo_outlier",
    "flag_meteo_artifact",
]

MPPT_FIELD_CANDIDATES: List[str] = []
for i in range(1, 9):
    MPPT_FIELD_CANDIDATES += [
        f"mppt{i}_pac_w",
        f"mppt{i}_pdc_w",
        f"mppt{i}_vdc_v",
        f"mppt{i}_idc_a",
        f"mppt{i}_warning",
        f"mppt{i}_warnings",
        f"mppt{i}_alarm",
        f"mppt{i}_alarms",
        f"mppt{i}_status",
    ]


# ============================================================
# Helpers gerais
# ============================================================
def _plant_tz(plant: PVPlant) -> ZoneInfo:
    tz_name = getattr(plant, "timezone", None) or getattr(settings, "TIME_ZONE", "UTC")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _parse_date(s: Optional[str], default: date) -> date:
    if not s:
        return default
    try:
        return date.fromisoformat(s)
    except Exception:
        return default


def _parse_int(s: Optional[str], default: int, lo: int, hi: int) -> int:
    try:
        v = int(s) if s is not None else default
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _parse_float(s: Optional[str], default: float) -> float:
    try:
        return float(s) if s is not None else float(default)
    except Exception:
        return float(default)



def _normalize_dt_minutes(v: Optional[str]) -> int:
    try:
        x = int(v) if v is not None else 15
    except Exception:
        x = 15
    if x <= 15:
        return 15
    if x <= 30:
        return 30
    return 60


def _json_body(request: HttpRequest) -> dict:
    try:
        raw = request.body.decode("utf-8") if request.body else ""
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _bins_per_day(dt_minutes: int) -> int:
    return int(24 * 60 // dt_minutes)


def _tkey(dt_local: datetime) -> str:
    return dt_local.strftime("%Y-%m-%dT%H:%M")


def _parse_tkey_to_local(tkey: str, tz: ZoneInfo) -> Optional[datetime]:
    try:
        x = str(tkey or "").strip()
        if not x:
            return None
        if " " in x and "T" not in x:
            x = x.replace(" ", "T")
        dt = datetime.fromisoformat(x)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        return dt
    except Exception:
        return None


def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _resolve_event_classifier_version(params: Any, default: Optional[str] = None) -> Optional[str]:
    raw = None
    try:
        raw = params.get("event_classifier_version")
    except Exception:
        raw = None
    if raw in (None, ""):
        try:
            raw = params.get("model_version")
        except Exception:
            raw = None
    val = str(raw or "").strip()
    return val or default


def _resolve_trained_model_version(params: Any, default: Optional[str] = None) -> Optional[str]:
    raw = None
    try:
        raw = params.get("trained_model_version")
    except Exception:
        raw = None
    val = str(raw or "").strip()
    return val or default


def _list_trained_model_versions() -> List[str]:
    if list_available_model_versions is None:
        return []
    try:
        return [str(v) for v in list_available_model_versions() if str(v or "").strip()]
    except Exception:
        logger.exception("trained model version listing failed")
        return []


def _build_version_summary(*, detector_version: Optional[str], event_classifier_version: Optional[str], trained_model_version: Optional[str], view_name: str) -> Dict[str, Any]:
    detector = (detector_version or "").strip() or None
    event_classifier = (event_classifier_version or "").strip() or None
    trained = (trained_model_version or "").strip() or None
    if view_name == "mppt_gnn_fdd":
        event_note = "Classificador event-level/MPPT persistido em FaultEventMPPT."
        trained_note = "Bundle treinado usado apenas para métricas e auditoria do bloco Saúde do modelo."
    else:
        event_note = "Não aplicável nesta tela."
        trained_note = "Não aplicável nesta tela."
    return {
        "detector_version": detector,
        "event_classifier_version": event_classifier,
        "trained_model_version": trained,
        "detector_note": "Detector plant-level que encontra bins/eventos suspeitos.",
        "event_classifier_note": event_note,
        "trained_model_note": trained_note,
    }


def _load_model_health_payload(trained_model_version: Optional[str]) -> Optional[Dict[str, Any]]:
    if not trained_model_version or load_model_health is None:
        return None
    try:
        return load_model_health(model_version=str(trained_model_version))
    except Exception:
        logger.exception("model health load failed", extra={"trained_model_version": trained_model_version})
        return None


def _has_useful_oper_data_from_diag_row(r: Dict[str, Any]) -> bool:
    """
    Coverage-first:
    considera o bin "operativo" quando existe algum dado útil do inversor
    persistido na linha diagnóstica, mesmo que o ponto não seja elegível
    para diagnóstico fino (`valid=False`).
    """
    candidates = [
        "pac_real_w", "p_ac_real_w",
        "v_ac_v", "vac_v",
        "i_ac_a", "iac_a",
        "v_dc_v", "vdc_total_v",
        "i_dc_a",
    ]
    for k in candidates:
        v = _safe_float(r.get(k), None)
        if v is not None:
            return True
    return False


def _has_useful_oper_data_from_merged_row(r: Dict[str, Any]) -> bool:
    """
    Coverage-first usando a merged 15 min.

    Regras:
    - se a linha explicita que faltou inversor -> sem cobertura
    - se houver inv_coverage > 0 -> com cobertura
    - se houver ao menos uma variável operativa do inversor preenchida -> com cobertura

    Campos meramente estruturais/default (ex.: inv_n=0, alarm_sev=0)
    não devem, sozinhos, pintar o bin de verde.
    """
    flag_inv_missing = r.get("flag_inv_missing", None)
    if flag_inv_missing is True:
        return False

    inv_coverage = _safe_float(r.get("inv_coverage"), None)
    if inv_coverage is not None and inv_coverage > 0:
        return True

    operative_numeric_fields = [
        "p_ac_w",
        "p_dc_w",
        "v_ac_v",
        "i_ac_a",
        "v_dc_v",
        "i_dc_a",
        "e_ac_wh_15",
        "mppt1_vdc_v",
        "mppt2_vdc_v",
        "mppt3_vdc_v",
        "mppt4_vdc_v",
        "mppt1_idc_a",
        "mppt2_idc_a",
        "mppt3_idc_a",
        "mppt4_idc_a",
    ]

    for k in operative_numeric_fields:
        v = _safe_float(r.get(k), None)
        if v is not None:
            return True

    return False


def _error_json(msg: str, *, trace: Optional[str] = None) -> JsonResponse:
    payload: Dict[str, Any] = {"ok": False, "error": msg}
    if getattr(settings, "DEBUG", False) and trace:
        payload["trace"] = trace
    return JsonResponse(payload, status=200)


def _label_state(label: str) -> int:
    lab = (label or "").strip().lower()
    if not lab or lab in {"invalid", "no_oper_data"}:
        return HEATMAP_STATE_NONE
    if lab in BENIGN_LABELS:
        return HEATMAP_STATE_OK
    return HEATMAP_STATE_CRIT


def _mismatch_underperformance(mismatch_rel: Any) -> Optional[float]:
    mm = _safe_float(mismatch_rel, None)
    if mm is None:
        return None
    return max(0.0, -mm)


def _diag_gpoa_wm2(row: Dict[str, Any]) -> Optional[float]:
    for key in ("g_poa", "gpoa", "gti", "ghi"):
        v = _safe_float(row.get(key), None)
        if v is not None:
            return v
    return None


def _diag_meteo_good_for_warn(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    if bool(row.get("flag_meteo_missing")):
        return False
    if bool(row.get("flag_meteo_low_confidence")):
        return False
    score = _safe_float(row.get("meteo_qc_score"), None)
    if score is not None and score < WARN_MIN_QC_SCORE:
        return False
    return True


def _diag_meteo_allows_green_refinement(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    if bool(row.get("flag_meteo_missing")):
        return False
    if bool(row.get("flag_meteo_low_confidence")):
        return False
    return True


def _compute_warn_bin_map(
    *,
    diag_rows: List[Dict[str, Any]],
    tz: ZoneInfo,
    d_start: date,
    days_len: int,
    bpd: int,
    dt_minutes: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    candidate_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    per_day: Dict[int, List[Tuple[int, float]]] = {}

    for row in diag_rows:
        tsu = row.get("ts_utc")
        if tsu is None or not bool(row.get("valid")):
            continue
        if not _has_useful_oper_data_from_diag_row(row):
            continue
        if not _diag_meteo_good_for_warn(row):
            continue

        underperf = _mismatch_underperformance(row.get("mismatch_rel"))
        if underperf is None or underperf < WARN_MISMATCH_REL:
            continue

        gpoa = _diag_gpoa_wm2(row)
        if gpoa is not None and gpoa < WARN_MIN_GPOA_WM2:
            continue

        stable_sky = row.get("stable_sky")
        if stable_sky is not None and not bool(stable_sky):
            continue

        ts_local = tsu.astimezone(tz)
        di = (ts_local.date() - d_start).days
        if not (0 <= di < days_len):
            continue
        minutes = ts_local.hour * 60 + ts_local.minute
        bi = int(minutes // dt_minutes)
        if not (0 <= bi < bpd):
            continue

        key = (di, bi)
        meteo_interpolated = bool(row.get("flag_meteo_interpolated"))
        candidate_map[key] = {
            "state": HEATMAP_STATE_WARN,
            "label": WARN_LABEL_MISMATCH_INTERP if meteo_interpolated else WARN_LABEL_MISMATCH,
            "tkey": _tkey(ts_local),
            "mismatch_rel": _safe_float(row.get("mismatch_rel"), None),
            "underperf_rel": underperf,
            "gpoa_wm2": gpoa,
            "meteo_interpolated": meteo_interpolated,
            "meteo_qc_score": _safe_float(row.get("meteo_qc_score"), None),
        }
        per_day.setdefault(di, []).append((bi, underperf))

    warn_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for di, items in per_day.items():
        items = sorted(items, key=lambda x: x[0])
        for idx, (bi, underperf) in enumerate(items):
            prev_adj = idx > 0 and (bi - items[idx - 1][0] == 1)
            next_adj = idx + 1 < len(items) and (items[idx + 1][0] - bi == 1)
            persistent = prev_adj or next_adj
            if persistent or underperf >= WARN_MISMATCH_REL_SINGLE:
                key = (di, bi)
                info = dict(candidate_map[key])
                if bool(info.get("meteo_interpolated")):
                    info["reason"] = "persistent_mismatch_interpolated_meteo" if persistent else "single_bin_high_mismatch_interpolated_meteo"
                    info["warn_policy"] = "warn_under_interpolated_meteo_qc_guard"
                else:
                    info["reason"] = "persistent_mismatch" if persistent else "single_bin_high_mismatch"
                    info["warn_policy"] = "warn_under_good_meteo"
                warn_map[key] = info
    return warn_map


def _label_sev(label: str) -> int:
    return int(SEV_BY_LABEL.get((label or "").strip().lower(), 2))


def _event_score(label: str, severity_score: Any, confidence: Any, novelty_score: Any) -> float:
    sev_rank = _label_sev(label)
    sev_val = abs(_safe_float(severity_score, 0.0) or 0.0)
    conf = _safe_float(confidence, 0.0) or 0.0
    nov = _safe_float(novelty_score, 1.0) or 1.0
    return (sev_rank * 1_000_000_000.0) + (sev_val * 1_000_000.0) + (conf * 10_000.0) - (nov * 1000.0)


def _pred_rank(pred_label: Any, confidence: Any, novelty_score: Any) -> float:
    lab = str(pred_label or "")
    conf = _safe_float(confidence, 0.0) or 0.0
    nov = _safe_float(novelty_score, 1.0) or 1.0
    known_bonus = 1.0 if lab and lab != "unknown_fault" else 0.0
    sev = _label_sev(lab)
    return sev * 1_000_000.0 + known_bonus * 100_000.0 + conf * 10_000.0 - nov * 100.0


def _model_field_names(model_cls: Any) -> set[str]:
    names: set[str] = set()
    try:
        for f in model_cls._meta.get_fields():
            if getattr(f, "attname", None):
                names.add(str(f.attname))
            if getattr(f, "name", None):
                names.add(str(f.name))
    except Exception:
        pass
    return names


def _existing_fields(model_cls: Any, candidates: List[str]) -> List[str]:
    names = _model_field_names(model_cls)
    return [c for c in candidates if c in names]


def _all_concrete_field_names(model_cls: Any) -> List[str]:
    names: List[str] = []
    try:
        for f in model_cls._meta.concrete_fields:
            nm = getattr(f, "name", None)
            if nm:
                names.append(str(nm))
    except Exception:
        pass
    return names



def _distinct_nonempty_values(model_cls: Any, field_name: str, **filters: Any) -> List[str]:
    try:
        q = model_cls.objects.filter(**filters)
        vals = q.values_list(field_name, flat=True).distinct().order_by(field_name)
        return [str(v) for v in vals if str(v or "").strip()]
    except Exception:
        return []


def _detect_available_mppts(
    *,
    plant_id: int,
    source_oper: Optional[str] = None,
    source_meteo: Optional[str] = None,
    dt0_utc: Optional[datetime] = None,
    dt1_utc: Optional[datetime] = None,
    max_mppt: int = 8,
) -> List[int]:
    if PVPlantMergedRecord15m is None:
        return [1, 2, 3, 4]

    model_names = _model_field_names(PVPlantMergedRecord15m)
    candidate_idxs: List[int] = []
    for i in range(1, max_mppt + 1):
        if f"mppt{i}_vdc_v" in model_names or f"mppt{i}_idc_a" in model_names:
            candidate_idxs.append(i)

    if not candidate_idxs:
        return [1, 2, 3, 4]

    q = PVPlantMergedRecord15m.objects.filter(plant_id=plant_id)
    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)
    if source_meteo:
        q = q.filter(source_meteo=source_meteo)
    if dt0_utc is not None:
        q = q.filter(ts_utc__gte=dt0_utc)
    if dt1_utc is not None:
        q = q.filter(ts_utc__lt=dt1_utc)

    observed: List[int] = []
    for i in candidate_idxs:
        vf = f"mppt{i}_vdc_v"
        inf = f"mppt{i}_idc_a"
        has_v = False
        has_i = False
        try:
            if vf in model_names:
                has_v = q.exclude(**{f"{vf}__isnull": True}).exists()
            if inf in model_names:
                has_i = q.exclude(**{f"{inf}__isnull": True}).exists()
        except Exception:
            pass
        if has_v or has_i:
            observed.append(i)

    return observed or candidate_idxs


def _coerce_jsonish(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_coerce_jsonish(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce_jsonish(val) for k, val in v.items()}
    return v


def _decode_structured_text(v: Any) -> Any:
    if not isinstance(v, str):
        return v

    s = v.strip()
    if not s:
        return v

    looks_structured = (
        (s.startswith("{") and s.endswith("}")) or
        (s.startswith("[") and s.endswith("]"))
    )
    if not looks_structured:
        return v

    try:
        return _coerce_jsonish(json.loads(s))
    except Exception:
        pass

    try:
        return _coerce_jsonish(ast.literal_eval(s))
    except Exception:
        pass

    return v


def _coerce_jsonish_deep(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _coerce_jsonish_deep(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce_jsonish_deep(x) for x in v]

    v2 = _coerce_jsonish(v)
    v3 = _decode_structured_text(v2)

    if isinstance(v3, dict):
        return {str(k): _coerce_jsonish_deep(val) for k, val in v3.items()}
    if isinstance(v3, (list, tuple)):
        return [_coerce_jsonish_deep(x) for x in v3]

    return v3


def _merge_prefixed(dst: Dict[str, Any], src: Dict[str, Any], prefix: str) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[f"{prefix}{k}"] = v


def _append_mppt_arrays_from_window(selected_bin: Dict[str, Any], win: Any, idx: int) -> None:
    try:
        n_mppt = 0
        if getattr(win, "mppt_vdc", None) is not None:
            n_mppt = int(win.mppt_vdc.shape[0])
        elif getattr(win, "mppt_idc", None) is not None:
            n_mppt = int(win.mppt_idc.shape[0])

        n_mppt = max(0, min(n_mppt, 8))
        for i in range(n_mppt):
            tag = f"mppt{i+1}"
            vdc = None
            idc = None
            pdc = None

            if getattr(win, "mppt_vdc", None) is not None:
                val = win.mppt_vdc[i, idx]
                vdc = float(val) if val == val else None
            if getattr(win, "mppt_idc", None) is not None:
                val = win.mppt_idc[i, idx]
                idc = float(val) if val == val else None

            if vdc is not None and idc is not None:
                pdc = float(vdc * idc)

            selected_bin[f"{tag}_vdc_v"] = vdc
            selected_bin[f"{tag}_idc_a"] = idc
            selected_bin[f"{tag}_pdc_w"] = pdc
    except Exception:
        logger.exception("failed to append mppt arrays from window")


def _sum_none_vals(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    ok = False
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        ok = True
    return acc if ok else None


def _mean_none_vals(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    n = 0
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        n += 1
    return (acc / n) if n else None


def _mean_nonzero_vals(xs: List[Optional[float]], *, eps: float = 1e-9) -> Optional[float]:
    vals: List[float] = []
    for v in xs:
        if v is None:
            continue
        fv = float(v)
        if abs(fv) <= eps:
            continue
        vals.append(fv)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _is_mppt_source(src: str) -> bool:
    return "|MPPT" in str(src or "").upper()


def _is_agg_source(src: str) -> bool:
    s = str(src or "").strip()
    if not s:
        return False
    u = s.upper()
    if "|" not in u:
        return True
    if u.endswith("|AGG"):
        return True
    return False


def _extract_mppt_index_from_source(src: str) -> Optional[int]:
    s = str(src or "")
    import re
    m = re.search(r"(?:\||\b)MPPT\s*([0-9]+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_effectively_active_source(vals: Dict[str, Any]) -> bool:
    pdc = _safe_float(vals.get("p_dc_w"), None)
    pac = _safe_float(vals.get("p_ac_w"), None)
    idc = _safe_float(vals.get("i_dc_a"), None)
    vdc = _safe_float(vals.get("v_dc_v"), None)

    if pdc is not None and abs(pdc) > 1.0:
        return True
    if pac is not None and abs(pac) > 1.0:
        return True
    if idc is not None and abs(idc) > 0.2 and vdc is not None and abs(vdc) > 1.0:
        return True
    return False


def _build_canonical_mppt_from_sources(
    sources: Dict[str, Dict[str, Any]],
    *,
    max_mppt: int = 8,
) -> Dict[str, Any]:
    """
    Bloco canônico por MPPT:
      - mantém somente grandezas coerentes por MPPT:
        Pac, Pdc, Vdc, Idc e metadados/warning/alarm/status
      - NÃO replica Vac/Iac/cobertura/flag do inversor em cada MPPT
    """
    out: Dict[str, Any] = {}

    for i in range(1, max_mppt + 1):
        out[f"mppt{i}_source_oper"] = None
        out[f"mppt{i}_pac_w"] = None
        out[f"mppt{i}_pdc_w"] = None
        out[f"mppt{i}_vdc_v"] = None
        out[f"mppt{i}_idc_a"] = None
        out[f"mppt{i}_warning"] = None
        out[f"mppt{i}_warnings"] = None
        out[f"mppt{i}_alarm"] = None
        out[f"mppt{i}_alarms"] = None
        out[f"mppt{i}_status"] = None

    for src, vals in (sources or {}).items():
        idx = _extract_mppt_index_from_source(src)
        if idx is None or idx < 1 or idx > max_mppt:
            continue

        def _first_non_null(*keys):
            for k in keys:
                if k in vals and vals.get(k) is not None:
                    return vals.get(k)
            return None

        out[f"mppt{idx}_source_oper"] = src
        out[f"mppt{idx}_pac_w"] = _first_non_null("p_ac_w")
        out[f"mppt{idx}_pdc_w"] = _first_non_null("p_dc_w")
        out[f"mppt{idx}_vdc_v"] = _first_non_null("v_dc_v")
        out[f"mppt{idx}_idc_a"] = _first_non_null("i_dc_a")
        out[f"mppt{idx}_warning"] = _first_non_null("warning", "inv_warning")
        out[f"mppt{idx}_warnings"] = _first_non_null("warnings", "inv_warnings")
        out[f"mppt{idx}_alarm"] = _first_non_null("alarm", "inv_alarm", "alarm_code")
        out[f"mppt{idx}_alarms"] = _first_non_null("alarms", "inv_alarms", "alarm_sev")
        out[f"mppt{idx}_status"] = _first_non_null("status", "mode")

    return out


def _best_pred_rows_for_events(
    event_ids: List[int],
    *,
    event_classifier_version: Optional[str],
    mppt: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Retorna o melhor prediction row por event_id.
    Se mppt == 0 => considera todos os MPPTs e escolhe o melhor por score.
    Se mppt > 0 => filtra naquele MPPT.
    """
    out: Dict[int, Dict[str, Any]] = {}
    if FaultEventMPPT is None or not event_ids:
        return out

    q = FaultEventMPPT.objects.filter(event_id__in=event_ids)
    if event_classifier_version:
        q = q.filter(model_version=event_classifier_version)
    if mppt > 0:
        q = q.filter(mppt=mppt)

    rows = list(
        q.values(
            "event_id",
            "mppt",
            "pred_label",
            "pred_code",
            "confidence",
            "novelty_score",
            "proba",
            "contribution",
            "model_version",
            "source_oper",
        )
    )

    best_score: Dict[int, float] = {}
    for r in rows:
        eid = int(r["event_id"])
        sc = _pred_rank(r.get("pred_label"), r.get("confidence"), r.get("novelty_score"))
        prev = best_score.get(eid)
        if prev is None or sc > prev:
            best_score[eid] = sc
            out[eid] = r

    return out


def _build_merged_snapshot_for_ts(*, plant_id: int, ts_utc: datetime) -> Dict[str, Any]:
    """
    Snapshot do merged_15m num timestamp:
      - raw_operational_records: dump bruto completo por source_oper
      - chosen_total: agregado limpo/consistente
      - canonical_mppt: somente grandezas coerentes por MPPT
    """
    if PVPlantMergedRecord15m is None:
        return {
            "source_oper_list": [],
            "sources": {},
            "raw_operational_records": {},
            "meteo": {},
            "chosen_total": {},
            "canonical_mppt": {},
        }

    merged_fields = _all_concrete_field_names(PVPlantMergedRecord15m)

    mrows = list(
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc=ts_utc,
        ).values(*merged_fields).order_by("source_oper")
    )

    if not mrows:
        return {
            "source_oper_list": [],
            "sources": {},
            "raw_operational_records": {},
            "meteo": {},
            "chosen_total": {},
            "canonical_mppt": {},
        }

    by_src: Dict[str, Dict[str, Any]] = {}
    raw_by_src: Dict[str, Dict[str, Any]] = {}
    src_list: List[str] = []

    for r in mrows:
        src = str(r.get("source_oper") or "").strip()
        if not src:
            continue

        src_list.append(src)

        row_dump: Dict[str, Any] = {}
        for k, v in r.items():
            row_dump[k] = _coerce_jsonish_deep(v)

        by_src[src] = dict(row_dump)
        raw_by_src[src] = dict(row_dump)

    first = mrows[0]
    meteo: Dict[str, Any] = {}
    for k in [
        "gti", "ghi", "dni", "dhi",
        "temp_air", "wind_speed", "rh",
        "meteo_qc_score", "flag_meteo_low_confidence", "flag_meteo_interpolated",
        "flag_meteo_outlier", "flag_meteo_artifact",
        "flag_meteo_missing", "source_meteo"
    ]:
        if k in first:
            meteo[k] = _coerce_jsonish_deep(first.get(k))

    present = list(by_src.keys())
    present_mppt = [s for s in present if _is_mppt_source(s)]
    present_agg = [s for s in present if _is_agg_source(s)]

    if present_mppt:
        chosen = present_mppt
        policy = "mppt_sum"
    elif present_agg:
        chosen = present_agg
        policy = "agg_fallback"
    else:
        chosen = present
        policy = "any_fallback"

    active_chosen = [s for s in chosen if _is_effectively_active_source(by_src.get(s, {}))]
    chosen_for_dc = active_chosen if active_chosen else chosen

    pac_mppt = _sum_none_vals([_safe_float(by_src[s].get("p_ac_w"), None) for s in present_mppt]) if present_mppt else None
    pac_agg = _sum_none_vals([_safe_float(by_src[s].get("p_ac_w"), None) for s in present_agg]) if present_agg else None

    pac_l = [_safe_float(by_src[s].get("p_ac_w"), None) for s in chosen] if chosen else []
    pdc_l = [_safe_float(by_src[s].get("p_dc_w"), None) for s in chosen] if chosen else []

    vdc_active_l = [_safe_float(by_src[s].get("v_dc_v"), None) for s in chosen_for_dc] if chosen_for_dc else []
    idc_active_l = [_safe_float(by_src[s].get("i_dc_a"), None) for s in chosen_for_dc] if chosen_for_dc else []

    vac_l = [_safe_float(by_src[s].get("v_ac_v"), None) for s in chosen] if chosen else []
    iac_l = [_safe_float(by_src[s].get("i_ac_a"), None) for s in chosen] if chosen else []
    cov_l = [_safe_float(by_src[s].get("inv_coverage"), None) for s in chosen] if chosen else []

    miss_flags = [bool(by_src[s].get("flag_inv_missing") or False) for s in chosen] if chosen else []
    if not miss_flags:
        miss_all = True
        miss_partial = False
    else:
        miss_all = all(miss_flags)
        miss_partial = any(miss_flags) and (not miss_all)

    chosen_total = {
        "policy": policy,
        "p_ac_w": _sum_none_vals(pac_l),
        "p_ac_mppt_sum_w": pac_mppt,
        "p_ac_agg_w": pac_agg,
        "p_dc_w": _sum_none_vals(pdc_l),
        "v_dc_active_mean_v": _mean_nonzero_vals(vdc_active_l),
        "i_dc_sum_a": _sum_none_vals(idc_active_l),
        "v_ac_v": _mean_nonzero_vals(vac_l),
        "i_ac_a": _mean_nonzero_vals(iac_l),
        "inv_coverage": _mean_none_vals(cov_l),
        "flag_inv_missing_all": bool(miss_all),
        "flag_inv_missing_partial": bool(miss_partial),
        "active_mppt_n": len(active_chosen),
        "chosen_sources": chosen,
    }

    canonical_mppt = _build_canonical_mppt_from_sources(by_src)

    return {
        "source_oper_list": src_list,
        "sources": by_src,
        "raw_operational_records": raw_by_src,
        "meteo": meteo,
        "chosen_total": chosen_total,
        "canonical_mppt": canonical_mppt,
    }


# ============================================================
# Helpers de eventos
# ============================================================

def _find_event_for_tkey(
    *,
    plant_id: int,
    dt_local: datetime,
    tz: ZoneInfo,
    event_classifier_version: Optional[str],
    detector_version: Optional[str],
    source_oper: Optional[str],
    source_meteo: Optional[str],
    mppt: int,
) -> Optional[int]:
    if FaultEvent is None:
        return None

    tsu = dt_local.astimezone(dt_tz.utc)

    q = FaultEvent.objects.filter(
        plant_id=plant_id,
        ts_start_utc__lte=tsu,
        ts_end_utc__gte=tsu,
    )
    if detector_version:
        q = q.filter(detector_version=detector_version)
    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)
    if source_meteo:
        q = q.filter(source_meteo=source_meteo)

    events = list(q.order_by("ts_start_utc"))
    if not events:
        return None

    event_ids = [e.id for e in events]
    pred_map = _best_pred_rows_for_events(event_ids, event_classifier_version=event_classifier_version, mppt=mppt)

    best_id = None
    best_score = None
    for ev in events:
        pr = pred_map.get(ev.id)
        label = (
            (pr.get("pred_label") if pr else None)
            or ev.final_label
            or ev.event_label_prelim
            or "unknown_fault"
        )
        sc = _event_score(
            label=label,
            severity_score=ev.severity_score,
            confidence=(pr.get("confidence") if pr else ev.confidence),
            novelty_score=(pr.get("novelty_score") if pr else ev.novelty_score),
        )
        if best_score is None or sc > best_score:
            best_score = sc
            best_id = ev.id
    return best_id

def _build_event_bin_map(
    *,
    plant_id: int,
    tz: ZoneInfo,
    dt0_utc: datetime,
    dt1_utc: datetime,
    d_start: date,
    days_len: int,
    bpd: int,
    dt_minutes: int,
    source_oper: Optional[str],
    source_meteo: Optional[str],
    detector_version: Optional[str],
    event_classifier_version: Optional[str],
    mppt: int,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], List[dict], List[str], List[str], List[str], List[str]]:
    best_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if FaultEvent is None:
        return best_info, [], [], [], [], []

    q = FaultEvent.objects.filter(
        plant_id=plant_id,
        ts_start_utc__lt=dt1_utc,
        ts_end_utc__gte=dt0_utc,
    ).order_by("ts_start_utc")

    if detector_version:
        q = q.filter(detector_version=detector_version)
    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)
    if source_meteo:
        q = q.filter(source_meteo=source_meteo)

    events = list(
        q.values(
            "id",
            "ts_start_utc",
            "ts_end_utc",
            "source_oper",
            "source_meteo",
            "detector_version",
            "event_label_prelim",
            "final_label",
            "severity_score",
            "confidence",
            "novelty_score",
        )
    )

    if source_oper:
        # já filtrado no queryset por prefixo
        pass
    elif mppt > 0:
        events = [
            e for e in events
            if _extract_mppt_index_from_source(str(e.get("source_oper") or "")) == mppt
        ]
    else:
        # view plant-level/agregada: não misturar eventos MPPT específicos
        events = [
            e for e in events
            if _is_agg_source(str(e.get("source_oper") or ""))
        ]

    mv_list: List[str] = []
    if FaultEventMPPT is not None:
        qmv = FaultEventMPPT.objects.filter(event__plant_id=plant_id)
        if source_oper:
            qmv = qmv.filter(event__source_oper__startswith=source_oper)
        if source_meteo:
            qmv = qmv.filter(event__source_meteo=source_meteo)
        if detector_version:
            qmv = qmv.filter(event__detector_version=detector_version)
        mv_list = [str(v) for v in qmv.values_list("model_version", flat=True).distinct().order_by("model_version") if str(v or "").strip()]

    qe = FaultEvent.objects.filter(plant_id=plant_id)
    if detector_version:
        qe = qe.filter(detector_version=detector_version)
    if source_oper:
        qe = qe.filter(source_oper__startswith=source_oper)
    if source_meteo:
        qe = qe.filter(source_meteo=source_meteo)

    so_list = [str(v) for v in qe.values_list("source_oper", flat=True).distinct().order_by("source_oper") if str(v or "").strip()]
    sm_list = [str(v) for v in qe.values_list("source_meteo", flat=True).distinct().order_by("source_meteo") if str(v or "").strip()]
    dv_list = [str(v) for v in qe.values_list("detector_version", flat=True).distinct().order_by("detector_version") if str(v or "").strip()]

    pred_map = _best_pred_rows_for_events(
        [int(e["id"]) for e in events],
        event_classifier_version=event_classifier_version,
        mppt=mppt,
    )

    best_score: Dict[Tuple[int, int], float] = {}

    for ev in events:
        event_id = int(ev["id"])
        pred = pred_map.get(event_id)

        label = (
            (pred.get("pred_label") if pred else None)
            or ev.get("final_label")
            or ev.get("event_label_prelim")
            or "unknown_fault"
        )
        confidence = pred.get("confidence") if pred else ev.get("confidence")
        novelty_score = pred.get("novelty_score") if pred else ev.get("novelty_score")

        start_local = ev["ts_start_utc"].astimezone(tz)
        end_local = ev["ts_end_utc"].astimezone(tz)

        cur_bin = start_local.replace(second=0, microsecond=0)
        cur_bin = cur_bin.replace(minute=(cur_bin.minute // dt_minutes) * dt_minutes)

        while cur_bin <= end_local:
            di = (cur_bin.date() - d_start).days
            if 0 <= di < days_len:
                minutes = cur_bin.hour * 60 + cur_bin.minute
                bi = int(minutes // dt_minutes)
                if 0 <= bi < bpd:
                    sc = _event_score(
                        label=label,
                        severity_score=ev.get("severity_score"),
                        confidence=confidence,
                        novelty_score=novelty_score,
                    )
                    key = (di, bi)
                    prev = best_score.get(key)
                    if prev is None or sc > prev:
                        best_score[key] = sc
                        best_info[key] = {
                            "event_id": event_id,
                            "label": str(label),
                            "state": _label_state(str(label)),
                            "confidence": _safe_float(confidence, None),
                            "novelty_score": _safe_float(novelty_score, None),
                            "tkey": _tkey(cur_bin),
                            "source_oper": ev.get("source_oper"),
                            "source_meteo": ev.get("source_meteo"),
                            "detector_version": ev.get("detector_version"),
                            "score": sc,
                        }
            cur_bin += timedelta(minutes=dt_minutes)

    return best_info, events, mv_list, so_list, sm_list, dv_list
# ============================================================
# Página
# ============================================================
@require_GET
@login_required

def mppt_gnn_fdd_view(request: HttpRequest):
    qs = (
        PVPlant.objects.all().order_by("nome")
        if request.user.is_superuser
        else PVPlant.objects.filter(owner=request.user).order_by("nome")
    )
    plants = list(qs)

    d_end = date.today()
    d_start = d_end - timedelta(days=7)

    plant_id = request.GET.get("plant_id") or request.GET.get("pk") or request.GET.get("plant_pk")
    if not plant_id and plants:
        plant_id = str(plants[0].id)

    event_classifier_version = _resolve_event_classifier_version(request.GET, default="event_rules_v2") or "event_rules_v2"
    detector_version = (request.GET.get("detector_version") or "hybrid_rules_v1")
    trained_model_version = _resolve_trained_model_version(request.GET, default="") or ""
    source_oper = request.GET.get("source_oper") or ""
    source_meteo = request.GET.get("source_meteo") or ""
    view_mode = request.GET.get("view_mode") or "full"

    start_q = request.GET.get("start")
    end_q = request.GET.get("end")

    mppt_options = [1, 2, 3, 4]
    if plant_id:
        try:
            mppt_options = _detect_available_mppts(
                plant_id=int(plant_id),
                source_oper=(source_oper or None),
                source_meteo=(source_meteo or None),
            )
        except Exception:
            logger.exception("mppt option inference failed")

    if plant_id and (not start_q or not end_q):
        try:
            plant_obj = PVPlant.objects.filter(id=int(plant_id)).first()
            if plant_obj:
                tz = _plant_tz(plant_obj)
                agg = None

                if view_mode == "full" and PlantDiagnostic15m is not None:
                    qd = PlantDiagnostic15m.objects.filter(plant_id=int(plant_id))
                    if detector_version:
                        qd = qd.filter(detector_version=detector_version)
                    if source_oper and "source_oper" in _model_field_names(PlantDiagnostic15m):
                        qd = qd.filter(source_oper__startswith=source_oper)
                    if source_meteo and "source_meteo" in _model_field_names(PlantDiagnostic15m):
                        qd = qd.filter(source_meteo=source_meteo)
                    agg = qd.aggregate(ts_min=Min("ts_utc"), ts_max=Max("ts_utc"))
                elif FaultEvent is not None:
                    q = FaultEvent.objects.filter(plant_id=int(plant_id))
                    if detector_version:
                        q = q.filter(detector_version=detector_version)
                    if source_oper:
                        q = q.filter(source_oper__startswith=source_oper)
                    if source_meteo:
                        q = q.filter(source_meteo=source_meteo)
                    agg = q.aggregate(ts_min=Min("ts_start_utc"), ts_max=Max("ts_end_utc"))

                if agg and agg["ts_max"]:
                    end_local = agg["ts_max"].astimezone(tz).date()
                    start_local = end_local - timedelta(days=7)
                    start_q = start_q or start_local.isoformat()
                    end_q = end_q or end_local.isoformat()
        except Exception:
            logger.exception("mppt_gnn_fdd_view default date inference failed")

    return render(
        request,
        "dashboard/mppt_gnn_fdd.html",
        {
            "plants": plants,
            "plant_id": plant_id,
            "start": start_q or d_start.isoformat(),
            "end": end_q or d_end.isoformat(),
            "dt_minutes": _normalize_dt_minutes(request.GET.get("dt_minutes")),
            "mppt": request.GET.get("mppt") or "all",
            "dt_options": [15, 30, 60],
            "mppt_options": mppt_options,
            "event_classifier_version": event_classifier_version,
            "trained_model_version": trained_model_version,
            "detector_version": detector_version,
            "source_oper": source_oper,
            "source_meteo": source_meteo,
            "view_mode": view_mode,
            "api_url": reverse("mppt_gnn_fdd_api"),
            "dump_url": reverse("mppt_gnn_fdd_dump_api"),
            "actions_url": reverse("mppt_gnn_fdd_actions_api"),
            "export_pdf_url": reverse("mppt_gnn_fdd_export_pdf"),
        },
    )


# ============================================================
# API 1: Heatmap
# ============================================================

@require_GET
@login_required
def mppt_gnn_fdd_api(request: HttpRequest) -> JsonResponse:
    try:
        plant_id = int(request.GET.get("plant_id") or request.GET.get("plant") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        tz = _plant_tz(plant)

        d_end = _parse_date(request.GET.get("end"), default=date.today())
        d_start = _parse_date(request.GET.get("start"), default=(d_end - timedelta(days=7)))
        if d_start > d_end:
            d_start, d_end = d_end, d_start

        dt_minutes = _normalize_dt_minutes(request.GET.get("dt_minutes"))
        mppt = _parse_int(request.GET.get("mppt"), default=0, lo=0, hi=32)
        view_mode = (request.GET.get("view_mode") or "full").strip().lower()
        if view_mode not in {"full", "events"}:
            view_mode = "full"

        event_classifier_version = _resolve_event_classifier_version(request.GET, default=None)
        trained_model_version = _resolve_trained_model_version(request.GET, default=None)
        detector_version = (request.GET.get("detector_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None
        source_meteo = (request.GET.get("source_meteo") or "").strip() or None

        dt0_local = datetime.combine(d_start, time.min, tzinfo=tz)
        dt1_local = datetime.combine(d_end + timedelta(days=1), time.min, tzinfo=tz)
        dt0_utc = dt0_local.astimezone(dt_tz.utc)
        dt1_utc = dt1_local.astimezone(dt_tz.utc)

        bpd = _bins_per_day(dt_minutes)

        days: List[str] = []
        cur = d_start
        while cur <= d_end:
            days.append(cur.isoformat())
            cur += timedelta(days=1)

        grid = [[0 for _ in range(bpd)] for _ in range(len(days))]
        tkeys: List[List[Optional[str]]] = [[None for _ in range(bpd)] for _ in range(len(days))]
        event_ids: List[List[Optional[int]]] = [[None for _ in range(bpd)] for _ in range(len(days))]
        labels: List[List[Optional[str]]] = [[None for _ in range(bpd)] for _ in range(len(days))]

        event_best_info, events, mv_list, so_list, sm_list, dv_list = _build_event_bin_map(
            plant_id=plant_id,
            tz=tz,
            dt0_utc=dt0_utc,
            dt1_utc=dt1_utc,
            d_start=d_start,
            days_len=len(days),
            bpd=bpd,
            dt_minutes=dt_minutes,
            source_oper=source_oper,
            source_meteo=source_meteo,
            detector_version=detector_version,
            event_classifier_version=event_classifier_version,
            mppt=mppt,
        )

        available_mppts = _detect_available_mppts(
            plant_id=plant_id,
            source_oper=source_oper,
            source_meteo=source_meteo,
            dt0_utc=dt0_utc,
            dt1_utc=dt1_utc,
        )

        available_common = {
            "event_classifier_versions": mv_list,
            "trained_model_versions": _list_trained_model_versions(),
            "source_opers": so_list,
            "source_meteos": sm_list,
            "detector_versions": dv_list,
            "mppt_options": available_mppts,
        }
        model_health = _load_model_health_payload(trained_model_version)
        version_summary = _build_version_summary(detector_version=detector_version, event_classifier_version=event_classifier_version, trained_model_version=trained_model_version, view_name="mppt_gnn_fdd")

        if view_mode == "events":
            event_count = len(events)

            if event_count == 0:
                avail = None
                if FaultEvent is not None:
                    qev = FaultEvent.objects.filter(plant_id=plant_id)
                    if detector_version:
                        qev = qev.filter(detector_version=detector_version)
                    if source_oper:
                        qev = qev.filter(source_oper__startswith=source_oper)
                    if source_meteo:
                        qev = qev.filter(source_meteo=source_meteo)
                    avail = qev.aggregate(ts_min=Min("ts_start_utc"), ts_max=Max("ts_end_utc"))
                return JsonResponse(
                    {
                        "ok": True,
                        "plant_id": plant_id,
                        "timezone": str(tz),
                        "view_mode": "events",
                        "start": d_start.isoformat(),
                        "end": d_end.isoformat(),
                        "dt_minutes": dt_minutes,
                        "bins_per_day": bpd,
                        "days": days,
                        "grid": grid,
                        "tkeys": tkeys,
                        "event_ids": event_ids,
                        "labels": labels,
                        "pred_count": 0,
                        "available": {
                            **available_common,
                            "event_min_utc": avail["ts_min"].isoformat() if avail and avail["ts_min"] else None,
                            "event_max_utc": avail["ts_max"].isoformat() if avail and avail["ts_max"] else None,
                        },
                        "model_health": model_health,
                        "versions": version_summary,
                        "echo": {
                            "event_classifier_version": event_classifier_version,
                            "trained_model_version": trained_model_version,
                            "detector_version": detector_version,
                            "source_oper": source_oper,
                            "source_meteo": source_meteo,
                            "mppt": mppt,
                        },
                        "hint": "Sem eventos no período/filtros atuais.",
                    },
                    status=200,
                )

            counts_by_label: Dict[str, int] = {}
            counts_by_state: Dict[str, int] = {"none": 0, "ok": 0, "fault": 0}

            for di in range(len(days)):
                for bi in range(bpd):
                    info = event_best_info.get((di, bi))
                    if not info:
                        counts_by_state["none"] += 1
                        continue

                    grid[di][bi] = int(info["state"])
                    tkeys[di][bi] = info["tkey"]
                    event_ids[di][bi] = int(info["event_id"])
                    labels[di][bi] = str(info["label"])

                    counts_by_label[str(info["label"])] = counts_by_label.get(str(info["label"]), 0) + 1
                    if int(info["state"]) == HEATMAP_STATE_OK:
                        counts_by_state["ok"] += 1
                    elif int(info["state"]) == HEATMAP_STATE_CRIT:
                        counts_by_state["fault"] += 1
                    else:
                        counts_by_state["none"] += 1

            return JsonResponse(
                {
                    "ok": True,
                    "plant_id": plant_id,
                    "timezone": str(tz),
                    "view_mode": "events",
                    "start": d_start.isoformat(),
                    "end": d_end.isoformat(),
                    "dt_minutes": dt_minutes,
                    "bins_per_day": bpd,
                    "days": days,
                    "grid": grid,
                    "tkeys": tkeys,
                    "event_ids": event_ids,
                    "labels": labels,
                    "pred_count": event_count,
                    "counts_by_label": counts_by_label,
                    "counts_by_state": counts_by_state,
                    "available": available_common,
                    "model_health": model_health,
                    "versions": version_summary,
                    "echo": {
                        "event_classifier_version": event_classifier_version,
                        "trained_model_version": trained_model_version,
                        "detector_version": detector_version,
                        "source_oper": source_oper,
                        "source_meteo": source_meteo,
                        "mppt": mppt,
                    },
                },
                status=200,
            )

        if PlantDiagnostic15m is None:
            return _error_json("PlantDiagnostic15m não está disponível.")

        # ------------------------------------------------------------
        # Coverage-first: base visual vem da tabela merged 15 min.
        # Diagnóstico 15 min entra apenas como overlay de anomalia/label.
        # ------------------------------------------------------------
        merged_names = _model_field_names(PVPlantMergedRecord15m)
        qm = PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).order_by("ts_utc")
        if source_oper and "source_oper" in merged_names:
            qm = qm.filter(source_oper__startswith=source_oper)
        if source_meteo and "source_meteo" in merged_names:
            qm = qm.filter(source_meteo=source_meteo)

        merged_fields = _existing_fields(
            PVPlantMergedRecord15m,
            [
                "ts_utc",
                "source_oper",
                "source_meteo",
                "p_ac_w",
                "p_dc_w",
                "v_ac_v",
                "i_ac_a",
                "v_dc_v",
                "i_dc_a",
                "e_ac_wh_15",
                "alarm_code",
                "alarm_sev",
                "inv_n",
                "inv_coverage",
                "flag_inv_missing",
                "flag_low_coverage",
                "mppt1_vdc_v",
                "mppt2_vdc_v",
                "mppt3_vdc_v",
                "mppt4_vdc_v",
                "mppt1_idc_a",
                "mppt2_idc_a",
                "mppt3_idc_a",
                "mppt4_idc_a",
            ],
        )
        merged_rows = list(qm.values(*merged_fields))

        qd = PlantDiagnostic15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).order_by("ts_utc")
        diag_names = _model_field_names(PlantDiagnostic15m)
        if detector_version and "detector_version" in diag_names:
            qd = qd.filter(detector_version=detector_version)
        if source_oper and "source_oper" in diag_names:
            qd = qd.filter(source_oper__startswith=source_oper)
        if source_meteo and "source_meteo" in diag_names:
            qd = qd.filter(source_meteo=source_meteo)

        diag_fields = _existing_fields(
            PlantDiagnostic15m,
            [
                "ts_utc",
                "valid",
                "anomaly_flag",
                "rca_code",
                "rca_label",
                "detector_score",
                "mismatch_rel",
                "ewma_z",
                "cusum_score",
                "stable_sky",
                "pac_real_w",
                "p_ac_real_w",
                "pac_model_w",
                "g_poa",
                "gpoa",
                "gti",
                "ghi",
                "tcell_c",
                "t_air_c",
                "temp_air_c",
                "temp_air",
                "v_ac_v",
                "vac_v",
                "i_ac_a",
                "iac_a",
                "v_dc_v",
                "vdc_total_v",
                "i_dc_a",
                "meteo_qc_score",
                "flag_meteo_missing",
                "flag_meteo_low_confidence",
                "flag_meteo_interpolated",
                "flag_meteo_outlier",
                "flag_meteo_artifact",
                "source_oper",
                "source_meteo",
                "detector_version",
            ],
        )
        diag_rows = list(qd.values(*diag_fields))

        if "source_oper" in diag_fields:
            if source_oper:
                pass
            elif mppt > 0:
                diag_rows = [
                    r for r in diag_rows
                    if _extract_mppt_index_from_source(str(r.get("source_oper") or "")) == mppt
                ]
            else:
                diag_rows = [
                    r for r in diag_rows
                    if _is_agg_source(str(r.get("source_oper") or "")) or str(r.get("source_oper") or "").strip().upper() in {"SHINEMONITOR", "GROWATT", "MANUAL"}
                ]

        if not diag_rows and not merged_rows and not event_best_info:
            qavail = PlantDiagnostic15m.objects.filter(plant_id=plant_id)
            if detector_version and "detector_version" in diag_names:
                qavail = qavail.filter(detector_version=detector_version)
            if source_oper and "source_oper" in diag_names:
                qavail = qavail.filter(source_oper__startswith=source_oper)
            if source_meteo and "source_meteo" in diag_names:
                qavail = qavail.filter(source_meteo=source_meteo)

            avail = qavail.aggregate(ts_min=Min("ts_utc"), ts_max=Max("ts_utc"))
            return JsonResponse(
                {
                    "ok": True,
                    "plant_id": plant_id,
                    "timezone": str(tz),
                    "view_mode": "full",
                    "start": d_start.isoformat(),
                    "end": d_end.isoformat(),
                    "dt_minutes": dt_minutes,
                    "bins_per_day": bpd,
                    "days": days,
                    "grid": grid,
                    "tkeys": tkeys,
                    "event_ids": event_ids,
                    "labels": labels,
                    "pred_count": 0,
                    "available": {
                        **available_common,
                        "diag_min_utc": avail["ts_min"].isoformat() if avail["ts_min"] else None,
                        "diag_max_utc": avail["ts_max"].isoformat() if avail["ts_max"] else None,
                    },
                    "model_health": model_health,
                    "versions": version_summary,
                    "echo": {
                        "event_classifier_version": event_classifier_version,
                        "trained_model_version": trained_model_version,
                        "detector_version": detector_version,
                        "source_oper": source_oper,
                        "source_meteo": source_meteo,
                        "mppt": mppt,
                    },
                    "hint": "Sem diagnósticos 15 min no período/filtros atuais.",
                },
                status=200,
            )

        counts_by_label: Dict[str, int] = {}
        counts_by_state: Dict[str, int] = {"none": 0, "ok": 0, "warn": 0, "fault": 0}
        best_score: Dict[Tuple[int, int], float] = {}
        coverage_map: Dict[Tuple[int, int], bool] = {}
        diag_context: Dict[Tuple[int, int], Dict[str, Any]] = {}

        warn_bin_map = _compute_warn_bin_map(
            diag_rows=diag_rows,
            tz=tz,
            d_start=d_start,
            days_len=len(days),
            bpd=bpd,
            dt_minutes=dt_minutes,
        )

        # Base do heatmap = cobertura operativa observada na merged
        for r in merged_rows:
            tsu = r.get("ts_utc")
            if tsu is None:
                continue

            ts_local = tsu.astimezone(tz)
            di = (ts_local.date() - d_start).days
            if not (0 <= di < len(days)):
                continue

            minutes = ts_local.hour * 60 + ts_local.minute
            bi = int(minutes // dt_minutes)
            if not (0 <= bi < bpd):
                continue

            key = (di, bi)
            has_oper = _has_useful_oper_data_from_merged_row(r)
            if not has_oper:
                continue

            coverage_map[key] = True
            grid[di][bi] = HEATMAP_STATE_OK
            if not tkeys[di][bi]:
                tkeys[di][bi] = _tkey(ts_local)
            if not labels[di][bi]:
                labels[di][bi] = "coverage_only"

        # Overlay do diagnóstico 15 min (detector plant-level)
        for r in diag_rows:
            tsu = r.get("ts_utc")
            if tsu is None:
                continue

            ts_local = tsu.astimezone(tz)
            di = (ts_local.date() - d_start).days
            if not (0 <= di < len(days)):
                continue

            minutes = ts_local.hour * 60 + ts_local.minute
            bi = int(minutes // dt_minutes)
            if not (0 <= bi < bpd):
                continue

            key = (di, bi)
            valid = bool(r.get("valid"))
            anomaly = bool(r.get("anomaly_flag"))
            rca_label = (r.get("rca_label") or "").strip().lower()
            has_oper_diag = _has_useful_oper_data_from_diag_row(r)
            has_oper = bool(coverage_map.get(key)) or has_oper_diag

            if anomaly and has_oper:
                state = HEATMAP_STATE_CRIT
                label = rca_label or "anomaly"
            elif has_oper:
                state = HEATMAP_STATE_OK
                label = rca_label or ("normal" if valid else "coverage_only")
            else:
                state = HEATMAP_STATE_NONE
                label = rca_label or ("invalid" if not valid else "no_oper_data")

            detector_score = _safe_float(r.get("detector_score"), 0.0) or 0.0
            score = (10_000_000 if state == HEATMAP_STATE_CRIT else 1_000_000) + detector_score

            prev = best_score.get(key)
            if prev is not None and score <= prev:
                continue
            best_score[key] = score
            diag_context[key] = dict(r)

            grid[di][bi] = state
            tkeys[di][bi] = _tkey(ts_local)
            labels[di][bi] = label

            einfo = event_best_info.get((di, bi))
            if einfo:
                event_ids[di][bi] = int(einfo["event_id"])

        # Overlay do event-level com política híbrida conservadora
        for key, einfo in event_best_info.items():
            di, bi = key
            cur_state = int(grid[di][bi] or 0)
            cur_label = labels[di][bi] or ""
            diag_row = diag_context.get(key)

            event_state = int(einfo.get("state") or HEATMAP_STATE_NONE)
            event_label = str(einfo.get("label") or "")
            event_conf = _safe_float(einfo.get("confidence"), None)
            meteo_allows_green = _diag_meteo_allows_green_refinement(diag_row)

            resolved_state = cur_state
            resolved_label = cur_label or event_label

            if event_state == HEATMAP_STATE_CRIT:
                resolved_state = HEATMAP_STATE_CRIT
                resolved_label = event_label or cur_label or "anomaly"
            elif event_label in BENIGN_LABELS:
                if cur_state == HEATMAP_STATE_CRIT:
                    if event_conf is not None and event_conf >= EVENT_GREEN_CONFIDENCE_MIN and meteo_allows_green:
                        resolved_state = HEATMAP_STATE_OK
                        resolved_label = event_label
                    elif event_conf is not None and event_conf >= EVENT_WARN_CONFIDENCE_MIN and meteo_allows_green:
                        resolved_state = HEATMAP_STATE_WARN
                        resolved_label = WARN_LABEL_GUARDED
                elif cur_state == HEATMAP_STATE_NONE:
                    resolved_state = HEATMAP_STATE_OK
                    resolved_label = event_label
                else:
                    resolved_label = event_label or resolved_label

            if resolved_state != cur_state or resolved_label != cur_label:
                grid[di][bi] = resolved_state
                labels[di][bi] = resolved_label

            if not event_ids[di][bi]:
                event_ids[di][bi] = int(einfo["event_id"])
            if not tkeys[di][bi]:
                tkeys[di][bi] = einfo["tkey"]

        # Overlay warn por mismatch moderado/persistente (não rebaixa bins críticos)
        for key, winfo in warn_bin_map.items():
            di, bi = key
            cur_state = int(grid[di][bi] or 0)
            if cur_state == HEATMAP_STATE_OK:
                grid[di][bi] = HEATMAP_STATE_WARN
                if not labels[di][bi] or labels[di][bi] in {"coverage_only", "normal", "curtailment_clipping", "meteo_bias"}:
                    labels[di][bi] = str(winfo.get("label") or WARN_LABEL_MISMATCH)
                if not tkeys[di][bi]:
                    tkeys[di][bi] = str(winfo.get("tkey") or "")

        for di in range(len(days)):
            for bi in range(bpd):
                state = grid[di][bi]
                label = labels[di][bi] or (
                    "normal" if state == HEATMAP_STATE_OK else
                    WARN_LABEL_MISMATCH if state == HEATMAP_STATE_WARN else
                    "anomaly" if state == HEATMAP_STATE_CRIT else
                    "sem_diagnostico"
                )
                counts_by_label[label] = counts_by_label.get(label, 0) + 1

                if state == HEATMAP_STATE_OK:
                    counts_by_state["ok"] += 1
                elif state == HEATMAP_STATE_WARN:
                    counts_by_state["warn"] += 1
                elif state == HEATMAP_STATE_CRIT:
                    counts_by_state["fault"] += 1
                else:
                    counts_by_state["none"] += 1

        return JsonResponse(
            {
                "ok": True,
                "plant_id": plant_id,
                "timezone": str(tz),
                "view_mode": "full",
                "start": d_start.isoformat(),
                "end": d_end.isoformat(),
                "dt_minutes": dt_minutes,
                "bins_per_day": bpd,
                "days": days,
                "grid": grid,
                "tkeys": tkeys,
                "event_ids": event_ids,
                "labels": labels,
                "pred_count": len(diag_rows),
                "diag_rows_total": len(diag_rows),
                "diag_rows_valid": sum(1 for r in diag_rows if bool(r.get("valid"))),
                "diag_rows_invalid": sum(1 for r in diag_rows if not bool(r.get("valid"))),
                "diag_rows_anomaly": sum(1 for r in diag_rows if bool(r.get("anomaly_flag"))),
                "merged_rows_total": len(merged_rows),
                "merged_rows_with_oper": sum(1 for r in merged_rows if _has_useful_oper_data_from_merged_row(r)),
                "counts_by_label": counts_by_label,
                "counts_by_state": counts_by_state,
                "state_policy": {
                    "name": "hybrid_warn_mismatch_v2",
                    "warn_mismatch_rel": WARN_MISMATCH_REL,
                    "warn_single_rel": WARN_MISMATCH_REL_SINGLE,
                    "warn_min_gpoa_wm2": WARN_MIN_GPOA_WM2,
                    "warn_min_qc_score": WARN_MIN_QC_SCORE,
                    "warn_allows_interpolated_meteo": True,
                    "event_green_confidence_min": EVENT_GREEN_CONFIDENCE_MIN,
                    "event_warn_confidence_min": EVENT_WARN_CONFIDENCE_MIN,
                },
                "available": available_common,
                "model_health": model_health,
                "versions": version_summary,
                "echo": {
                    "event_classifier_version": event_classifier_version,
                    "trained_model_version": trained_model_version or None,
                    "detector_version": detector_version,
                    "source_oper": source_oper,
                    "source_meteo": source_meteo,
                    "mppt": mppt,
                },
            },
            status=200,
        )

    except Exception as e:
        logger.exception("mppt_gnn_fdd_api failed")
        return _error_json(str(e), trace=traceback.format_exc())


# ============================================================
# API 2: Dump do bin / evento
# ============================================================
@require_GET
@login_required
def mppt_gnn_fdd_dump_api(request: HttpRequest) -> JsonResponse:
    try:
        plant_id = int(request.GET.get("plant_id") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        tz = _plant_tz(plant)

        mppt = _parse_int(request.GET.get("mppt"), default=0, lo=0, hi=32)
        event_classifier_version = _resolve_event_classifier_version(request.GET, default=None)
        trained_model_version = _resolve_trained_model_version(request.GET, default=None)
        detector_version = (request.GET.get("detector_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None
        source_meteo = (request.GET.get("source_meteo") or "").strip() or None

        event_id = request.GET.get("event_id")
        tkey = (request.GET.get("tkey") or request.GET.get("ts_local") or "").strip()

        event = None
        dt_local = None

        if event_id and FaultEvent is not None:
            qev = FaultEvent.objects.filter(id=int(event_id), plant_id=plant_id)
            if detector_version:
                qev = qev.filter(detector_version=detector_version)
            if source_oper:
                qev = qev.filter(source_oper__startswith=source_oper)
            if source_meteo:
                qev = qev.filter(source_meteo=source_meteo)
            event = qev.first()
            if event is not None and tkey:
                dt_local = _parse_tkey_to_local(tkey, tz)

        if event is None and tkey:
            dt_local = _parse_tkey_to_local(tkey, tz)
            if not dt_local:
                return _error_json("tkey inválido")

            found_event_id = _find_event_for_tkey(
                plant_id=plant_id,
                dt_local=dt_local,
                tz=tz,
                event_classifier_version=event_classifier_version,
                detector_version=detector_version,
                source_oper=source_oper,
                source_meteo=source_meteo,
                mppt=mppt,
            )
            if found_event_id and FaultEvent is not None:
                qev = FaultEvent.objects.filter(id=found_event_id, plant_id=plant_id)
                if detector_version:
                    qev = qev.filter(detector_version=detector_version)
                if source_oper:
                    qev = qev.filter(source_oper__startswith=source_oper)
                if source_meteo:
                    qev = qev.filter(source_meteo=source_meteo)
                event = qev.first()

        pred = None
        if event is not None:
            pred_map = _best_pred_rows_for_events([event.id], event_classifier_version=event_classifier_version, mppt=mppt)
            pred = pred_map.get(event.id)

        selected_bin: Dict[str, Any] = {}

        if event is not None and load_event_window is not None and dt_local is not None:
            try:
                win, ts_grid, meta = load_event_window(event_id=event.id, pre_bins=8, post_bins=8, n_mppt=8)
                tsu = dt_local.astimezone(dt_tz.utc)

                idx = None
                for i, t in enumerate(ts_grid):
                    if t.astimezone(dt_tz.utc) == tsu:
                        idx = i
                        break

                if idx is not None:
                    selected_bin = {
                        "ts_local": dt_local.isoformat(),
                        "ts_utc": tsu.isoformat(),
                    }

                    if getattr(win, "pac", None) is not None:
                        selected_bin["pac_w"] = float(win.pac[idx]) if win.pac[idx] == win.pac[idx] else None
                    if getattr(win, "pac_model", None) is not None:
                        selected_bin["pac_model_w"] = float(win.pac_model[idx]) if win.pac_model[idx] == win.pac_model[idx] else None
                    if getattr(win, "mismatch", None) is not None:
                        selected_bin["mismatch"] = float(win.mismatch[idx]) if win.mismatch[idx] == win.mismatch[idx] else None
                    if getattr(win, "g", None) is not None:
                        selected_bin["g_wm2"] = float(win.g[idx]) if win.g[idx] == win.g[idx] else None
                    if getattr(win, "t", None) is not None:
                        selected_bin["t_air_c"] = float(win.t[idx]) if win.t[idx] == win.t[idx] else None
                    if getattr(win, "vdc_total", None) is not None:
                        selected_bin["vdc_total_v"] = float(win.vdc_total[idx]) if win.vdc_total[idx] == win.vdc_total[idx] else None
                    if getattr(win, "iac", None) is not None:
                        selected_bin["iac_a"] = float(win.iac[idx]) if win.iac[idx] == win.iac[idx] else None

                    _append_mppt_arrays_from_window(selected_bin, win, idx)

                    if mppt > 0:
                        mppt_key = f"mppt{mppt}"
                        selected_bin["mppt_vdc_v"] = selected_bin.get(f"{mppt_key}_vdc_v")
                        selected_bin["mppt_idc_a"] = selected_bin.get(f"{mppt_key}_idc_a")
                        selected_bin["mppt_pdc_w"] = selected_bin.get(f"{mppt_key}_pdc_w")

                    if meta is not None:
                        try:
                            meta_dict = _coerce_jsonish(meta if isinstance(meta, dict) else vars(meta))
                            if isinstance(meta_dict, dict):
                                selected_bin["window_meta"] = meta_dict
                        except Exception:
                            logger.exception("failed to merge load_event_window meta")
            except Exception:
                logger.exception("load_event_window failed inside dump_api")

        if PlantDiagnostic15m is not None and dt_local is not None:
            tsu = dt_local.astimezone(dt_tz.utc)

            diag_fields = _existing_fields(
                PlantDiagnostic15m,
                DIAG_BASE_CANDIDATES + MPPT_FIELD_CANDIDATES + ["source_oper", "source_meteo"],
            )

            qdiag = PlantDiagnostic15m.objects.filter(plant_id=plant_id, ts_utc=tsu)
            diag_names = _model_field_names(PlantDiagnostic15m)
            if detector_version and "detector_version" in diag_names:
                qdiag = qdiag.filter(detector_version=detector_version)
            if source_oper and "source_oper" in diag_names:
                qdiag = qdiag.filter(source_oper__startswith=source_oper)
            if source_meteo and "source_meteo" in diag_names:
                qdiag = qdiag.filter(source_meteo=source_meteo)

            drow = qdiag.values(*diag_fields).first()

            if drow:
                diag_payload: Dict[str, Any] = {}
                for k, v in drow.items():
                    diag_payload[f"diag_{k}" if not str(k).startswith("diag_") else str(k)] = _coerce_jsonish(v)

                _merge_prefixed(selected_bin, diag_payload, "diag_")

                alias_map = {
                    "pac_real_w": "pac_w",
                    "pac_model_w": "pac_model_w",
                    "g_poa": "g_wm2",
                    "temp_air_c": "t_air_c",
                    "v_ac_v": "vac_v",
                    "i_ac_a": "iac_a",
                    "v_dc_v": "vdc_total_v",
                    "freq_hz": "fac_hz",
                }
                for src_key, dst_key in alias_map.items():
                    if src_key in drow and dst_key not in selected_bin:
                        selected_bin[dst_key] = _coerce_jsonish(drow.get(src_key))

                if not selected_bin.get("ts_local"):
                    selected_bin["ts_local"] = dt_local.isoformat()
                    selected_bin["ts_utc"] = tsu.isoformat()

        merged_snapshot = {
            "source_oper_list": [],
            "sources": {},
            "raw_operational_records": {},
            "meteo": {},
            "chosen_total": {},
            "canonical_mppt": {},
        }

        if dt_local is not None:
            tsu = dt_local.astimezone(dt_tz.utc)
            merged_snapshot = _build_merged_snapshot_for_ts(plant_id=plant_id, ts_utc=tsu)

            chosen_total = merged_snapshot.get("chosen_total") or {}
            for k, v in chosen_total.items():
                if k not in selected_bin:
                    selected_bin[k] = _coerce_jsonish(v)

            meteo_dump = merged_snapshot.get("meteo") or {}
            for k, v in meteo_dump.items():
                if k not in selected_bin:
                    selected_bin[k] = _coerce_jsonish(v)

            canonical_mppt = merged_snapshot.get("canonical_mppt") or {}
            for k, v in canonical_mppt.items():
                if v is not None:
                    selected_bin[k] = _coerce_jsonish(v)

            if mppt > 0:
                mp_tag = f"mppt{mppt}"
                for suffix in ["pac_w", "pdc_w", "vdc_v", "idc_a", "warning", "warnings", "alarm", "alarms", "status"]:
                    val = canonical_mppt.get(f"{mp_tag}_{suffix}")
                    if val is not None:
                        selected_bin[f"mppt_{suffix}"] = _coerce_jsonish(val)

        if event is None and not selected_bin and not merged_snapshot.get("sources"):
            return JsonResponse(
                {"ok": True, "found": False, "hint": "Nenhum evento, diagnóstico ou snapshot merged encontrado para esse bin."},
                status=200,
            )

        event_meta = _coerce_jsonish_deep(event.meta) if event else None
        plant_summary = event_meta.get("plant_summary", {}) if isinstance(event_meta, dict) else {}
        feature_summary = {}
        if pred and isinstance(pred.get("contribution"), dict):
            feature_summary = _coerce_jsonish_deep((pred.get("contribution") or {}).get("features") or {}) or {}

        selected_mppt_summary = {}
        if mppt > 0:
            selected_mppt_summary = {
                "mppt": mppt,
                "source_oper": selected_bin.get(f"mppt{mppt}_source_oper"),
                "pac_w": selected_bin.get("mppt_pac_w"),
                "pdc_w": selected_bin.get("mppt_pdc_w"),
                "vdc_v": selected_bin.get("mppt_vdc_v"),
                "idc_a": selected_bin.get("mppt_idc_a"),
                "warning": selected_bin.get("mppt_warning"),
                "warnings": selected_bin.get("mppt_warnings"),
                "alarm": selected_bin.get("mppt_alarm"),
                "alarms": selected_bin.get("mppt_alarms"),
                "status": selected_bin.get("mppt_status"),
            }

        mismatch_rel_bin = _safe_float(selected_bin.get("mismatch"), None)
        if mismatch_rel_bin is None:
            mismatch_rel_bin = _safe_float(selected_bin.get("diag_mismatch_rel"), None)
        warn_underperf_rel = _mismatch_underperformance(mismatch_rel_bin)

        bin_flags = {
            "diag_valid": selected_bin.get("diag_valid"),
            "diag_anomaly_flag": selected_bin.get("diag_anomaly_flag"),
            "diag_detector_version": selected_bin.get("diag_detector_version"),
            "diag_source_oper": selected_bin.get("diag_source_oper"),
            "diag_source_meteo": selected_bin.get("diag_source_meteo"),
            "flag_inv_missing": selected_bin.get("flag_inv_missing"),
            "flag_meteo_missing": selected_bin.get("flag_meteo_missing"),
            "meteo_qc_score": selected_bin.get("meteo_qc_score"),
            "flag_meteo_low_confidence": selected_bin.get("flag_meteo_low_confidence"),
            "flag_meteo_interpolated": selected_bin.get("flag_meteo_interpolated"),
            "flag_meteo_outlier": selected_bin.get("flag_meteo_outlier"),
            "flag_meteo_artifact": selected_bin.get("flag_meteo_artifact"),
            "flag_inv_missing_all": selected_bin.get("flag_inv_missing_all"),
            "flag_inv_missing_partial": selected_bin.get("flag_inv_missing_partial"),
            "inv_coverage": selected_bin.get("inv_coverage"),
            "heatmap_state_policy": "hybrid_warn_mismatch_v2",
            "warn_mismatch_rel_threshold": WARN_MISMATCH_REL,
            "warn_mismatch_rel_single": WARN_MISMATCH_REL_SINGLE,
            "warn_min_qc_score": WARN_MIN_QC_SCORE,
            "warn_allows_interpolated_meteo": True,
            "warn_event_green_confidence_min": EVENT_GREEN_CONFIDENCE_MIN,
            "warn_event_confidence_min": EVENT_WARN_CONFIDENCE_MIN,
            "warn_underperf_rel": warn_underperf_rel,
            "warn_interpolated_meteo_note": "Warn por mismatch pode permanecer ativo sob meteo 15 min interpolada quando QC local e confiança basal permitirem.",
        }

        source_summary = {
            "requested_source_oper": source_oper,
            "requested_source_meteo": source_meteo,
            "requested_detector_version": detector_version,
            "requested_event_classifier_version": event_classifier_version,
            "requested_trained_model_version": trained_model_version,
            "event_source_oper": event.source_oper if event else None,
            "event_source_meteo": event.source_meteo if event else None,
            "event_detector_version": event.detector_version if event else None,
            "prediction_source_oper": (pred or {}).get("source_oper"),
            "prediction_model_version": (pred or {}).get("model_version"),
            "available_source_oper_list": merged_snapshot.get("source_oper_list") or [],
            "chosen_sources": (merged_snapshot.get("chosen_total") or {}).get("chosen_sources"),
            "chosen_policy": (merged_snapshot.get("chosen_total") or {}).get("policy"),
        }

        event_summary = {
            "event_id": event.id if event else None,
            "status": event.status if event else None,
            "event_label_prelim": event.event_label_prelim if event else None,
            "final_label": event.final_label if event else None,
            "known_vs_unknown": event.known_vs_unknown if event else None,
            "confidence": event.confidence if event else None,
            "data_reliability_score": event.data_reliability_score if event else None,
            "data_reliability_level": event.data_reliability_level if event else None,
            "detection_confidence_score": event.detection_confidence_score if event else None,
            "detection_confidence_level": event.detection_confidence_level if event else None,
            "diagnosis_confidence_score": event.diagnosis_confidence_score if event else None,
            "diagnosis_confidence_level": event.diagnosis_confidence_level if event else None,
            "confidence_notes_json": _coerce_jsonish(event.confidence_notes_json) if event else None,
            "novelty_score": event.novelty_score if event else None,
            "severity_score": event.severity_score if event else None,
            "energy_loss_wh": event.energy_loss_wh if event else None,
            "detector_score_max": event.detector_score_max if event else None,
            "detector_score_mean": event.detector_score_mean if event else None,
            **_coerce_jsonish_deep(plant_summary),
        }

        version_summary = _build_version_summary(detector_version=detector_version, event_classifier_version=event_classifier_version, trained_model_version=trained_model_version, view_name="mppt_gnn_fdd")

        dump = {
            "plant_id": plant_id,
            "mppt": mppt,
            "event": {
                "id": event.id if event else None,
                "source_oper": event.source_oper if event else None,
                "source_meteo": event.source_meteo if event else None,
                "ts_start_utc": event.ts_start_utc.isoformat() if event else None,
                "ts_end_utc": event.ts_end_utc.isoformat() if event else None,
                "ts_start_local": event.ts_start_utc.astimezone(tz).isoformat() if event else None,
                "ts_end_local": event.ts_end_utc.astimezone(tz).isoformat() if event else None,
                "status": event.status if event else None,
                "detector_version": event.detector_version if event else None,
                "detector_score_max": event.detector_score_max if event else None,
                "detector_score_mean": event.detector_score_mean if event else None,
                "severity_score": event.severity_score if event else None,
                "energy_loss_wh": event.energy_loss_wh if event else None,
                "event_label_prelim": event.event_label_prelim if event else None,
                "final_label": event.final_label if event else None,
                "known_vs_unknown": event.known_vs_unknown if event else None,
                "confidence": event.confidence if event else None,
                "data_reliability_score": event.data_reliability_score if event else None,
                "data_reliability_level": event.data_reliability_level if event else None,
                "detection_confidence_score": event.detection_confidence_score if event else None,
                "detection_confidence_level": event.detection_confidence_level if event else None,
                "diagnosis_confidence_score": event.diagnosis_confidence_score if event else None,
                "diagnosis_confidence_level": event.diagnosis_confidence_level if event else None,
                "confidence_notes_json": _coerce_jsonish(event.confidence_notes_json) if event else None,
                "novelty_score": event.novelty_score if event else None,
                "meta": event_meta,
            },
            "mppt_pred": pred or {
                "pred_code": None,
                "pred_label": None,
                "confidence": None,
                "novelty_score": None,
                "proba": None,
                "contribution": None,
                "source_oper": source_oper,
                "mppt": mppt,
                "event_classifier_version": event_classifier_version,
            },
            "versions": version_summary,
            "event_summary": _coerce_jsonish(event_summary),
            "mppt_feature_summary": _coerce_jsonish(feature_summary),
            "selected_mppt_summary": _coerce_jsonish(selected_mppt_summary),
            "bin_flags": _coerce_jsonish(bin_flags),
            "source_summary": _coerce_jsonish(source_summary),
            "selected_bin": _coerce_jsonish(selected_bin),
            "source_oper_list": merged_snapshot.get("source_oper_list") or [],
            "sources": _coerce_jsonish(merged_snapshot.get("sources") or {}),
            "raw_operational_records": _coerce_jsonish(merged_snapshot.get("raw_operational_records") or {}),
            "meteo": _coerce_jsonish(merged_snapshot.get("meteo") or {}),
            "chosen_total": _coerce_jsonish(merged_snapshot.get("chosen_total") or {}),
            "canonical_mppt": _coerce_jsonish(merged_snapshot.get("canonical_mppt") or {}),
        }

        return JsonResponse({"ok": True, "found": True, "dump": dump, "versions": version_summary}, status=200)

    except Exception as e:
        logger.exception("mppt_gnn_fdd_dump_api failed")
        return _error_json(str(e), trace=traceback.format_exc())




def _build_export_event_rows(*, plant_id: int, tz: ZoneInfo, dt0_utc: datetime, dt1_utc: datetime, detector_version: Optional[str], source_oper: Optional[str], source_meteo: Optional[str], event_classifier_version: Optional[str], mppt: int) -> List[Dict[str, Any]]:
    if FaultEvent is None:
        return []

    q = FaultEvent.objects.filter(
        plant_id=plant_id,
        ts_start_utc__lt=dt1_utc,
        ts_end_utc__gte=dt0_utc,
    ).order_by("ts_start_utc")
    if detector_version:
        q = q.filter(detector_version=detector_version)
    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)
    if source_meteo:
        q = q.filter(source_meteo=source_meteo)

    events = list(q[:180])
    if not events:
        return []

    pred_map = _best_pred_rows_for_events(
        [int(e.id) for e in events],
        event_classifier_version=event_classifier_version,
        mppt=mppt,
    )

    rows: List[Dict[str, Any]] = []
    for ev in events:
        pr = pred_map.get(int(ev.id)) or {}
        pred_label = str(pr.get("pred_label") or "").strip()
        pred_mppt = pr.get("mppt")
        pred_txt = pred_label or "-"
        if pred_mppt:
            pred_txt = f"MPPT{pred_mppt}: {pred_txt}"
        rows.append({
            "event_id": int(ev.id),
            "start_local": ev.ts_start_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M"),
            "end_local": ev.ts_end_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M"),
            "status": str(getattr(ev, "status", "") or "-"),
            "final_label": str(getattr(ev, "final_label", "") or getattr(ev, "event_label_prelim", "") or "-"),
            "mppt_pred": pred_txt,
            "confidence": pr.get("confidence", getattr(ev, "confidence", None)),
            "event_data_reliability": getattr(ev, "data_reliability_score", None),
            "event_detection_confidence": getattr(ev, "detection_confidence_score", None),
            "event_diagnosis_confidence": getattr(ev, "diagnosis_confidence_score", None),
            "event_diagnosis_level": getattr(ev, "diagnosis_confidence_level", None),
            "severity_score": getattr(ev, "severity_score", None),
            "energy_loss_wh": getattr(ev, "energy_loss_wh", None),
        })
    return rows


@require_GET
@login_required
def mppt_gnn_fdd_export_pdf(request: HttpRequest) -> HttpResponse:
    try:
        if build_mppt_gnn_pdf_report is None:
            return HttpResponse("Serviço de geração PDF não disponível.", content_type="text/plain; charset=utf-8", status=500)

        plant_id = int(request.GET.get("plant_id") or request.GET.get("plant") or 0)
        if not plant_id:
            return HttpResponse("plant_id obrigatório", content_type="text/plain; charset=utf-8", status=400)

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return HttpResponse("Plant not found", content_type="text/plain; charset=utf-8", status=404)
        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return HttpResponse("Forbidden", content_type="text/plain; charset=utf-8", status=403)

        api_response = mppt_gnn_fdd_api(request)
        payload = json.loads(api_response.content.decode("utf-8"))
        if not payload.get("ok"):
            return HttpResponse(str(payload.get("error") or "Falha ao montar payload do relatório."), content_type="text/plain; charset=utf-8", status=400)

        tz = _plant_tz(plant)
        d_end = _parse_date(request.GET.get("end"), default=date.today())
        d_start = _parse_date(request.GET.get("start"), default=(d_end - timedelta(days=7)))
        if d_start > d_end:
            d_start, d_end = d_end, d_start
        dt0_utc = datetime.combine(d_start, time.min, tzinfo=tz).astimezone(dt_tz.utc)
        dt1_utc = datetime.combine(d_end + timedelta(days=1), time.min, tzinfo=tz).astimezone(dt_tz.utc)

        detector_version = (request.GET.get("detector_version") or "").strip() or None
        event_classifier_version = _resolve_event_classifier_version(request.GET, default=None)
        source_oper = (request.GET.get("source_oper") or "").strip() or None
        source_meteo = (request.GET.get("source_meteo") or "").strip() or None
        mppt = _parse_int(request.GET.get("mppt"), default=0, lo=0, hi=32)

        event_rows = _build_export_event_rows(
            plant_id=plant_id,
            tz=tz,
            dt0_utc=dt0_utc,
            dt1_utc=dt1_utc,
            detector_version=detector_version,
            source_oper=source_oper,
            source_meteo=source_meteo,
            event_classifier_version=event_classifier_version,
            mppt=mppt,
        )

        filters = {
            "mppt": mppt,
            "mppt_ui": request.GET.get("mppt") or "all",
            "view_mode": (request.GET.get("view_mode") or "full").strip().lower(),
            "detector_version": detector_version,
            "event_classifier_version": event_classifier_version,
            "trained_model_version": _resolve_trained_model_version(request.GET, default=None),
            "source_oper": source_oper,
            "source_meteo": source_meteo,
        }

        generated_at_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        pdf_bytes = build_mppt_gnn_pdf_report(
            plant_name=str(getattr(plant, "nome", f"Plant {plant_id}")),
            filters=filters,
            payload=payload,
            event_rows=event_rows,
            generated_at_local=generated_at_local,
            user_label=str(getattr(request.user, "username", "") or getattr(request.user, "email", "") or request.user.pk),
        )

        filename = f"mppt_gnn_fdd_report_plant{plant_id}_{d_start.isoformat()}_{d_end.isoformat()}.pdf"
        resp = HttpResponse(pdf_bytes, content_type="application/pdf")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        logger.exception("mppt_gnn_fdd_export_pdf failed")
        return HttpResponse(f"Erro ao gerar PDF: {e}", content_type="text/plain; charset=utf-8", status=500)


# ============================================================
# API 3: Actions
# ============================================================
@csrf_exempt
@require_POST
@login_required
def mppt_gnn_fdd_actions_api(request: HttpRequest) -> JsonResponse:
    try:
        body = _json_body(request)
        action = str(body.get("action") or "").strip().lower()
        if not action:
            return _error_json("action obrigatório (infer | train)")

        plant_id = int(body.get("plant_id") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        start = _parse_date(str(body.get("start") or ""), default=date.today())
        end = _parse_date(str(body.get("end") or ""), default=start)
        if start > end:
            start, end = end, start

        event_classifier_version = _resolve_event_classifier_version(body, default="event_rules_v2") or "event_rules_v2"
        trained_model_version = _resolve_trained_model_version(body, default="") or ""
        detector_version = str(body.get("detector_version") or "hybrid_rules_v1").strip()
        source_oper = str(body.get("source_oper") or "").strip()
        source_meteo = str(body.get("source_meteo") or "").strip()
        confidence_threshold = _parse_float(str(body.get("confidence_threshold") or "0.60"), 0.60)
        delete_existing = bool(int(body.get("delete_existing") or 1))
        version_summary = _build_version_summary(
            detector_version=detector_version,
            event_classifier_version=event_classifier_version,
            trained_model_version=trained_model_version,
            view_name="mppt_gnn_fdd",
        )

        tz = _plant_tz(plant)
        ts_start_utc = datetime.combine(start, time.min, tzinfo=tz).astimezone(dt_tz.utc)
        ts_end_utc = datetime.combine(end + timedelta(days=1), time.min, tzinfo=tz).astimezone(dt_tz.utc)

        if action == "infer":
            if run_detection_pipeline is None:
                return _error_json("run_detection_pipeline não disponível.")
            if infer_events_and_persist is None:
                return _error_json("infer_events_and_persist não disponível.")

            det_out = run_detection_pipeline(
                plant_id=plant_id,
                ts_start_utc=ts_start_utc,
                ts_end_utc=ts_end_utc,
                source_oper=(source_oper or None),
                source_meteo=(source_meteo or None),
                detector_version=detector_version,
                delete_existing=delete_existing,
            )

            infer_outs: List[dict] = []
            if FaultEvent is not None:
                eq = FaultEvent.objects.filter(
                    plant_id=plant_id,
                    ts_start_utc__lt=ts_end_utc,
                    ts_end_utc__gte=ts_start_utc,
                ).order_by("ts_start_utc")

                if detector_version:
                    eq = eq.filter(detector_version=detector_version)
                if source_oper:
                    eq = eq.filter(source_oper__startswith=source_oper)
                if source_meteo:
                    eq = eq.filter(source_meteo=source_meteo)

                event_ids = list(eq.values_list("id", flat=True))

                infer_outs = infer_events_and_persist(
                    plant_id=plant_id,
                    event_ids=event_ids,
                    statuses=["open", "closed", "reviewed", "dismissed"],
                    model_version=event_classifier_version,
                    confidence_threshold=confidence_threshold,
                    replace_existing=delete_existing,
                )

            return JsonResponse(
                {
                    "ok": True,
                    "action": "infer",
                    "plant_id": plant_id,
                    "event_classifier_version": event_classifier_version,
                    "trained_model_version": trained_model_version or None,
                    "detector_version": detector_version,
                    "versions": version_summary,
                    "source_oper": source_oper or None,
                    "source_meteo": source_meteo or None,
                    "events_detected": int(det_out.get("events", 0)),
                    "events_inferred": len(infer_outs),
                    "detector": det_out,
                    "details": infer_outs,
                },
                status=200,
            )

        if action == "train":
            return JsonResponse(
                {
                    "ok": True,
                    "action": "train",
                    "skipped": True,
                    "message": "O treino nesta tela usa o trained_model_version apenas para auditoria do bundle. O classificador event-level continua sendo executado via 'Detectar + Inferir'.",
                    "versions": version_summary,
                },
                status=200,
            )

        return _error_json("action inválido (use infer | train)")

    except Exception as e:
        logger.exception("mppt_gnn_fdd_actions_api failed")
        return _error_json(str(e), trace=traceback.format_exc())
