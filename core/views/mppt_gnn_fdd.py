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
from django.http import HttpRequest, JsonResponse
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


def _error_json(msg: str, *, trace: Optional[str] = None) -> JsonResponse:
    payload: Dict[str, Any] = {"ok": False, "error": msg}
    if getattr(settings, "DEBUG", False) and trace:
        payload["trace"] = trace
    return JsonResponse(payload, status=200)


def _label_state(label: str) -> int:
    lab = (label or "").strip().lower()
    if not lab or lab in {"invalid", "no_oper_data"}:
        return 0
    if lab in BENIGN_LABELS:
        return 1
    return 2


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
    model_version: Optional[str],
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
    if model_version:
        q = q.filter(model_version=model_version)
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
    model_version: Optional[str],
    source_oper: Optional[str],
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
    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)

    events = list(q.order_by("ts_start_utc"))
    if not events:
        return None

    event_ids = [e.id for e in events]
    pred_map = _best_pred_rows_for_events(event_ids, model_version=model_version, mppt=mppt)

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
    model_version: Optional[str],
    mppt: int,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], List[dict], List[str], List[str]]:
    best_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if FaultEvent is None:
        return best_info, [], [], []

    q = FaultEvent.objects.filter(
        plant_id=plant_id,
        ts_start_utc__lt=dt1_utc,
        ts_end_utc__gte=dt0_utc,
    ).order_by("ts_start_utc")

    if source_oper:
        q = q.filter(source_oper__startswith=source_oper)

    events = list(
        q.values(
            "id",
            "ts_start_utc",
            "ts_end_utc",
            "source_oper",
            "event_label_prelim",
            "final_label",
            "severity_score",
            "confidence",
            "novelty_score",
        )
    )

    mv_list: List[str] = []
    if FaultEventMPPT is not None:
        mv_list = list(
            FaultEventMPPT.objects.filter(event__plant_id=plant_id)
            .values_list("model_version", flat=True)
            .distinct()
            .order_by("model_version")
        )

    so_list = list(
        FaultEvent.objects.filter(plant_id=plant_id)
        .values_list("source_oper", flat=True)
        .distinct()
        .order_by("source_oper")
    )

    pred_map = _best_pred_rows_for_events(
        [int(e["id"]) for e in events],
        model_version=model_version,
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
                            "tkey": _tkey(cur_bin),
                        }
            cur_bin += timedelta(minutes=dt_minutes)

    return best_info, events, mv_list, so_list


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

    model_version = request.GET.get("model_version") or "event_rules_v1"
    detector_version = request.GET.get("detector_version") or "residual_v1"
    source_oper = request.GET.get("source_oper") or ""
    source_meteo = request.GET.get("source_meteo") or ""
    view_mode = request.GET.get("view_mode") or "full"

    start_q = request.GET.get("start")
    end_q = request.GET.get("end")

    if plant_id and (not start_q or not end_q):
        try:
            plant_obj = PVPlant.objects.filter(id=int(plant_id)).first()
            if plant_obj:
                tz = _plant_tz(plant_obj)
                agg = None

                if view_mode == "full" and PlantDiagnostic15m is not None:
                    agg = PlantDiagnostic15m.objects.filter(plant_id=int(plant_id)).aggregate(
                        ts_min=Min("ts_utc"),
                        ts_max=Max("ts_utc"),
                    )
                elif FaultEvent is not None:
                    q = FaultEvent.objects.filter(plant_id=int(plant_id))
                    if source_oper:
                        q = q.filter(source_oper__startswith=source_oper)
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
            "dt_minutes": int(float(request.GET.get("dt_minutes") or 15)),
            "mppt": request.GET.get("mppt") or "all",
            "dt_options": [5, 10, 15, 30, 60],
            "mppt_options": [1, 2, 3, 4],
            "model_version": model_version,
            "detector_version": detector_version,
            "source_oper": source_oper,
            "source_meteo": source_meteo,
            "view_mode": view_mode,
            "api_url": reverse("mppt_gnn_fdd_api"),
            "dump_url": reverse("mppt_gnn_fdd_dump_api"),
            "actions_url": reverse("mppt_gnn_fdd_actions_api"),
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

        dt_minutes = _parse_int(request.GET.get("dt_minutes"), default=15, lo=5, hi=60)
        mppt = _parse_int(request.GET.get("mppt"), default=0, lo=0, hi=32)
        view_mode = (request.GET.get("view_mode") or "full").strip().lower()
        if view_mode not in {"full", "events"}:
            view_mode = "full"

        model_version = (request.GET.get("model_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None

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

        event_best_info, events, mv_list, so_list = _build_event_bin_map(
            plant_id=plant_id,
            tz=tz,
            dt0_utc=dt0_utc,
            dt1_utc=dt1_utc,
            d_start=d_start,
            days_len=len(days),
            bpd=bpd,
            dt_minutes=dt_minutes,
            source_oper=source_oper,
            model_version=model_version,
            mppt=mppt,
        )

        if view_mode == "events":
            event_count = len(events)

            if event_count == 0:
                avail = None
                if FaultEvent is not None:
                    avail = FaultEvent.objects.filter(plant_id=plant_id).aggregate(
                        ts_min=Min("ts_start_utc"),
                        ts_max=Max("ts_end_utc"),
                    )
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
                            "event_min_utc": avail["ts_min"].isoformat() if avail and avail["ts_min"] else None,
                            "event_max_utc": avail["ts_max"].isoformat() if avail and avail["ts_max"] else None,
                            "model_versions": mv_list,
                            "source_opers": so_list,
                        },
                        "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
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
                    if int(info["state"]) == 1:
                        counts_by_state["ok"] += 1
                    elif int(info["state"]) == 2:
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
                    "available": {
                        "model_versions": mv_list,
                        "source_opers": so_list,
                    },
                    "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
                },
                status=200,
            )

        if PlantDiagnostic15m is None:
            return _error_json("PlantDiagnostic15m não está disponível.")

        qd = PlantDiagnostic15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).order_by("ts_utc")

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
                "pac_real_w",
                "pac_model_w",
                "g_poa",
                "tcell_c",
            ],
        )
        diag_rows = list(qd.values(*diag_fields))

        if not diag_rows:
            avail = PlantDiagnostic15m.objects.filter(plant_id=plant_id).aggregate(
                ts_min=Min("ts_utc"),
                ts_max=Max("ts_utc"),
            )
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
                        "diag_min_utc": avail["ts_min"].isoformat() if avail["ts_min"] else None,
                        "diag_max_utc": avail["ts_max"].isoformat() if avail["ts_max"] else None,
                        "model_versions": mv_list,
                        "source_opers": so_list,
                    },
                    "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
                    "hint": "Sem diagnósticos 15 min no período/filtros atuais.",
                },
                status=200,
            )

        counts_by_label: Dict[str, int] = {}
        counts_by_state: Dict[str, int] = {"none": 0, "ok": 0, "fault": 0}
        best_score: Dict[Tuple[int, int], float] = {}

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
            anomaly = bool(r.get("anomaly_flag"))
            rca_label = (r.get("rca_label") or "").strip()

            state = 1
            label = "normal"

            if anomaly:
                state = 2
                label = rca_label or "anomaly"

            detector_score = _safe_float(r.get("detector_score"), 0.0) or 0.0
            score = (10_000_000 if state == 2 else 1_000_000) + detector_score

            prev = best_score.get(key)
            if prev is not None and score <= prev:
                continue
            best_score[key] = score

            grid[di][bi] = state
            tkeys[di][bi] = _tkey(ts_local)
            labels[di][bi] = label

            einfo = event_best_info.get((di, bi))
            if einfo:
                event_ids[di][bi] = int(einfo["event_id"])

        for di in range(len(days)):
            for bi in range(bpd):
                state = grid[di][bi]
                label = labels[di][bi] or ("normal" if state == 1 else "anomaly" if state == 2 else "no_oper_data")
                counts_by_label[label] = counts_by_label.get(label, 0) + 1

                if state == 1:
                    counts_by_state["ok"] += 1
                elif state == 2:
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
                "counts_by_label": counts_by_label,
                "counts_by_state": counts_by_state,
                "available": {
                    "model_versions": mv_list,
                    "source_opers": so_list,
                },
                "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
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
        model_version = (request.GET.get("model_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None

        event_id = request.GET.get("event_id")
        tkey = (request.GET.get("tkey") or request.GET.get("ts_local") or "").strip()

        event = None
        dt_local = None

        if event_id and FaultEvent is not None:
            event = FaultEvent.objects.filter(id=int(event_id), plant_id=plant_id).first()
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
                model_version=model_version,
                source_oper=source_oper,
                mppt=mppt,
            )
            if found_event_id and FaultEvent is not None:
                event = FaultEvent.objects.filter(id=found_event_id, plant_id=plant_id).first()

        pred = None
        if event is not None:
            pred_map = _best_pred_rows_for_events([event.id], model_version=model_version, mppt=mppt)
            pred = pred_map.get(event.id)

        selected_bin: Dict[str, Any] = {}

        # -----------------------------------------------------------------
        # 1) Window do evento
        # -----------------------------------------------------------------
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

                    if getattr(win, "vac", None) is not None:
                        selected_bin["vac_v"] = float(win.vac[idx]) if win.vac[idx] == win.vac[idx] else None

                    if getattr(win, "fac", None) is not None:
                        selected_bin["fac_hz"] = float(win.fac[idx]) if win.fac[idx] == win.fac[idx] else None

                    if getattr(win, "qac", None) is not None:
                        selected_bin["inv_qac_var"] = float(win.qac[idx]) if win.qac[idx] == win.qac[idx] else None

                    if getattr(win, "pf", None) is not None:
                        selected_bin["inv_pf"] = float(win.pf[idx]) if win.pf[idx] == win.pf[idx] else None

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
                                if "warning" in meta_dict:
                                    selected_bin["inv_warning"] = meta_dict.get("warning")
                                if "warnings" in meta_dict:
                                    selected_bin["inv_warnings"] = meta_dict.get("warnings")
                                if "alarm" in meta_dict:
                                    selected_bin["inv_alarm"] = meta_dict.get("alarm")
                                if "alarms" in meta_dict:
                                    selected_bin["inv_alarms"] = meta_dict.get("alarms")
                                if "status" in meta_dict:
                                    selected_bin["inv_status"] = meta_dict.get("status")
                                if "mode" in meta_dict:
                                    selected_bin["inv_mode"] = meta_dict.get("mode")
                        except Exception:
                            logger.exception("failed to merge load_event_window meta")
            except Exception:
                logger.exception("load_event_window failed inside dump_api")

        # -----------------------------------------------------------------
        # 2) PlantDiagnostic15m dinâmica
        # -----------------------------------------------------------------
        if PlantDiagnostic15m is not None and dt_local is not None:
            tsu = dt_local.astimezone(dt_tz.utc)

            diag_fields = _existing_fields(
                PlantDiagnostic15m,
                DIAG_BASE_CANDIDATES + MPPT_FIELD_CANDIDATES,
            )

            drow = (
                PlantDiagnostic15m.objects.filter(plant_id=plant_id, ts_utc=tsu)
                .values(*diag_fields)
                .first()
            )

            if drow:
                diag_payload: Dict[str, Any] = {}
                for k, v in drow.items():
                    diag_payload[f"diag_{k}" if not str(k).startswith("diag_") else str(k)] = _coerce_jsonish(v)

                _merge_prefixed(selected_bin, diag_payload, "diag_")

                alias_map = {
                    "pac_real_w": "pac_w",
                    "pac_model_w": "pac_model_w",
                    "p_ac_real_w": "pac_w",
                    "p_ac_model_w": "pac_model_w",
                    "g_poa": "g_wm2",
                    "gpoa": "g_wm2",
                    "temp_air_c": "t_air_c",
                    "temp_air": "t_air_c",
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

        # -----------------------------------------------------------------
        # 3) Snapshot merged_15m
        # -----------------------------------------------------------------
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
            merged_snapshot = _build_merged_snapshot_for_ts(
                plant_id=plant_id,
                ts_utc=tsu,
            )

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
                if canonical_mppt.get(f"{mp_tag}_pac_w") is not None:
                    selected_bin["mppt_pac_w"] = _coerce_jsonish(canonical_mppt.get(f"{mp_tag}_pac_w"))
                if canonical_mppt.get(f"{mp_tag}_pdc_w") is not None:
                    selected_bin["mppt_pdc_w"] = _coerce_jsonish(canonical_mppt.get(f"{mp_tag}_pdc_w"))
                if canonical_mppt.get(f"{mp_tag}_vdc_v") is not None:
                    selected_bin["mppt_vdc_v"] = _coerce_jsonish(canonical_mppt.get(f"{mp_tag}_vdc_v"))
                if canonical_mppt.get(f"{mp_tag}_idc_a") is not None:
                    selected_bin["mppt_idc_a"] = _coerce_jsonish(canonical_mppt.get(f"{mp_tag}_idc_a"))

        if event is None and not selected_bin and not merged_snapshot.get("sources"):
            return JsonResponse(
                {"ok": True, "found": False, "hint": "Nenhum evento, diagnóstico ou snapshot merged encontrado para esse bin."},
                status=200,
            )

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
                "novelty_score": event.novelty_score if event else None,
                "meta": _coerce_jsonish(event.meta) if event else None,
            },
            "mppt_pred": pred or {
                "pred_code": None,
                "pred_label": None,
                "confidence": None,
                "novelty_score": None,
                "proba": None,
                "contribution": None,
                "model_version": model_version,
                "source_oper": source_oper,
                "mppt": mppt,
            },
            "selected_bin": _coerce_jsonish(selected_bin),
            "source_oper_list": merged_snapshot.get("source_oper_list") or [],
            "sources": _coerce_jsonish(merged_snapshot.get("sources") or {}),
            "raw_operational_records": _coerce_jsonish(merged_snapshot.get("raw_operational_records") or {}),
            "meteo": _coerce_jsonish(merged_snapshot.get("meteo") or {}),
            "chosen_total": _coerce_jsonish(merged_snapshot.get("chosen_total") or {}),
            "canonical_mppt": _coerce_jsonish(merged_snapshot.get("canonical_mppt") or {}),
        }

        return JsonResponse({"ok": True, "found": True, "dump": dump}, status=200)

    except Exception as e:
        logger.exception("mppt_gnn_fdd_dump_api failed")
        return _error_json(str(e), trace=traceback.format_exc())


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

        model_version = str(body.get("model_version") or "event_rules_v1").strip()
        detector_version = str(body.get("detector_version") or "residual_v1").strip()
        source_oper = str(body.get("source_oper") or "").strip()
        source_meteo = str(body.get("source_meteo") or "").strip()
        confidence_threshold = _parse_float(str(body.get("confidence_threshold") or "0.60"), 0.60)
        delete_existing = bool(int(body.get("delete_existing") or 1))

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

                if source_oper:
                    eq = eq.filter(source_oper__startswith=source_oper)

                event_ids = list(eq.values_list("id", flat=True))

                infer_outs = infer_events_and_persist(
                    plant_id=plant_id,
                    event_ids=event_ids,
                    statuses=["open", "closed", "reviewed", "dismissed"],
                    model_version=model_version,
                    confidence_threshold=confidence_threshold,
                    replace_existing=delete_existing,
                )

            return JsonResponse(
                {
                    "ok": True,
                    "action": "infer",
                    "plant_id": plant_id,
                    "model_version": model_version,
                    "detector_version": detector_version,
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
                    "message": "No modo event_rules_v1 o treino está desabilitado. Use o botão 'Detectar + Inferir'.",
                },
                status=200,
            )

        return _error_json("action inválido (use infer | train)")

    except Exception as e:
        logger.exception("mppt_gnn_fdd_actions_api failed")
        return _error_json(str(e), trace=traceback.format_exc())