from __future__ import annotations

from core.views._imports import *  # mantém o padrão do projeto

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import math

from zoneinfo import ZoneInfo
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods
from django.contrib.auth.decorators import login_required

from core.models import PVPlant, FaultEvent, GroundTruthEvent
from core.services.fdd.dashboard_common import MISMATCH_VERSION_SUMMARY, DashboardServiceError
from core.services.fdd.dashboard_runtime import build_mismatch_dashboard_payload, parse_dashboard_params
from core.services.fdd.param_catalog import BASIC_PARAM_DEFAULTS, ADVANCED_PARAM_KEYS, advanced_groups, RANDOM_SEARCH_DEFAULT_TRIALS, RANDOM_SEARCH_DEFAULT_SEED
from core.services.fdd.random_search import run_typology_random_search
from core.services.fdd.validation import compute_validation_report_from_db, infer_truth_group

try:
    from core.services.fdd.report_pdf import build_mismatch_pdf_report  # type: ignore
except Exception:
    build_mismatch_pdf_report = None  # type: ignore

logger = logging.getLogger(__name__)


def _json_sanitize(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    try:
        import numpy as np  # type: ignore
        if isinstance(x, np.generic):
            return _json_sanitize(x.item())
        if isinstance(x, np.ndarray):
            return [_json_sanitize(v) for v in x.tolist()]
    except Exception:
        pass
    if is_dataclass(x):
        return _json_sanitize(asdict(x))
    return x


def _json_response_strict(payload: Any, *, status: int = 200) -> JsonResponse:
    safe = isinstance(payload, dict)
    payload = _json_sanitize(payload)
    return JsonResponse(
        payload,
        status=status,
        safe=safe,
        json_dumps_params={"ensure_ascii": False, "allow_nan": False},
    )


def _plant_queryset_for_user(user) -> Any:
    if user.is_superuser:
        return PVPlant.objects.all().order_by("nome")
    return PVPlant.objects.filter(owner=user).order_by("nome")


def _load_authorized_plant(request: HttpRequest, plant_id: int) -> PVPlant:
    plant = (
        PVPlant.objects.filter(id=plant_id)
        .select_related("details", "details__module", "details__inverter")
        .first()
    )
    if plant is None:
        raise DashboardServiceError("Planta não encontrada", status_code=404)
    if (not request.user.is_superuser) and plant.owner_id and (plant.owner_id != request.user.id):
        raise DashboardServiceError("Sem permissão para esta planta", status_code=403)
    return plant


def _parse_plant_id(data: Any) -> int:
    raw = data.get("plant_id") or data.get("plant_pk") or data.get("pk") or "0"
    try:
        return int(str(raw).strip())
    except Exception:
        raise DashboardServiceError("plant_id inválido", status_code=400)


def _build_payload_from_request(request: HttpRequest, *, allow_post: bool) -> Tuple[PVPlant, Any, Dict[str, Any]]:
    data = request.POST if allow_post and request.method == "POST" else request.GET
    plant = _load_authorized_plant(request, _parse_plant_id(data))
    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    params = parse_dashboard_params(data, tz_name)
    payload = build_mismatch_dashboard_payload(plant, params)
    return plant, params, payload




def _read_request_data(request: HttpRequest) -> Dict[str, Any]:
    if request.content_type and "application/json" in request.content_type.lower():
        try:
            body = request.body.decode("utf-8") if request.body else "{}"
            parsed = json.loads(body or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if request.method == "POST":
        return dict(request.POST.items())
    return dict(request.GET.items())


def _parse_iso_dt(value: Any, *, field_name: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise DashboardServiceError(f"{field_name} obrigatório", status_code=400)
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except Exception:
        raise DashboardServiceError(f"{field_name} inválido; use ISO-8601", status_code=400)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("UTC"))


def _update_fault_event_review(*, event: FaultEvent, review_state: str, final_label: str, notes: str, annotation_source: str, annotation_confidence: str, reviewer: str) -> FaultEvent:
    review_state = str(review_state or "pending").strip().lower()
    if review_state not in {"pending", "confirmed", "dismissed", "uncertain"}:
        raise DashboardServiceError("review_state inválido", status_code=400)

    meta = dict(event.meta or {})
    meta["review"] = {
        "review_state": review_state,
        "annotated_by": reviewer,
        "annotation_source": annotation_source or "specialist_review",
        "annotation_confidence": annotation_confidence or "B",
        "notes": notes or "",
        "reviewed_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }
    event.meta = meta

    if review_state == "dismissed":
        event.status = FaultEvent.STATUS_DISMISSED
        event.known_vs_unknown = "dismissed"
    elif review_state == "confirmed":
        event.status = FaultEvent.STATUS_REVIEWED
        event.known_vs_unknown = "known" if (final_label and final_label != "unknown") else "unknown"
    elif review_state == "uncertain":
        event.status = FaultEvent.STATUS_REVIEWED
        event.known_vs_unknown = "unknown"
    else:
        event.status = FaultEvent.STATUS_OPEN
        event.known_vs_unknown = "pending"

    if final_label:
        event.final_label = final_label
    event.save(update_fields=["status", "known_vs_unknown", "final_label", "meta", "updated_at"])
    return event


def _upsert_truth_from_review(*, event: FaultEvent, review_state: str, final_label: str, notes: str, annotation_source: str, annotation_confidence: str, reviewer: str) -> Optional[GroundTruthEvent]:
    if review_state not in {"confirmed", "uncertain", "dismissed"}:
        return None
    truth_state = {
        "confirmed": GroundTruthEvent.STATE_CONFIRMED,
        "uncertain": GroundTruthEvent.STATE_UNCERTAIN,
        "dismissed": GroundTruthEvent.STATE_DISMISSED,
    }[review_state]
    defaults = {
        "plant": event.plant,
        "source_oper": event.source_oper,
        "source_meteo": event.source_meteo,
        "detector_version": event.detector_version or "mismatch_runtime_v1",
        "ts_start_utc": event.ts_start_utc,
        "ts_end_utc": event.ts_end_utc,
        "truth_state": truth_state,
        "truth_label": final_label or event.final_label or event.event_label_prelim or "unknown",
        "truth_group": infer_truth_group(final_label or event.final_label or event.event_label_prelim or "unknown", truth_state),
        "annotation_source": annotation_source or "specialist_review",
        "annotation_confidence": annotation_confidence or "B",
        "created_by": reviewer,
        "notes": notes or "",
        "linked_fault_event": event,
        "meta": {"origin": "fault_event_review", "fault_event_id": event.id},
    }
    gt = GroundTruthEvent.objects.filter(linked_fault_event=event).first()
    if gt is None:
        gt = GroundTruthEvent.objects.create(**defaults)
    else:
        for key, value in defaults.items():
            setattr(gt, key, value)
        gt.save()
    return gt

@require_GET
@login_required
def mismatch_fdd_view(request: HttpRequest):
    plants = list(_plant_queryset_for_user(request.user))
    d_end = date.today()
    d_start = d_end - timedelta(days=7)
    plant_id = request.GET.get("plant_id") or request.GET.get("pk") or request.GET.get("plant_pk")
    if not plant_id and plants:
        plant_id = str(plants[0].id)

    return render(
        request,
        "dashboard/mismatch_fdd.html",
        {
            "plants": plants,
            "plant_id": plant_id,
            "start": request.GET.get("start") or d_start.isoformat(),
            "end": request.GET.get("end") or d_end.isoformat(),
            "dt_minutes": int(float(request.GET.get("dt_minutes") or 15)),
            "warn_abs": float(request.GET.get("warn_abs") or BASIC_PARAM_DEFAULTS["warn_abs"]),
            "fault_abs": float(request.GET.get("fault_abs") or BASIC_PARAM_DEFAULTS["fault_abs"]),
            "gpoa_min": float(request.GET.get("gpoa_min") or request.GET.get("gpoa_gate") or BASIC_PARAM_DEFAULTS["gpoa_min"]),
            "pmin_w": float(request.GET.get("pmin_w") or BASIC_PARAM_DEFAULTS["pmin_w"]),
            "api_url": reverse("mismatch_fdd_api"),
            "export_pdf_url": reverse("mismatch_fdd_export_pdf"),
            "random_search_url": reverse("mismatch_fdd_random_search_api"),
            "display_mode": (request.GET.get("display_mode") or "mismatch"),
            "version_summary": MISMATCH_VERSION_SUMMARY,
            "advanced_param_groups": advanced_groups(),
            "advanced_param_keys": ADVANCED_PARAM_KEYS,
            "random_search_defaults": {"trials": RANDOM_SEARCH_DEFAULT_TRIALS, "seed": RANDOM_SEARCH_DEFAULT_SEED},
        },
    )


@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_api(request: HttpRequest) -> JsonResponse:
    try:
        _, _, payload = _build_payload_from_request(request, allow_post=True)
        return _json_response_strict(payload, status=200)
    except DashboardServiceError as exc:
        return _json_response_strict({"ok": False, "error": exc.message}, status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_api failed")
        return _json_response_strict({"ok": False, "error": f"Erro interno: {type(exc).__name__}: {exc}"}, status=500)


@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_random_search_api(request: HttpRequest) -> JsonResponse:
    try:
        data = request.POST if request.method == "POST" else request.GET
        plant = _load_authorized_plant(request, _parse_plant_id(data))
        result = run_typology_random_search(plant=plant, base_data=data, tz_name=(getattr(plant, "timezone", "UTC") or "UTC"))
        return _json_response_strict(result, status=200)
    except DashboardServiceError as exc:
        return _json_response_strict({"ok": False, "error": exc.message}, status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_random_search_api failed")
        return _json_response_strict({"ok": False, "error": f"Erro interno: {type(exc).__name__}: {exc}"}, status=500)




@require_http_methods(["POST"])
@login_required
def mismatch_fdd_review_event_api(request: HttpRequest) -> JsonResponse:
    try:
        data = _read_request_data(request)
        event_id = int(str(data.get("fault_event_id") or data.get("event_id") or "0").strip())
        event = FaultEvent.objects.select_related("plant").filter(id=event_id).first()
        if event is None:
            raise DashboardServiceError("Evento não encontrado", status_code=404)
        _load_authorized_plant(request, event.plant_id)

        review_state = str(data.get("review_state") or "pending").strip().lower()
        final_label = str(data.get("final_label") or event.final_label or event.event_label_prelim or "").strip()
        notes = str(data.get("notes") or "").strip()
        annotation_source = str(data.get("annotation_source") or "specialist_review").strip()
        annotation_confidence = str(data.get("annotation_confidence") or "B").strip()
        reviewer = str(getattr(request.user, "username", "") or getattr(request.user, "email", "") or request.user.pk)

        event = _update_fault_event_review(
            event=event,
            review_state=review_state,
            final_label=final_label,
            notes=notes,
            annotation_source=annotation_source,
            annotation_confidence=annotation_confidence,
            reviewer=reviewer,
        )
        gt = _upsert_truth_from_review(
            event=event,
            review_state=review_state,
            final_label=final_label,
            notes=notes,
            annotation_source=annotation_source,
            annotation_confidence=annotation_confidence,
            reviewer=reviewer,
        )
        return _json_response_strict({
            "ok": True,
            "fault_event_id": event.id,
            "status": event.status,
            "known_vs_unknown": event.known_vs_unknown,
            "final_label": event.final_label,
            "ground_truth_event_id": gt.id if gt else None,
        }, status=200)
    except DashboardServiceError as exc:
        return _json_response_strict({"ok": False, "error": exc.message}, status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_review_event_api failed")
        return _json_response_strict({"ok": False, "error": f"Erro interno: {type(exc).__name__}: {exc}"}, status=500)


@require_http_methods(["POST"])
@login_required
def mismatch_fdd_create_truth_event_api(request: HttpRequest) -> JsonResponse:
    try:
        data = _read_request_data(request)
        plant = _load_authorized_plant(request, _parse_plant_id(data))
        gt_id_raw = str(data.get("ground_truth_event_id") or "").strip()
        fault_event_id_raw = str(data.get("fault_event_id") or "").strip()
        linked_fault_event = None
        if fault_event_id_raw:
            linked_fault_event = FaultEvent.objects.filter(id=int(fault_event_id_raw), plant_id=plant.id).first()

        ts_start_utc = _parse_iso_dt(data.get("ts_start_utc") or data.get("ts_start") or data.get("start"), field_name="ts_start_utc")
        ts_end_utc = _parse_iso_dt(data.get("ts_end_utc") or data.get("ts_end") or data.get("end"), field_name="ts_end_utc")
        truth_state = str(data.get("truth_state") or GroundTruthEvent.STATE_CONFIRMED).strip().lower()
        if truth_state not in {GroundTruthEvent.STATE_CONFIRMED, GroundTruthEvent.STATE_NORMAL, GroundTruthEvent.STATE_UNCERTAIN, GroundTruthEvent.STATE_DISMISSED}:
            raise DashboardServiceError("truth_state inválido", status_code=400)
        truth_label = str(data.get("truth_label") or ("normal" if truth_state == GroundTruthEvent.STATE_NORMAL else "unknown")).strip()
        annotation_source = str(data.get("annotation_source") or "specialist_review").strip()
        annotation_confidence = str(data.get("annotation_confidence") or "B").strip()
        notes = str(data.get("notes") or "").strip()
        reviewer = str(getattr(request.user, "username", "") or getattr(request.user, "email", "") or request.user.pk)
        defaults = {
            "plant": plant,
            "source_oper": str(data.get("source_oper") or getattr(linked_fault_event, "source_oper", "") or "").strip(),
            "source_meteo": str(data.get("source_meteo") or getattr(linked_fault_event, "source_meteo", "") or "").strip(),
            "detector_version": str(data.get("detector_version") or getattr(linked_fault_event, "detector_version", "mismatch_runtime_v1") or "mismatch_runtime_v1").strip(),
            "ts_start_utc": ts_start_utc,
            "ts_end_utc": ts_end_utc,
            "truth_state": truth_state,
            "truth_label": truth_label,
            "truth_group": infer_truth_group(truth_label, truth_state),
            "annotation_source": annotation_source,
            "annotation_confidence": annotation_confidence,
            "created_by": reviewer,
            "notes": notes,
            "linked_fault_event": linked_fault_event,
            "meta": {"origin": "manual_truth_event"},
        }
        if gt_id_raw:
            gt = GroundTruthEvent.objects.filter(id=int(gt_id_raw), plant_id=plant.id).first()
            if gt is None:
                raise DashboardServiceError("Ground truth event não encontrado", status_code=404)
            for key, value in defaults.items():
                setattr(gt, key, value)
            gt.save()
        else:
            gt = GroundTruthEvent.objects.create(**defaults)
        return _json_response_strict({"ok": True, "ground_truth_event_id": gt.id, "truth_state": gt.truth_state, "truth_label": gt.truth_label}, status=200)
    except DashboardServiceError as exc:
        return _json_response_strict({"ok": False, "error": exc.message}, status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_create_truth_event_api failed")
        return _json_response_strict({"ok": False, "error": f"Erro interno: {type(exc).__name__}: {exc}"}, status=500)


@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_validation_api(request: HttpRequest) -> JsonResponse:
    try:
        data = request.POST if request.method == "POST" else request.GET
        plant = _load_authorized_plant(request, _parse_plant_id(data))
        tz_name = getattr(plant, "timezone", "UTC") or "UTC"
        params = parse_dashboard_params(data, tz_name)
        detector_version = str(data.get("detector_version") or "mismatch_runtime_v1").strip()
        source_oper = str(data.get("source_oper") or data.get("src_oper") or "").strip()
        source_meteo = str(data.get("source_meteo") or data.get("src_meteo") or "").strip()
        report = compute_validation_report_from_db(
            plant_id=plant.id,
            ts_start_utc=params.dt0_utc,
            ts_end_utc=params.dt1_utc,
            detector_version=detector_version,
            source_oper=source_oper,
            source_meteo=source_meteo,
        )
        return _json_response_strict({"ok": True, "validation": report}, status=200)
    except DashboardServiceError as exc:
        return _json_response_strict({"ok": False, "error": exc.message}, status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_validation_api failed")
        return _json_response_strict({"ok": False, "error": f"Erro interno: {type(exc).__name__}: {exc}"}, status=500)


@require_GET
@login_required
def mismatch_fdd_export_pdf(request: HttpRequest) -> HttpResponse:
    try:
        if build_mismatch_pdf_report is None:
            return HttpResponse("Serviço de geração PDF não disponível.", content_type="text/plain; charset=utf-8", status=500)

        plant, params, payload = _build_payload_from_request(request, allow_post=False)
        if not payload.get("ok"):
            return HttpResponse(
                str(payload.get("error") or "Falha ao montar payload do relatório."),
                content_type="text/plain; charset=utf-8",
                status=400,
            )

        try:
            tz = ZoneInfo(getattr(plant, "timezone", "UTC") or "UTC")
        except Exception:
            tz = ZoneInfo("UTC")

        filters = {
            "warn_abs": request.GET.get("warn_abs") or payload.get("thresholds", {}).get("warn_abs"),
            "fault_abs": request.GET.get("fault_abs") or payload.get("thresholds", {}).get("fault_abs"),
            "gpoa_min": request.GET.get("gpoa_min") or request.GET.get("gpoa_gate") or payload.get("thresholds", {}).get("gpoa_gate"),
            "pmin_w": request.GET.get("pmin_w") or payload.get("thresholds", {}).get("pmin_w"),
            "dt_minutes": request.GET.get("dt_minutes") or request.GET.get("bin_minutes") or 15,
            "source_oper": request.GET.get("source_oper") or request.GET.get("src_oper") or params.source_oper_raw or None,
            "source_meteo": request.GET.get("source_meteo") or request.GET.get("src_meteo") or payload.get("sources", {}).get("source_meteo"),
            "pipeline": payload.get("pipeline"),
            "display_mode": request.GET.get("display_mode") or payload.get("display_mode") or "mismatch",
        }

        generated_at_local = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        pdf_bytes = build_mismatch_pdf_report(
            plant_name=str(getattr(plant, "nome", f"Plant {plant.id}")),
            payload=payload,
            filters=filters,
            generated_at_local=generated_at_local,
            user_label=str(getattr(request.user, "username", "") or getattr(request.user, "email", "") or request.user.pk),
        )

        filename = f"mismatch_fdd_report_plant{plant.id}_{params.start.isoformat()}_{params.end.isoformat()}.pdf"
        resp = HttpResponse(pdf_bytes, content_type="application/pdf")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except DashboardServiceError as exc:
        return HttpResponse(exc.message, content_type="text/plain; charset=utf-8", status=exc.status_code)
    except Exception as exc:
        logger.exception("mismatch_fdd_export_pdf failed")
        return HttpResponse(f"Erro ao gerar PDF: {exc}", content_type="text/plain; charset=utf-8", status=500)
