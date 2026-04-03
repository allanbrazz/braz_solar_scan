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

from core.models import PVPlant
from core.services.fdd.dashboard_common import MISMATCH_VERSION_SUMMARY, DashboardServiceError
from core.services.fdd.dashboard_runtime import build_mismatch_dashboard_payload, parse_dashboard_params

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
            "warn_abs": float(request.GET.get("warn_abs") or 0.35),
            "fault_abs": float(request.GET.get("fault_abs") or 0.90),
            "gpoa_min": float(request.GET.get("gpoa_min") or request.GET.get("gpoa_gate") or 50),
            "pmin_w": float(request.GET.get("pmin_w") or 0),
            "api_url": reverse("mismatch_fdd_api"),
            "export_pdf_url": reverse("mismatch_fdd_export_pdf"),
            "display_mode": (request.GET.get("display_mode") or "mismatch"),
            "version_summary": MISMATCH_VERSION_SUMMARY,
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
            "heatmap_mode": request.GET.get("display_mode") or payload.get("display_mode") or "mismatch",
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
