#core/views/inversor
from __future__ import annotations
from core.views._imports import *
from django.core.paginator import Paginator
from django.db import IntegrityError
# Forms
from core.forms import (
    PVInverterForm,
)

# Models
from core.models import (
    PVInverter, 
)
#---------------------------
#---------------------------  I N V E R S O R
#---------------------------

@login_required
@require_GET
def inverter_list_view(request: HttpRequest) -> HttpResponse:
    q = (request.GET.get("q") or "").strip()
    fab = (request.GET.get("fab") or "").strip()

    sort = (request.GET.get("sort") or "fabricante").strip()
    order = (request.GET.get("order") or "asc").strip().lower()

    allowed_sort = {
        "fabricante", "modelo",
        "p_ac_nom_w", "v_ac_nom_v",
        "vdc_mppt_min_v", "vdc_mppt_max_v", "vdc_abs_max_v",
        "mppt_count", "strings_por_mppt_max",
        "eficiencia_max_pct",
    }
    if sort not in allowed_sort:
        sort = "fabricante"
    if order not in {"asc", "desc"}:
        order = "asc"

    ordering = sort if order == "asc" else f"-{sort}"

    qs = PVInverter.objects.all()

    if q:
        qs = qs.filter(Q(fabricante__icontains=q) | Q(modelo__icontains=q))

    if fab:
        qs = qs.filter(fabricante=fab)

    fabricantes = list(
        PVInverter.objects.values_list("fabricante", flat=True)
        .distinct()
        .order_by("fabricante")
    )

    qs = qs.order_by(ordering, "id")

    paginator = Paginator(qs, 50)
    page_obj = paginator.get_page(request.GET.get("page") or 1)

    context = {
        "inversores": page_obj.object_list,      # <- NOME QUE O TEMPLATE ESPERA
        "fabricantes": fabricantes,
        "q": q,
        "fab": fab,
        "sort": sort,
        "order": order,
        "is_paginated": page_obj.has_other_pages(),
        "page_obj": page_obj,
    }
    return render(request, "inverters/inverter_list.html", context)

@login_required
@require_http_methods(["GET", "POST"])
def inverter_create_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = PVInverterForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    obj = form.save()
                messages.success(request, f"Inversor cadastrado: {obj.fabricante} {obj.modelo}.")
                return redirect("inverter_list")
            except IntegrityError:
                form.add_error(None, "Já existe um inversor com este fabricante e modelo.")
        # se inválido, cai e re-renderiza com erros
    else:
        form = PVInverterForm()

    return render(request, "inverters/inverter_form.html", {"form": form})

@login_required
@require_http_methods(["GET", "POST"])
def inverter_edit_view(request: HttpRequest, pk: int) -> HttpResponse:
    inv = get_object_or_404(PVInverter, pk=pk)

    if request.method == "POST":
        form = PVInverterForm(request.POST, instance=inv)
        if form.is_valid():
            try:
                with transaction.atomic():
                    obj = form.save()
                messages.success(request, f"Inversor atualizado: {obj.fabricante} {obj.modelo}.")
                return redirect("inverter_list")
            except IntegrityError:
                form.add_error(None, "Já existe um inversor com este fabricante e modelo.")
    else:
        form = PVInverterForm(instance=inv)

    return render(
        request,
        "inverters/inverter_form.html",  # pode reutilizar o mesmo template do create
        {"form": form, "is_edit": True, "inv": inv},
    )

