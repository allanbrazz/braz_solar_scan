#core/views/basicas.py
from __future__ import annotations
from core.views._imports import *
from core.models import PVPlantMergedRecord15m

from core.models import (
    PVPlant,
    InverterOperationalData,
    MeteoRecord
)

# -------------------------
# Views básicas (auth / home)
# -------------------------

def _user_can_manage_plant(user, plant: PVPlant) -> bool:
    # Mais seguro: apenas superuser
    if user.is_superuser:
        return True

    # Se você quiser permitir ao owner (opcional):
    return bool(plant.owner_id and plant.owner_id == user.id)

@login_required
def home(request: HttpRequest) -> HttpResponse:
    qs = PVPlant.objects.all().order_by("nome")
    if not request.user.is_superuser:
        qs = qs.filter(owner=request.user)

    # Agora inclui latitude/longitude para o mapa
    plants = list(qs.values("id", "nome", "latitude", "longitude"))

    if request.method == "POST":
        action = (request.POST.get("action") or "").strip()

        if action == "purge_plant_data":
            plant_id = request.POST.get("plant_id")
            if not plant_id:
                messages.error(request, "Selecione uma planta.")
                return redirect("home")

            try:
                plant_id_int = int(plant_id)
            except Exception:
                messages.error(request, "plant_id inválido.")
                return redirect("home")

            plant = PVPlant.objects.filter(id=plant_id_int).first()
            if not plant:
                messages.error(request, "Planta não encontrada.")
                return redirect("home")

            if not _user_can_manage_plant(request.user, plant):
                messages.error(request, "Sem permissão para excluir dados desta planta.")
                return redirect("home")

            confirm = (request.POST.get("confirm_text") or "").strip().upper()
            if confirm != "EXCLUIR":
                messages.error(request, "Confirmação inválida. Digite EXCLUIR para confirmar.")
                return redirect("home")

            purge_oper = request.POST.get("purge_oper") == "on"
            purge_meteo = request.POST.get("purge_meteo") == "on"
            purge_merged = request.POST.get("purge_merged") == "on"

            if not (purge_oper or purge_meteo or purge_merged):
                messages.error(
                    request,
                    "Selecione ao menos uma base para apagar (operativo, meteo, merged).",
                )
                return redirect("home")

            op_qs = InverterOperationalData.objects.filter(plant=plant)
            met_qs = MeteoRecord.objects.filter(plant=plant)
            merged_qs = PVPlantMergedRecord15m.objects.filter(plant=plant)

            with transaction.atomic():
                op_deleted = op_qs.count() if purge_oper else 0
                met_deleted = met_qs.count() if purge_meteo else 0
                merged_deleted = merged_qs.count() if purge_merged else 0

                if purge_oper:
                    op_qs.delete()
                if purge_meteo:
                    met_qs.delete()
                if purge_merged:
                    merged_qs.delete()

            messages.success(
                request,
                f"Dados apagados da planta '{plant.nome}': "
                f"Operativo={op_deleted}, Meteo={met_deleted}, Merged={merged_deleted}.",
            )
            return redirect("home")

        messages.error(request, "Ação inválida.")
        return redirect("home")

    return render(request, "home.html", {"plants": plants})

def signup(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Conta criada com sucesso! Faça login.")
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "registration/signup.html", {"form": form})

