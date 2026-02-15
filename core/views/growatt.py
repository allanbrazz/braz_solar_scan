#core/views/growatt
from __future__ import annotations
from core.views._imports import *

# Models
from core.models import (
    PVPlant,
)


#GROWATT
from core.services.dados_inversor.growatt_client import (
    fetch_growatt_plant_data,
    GrowattAuthError,
    GrowattReadError,
)

#---------------------------
#---------------------------  GROWATT
#---------------------------

class PlantGrowattDebugView(LoginRequiredMixin, View):
    def get(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        cred = plant.credentials.filter(provedor="GROWATT").first()
        if not cred:
            messages.error(request, "Nenhuma credencial Growatt cadastrada para esta planta.")
            return redirect("plants:detail", pk=plant.pk)

        try:
            data = fetch_growatt_plant_data(
                cred.username,
                cred.password,
                debug=True,
            )
        except GrowattAuthError as exc:
            messages.error(request, f"Erro de autenticação Growatt: {exc}")
            return redirect("plants:detail", pk=plant.pk)
        except GrowattReadError as exc:
            messages.error(request, f"Erro ao ler dados Growatt: {exc}")
            return redirect("plants:detail", pk=plant.pk)

        # só para inspecionar, devolve JSON bruto
        return JsonResponse(data, json_dumps_params={"ensure_ascii": False, "indent": 2})


class PlantGrowattDailyJsonView(LoginRequiredMixin, View):
    def get(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        cred = plant.credentials.filter(provedor="GROWATT").first()
        if not cred:
            return JsonResponse(
                {"error": "Nenhuma credencial Growatt cadastrada para esta planta."},
                status=400,
            )

        try:
            data = fetch_growatt_plant_data(
                cred.username,
                cred.password,
                debug=False,
            )
        except GrowattAuthError as exc:
            return JsonResponse({"error": f"auth_error: {exc}"}, status=401)
        except GrowattReadError as exc:
            return JsonResponse({"error": f"read_error: {exc}"}, status=502)

        return JsonResponse(data, json_dumps_params={"ensure_ascii": False})
    

