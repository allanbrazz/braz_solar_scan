#core/views/plantas
from __future__ import annotations
from core.views._imports import *

# Forms
from core.forms import (
    PVPlantForm, PlantMonitoringCredentialForm,
    PVPlantDetailsForm,
    PlantCableFormSet,
    PlantCableSegmentForm,
    PVStringConfigFormSet,
)

# Models
from core.models import (
    PVPlant,
    PlantMonitoringCredential,
    PVPlantDetails,
    PVPlantStringConfig,
    PlantCableSegment,
)
#---------------------------
#---------------------------  P L A N T A S
#---------------------------

class PlantListView(LoginRequiredMixin, ListView):
    template_name = "plants/list.html"
    context_object_name = "plantas"
    paginate_by = 20

    def get_queryset(self):
        # Cada usuário vê apenas as SUAS plantas
        qs = PVPlant.objects.filter(owner=self.request.user)
        q = self.request.GET.get("q", "").strip()
        if q:
            qs = qs.filter(nome__icontains=q)
        return qs

class PlantCreateView(LoginRequiredMixin, CreateView):
    model = PVPlant
    form_class = PVPlantForm
    template_name = "plants/form.html"
    success_url = reverse_lazy("plants:list")

    def form_valid(self, form):
        form.instance.owner = self.request.user
        resp = super().form_valid(form)
        messages.success(self.request, "Planta criada com sucesso.")
        return resp


class PlantDetailView(LoginRequiredMixin, DetailView):
    model = PVPlant
    template_name = "plants/detail.html"
    context_object_name = "p"

    def get_queryset(self):
        # melhora performance e já traz configs
        return (
            PVPlant.objects
            .filter(owner=self.request.user)
            .select_related("details")
            .prefetch_related("details__string_configs")
        )

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        p: PVPlant = self.object

        d = getattr(p, "details", None)
        ctx["d"] = d

        # Se existem configs (linhas), recomputa totais em memória para exibir corretamente
        string_configs = []
        if d is not None:
            try:
                # só recalcula em memória (não grava)
                d.recompute_totals_from_configs(commit=False)
            except Exception:
                pass
            try:
                string_configs = list(d.string_configs.all())
            except Exception:
                string_configs = []

        ctx["string_configs"] = string_configs
        ctx["has_string_configs"] = bool(string_configs)

        # Form em branco para NOVA credencial
        ctx["cred_form"] = PlantMonitoringCredentialForm()

        # ======= botão Renovigi =======
        ctx["has_renovigi_cred"] = PlantMonitoringCredential.objects.filter(
            plant=p,
            provedor="RENOVIGI",
        ).exists()
        ctx["renovigi_console_url"] = reverse("renovigi_console", kwargs={"pk": p.pk})

        return ctx

class PlantDetailsEditView(LoginRequiredMixin, View):
    template_name = "plants/details_form.html"
    FORMSET_PREFIX = "strings"

    def _legacy_initial_strings(self, details: PVPlantDetails):
        """
        Se ainda não existirem PVPlantStringConfig persistidas, converte a configuração
        legada (strings_count/modules_per_string) em 1 linha inicial do formset.
        """
        if not details:
            return []
        try:
            if details.string_configs.exists():
                return []
        except Exception:
            return []

        if details.strings_count and details.modules_per_string:
            return [{
                "name": "S1",
                "strings_qty": int(details.strings_count),
                "modules_per_string": int(details.modules_per_string),
            }]
        return []

    def _build_strings_formset(self, *, details: PVPlantDetails, data=None):
        kwargs = {
            "instance": details,
            "prefix": self.FORMSET_PREFIX,
        }
        if data is not None:
            kwargs["data"] = data
        else:
            kwargs["initial"] = self._legacy_initial_strings(details)
        return PVStringConfigFormSet(**kwargs)

    def get(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        details, _ = PVPlantDetails.objects.get_or_create(plant=plant)

        form = PVPlantDetailsForm(instance=details)
        strings_formset = self._build_strings_formset(details=details)

        ctx = {"plant": plant, "pk": plant.pk, "form": form, "strings_formset": strings_formset}
        return render(request, self.template_name, ctx)

    def post(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        details, _ = PVPlantDetails.objects.get_or_create(plant=plant)

        form = PVPlantDetailsForm(request.POST, instance=details)
        strings_formset = self._build_strings_formset(details=details, data=request.POST)

        ok_form = form.is_valid()
        ok_fs = strings_formset.is_valid()

        if ok_form and ok_fs:
            with transaction.atomic():
                details_obj = form.save(commit=False)
                details_obj.plant = plant
                details_obj.save()

                strings_formset.instance = details_obj
                strings_formset.save()

                details_obj.refresh_from_db()
                if details_obj.string_configs.exists():
                    details_obj.recompute_totals_from_configs(commit=True)
                else:
                    PVPlantDetails.objects.filter(pk=details_obj.pk).update(
                        strings_count=None,
                        modules_total=None,
                        modules_per_string=None,
                    )
                    details_obj.refresh_from_db()

            messages.success(request, "Detalhes da planta salvos.")
            return redirect("plants:detail", pk=plant.pk)

        logger.warning("PVPlantDetailsForm errors: %s", form.errors.as_json())
        logger.warning("String formset errors: %s", strings_formset.errors)

        messages.error(request, "Verifique os campos destacados.")
        ctx = {"plant": plant, "pk": plant.pk, "form": form, "strings_formset": strings_formset}
        return render(request, self.template_name, ctx)

class PlantCablesEditView(LoginRequiredMixin, View):
    template_name = "plants/cables_form.html"

    def _get_plant(self, request, pk):
        return get_object_or_404(PVPlant, pk=pk, owner=request.user)

    def get(self, request, pk):
        plant = self._get_plant(request, pk)
        formset = PlantCableFormSet(instance=plant, prefix="cables")
        return render(request, self.template_name, {"plant": plant, "formset": formset})

    def post(self, request, pk):
        plant = self._get_plant(request, pk)
        formset = PlantCableFormSet(request.POST, instance=plant, prefix="cables")

        # (Opcional) botão “Adicionar linha”
        if "_addrow" in request.POST:
            # re-renderiza com um extra a mais (sem salvar ainda)
            from django.forms import inlineformset_factory
            ExtraFormSet = inlineformset_factory(
                PVPlant, PlantCableSegment, form=PlantCableSegmentForm,
                extra=len(formset.forms) + 1, can_delete=True
            )
            formset = ExtraFormSet(instance=plant, prefix="cables")
            return render(request, self.template_name, {"plant": plant, "formset": formset})

        if formset.is_valid():
            formset.save()
            messages.success(request, "Cabeamento salvo.")
            return redirect("plants:detail", pk=plant.pk)

        messages.error(request, "Corrija os erros abaixo.")
        return render(request, self.template_name, {"plant": plant, "formset": formset})
    
class PlantUpdateView(LoginRequiredMixin, UpdateView):
    model = PVPlant
    form_class = PVPlantForm
    template_name = "plants/form.html"
    success_url = reverse_lazy("plants:list")

    def get_queryset(self):
        return PVPlant.objects.filter(owner=self.request.user)

    def form_valid(self, form):
        resp = super().form_valid(form)
        messages.success(self.request, "Planta atualizada com sucesso.")
        return resp

class PlantCredSaveView(LoginRequiredMixin, View):
    def post(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        provedor_val = request.POST.get("provedor") or None

        cred = None
        if provedor_val:
            cred = PlantMonitoringCredential.objects.filter(
                plant=plant,
                provedor=provedor_val
            ).first()

        if cred is None:
            cred = PlantMonitoringCredential(plant=plant)
            if provedor_val:
                cred.provedor = provedor_val

        form = PlantMonitoringCredentialForm(request.POST, instance=cred)

        if form.is_valid():
            obj = form.save(commit=False)
            obj.plant = plant

            # defaults i18n/lang para Renovigi (se quiser)
            if obj.provedor == "RENOVIGI":
                if not obj.shinemonitor_i18n:
                    obj.shinemonitor_i18n = "pt_BR"
                if not obj.shinemonitor_lang:
                    obj.shinemonitor_lang = "pt_BR"

            obj.save()
            messages.success(request, "Credenciais salvas/atualizadas com sucesso.")

            if obj.provedor == "RENOVIGI":
                return redirect("core:renovigi_console", pk=plant.pk)

            return redirect("plants:detail", pk=plant.pk)

        messages.error(request, "Erro ao salvar credenciais.")
        return redirect("plants:detail", pk=plant.pk)
    
