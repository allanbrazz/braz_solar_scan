#core/views/plantas
from __future__ import annotations
from core.views._imports import *

# Forms
from core.forms import (
    PVPlantForm, PlantMonitoringCredentialForm,
    PVPlantDetailsForm,     
    PlantCableFormSet,      
    PlantCableFormSet,
    PlantCableSegmentForm, 
    PVStringGroupFormSet,
)

# Models
from core.models import (
    PVPlant,
    PlantMonitoringCredential,
    PVPlantDetails,    
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

    def _initial_strings_from_details(self, details: PVPlantDetails):
        """
        Se já houver strings_count/modules_per_string, preenche 1 linha inicial.
        Se modules_total existir mas modules_per_string for None (MIX), deixa em branco (usuário preenche).
        """
        if details and details.strings_count and details.modules_per_string:
            return [{
                "label": "S1",
                "strings_qty": int(details.strings_count),
                "modules_per_string": int(details.modules_per_string),
            }]
        return []

    def get(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        details, _ = PVPlantDetails.objects.get_or_create(plant=plant)

        form = PVPlantDetailsForm(instance=details)
        strings_formset = PVStringGroupFormSet(
            prefix=self.FORMSET_PREFIX,
            initial=self._initial_strings_from_details(details),
        )

        ctx = {"plant": plant, "pk": plant.pk, "form": form, "strings_formset": strings_formset}
        return render(request, self.template_name, ctx)

    def post(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        details, _ = PVPlantDetails.objects.get_or_create(plant=plant)

        form = PVPlantDetailsForm(request.POST, instance=details)
        strings_formset = PVStringGroupFormSet(request.POST, prefix=self.FORMSET_PREFIX)

        ok_form = form.is_valid()
        ok_fs = strings_formset.is_valid()

        if ok_form and ok_fs:
            details_obj = form.save(commit=False)

            # agrega totais a partir do formset
            total_strings = 0
            total_modules = 0
            mps_vals = []

            for row in strings_formset.cleaned_data:
                if not row or row.get("DELETE"):
                    continue
                qty = int(row["strings_qty"])
                mps = int(row["modules_per_string"])
                total_strings += qty
                total_modules += qty * mps
                mps_vals.append(mps)

            # grava agregados no details
            if total_strings == 0:
                details_obj.strings_count = None
                details_obj.modules_total = None
                details_obj.modules_per_string = None
            else:
                details_obj.strings_count = total_strings
                details_obj.modules_total = total_modules

                # se todos os mps iguais, salva; senão, deixa None (MIX)
                details_obj.modules_per_string = mps_vals[0] if all(v == mps_vals[0] for v in mps_vals) else None

            details_obj.save()
            messages.success(request, "Detalhes da planta salvos.")
            return redirect("plants:detail", pk=plant.pk)

        # DEBUG: agora você enxerga exatamente o que quebrou
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
    
