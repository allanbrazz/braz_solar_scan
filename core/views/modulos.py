#core/views/modulos
from __future__ import annotations
from core.views._imports import *

# Forms
from core.forms import (
    PVModuleForm, CSVUploadForm,
)

# Models
from core.models import (
    PVModule,
)

#---------------------------
#---------------------------  MÓDULOS
#---------------------------

class ModuleListView(ListView):
    model = PVModule
    paginate_by = 20
    template_name = "pvmodules/list.html"
    context_object_name = "modulos"

    def get_queryset(self):
        qs = super().get_queryset()
        q = self.request.GET.get("q", "").strip()
        fab = self.request.GET.get("fab", "").strip()
        sort = self.request.GET.get("sort", "").strip()
        order = self.request.GET.get("order", "asc")

        if q:
            qs = qs.filter(Q(nome__icontains=q) | Q(fabricante__icontains=q))
        if fab:
            qs = qs.filter(fabricante__iexact=fab)

        allowed = {
            "fabricante": "fabricante",
            "nome": "nome",
            "pmp": "pmp_w",
            "vmp": "vmp_v",
            "imp": "imp_a",
            "voc": "voc_v",
            "isc": "isc_a",
            "eficiencia": "eficiencia_pct",
            "celulas": "num_celulas",
            "rs": "rs_ohm",
            "rp": "rp_ohm",
            "a": "diode_a",
        }
        if sort in allowed:
            key = allowed[sort]
            if order == "desc":
                key = f"-{key}"
            qs = qs.order_by(key)

        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["q"] = self.request.GET.get("q", "").strip()
        ctx["fab"] = self.request.GET.get("fab", "").strip()
        ctx["order"] = self.request.GET.get("order", "asc")
        ctx["sort"] = self.request.GET.get("sort", "")
        ctx["fabricantes"] = (
            PVModule.objects.values_list("fabricante", flat=True).distinct().order_by("fabricante")
        )
        return ctx
    
class ModuleDetailView(DetailView):
    model = PVModule
    template_name = "pvmodules/detail.html"
    context_object_name = "m"

class ModuleCreateView(CreateView):
    model = PVModule
    form_class = PVModuleForm
    template_name = "pvmodules/form.html"

    def get_success_url(self):
        messages.success(self.request, "Módulo criado com sucesso.")
        return reverse("pvmodules:list")

class ModuleUpdateView(UpdateView):
    model = PVModule
    form_class = PVModuleForm
    template_name = "pvmodules/form.html"

    def get_success_url(self):
        messages.success(self.request, "Módulo atualizado com sucesso.")
        return reverse("pvmodules:detail", args=[self.object.pk])

class CSVUploadView(FormView):
    template_name = "pvmodules/upload.html"
    form_class = CSVUploadForm
    success_url = reverse_lazy("pvmodules:list")

    expected_headers = [
        "nome","fabricante","pmp_w","vmp_v","imp_a","voc_v","isc_a",
        "eficiencia_pct","power_tolerance","num_celulas",
        "temp_coeff_voc_pct_c","temp_coeff_isc_pct_c",
        "rs_ohm","rp_ohm","diode_a"
    ]

    def form_valid(self, form):
        f = form.cleaned_data["arquivo"]
        atualizar = form.cleaned_data["atualizar_existentes"]
        decoded = f.read().decode("utf-8", errors="ignore").splitlines()
        reader = csv.DictReader(decoded)

        missing = [h for h in self.expected_headers if h not in reader.fieldnames]
        if missing:
            messages.error(self.request, f"CSV faltando cabeçalhos: {', '.join(missing)}")
            return HttpResponseRedirect(self.get_success_url())

        criados, atualizados, erros = 0, 0, 0
        for i, row in enumerate(reader, start=2):
            try:
                nome = row["nome"].strip()
                fabricante = row["fabricante"].strip()
                if not nome or not fabricante:
                    raise ValueError("Nome e Fabricante são obrigatórios.")

                defaults = {
                    "pmp_w": _to_decimal(row["pmp_w"]),
                    "vmp_v": _to_decimal(row["vmp_v"]),
                    "imp_a": _to_decimal(row["imp_a"]),
                    "voc_v": _to_decimal(row["voc_v"]),
                    "isc_a": _to_decimal(row["isc_a"]),
                    "eficiencia_pct": _to_decimal(row["eficiencia_pct"]),
                    "power_tolerance": row.get("power_tolerance", "").strip(),
                    "num_celulas": int(row["num_celulas"]),
                    "temp_coeff_voc_pct_c": _to_decimal(row["temp_coeff_voc_pct_c"]),
                    "temp_coeff_isc_pct_c": _to_decimal(row["temp_coeff_isc_pct_c"]),
                    "rs_ohm": _to_decimal(row["rs_ohm"]),
                    "rp_ohm": _to_decimal(row["rp_ohm"]),
                    "diode_a": _to_decimal(row["diode_a"]),
                }

                obj = PVModule.objects.filter(nome=nome, fabricante=fabricante).first()
                if obj:
                    if atualizar:
                        for k, v in defaults.items():
                            setattr(obj, k, v)
                        obj.full_clean()
                        obj.save()
                        atualizados += 1
                else:
                    obj = PVModule(nome=nome, fabricante=fabricante, **defaults)
                    obj.full_clean()
                    obj.save()
                    criados += 1

            except Exception as e:
                erros += 1
                messages.error(self.request, f"Linha {i}: {e}")

        if criados:
            messages.success(self.request, f"{criados} módulo(s) criado(s).")
        if atualizados:
            messages.info(self.request, f"{atualizados} módulo(s) atualizado(s).")
        if erros:
            messages.warning(self.request, f"{erros} linha(s) com erro no CSV.")
        return super().form_valid(form)

