from __future__ import annotations
from django import forms
import datetime as date
from django.conf import settings
from functools import lru_cache
from django.forms import inlineformset_factory 
from datetime import timedelta
from django.utils import timezone
from django.forms import formset_factory

try:
    # Python 3.9+; no Windows instale: pip install tzdata
    from zoneinfo import available_timezones
except Exception:
    available_timezones = lambda: set()

from .models import PVModule, PVPlant, PlantMonitoringCredential, PVInverter, PlantCableSegment, PVPlantDetails, PVPlantStringConfig




@lru_cache(maxsize=1)
def get_timezone_choices():
    tzs = [tz for tz in available_timezones() if "/" in tz and not tz.startswith("Etc/")]
    tzs.sort()

    preferidas = [
        "America/Montevideo", "America/Sao_Paulo", "America/Buenos_Aires",
        "America/Santiago", "America/Lima", "America/Bogota", "UTC",
    ]
    seen = set()
    ordered = [tz for tz in preferidas if tz in tzs and not (tz in seen or seen.add(tz))]
    ordered += [tz for tz in tzs if tz not in seen]

    return [(tz, tz.replace("_", " ")) for tz in ordered]

# ---------- NSRDB (PSM3 CSV) ----------
class NSRDBForm(forms.Form):
    lat = forms.FloatField(
        label="Latitude",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "any"})
    )
    lon = forms.FloatField(
        label="Longitude",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "any"})
    )
    start = forms.DateField(
        label="Início",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )
    end = forms.DateField(
        label="Fim",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )

    interval = forms.ChoiceField(
        label="Intervalo (min)",
        choices=[("30", "30"), ("60", "60")],
        initial="60",
        widget=forms.Select(attrs={"class": "form-control"})
    )

    utc = forms.BooleanField(
        label="Timestamps em UTC",
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )

    def clean(self):
        cleaned = super().clean()
        s, e = cleaned.get("start"), cleaned.get("end")
        if s and e and s > e:
            self.add_error("end", "A data final deve ser maior ou igual à data inicial.")
        return cleaned

# ---------- PV Module ----------
class PVModuleForm(forms.ModelForm):
    class Meta:
        model = PVModule
        fields = [
            "nome", "fabricante",
            "pmp_w", "vmp_v", "imp_a", "voc_v", "isc_a",
            "eficiencia_pct", "power_tolerance",
            "num_celulas",
            "temp_coeff_voc_pct_c", "temp_coeff_isc_pct_c",
            "rs_ohm", "rp_ohm", "diode_a",
        ]
        widgets = {
            "nome": forms.TextInput(attrs={"class": "form-control"}),
            "fabricante": forms.TextInput(attrs={"class": "form-control"}),
            "pmp_w": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "vmp_v": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "imp_a": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "voc_v": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "isc_a": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "eficiencia_pct": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "power_tolerance": forms.TextInput(attrs={
                "class": "form-control", "placeholder": "ex.: ±3% ou -0/+5W"
            }),
            "num_celulas": forms.NumberInput(attrs={"class": "form-control", "min": "1"}),
            "temp_coeff_voc_pct_c": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "temp_coeff_isc_pct_c": forms.NumberInput(attrs={"class": "form-control", "step": "0.001"}),
            "rs_ohm": forms.NumberInput(attrs={"class": "form-control", "step": "0.0001", "min": "0"}),
            "rp_ohm": forms.NumberInput(attrs={"class": "form-control", "step": "0.001", "min": "0"}),
            "diode_a": forms.NumberInput(attrs={"class": "form-control", "step": "0.001", "min": "1.0", "max": "2.0"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # garantia extra: qualquer campo sem 'form-control' recebe
        for f in self.fields.values():
            f.widget.attrs.setdefault("class", "form-control")

    def clean_eficiencia_pct(self):
        e = self.cleaned_data.get("eficiencia_pct")
        if e is None:
            return e
        if not (0 <= e <= 100):
            raise forms.ValidationError("Eficiência deve estar entre 0 e 100 (%).")
        return e

# ---------- PV Inverter ----------
class PVInverterForm(forms.ModelForm):
    class Meta:
        model = PVInverter
        fields = [
            "fabricante",
            "modelo",
            "p_ac_nom_w",
            "v_ac_nom_v",
            "vdc_mppt_min_v",
            "vdc_mppt_max_v",
            "vdc_abs_max_v",
            "mppt_count",
            "strings_por_mppt_max",
            "eficiencia_max_pct",
        ]
        widgets = {
            "fabricante": forms.TextInput(attrs={"class": "form-control", "placeholder": "Ex.: Growatt"}),
            "modelo": forms.TextInput(attrs={"class": "form-control", "placeholder": "Ex.: MIN 6000TL-X"}),
            "p_ac_nom_w": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "v_ac_nom_v": forms.NumberInput(attrs={"class": "form-control"}),
            "vdc_mppt_min_v": forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
            "vdc_mppt_max_v": forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
            "vdc_abs_max_v": forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
            "mppt_count": forms.NumberInput(attrs={"class": "form-control"}),
            "strings_por_mppt_max": forms.NumberInput(attrs={"class": "form-control"}),
            "eficiencia_max_pct": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
        }

    def clean(self):
        cleaned = super().clean()
        vmin = cleaned.get("vdc_mppt_min_v")
        vmax = cleaned.get("vdc_mppt_max_v")
        vabs = cleaned.get("vdc_abs_max_v")

        if vmin is not None and vmax is not None and vmin > vmax:
            self.add_error("vdc_mppt_max_v", "MPPT max deve ser >= MPPT min.")

        if vabs is not None and vmax is not None and vmax > vabs:
            self.add_error("vdc_abs_max_v", "VDC máx. absoluto deve ser >= MPPT max.")

        return cleaned

# ---------- CSV Upload ----------
class CSVUploadForm(forms.Form):
    arquivo = forms.FileField(label="Arquivo CSV")
    atualizar_existentes = forms.BooleanField(
        required=False, initial=True,
        help_text="Se marcado, atualiza registros que tenham mesmo (Nome, Fabricante)."
    )
    # Cabeçalhos esperados:
    # nome,fabricante,pmp_w,vmp_v,imp_a,voc_v,isc_a,eficiencia_pct,power_tolerance,num_celulas,temp_coeff_voc_pct_c,temp_coeff_isc_pct_c,rs_ohm,rp_ohm,diode_a


# ---------- PV Plant ----------
class PVPlantForm(forms.ModelForm):
    # transforma timezone em <select>
    timezone = forms.ChoiceField(
        label="Fuso horário",
        choices=[],
        widget=forms.Select(attrs={"class": "form-control"})
    )

    class Meta:
        model = PVPlant
        fields = ["nome", "latitude", "longitude", "timezone"]
        widgets = {
            "nome": forms.TextInput(attrs={"class": "form-control", "placeholder": "Ex.: UTEC Durazno"}),
            "latitude": forms.NumberInput(attrs={"class": "form-control", "step": "0.000001"}),
            "longitude": forms.NumberInput(attrs={"class": "form-control", "step": "0.000001"}),
            # timezone vem do ChoiceField acima
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["timezone"].choices = get_timezone_choices()
        if not self.initial.get("timezone") and not getattr(self.instance, "timezone", None):
            self.initial["timezone"] = getattr(settings, "TIME_ZONE", "UTC")

class PVPlantStringConfigEditForm(forms.ModelForm):
    class Meta:
        model = PVPlantStringConfig
        fields = ["order", "name", "mppt", "strings_qty", "modules_per_string"]
        widgets = {
            "order": forms.NumberInput(attrs={"class": "form-control", "min": 0}),
            "name": forms.TextInput(attrs={"class": "form-control", "placeholder": "Ex.: S1 / MPPT1"}),
            "mppt": forms.NumberInput(attrs={"class": "form-control", "min": 1, "placeholder": "Ex.: 1"}),
            "strings_qty": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
            "modules_per_string": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
        }
        labels = {
            "order": "Ordem",
            "name": "Nome/Label",
            "mppt": "MPPT",
            "strings_qty": "Qtd. strings",
            "modules_per_string": "Módulos/string",
        }

    def clean(self):
        cleaned = super().clean()
        mppt = cleaned.get("mppt")
        details = getattr(self.instance, "details", None)
        inverter = getattr(details, "inverter", None) if details is not None else None
        mppt_count = getattr(inverter, "mppt_count", None) if inverter is not None else None
        if mppt is not None and mppt_count is not None:
            try:
                if int(mppt) > int(mppt_count):
                    self.add_error("mppt", f"MPPT deve estar entre 1 e {int(mppt_count)} para o inversor associado.")
            except Exception:
                pass
        return cleaned


class PVPlantDetailsForm(forms.ModelForm):
    class Meta:
        model = PVPlantDetails
        fields = ["module", "inverter", "tilt_deg", "azimuth_deg", "k_sys", "noct_c"]

    strings_count = forms.IntegerField(required=False, disabled=True)
    modules_per_string = forms.IntegerField(required=False, disabled=True)
    modules_total = forms.IntegerField(required=False, disabled=True)


PVStringConfigFormSet = inlineformset_factory(
    PVPlantDetails,
    PVPlantStringConfig,
    form=PVPlantStringConfigEditForm,
    fields=("order", "name", "mppt", "strings_qty", "modules_per_string"),
    extra=1,
    can_delete=True,
)

# ---------- Credenciais de Monitoramento ----------
class PlantMonitoringCredentialForm(forms.ModelForm):
    class Meta:
        model = PlantMonitoringCredential
        fields = ["provedor", "username", "password"]
        widgets = {
            "provedor": forms.Select(attrs={"class": "form-control"}),
            "username": forms.TextInput(attrs={"class": "form-control"}),
            "password": forms.PasswordInput(render_value=True, attrs={"class": "form-control"}),
        }

class PlantCableSegmentForm(forms.ModelForm):
    class Meta:
        model = PlantCableSegment
        fields = ["segment", "description", "length_m", "cross_section_mm2", "material", "qty_parallel"]
        widgets = {
            "segment": forms.Select(attrs={"class": "form-control"}),
            "description": forms.TextInput(attrs={"class": "form-control"}),
            "length_m": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "cross_section_mm2": forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
            "material": forms.Select(attrs={"class": "form-control"}),
            "qty_parallel": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
        }


PlantCableFormSet = inlineformset_factory(
    parent_model=PVPlant,
    model=PlantCableSegment,
    form=PlantCableSegmentForm,
    extra=1,
    can_delete=True
)


PlantCableFormSet = inlineformset_factory(
    PVPlant, PlantCableSegment, form=PlantCableSegmentForm,
    extra=1, can_delete=True
)




#----------- METEO -------------
class MeteoRequestForm(forms.Form):
    plant = forms.ModelChoiceField(
        queryset=PVPlant.objects.all(),
        label="Planta",
        widget=forms.Select(attrs={"class": "form-control"})
    )

    start_date = forms.DateField(
        label="Data inicial",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )

    end_date = forms.DateField(
        label="Data final",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )

    include_gti = forms.BooleanField(
        label="Incluir GTI (POA) usando tilt/azimuth da planta",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )

    model = forms.CharField(
        label="Modelo (opcional)",
        required=False,
        initial="",
        help_text="Ex.: ERA5-Seamless, ERA5, ERA5-Land (histórico). Deixe vazio para padrão/best match.",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Opcional"})
    )

    def clean(self):
        cleaned = super().clean()
        s = cleaned.get("start_date")
        e = cleaned.get("end_date")
        if s and e and e < s:
            raise forms.ValidationError("Data final deve ser >= data inicial.")
        if s and e and (e - s).days > 370:
            raise forms.ValidationError("Intervalo muito grande. Quebre em janelas (ex.: 60-120 dias).")
        return cleaned


# ---------- RENOVIGI ----------
class ShineMonitorConsoleForm(forms.Form):
    # Credenciais (não vamos persistir por padrão)
    token = forms.CharField(label="Token", widget=forms.PasswordInput(render_value=False))
    secret = forms.CharField(label="Secret", widget=forms.PasswordInput(render_value=False))

    # Identificadores do dispositivo
    pn = forms.CharField(label="pn", max_length=40)
    devcode = forms.CharField(label="devcode", max_length=40)
    devaddr = forms.CharField(label="devaddr", max_length=16)
    sn = forms.CharField(label="sn", max_length=64)

    i18n = forms.CharField(label="i18n", max_length=16, required=False, initial="pt_BR")
    lang = forms.CharField(label="lang", max_length=8, required=False, initial="pt")

    odd_even_row = forms.CharField(
        label="oddEvenRow (opcional; use 'null' se necessário)",
        max_length=16,
        required=False,
        initial="",
    )

    # Consulta
    start = forms.DateField(label="Data inicial", widget=forms.DateInput(attrs={"type": "date"}))
    end = forms.DateField(label="Data final", widget=forms.DateInput(attrs={"type": "date"}))

    pagesize = forms.IntegerField(label="Page size", min_value=10, max_value=500, required=False, initial=50)
    max_rows = forms.IntegerField(label="Máx. linhas exibidas", min_value=50, max_value=5000, required=False, initial=500)

    def clean(self):
        cleaned = super().clean()
        start = cleaned.get("start")
        end = cleaned.get("end")
        if start and end and end < start:
            raise forms.ValidationError("A data final deve ser >= data inicial.")
        return cleaned
    

#----------- JUNTAR BASES -------------
class MergeRunForm(forms.Form):
    plant = forms.ModelChoiceField(
        label="Planta",
        queryset=PVPlant.objects.none(),
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    start_date = forms.DateField(
        label="Data inicial (local da planta)",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}),
    )
    end_date = forms.DateField(
        label="Data final (local da planta)",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}),
    )

    persist = forms.BooleanField(
        label="Persistir base casada (15 min) no banco",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    want_hourly = forms.BooleanField(
        label="Gerar também roll-up horário (preview)",
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    # Se quiser expor, mantenha; caso contrário, fixe na view.
    source_oper = forms.CharField(
        label="Fonte operativa (tag)",
        required=False,
        initial="SHINEMONITOR",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )
    source_meteo = forms.CharField(
        label="Fonte meteo (tag)",
        required=False,
        initial="OPENMETEO",
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)

        qs = PVPlant.objects.all().order_by("nome")
        if user and user.is_authenticated:
            # se você usa owner, filtra; caso contrário, remove esse filtro
            qs = qs.filter(owner=user)
        self.fields["plant"].queryset = qs

        # defaults (últimos 7 dias no fuso do servidor; serve para teste)
        if not self.initial.get("start_date") and not self.initial.get("end_date"):
            today = timezone.now().date()
            self.initial["end_date"] = today
            self.initial["start_date"] = today - timedelta(days=7)

    def clean(self):
        cleaned = super().clean()
        s = cleaned.get("start_date")
        e = cleaned.get("end_date")
        if s and e and e < s:
            raise forms.ValidationError("A data final deve ser maior ou igual à data inicial.")
        return cleaned


#----------- DASHBOARD -------------

class TimeseriesDashboardForm(forms.Form):
    plant = forms.ModelChoiceField(queryset=PVPlant.objects.none(), label="Planta")
    start_date = forms.DateField(label="Data inicial (local da planta)")
    end_date = forms.DateField(label="Data final (local da planta)")
    want_15m = forms.BooleanField(label="Alinhar em 15 min (recomendado)", required=False, initial=True)
    want_raw_inv = forms.BooleanField(label="Mostrar inversor bruto (se disponível)", required=False, initial=False)

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        qs = PVPlant.objects.all()
        if user is not None and not user.is_superuser:
            qs = qs.filter(owner=user)
        self.fields["plant"].queryset = qs.order_by("nome")

        # defaults úteis
        today = date.today()
        self.fields["start_date"].initial = today - timedelta(days=2)
        self.fields["end_date"].initial = today



OPENMETEO_MODEL_CHOICES = [
    ("", "Best Match (padrão)"),
    ("era5", "ERA5"),
    ("era5_land", "ERA5-Land"),
    ("cerra", "CERRA"),
]


class MeteoRequestForm(forms.Form):
    plant = forms.ModelChoiceField(
        queryset=PVPlant.objects.all(),
        label="Planta",
        widget=forms.Select(attrs={"class": "form-control"})
    )

    start_date = forms.DateField(
        label="Data inicial",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )

    end_date = forms.DateField(
        label="Data final",
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"})
    )

    include_gti = forms.BooleanField(
        label="Incluir GTI (POA) usando tilt/azimuth da planta",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )

    model = forms.ChoiceField(
        label="Modelo meteorológico",
        required=False,
        initial="",
        choices=OPENMETEO_MODEL_CHOICES,
        help_text="Modelo requisitado à Open-Meteo. Em branco, usa o modo padrão (Best Match).",
        widget=forms.Select(attrs={"class": "form-control"})
    )

    def clean(self):
        cleaned = super().clean()
        s = cleaned.get("start_date")
        e = cleaned.get("end_date")
        if s and e and e < s:
            raise forms.ValidationError("Data final deve ser >= data inicial.")
        if s and e and (e - s).days > 370:
            raise forms.ValidationError("Intervalo muito grande. Quebre em janelas (ex.: 60-120 dias).")
        return cleaned