#core/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import timezone
from decimal import Decimal
from django.db.models import Q, F

#---------------------------
#--------------------------- P     M
#---------------------------   V     O
#---------------------------

class PVModule(models.Model):
    """
    Modelo para armazenar especificações de um módulo fotovoltaico.
    Unicidade: (nome, fabricante)
    """
    nome = models.CharField("Nome", max_length=120)
    fabricante = models.CharField("Fabricante", max_length=120)

    # Potência nominal no MPP (W)
    pmp_w = models.DecimalField("Pmp (W)", max_digits=10, decimal_places=2,
                                validators=[MinValueValidator(0)])

    # Ponto de máxima potência (V, A)
    vmp_v = models.DecimalField("Vmp (V)", max_digits=8, decimal_places=3,
                                validators=[MinValueValidator(0)])
    imp_a = models.DecimalField("Imp (A)", max_digits=8, decimal_places=3,
                                validators=[MinValueValidator(0)])

    # Circuito aberto / curto-circuito (V, A)
    voc_v = models.DecimalField("Voc (V)", max_digits=8, decimal_places=3,
                                validators=[MinValueValidator(0)])
    isc_a = models.DecimalField("Isc (A)", max_digits=8, decimal_places=3,
                                validators=[MinValueValidator(0)])

    # Eficiência em % (0–100)
    eficiencia_pct = models.DecimalField("Eficiência (%)", max_digits=5, decimal_places=2,
                                         validators=[MinValueValidator(0), MaxValueValidator(100)])

    # Tolerância de potência – texto livre para suportar “±3%”, “-0/+5W” etc.
    power_tolerance = models.CharField("Power Tolerance", max_length=32, blank=True, default="")

    # Número de células
    num_celulas = models.PositiveSmallIntegerField("Número de células",
                                                   validators=[MinValueValidator(1)])

    # Coeficientes de temperatura em %/°C (Voc tende a negativo; Isc tende a positivo)
    temp_coeff_voc_pct_c = models.DecimalField(
        "Temperature Coefficient (Voc) (%/°C)", max_digits=6, decimal_places=3)
    temp_coeff_isc_pct_c = models.DecimalField(
        "Temperature Coefficient (Isc) (%/°C)", max_digits=6, decimal_places=3)

    # Parâmetros do modelo elétrico
    rs_ohm = models.DecimalField("Resistência série Rs (Ω)", max_digits=8, decimal_places=4,
                                 validators=[MinValueValidator(0)])
    rp_ohm = models.DecimalField("Resistência paralelo Rp (Ω)", max_digits=10, decimal_places=3,
                                 validators=[MinValueValidator(0)])
    diode_a = models.DecimalField("Fator de idealidade (a)", max_digits=4, decimal_places=3,
                                  validators=[MinValueValidator(0.5), MaxValueValidator(2.5)])

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["nome", "fabricante"], name="uniq_nome_fabricante")
        ]
        ordering = ["fabricante", "nome"]

    def __str__(self):
        return f"{self.fabricante} — {self.nome}"


#---------------------------
#--------------------------- I N V E R T E R
#---------------------------

class PVInverter(models.Model):
    fabricante = models.CharField("Fabricante", max_length=120)
    modelo = models.CharField("Modelo", max_length=120)

    p_ac_nom_w = models.DecimalField("Potência AC nominal (W)", max_digits=10, decimal_places=2,
                                     validators=[MinValueValidator(0)])
    v_ac_nom_v = models.PositiveIntegerField("Tensão AC nominal (V)", default=230)

    vdc_mppt_min_v = models.DecimalField("MPPT min (V)", max_digits=8, decimal_places=1, null=True, blank=True)
    vdc_mppt_max_v = models.DecimalField("MPPT max (V)", max_digits=8, decimal_places=1, null=True, blank=True)
    vdc_abs_max_v = models.DecimalField("VDC máx. absoluto (V)", max_digits=8, decimal_places=1, null=True, blank=True)

    mppt_count = models.PositiveSmallIntegerField("Qtd MPPT", default=1)
    strings_por_mppt_max = models.PositiveSmallIntegerField("Strings/MPPT (máx.)", default=1)

    eficiencia_max_pct = models.DecimalField("Eficiência máx. (%)", max_digits=5, decimal_places=2,
                                             null=True, blank=True,
                                             validators=[MinValueValidator(0), MaxValueValidator(100)])

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["fabricante", "modelo"]
        constraints = [
            models.UniqueConstraint(fields=("fabricante", "modelo"), name="uniq_inverter_fabricante_modelo")
        ]

    def __str__(self):
        return f"{self.fabricante} {self.modelo}"


#---------------------------
#--------------------------- P     P
#---------------------------   V     L
#---------------------------

class PVPlant(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              null=True, blank=True, related_name="pvplants")
    nome = models.CharField("Nome da planta", max_length=120)
    latitude = models.DecimalField("Latitude", max_digits=9, decimal_places=6,
                                   validators=[MinValueValidator(-90), MaxValueValidator(90)])
    longitude = models.DecimalField("Longitude", max_digits=9, decimal_places=6,
                                    validators=[MinValueValidator(-180), MaxValueValidator(180)])
    timezone = models.CharField("Fuso horário", max_length=64, default=settings.TIME_ZONE)
    # (sem module/inverter/strings/tilt/azimuth aqui!)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    ...
    def __str__(self):
        return self.nome





class PVPlantDetails(models.Model):
    plant = models.OneToOneField("PVPlant", on_delete=models.CASCADE, related_name="details")

    # associações
    module = models.ForeignKey(
        "PVModule", on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="plant_details",
    )
    inverter = models.ForeignKey(
        "PVInverter", on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="plant_details",
    )

    # configuração elétrica (DERIVADA quando existir string_configs)
    strings_count = models.PositiveIntegerField(
        "Strings (qtd)", null=True, blank=True,
        validators=[MinValueValidator(1)],
    )
    # NOTA: quando houver configs heterogêneas, isso vira None
    modules_per_string = models.PositiveIntegerField(
        "Módulos por string", null=True, blank=True,
        validators=[MinValueValidator(1)],
    )
    modules_total = models.PositiveIntegerField(
        "Módulos totais", null=True, blank=True,
        validators=[MinValueValidator(1)],
    )

    tilt_deg = models.DecimalField(
        "Inclinação (graus)", max_digits=5, decimal_places=2,
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(90)],
    )
    azimuth_deg = models.DecimalField(
        "Azimute (graus)", max_digits=6, decimal_places=2,
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(359)],
    )
    k_sys = models.DecimalField(
        "k_sys (DC→AC)", max_digits=6, decimal_places=3,
        default=Decimal("0.900"),
        validators=[MinValueValidator(Decimal("0.5")), MaxValueValidator(Decimal("1.2"))],
    )
    noct_c = models.DecimalField(
        "TNOCT (°C)", max_digits=5, decimal_places=2,
        default=Decimal("45.00"),
        validators=[MinValueValidator(Decimal("20")), MaxValueValidator(Decimal("70"))],
    )

    updated_at = models.DateTimeField(auto_now=True)

    # ------------ DERIVAÇÕES ------------

    def recompute_totals_from_configs(self, commit: bool = True) -> None:
        """
        Se existirem configs (PVPlantStringConfig), deriva:
        - strings_count = soma(strings_qty)
        - modules_total = soma(strings_qty * modules_per_string)
        - modules_per_string = valor único se todas configs tiverem mesmo modules_per_string (senão None)
        """
        cfgs = list(self.string_configs.all())
        if not cfgs:
            return

        strings_count = sum(int(c.strings_qty or 0) for c in cfgs)
        modules_total = sum(int(c.strings_qty or 0) * int(c.modules_per_string or 0) for c in cfgs)

        mps_set = {int(c.modules_per_string) for c in cfgs if c.modules_per_string}
        modules_per_string = (mps_set.pop() if len(mps_set) == 1 else None)

        self.strings_count = strings_count or None
        self.modules_total = modules_total or None
        self.modules_per_string = modules_per_string  # pode ser None se heterogêneo

        if commit and self.pk:
            PVPlantDetails.objects.filter(pk=self.pk).update(
                strings_count=self.strings_count,
                modules_total=self.modules_total,
                modules_per_string=self.modules_per_string,
            )

    def clean(self):
        super().clean()

        has_cfg = bool(self.pk) and self.string_configs.exists()

        if has_cfg:
            # configs são fonte de verdade: derive e valide apenas consistências
            self.recompute_totals_from_configs(commit=False)

            # ✅ ACEITA heterogêneo: modules_per_string pode ser None.
            # O que ainda precisa bater sempre:
            # modules_total == soma(strings_qty * modules_per_string) (já derivado)
            # strings_count == soma(strings_qty) (já derivado)
            #
            # Aqui só validamos coerência caso o usuário tenha preenchido manualmente
            # algo que conflite — mas, como derivamos, normalmente não conflita.
            if self.strings_count is not None and int(self.strings_count) < 1:
                raise ValidationError({"strings_count": "Deve ser >= 1."})
            if self.modules_total is not None and int(self.modules_total) < 1:
                raise ValidationError({"modules_total": "Deve ser >= 1."})

            return

        # -------- Modo “simples/legado” (sem configs) --------
        if self.strings_count and self.modules_per_string:
            expected = int(self.strings_count) * int(self.modules_per_string)
            if self.modules_total and int(self.modules_total) != expected:
                raise ValidationError({"modules_total": f"Deve ser {expected} (= strings × módulos/string)."})
        # se o usuário preencher modules_total sozinho sem os outros, não dá p/ validar

    def save(self, *args, **kwargs):
        """
        Regras:
        - Se existirem configs: NÃO tenta recalcular aqui (deixa para recompute_totals_from_configs
          ser chamado por você após salvar configs, ou por signals/admin).
        - Se não existirem configs: mantém comportamento atual (strings_count × modules_per_string).
        """
        has_cfg = bool(self.pk) and getattr(self, "string_configs", None) and self.string_configs.exists()

        if not has_cfg:
            if self.strings_count and self.modules_per_string:
                self.modules_total = int(self.strings_count) * int(self.modules_per_string)

        super().save(*args, **kwargs)


class PVPlantStringConfig(models.Model):
    """
    Linha configurável: permite “agregar quantas strings quiser”.
    Cada linha pode representar 1 ou várias strings com mesmo modules_per_string.

    mppt é opcional (planejamento futuro).
    """
    details = models.ForeignKey(PVPlantDetails, on_delete=models.CASCADE, related_name="string_configs")

    name = models.CharField("Nome", max_length=60, blank=True, default="")
    order = models.PositiveSmallIntegerField("Ordem", default=0)

    mppt = models.PositiveSmallIntegerField(
        "MPPT (opcional)", null=True, blank=True,
        validators=[MinValueValidator(1)],
    )
    strings_qty = models.PositiveIntegerField(
        "Qtd de strings", default=1,
        validators=[MinValueValidator(1)],
    )
    modules_per_string = models.PositiveIntegerField(
        "Módulos por string",
        validators=[MinValueValidator(1)],
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order", "id"]
        indexes = [models.Index(fields=["details"])]

    def clean(self):
        super().clean()
        if self.strings_qty is None or int(self.strings_qty) < 1:
            raise ValidationError({"strings_qty": "Deve ser >= 1."})
        if self.modules_per_string is None or int(self.modules_per_string) < 1:
            raise ValidationError({"modules_per_string": "Deve ser >= 1."})

    def _sync_parent_details(self) -> None:
        details = getattr(self, "details", None)
        if not details or not details.pk:
            return
        qs = details.string_configs.all()
        if qs.exists():
            details.recompute_totals_from_configs(commit=True)
        else:
            PVPlantDetails.objects.filter(pk=details.pk).update(
                strings_count=None,
                modules_total=None,
                modules_per_string=None,
            )

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self._sync_parent_details()

    def delete(self, *args, **kwargs):
        details = getattr(self, "details", None)
        super().delete(*args, **kwargs)
        if details and details.pk:
            qs = details.string_configs.all()
            if qs.exists():
                details.recompute_totals_from_configs(commit=True)
            else:
                PVPlantDetails.objects.filter(pk=details.pk).update(
                    strings_count=None,
                    modules_total=None,
                    modules_per_string=None,
                )

    def __str__(self) -> str:
        label = self.name.strip() or f"Config #{self.pk or 'new'}"
        return f"{label}: {self.strings_qty} strings × {self.modules_per_string} módulos/string"


class PlantCableSegment(models.Model):
    SEG_CHOICES = [
        ("PV_CHAIN", "Módulo–Módulo (string)"),
        ("STRING_HOME", "String–Caixa/Combiner"),
        ("DC_MAIN", "DC principal – Combiner/Inv"),
        ("AC_GRID", "AC – Inversor–QDG/Rede"),
        ("GND", "Aterramento"),
        ("OUTRO", "Outro"),
    ]
    plant = models.ForeignKey(PVPlant, on_delete=models.CASCADE, related_name="cable_segments")
    segment = models.CharField("Trecho", max_length=20, choices=SEG_CHOICES)
    description = models.CharField("Descrição", max_length=120, blank=True, default="")
    length_m = models.DecimalField("Comprimento (m)", max_digits=8, decimal_places=2,
                                   validators=[MinValueValidator(0)])
    cross_section_mm2 = models.DecimalField("Seção (mm²)", max_digits=6, decimal_places=2,
                                            validators=[MinValueValidator(0)])
    material = models.CharField("Material", max_length=2, choices=[("CU", "Cobre"), ("AL", "Alumínio")], default="CU")
    qty_parallel = models.PositiveSmallIntegerField("Condutores em paralelo", default=1,
                                                    validators=[MinValueValidator(1)])

    class Meta:
        ordering = ["plant", "segment", "id"]
        verbose_name = "Trecho de cabeamento"
        verbose_name_plural = "Cabeamento da planta"

    def __str__(self):
        return f"{self.plant.nome} • {self.get_segment_display()} ({self.length_m} m)"

class PlantMonitoringCredential(models.Model):
    plant = models.ForeignKey("PVPlant", on_delete=models.CASCADE, related_name="credentials")
    provedor = models.CharField(
        "Provedor",
        max_length=60,
        choices=[("GROWATT", "Growatt"),
                 ("RENOVIGI", "Renovigi")],
        default="GROWATT",
    )
    username = models.CharField("Login/usuário", max_length=120, blank=True, default="")
    password = models.CharField("Senha (plain por enquanto)", max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # =========================
    # Renovigi/ShineMonitor (preenchido automaticamente)
    # =========================
    shinemonitor_plantid = models.BigIntegerField(null=True, blank=True)

    shinemonitor_pn = models.CharField(max_length=80, blank=True, default="")
    shinemonitor_devcode = models.CharField(max_length=40, blank=True, default="")
    shinemonitor_devaddr = models.IntegerField(null=True, blank=True)
    shinemonitor_sn = models.CharField(max_length=80, blank=True, default="")

    shinemonitor_i18n = models.CharField(max_length=20, blank=True, default="pt_BR")
    shinemonitor_lang = models.CharField(max_length=20, blank=True, default="pt_BR")

    # cache do “discovery” (para renderizar selects no template sem refazer chamadas toda hora)
    shinemonitor_plants_cache = models.JSONField(blank=True, default=list)   # list[dict]
    shinemonitor_devices_cache = models.JSONField(blank=True, default=list)  # list[dict]

    class Meta:
        verbose_name = "Credencial de monitoramento"
        verbose_name_plural = "Credenciais de monitoramento"
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "provedor"],
                name="uniq_plant_provedor",
            )
        ]

    def __str__(self):
        return f"Credenciais {self.plant.nome} ({self.provedor})"


#---------------------------
#--------------------------- RENOVIGI
#---------------------------

class ShineCredential(models.Model):
    """
    Guarda token/secret do ShineMonitor.
    Recomendo criptografar em produção (ex.: django-fernet-fields), mas aqui deixo simples.
    """
    name = models.CharField(max_length=100, unique=True)

    token = models.TextField()
    secret = models.TextField()

    # opcional (se você quiser controlar expiração)
    expires_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_expired(self) -> bool:
        return bool(self.expires_at and timezone.now() >= self.expires_at)

    def __str__(self):
        return self.name


class ShineDevice(models.Model):
    """
    Identificadores do dispositivo, conforme o seu request:
    pn, devcode, devaddr, sn (+ i18n/lang) e oddEvenRow=null se necessário.
    """
    name = models.CharField(max_length=120)
    credential = models.ForeignKey(ShineCredential, on_delete=models.PROTECT)

    pn = models.CharField(max_length=40)
    devcode = models.CharField(max_length=40)
    devaddr = models.CharField(max_length=16)
    sn = models.CharField(max_length=64)

    i18n = models.CharField(max_length=16, default="pt_BR")
    lang = models.CharField(max_length=8, default="pt")

    # Se seu endpoint exige oddEvenRow=null, preencha com "null"
    odd_even_row = models.CharField(max_length=16, blank=True, default="")

    # Timestamp vem como string local do dispositivo; convertemos para UTC
    timezone_name = models.CharField(max_length=64, default="America/Sao_Paulo")

    # Em geral o timestamp está em field_1 (índice 1), mas deixamos configurável
    timestamp_index = models.PositiveSmallIntegerField(default=1)

    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} (sn={self.sn})"


class ShineProtocolSchema(models.Model):
    """
    Guarda o 'title' retornado pela API para mapear field_0..field_n -> nomes.
    """
    device = models.OneToOneField(ShineDevice, on_delete=models.CASCADE)
    titles = models.JSONField(default=list)  # lista de dicts (ex.: {"title": "...", "unit": "..."})
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Schema {self.device_id}"


class ShineReading(models.Model):
    """
    Leitura “linha” do dia. Idempotência: unique(device, ts_utc).
    """
    device = models.ForeignKey(ShineDevice, on_delete=models.CASCADE)
    ts_utc = models.DateTimeField()

    # Mapeado por título quando disponível
    fields = models.JSONField(default=dict)

    # Se quiser auditoria: raw da linha (filed list / dict)
    raw = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["device", "ts_utc"], name="uniq_shine_device_ts"),
        ]
        indexes = [
            models.Index(fields=["device", "ts_utc"]),
        ]

    def __str__(self):
        return f"{self.device_id} @ {self.ts_utc.isoformat()}"
    

#---------------------------
#--------------------------- M E T E O
#---------------------------

# core/models.py

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import timezone
from decimal import Decimal
from django.db.models import Q, F

# ... resto do arquivo permanece igual ...


class MeteoSource(models.TextChoices):
    OPENMETEO = "OPENMETEO", "Open-Meteo"
    NSRDB = "NSRDB", "NSRDB (NREL)"


class MeteoDataTypology(models.TextChoices):
    REANALYSIS_MODELED = "REANALYSIS_MODELED", "Reanálise / modelado"
    DERIVED_MODELED = "DERIVED_MODELED", "Derivado / modelado"
    MEASURED = "MEASURED", "Medido"
    OTHER = "OTHER", "Outro"


class MeteoImportBatch(models.Model):
    """
    Lote de importação meteorológica.
    Guarda a proveniência da requisição feita ao provedor.
    """
    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="meteo_import_batches",
    )

    source = models.CharField(
        max_length=20,
        choices=MeteoSource.choices,
        default=MeteoSource.OPENMETEO,
        db_index=True,
    )

    source_endpoint = models.CharField(max_length=255, blank=True, default="")
    dataset_model = models.CharField(
        max_length=64,
        blank=True,
        default="best_match",
        db_index=True,
        help_text="Modelo requisitado à Open-Meteo, ex.: best_match, era5, era5_land, cerra.",
    )
    data_typology = models.CharField(
        max_length=32,
        choices=MeteoDataTypology.choices,
        default=MeteoDataTypology.REANALYSIS_MODELED,
    )

    interval_min = models.PositiveSmallIntegerField(
        default=15,
        validators=[MinValueValidator(1)],
    )

    start_date = models.DateField()
    end_date = models.DateField()

    request_url = models.TextField(blank=True, default="")
    request_params = models.JSONField(default=dict, blank=True)
    response_meta = models.JSONField(default=dict, blank=True)

    imported_rows = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["plant", "source", "created_at"]),
            models.Index(fields=["plant", "dataset_model", "created_at"]),
        ]
        verbose_name = "Lote de importação meteorológica"
        verbose_name_plural = "Lotes de importação meteorológica"

    def __str__(self):
        return (
            f"{self.plant.nome} | {self.source} | {self.dataset_model} | "
            f"{self.start_date}..{self.end_date}"
        )


class MeteoRecord(models.Model):
    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="meteo_records",
    )

    source = models.CharField(
        max_length=20,
        choices=MeteoSource.choices,
        default=MeteoSource.OPENMETEO,
    )

    import_batch = models.ForeignKey(
        "core.MeteoImportBatch",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="records",
    )

    source_endpoint = models.CharField(max_length=255, blank=True, default="")
    dataset_model = models.CharField(
        max_length=64,
        blank=True,
        default="best_match",
        db_index=True,
        help_text="Modelo meteorológico requisitado ao provedor.",
    )
    data_typology = models.CharField(
        max_length=32,
        choices=MeteoDataTypology.choices,
        default=MeteoDataTypology.REANALYSIS_MODELED,
    )

    # CANÔNICO: timestamp em UTC
    ts_utc = models.DateTimeField(db_index=True)
    interval_min = models.PositiveSmallIntegerField(default=60, validators=[MinValueValidator(1)])

    # Radiação (W/m²)
    ghi = models.FloatField(null=True, blank=True)
    dni = models.FloatField(null=True, blank=True)
    dhi = models.FloatField(null=True, blank=True)

    # Opcional
    gti = models.FloatField(null=True, blank=True)

    # Meteorologia
    temp_air = models.FloatField(null=True, blank=True)
    wind_speed = models.FloatField(null=True, blank=True)
    rh = models.FloatField(null=True, blank=True)
    pressure = models.FloatField(null=True, blank=True)

    # Qualidade meteo / audit trail para FDD
    meteo_qc_score = models.FloatField(null=True, blank=True)
    flag_meteo_low_confidence = models.BooleanField(default=False)
    flag_meteo_interpolated = models.BooleanField(default=False)
    flag_meteo_outlier = models.BooleanField(default=False)
    flag_meteo_artifact = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "source", "ts_utc"],
                name="uniq_meteo_plant_source_tsutc",
            )
        ]
        indexes = [
            models.Index(fields=["plant", "source", "ts_utc"]),
            models.Index(fields=["plant", "dataset_model", "ts_utc"]),
        ]

    def __str__(self):
        return f"{self.plant.nome} {self.source} {self.ts_utc.isoformat()}"
#---------------------------
#--------------------------- S I N C R O N I Z A Ç Ã O
#---------------------------


class InverterOperationalData(models.Model):
    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="opdata",
    )

    provedor = models.CharField(max_length=60, default="RENOVIGI")

    # Identidade do device (ShineMonitor/Renovigi)
    pn = models.CharField(max_length=80)
    devcode = models.CharField(max_length=80)
    devaddr = models.IntegerField()
    sn = models.CharField(max_length=120)

    # Timestamp canônico (UTC) – use sempre timezone-aware
    ts_utc = models.DateTimeField(db_index=True)

    # Payload cru da API (linhas/medidas)
    payload = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  # útil p/ auditoria/reprocessamento

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "provedor", "pn", "devcode", "devaddr", "sn", "ts_utc"],
                name="uniq_opdata_device_ts",
            )
        ]
        indexes = [
            # Para consultas do tipo: "me dê a série da planta no período"
            models.Index(fields=["plant", "ts_utc"], name="idx_opdata_plant_ts"),

            # Para consultas do tipo: "último ponto do device" / "série de um device"
            models.Index(
                fields=["plant", "provedor", "pn", "devcode", "devaddr", "sn", "ts_utc"],
                name="idx_opdata_device_ts",
            ),

            # Útil quando você filtra por provedor e planta (ex.: múltiplos provedores no futuro)
            models.Index(fields=["plant", "provedor"], name="idx_opdata_plant_provider"),
        ]

        # opcional, mas costuma ajudar
        ordering = ["ts_utc"]

    def __str__(self) -> str:
        return f"{self.plant_id} {self.provedor} {self.pn}/{self.sn} @ {self.ts_utc}"

class InverterSample(models.Model):
    """
    Amostra bruta (raw) do inversor, em timestamp.
    O payload do ShineMonitor vem com headers + rows (matriz).
    Guardamos como JSON (raw) para não travar no mapeamento de colunas agora.
    Depois você pode normalizar em colunas (p_ac, v_dc, etc.) conforme quiser.
    """
    plant = models.ForeignKey(PVPlant, on_delete=models.CASCADE, related_name="inverter_samples")

    # Identificador do dispositivo (completo o suficiente para não colidir)
    device_key = models.CharField(max_length=255)

    ts = models.DateTimeField(db_index=True)

    # Linha bruta em dict: {"header1": value1, ...}
    data = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        indexes = [
            models.Index(fields=["plant", "device_key", "ts"]),
        ]
        constraints = [
            models.UniqueConstraint(fields=["plant", "device_key", "ts"], name="uniq_inverter_sample"),
        ]

    def __str__(self) -> str:
        return f"{self.plant_id} {self.device_key} {self.ts.isoformat()}"
    
class DataIngestState(models.Model):
    """
    Guarda o watermark da ingestão por planta + fonte + série (device_key).
    """
    plant = models.ForeignKey(PVPlant, on_delete=models.CASCADE, related_name="ingest_states")

    source = models.CharField(max_length=40)  # ex: "RENOVIGI"
    series_key = models.CharField(max_length=255)  # ex: device_key

    last_ok_day = models.DateField(null=True, blank=True)  # sincronização por dia
    last_run_at = models.DateTimeField(null=True, blank=True)
    last_status = models.CharField(max_length=20, default="never")  # never|ok|error
    last_error = models.TextField(blank=True, default="")

    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["plant", "source", "series_key"], name="uniq_ingest_state"),
        ]

    def mark_ok(self, day):
        self.last_ok_day = day
        self.last_run_at = timezone.now()
        self.last_status = "ok"
        self.last_error = ""
        self.updated_at = timezone.now()

    def mark_error(self, msg: str):
        self.last_run_at = timezone.now()
        self.last_status = "error"
        self.last_error = (msg or "")[:5000]
        self.updated_at = timezone.now()    




# ---------------------------
# C A S A R - B D
# ---------------------------

class MergedSourceOper(models.TextChoices):
    SHINEMONITOR = "SHINEMONITOR", "ShineMonitor/Renovigi"
    GROWATT = "GROWATT", "Growatt"
    MANUAL = "MANUAL", "Manual/Outros"


class MergedSourceMeteo(models.TextChoices):
    OPENMETEO = "OPENMETEO", "Open-Meteo"
    NSRDB = "NSRDB", "NSRDB (NREL)"


class PVPlantMergedRecord15m(models.Model):
    """
    Base casada em 15 minutos: inversor (5->15) + meteo (15) alinhados.
    Armazenar em UTC.
    """

    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="merged_15m",
    )

    # rastreio de proveniência
    source_oper = models.CharField(
        max_length=30,
        choices=MergedSourceOper.choices,
        default=MergedSourceOper.SHINEMONITOR,
        db_index=True,
    )

    source_meteo = models.CharField(
        max_length=20,
        choices=MergedSourceMeteo.choices,
        default=MergedSourceMeteo.OPENMETEO,
        db_index=True,
    )

    # Canônico
    ts_utc = models.DateTimeField(db_index=True)
    interval_min = models.PositiveSmallIntegerField(
        default=15,
        validators=[MinValueValidator(1)],
        help_text="Intervalo do bucket (min). Para esta tabela, deve ser 15.",
    )

    # -------- Operativo agregado (bucket 15 min) --------
    p_dc_w = models.FloatField(null=True, blank=True)
    p_ac_w = models.FloatField(null=True, blank=True)
    v_dc_v = models.FloatField(null=True, blank=True)
    i_dc_a = models.FloatField(null=True, blank=True)
    v_ac_v = models.FloatField(null=True, blank=True)
    i_ac_a = models.FloatField(null=True, blank=True)
    freq_hz = models.FloatField(null=True, blank=True)

    # -------- MPPT-level (para features internas / GNN) --------
    mppt1_vdc_v = models.FloatField(null=True, blank=True)
    mppt2_vdc_v = models.FloatField(null=True, blank=True)
    mppt3_vdc_v = models.FloatField(null=True, blank=True)
    mppt4_vdc_v = models.FloatField(null=True, blank=True)

    mppt1_idc_a = models.FloatField(null=True, blank=True)
    mppt2_idc_a = models.FloatField(null=True, blank=True)
    mppt3_idc_a = models.FloatField(null=True, blank=True)
    mppt4_idc_a = models.FloatField(null=True, blank=True)

    # -------- Alarmes (weak labels / features) --------
    alarm_code = models.IntegerField(null=True, blank=True, help_text="Código do alarme/falha (se disponível).")
    alarm_sev = models.PositiveSmallIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(10)],
        help_text="Severidade agregada no bucket (MVP: 0 OK, 2 fault).",
    )

    # Energia do inversor no bucket (Wh/15min)
    e_ac_wh_15 = models.FloatField(null=True, blank=True)

    # Qualidade do inversor
    inv_n = models.PositiveSmallIntegerField(null=True, blank=True)
    inv_coverage = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Cobertura (0..1) de amostras do inversor no bucket.",
    )
    flag_low_coverage = models.BooleanField(default=False)

    # -------- Meteo (15 min) --------
    ghi = models.FloatField(null=True, blank=True)
    dni = models.FloatField(null=True, blank=True)
    dhi = models.FloatField(null=True, blank=True)
    gti = models.FloatField(null=True, blank=True, help_text="POA/GTI se disponível")

    temp_air = models.FloatField(null=True, blank=True)
    wind_speed = models.FloatField(null=True, blank=True)
    rh = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Umidade relativa (%)",
    )
    pressure = models.FloatField(null=True, blank=True, help_text="Pressão (Pa)")

    # Qualidade meteo derivada do QC da fonte
    meteo_qc_score = models.FloatField(null=True, blank=True)
    flag_meteo_low_confidence = models.BooleanField(default=False)
    flag_meteo_interpolated = models.BooleanField(default=False)
    flag_meteo_outlier = models.BooleanField(default=False)
    flag_meteo_artifact = models.BooleanField(default=False)

    # Flags de integridade
    flag_meteo_missing = models.BooleanField(default=False)
    flag_inv_missing = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "source_oper", "source_meteo", "interval_min", "ts_utc"],
                name="uniq_merged15m_plant_sources_interval_tsutc",
            ),
            models.CheckConstraint(
                condition=Q(interval_min=15),
                name="chk_merged15m_interval_is_15",
            ),
            models.CheckConstraint(
                condition=Q(inv_coverage__isnull=True)
                | (Q(inv_coverage__gte=0.0) & Q(inv_coverage__lte=1.0)),
                name="chk_merged15m_inv_coverage_0_1",
            ),
            models.CheckConstraint(
                condition=Q(e_ac_wh_15__isnull=True) | Q(e_ac_wh_15__gte=0.0),
                name="chk_merged15m_e_ac_wh_15_nonneg",
            ),
        ]
        indexes = [
            models.Index(fields=["plant", "ts_utc"]),
            models.Index(fields=["plant", "source_oper", "source_meteo", "ts_utc"]),
        ]

    def __str__(self) -> str:
        plant_label = getattr(self.plant, "nome", None) or getattr(self.plant, "name", None) or str(self.plant_id)
        return f"{plant_label} merged15m {self.ts_utc.isoformat()}"

# ---------------------------
# F A L H A S
# ---------------------------

class PlantDiagnostic15m(models.Model):
    """
    Diagnóstico plant-level por timestamp com suporte a tiers de irradiância,
    estado operativo, domínio provável da falha e confiança do diagnóstico.
    """

    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="diagnostics_15m",
        db_index=True,
    )

    ts_utc = models.DateTimeField("Timestamp (UTC)", db_index=True)

    source_oper = models.CharField(
        max_length=30,
        blank=True,
        default="",
        db_index=True,
        help_text="Fonte operativa usada para este diagnóstico 15 min.",
    )
    source_meteo = models.CharField(
        max_length=20,
        blank=True,
        default="",
        db_index=True,
        help_text="Fonte meteorológica usada para este diagnóstico 15 min.",
    )

    rca_code = models.SmallIntegerField("RCA code", default=0, validators=[MinValueValidator(0)])
    rca_label = models.CharField("RCA label", max_length=64, default="invalid", blank=True)

    valid = models.BooleanField(default=False)
    anomaly_flag = models.BooleanField(default=False, db_index=True)
    detector_score = models.FloatField(null=True, blank=True)
    ewma_z = models.FloatField(null=True, blank=True)
    cusum_score = models.FloatField(null=True, blank=True)
    stable_sky = models.BooleanField(default=False)
    detector_version = models.CharField(max_length=64, default="hybrid_rules_v1", blank=True)

    g_poa = models.FloatField("GPOA/POA (W/m²)", null=True, blank=True, validators=[MinValueValidator(0.0)])
    tcell_c = models.FloatField("Tcell (°C)", null=True, blank=True)

    pac_real_w = models.FloatField("Pac real (W)", null=True, blank=True)
    pac_model_w = models.FloatField("Pac modelo (W)", null=True, blank=True)
    mismatch_rel = models.FloatField("Mismatch relativo", null=True, blank=True, validators=[MinValueValidator(-5.0), MaxValueValidator(5.0)])

    irradiance_tier = models.CharField(max_length=1, default="N", blank=True)
    fine_diag_allowed = models.BooleanField(default=False)
    meteo_quality_ok = models.BooleanField(default=False)
    direct_grid_evidence = models.BooleanField(default=False)
    zero_injection_flag = models.BooleanField(default=False)

    state_label = models.CharField(max_length=64, default="unknown", blank=True)
    domain_label = models.CharField(max_length=64, default="unknown", blank=True)
    diagnosis_label = models.CharField(max_length=64, default="invalid", blank=True)
    diagnosis_confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])

    data_reliability_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    data_reliability_level = models.CharField(max_length=16, default="", blank=True)
    detection_confidence_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    detection_confidence_level = models.CharField(max_length=16, default="", blank=True)
    diagnosis_confidence_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    diagnosis_confidence_level = models.CharField(max_length=16, default="", blank=True)

    v_ac_v = models.FloatField(null=True, blank=True)
    i_ac_a = models.FloatField(null=True, blank=True)
    freq_hz = models.FloatField(null=True, blank=True)
    alarm_code_oper = models.IntegerField(null=True, blank=True)
    alarm_sev_oper = models.PositiveSmallIntegerField(null=True, blank=True)

    evidence_json = models.JSONField(null=True, blank=True)
    confidence_notes_json = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Diagnóstico 15 min"
        verbose_name_plural = "Diagnósticos 15 min"
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "source_oper", "source_meteo", "detector_version", "ts_utc"],
                name="uniq_diag15m_plant_sources_detector_ts",
            ),
        ]
        indexes = [
            models.Index(fields=["plant", "ts_utc"], name="idx_diag15m_plant_ts"),
            models.Index(fields=["plant", "source_oper", "source_meteo", "ts_utc"], name="idx_diag15m_sources_ts"),
            models.Index(fields=["plant", "detector_version", "ts_utc"], name="idx_diag15m_detector_ts"),
            models.Index(fields=["plant", "rca_code", "ts_utc"], name="idx_diag15m_plant_code_ts"),
            models.Index(fields=["plant", "anomaly_flag", "ts_utc"], name="idx_diag15m_plant_anom_ts"),
        ]
        ordering = ["plant_id", "ts_utc"]

    def __str__(self) -> str:
        ts = self.ts_utc.isoformat() if self.ts_utc else "n/a"
        src = f"{self.source_oper or '-'}|{self.source_meteo or '-'}|{self.detector_version or '-'}"
        return f"{self.plant_id} {ts} {src} {self.diagnosis_label or self.rca_label}"


# ---------------------------
# MPPT-level FDD predictions (GNN/GRU)
# ---------------------------

class MPPTDiagnostic15m(models.Model):
    """
    Um registro por (planta, inversor/source_oper, mppt, timestamp_utc).

    Guarda predição por MPPT (nó do grafo) para ser consumida no drawer/heatmap.
    """
    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="mppt_diagnostics_15m",
        db_index=True,
    )

    source_oper = models.CharField(max_length=30, db_index=True)
    mppt = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(16)])

    ts_utc = models.DateTimeField("Timestamp (UTC)", db_index=True)

    model_version = models.CharField(max_length=64, default="mppt_gnn_v1", blank=True)

    pred_code = models.SmallIntegerField(default=0)   # 0 normal, 1 disconnected (por enquanto)
    pred_label = models.CharField(max_length=40, default="normal", blank=True)
    pred_pmax = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])

    proba = models.JSONField(null=True, blank=True)   # opcional: {"normal":0.9,"disconnected":0.1}

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Diagnóstico MPPT 15 min"
        verbose_name_plural = "Diagnósticos MPPT 15 min"
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "source_oper", "mppt", "ts_utc"],
                name="uniq_mpptdiag_plant_src_mppt_ts",
            ),
        ]
        indexes = [
            models.Index(fields=["plant", "ts_utc"], name="idx_mpptdiag_plant_ts"),
            models.Index(fields=["plant", "source_oper", "ts_utc"], name="idx_mpptdiag_src_ts"),
            models.Index(fields=["plant", "source_oper", "mppt", "ts_utc"], name="idx_mpptdiag_src_mppt_ts"),
        ]
        ordering = ["plant_id", "source_oper", "mppt", "ts_utc"]

    def __str__(self) -> str:
        ts = self.ts_utc.isoformat() if self.ts_utc else "n/a"
        return f"{self.plant_id} {self.source_oper} mppt{self.mppt} {ts} {self.pred_label}"
    

# ---------------------------
# Event-level FDD
# ---------------------------

class FaultEvent(models.Model):
    """
    Evento anômalo persistido (plant-level), derivado dos bins de PlantDiagnostic15m.
    """
    STATUS_OPEN = "open"
    STATUS_CLOSED = "closed"
    STATUS_REVIEWED = "reviewed"
    STATUS_DISMISSED = "dismissed"

    STATUS_CHOICES = [
        (STATUS_OPEN, "Open"),
        (STATUS_CLOSED, "Closed"),
        (STATUS_REVIEWED, "Reviewed"),
        (STATUS_DISMISSED, "Dismissed"),
    ]

    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="fault_events",
        db_index=True,
    )
    source_oper = models.CharField(max_length=30, blank=True, default="", db_index=True)
    source_meteo = models.CharField(max_length=20, blank=True, default="", db_index=True)

    ts_start_utc = models.DateTimeField(db_index=True)
    ts_end_utc = models.DateTimeField(db_index=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)
    detector_version = models.CharField(max_length=64, default="residual_v1", blank=True)

    detector_score_max = models.FloatField(null=True, blank=True)
    detector_score_mean = models.FloatField(null=True, blank=True)
    severity_score = models.FloatField(null=True, blank=True)
    energy_loss_wh = models.FloatField(null=True, blank=True)

    event_label_prelim = models.CharField(max_length=64, default="unknown", blank=True)
    known_vs_unknown = models.CharField(max_length=16, default="pending", db_index=True)
    final_label = models.CharField(max_length=64, default="", blank=True)
    confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
    )
    data_reliability_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    data_reliability_level = models.CharField(max_length=16, default="", blank=True)
    detection_confidence_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    detection_confidence_level = models.CharField(max_length=16, default="", blank=True)
    diagnosis_confidence_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    diagnosis_confidence_level = models.CharField(max_length=16, default="", blank=True)
    novelty_score = models.FloatField(null=True, blank=True)

    meta = models.JSONField(null=True, blank=True)
    confidence_notes_json = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Fault Event"
        verbose_name_plural = "Fault Events"
        constraints = [
            models.UniqueConstraint(
                fields=["plant", "source_oper", "ts_start_utc", "ts_end_utc", "detector_version"],
                name="uniq_faultevent_window_detector",
            ),
            models.CheckConstraint(
                condition=Q(ts_end_utc__gte=F("ts_start_utc")),
                name="chk_faultevent_end_after_start",
            ),
        ]
        indexes = [
            models.Index(fields=["plant", "ts_start_utc"], name="idx_faultevent_plant_start"),
            models.Index(fields=["plant", "status", "ts_start_utc"], name="idx_faultevent_status_start"),
            models.Index(fields=["plant", "final_label", "ts_start_utc"], name="idx_faultevent_label_start"),
        ]
        ordering = ["plant_id", "ts_start_utc"]

    def __str__(self) -> str:
        return f"event {self.plant_id} {self.ts_start_utc.isoformat()}..{self.ts_end_utc.isoformat()}"


class GroundTruthEvent(models.Model):
    """
    Verdade de referência mínima para a campanha de validação do FDD mismatch.

    A anotação nasce em nível de evento e é posteriormente discretizada para bins
    de 15 minutos pelo serviço de validação. O objeto pode representar tanto um
    evento de falha confirmado quanto uma janela normal revisada manualmente.
    """

    STATE_CONFIRMED = "confirmed"
    STATE_NORMAL = "normal"
    STATE_UNCERTAIN = "uncertain"
    STATE_DISMISSED = "dismissed"

    STATE_CHOICES = [
        (STATE_CONFIRMED, "Confirmed fault"),
        (STATE_NORMAL, "Normal window"),
        (STATE_UNCERTAIN, "Uncertain"),
        (STATE_DISMISSED, "Dismissed"),
    ]

    plant = models.ForeignKey(
        "core.PVPlant",
        on_delete=models.CASCADE,
        related_name="ground_truth_events",
        db_index=True,
    )
    source_oper = models.CharField(max_length=30, blank=True, default="", db_index=True)
    source_meteo = models.CharField(max_length=20, blank=True, default="", db_index=True)
    detector_version = models.CharField(max_length=64, default="mismatch_runtime_v1", blank=True)

    ts_start_utc = models.DateTimeField(db_index=True)
    ts_end_utc = models.DateTimeField(db_index=True)

    truth_state = models.CharField(max_length=16, choices=STATE_CHOICES, default=STATE_CONFIRMED, db_index=True)
    truth_label = models.CharField(max_length=64, default="unknown", blank=True, db_index=True)
    truth_group = models.CharField(max_length=32, default="unknown", blank=True, db_index=True)

    annotation_source = models.CharField(max_length=32, default="specialist_review", blank=True)
    annotation_confidence = models.CharField(max_length=8, default="B", blank=True)
    created_by = models.CharField(max_length=128, default="", blank=True)
    notes = models.TextField(blank=True, default="")
    meta = models.JSONField(null=True, blank=True)

    linked_fault_event = models.ForeignKey(
        "core.FaultEvent",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="linked_ground_truth_events",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Ground Truth Event"
        verbose_name_plural = "Ground Truth Events"
        constraints = [
            models.CheckConstraint(
                condition=Q(ts_end_utc__gte=F("ts_start_utc")),
                name="chk_groundtruth_end_after_start",
            ),
        ]
        indexes = [
            models.Index(fields=["plant", "ts_start_utc"], name="idx_gt_event_plant_start"),
            models.Index(fields=["plant", "truth_state", "ts_start_utc"], name="idx_gt_event_state_start"),
            models.Index(fields=["plant", "truth_label", "ts_start_utc"], name="idx_gt_event_label_start"),
        ]
        ordering = ["plant_id", "ts_start_utc"]

    def __str__(self) -> str:
        return f"gt {self.plant_id} {self.truth_state} {self.truth_label} {self.ts_start_utc.isoformat()}..{self.ts_end_utc.isoformat()}"


class FaultEventMPPT(models.Model):
    """
    Diagnóstico por MPPT associado a um FaultEvent.
    """
    event = models.ForeignKey(
        "core.FaultEvent",
        on_delete=models.CASCADE,
        related_name="mppt_predictions",
        db_index=True,
    )
    source_oper = models.CharField(max_length=30, blank=True, default="", db_index=True)
    mppt = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(16)])

    model_version = models.CharField(max_length=64, default="event_rules_v1", blank=True)
    pred_code = models.SmallIntegerField(default=99)
    pred_label = models.CharField(max_length=64, default="unknown_fault", blank=True)
    confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
    )
    novelty_score = models.FloatField(null=True, blank=True)
    contribution = models.JSONField(null=True, blank=True)
    proba = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Fault Event MPPT"
        verbose_name_plural = "Fault Event MPPT"
        constraints = [
            models.UniqueConstraint(
                fields=["event", "mppt", "model_version"],
                name="uniq_faulteventmppt_event_mppt_model",
            ),
        ]
        indexes = [
            models.Index(fields=["event", "mppt"], name="idx_faulteventmppt_event_mppt"),
            models.Index(fields=["event", "pred_code"], name="idx_faulteventmppt_event_code"),
        ]
        ordering = ["event_id", "mppt"]

    def __str__(self) -> str:
        return f"event={self.event_id} mppt={self.mppt} {self.pred_label}"    