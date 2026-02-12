#core/admin.py
from django.contrib import admin
from .models import ShineCredential, ShineDevice, ShineProtocolSchema, ShineReading, InverterOperationalData, MeteoRecord, PVPlantDetails, PVPlantStringConfig, PVPlantMergedRecord15m
from django.utils import timezone as dj_tz
from zoneinfo import ZoneInfo


# Register your models here.

@admin.register(PVPlantMergedRecord15m)
class PVPlantMergedRecord15mAdmin(admin.ModelAdmin):
    # Colunas que aparecerão na lista principal
    list_display = (
        'plant', 
        'ts_utc', 
        'source_oper', 
        'source_meteo', 
        'p_ac_w', 
        'gti', 
        'flag_low_coverage',
        'flag_meteo_missing'
    )
    
    # Filtros na lateral direita
    list_filter = (
        'plant', 
        'source_oper', 
        'source_meteo', 
        'flag_low_coverage', 
        'flag_meteo_missing',
        'ts_utc'
    )
    
    # Campo de busca (útil se você tiver o nome da planta no modelo relacionado)
    search_fields = ('plant__nome',)
    
    # Ordenação padrão (do mais recente para o mais antigo)
    ordering = ('-ts_utc',)
    
    # Organização dos detalhes dentro do formulário de edição
    fieldsets = (
        ('Identificação e Tempo', {
            'fields': ('plant', 'ts_utc', 'interval_min', 'source_oper', 'source_meteo')
        }),
        ('Dados do Inversor (Operativo)', {
            'fields': (('p_dc_w', 'p_ac_w'), ('v_dc_v', 'i_dc_a'), ('v_ac_v', 'i_ac_a'), 'e_ac_wh_15')
        }),
        ('Dados Meteorológicos', {
            'fields': (('ghi', 'dni', 'dhi', 'gti'), ('temp_air', 'wind_speed', 'rh', 'pressure'))
        }),
        ('Qualidade e Diagnóstico', {
            'fields': ('inv_n', 'inv_coverage', 'flag_low_coverage', 'flag_meteo_missing', 'flag_inv_missing')
        }),
        ('Metadados', {
            'fields': ('created_at',),
            'classes': ('collapse',), # Esconde por padrão para não poluir
        }),
    )
    
    # Define campos como apenas leitura (created_at não pode ser editado)
    readonly_fields = ('created_at',)


class PVPlantStringConfigInline(admin.TabularInline):
    model = PVPlantStringConfig
    extra = 1

@admin.register(PVPlantDetails)
class PVPlantDetailsAdmin(admin.ModelAdmin):
    inlines = [PVPlantStringConfigInline]

@admin.register(MeteoRecord)
class MeteoRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "plant", "source", "ts_utc", "ts_local", "interval_min", "ghi", "dni", "dhi", "temp_air")
    list_filter = ("plant", "source", "interval_min")
    date_hierarchy = "ts_utc"
    ordering = ("-ts_utc",)

    @admin.display(description="ts_local")
    def ts_local(self, obj: MeteoRecord):
        tz_name = getattr(obj.plant, "timezone", None) or "UTC"
        return dj_tz.localtime(obj.ts_utc, timezone=ZoneInfo(tz_name))


@admin.register(InverterOperationalData)
class InverterOperationalDataAdmin(admin.ModelAdmin):
    list_display = ("id", "plant", "provedor", "ts_utc", "ts_local", "sn", "pn", "devcode", "devaddr")
    list_filter = ("plant", "provedor")
    date_hierarchy = "ts_utc"
    ordering = ("-ts_utc",)

    @admin.display(description="ts_local")
    def ts_local(self, obj: InverterOperationalData):
        tz_name = getattr(obj.plant, "timezone", None) or "UTC"
        return dj_tz.localtime(obj.ts_utc, timezone=ZoneInfo(tz_name))


@admin.register(ShineCredential)
class ShineCredentialAdmin(admin.ModelAdmin):
    list_display = ("name", "expires_at", "updated_at")
    search_fields = ("name",)

@admin.register(ShineDevice)
class ShineDeviceAdmin(admin.ModelAdmin):
    list_display = ("name", "pn", "devcode", "devaddr", "sn", "i18n", "lang", "odd_even_row", "is_active")
    search_fields = ("name", "sn", "pn")
    list_filter = ("is_active", "i18n", "lang")

@admin.register(ShineProtocolSchema)
class ShineProtocolSchemaAdmin(admin.ModelAdmin):
    list_display = ("device", "updated_at")

@admin.register(ShineReading)
class ShineReadingAdmin(admin.ModelAdmin):
    list_display = ("device", "ts_utc")
    list_filter = ("device",)
    date_hierarchy = "ts_utc"