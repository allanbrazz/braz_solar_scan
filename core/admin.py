# core/admin.py

from django.contrib import admin
from .models import (
    ShineCredential,
    ShineDevice,
    ShineProtocolSchema,
    ShineReading,
    InverterOperationalData,
    MeteoRecord,
    MeteoImportBatch,
    PVPlantDetails,
    PVPlantStringConfig,
    PVPlantMergedRecord15m,
    GroundTruthEvent,
    FaultEvent,
    PlantDiagnostic15m,
)
from django.utils import timezone as dj_tz
from zoneinfo import ZoneInfo


@admin.register(PVPlantMergedRecord15m)
class PVPlantMergedRecord15mAdmin(admin.ModelAdmin):
    list_display = (
        "plant",
        "ts_utc",
        "source_oper",
        "source_meteo",
        "p_ac_w",
        "gti",
        "flag_low_coverage",
        "flag_meteo_missing",
    )
    list_filter = (
        "plant",
        "source_oper",
        "source_meteo",
        "flag_low_coverage",
        "flag_meteo_missing",
        "ts_utc",
    )
    search_fields = ("plant__nome",)
    ordering = ("-ts_utc",)
    fieldsets = (
        ("Identificação e Tempo", {
            "fields": ("plant", "ts_utc", "interval_min", "source_oper", "source_meteo")
        }),
        ("Dados do Inversor (Operativo)", {
            "fields": (("p_dc_w", "p_ac_w"), ("v_dc_v", "i_dc_a"), ("v_ac_v", "i_ac_a"), "e_ac_wh_15")
        }),
        ("Dados Meteorológicos", {
            "fields": (("ghi", "dni", "dhi", "gti"), ("temp_air", "wind_speed", "rh", "pressure"))
        }),
        ("Qualidade e Diagnóstico", {
            "fields": ("inv_n", "inv_coverage", "flag_low_coverage", "flag_meteo_missing", "flag_inv_missing")
        }),
        ("Metadados", {
            "fields": ("created_at",),
            "classes": ("collapse",),
        }),
    )
    readonly_fields = ("created_at",)


class PVPlantStringConfigInline(admin.TabularInline):
    model = PVPlantStringConfig
    extra = 1


@admin.register(PVPlantDetails)
class PVPlantDetailsAdmin(admin.ModelAdmin):
    inlines = [PVPlantStringConfigInline]


@admin.register(MeteoImportBatch)
class MeteoImportBatchAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "plant",
        "source",
        "dataset_model",
        "data_typology",
        "interval_min",
        "start_date",
        "end_date",
        "imported_rows",
        "created_at",
    )
    list_filter = ("source", "dataset_model", "data_typology", "interval_min")
    search_fields = ("plant__nome", "request_url", "source_endpoint")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)


@admin.register(MeteoRecord)
class MeteoRecordAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "plant",
        "source",
        "dataset_model",
        "data_typology",
        "ts_utc",
        "ts_local",
        "interval_min",
        "ghi",
        "dni",
        "dhi",
        "temp_air",
    )
    list_filter = ("plant", "source", "dataset_model", "data_typology", "interval_min")
    date_hierarchy = "ts_utc"
    ordering = ("-ts_utc",)
    search_fields = ("plant__nome", "source_endpoint")

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


@admin.register(PlantDiagnostic15m)
class PlantDiagnostic15mAdmin(admin.ModelAdmin):
    list_display = ("plant", "ts_utc", "source_oper", "source_meteo", "detector_version", "anomaly_flag", "diagnosis_label")
    list_filter = ("plant", "source_oper", "source_meteo", "detector_version", "anomaly_flag", "diagnosis_label")
    search_fields = ("plant__nome", "diagnosis_label", "rca_label")
    ordering = ("-ts_utc",)


@admin.register(FaultEvent)
class FaultEventAdmin(admin.ModelAdmin):
    list_display = ("plant", "ts_start_utc", "ts_end_utc", "status", "event_label_prelim", "final_label", "known_vs_unknown")
    list_filter = ("plant", "status", "source_oper", "source_meteo", "detector_version", "known_vs_unknown")
    search_fields = ("plant__nome", "event_label_prelim", "final_label")
    ordering = ("-ts_start_utc",)


@admin.register(GroundTruthEvent)
class GroundTruthEventAdmin(admin.ModelAdmin):
    list_display = ("plant", "ts_start_utc", "ts_end_utc", "truth_state", "truth_label", "annotation_source", "annotation_confidence")
    list_filter = ("plant", "truth_state", "truth_label", "truth_group", "annotation_source")
    search_fields = ("plant__nome", "truth_label", "notes", "created_by")
    ordering = ("-ts_start_utc",)