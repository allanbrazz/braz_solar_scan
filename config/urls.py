# config/urls.py
from django.contrib import admin
from django.urls import path, include
from core.views import (
    # Home/Auth/Met
    home, signup, nsrdb_view, nsrdb_api_json,
    # Módulos
    ModuleListView, ModuleCreateView, CSVUploadView,
    ModuleDetailView, ModuleUpdateView,
    # Plantas
    PlantListView, PlantCreateView, PlantDetailView,
    PlantUpdateView, PlantCredSaveView,
    PlantDetailsEditView, PlantCablesEditView, PlantGrowattDebugView, 
    PlantGrowattDailyJsonView, RenovigiConsoleView, 
    PlantOperationalDataListView, merge_run_view,
    pv_dashboard_view, pv_dashboard_timeseries_api,inverter_list_view,
    inverter_create_view, inverter_edit_view, mismatch_fdd_api, mismatch_fdd_view,
)

# ---------- pvmodules agrupado e namespaced ----------
pvmodules_patterns = [
    path("modulos",            ModuleListView.as_view(),   name="list"),
    path("novo/",              ModuleCreateView.as_view(), name="create"),
    path("upload/",            CSVUploadView.as_view(),    name="upload"),
    path("<int:pk>/",          ModuleDetailView.as_view(), name="detail"),
    path("<int:pk>/editar/",   ModuleUpdateView.as_view(), name="edit"),
]

# ---------- plants agrupado e namespaced (tudo neste arquivo) ----------
plants_patterns = [
    path("",                   PlantListView.as_view(),        name="list"),
    path("nova/",              PlantCreateView.as_view(),      name="create"),
    path("<int:pk>/",          PlantDetailView.as_view(),      name="detail"),
    path("<int:pk>/editar/",   PlantUpdateView.as_view(),      name="edit"),
    path("<int:pk>/detalhes/", PlantDetailsEditView.as_view(), name="details_edit"),
    path("<int:pk>/cabos/",    PlantCablesEditView.as_view(),  name="cables_edit"),
    path("<int:pk>/credenciais/", PlantCredSaveView.as_view(), name="cred_save"),
    # API Growatt read-only
    # === GROWATT ===
    path("<int:pk>/growatt/debug/", PlantGrowattDebugView.as_view(),
         name="growatt_debug"),
    path("<int:pk>/growatt/daily.json", PlantGrowattDailyJsonView.as_view(),
         name="growatt_daily_json"),
]

urlpatterns = [
    path("admin/", admin.site.urls),

    # Home
    path("", home, name="home"),

    # Auth
    path("accounts/", include("django.contrib.auth.urls")),
    path("accounts/signup/", signup, name="signup"),

    # Meteo
    path("nsrdb", nsrdb_view, name="nsrdb_view"),
    path("nsrdb/api", nsrdb_api_json, name="nsrdb_api_json"),

    # Namespaces
    path("pvmodules/", include((pvmodules_patterns, "pvmodules"), namespace="pvmodules")),
    path("plantas/",   include((plants_patterns,   "plants"),   namespace="plants")),

    # RENOVIGI
    path("plants/<int:pk>/renovigi/console/", RenovigiConsoleView.as_view(), name="renovigi_console"),
    # DADOS ARQUIVADOS RENOVIGI
    path("<int:pk>/opdata/", PlantOperationalDataListView.as_view(), name="opdata_list"),

    #JUNÇÃO
    path("merge", merge_run_view, name="merge_run_view"),

    #DASHBOARD
    path("dashboard/pv/", pv_dashboard_view, name="pv_dashboard"),
    path("dashboard/pv/api/timeseries/", pv_dashboard_timeseries_api, name="pv_dashboard_api_timeseries"),

    # FDD (Mismatch)
    path("dashboard/fdd/mismatch/", mismatch_fdd_view, name="mismatch_fdd"),
    path("dashboard/fdd/mismatch/api/", mismatch_fdd_api, name="mismatch_fdd_api"),


    #INVERSOR
    path("inverters/", inverter_list_view, name="inverter_list"),
    path("inverters/new/", inverter_create_view, name="inverter_create"),
    path("inverters/<int:pk>/edit/", inverter_edit_view, name="inverter_edit"),  # NOVO
]
