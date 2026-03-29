# config/urls.py
from django.contrib import admin
from django.urls import path, include
from core.views.basicas import (
    home,
    signup,
)
from core.views.meteo import (
    open_meteo_view,
    open_meteo_view_api_json,
)
from core.views.modulos import (
    # Módulos
    ModuleListView,
    ModuleCreateView,
    CSVUploadView,
    ModuleDetailView,
    ModuleUpdateView,
)
from core.views.plantas import (
    PlantListView, 
    PlantCreateView, 
    PlantDetailView,
    PlantUpdateView, 
    PlantCredSaveView,
    PlantDetailsEditView, 
    PlantCablesEditView
)
from core.views.growatt import (
    PlantGrowattDebugView, 
    PlantGrowattDailyJsonView,
 )
from core.views.renovigi import (
    RenovigiConsoleView, 
    PlantOperationalDataListView, 
)
from core.views.juntar import (
    merge_run_view,
)
from core.views.dashboard import (
    pv_dashboard_view, 
    pv_dashboard_timeseries_api,
)
from core.views.inversor import (
    inverter_list_view,
    inverter_create_view, 
    inverter_edit_view, 
)
from core.views.fdd import (
    mismatch_fdd_api, 
    mismatch_fdd_view,
    mismatch_fdd_export_pdf,
)

from core.views.mppt_gnn_fdd import (
    mppt_gnn_fdd_view,
    mppt_gnn_fdd_api,
    mppt_gnn_fdd_dump_api,
    mppt_gnn_fdd_actions_api,
    mppt_gnn_fdd_export_pdf,
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
    path("openmeteo", open_meteo_view, name="open_meteo_view"),
    path("openmeteo/api", open_meteo_view_api_json, name="open_meteo_view_api_json"),

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
    path("dashboard/fdd/mismatch/export-pdf/", mismatch_fdd_export_pdf, name="mismatch_fdd_export_pdf"),
    path("dashboard/fdd/mppt-gnn/", mppt_gnn_fdd_view, name="mppt_gnn_fdd_view"),
    path("dashboard/fdd/mppt-gnn/api/", mppt_gnn_fdd_api, name="mppt_gnn_fdd_api"),
    path("dashboard/fdd/mppt-gnn/dump/", mppt_gnn_fdd_dump_api, name="mppt_gnn_fdd_dump_api"),
    path("dashboard/fdd/mppt-gnn/actions/", mppt_gnn_fdd_actions_api, name="mppt_gnn_fdd_actions_api"),
    path("dashboard/fdd/mppt-gnn/export-pdf/", mppt_gnn_fdd_export_pdf, name="mppt_gnn_fdd_export_pdf"),

    #INVERSOR
    path("inverters/", inverter_list_view, name="inverter_list"),
    path("inverters/new/", inverter_create_view, name="inverter_create"),
    path("inverters/<int:pk>/edit/", inverter_edit_view, name="inverter_edit"),  # NOVO
]
