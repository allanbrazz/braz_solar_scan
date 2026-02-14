from __future__ import annotations

import math
import logging
import os, re
import csv
from decimal import Decimal, InvalidOperation
import datetime as dt
from datetime import date, time
import pandas as pd
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from django.db.models import Q
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpRequest, HttpResponse, JsonResponse, HttpResponseRedirect, HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse, reverse_lazy
from django.views import View
from django.conf import settings
from django.views.generic import ListView, DetailView, CreateView, UpdateView, FormView
from django.core.cache import cache
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.http import require_GET, require_http_methods
from core.models import ShineDevice, InverterOperationalData, MeteoRecord, MeteoSource ,PVPlant, PVPlantMergedRecord15m, PlantDiagnostic15m, PVPlantMergedRecord15m
from django.core.paginator import Paginator
from datetime import datetime, timedelta, timezone
from django.utils.dateparse import parse_date
from .services.dados_satelite.nsrdb import fetch_nsrdb_goes_full_disc_csv, ingest_nsrdb_range 
from core.services.dados_inversor.renovigi_gateway import discover_plants, discover_devices, fetch_range_table
from core.services.dados_inversor.renovigi_ingest import sync_operational_data_for_device
from django.utils.timezone import make_aware
from core.services.coverage import compute_time_coverage
from core.services.dados_satelite.openmeteo import ingest_openmeteo_range
from core.forms import MergeRunForm, PVInverterForm, PVStringConfigFormSet, MeteoRequestForm, PVStringGroupFormSet
from core.services.series_juntar.build_merged_dataset import build_plant_merged_dataset
from typing import Any, Dict, List, Optional, Tuple
from django.apps import apps
import inspect
from django.db import transaction, IntegrityError
from collections import OrderedDict, defaultdict
from core.services.power_model.power_model import (
    module_from_pvmodule,
    plant_from_details,
    expected_and_mismatch,
)




#NSRDB

def _nsrdb_env() -> dict:
    # Ajuste os nomes das env vars conforme você configurou no passo 1/2
    api_key = os.environ.get("NREL_API_KEY") or os.environ.get("NSRDB_API_KEY")
    full_name = os.environ.get("NREL_FULL_NAME") or os.environ.get("NSRDB_FULL_NAME")
    email = os.environ.get("NREL_EMAIL") or os.environ.get("NSRDB_EMAIL")
    affiliation = os.environ.get("NREL_AFFILIATION") or os.environ.get("NSRDB_AFFILIATION", "UTEC")
    reason = os.environ.get("NREL_REASON") or os.environ.get("NSRDB_REASON", "research")

    missing = [k for k,v in {
        "API_KEY": api_key, "FULL_NAME": full_name, "EMAIL": email
    }.items() if not v]

    if missing:
        raise ValueError(f"Credenciais NSRDB ausentes em env: {', '.join(missing)}")

    return dict(
        api_key=api_key,
        full_name=full_name,
        email=email,
        affiliation=affiliation,
        reason=reason,
    )


#GROWATT
from .services.dados_inversor.growatt_client import (
    fetch_growatt_plant_data,
    GrowattAuthError,
    GrowattReadError,
)
# Forms
from .forms import (
    NSRDBForm,
    PVModuleForm, CSVUploadForm,
    PVPlantForm, PlantMonitoringCredentialForm,
    # Estes dois são opcionais no seu projeto; mantenha se você os definiu:
    PVPlantDetailsForm,        # formulário para strings/tilt/azimute etc.
    PlantCableFormSet,         # inline formset para PlantCableSegment
    PlantCableFormSet,
    PlantCableSegmentForm,     # <<< ADICIONE ESTE
)

# Models
from .models import (
    PVModule,
    PVPlant,
    PlantMonitoringCredential,
    PVInverter,         # se você criou este modelo
    PVPlantDetails,     # <-- ADICIONE
    PlantCableSegment,  # <-- REMOVA (não é usado aqui)
)

LOCAL_TZ = ZoneInfo("America/Montevideo")


# -------------------------
# NSRDB
# -------------------------
# -------------------------
# NSRDB helpers
# -------------------------

def _nsrdb_env() -> dict:
    api_key = os.environ.get("NREL_API_KEY") or os.environ.get("NSRDB_API_KEY") or getattr(settings, "NREL_API_KEY", None)
    full_name = os.environ.get("NREL_FULL_NAME") or os.environ.get("NSRDB_FULL_NAME") or getattr(settings, "NREL_FULL_NAME", None)
    email = os.environ.get("NREL_EMAIL") or os.environ.get("NSRDB_EMAIL") or getattr(settings, "NREL_EMAIL", None)

    affiliation = os.environ.get("NREL_AFFILIATION") or os.environ.get("NSRDB_AFFILIATION") or getattr(settings, "NREL_AFFILIATION", "UTEC")
    reason = os.environ.get("NREL_REASON") or os.environ.get("NSRDB_REASON") or getattr(settings, "NREL_REASON", "research")

    missing = [k for k, v in {"API_KEY": api_key, "FULL_NAME": full_name, "EMAIL": email}.items() if not v]
    if missing:
        raise ValueError(f"Credenciais NSRDB ausentes em env: {', '.join(missing)}")

    return dict(api_key=api_key, full_name=full_name, email=email, affiliation=affiliation, reason=reason)


def _nsrdb_make_datetime_index(df: pd.DataFrame, utc_flag: bool) -> pd.DataFrame:
    needed = {"Year", "Month", "Day", "Hour", "Minute"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV NSRDB sem colunas de tempo esperadas. Achei: {list(df.columns)[:25]} ...")

    ts = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]], errors="coerce")
    out = df.copy()
    out["datetime"] = ts
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # NSRDB retorna timestamps "naive":
    # - utc=true: representa UTC
    # - utc=false: representa Local Standard Time (LST)
    if utc_flag:
        out.index = out.index.tz_localize("UTC")

    return out


def _nsrdb_normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes compatíveis com o front anterior:
      - air_temperature -> temperature_2m
      - wind_speed -> wind_speed_10m
    """
    out = df.copy()
    if "air_temperature" in out.columns and "temperature_2m" not in out.columns:
        out = out.rename(columns={"air_temperature": "temperature_2m"})
    if "wind_speed" in out.columns and "wind_speed_10m" not in out.columns:
        out = out.rename(columns={"wind_speed": "wind_speed_10m"})
    return out


def _slice_by_dates(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Recorta [start, end], incluindo todo o dia final.
    Funciona para índice tz-naive e tz-aware.
    """
    if df.empty:
        return df

    tz = df.index.tz
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end) + pd.Timedelta(days=1)  # exclusivo

    if tz is not None:
        t0 = t0.tz_localize(tz)
        t1 = t1.tz_localize(tz)

    return df.loc[(df.index >= t0) & (df.index < t1)]


def _nsrdb_cache_key(lat: float, lon: float, year: int, interval_min: int, utc_flag: bool, attributes: str) -> str:
    # normaliza attributes para evitar variações por espaços
    attrs = ",".join([a.strip() for a in (attributes or "").split(",") if a.strip()])
    return f"nsrdb:psm3:{lat:.5f}:{lon:.5f}:{year}:{interval_min}:{int(utc_flag)}:{attrs}"


def _nsrdb_fetch_year_cached(
    *, lat: float, lon: float, year: int,
    interval_min: int, utc_flag: bool, attributes: str,
    timeout_s: int = 120,
) -> tuple[dict, pd.DataFrame]:
    """
    Busca e faz cache de um ano do NSRDB.
    Cache é importante porque a API devolve CSV anual (pesado).
    """
    key = _nsrdb_cache_key(lat, lon, year, interval_min, utc_flag, attributes)
    cached = cache.get(key)
    if cached is not None:
        meta, df = cached
        return meta, df

    creds = _nsrdb_env()
    info, df_raw = fetch_nsrdb_goes_full_disc_csv(
        lat=lat,
        lon=lon,
        year=year,
        api_key=creds["api_key"],
        full_name=creds["full_name"],
        email=creds["email"],
        affiliation=creds["affiliation"],
        reason=creds["reason"],
        interval_min=interval_min,
        utc=utc_flag,
        attributes=attributes,
        timeout_s=timeout_s,
    )

    meta = info.iloc[0].to_dict() if info is not None and len(info) else {}
    df = _nsrdb_make_datetime_index(df_raw, utc_flag=utc_flag)
    df = _nsrdb_normalize_cols(df)

    # Cache por 6h (ajuste conforme uso)
    cache.set(key, (meta, df), timeout=6 * 3600)
    return meta, df


def _nsrdb_fetch_range(
    *, lat: float, lon: float, start: dt.date, end: dt.date,
    interval_min: int, utc_flag: bool, attributes: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Busca 1 ou 2 anos (se o período cruza virada de ano), concatena e recorta.
    Retorna df_all (index datetime) e meta (dict).
    """
    years = sorted(set([start.year, end.year]))
    frames: list[pd.DataFrame] = []
    meta_out: dict = {}

    for y in years:
        meta_y, df_y = _nsrdb_fetch_year_cached(
            lat=lat, lon=lon, year=y,
            interval_min=interval_min,
            utc_flag=utc_flag,
            attributes=attributes,
        )
        # mantém o primeiro meta não-vazio
        if not meta_out and meta_y:
            meta_out = meta_y
        frames.append(df_y)

    df_all = pd.concat(frames).sort_index()
    df_all = _slice_by_dates(df_all, start, end)
    return df_all, meta_out


# -------------------------
# NSRDB: JSON (API/debug)
# -------------------------

def nsrdb_api_json(request: HttpRequest) -> JsonResponse:
    """
    JSON com ghi/dni/dhi + temperature_2m + wind_speed_10m.
    Querystring:
      - lat, lon
      - start, end (YYYY-MM-DD)
      - interval (30|60), utc (0|1)
      - attributes (default: ghi,dhi,dni,wind_speed,air_temperature)
    """
    lat = _get_float(request, "lat", float(os.environ.get("PV_LAT", -34.9)))
    lon = _get_float(request, "lon", float(os.environ.get("PV_LON", -56.2)))

    today = dt.date.today()
    end_str = request.GET.get("end")
    start_str = request.GET.get("start")
    end = dtparser.parse(end_str).date() if end_str else today
    start = dtparser.parse(start_str).date() if start_str else (end - dt.timedelta(days=3))

    interval = int(request.GET.get("interval", 60))
    if interval not in (30, 60):
        interval = 60

    utc_flag = request.GET.get("utc", "0") in ("1", "true", "True", "yes", "on")
    attributes = request.GET.get("attributes", "ghi,dhi,dni,wind_speed,air_temperature")

    try:
        df_all, meta = _nsrdb_fetch_range(
            lat=lat, lon=lon, start=start, end=end,
            interval_min=interval, utc_flag=utc_flag, attributes=attributes
        )
    except Exception as exc:
        return JsonResponse({"error": f"Falha ao consultar NSRDB: {exc}"}, status=502, safe=False)

    if df_all.empty:
        return JsonResponse({"records": [], "meta": {"count": 0, "msg": "sem dados"}}, safe=False)

    out = df_all.reset_index().rename(columns={"datetime": "datetime_ref"})

    # serialização ISO
    if utc_flag:
        # index tz-aware UTC -> ISO Z
        out["datetime_ref"] = out["datetime_ref"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # local standard time (naive)
        out["datetime_ref"] = out["datetime_ref"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    keep = ["datetime_ref"]
    for c in ["ghi", "dni", "dhi", "temperature_2m", "wind_speed_10m"]:
        if c in out.columns:
            keep.append(c)

    out = out[keep].where(pd.notna(out[keep]), None)

    return JsonResponse(
        {
            "records": out.to_dict(orient="records"),
            "meta": {
                "count": len(out),
                "lat": lat,
                "lon": lon,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "interval_min": interval,
                "utc": utc_flag,
                "nsrdb_meta": meta,
            },
        },
        safe=False,
    )


# -------------------------
# Página HTML (usa radiation_view.html)
# -------------------------

@login_required
def nsrdb_view(request: HttpRequest) -> HttpResponse:
    """
    View HTML que reutiliza radiation_view.html.
    Mantive OpenMeteoForm para compatibilidade imediata com o template,
    mas aqui tilt/azimuth NÃO são usados (NSRDB entrega GHI/DNI/DHI).
    """
    today = dt.date.today()
    initial = dict(
        lat=float(os.environ.get("PV_LAT", -34.9)),
        lon=float(os.environ.get("PV_LON", -56.2)),
        tilt=int(os.environ.get("PV_TILT", 25)),      # mantido no form/template
        azimuth=int(os.environ.get("PV_AZIMUTH", 0)), # mantido no form/template
        start=today - dt.timedelta(days=3),
        end=today,
    )
    form = NSRDBForm(request.GET or None, initial=initial)

    df_show = None
    meta: dict = {}

    if form.is_valid():
        cd = form.cleaned_data
        start = cd["start"]
        end = cd["end"]

        interval = int(request.GET.get("interval", 60))
        if interval not in (30, 60):
            interval = 60

        utc_flag = request.GET.get("utc", "0") in ("1", "true", "True", "yes", "on")
        attributes = request.GET.get("attributes", "ghi,dhi,dni,wind_speed,air_temperature")

        try:
            df_all, meta_nsrdb = _nsrdb_fetch_range(
                lat=cd["lat"], lon=cd["lon"],
                start=start, end=end,
                interval_min=interval,
                utc_flag=utc_flag,
                attributes=attributes
            )

            if df_all.empty:
                messages.warning(request, "Sem dados NSRDB para o intervalo escolhido.")
            else:
                # Para exibir em horário local:
                # - se utc_flag=True, converte UTC -> LOCAL_TZ
                # - se utc_flag=False, é LST (naive) e não tem tz; exibimos “como está”
                if utc_flag:
                    df_all = df_all.tz_convert(LOCAL_TZ)

                # RECOMENDADO: manter cadência nativa (evita NaN e é bem mais rápido)
                limit = int(request.GET.get("limit", 500))
                df_show = df_all.tail(limit).reset_index()
                df_show = df_show.where(pd.notna(df_show), None)

                meta = {
                    "count_total": len(df_all),
                    "count_shown": len(df_show),
                    "cadence": f"{interval} min (NSRDB nativo)",
                    "interval_in": interval,
                    "utc": utc_flag,
                    "timezone_display": str(LOCAL_TZ) if utc_flag else "Local Standard Time (NSRDB)",
                    "nsrdb_meta": meta_nsrdb,
                }

        except Exception as exc:
            messages.error(request, f"Falha ao buscar NSRDB: {exc}")

    ctx = {
        "form": form,
        "cols": list(df_show.columns) if df_show is not None else [],
        "rows": list(df_show.itertuples(index=False, name=None)) if df_show is not None else [],
        "meta": meta,
    }
    return render(request, "meteo/radiation_view.html", ctx)

# -------------------------
# Views básicas (auth / home)
# -------------------------

def _user_can_manage_plant(user, plant: PVPlant) -> bool:
    # Mais seguro: apenas superuser
    if user.is_superuser:
        return True

    # Se você quiser permitir ao owner (opcional):
    return bool(plant.owner_id and plant.owner_id == user.id)

@login_required
def home(request: HttpRequest) -> HttpResponse:
    qs = PVPlant.objects.all().order_by("nome")
    if not request.user.is_superuser:
        qs = qs.filter(owner=request.user)

    # Agora inclui latitude/longitude para o mapa
    plants = list(qs.values("id", "nome", "latitude", "longitude"))

    if request.method == "POST":
        action = (request.POST.get("action") or "").strip()

        if action == "purge_plant_data":
            plant_id = request.POST.get("plant_id")
            if not plant_id:
                messages.error(request, "Selecione uma planta.")
                return redirect("home")

            try:
                plant_id_int = int(plant_id)
            except Exception:
                messages.error(request, "plant_id inválido.")
                return redirect("home")

            plant = PVPlant.objects.filter(id=plant_id_int).first()
            if not plant:
                messages.error(request, "Planta não encontrada.")
                return redirect("home")

            if not _user_can_manage_plant(request.user, plant):
                messages.error(request, "Sem permissão para excluir dados desta planta.")
                return redirect("home")

            confirm = (request.POST.get("confirm_text") or "").strip().upper()
            if confirm != "EXCLUIR":
                messages.error(request, "Confirmação inválida. Digite EXCLUIR para confirmar.")
                return redirect("home")

            purge_oper = request.POST.get("purge_oper") == "on"
            purge_meteo = request.POST.get("purge_meteo") == "on"
            purge_merged = request.POST.get("purge_merged") == "on"

            if not (purge_oper or purge_meteo or purge_merged):
                messages.error(
                    request,
                    "Selecione ao menos uma base para apagar (operativo, meteo, merged).",
                )
                return redirect("home")

            op_qs = InverterOperationalData.objects.filter(plant=plant)
            met_qs = MeteoRecord.objects.filter(plant=plant)
            merged_qs = PVPlantMergedRecord15m.objects.filter(plant=plant)

            with transaction.atomic():
                op_deleted = op_qs.count() if purge_oper else 0
                met_deleted = met_qs.count() if purge_meteo else 0
                merged_deleted = merged_qs.count() if purge_merged else 0

                if purge_oper:
                    op_qs.delete()
                if purge_meteo:
                    met_qs.delete()
                if purge_merged:
                    merged_qs.delete()

            messages.success(
                request,
                f"Dados apagados da planta '{plant.nome}': "
                f"Operativo={op_deleted}, Meteo={met_deleted}, Merged={merged_deleted}.",
            )
            return redirect("home")

        messages.error(request, "Ação inválida.")
        return redirect("home")

    return render(request, "home.html", {"plants": plants})

def signup(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Conta criada com sucesso! Faça login.")
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "registration/signup.html", {"form": form})

# -------------------------
# Helpers
# -------------------------

def _get_float(request: HttpRequest, key: str, default: float) -> float:
    try:
        return float(request.GET.get(key, default))
    except Exception:
        return default

def _to_decimal(value):
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        raise ValueError(f"Valor numérico inválido: '{value}'")




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


#---------------------------
#---------------------------  I N V E R S O R
#---------------------------

@login_required
@require_GET
def inverter_list_view(request: HttpRequest) -> HttpResponse:
    q = (request.GET.get("q") or "").strip()
    fab = (request.GET.get("fab") or "").strip()

    sort = (request.GET.get("sort") or "fabricante").strip()
    order = (request.GET.get("order") or "asc").strip().lower()

    allowed_sort = {
        "fabricante", "modelo",
        "p_ac_nom_w", "v_ac_nom_v",
        "vdc_mppt_min_v", "vdc_mppt_max_v", "vdc_abs_max_v",
        "mppt_count", "strings_por_mppt_max",
        "eficiencia_max_pct",
    }
    if sort not in allowed_sort:
        sort = "fabricante"
    if order not in {"asc", "desc"}:
        order = "asc"

    ordering = sort if order == "asc" else f"-{sort}"

    qs = PVInverter.objects.all()

    if q:
        qs = qs.filter(Q(fabricante__icontains=q) | Q(modelo__icontains=q))

    if fab:
        qs = qs.filter(fabricante=fab)

    fabricantes = list(
        PVInverter.objects.values_list("fabricante", flat=True)
        .distinct()
        .order_by("fabricante")
    )

    qs = qs.order_by(ordering, "id")

    paginator = Paginator(qs, 50)
    page_obj = paginator.get_page(request.GET.get("page") or 1)

    context = {
        "inversores": page_obj.object_list,      # <- NOME QUE O TEMPLATE ESPERA
        "fabricantes": fabricantes,
        "q": q,
        "fab": fab,
        "sort": sort,
        "order": order,
        "is_paginated": page_obj.has_other_pages(),
        "page_obj": page_obj,
    }
    return render(request, "inverters/inverter_list.html", context)

@login_required
@require_http_methods(["GET", "POST"])
def inverter_create_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = PVInverterForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    obj = form.save()
                messages.success(request, f"Inversor cadastrado: {obj.fabricante} {obj.modelo}.")
                return redirect("inverter_list")
            except IntegrityError:
                form.add_error(None, "Já existe um inversor com este fabricante e modelo.")
        # se inválido, cai e re-renderiza com erros
    else:
        form = PVInverterForm()

    return render(request, "inverters/form.html", {"form": form})

@login_required
@require_http_methods(["GET", "POST"])
def inverter_edit_view(request: HttpRequest, pk: int) -> HttpResponse:
    inv = get_object_or_404(PVInverter, pk=pk)

    if request.method == "POST":
        form = PVInverterForm(request.POST, instance=inv)
        if form.is_valid():
            try:
                with transaction.atomic():
                    obj = form.save()
                messages.success(request, f"Inversor atualizado: {obj.fabricante} {obj.modelo}.")
                return redirect("inverter_list")
            except IntegrityError:
                form.add_error(None, "Já existe um inversor com este fabricante e modelo.")
    else:
        form = PVInverterForm(instance=inv)

    return render(
        request,
        "inverters/inverter_form.html",  # pode reutilizar o mesmo template do create
        {"form": form, "is_edit": True, "inv": inv},
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
    

#---------------------------
#---------------------------  M E T E O
#---------------------------

UTC = ZoneInfo("UTC")


def _safe_zoneinfo(tzname: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tzname) if tzname else ZoneInfo("UTC")
    except Exception:
        return ZoneInfo("UTC")


def _local_dates_to_utc_range(*, plant_tz: str | None, start_date, end_date):
    """
    Converte datas (date) em intervalo UTC semiaberto:
      [start_local 00:00, (end_date+1) 00:00) em tz local
      -> retorna (start_utc, end_utc_exclusive, tz_local)
    Isso evita off-by-one e funciona bem com __gte / __lt e com grades (15min/5min).
    """
    tz_local = _safe_zoneinfo(plant_tz)

    start_local = make_aware(datetime.combine(start_date, time.min), timezone=tz_local)

    # end EXCLUSIVO: começo do dia seguinte
    end_local_excl = make_aware(
        datetime.combine(end_date + timedelta(days=1), time.min),
        timezone=tz_local,
    )

    return start_local.astimezone(UTC), end_local_excl.astimezone(UTC), tz_local


def _align_utc_range_to_interval(*, start_utc, end_utc, interval_min: int):
    """
    Opcional: alinha start/end para a grade do intervalo.
    - start: floor
    - end: ceil (mantém end exclusivo)
    """
    import pandas as pd

    freq = f"{int(interval_min)}min"
    s = pd.Timestamp(start_utc).floor(freq)
    e = pd.Timestamp(end_utc).ceil(freq)

    # garante tz
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")

    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")

    return s.to_pydatetime(), e.to_pydatetime()


@require_http_methods(["GET", "POST"])
def nsrdb_view(request):
    """
    Mantive o nome da view/URL para não quebrar a rota.
    Opera com Open-Meteo (ingest).
    """
    form = MeteoRequestForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        plant = form.cleaned_data["plant"]
        start_date = form.cleaned_data["start_date"]
        end_date = form.cleaned_data["end_date"]
        include_gti = form.cleaned_data["include_gti"]
        model = (form.cleaned_data.get("model") or "").strip() or None

        try:
            count, meta = ingest_openmeteo_range(
                plant=plant,
                start_date=start_date,
                end_date=end_date,
                include_gti=include_gti,
                model=model,
            )
            messages.success(request, f"Open-Meteo: {count} registros ingeridos/atualizados.")
        except Exception as e:
            messages.error(request, f"Falha ao ingerir Open-Meteo: {e}")

    return render(request, "meteo/nsrdb_request.html", {"form": form})


def nsrdb_api_json(request):
    """
    Endpoint reutilizável para cobertura/consistência no banco.
    GET params via MeteoRequestForm: plant, start_date, end_date, interval_min
    """
    form = MeteoRequestForm(request.GET)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    plant = form.cleaned_data["plant"]
    start_date = form.cleaned_data["start_date"]
    end_date = form.cleaned_data["end_date"]
    interval_min = int(form.cleaned_data["interval_min"])

    start_utc, end_utc, tz_local = _local_dates_to_utc_range(
        plant_tz=getattr(plant, "timezone", None),
        start_date=start_date,
        end_date=end_date,
    )

    # Opcional, mas ajuda muito a bater contagens com “expected_count”
    start_utc, end_utc = _align_utc_range_to_interval(
        start_utc=start_utc, end_utc=end_utc, interval_min=interval_min
    )

    # >>> CORREÇÃO PRINCIPAL AQUI: source do Open-Meteo <<<
    qs = MeteoRecord.objects.filter(plant=plant, source=MeteoSource.OPENMETEO)

    cov = compute_time_coverage(
        queryset=qs,
        start_utc=start_utc,
        end_utc=end_utc,
        interval_min=interval_min,
    )

    ranges_local = [
        {"start": a.astimezone(tz_local).isoformat(), "end": b.astimezone(tz_local).isoformat()}
        for (a, b) in cov.missing_ranges_utc[:50]
    ]

    return JsonResponse({
        "ok": True,
        "plant_id": plant.id,
        "plant_tz": str(tz_local),
        "interval_min": cov.interval_min,
        "start_utc": cov.start_utc.isoformat(),
        "end_utc": cov.end_utc.isoformat(),
        "expected_count": cov.expected_count,
        "existing_count": cov.existing_count,
        "missing_count": cov.missing_count,
        "coverage_pct": round(cov.coverage_pct, 2),
        "missing_ranges_local": ranges_local,
        "missing_ranges_truncated": len(cov.missing_ranges_utc) > 50,
    })

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
    


#---------------------------
#---------------------------  RENOVIGI
#---------------------------

class RenovigiConsoleView(LoginRequiredMixin, View):
    template_name = "inverters/renovigi_console.html"

    def get_cred(self, plant: PVPlant) -> PlantMonitoringCredential | None:
        return PlantMonitoringCredential.objects.filter(
            plant=plant,
            provedor="RENOVIGI"
        ).first()

    # ---- helpers ----
    def _base_url(self) -> str:
        return getattr(settings, "RENOVIGI_BASE_URL", getattr(settings, "SHINEMONITOR_BASE_URL", ""))

    def _extract_plantid(self, p: dict) -> int | None:
        if not isinstance(p, dict):
            return None
        for k in ("pid", "plantid", "id"):
            v = p.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return None

    def _device_key(self, d: dict) -> str:
        # formato usado no bind: pn|devcode|devaddr|sn
        if not isinstance(d, dict):
            return "|||"
        return f"{d.get('pn','')}|{d.get('devcode','')}|{d.get('devaddr','')}|{d.get('sn','')}"

    def _normalize_result_rows(self, obj):
        """
        Normaliza retornos para um formato consistente, evitando:
          - 'list' object has no attribute 'get'

        Retorna: (result_dict, rows_list)
          - se vier dict e tiver 'rows'/'datas', usa isso
          - se vier list, assume que é a lista de linhas
        """
        if isinstance(obj, dict):
            rows = obj.get("rows")
            if rows is None:
                rows = obj.get("datas")
            if rows is None:
                rows = []
            if not isinstance(rows, list):
                rows = []
            result = dict(obj)
            result.setdefault("rows", rows)
            return result, rows

        if isinstance(obj, list):
            return {"rows": obj}, obj

        return {"rows": []}, []

    def _coerce_device_fields(self, d: dict) -> tuple[str, str, int | None, str, list[str]]:
        """
        Extrai e normaliza (pn, devcode, devaddr_int, sn) de um dict.
        Retorna também lista de campos faltantes para mensagem de erro.
        """
        if not isinstance(d, dict):
            return "", "", None, "", ["pn", "devcode", "devaddr", "sn"]

        pn = (d.get("pn") or "").strip()
        devcode = (d.get("devcode") or "").strip()
        sn = (d.get("sn") or "").strip()

        devaddr_raw = d.get("devaddr", None)
        devaddr: int | None
        try:
            devaddr = int(devaddr_raw) if devaddr_raw is not None and str(devaddr_raw).strip() != "" else None
        except Exception:
            devaddr = None

        missing = []
        if not pn:
            missing.append("pn")
        if not devcode:
            missing.append("devcode")
        if devaddr is None:
            missing.append("devaddr")
        if not sn:
            missing.append("sn")

        return pn, devcode, devaddr, sn, missing

    def _parse_device_key(self, s: str) -> tuple[str, str, int, str] | None:
        """
        Aceita:
        A) "pn|devcode|devaddr|sn"
        B) "pn=XXX | devcode=YYY | devaddr=1 | sn=ZZZ"
        Retorna (pn, devcode, devaddr_int, sn)
        """
        s = (s or "").strip()
        if not s:
            return None

        parts = [p.strip() for p in s.split("|")]
        if len(parts) == 4:
            pn, devcode, devaddr_s, sn = parts
            if pn and devcode and devaddr_s and sn:
                try:
                    return pn, devcode, int(devaddr_s), sn
                except Exception:
                    return None

        m_pn = re.search(r"\bpn\s*=\s*([^\|]+)", s, re.IGNORECASE)
        m_dc = re.search(r"\bdevcode\s*=\s*([^\|]+)", s, re.IGNORECASE)
        m_da = re.search(r"\bdevaddr\s*=\s*([0-9]+)", s, re.IGNORECASE)
        m_sn = re.search(r"\bsn\s*=\s*([^\|]+)", s, re.IGNORECASE)
        if m_pn and m_dc and m_da and m_sn:
            pn = m_pn.group(1).strip()
            devcode = m_dc.group(1).strip()
            devaddr_i = int(m_da.group(1).strip())
            sn = m_sn.group(1).strip()
            if pn and devcode and sn:
                return pn, devcode, devaddr_i, sn

        return None

    def _ctx(self, plant: PVPlant, cred: PlantMonitoringCredential, result=None):
        plants_cache = getattr(cred, "shinemonitor_plants_cache", None) or []
        devices_cache = getattr(cred, "shinemonitor_devices_cache", None) or []

        selected_device_key = ""
        if getattr(cred, "shinemonitor_pn", None) and getattr(cred, "shinemonitor_sn", None):
            selected_device_key = (
                f"{cred.shinemonitor_pn}|{cred.shinemonitor_devcode}|"
                f"{cred.shinemonitor_devaddr}|{cred.shinemonitor_sn}"
            )

        return {
            "plant": plant,
            "cred": cred,
            "company_key": getattr(settings, "RENOVIGI_COMPANY_KEY", ""),
            "base_url": self._base_url(),
            "result": result,

            # para o template popular selects
            "plants": plants_cache,
            "devices": devices_cache,
            "selected_plantid": getattr(cred, "shinemonitor_plantid", "") or "",
            "selected_device_key": selected_device_key,
        }

    def get(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        cred = self.get_cred(plant)

        if not cred:
            messages.error(request, "Salve primeiro as credenciais RENOVIGI para esta planta.")
            return redirect("plants:detail", pk=plant.pk)

        return render(request, self.template_name, self._ctx(plant, cred, result=None))

    def post(self, request, pk):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)
        cred = self.get_cred(plant)

        if not cred:
            messages.error(request, "Salve primeiro as credenciais RENOVIGI para esta planta.")
            return redirect("plants:detail", pk=plant.pk)

        action = (request.POST.get("action") or "").strip()

        use_saved_password = (request.POST.get("use_saved_password") == "on")
        username = (request.POST.get("username") or cred.username or "").strip()

        password = (request.POST.get("password") or "").strip()
        if not password and use_saved_password:
            password = cred.password

        if username and username != cred.username:
            cred.username = username
            cred.save(update_fields=["username", "updated_at"])

        if not username or not password:
            messages.error(request, "Informe usuário e senha (ou marque 'usar senha salva').")
            return redirect("core:renovigi_console", pk=plant.pk)

        try:
            # ---------------- DISCOVER ----------------
            if action == "discover":
                plants = discover_plants(username, password)
                cred.shinemonitor_plants_cache = plants

                plantid_post = (request.POST.get("plantid") or "").strip()
                if plantid_post:
                    try:
                        cred.shinemonitor_plantid = int(plantid_post)
                    except Exception:
                        pass

                if not getattr(cred, "shinemonitor_plantid", None):
                    if plants and isinstance(plants[0], dict):
                        pid0 = self._extract_plantid(plants[0])
                        if pid0 is not None:
                            cred.shinemonitor_plantid = pid0

                devices = []
                if getattr(cred, "shinemonitor_plantid", None):
                    devices = discover_devices(username, password, int(cred.shinemonitor_plantid))
                    cred.shinemonitor_devices_cache = devices

                    # pré-preenche apenas se ainda não há device salvo
                    if devices and not (getattr(cred, "shinemonitor_pn", "") and getattr(cred, "shinemonitor_sn", "")):
                        d0 = devices[0] if isinstance(devices[0], dict) else {}
                        pn0, devcode0, devaddr0, sn0, missing = self._coerce_device_fields(d0)

                        if missing:
                            messages.error(
                                request,
                                "Discovery retornou dispositivo incompleto (faltando: "
                                + ", ".join(missing)
                                + "). Ajuste o mapeamento do discover_devices(). "
                                + f"Keys recebidas: {list(d0.keys())}"
                            )
                        else:
                            cred.shinemonitor_pn = pn0
                            cred.shinemonitor_devcode = devcode0
                            cred.shinemonitor_devaddr = devaddr0
                            cred.shinemonitor_sn = sn0

                cred.save()
                messages.success(request, f"Discovery OK: {len(plants)} planta(s) | {len(devices)} device(s).")
                return render(request, self.template_name, self._ctx(plant, cred, result=None))

            # ---------------- BIND ----------------
            if action == "bind":
                plantid = (request.POST.get("plantid") or "").strip()
                device_key = (request.POST.get("device_key") or "").strip()

                if not plantid:
                    raise ValueError("Selecione uma planta (plantid).")

                cred.shinemonitor_plantid = int(plantid)

                devices = discover_devices(username, password, int(plantid))
                cred.shinemonitor_devices_cache = devices

                parsed = self._parse_device_key(device_key)

                if parsed:
                    pn, devcode, devaddr_i, sn = parsed
                    cred.shinemonitor_pn = pn
                    cred.shinemonitor_devcode = devcode
                    cred.shinemonitor_devaddr = devaddr_i
                    cred.shinemonitor_sn = sn
                elif devices:
                    d0 = devices[0] if isinstance(devices[0], dict) else {}
                    pn0, devcode0, devaddr0, sn0, missing = self._coerce_device_fields(d0)
                    if missing:
                        raise ValueError(
                            "Não foi possível derivar dispositivo do dropdown. "
                            f"Faltando: {', '.join(missing)}. Keys: {list(d0.keys())}"
                        )
                    cred.shinemonitor_pn = pn0
                    cred.shinemonitor_devcode = devcode0
                    cred.shinemonitor_devaddr = int(devaddr0)
                    cred.shinemonitor_sn = sn0
                else:
                    raise ValueError("Nenhum dispositivo disponível. Execute o discovery primeiro.")

                cred.save()
                messages.success(request, "Planta e dispositivo vinculados. Campos foram pré-preenchidos.")
                return redirect("core:renovigi_console", pk=plant.pk)

            # ---------------- SYNC (salvar no banco) ----------------
            if action == "sync":
                start_day = (request.POST.get("start_day") or "").strip()
                end_day = (request.POST.get("end_day") or "").strip()

                pn = (request.POST.get("pn") or getattr(cred, "shinemonitor_pn", "")).strip()
                devcode = (request.POST.get("devcode") or getattr(cred, "shinemonitor_devcode", "")).strip()
                devaddr = request.POST.get("devaddr") or getattr(cred, "shinemonitor_devaddr", None)
                sn = (request.POST.get("sn") or getattr(cred, "shinemonitor_sn", "")).strip()

                if not (start_day and end_day):
                    raise ValueError("Informe start_day e end_day (YYYY-MM-DD).")
                if not (pn and devcode and devaddr is not None and sn):
                    raise ValueError("Dispositivo incompleto. Faça o discovery/vínculo primeiro.")

                start_dt = date.fromisoformat(start_day)
                end_dt = date.fromisoformat(end_day)
                if end_dt < start_dt:
                    raise ValueError("end_day deve ser >= start_day.")

                # Se vier como string vazia, explode no int(): normalize antes
                devaddr_i = int(devaddr)

                # Chama o ingest (retorna dict com inserted/requested_rows/bad_ts/per_day/range etc.)
                stats = sync_operational_data_for_device(
                    plant=plant,
                    cred=cred,
                    username=username,
                    password=password,
                    pn=pn,
                    devcode=devcode,
                    devaddr=devaddr_i,
                    sn=sn,
                    start_day=start_dt,
                    end_day=end_dt,
                    # opcional: deixe defaults do ingest fazerem incremental/backfill automaticamente
                    # incremental_from_last=True,
                    # skip_days_if_exists=True,
                    # safety_days=1,
                )

                if not isinstance(stats, dict):
                    raise RuntimeError(f"Retorno inesperado do sync (esperava dict): {type(stats)}")

                inserted = int(stats.get("inserted") or 0)
                requested_rows = int(stats.get("requested_rows") or 0)
                bad_ts = int(stats.get("bad_ts") or 0)

                # Pequeno resumo “por dia” (útil p/ você ver se está pulando dias)
                per_day = stats.get("per_day") or []
                days_with_insert = sum(1 for d in per_day if (d.get("inserted") or 0) > 0)
                days_skipped = sum(1 for d in per_day if d.get("skipped"))
                days_total = len(per_day)

                # Range efetivo (para debug de backfill)
                rng = stats.get("range") or {}
                eff_start = rng.get("effective_start") or ""
                eff_reason = rng.get("effective_reason") or ""

                messages.success(
                    request,
                    "Sync OK: inserted=%s (requested_rows=%s, bad_ts=%s). "
                    "days_total=%s, days_with_insert=%s, days_skipped=%s. "
                    "effective_start=%s %s"
                    % (
                        inserted,
                        requested_rows,
                        bad_ts,
                        days_total,
                        days_with_insert,
                        days_skipped,
                        eff_start,
                        f"[{eff_reason}]" if eff_reason else "",
                    )
                )

                # Renderiza com o dict cru (o template já está preparado para mostrar dict)
                return render(request, self.template_name, self._ctx(plant, cred, result=stats))

            # ---------------- FETCH (somente mostrar tabela) ----------------
            if action == "fetch":
                start_day = (request.POST.get("start_day") or "").strip()
                end_day = (request.POST.get("end_day") or "").strip()
                if not (start_day and end_day):
                    raise ValueError("Informe start_day e end_day (YYYY-MM-DD).")

                pn = (request.POST.get("pn") or getattr(cred, "shinemonitor_pn", "")).strip()
                devcode = (request.POST.get("devcode") or getattr(cred, "shinemonitor_devcode", "")).strip()
                devaddr = request.POST.get("devaddr") or getattr(cred, "shinemonitor_devaddr", None)
                sn = (request.POST.get("sn") or getattr(cred, "shinemonitor_sn", "")).strip()

                if not (pn and devcode and devaddr is not None and sn):
                    raise ValueError("Dispositivo incompleto. Faça o discovery/vínculo primeiro.")

                table = fetch_range_table(
                    username, password,
                    pn=pn, devcode=devcode, devaddr=int(devaddr), sn=sn,
                    start_day=start_day, end_day=end_day,
                    i18n=getattr(cred, "shinemonitor_i18n", None) or "pt_BR",
                    lang=getattr(cred, "shinemonitor_lang", None) or "pt_BR",
                    pagesize=50
                )

                table_norm, rows_norm = self._normalize_result_rows(table)

                ctx = self._ctx(plant, cred, result=table_norm)
                ctx["rows"] = rows_norm
                ctx["meta"] = table_norm.get("meta", {}) if isinstance(table_norm, dict) else {}

                messages.success(request, f"Dados carregados: {len(rows_norm)} linhas.")
                return render(request, self.template_name, ctx)

            messages.error(request, "Ação inválida.")
            return redirect("core:renovigi_console", pk=plant.pk)

        except Exception as exc:
            messages.error(request, f"Falha: {exc}")
            return render(request, self.template_name, self._ctx(plant, cred, result=None))


class PlantOperationalDataListView(LoginRequiredMixin, View):
    template_name = "plants/opdata_list.html"

    def get(self, request, pk: int):
        plant = get_object_or_404(PVPlant, pk=pk, owner=request.user)

        # Defaults: últimos 7 dias (UTC)
        today_utc = datetime.now(timezone.utc).date()
        default_start = today_utc - timedelta(days=7)
        default_end = today_utc

        start_s = (request.GET.get("start") or str(default_start)).strip()
        end_s = (request.GET.get("end") or str(default_end)).strip()

        start_d = parse_date(start_s) or default_start
        end_d = parse_date(end_s) or default_end

        # Range UTC [start 00:00, end+1 00:00)
        start_dt = datetime(start_d.year, start_d.month, start_d.day, tzinfo=timezone.utc)
        end_dt = datetime(end_d.year, end_d.month, end_d.day, tzinfo=timezone.utc) + timedelta(days=1)

        pn = (request.GET.get("pn") or "").strip()
        sn = (request.GET.get("sn") or "").strip()

        qs = (
            InverterOperationalData.objects
            .filter(plant=plant, ts_utc__gte=start_dt, ts_utc__lt=end_dt)
            .order_by("-ts_utc")
            .only("id", "ts_utc", "pn", "devcode", "devaddr", "sn", "payload")
        )

        if pn:
            qs = qs.filter(pn=pn)
        if sn:
            qs = qs.filter(sn=sn)

        page_size = int(request.GET.get("page_size") or 200)
        page_size = max(20, min(page_size, 1000))

        paginator = Paginator(qs, page_size)
        page_number = request.GET.get("page") or 1
        page_obj = paginator.get_page(page_number)

        devices = (
            InverterOperationalData.objects
            .filter(plant=plant)
            .values("pn", "sn")
            .distinct()
            .order_by("pn", "sn")
        )

        ctx = {
            "plant": plant,
            "page_obj": page_obj,
            "start": start_d.isoformat(),
            "end": end_d.isoformat(),
            "pn": pn,
            "sn": sn,
            "page_size": page_size,
            "devices": devices,
        }
        return render(request, self.template_name, ctx)


#---------------------------
#---------------------------  JUNTAR BASES
#---------------------------

def _local_dates_to_utc_range(start_date, end_date, tz_name: str) -> tuple[datetime, datetime]:
    """
    Converte [start_date, end_date] (datas locais da planta) em intervalo UTC [start, end).

    Exemplo (America/Maceio, UTC-03):
      start_date=2025-12-31 -> start_utc=2025-12-31T03:00Z
      end_date=2025-12-31   -> end_utc  =2026-01-01T03:00Z
    """
    tz = ZoneInfo(tz_name or "UTC")

    # Crie datetimes locais tz-aware corretamente (sem replace(tzinfo=...))
    start_local = datetime.combine(start_date, time.min, tzinfo=tz)
    end_local_excl = datetime.combine(end_date, time.min, tzinfo=tz) + timedelta(days=1)

    return start_local.astimezone(timezone.utc), end_local_excl.astimezone(timezone.utc)


def _df_preview(df: pd.DataFrame, tz_name: str, n: int = 60) -> tuple[list[str], list[list]]:
    """
    Converte o índice (assumindo que o índice representa UTC) para o timezone da planta
    e retorna:
      - cols: lista de nomes de colunas
      - rows: lista de listas (valores alinhados com cols)

    Regras:
    - Se o índice for tz-naive: assume UTC e tz_localize("UTC")
    - Se for tz-aware: tz_convert("UTC") antes, e depois tz_convert(tz da planta)
    - Formata ts_local como string para a UI (evita exibir offset -03:00)
    """
    if df is None or df.empty:
        return [], []

    tz = ZoneInfo(tz_name or "UTC")
    d = df.copy()

    # --- Garantir índice datetime tz-aware (assumindo que o índice representa UTC) ---
    if not isinstance(d.index, pd.DatetimeIndex):
        # se vier sem DatetimeIndex, tenta converter
        d.index = pd.to_datetime(d.index, errors="coerce")

    if not isinstance(d.index, pd.DatetimeIndex):
        return [], []

    # Se index tz-naive, assume UTC
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        # normaliza para UTC antes de converter para tz local
        d.index = d.index.tz_convert("UTC")

    # Converte para tz local da planta
    d.index = d.index.tz_convert(tz)

    # Recorta e leva índice para coluna
    d = d.head(n).reset_index()

    # Normaliza nome da coluna de tempo (o reset_index usa o nome do índice; se não houver, "index")
    # No seu merge: d.index.name == "ts_15" (bom), então a coluna será "ts_15".
    if "ts_15" in d.columns:
        d = d.rename(columns={"ts_15": "ts_local"})
    elif "ts_utc" in d.columns:
        d = d.rename(columns={"ts_utc": "ts_local"})
    elif "index" in d.columns:
        d = d.rename(columns={"index": "ts_local"})
    else:
        # fallback: primeira coluna
        d = d.rename(columns={d.columns[0]: "ts_local"})

    # Formata timestamp para UI (sem offset/timezone)
    if "ts_local" in d.columns:
        d["ts_local"] = pd.to_datetime(d["ts_local"], errors="coerce")
        d["ts_local"] = d["ts_local"].dt.strftime("%d/%m/%Y %H:%M")

    # Arredonda numéricos para preview (inclui floats/ints)
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            # não mexe em flags booleanas
            if pd.api.types.is_bool_dtype(d[c]):
                continue
            d[c] = pd.to_numeric(d[c], errors="coerce").round(3)

    cols = list(d.columns)
    rows = d.to_numpy(dtype=object).tolist()
    return cols, rows


# -----------------------------------------------------------------------------
# View
# -----------------------------------------------------------------------------

@login_required
def merge_run_view(request: HttpRequest) -> HttpResponse:
    """
    Tela para executar merge e (opcionalmente) persistir base casada 15 min.
    """
    stats = None
    cols15, rows15 = [], []
    colsh, rowsh = [], []

    if request.method == "POST":
        form = MergeRunForm(request.POST, user=request.user)
        if form.is_valid():
            plant = form.cleaned_data["plant"]
            start_date = form.cleaned_data["start_date"]
            end_date = form.cleaned_data["end_date"]
            persist = bool(form.cleaned_data.get("persist"))
            want_hourly = bool(form.cleaned_data.get("want_hourly"))
            source_oper = (form.cleaned_data.get("source_oper") or "SHINEMONITOR").strip()
            source_meteo = (form.cleaned_data.get("source_meteo") or "OPENMETEO").strip()

            tz_name = getattr(plant, "timezone", None) or "UTC"

            # intervalo UTC [start,end)
            dt_start_utc, dt_end_utc = _local_dates_to_utc_range(start_date, end_date, tz_name)

            # executa merge
            run = build_plant_merged_dataset(
                plant=plant,
                dt_start_utc=dt_start_utc,
                dt_end_utc=dt_end_utc,
                want_hourly=want_hourly,
                persist=persist,
                source_oper=source_oper,
                source_meteo=source_meteo,
                interval_min=15,
            )

            stats = run.stats

            # Preview com conversão correta para timezone local
            cols15, rows15 = _df_preview(run.df15, tz_name=tz_name, n=120)

            if want_hourly:
                colsh, rowsh = _df_preview(run.df_hour, tz_name=tz_name, n=72)

            if persist:
                messages.success(
                    request,
                    f"Merge executado e persistido: {stats.get('saved_rows_15m', 0)} linhas (15 min).",
                )
            else:
                messages.info(
                    request,
                    f"Merge executado (sem persistir). Linhas 15 min: {stats.get('merged_rows_15', 0)}.",
                )
    else:
        form = MergeRunForm(user=request.user)

    return render(
        request,
        "merge/merge_run.html",
        {
            "form": form,
            "stats": stats,
            "cols15": cols15,
            "rows15": rows15,
            "colsh": colsh,
            "rowsh": rowsh,
        },
    )






#---------------------------
#---------------------------  D A S H B O A R D
#---------------------------

def _safe_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "UTC")
    except Exception:
        return ZoneInfo("UTC")


def _local_dates_to_utc_range(start_date: date, end_date: date, tz_name: str) -> tuple[datetime, datetime]:
    """
    Converte [start_date, end_date] (datas locais) -> intervalo UTC [start, end).
    """
    tz = _safe_zoneinfo(tz_name)
    start_local = datetime.combine(start_date, time.min, tzinfo=tz)
    end_local_excl = datetime.combine(end_date, time.min, tzinfo=tz) + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local_excl.astimezone(timezone.utc)


def _to_local_iso(ts_utc: datetime, tz: ZoneInfo) -> str:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    return ts_utc.astimezone(tz).isoformat()


def _to_local_label(ts_utc: datetime, tz: ZoneInfo) -> str:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    dt_local = ts_utc.astimezone(tz)
    return dt_local.strftime("%d/%m %H:%M")


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _pick_sources_for_plant_in_range(
    plant: PVPlant,
    dt0_utc: datetime,
    dt1_utc: datetime,
    *,
    interval_min: int = 15,
) -> tuple[Optional[str], Optional[str]]:
    """
    Escolhe source_oper/source_meteo baseado no registro MAIS RECENTE DENTRO DO INTERVALO.
    Se não houver registro no intervalo, retorna (None, None).
    """
    last = (
        PVPlantMergedRecord15m.objects
        .filter(plant=plant, interval_min=interval_min, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc)
        .order_by("-ts_utc")
        .values("source_oper", "source_meteo")
        .first()
    )
    if not last:
        return None, None
    return last.get("source_oper"), last.get("source_meteo")


def _pick_sources_for_plant_global(
    plant: PVPlant,
    *,
    interval_min: int = 15,
) -> tuple[Optional[str], Optional[str]]:
    """
    Fallback: escolhe source_oper/source_meteo baseado no último registro global.
    """
    last = (
        PVPlantMergedRecord15m.objects
        .filter(plant=plant, interval_min=interval_min)
        .order_by("-ts_utc")
        .values("source_oper", "source_meteo")
        .first()
    )
    if not last:
        return None, None
    return last.get("source_oper"), last.get("source_meteo")


def _available_source_pairs_in_range(
    plant: PVPlant,
    dt0_utc: datetime,
    dt1_utc: datetime,
    *,
    interval_min: int = 15,
    limit: int = 20,
) -> List[Dict[str, str]]:
    """
    Lista combinações existentes no intervalo, para debug/UX.
    """
    pairs = (
        PVPlantMergedRecord15m.objects
        .filter(plant=plant, interval_min=interval_min, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc)
        .values("source_oper", "source_meteo")
        .distinct()
        .order_by("source_oper", "source_meteo")[:limit]
    )
    out = []
    for p in pairs:
        so = p.get("source_oper")
        sm = p.get("source_meteo")
        if so and sm:
            out.append({"source_oper": so, "source_meteo": sm})
    return out

def _get_merged15m_model():
    # Ajuste "core" se o app_label for outro
    return apps.get_model("core", "PVPlantMergedRecord15m")


def _pick_latest_sources_for_plant(plant):
    """
    Descobre automaticamente quais sources (oper/meteo) existem na base merged para a planta,
    usando o registro mais recente como referência.
    """
    M = _get_merged15m_model()

    last = (
        M.objects
        .filter(plant=plant, interval_min=15)
        .exclude(source_oper__isnull=True)
        .exclude(source_meteo__isnull=True)
        .order_by("-ts_utc")
        .values("source_oper", "source_meteo")
        .first()
    )

    if not last:
        return None, None

    return last.get("source_oper"), last.get("source_meteo")
# ---------------------------
# Views
# ---------------------------
logger = logging.getLogger(__name__)

# ----------------------------
# JSON estrito (evita NaN/Inf)
# ----------------------------
try:
    import numpy as np  # type: ignore
except Exception:
    np = None


def _json_safe(x: Any) -> Any:
    """
    Converte payload para JSON estrito:
      - NaN/Inf -> None (null)
      - numpy types -> python types
      - ndarray -> list
      - datetime/date -> isoformat
    """
    if x is None:
        return None

    # numpy
    if np is not None:
        if isinstance(x, (np.floating,)):
            xf = float(x)
            return xf if math.isfinite(xf) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.ndarray,)):
            return [_json_safe(v) for v in x.tolist()]

    # python float
    if isinstance(x, float):
        return x if math.isfinite(x) else None

    # datetime/date
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # containers
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    return x


def _json_response_strict(payload: Dict[str, Any], *, status: int = 200) -> JsonResponse:
    payload = _json_safe(payload)
    return JsonResponse(
        payload,
        status=status,
        json_dumps_params={"ensure_ascii": False, "allow_nan": False},
    )


# -------------------------------------------------------------------
# Helpers externos (assumidos já existentes no seu projeto)
# -------------------------------------------------------------------
def _local_dates_to_utc_range(start_date: date, end_date: date, tz_name: str):
    """
    Deve retornar (dt0_utc, dt1_utc) no formato timezone-aware UTC,
    onde dt1_utc é exclusivo.
    """
    tz = ZoneInfo(tz_name or "UTC")
    dt0_local = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=tz)
    dt1_local = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=tz)
    return dt0_local.astimezone(timezone.utc), dt1_local.astimezone(timezone.utc)


def _pick_latest_sources_for_plant(plant: PVPlant):
    """
    Deve retornar (source_oper, source_meteo) mais recentes disponíveis para a planta,
    ou (None, None) se não houver.
    """
    # Implementação típica (exemplo): pegue a última combinação de sources no merged_15m
    q = (
        PVPlantMergedRecord15m.objects.filter(plant=plant, interval_min=15)
        .order_by("-ts_utc")
        .values_list("source_oper", "source_meteo")
        .first()
    )
    if not q:
        return None, None
    return q[0], q[1]


# ---------------------------
# Views
# ---------------------------
@login_required
def pv_dashboard_view(request: HttpRequest) -> HttpResponse:
    qs = PVPlant.objects.all().order_by("nome")
    if not request.user.is_superuser:
        qs = qs.filter(owner=request.user)

    plants = list(qs.values("id", "nome", "timezone"))

    today = date.today()
    default_start = today - timedelta(days=2)
    default_end = today

    default_plant_id = plants[0]["id"] if plants else None

    return render(
        request,
        "dashboard/pv_dashboard.html",
        {
            "plants": plants,
            "default_plant_id": default_plant_id,
            "default_start": default_start.isoformat(),
            "default_end": default_end.isoformat(),
            "heatmap_year": today.year,
        },
    )


@require_GET
@login_required
def pv_dashboard_timeseries_api(request: HttpRequest) -> JsonResponse:
    """
    Retorna JSON com séries e KPIs (eixo X em horário local),
    baseado em PVPlantMergedRecord15m.

    - Agrega TODAS as sources operativas disponíveis no intervalo (ou as escolhidas),
      e também retorna `series_by_source`.

    IMPORTANTE:
    - JSON estrito (sem NaN/Inf).
    - Esta view NÃO referencia fuzzy (não importa nem chama).
      Se o power_model retornar rca_label/valid, apenas repassa.
    """
    import math
    import inspect
    from collections import OrderedDict
    from datetime import datetime, timezone, date
    from zoneinfo import ZoneInfo
    from typing import Any, Dict, List, Optional

    # ----------------------------
    # Helpers locais
    # ----------------------------
    def f(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if not math.isfinite(x):
            return None
        return x

    def s_or_none(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _safe_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    def _safe_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            x = float(v)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def _mean_none(vals: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in vals if x is not None and math.isfinite(x)]
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    def _sum_none(vals: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in vals if x is not None and math.isfinite(x)]
        if not xs:
            return None
        return float(sum(xs))

    def _build_plant_info(plant_obj: PVPlant) -> Dict[str, Any]:
        try:
            details_obj = plant_obj.details
        except Exception:
            details_obj = None

        module_obj = getattr(details_obj, "module", None) if details_obj else None
        inverter_obj = getattr(details_obj, "inverter", None) if details_obj else None

        strings_count = _safe_int(getattr(details_obj, "strings_count", None)) if details_obj else None
        mps = _safe_int(getattr(details_obj, "modules_per_string", None)) if details_obj else None
        modules_total = _safe_int(getattr(details_obj, "modules_total", None)) if details_obj else None

        string_groups = []
        if details_obj and getattr(details_obj, "pk", None):
            try:
                string_groups = list(
                    details_obj.string_configs.order_by("order", "id").values(
                        "id", "name", "order", "mppt", "strings_qty", "modules_per_string"
                    )
                )
            except Exception:
                string_groups = []

        if string_groups:
            sc2 = 0
            mt2 = 0
            for g in string_groups:
                sq = _safe_int(g.get("strings_qty")) or 0
                ns = _safe_int(g.get("modules_per_string")) or 0
                sc2 += sq
                mt2 += sq * ns
            if sc2 > 0:
                strings_count = sc2
            if mt2 > 0:
                modules_total = mt2

            mps_set = {int(g["modules_per_string"]) for g in string_groups if g.get("modules_per_string")}
            mps = (mps_set.pop() if len(mps_set) == 1 else None)

        if (modules_total is None or modules_total == 0) and (strings_count is not None) and (mps is not None):
            modules_total = strings_count * mps

        tilt = _safe_float(getattr(details_obj, "tilt_deg", None)) if details_obj else None
        az = _safe_float(getattr(details_obj, "azimuth_deg", None)) if details_obj else None
        ksys = _safe_float(getattr(details_obj, "k_sys", None)) if details_obj else None
        noct = _safe_float(getattr(details_obj, "noct_c", None)) if details_obj else None

        mod_model = s_or_none(getattr(module_obj, "model", None) or getattr(module_obj, "nome", None) or getattr(module_obj, "name", None))
        mod_mfr = s_or_none(getattr(module_obj, "manufacturer", None) or getattr(module_obj, "marca", None))
        mod_pstc = getattr(module_obj, "p_stc_w", None) or getattr(module_obj, "pmp_stc_w", None) or getattr(module_obj, "p_stc", None)

        inv_model = s_or_none(getattr(inverter_obj, "model", None) or getattr(inverter_obj, "nome", None) or getattr(inverter_obj, "name", None))
        inv_mfr = s_or_none(getattr(inverter_obj, "manufacturer", None) or getattr(inverter_obj, "marca", None))
        inv_pac = getattr(inverter_obj, "p_ac_rated_w", None) or getattr(inverter_obj, "pac_rated_w", None) or getattr(inverter_obj, "p_nom_w", None)

        return {
            "module": {
                "id": getattr(module_obj, "id", None),
                "model": mod_model,
                "manufacturer": mod_mfr,
                "p_stc_w": _safe_float(mod_pstc),
            },
            "inverter": {
                "id": getattr(inverter_obj, "id", None),
                "model": inv_model,
                "manufacturer": inv_mfr,
                "p_ac_rated_w": _safe_float(inv_pac),
            },
            "electrical": {
                "strings_count": strings_count,
                "modules_per_string": mps,
                "modules_total": modules_total,
                "string_groups": string_groups,
            },
            "geometry": {
                "tilt_deg": tilt,
                "azimuth_deg": az,
                "k_sys": ksys,
                "noct_c": noct,
            },
        }

    def _empty_payload(
        *,
        plant: Optional[PVPlant],
        tz_name: str,
        start_s: str,
        end_s: str,
        dt0_utc=None,
        dt1_utc=None,
        src_oper=None,
        src_oper_list=None,
        src_meteo=None,
        message: str = "",
    ):
        plant_info = {}
        if plant is not None:
            try:
                plant_info = _build_plant_info(plant)
            except Exception:
                plant_info = {}

        return {
            "ok": True,
            "empty": True,
            "message": message or "Sem dados.",
            "plant": {"id": getattr(plant, "id", None), "nome": getattr(plant, "nome", None), "tz": tz_name},
            "plant_info": plant_info,
            "range": {
                "start_local": start_s or None,
                "end_local": end_s or None,
                "dt0_utc": dt0_utc.isoformat() if dt0_utc else None,
                "dt1_utc": dt1_utc.isoformat() if dt1_utc else None,
            },
            "sources": {
                "source_oper": src_oper,
                "source_oper_list": src_oper_list or [],
                "source_meteo": src_meteo
            },
            "x": [],
            "x_label": [],
            "series": {},
            "series_by_source": {},
            "charts": {
                "gauge": {"score": None, "gcv_stat": None, "label": "Indisponível"},
                "scatter": {"times": [], "x_gcv": [], "y_mismatch": [], "code": [], "code_name": []},
                "sankey": {"nodes": [], "links": [], "values_kwh": {}},
                "timeline": {"times": [], "code": [], "code_hyst": [], "dt_minutes": 15.0, "min_persist_minutes": 60.0},
            },
            "kpis": {"points": 0},
            "audit": {"model_ok": False, "model_error": None, "model_meta": {}, "g_used": "—"},
            "debug": {"has_model": False, "model_error": None},
        }

    # ----------------------------
    # Inputs
    # ----------------------------
    try:
        plant_id = int(request.GET.get("plant_id", "0"))
    except Exception:
        return _json_response_strict({"ok": False, "error": "plant_id inválido"}, status=400)

    start_s = (request.GET.get("start") or "").strip()
    end_s = (request.GET.get("end") or "").strip()

    if not start_s or not end_s:
        return _json_response_strict({"ok": False, "error": "start e end são obrigatórios (YYYY-MM-DD)"}, status=400)

    try:
        start_date = date.fromisoformat(start_s)
        end_date = date.fromisoformat(end_s)
    except Exception:
        return _json_response_strict({"ok": False, "error": "start/end devem estar no formato YYYY-MM-DD"}, status=400)

    if end_date < start_date:
        return _json_response_strict({"ok": False, "error": "end não pode ser menor que start"}, status=400)

    # ----------------------------
    # Planta + permissão
    # ----------------------------
    plant = PVPlant.objects.filter(id=plant_id).first()
    if not plant:
        return _json_response_strict({"ok": False, "error": "Planta não encontrada"}, status=404)

    if (not request.user.is_superuser) and (plant.owner_id != request.user.id):
        return _json_response_strict({"ok": False, "error": "Sem permissão para esta planta"}, status=403)

    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"
        tz = ZoneInfo("UTC")

    dt0_utc, dt1_utc = _local_dates_to_utc_range(start_date, end_date, tz_name)

    try:
        plant_info = _build_plant_info(plant)
    except Exception:
        plant_info = {}

    # ----------------------------
    # Sources (multi source_oper)
    # ----------------------------
    src_oper_raw = (request.GET.get("source_oper") or "").strip() or None
    src_meteo = (request.GET.get("source_meteo") or "").strip() or None

    if not src_oper_raw or not src_meteo:
        auto_oper, auto_meteo = _pick_latest_sources_for_plant(plant)
        src_meteo = src_meteo or auto_meteo

    if not src_meteo:
        payload = _empty_payload(
            plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
            dt0_utc=dt0_utc, dt1_utc=dt1_utc,
            src_oper=None, src_oper_list=[], src_meteo=src_meteo,
            message="Não há dados merged_15m para esta planta ainda (source_meteo ausente).",
        )
        return _json_response_strict(payload)

    avail_oper = list(
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_meteo=src_meteo,
            interval_min=15,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        ).values_list("source_oper", flat=True).distinct()
    )

    if not avail_oper:
        avail_oper = list(
            PVPlantMergedRecord15m.objects.filter(
                plant=plant,
                interval_min=15,
                ts_utc__gte=dt0_utc,
                ts_utc__lt=dt1_utc,
            ).values_list("source_oper", flat=True).distinct()
        )

    src_oper_list: List[str] = []
    if src_oper_raw:
        if src_oper_raw.strip().upper() == "ALL":
            src_oper_list = list(avail_oper)
        else:
            src_oper_list = [s.strip() for s in src_oper_raw.split(",") if s.strip()]
    else:
        src_oper_list = list(avail_oper)

    if avail_oper:
        avail_set = set(avail_oper)
        src_oper_list = [s for s in src_oper_list if s in avail_set]

    if not src_oper_list:
        payload = _empty_payload(
            plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
            dt0_utc=dt0_utc, dt1_utc=dt1_utc,
            src_oper=src_oper_raw, src_oper_list=[], src_meteo=src_meteo,
            message="Sem dados merged_15m no intervalo para as fontes operativas selecionadas.",
        )
        return _json_response_strict(payload)

    src_oper = src_oper_list[0] if src_oper_list else None

    # ----------------------------
    # Query merged_15m (source_oper__in)
    # ----------------------------
    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_oper__in=src_oper_list,
            source_meteo=src_meteo,
            interval_min=15,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .order_by("ts_utc", "source_oper")
        .values(
            "ts_utc",
            "source_oper",
            "p_ac_w", "p_dc_w", "e_ac_wh_15",
            "v_dc_v", "i_dc_a", "v_ac_v", "i_ac_a",
            "ghi", "gti", "dni", "dhi",
            "temp_air", "wind_speed", "rh",
            "inv_coverage",
            "flag_meteo_missing", "flag_inv_missing",
        )
    )

    rows = list(qs)
    if not rows:
        payload = _empty_payload(
            plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
            dt0_utc=dt0_utc, dt1_utc=dt1_utc,
            src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo,
            message="Sem pontos no intervalo selecionado.",
        )
        return _json_response_strict(payload)

    # ----------------------------
    # Pivô por timestamp
    # ----------------------------
    rec_by_ts: "OrderedDict[datetime, Dict[str, Dict[str, Any]]]" = OrderedDict()
    sources_set = set()

    for r in rows:
        ts_utc = r.get("ts_utc")
        if isinstance(ts_utc, str):
            ts_utc = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        if ts_utc is None:
            continue
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)

        src = (r.get("source_oper") or "").strip() or "unknown"
        sources_set.add(src)
        rec_by_ts.setdefault(ts_utc, {})[src] = r

    if not rec_by_ts:
        payload = _empty_payload(
            plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
            dt0_utc=dt0_utc, dt1_utc=dt1_utc,
            src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo,
            message="Sem timestamps válidos no intervalo (ts_utc ausente/inválido).",
        )
        return _json_response_strict(payload)

    sources = sorted(sources_set)

    # ----------------------------
    # Séries (agregadas + por source)
    # ----------------------------
    x_iso: List[str] = []
    x_label: List[str] = []
    t_utc: List[datetime] = []

    p_ac: List[Optional[float]] = []
    p_dc: List[Optional[float]] = []
    v_dc: List[Optional[float]] = []
    i_dc: List[Optional[float]] = []
    v_ac: List[Optional[float]] = []
    i_ac: List[Optional[float]] = []

    ghi: List[Optional[float]] = []
    gti: List[Optional[float]] = []
    dni: List[Optional[float]] = []
    dhi: List[Optional[float]] = []
    temp_air: List[Optional[float]] = []
    wind: List[Optional[float]] = []
    rh: List[Optional[float]] = []

    e_wh_total = 0.0
    p_ac_max = None
    ghi_max = None

    inv_cov_vals: List[float] = []
    met_missing_ts = 0
    inv_missing_count = 0
    inv_total_points = 0

    series_by_source: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for src in sources:
        series_by_source[src] = {
            "p_ac_w": [],
            "p_dc_w": [],
            "v_dc_v": [],
            "i_dc_a": [],
            "v_ac_v": [],
            "i_ac_a": [],
            "inv_coverage": [],
            "flag_inv_missing": [],
        }

    for ts_utc, per_src in rec_by_ts.items():
        t_utc.append(ts_utc)

        ts_local = ts_utc.astimezone(tz)
        x_iso.append(ts_local.isoformat())
        x_label.append(ts_local.strftime("%d/%m %H:%M"))

        pac_vals: List[Optional[float]] = []
        pdc_vals: List[Optional[float]] = []
        e15_vals: List[Optional[float]] = []
        vdc_vals: List[Optional[float]] = []
        idc_vals: List[Optional[float]] = []
        vac_vals: List[Optional[float]] = []
        iac_vals: List[Optional[float]] = []
        cov_vals: List[Optional[float]] = []

        first_row = None
        if per_src:
            for s0 in sources:
                if s0 in per_src:
                    first_row = per_src[s0]
                    break
            if first_row is None:
                first_row = next(iter(per_src.values()))

        g_ghi = f(first_row.get("ghi")) if first_row else None
        g_gti = f(first_row.get("gti")) if first_row else None
        g_dni = f(first_row.get("dni")) if first_row else None
        g_dhi = f(first_row.get("dhi")) if first_row else None
        g_ta = f(first_row.get("temp_air")) if first_row else None
        g_ws = f(first_row.get("wind_speed")) if first_row else None
        g_rh = f(first_row.get("rh")) if first_row else None

        ghi.append(g_ghi)
        gti.append(g_gti)
        dni.append(g_dni)
        dhi.append(g_dhi)
        temp_air.append(g_ta)
        wind.append(g_ws)
        rh.append(g_rh)

        if g_ghi is not None:
            ghi_max = g_ghi if (ghi_max is None or g_ghi > ghi_max) else ghi_max

        if first_row and bool(first_row.get("flag_meteo_missing")):
            met_missing_ts += 1

        for src in sources:
            r = per_src.get(src)

            pac = f(r.get("p_ac_w")) if r else None
            pdc = f(r.get("p_dc_w")) if r else None
            e15 = f(r.get("e_ac_wh_15")) if r else None
            vdc = f(r.get("v_dc_v")) if r else None
            idc = f(r.get("i_dc_a")) if r else None
            vac = f(r.get("v_ac_v")) if r else None
            iac = f(r.get("i_ac_a")) if r else None
            cov = f(r.get("inv_coverage")) if r else None

            flag_inv_missing = bool(r.get("flag_inv_missing")) if r else True

            series_by_source[src]["p_ac_w"].append(pac)
            series_by_source[src]["p_dc_w"].append(pdc)
            series_by_source[src]["v_dc_v"].append(vdc)
            series_by_source[src]["i_dc_a"].append(idc)
            series_by_source[src]["v_ac_v"].append(vac)
            series_by_source[src]["i_ac_a"].append(iac)
            series_by_source[src]["inv_coverage"].append(cov)
            series_by_source[src]["flag_inv_missing"].append(bool(flag_inv_missing))

            pac_vals.append(pac)
            pdc_vals.append(pdc)
            e15_vals.append(e15)
            vdc_vals.append(vdc)
            idc_vals.append(idc)
            vac_vals.append(vac)
            iac_vals.append(iac)
            cov_vals.append(cov)

            if cov is not None:
                inv_cov_vals.append(float(cov))

            inv_total_points += 1
            if flag_inv_missing:
                inv_missing_count += 1

        pac_total = _sum_none(pac_vals)
        pdc_total = _sum_none(pdc_vals)

        vdc_agg = _mean_none(vdc_vals)
        idc_agg = _sum_none(idc_vals)  # <<< corrigido p/ multi-fontes: soma

        vac_agg = _mean_none(vac_vals)
        iac_agg = _sum_none(iac_vals)  # <<< coerente com multi-fontes

        p_ac.append(pac_total)
        p_dc.append(pdc_total)
        v_dc.append(vdc_agg)
        i_dc.append(idc_agg)
        v_ac.append(vac_agg)
        i_ac.append(iac_agg)

        if pac_total is not None:
            p_ac_max = pac_total if (p_ac_max is None or pac_total > p_ac_max) else p_ac_max

        e15_total = _sum_none(e15_vals)
        if e15_total is not None:
            e_wh_total += float(e15_total)

    n = len(x_iso)
    if n == 0:
        payload = _empty_payload(
            plant=plant, tz_name=tz_name, start_s=start_s, end_s=end_s,
            dt0_utc=dt0_utc, dt1_utc=dt1_utc,
            src_oper=src_oper, src_oper_list=src_oper_list, src_meteo=src_meteo,
            message="Sem pontos válidos após agregação por timestamp.",
        )
        return _json_response_strict(payload)

    # ----------------------------
    # MODELO: power_model.py (sem fuzzy)
    # ----------------------------
    dt_minutes = 15.0
    persist_minutes = 60.0

    p_ac_model_w = None
    mismatch_rel = None
    tcell_c = None
    e_model_kwh = None

    p_ac_pu_model = None
    p_ac_pu_real = None
    pr_model_inst = None
    pr_real_inst = None
    g_std_60m = None
    g_cv_60m = None
    csi = None
    eta_inv = None

    pac_model_pu_stc = None
    pac_real_pu_stc = None
    k_cs = None
    g_poa_used = None

    # apenas repassa se power_model fornecer (sem fuzzy)
    rca_label = None
    valid_model = None

    charts = {
        "gauge": {"score": None, "gcv_stat": None, "label": "Indisponível"},
        "scatter": {"times": [], "x_gcv": [], "y_mismatch": [], "code": [], "code_name": []},
        "sankey": {"nodes": [], "links": [], "values_kwh": {}},
        "timeline": {"times": [], "code": [], "code_hyst": [], "dt_minutes": dt_minutes, "min_persist_minutes": persist_minutes},
    }

    audit = {"model_ok": False, "model_error": None, "model_meta": {}, "g_used": "—"}

    try:
        import numpy as _np
        from dataclasses import asdict

        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
            transpose_ghi_to_poa_isotropic,
        )

        def list_to_np_nan(xs):
            return _np.array([_np.nan if v is None else float(v) for v in (xs or [])], dtype=float)

        def np_to_list_none(a):
            a = _np.asarray(a, dtype=float)
            return [None if (not _np.isfinite(v)) else float(v) for v in a.tolist()]

        def np_to_list_str(a):
            a = _np.asarray(a, dtype=object)
            out = []
            for v in a.tolist():
                out.append(None if v is None else str(v))
            return out

        def np_to_list_bool(a):
            a = _np.asarray(a, dtype=bool)
            return [bool(v) for v in a.tolist()]

        # charts builder (se existir)
        build_dashboard_payload = None
        try:
            from core.services.dashboard.dashboard_charts import build_dashboard_payload as _build
            build_dashboard_payload = _build
        except Exception:
            build_dashboard_payload = None

        details = getattr(plant, "details", None)

        if details and getattr(details, "module_id", None):
            n_mod = int(getattr(details, "modules_total", 0) or 0)

            if n_mod > 0:
                mod = module_from_pvmodule(details.module)
                inv = getattr(details, "inverter", None)

                pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

                # fallback de geometria via PVPlant (se necessário)
                pld = asdict(pl)
                if pld.get("lat_deg") is None:
                    pld["lat_deg"] = _safe_float(getattr(plant, "latitude", None) or getattr(plant, "latitude_deg", None) or getattr(plant, "lat_deg", None))
                if pld.get("lon_deg") is None:
                    pld["lon_deg"] = _safe_float(getattr(plant, "longitude", None) or getattr(plant, "longitude_deg", None) or getattr(plant, "lon_deg", None))
                if pld.get("tilt_deg") is None:
                    pld["tilt_deg"] = _safe_float(getattr(details, "tilt_deg", None) or getattr(plant, "tilt_deg", None))
                if pld.get("azimuth_deg") is None:
                    pld["azimuth_deg"] = _safe_float(getattr(details, "azimuth_deg", None) or getattr(plant, "azimuth_deg", None))
                pl = pl.__class__(**pld)

                gti_np = list_to_np_nan(gti)
                ghi_np = list_to_np_nan(ghi)
                dni_np = list_to_np_nan(dni)
                dhi_np = list_to_np_nan(dhi)

                gti_ok = _np.isfinite(gti_np)
                ghi_ok = _np.isfinite(ghi_np)

                gpoa_np = _np.where(gti_ok, gti_np, _np.nan)

                can_transpose = (
                    (pl.lat_deg is not None) and (pl.lon_deg is not None) and
                    (pl.tilt_deg is not None) and (pl.azimuth_deg is not None) and
                    (len(t_utc) == len(ghi)) and _np.any(ghi_ok)
                )

                if can_transpose and (not _np.all(gti_ok)):
                    trans = transpose_ghi_to_poa_isotropic(
                        ghi=ghi_np,
                        dhi=dhi_np if _np.any(_np.isfinite(dhi_np)) else None,
                        dni=dni_np if _np.any(_np.isfinite(dni_np)) else None,
                        times_utc=t_utc,
                        lat_deg=float(pl.lat_deg),
                        lon_deg=float(pl.lon_deg),
                        tilt_deg=float(pl.tilt_deg),
                        azimuth_deg=float(pl.azimuth_deg),
                        albedo=float(pl.albedo),
                    )
                    gpoa_tr = _np.asarray(trans.get("g_poa"), dtype=float)
                    gpoa_np = _np.where(gti_ok, gti_np, gpoa_tr)
                    audit["g_used"] = "GTI + POA(transposição) p/ faltas" if _np.any(gti_ok) else "POA(transposição)"
                elif _np.any(gti_ok):
                    audit["g_used"] = "GTI"
                else:
                    gpoa_np = ghi_np
                    audit["g_used"] = "GHI (sem POA/geo)"

                g_poa_used = np_to_list_none(gpoa_np)

                tamb_np = list_to_np_nan(temp_air)
                pac_real_np = list_to_np_nan(p_ac)

                # medidos (agregados) para v/i ratio se o model suportar
                vdc_np = list_to_np_nan(v_dc)
                idc_np = list_to_np_nan(i_dc)

                sig = inspect.signature(expected_and_mismatch)
                kwargs = dict(
                    g_poa=gpoa_np,
                    tamb_c=tamb_np,
                    pac_real_w=pac_real_np,
                    module=mod,
                    plant=pl,
                    g_min_valid=0.0,
                    n_points=60,
                    eps_w=50.0,
                )

                if "dt_minutes" in sig.parameters:
                    kwargs["dt_minutes"] = dt_minutes
                if "window_minutes" in sig.parameters:
                    kwargs["window_minutes"] = 60.0

                # >>> alinhamento temporal (se suportado)
                if "times_utc" in sig.parameters:
                    kwargs["times_utc"] = t_utc

                # >>> nomes “novos” (e compat antigos)
                if "v_dc_real_v" in sig.parameters:
                    kwargs["v_dc_real_v"] = vdc_np
                elif "vdc_meas_v" in sig.parameters:
                    kwargs["vdc_meas_v"] = vdc_np

                if "i_dc_real_a" in sig.parameters:
                    kwargs["i_dc_real_a"] = idc_np
                elif "idc_meas_a" in sig.parameters:
                    kwargs["idc_meas_a"] = idc_np

                out_model = expected_and_mismatch(**kwargs) or {}
                meta = out_model.get("meta", {}) if isinstance(out_model, dict) else {}

                audit["model_ok"] = True
                audit["model_meta"] = meta

                pac_exp = out_model.get("pac_expected_w")
                if pac_exp is not None:
                    p_ac_model_w = np_to_list_none(pac_exp)

                    dt_h = dt_minutes / 60.0
                    pac_exp_np = _np.asarray(pac_exp, dtype=float)
                    e_model_kwh = float(_np.nansum(_np.clip(pac_exp_np, 0.0, None)) * dt_h / 1000.0)

                if out_model.get("mismatch_rel") is not None:
                    mismatch_rel = np_to_list_none(out_model["mismatch_rel"])
                if out_model.get("tcell_c") is not None:
                    tcell_c = np_to_list_none(out_model["tcell_c"])

                if out_model.get("p_ac_pu_model") is not None:
                    p_ac_pu_model = np_to_list_none(out_model["p_ac_pu_model"])
                if out_model.get("p_ac_pu_real") is not None:
                    p_ac_pu_real = np_to_list_none(out_model["p_ac_pu_real"])
                if out_model.get("pr_model_inst") is not None:
                    pr_model_inst = np_to_list_none(out_model["pr_model_inst"])
                if out_model.get("pr_real_inst") is not None:
                    pr_real_inst = np_to_list_none(out_model["pr_real_inst"])
                if out_model.get("g_std_60m") is not None:
                    g_std_60m = np_to_list_none(out_model["g_std_60m"])
                if out_model.get("g_cv_60m") is not None:
                    g_cv_60m = np_to_list_none(out_model["g_cv_60m"])
                if out_model.get("csi") is not None:
                    csi = np_to_list_none(out_model["csi"])
                    k_cs = csi
                if out_model.get("eta_inv") is not None:
                    eta_inv = np_to_list_none(out_model["eta_inv"])

                pac_model_pu_stc = p_ac_pu_model
                pac_real_pu_stc = p_ac_pu_real

                # repasse (sem fuzzy)
                if out_model.get("rca_label") is not None:
                    rca_label = np_to_list_str(out_model["rca_label"])
                if out_model.get("valid") is not None:
                    valid_model = np_to_list_bool(out_model["valid"])

                # charts (se builder existir)
                if callable(build_dashboard_payload):
                    charts = build_dashboard_payload(
                        times=x_iso,
                        out_model=out_model,
                        dt_minutes=dt_minutes,
                        min_persist_minutes=persist_minutes,
                    )

    except Exception as e:
        audit["model_ok"] = False
        audit["model_error"] = f"{type(e).__name__}: {e}"
        logger.exception("Falha ao calcular modelo físico no pv_dashboard_timeseries_api (plant_id=%s).", plant_id)

        # mantém charts default e séries de modelo como None

    # ----------------------------
    # KPIs + payload
    # ----------------------------
    inv_cov_mean = (sum(inv_cov_vals) / len(inv_cov_vals)) if inv_cov_vals else None
    e_kwh = e_wh_total / 1000.0

    met_missing_frac = round(met_missing_ts / n, 3) if n else 0.0
    inv_missing_frac = round(inv_missing_count / inv_total_points, 3) if inv_total_points else 0.0

    payload: Dict[str, Any] = {
        "ok": True,
        "empty": False,
        "plant": {"id": plant.id, "nome": plant.nome, "tz": tz_name},
        "plant_info": plant_info,
        "range": {
            "start_local": start_s,
            "end_local": end_s,
            "dt0_utc": dt0_utc.isoformat() if hasattr(dt0_utc, "isoformat") else dt0_utc,
            "dt1_utc": dt1_utc.isoformat() if hasattr(dt1_utc, "isoformat") else dt1_utc,
        },
        "sources": {
            "source_oper": src_oper,
            "source_oper_list": src_oper_list,
            "source_meteo": src_meteo,
        },
        "x": x_iso,
        "x_label": x_label,

        "charts": charts,

        "series": {
            "p_ac_w": p_ac,
            "p_dc_w": p_dc,
            "ghi": ghi,
            "gti": gti,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "wind_speed": wind,
            "rh": rh,
            "v_dc_v": v_dc,
            "i_dc_a": i_dc,
            "v_ac_v": v_ac,
            "i_ac_a": i_ac,

            # modelo
            "p_ac_model_w": p_ac_model_w,
            "mismatch_rel": mismatch_rel,
            "tcell_c": tcell_c,
            "p_ac_pu_model": p_ac_pu_model,
            "p_ac_pu_real": p_ac_pu_real,
            "pr_model_inst": pr_model_inst,
            "pr_real_inst": pr_real_inst,
            "g_std_60m": g_std_60m,
            "g_cv_60m": g_cv_60m,
            "csi": csi,
            "eta_inv": eta_inv,
            "pac_model_pu_stc": pac_model_pu_stc,
            "pac_real_pu_stc": pac_real_pu_stc,
            "k_cs": k_cs,
            "g_poa_used": g_poa_used,

            # repasse (sem fuzzy)
            "valid_model": valid_model,
            "rca_label": rca_label,
        },

        "series_by_source": series_by_source,

        "kpis": {
            "energy_kwh": round(e_kwh, 3),
            "energy_model_kwh": None if e_model_kwh is None else round(e_model_kwh, 3),
            "p_ac_max_w": None if p_ac_max is None else round(p_ac_max, 1),
            "ghi_max_wm2": None if ghi_max is None else round(ghi_max, 1),
            "inv_coverage_mean": None if inv_cov_mean is None else round(inv_cov_mean, 3),
            "meteo_missing_frac": met_missing_frac,
            "inv_missing_frac": inv_missing_frac,
            "points": n,
            "sources_oper_qty": len(src_oper_list),
            "meteo_reliability_score": (charts.get("gauge") or {}).get("score") if isinstance(charts, dict) else None,
            "meteo_reliability_label": (charts.get("gauge") or {}).get("label") if isinstance(charts, dict) else None,
        },
        "audit": audit,
        "debug": {
            "len_x": len(x_iso),
            "len_sources": len(src_oper_list),
            "has_model": bool(audit.get("model_ok")) and (p_ac_model_w is not None),
            "model_error": audit.get("model_error"),
        },
    }

    return _json_response_strict(payload)



# ----------------------------
# ----------- S A N I D A D E  D O  S I S T E M A  (Mismatch FDD)
# ----------------------------
import inspect
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Count
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods

from core.models import PVPlant, PVPlantMergedRecord15m, PlantDiagnostic15m
from core.services.fdd_mismatch import (
    MismatchThresholds,
    classify_mismatch_series,
    CODE_INVALID,
)

logger = logging.getLogger(__name__)


# ----------------------------
# JSON strict/robusto
# ----------------------------
def _json_response_strict(payload: Any, *, status: int = 200) -> JsonResponse:
    """JsonResponse com serializer robusto (datetime, date, numpy, etc.)."""

    def _default(o: Any):
        if isinstance(o, (datetime, date)):
            return o.isoformat()

        # numpy (opcional)
        try:
            import numpy as np  # type: ignore

            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass

        # dataclasses
        if is_dataclass(o):
            return asdict(o)

        return str(o)

    safe = isinstance(payload, dict)
    return JsonResponse(
        payload,
        status=status,
        safe=safe,
        json_dumps_params={"ensure_ascii": False, "default": _default},
    )


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _sum_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    ok = False
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        ok = True
    return acc if ok else None


def _mean_none(xs: List[Optional[float]]) -> Optional[float]:
    acc = 0.0
    n = 0
    for v in xs:
        if v is None:
            continue
        acc += float(v)
        n += 1
    return (acc / n) if n else None


def _pick_best_sources(
    plant_id: int,
    dt0_utc: datetime,
    dt1_utc: datetime,
) -> Tuple[Optional[str], Optional[str]]:
    """Escolhe (source_oper, source_meteo) com maior n no range."""
    row = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper", "source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    if not row:
        return None, None
    return row.get("source_oper"), row.get("source_meteo")


def _upsert_diag15m(
    *,
    plant: PVPlant,
    times_utc: List[datetime],
    codes: List[int],
    labels: List[str],
    valid: List[bool],
    g_poa: List[Optional[float]],
    tcell_c: List[Optional[float]],
    pac_real_w: List[Optional[float]],
    pac_model_w: List[Optional[float]],
    mismatch_rel: List[Optional[float]],
) -> Dict[str, Any]:
    """
    Upsert em PlantDiagnostic15m para o range.
    FIX importante:
      - bulk_create/bulk_update NÃO disparam auto_now/auto_now_add.
      - Portanto, setamos updated_at (e created_at se existir) manualmente.
    """
    assert (
        len(times_utc)
        == len(codes)
        == len(labels)
        == len(valid)
        == len(g_poa)
        == len(tcell_c)
        == len(pac_real_w)
        == len(pac_model_w)
        == len(mismatch_rel)
    )

    existing: Dict[datetime, PlantDiagnostic15m] = {}
    chunk = 1000
    for i in range(0, len(times_utc), chunk):
        ts_chunk = times_utc[i : i + chunk]
        qs = PlantDiagnostic15m.objects.filter(plant=plant, ts_utc__in=ts_chunk)
        for obj in qs:
            existing[obj.ts_utc] = obj

    to_create: List[PlantDiagnostic15m] = []
    to_update: List[PlantDiagnostic15m] = []

    now = timezone.now()

    for i, ts in enumerate(times_utc):
        obj = existing.get(ts)
        is_new = obj is None
        if is_new:
            obj = PlantDiagnostic15m(plant=plant, ts_utc=ts)
            to_create.append(obj)
        else:
            to_update.append(obj)

        obj.rca_code = int(codes[i])
        obj.rca_label = str(labels[i] or "invalid")
        obj.valid = bool(valid[i])

        obj.g_poa = g_poa[i]
        obj.tcell_c = tcell_c[i]
        obj.pac_real_w = pac_real_w[i]
        obj.pac_model_w = pac_model_w[i]
        obj.mismatch_rel = mismatch_rel[i]

        # timestamps (bulk_* não chama save())
        if hasattr(obj, "updated_at"):
            setattr(obj, "updated_at", now)
        if is_new and hasattr(obj, "created_at") and getattr(obj, "created_at", None) is None:
            setattr(obj, "created_at", now)

    with transaction.atomic():
        if to_create:
            PlantDiagnostic15m.objects.bulk_create(to_create, ignore_conflicts=True)
        if to_update:
            fields = [
                "rca_code",
                "rca_label",
                "valid",
                "g_poa",
                "tcell_c",
                "pac_real_w",
                "pac_model_w",
                "mismatch_rel",
            ]
            if hasattr(PlantDiagnostic15m, "updated_at"):
                fields.append("updated_at")

            PlantDiagnostic15m.objects.bulk_update(to_update, fields=fields)

    return {"created": len(to_create), "updated": len(to_update)}


# ----------------------------
# View (página)
# ----------------------------
@require_GET
@login_required
def mismatch_fdd_view(request: HttpRequest):
    """Página: Heatmap de mismatch + detalhes ao clicar."""
    qs = (
        PVPlant.objects.all().order_by("nome")
        if request.user.is_superuser
        else PVPlant.objects.filter(owner=request.user).order_by("nome")
    )
    plants = list(qs)

    d_end = date.today()
    d_start = d_end - timedelta(days=7)

    # FIX: adiciona "today" para templates antigos que usam default:today
    return render(
        request,
        "dashboard/mismatch_fdd.html",
        {
            "plants": plants,
            "default_start": d_start.isoformat(),
            "default_end": d_end.isoformat(),
            "today": d_end.isoformat(),
            "api_url": reverse("mismatch_fdd_api"),
        },
    )


# ----------------------------
# API (GET/POST)
# ----------------------------
@require_http_methods(["GET", "POST"])
@login_required
def mismatch_fdd_api(request: HttpRequest) -> JsonResponse:
    data = request.POST if request.method == "POST" else request.GET

    try:
        plant_id = int((data.get("plant_id") or "0").strip())
    except Exception:
        return _json_response_strict({"ok": False, "error": "plant_id inválido"}, status=400)

    plant = (
        PVPlant.objects.filter(id=plant_id)
        .select_related("details", "details__module", "details__inverter")
        .first()
    )
    if not plant:
        return _json_response_strict({"ok": False, "error": "Planta não encontrada"}, status=404)

    if (not request.user.is_superuser) and plant.owner_id and (plant.owner_id != request.user.id):
        return _json_response_strict({"ok": False, "error": "Sem permissão para esta planta"}, status=403)

    tz_name = getattr(plant, "timezone", "UTC") or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"
        tz = ZoneInfo("UTC")

    d0 = _parse_date(data.get("start") or "")
    d1 = _parse_date(data.get("end") or "")
    if not d0 or not d1:
        return _json_response_strict({"ok": False, "error": "start/end (YYYY-MM-DD) são obrigatórios"}, status=400)
    if d1 < d0:
        return _json_response_strict({"ok": False, "error": "end < start"}, status=400)

    # range local -> UTC (fim exclusivo)
    dt0_local = datetime.combine(d0, time.min, tzinfo=tz)
    dt1_local = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    src_oper_raw = (data.get("source_oper") or data.get("src_oper") or "").strip()
    src_meteo = (data.get("source_meteo") or data.get("src_meteo") or "").strip() or None

    if not src_meteo:
        _, _sm = _pick_best_sources(plant_id, dt0_utc, dt1_utc)
        src_meteo = _sm

    if not src_meteo:
        return _json_response_strict({"ok": False, "error": "Sem registros no range (PVPlantMergedRecord15m)."}, status=404)

    src_oper_rows = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=src_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    source_oper_list = [r["source_oper"] for r in src_oper_rows if r.get("source_oper")]
    if not source_oper_list:
        return _json_response_strict(
            {"ok": False, "error": "Sem source_oper para a fonte meteo selecionada no range."},
            status=404,
        )

    want_all = (not src_oper_raw) or (src_oper_raw.upper() == "ALL")
    if want_all:
        selected_sources = list(source_oper_list)
    else:
        if src_oper_raw not in source_oper_list:
            return _json_response_strict(
                {"ok": False, "error": f"source_oper '{src_oper_raw}' não existe no range."},
                status=404,
            )
        selected_sources = [src_oper_raw]

    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=src_meteo,
            source_oper__in=selected_sources,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .order_by("ts_utc")
        .values(
            "ts_utc",
            "source_oper",
            "p_ac_w",
            "p_dc_w",
            "e_ac_wh_15",
            "v_dc_v",
            "i_dc_a",
            "v_ac_v",
            "i_ac_a",
            "inv_coverage",
            "flag_inv_missing",
            "gti",
            "ghi",
            "dni",
            "dhi",
            "temp_air",
            "wind_speed",
            "rh",
            "flag_meteo_missing",
        )
    )
    rows = list(qs)
    if not rows:
        return _json_response_strict({"ok": False, "error": "Sem registros no range para as fontes selecionadas."}, status=404)

    # FIX: normaliza ts_utc (aware UTC, sem segundos/micros) para evitar drift
    def _norm_ts_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt_tz.utc)
        ts = ts.astimezone(dt_tz.utc)
        return ts.replace(second=0, microsecond=0, tzinfo=dt_tz.utc)

    per_ts: Dict[datetime, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        ts = _norm_ts_utc(r["ts_utc"])
        src = r.get("source_oper") or ""
        if not src:
            continue
        per_ts.setdefault(ts, {})[src] = r

    times_utc = sorted(per_ts.keys())
    n = len(times_utc)

    p_ac_w: List[Optional[float]] = [None] * n
    p_dc_w: List[Optional[float]] = [None] * n
    e_ac_wh_15: List[Optional[float]] = [None] * n
    v_dc_v: List[Optional[float]] = [None] * n
    i_dc_a: List[Optional[float]] = [None] * n
    v_ac_v: List[Optional[float]] = [None] * n
    i_ac_a: List[Optional[float]] = [None] * n
    inv_cov: List[Optional[float]] = [None] * n
    flag_inv_missing: List[bool] = [False] * n

    gti: List[Optional[float]] = [None] * n
    ghi: List[Optional[float]] = [None] * n
    dni: List[Optional[float]] = [None] * n
    dhi: List[Optional[float]] = [None] * n
    temp_air: List[Optional[float]] = [None] * n
    wind_speed: List[Optional[float]] = [None] * n
    rh: List[Optional[float]] = [None] * n
    flag_meteo_missing: List[bool] = [False] * n

    series_by_source: Dict[str, Dict[str, List[Optional[float]]]] = {
        src: {
            "p_ac_w": [None] * n,
            "p_dc_w": [None] * n,
            "e_ac_wh_15": [None] * n,
            "v_dc_v": [None] * n,
            "i_dc_a": [None] * n,
            "v_ac_v": [None] * n,
            "i_ac_a": [None] * n,
        }
        for src in selected_sources
    }

    for i, ts in enumerate(times_utc):
        by_src = per_ts.get(ts, {})

        pac_l: List[Optional[float]] = []
        pdc_l: List[Optional[float]] = []
        e15_l: List[Optional[float]] = []
        vdc_l: List[Optional[float]] = []
        idc_l: List[Optional[float]] = []
        vac_l: List[Optional[float]] = []
        iac_l: List[Optional[float]] = []
        cov_l: List[Optional[float]] = []

        inv_missing_any = False
        first_row: Optional[Dict[str, Any]] = None

        for src in selected_sources:
            r = by_src.get(src)
            if r is None:
                continue
            if first_row is None:
                first_row = r

            pac = _as_float(r.get("p_ac_w"))
            pdc = _as_float(r.get("p_dc_w"))
            e15 = _as_float(r.get("e_ac_wh_15"))
            vdc = _as_float(r.get("v_dc_v"))
            idc = _as_float(r.get("i_dc_a"))
            vac = _as_float(r.get("v_ac_v"))
            iac = _as_float(r.get("i_ac_a"))
            cov = _as_float(r.get("inv_coverage"))
            inv_missing_any = inv_missing_any or bool(r.get("flag_inv_missing") or False)

            pac_l.append(pac)
            pdc_l.append(pdc)
            e15_l.append(e15)
            vdc_l.append(vdc)
            idc_l.append(idc)
            vac_l.append(vac)
            iac_l.append(iac)
            cov_l.append(cov)

            sb = series_by_source.get(src)
            if sb is not None:
                sb["p_ac_w"][i] = pac
                sb["p_dc_w"][i] = pdc
                sb["e_ac_wh_15"][i] = e15
                sb["v_dc_v"][i] = vdc
                sb["i_dc_a"][i] = idc
                sb["v_ac_v"][i] = vac
                sb["i_ac_a"][i] = iac

        p_ac_w[i] = _sum_none(pac_l)
        p_dc_w[i] = _sum_none(pdc_l)
        e_ac_wh_15[i] = _sum_none(e15_l)
        v_dc_v[i] = _mean_none(vdc_l)
        i_dc_a[i] = _sum_none(idc_l)
        v_ac_v[i] = _mean_none(vac_l)
        i_ac_a[i] = _sum_none(iac_l)
        inv_cov[i] = _mean_none(cov_l)
        flag_inv_missing[i] = inv_missing_any

        if first_row is not None:
            gti[i] = _as_float(first_row.get("gti"))
            ghi[i] = _as_float(first_row.get("ghi"))
            dni[i] = _as_float(first_row.get("dni"))
            dhi[i] = _as_float(first_row.get("dhi"))
            temp_air[i] = _as_float(first_row.get("temp_air"))
            wind_speed[i] = _as_float(first_row.get("wind_speed"))
            rh[i] = _as_float(first_row.get("rh"))
            flag_meteo_missing[i] = bool(first_row.get("flag_meteo_missing") or False)

    # FIX p/ alinhamento de heatmap no front:
    # - t_local: ISO com offset (bom p/ exibição)
    # - t_local_wall: ISO sem offset (bom p/ binning wall-clock no JS, sem Date timezone conversion)
    x_local_dt = [t.astimezone(tz) for t in times_utc]
    x_local = [d.isoformat() for d in x_local_dt]
    x_local_wall = [d.replace(tzinfo=None).isoformat(timespec="minutes") for d in x_local_dt]
    x_utc = [t.isoformat() for t in times_utc]

    g_poa_used = [gti_i if gti_i is not None else ghi_i for gti_i, ghi_i in zip(gti, ghi)]

    def _gf(key: str, default: float) -> float:
        raw = (data.get(key) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _gi(key: str, default: int) -> int:
        raw = (data.get(key) or "").strip()
        if not raw:
            return int(default)
        try:
            return int(float(raw))
        except Exception:
            return int(default)

    thr = MismatchThresholds(
        gpoa_gate_wm2=_gf("gpoa_gate", 200.0),
        warn_abs=_gf("warn_abs", 0.35),
        fault_abs=_gf("fault_abs", 0.80),
        meteo_pos_abs=_gf("meteo_pos_abs", 0.25),
        shading_std_abs=_gf("shading_std_abs", 0.22),
        shading_window_points=_gi("shading_window_points", 6),
        dt_minutes=15.0,
        max_gap_minutes=_gf("max_gap_minutes", 30.0),
    )

    details = getattr(plant, "details", None)
    if not details or not getattr(details, "module_id", None):
        return _json_response_strict(
            {"ok": False, "error": "PVPlantDetails.module não configurado. Cadastre o módulo em 'Planta > Detalhes'."},
            status=400,
        )

    n_mod = int(getattr(details, "modules_total", 0) or 0)
    if n_mod <= 0:
        return _json_response_strict(
            {"ok": False, "error": "PVPlantDetails.modules_total inválido. Configure strings/módulos totais."},
            status=400,
        )

    # ----------------------------
    # Power model
    # ----------------------------
    try:
        import numpy as np
        from core.services.power_model.power_model import (
            expected_and_mismatch,
            module_from_pvmodule,
            plant_from_details,
        )

        mod = module_from_pvmodule(details.module)
        inv = getattr(details, "inverter", None)
        pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

        # FIX: reconstrução robusta do dataclass/class filtrando kwargs pelo __init__
        pld = asdict(pl) if is_dataclass(pl) else dict(getattr(pl, "__dict__", {}))
        if pld.get("lat_deg") is None:
            pld["lat_deg"] = _as_float(getattr(plant, "latitude", None))
        if pld.get("lon_deg") is None:
            pld["lon_deg"] = _as_float(getattr(plant, "longitude", None))
        if pld.get("tilt_deg") is None:
            pld["tilt_deg"] = _as_float(getattr(details, "tilt_deg", None))
        if pld.get("azimuth_deg") is None:
            pld["azimuth_deg"] = _as_float(getattr(details, "azimuth_deg", None))

        sig_pl = inspect.signature(pl.__class__)
        allowed = set(sig_pl.parameters.keys())
        pld = {k: v for k, v in pld.items() if k in allowed}
        pl = pl.__class__(**pld)

        def list_to_np_nan(xs):
            out = np.empty(len(xs), dtype=np.float64)
            for j, v in enumerate(xs):
                try:
                    out[j] = np.nan if v is None else float(v)
                except Exception:
                    out[j] = np.nan
            return out

        gpoa_np = list_to_np_nan(g_poa_used)
        tamb_np = list_to_np_nan(temp_air)
        pac_real_np = list_to_np_nan(p_ac_w)

        sig = inspect.signature(expected_and_mismatch)
        kwargs: Dict[str, Any] = dict(
            g_poa=gpoa_np,
            tamb_c=tamb_np,
            pac_real_w=pac_real_np,
            module=mod,
            plant=pl,
            g_min_valid=0.0,
            n_points=60,
            eps_w=50.0,
        )
        if "dt_minutes" in sig.parameters:
            kwargs["dt_minutes"] = 15.0
        if "window_minutes" in sig.parameters:
            kwargs["window_minutes"] = 60.0
        if "times_utc" in sig.parameters:
            kwargs["times_utc"] = times_utc

        out_model = expected_and_mismatch(**kwargs) or {}
        pac_expected = out_model.get("pac_expected_w")
        mismatch = out_model.get("mismatch_rel")
        valid_model_np = out_model.get("valid")
        tcell_np = out_model.get("tcell_c")

        if pac_expected is None:
            return _json_response_strict({"ok": False, "error": "power_model não retornou pac_expected_w."}, status=500)

        pac_expected_list = np.asarray(pac_expected, dtype=float).tolist()
        if len(pac_expected_list) != n:
            return _json_response_strict(
                {"ok": False, "error": f"power_model retornou pac_expected_w com len={len(pac_expected_list)} != n={n}"},
                status=500,
            )

        pac_model_w = [None if (not np.isfinite(v)) else float(v) for v in pac_expected_list]

        if mismatch is None:
            eps = 50.0
            mm: List[Optional[float]] = []
            for pr, pm in zip(p_ac_w, pac_model_w):
                if pr is None or pm is None:
                    mm.append(None)
                    continue
                den = max(abs(pm), eps)
                mm.append((float(pr) - float(pm)) / float(den))
            mismatch_rel = mm
        else:
            mismatch_list = np.asarray(mismatch, dtype=float).tolist()
            if len(mismatch_list) != n:
                return _json_response_strict(
                    {"ok": False, "error": f"power_model retornou mismatch_rel com len={len(mismatch_list)} != n={n}"},
                    status=500,
                )
            mismatch_rel = [None if (not np.isfinite(v)) else float(v) for v in mismatch_list]

        if valid_model_np is None:
            valid_model = [False if (m is None) else True for m in mismatch_rel]
        else:
            valid_list = list(np.asarray(valid_model_np, dtype=bool).tolist())
            if len(valid_list) != n:
                return _json_response_strict(
                    {"ok": False, "error": f"power_model retornou valid com len={len(valid_list)} != n={n}"},
                    status=500,
                )
            valid_model = [bool(v) for v in valid_list]

        if tcell_np is None:
            tcell_c = [None] * n
        else:
            tcell_list = np.asarray(tcell_np, dtype=float).tolist()
            if len(tcell_list) != n:
                return _json_response_strict(
                    {"ok": False, "error": f"power_model retornou tcell_c com len={len(tcell_list)} != n={n}"},
                    status=500,
                )
            tcell_c = [None if (not np.isfinite(v)) else float(v) for v in tcell_list]

    except Exception as e:
        logger.exception("Falha no power_model (mismatch_fdd_api) plant_id=%s", plant_id)
        return _json_response_strict({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)

    # ----------------------------
    # Classificação mismatch
    # ----------------------------
    out_cls = classify_mismatch_series(
        times_utc=times_utc,
        mismatch_rel=mismatch_rel,
        g_poa_wm2=g_poa_used,
        valid=valid_model,
        thresholds=thr,
    )
    codes = out_cls["codes"]
    labels = out_cls["labels"]

    persist = (data.get("persist") or "").strip().lower() in ("1", "true", "yes", "on")
    upsert = None
    if persist:
        upsert = _upsert_diag15m(
            plant=plant,
            times_utc=times_utc,
            codes=codes,
            labels=labels,
            valid=valid_model,
            g_poa=g_poa_used,
            tcell_c=tcell_c,
            pac_real_w=p_ac_w,
            pac_model_w=pac_model_w,
            mismatch_rel=mismatch_rel,
        )

    def _mean_abs_valid() -> Optional[float]:
        buf: List[float] = []
        for v_ok, mi, code in zip(valid_model, mismatch_rel, codes):
            if (not v_ok) or int(code) == CODE_INVALID:
                continue
            if mi is None:
                continue
            buf.append(abs(float(mi)))
        return (sum(buf) / len(buf)) if buf else None

    payload = {
        "ok": True,
        "plant": {"id": plant.id, "nome": plant.nome, "tz": tz_name},
        "range": {
            "start": d0.isoformat(),
            "end": d1.isoformat(),
            "start_utc": dt0_utc.isoformat(),
            "end_utc_excl": dt1_utc.isoformat(),
            "source_meteo": src_meteo,
            "selected_sources": selected_sources,
        },
        "sources": {
            "source_meteo": src_meteo,
            "source_oper_list": source_oper_list,
            "selected_sources": selected_sources,
        },
        "x_local": x_local,
        "x_local_wall": x_local_wall,  # ✅ útil para heatmap no front (wall-clock sem offset)
        "x_utc": x_utc,
        "series": {
            # ✅ aliases esperados pelo front
            "t_local": x_local,               # ISO com offset (exibição)
            "t_local_wall": x_local_wall,     # ISO sem offset (binnig estável no JS)
            "t_utc": x_utc,

            "g_poa": g_poa_used,
            "labels": labels,
            "codes": codes,

            # mantém nomes atuais também
            "x_local": x_local,
            "x_utc": x_utc,

            "p_ac_w": p_ac_w,
            "p_ac_real_w": p_ac_w,
            "p_dc_w": p_dc_w,
            "e_ac_wh_15": e_ac_wh_15,
            "v_dc_v": v_dc_v,
            "i_dc_a": i_dc_a,
            "v_ac_v": v_ac_v,
            "i_ac_a": i_ac_a,
            "inv_coverage": inv_cov,
            "flag_inv_missing": flag_inv_missing,

            "gti": gti,
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp_air,
            "wind_speed": wind_speed,
            "rh": rh,
            "flag_meteo_missing": flag_meteo_missing,

            "p_ac_model_w": pac_model_w,
            "tcell_c": tcell_c,
            "mismatch_rel": mismatch_rel,

            "g_poa_used": g_poa_used,
            "valid_model": valid_model,
            "valid": valid_model,

            "rca_code": codes,
            "rca_label": labels,
        },
        "series_by_source": series_by_source,
        "summary": {
            "counts": out_cls.get("summary", {}),
            "events": out_cls.get("events", []),
            "mean_abs_mismatch_valid": _mean_abs_valid(),
            "n_points": n,
        },
        "thresholds": out_cls.get("thresholds", {}),
        "persist": upsert,
    }

    return _json_response_strict(payload)
