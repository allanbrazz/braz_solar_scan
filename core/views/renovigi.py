# core/views/renovigi.py
from __future__ import annotations

from core.views._imports import *  # noqa: F401,F403

from core.services.dados_inversor.renovigi_ingest import sync_operational_data_for_device
from core.services.dados_inversor.renovigi_gateway import (
    discover_plants,
    discover_devices,
    fetch_range_table,
)

from datetime import datetime, timedelta, UTC
from django.core.paginator import Paginator
from django.utils.dateparse import parse_date

# Models
from core.models import (
    PVPlant,
    PlantMonitoringCredential,
    InverterOperationalData,
)

# ---------------------------
# RENOVIGI
# ---------------------------


class RenovigiConsoleView(LoginRequiredMixin, View):
    template_name = "inverters/renovigi_console.html"

    def get_cred(self, plant: PVPlant) -> PlantMonitoringCredential | None:
        return PlantMonitoringCredential.objects.filter(
            plant=plant,
            provedor="RENOVIGI",
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
                )

                if not isinstance(stats, dict):
                    raise RuntimeError(f"Retorno inesperado do sync (esperava dict): {type(stats)}")

                inserted = int(stats.get("inserted") or 0)
                requested_rows = int(stats.get("requested_rows") or 0)
                bad_ts = int(stats.get("bad_ts") or 0)

                # Pequeno resumo “por dia”
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
                    username,
                    password,
                    pn=pn,
                    devcode=devcode,
                    devaddr=int(devaddr),
                    sn=sn,
                    start_day=start_day,
                    end_day=end_day,
                    i18n=getattr(cred, "shinemonitor_i18n", None) or "pt_BR",
                    lang=getattr(cred, "shinemonitor_lang", None) or "pt_BR",
                    pagesize=50,
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
        today_utc = datetime.now(UTC).date()
        default_start = today_utc - timedelta(days=7)
        default_end = today_utc

        start_s = (request.GET.get("start") or str(default_start)).strip()
        end_s = (request.GET.get("end") or str(default_end)).strip()

        start_d = parse_date(start_s) or default_start
        end_d = parse_date(end_s) or default_end

        # Range UTC [start 00:00, end+1 00:00)
        start_dt = datetime(start_d.year, start_d.month, start_d.day, tzinfo=UTC)
        end_dt = datetime(end_d.year, end_d.month, end_d.day, tzinfo=UTC) + timedelta(days=1)

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
