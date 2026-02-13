from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.db import transaction

from core.models import PVPlant, PlantMonitoringCredential, InverterOperationalData
from core.services.dados_inversor.renovigi_client import RenovigiClient, RenovigiError


# -----------------------------
# Helpers (normalização)
# -----------------------------
def _as_list_of_dict(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("datas", "rows", "list", "data", "plants", "devices"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # fallback: primeira list[dict] em qualquer campo (1 nível)
        for v in obj.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, list) and (not vv or isinstance(vv[0], dict)):
                        return [x for x in vv if isinstance(x, dict)]
    return []


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _parse_ts_utc(row: Dict[str, Any]) -> Optional[datetime]:
    """
    Ajuste fino quando você souber a chave real do timestamp.
    Tentativas comuns:
      - epoch ms/s
      - strings ISO / 'YYYY-MM-DD HH:MM:SS'
    """
    # epoch ms/s
    for k in ("ts", "timestamp", "time", "collectTime", "collect_time", "dataTime", "dateTime", "datetime"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            if v > 1_000_000_000_000:  # ms
                return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            if v > 1_000_000_000:  # s
                return datetime.fromtimestamp(v, tz=timezone.utc)

    # strings
    for k in ("timeStr", "collectTimeStr", "dataTimeStr", "dateTimeStr", "datetimeStr", "dataTime", "dateTime"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            s = v.strip()
            # ISO
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
            # formatos comuns
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                except Exception:
                    continue

    return None


# -----------------------------
# Core: fetch 1 dia (paginado) com odd/even auto
# -----------------------------
def _fetch_one_day_rows(
    client: RenovigiClient,
    *,
    token: str,
    secret: str,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: str,  # YYYY-MM-DD
    i18n: str,
    lang: str,
    pagesize: int,
    odd_even: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Usa queryDeviceDataOneDayPaging e pagina até esgotar.
    - odd_even="auto" tenta ["odd","even"] e agrega.
    - Não aborta em dia vazio (ERR_NO_RECORD) -> retorna [].
    """
    if odd_even and odd_even.lower() != "auto":
        candidates = [odd_even]
    else:
        candidates = ["odd", "even"]

    all_rows: List[Dict[str, Any]] = []

    for oer in candidates:
        # tenta start page 0 e 1 (alguns portais variam)
        for start_page in (0, 1):
            page = start_page
            while True:
                try:
                    payload = client.query_device_data_one_day_paging(
                        token=token,
                        secret=secret,
                        devaddr=devaddr,
                        oddEvenRow=oer,
                        pn=pn,
                        devcode=devcode,
                        sn=sn,
                        day_yyyy_mm_dd=day,
                        page=page,
                        pagesize=pagesize,
                        i18n=i18n,
                        lang=lang,
                        use_retry=True,
                    )
                except RenovigiError as e:
                    # ERR_NO_RECORD deve ser tratado como vazio
                    if "ERR_NO_RECORD" in str(e) or "err=12" in str(e):
                        break
                    raise

                rows = _as_list_of_dict(payload)  # o client normaliza para {"datas":[...]} -> ok
                if not rows:
                    break

                all_rows.extend(rows)
                page += 1

            # Se já achou dados com esse start_page, não precisa testar o outro start_page
            if all_rows:
                break

    return all_rows


# -----------------------------
# Public API para as suas views
# -----------------------------
def discover_plants(username: str, password: str) -> List[Dict[str, Any]]:
    client = RenovigiClient()
    sess = client.auth(username, password, getattr(settings, "RENOVIGI_COMPANY_KEY", ""))

    dat = client.query_plants(sess.token, sess.secret, page=0, pagesize=200)
    return _as_list_of_dict(dat)


def discover_devices(username: str, password: str, plantid: int) -> List[Dict[str, Any]]:
    client = RenovigiClient()
    sess = client.auth(username, password, getattr(settings, "RENOVIGI_COMPANY_KEY", ""))

    dat = client.query_plant_device_view(sess.token, sess.secret, plantid=plantid)
    raw = _as_list_of_dict(dat)

    # Normaliza para o formato que sua UI espera
    devices: List[Dict[str, Any]] = []
    for d in raw:
        pn = d.get("pn") or d.get("PN") or d.get("devicePn")
        devcode = d.get("devcode") or d.get("devCode") or d.get("code")
        devaddr = d.get("devaddr") or d.get("devAddr") or d.get("addr")
        sn = d.get("sn") or d.get("SN") or d.get("serialNumber")

        if pn and devcode and devaddr is not None and sn:
            try:
                devaddr_i = int(devaddr)
            except Exception:
                continue
            devices.append(
                {
                    "pn": str(pn).strip(),
                    "devcode": str(devcode).strip(),
                    "devaddr": devaddr_i,
                    "sn": str(sn).strip(),
                }
            )

    return devices


def fetch_range_table(
    username: str,
    password: str,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    start_day: str,
    end_day: str,
    i18n: str = "pt_BR",
    lang: str = "pt_BR",
    pagesize: int = 50,
) -> Dict[str, Any]:
    """
    Retorna SEMPRE dict:
      {"rows":[...], "meta":{...}}
    """
    start_dt = date.fromisoformat(start_day)
    end_dt = date.fromisoformat(end_day)
    if end_dt < start_dt:
        raise ValueError("end_day deve ser >= start_day.")

    client = RenovigiClient()
    sess = client.auth(username, password, getattr(settings, "RENOVIGI_COMPANY_KEY", ""))

    rows_all: List[Dict[str, Any]] = []
    days_total = 0
    days_empty = 0
    days_with_data = 0
    first_day_with_data: Optional[str] = None
    last_day_with_data: Optional[str] = None

    # ✅ NÃO ABORTA no primeiro dia vazio
    for d in _daterange(start_dt, end_dt):
        days_total += 1
        day_s = d.isoformat()

        day_rows = _fetch_one_day_rows(
            client,
            token=sess.token,
            secret=sess.secret,
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
            day=day_s,
            i18n=i18n,
            lang=lang,
            pagesize=pagesize,
            odd_even="auto",
        )

        if not day_rows:
            days_empty += 1
            continue

        days_with_data += 1
        if first_day_with_data is None:
            first_day_with_data = day_s
        last_day_with_data = day_s
        rows_all.extend(day_rows)

    return {
        "rows": rows_all,
        "meta": {
            "days_total": days_total,
            "days_empty": days_empty,
            "days_with_data": days_with_data,
            "first_day_with_data": first_day_with_data,
            "last_day_with_data": last_day_with_data,
        },
    }


def sync_operational_data_for_device(
    *,
    plant: PVPlant,
    cred: PlantMonitoringCredential,
    username: str,
    password: str,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    start_day: date,
    end_day: date,
) -> Dict[str, Any]:
    client = RenovigiClient()
    sess = client.auth(username, password, getattr(settings, "RENOVIGI_COMPANY_KEY", ""))

    pagesize = int(getattr(settings, "RENOVIGI_PAGE_SIZE", 200))
    i18n = getattr(cred, "shinemonitor_i18n", None) or "pt_BR"
    lang = getattr(cred, "shinemonitor_lang", None) or "pt_BR"

    inserted = 0
    requested_rows = 0
    bad_ts = 0
    days_total = 0
    days_empty = 0

    # Evita transação gigantesca: atomiza por dia
    for d in _daterange(start_day, end_day):
        days_total += 1
        day_rows = _fetch_one_day_rows(
            client,
            token=sess.token,
            secret=sess.secret,
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
            day=d.isoformat(),
            i18n=i18n,
            lang=lang,
            pagesize=pagesize,
            odd_even="auto",
        )

        if not day_rows:
            days_empty += 1
            continue

        with transaction.atomic():
            for row in day_rows:
                requested_rows += 1
                ts = _parse_ts_utc(row)
                if ts is None:
                    bad_ts += 1
                    continue

                obj, created = InverterOperationalData.objects.get_or_create(
                    plant=plant,
                    ts_utc=ts,
                    pn=pn,
                    devcode=devcode,
                    devaddr=devaddr,
                    sn=sn,
                    defaults={"payload": row},
                )
                if created:
                    inserted += 1

    return {
        "inserted": inserted,
        "requested_rows": requested_rows,
        "bad_ts": bad_ts,
        "days_total": days_total,
        "days_empty": days_empty,
    }
