# core/services/renovigi_ingest.py
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

from django.apps import apps
from django.conf import settings
from django.db import transaction
from django.db.models import Max

# RenovigiClient pode lançar RenovigiError (ou algo equivalente).
# Mantemos import resiliente para não quebrar caso a exceção não exista no seu projeto.
try:
    from core.services.dados_inversor.renovigi_client import RenovigiClient, RenovigiError  # type: ignore
except Exception:  # pragma: no cover
    from core.services.dados_inversor.renovigi_client import RenovigiClient  # type: ignore

    class RenovigiError(Exception):  # type: ignore
        pass


# Você pode configurar no settings.py:
# RENOVIGI_OPDATA_MODEL = "core.InverterOperationalData"
DEFAULT_MODEL_LABEL = "core.InverterOperationalData"


def _get_opdata_model():
    label = getattr(settings, "RENOVIGI_OPDATA_MODEL", DEFAULT_MODEL_LABEL)
    try:
        app_label, model_name = label.split(".", 1)
    except ValueError as e:
        raise RuntimeError(f"RENOVIGI_OPDATA_MODEL inválido: {label}. Use 'app.ModelName'.") from e
    try:
        return apps.get_model(app_label, model_name)
    except LookupError as e:
        raise RuntimeError(
            f"Model {label} não encontrado. Crie esse model (ou ajuste RENOVIGI_OPDATA_MODEL)."
        ) from e


def _is_no_record_error(exc: Exception) -> bool:
    """
    ShineMonitor/Renovigi: err=12 / ERR_NO_RECORD significa "não há registro para o dia consultado".
    Isso NÃO deve abortar o range; deve ser tratado como "dia vazio".
    """
    err = getattr(exc, "err", None)
    if err is not None:
        try:
            if int(err) == 12:
                return True
        except Exception:
            pass

    msg = str(exc).upper()
    return ("ERR_NO_RECORD" in msg) or ("API ERR=12" in msg) or ("ERR=12" in msg)


def _parse_ts(value: Any, plant_tz: str) -> Optional[datetime]:
    """
    Converte o campo de timestamp em datetime timezone-aware (UTC).

    Regra chave:
      - Qualquer timestamp "naive" vindo como string (ex.: '2025-12-29 08:17:29')
        é interpretado como HORA LOCAL da planta e convertido para UTC.
      - Epoch (s/ms) é absoluto -> UTC.
      - Strings com TZ -> convertidas para UTC.
    """
    if value is None:
        return None

    tz_local = ZoneInfo(plant_tz or "UTC")

    # numérico (epoch)
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:  # ms
            return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
        if v > 1e9:  # s
            return datetime.fromtimestamp(v, tz=timezone.utc)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        # string numérica (epoch)
        if re.fullmatch(r"\d{10,13}", s):
            v = float(s)
            if v > 1e12:
                return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(v, tz=timezone.utc)

        # ISO-ish (aceita "YYYY-MM-DD HH:MM:SS" também)
        try:
            # normaliza espaço -> 'T' para fromisoformat aceitar melhor
            dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T"))
            if dt.tzinfo is None:
                # NAIVE => hora local da planta
                dt = dt.replace(tzinfo=tz_local)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        # BR: DD/MM/YYYY HH:MM(:SS)
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})(?::(\d{2}))?$", s)
        if m:
            dd, mm, yyyy, hh, mi, ss = m.groups()
            ss = ss or "0"
            dt_local = datetime(
                int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=tz_local
            )
            return dt_local.astimezone(timezone.utc)

    return None


def _normalize_titles(dat: Dict[str, Any], width_hint: int) -> List[str]:
    titles = dat.get("title") or dat.get("titles") or []
    if isinstance(titles, list) and titles:
        out: List[str] = []
        for i, t in enumerate(titles):
            if isinstance(t, dict) and t.get("title"):
                out.append(str(t["title"]))
            elif isinstance(t, str) and t.strip():
                out.append(t.strip())
            else:
                out.append(f"field_{i}")
        return out
    return [f"field_{i}" for i in range(width_hint)]


def _normalize_rows(dat: Dict[str, Any]) -> List[List[Any]]:
    rows_raw = dat.get("row") or dat.get("rows") or []
    out: List[List[Any]] = []
    for item in rows_raw:
        if isinstance(item, dict) and "filed" in item:
            out.append(item.get("filed") or [])
        elif isinstance(item, list):
            out.append(item)
        elif isinstance(item, dict) and "field" in item and isinstance(item["field"], list):
            out.append(item["field"])
    return out


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _day_bounds_utc(d: date, plant_tz: str) -> tuple[datetime, datetime]:
    """
    Limites do DIA LOCAL 'd' convertidos para UTC [start, end).
    Isso é essencial para 'skip_days_if_exists' e para deduplicação por dia.
    """
    tz = ZoneInfo(plant_tz or "UTC")
    start_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _query_one_day_page(
    client: RenovigiClient,
    token: str,
    secret: str,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: date,
    page: int,
    pagesize: int,
    oddEvenRow: str,
    i18n: str,
    lang: str,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper com tratamento de ERR_NO_RECORD (err=12).
    Retorna dict do payload ou None se o dia não tem registro.
    """
    try:
        return client.query_device_data_one_day_paging(
            token=token,
            secret=secret,
            devaddr=devaddr,
            oddEvenRow=oddEvenRow,
            pn=pn,
            devcode=devcode,
            sn=sn,
            day_yyyy_mm_dd=day.isoformat(),
            page=page,
            pagesize=pagesize,
            i18n=i18n,
            lang=lang,
        )
    except Exception as exc:
        if _is_no_record_error(exc):
            return None
        raise


def _detect_oddevenrow_for_day(
    client: RenovigiClient,
    token: str,
    secret: str,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: date,
    pagesize: int,
    i18n: str,
    lang: str,
) -> Tuple[Optional[str], List[str], List[List[Any]]]:
    """
    Tenta múltiplos valores de oddEvenRow e retorna o primeiro que produz rows.
    Retorna: (chosen_oddEvenRow, headers, first_rows_page0)

    Observação:
      - Se o dia não tiver registro (err=12), retorna (None, [], []) sem levantar exceção.
    """
    candidates = getattr(
        settings,
        "RENOVIGI_ODDEVENROW_CANDIDATES",
        ["odd", "even", "ODD", "EVEN", "0", "1", "", "null"],
    )

    for odd_even in candidates:
        dat = _query_one_day_page(
            client,
            token,
            secret,
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
            day=day,
            page=0,
            pagesize=pagesize,
            oddEvenRow=str(odd_even),
            i18n=i18n,
            lang=lang,
        )

        # ERR_NO_RECORD -> dia vazio
        if dat is None:
            return None, [], []

        rows0 = _normalize_rows(dat)
        if rows0:
            width = max((len(r) for r in rows0), default=0)
            headers = _normalize_titles(dat, width)
            if headers:
                w = len(headers)
                rows0 = [
                    list(r[:w]) + [None] * (w - len(r)) if len(r) < w else list(r[:w])
                    for r in rows0
                ]
            return str(odd_even), headers, rows0

    return None, [], []


def _fetch_one_day_rows(
    client: RenovigiClient,
    token: str,
    secret: str,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: date,
    pagesize: int,
    max_pages: int,
    i18n: str,
    lang: str,
) -> Tuple[List[str], List[List[Any]]]:
    """
    Retorna (headers, rows) para o dia.
    Estratégia:
      1) detecta oddEvenRow (page=0) que produz dados
      2) pagina usando esse oddEvenRow

    Correção:
      - ERR_NO_RECORD (err=12) => retorna ([], []) para o dia e segue range.
    """
    chosen, headers, rows0 = _detect_oddevenrow_for_day(
        client,
        token,
        secret,
        pn=pn,
        devcode=devcode,
        devaddr=devaddr,
        sn=sn,
        day=day,
        pagesize=pagesize,
        i18n=i18n,
        lang=lang,
    )

    if not chosen:
        return [], []

    all_rows: List[List[Any]] = []
    all_rows.extend(rows0)

    for page in range(1, max_pages):
        dat = _query_one_day_page(
            client,
            token,
            secret,
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
            day=day,
            page=page,
            pagesize=pagesize,
            oddEvenRow=chosen,
            i18n=i18n,
            lang=lang,
        )

        # Se por algum motivo retornar "no record" em páginas posteriores, trata como fim.
        if dat is None:
            break

        rows = _normalize_rows(dat)
        if not rows:
            break

        if headers:
            w = len(headers)
            rows = [
                list(r[:w]) + [None] * (w - len(r)) if len(r) < w else list(r[:w])
                for r in rows
            ]

        all_rows.extend(rows)

        if len(rows) < pagesize:
            break

    return headers, all_rows


def sync_operational_data_for_device(
    *,
    plant,
    cred,
    username: str,
    password: str,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    start_day: date,
    end_day: date,
    pagesize: int = 50,
    max_pages: int = 500,
    safety_days: int = 1,
    skip_days_if_exists: bool = True,
    incremental_from_last: bool = True,
) -> Dict[str, Any]:
    """
    Sincroniza dados operativos no banco.

    Correções principais:
      1) Backfill:
         - Só aplica "incremental from last(ts_utc)" se last_date <= end_day.
           Se o banco já tem dados mais novos (last_date > end_day),
           preserva start_day e permite sincronizar históricos antigos.
      2) ERR_NO_RECORD (err=12):
         - Dia sem dados não aborta o range; apenas resulta em 0 rows naquele dia.
      3) inserted real (determinístico):
         - Filtra timestamps já existentes no dia e conta inserts efetivos como len(objs).
      4) TIMEZONE CANÔNICO:
         - Timestamps "naive" vindos da API são tratados como HORA LOCAL DA PLANTA e convertidos para UTC
           antes de salvar em ts_utc.
         - Os "day bounds" para skip/dedup por dia são calculados como limites do DIA LOCAL convertidos para UTC.
    """
    OpData = _get_opdata_model()

    plant_tz = getattr(plant, "timezone", None) or "UTC"

    last = (
        OpData.objects.filter(
            plant=plant,
            provedor="RENOVIGI",
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
        )
        .aggregate(mx=Max("ts_utc"))
        .get("mx")
    )

    effective_start = start_day
    effective_reason = "user_range"

    if incremental_from_last and isinstance(last, datetime):
        last_date = last.astimezone(timezone.utc).date()
        # CORREÇÃO DO BACKFILL:
        # Só “puxa” effective_start se o último registro está dentro do range solicitado.
        if last_date <= end_day:
            effective_start = max(start_day, last_date - timedelta(days=safety_days))
            effective_reason = f"incremental_from_last({last_date.isoformat()})"
        else:
            effective_reason = f"backfill_preserved_start(last_date={last_date.isoformat()})"

    cli = RenovigiClient()
    session = cli.auth(username, password, getattr(settings, "RENOVIGI_COMPANY_KEY", ""))

    requested_rows = 0
    bad_ts = 0
    attempted_inserts = 0
    inserted = 0
    per_day: List[Dict[str, Any]] = []

    i18n = getattr(cred, "shinemonitor_i18n", None) or "pt_BR"
    lang = getattr(cred, "shinemonitor_lang", None) or "pt_BR"

    for d in _iter_days(effective_start, end_day):
        # LIMITES DO DIA LOCAL (d) -> UTC (para skip/dedup corretos)
        day_start, day_end = _day_bounds_utc(d, plant_tz)

        if skip_days_if_exists:
            exists = (
                OpData.objects.filter(
                    plant=plant,
                    provedor="RENOVIGI",
                    pn=pn,
                    devcode=devcode,
                    devaddr=devaddr,
                    sn=sn,
                    ts_utc__gte=day_start,
                    ts_utc__lt=day_end,
                )
                .only("id")
                .first()
                is not None
            )
            if exists:
                per_day.append(
                    {
                        "day": d.isoformat(),
                        "skipped": True,
                        "requested": 0,
                        "attempted": 0,
                        "inserted": 0,
                    }
                )
                continue

        headers, rows = _fetch_one_day_rows(
            cli,
            session.token,
            session.secret,
            pn=pn,
            devcode=devcode,
            devaddr=devaddr,
            sn=sn,
            day=d,
            pagesize=pagesize,
            max_pages=max_pages,
            i18n=i18n,
            lang=lang,
        )
        requested_rows += len(rows)

        # detecta coluna de timestamp
        ts_idx = 0
        if headers:
            for i, h in enumerate(headers):
                hh = (h or "").lower()
                if ("time" in hh) or ("hora" in hh) or ("data" in hh) or ("timestamp" in hh):
                    ts_idx = i
                    break

        # timestamps já existentes no dia (para inserted real)
        existing_ts = set(
            OpData.objects.filter(
                plant=plant,
                provedor="RENOVIGI",
                pn=pn,
                devcode=devcode,
                devaddr=devaddr,
                sn=sn,
                ts_utc__gte=day_start,
                ts_utc__lt=day_end,
            ).values_list("ts_utc", flat=True)
        )

        objs = []
        for r in rows:
            raw_ts = r[ts_idx] if ts_idx < len(r) else None
            ts = _parse_ts(raw_ts, plant_tz)
            if ts is None:
                bad_ts += 1
                continue

            # garante que caiu no dia local correto (janela UTC do dia local)
            # se cair fora, não inserir nesse dia (evita poluição por parsing ruim)
            if not (day_start <= ts < day_end):
                # ainda assim é útil contar como bad_ts para diagnóstico
                bad_ts += 1
                continue

            if ts in existing_ts:
                continue
            existing_ts.add(ts)

            if headers and len(headers) == len(r):
                payload = {headers[i]: r[i] for i in range(len(headers))}
            else:
                payload = {"row": r, "headers": headers}

            objs.append(
                OpData(
                    plant=plant,
                    provedor="RENOVIGI",
                    pn=pn,
                    devcode=devcode,
                    devaddr=devaddr,
                    sn=sn,
                    ts_utc=ts,
                    payload=payload,
                )
            )

        attempted_inserts += len(objs)
        inserted += len(objs)

        if objs:
            with transaction.atomic():
                OpData.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)

        per_day.append(
            {
                "day": d.isoformat(),
                "skipped": False,
                "requested": len(rows),
                "attempted": len(objs),
                "inserted": len(objs),
            }
        )

    return {
        "device": {"pn": pn, "devcode": devcode, "devaddr": devaddr, "sn": sn},
        "range": {
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "effective_start": effective_start.isoformat(),
            "effective_reason": effective_reason,
        },
        "plant_tz": plant_tz,
        "inserted": inserted,
        "attempted": attempted_inserts,
        "requested_rows": requested_rows,
        "bad_ts": bad_ts,
        "per_day": per_day,
        "note": (
            "Backfill suportado: effective_start preserva start_day quando last_date > end_day. "
            "ERR_NO_RECORD (err=12) tratado como dia vazio. "
            "Timezone corrigido: timestamps naive tratados como hora local da planta e convertidos para UTC; "
            "limites de dia (skip/dedup) calculados como dia local -> UTC."
        ),
    }
