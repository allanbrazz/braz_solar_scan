# core/services/renovigi_gateway.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings

# RenovigiClient pode expor RenovigiError (com err/desc). Mantemos import resiliente.
try:
    from .renovigi_client import RenovigiClient, RenovigiSession, RenovigiError  # type: ignore
except Exception:  # pragma: no cover
    from .renovigi_client import RenovigiClient, RenovigiSession  # type: ignore

    class RenovigiError(Exception):  # type: ignore
        pass


class RenovigiGatewayError(RuntimeError):
    pass


# ----------------------------
# Helpers de normalização
# ----------------------------
def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def _first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _is_no_record_error(exc: Exception) -> bool:
    """
    ShineMonitor/Renovigi: err=12 / ERR_NO_RECORD = "sem registro" para o dia consultado.
    Isso NÃO é falha do sistema; é dia vazio e deve ser tratado como rows=[].
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


def _normalize_headers(dat: Dict[str, Any]) -> List[str]:
    titles = _first_nonempty(
        dat.get("title"),
        dat.get("titles"),
        dat.get("header"),
        dat.get("headers"),
    )
    if isinstance(titles, dict):
        titles = _first_nonempty(titles.get("title"), titles.get("titles")) or []

    if not isinstance(titles, list):
        return []

    headers: List[str] = []
    for t in titles:
        if isinstance(t, str):
            s = t.strip()
            if s:
                headers.append(s)
        elif isinstance(t, dict):
            s = _first_nonempty(
                t.get("title"),
                t.get("name"),
                t.get("label"),
                t.get("field"),
            )
            if s is not None:
                s2 = str(s).strip()
                if s2:
                    headers.append(s2)
        else:
            s = str(t).strip()
            if s:
                headers.append(s)

    return headers


def _extract_row_values(row_item: Any) -> List[Any]:
    """
    Row pode vir como:
      - list de valores
      - dict com 'filed' (comum) ou 'field'
      - dict colunar
    """
    if row_item is None:
        return []

    if isinstance(row_item, list):
        return row_item

    if isinstance(row_item, dict):
        vals = _first_nonempty(
            row_item.get("filed"),   # grafia comum do ShineMonitor
            row_item.get("field"),
            row_item.get("values"),
            row_item.get("value"),
        )
        if isinstance(vals, list):
            return vals
        if vals is not None:
            return [vals]
        return list(row_item.values())

    return [row_item]


def _normalize_rows(dat: Dict[str, Any]) -> List[List[Any]]:
    row_container = _first_nonempty(dat.get("row"), dat.get("rows"), dat.get("data"), dat.get("datas"))
    row_list = _as_list(row_container)

    rows: List[List[Any]] = []
    for item in row_list:
        vals = _extract_row_values(item)
        if vals:
            rows.append(vals)
    return rows


# ----------------------------
# Flatten: Plants / Devices
# ----------------------------
def _flatten_plants(obj: Any) -> List[Dict[str, Any]]:
    """
    Retornos variam por OEM/versão:
      - list direto
      - dict com chaves: plants/plant/list/rows/row/data/datas/items/result
      - listas aninhadas
    """
    out: List[Dict[str, Any]] = []

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, list):
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            has_pid = any(k in x for k in ("pid", "plantid", "id"))
            has_name = any(k in x for k in ("name", "nome", "plantname", "title"))
            if has_pid and has_name:
                out.append(x)

            for k in ("plants", "plant", "list", "rows", "row", "data", "datas", "items", "result"):
                if k in x:
                    walk(x.get(k))
            return

    walk(obj)
    return out


def _flatten_devices(obj: Any) -> List[Dict[str, Any]]:
    """
    Retornos variam por OEM/versão:
      - list direto
      - dict com chaves: device/devices/list/rows/row/data/datas/items/result
      - listas aninhadas
    """
    out: List[Dict[str, Any]] = []

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, list):
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            has_any = any(k in x for k in ("pn", "devcode", "devaddr", "sn", "devsn", "serial", "serial_number"))
            if has_any:
                out.append(x)

            for k in ("device", "devices", "list", "rows", "row", "data", "datas", "items", "result"):
                if k in x:
                    walk(x.get(k))
            return

    walk(obj)
    return out


# ----------------------------
# Auth wrapper
# ----------------------------
def _client_and_session(username: str, password: str) -> Tuple[RenovigiClient, RenovigiSession]:
    company_key = getattr(settings, "RENOVIGI_COMPANY_KEY", None)
    if not company_key:
        raise RenovigiGatewayError("RENOVIGI_COMPANY_KEY não definido em settings.")
    cli = RenovigiClient()
    sess = cli.auth(username, password, company_key=company_key)
    return cli, sess


# ----------------------------
# Public API: Plants / Devices
# ----------------------------
def discover_plants(username: str, password: str) -> List[Dict[str, Any]]:
    cli, sess = _client_and_session(username, password)
    dat = cli.query_plants(sess.token, sess.secret, page=0, pagesize=200, i18n="pt_BR", lang="pt_BR")

    raw_plants = _flatten_plants(dat)

    out: List[Dict[str, Any]] = []
    seen = set()

    for p in raw_plants:
        if not isinstance(p, dict):
            continue

        pid = _first_nonempty(p.get("pid"), p.get("plantid"), p.get("id"))
        if pid is None:
            continue

        name = _first_nonempty(p.get("name"), p.get("nome"), p.get("plantname"), p.get("title"))
        status = p.get("status")

        key = str(pid)
        if key in seen:
            continue
        seen.add(key)

        out.append({"pid": pid, "name": name or f"Plant #{pid}", "status": status})

    return out


def discover_devices(username: str, password: str, plantid: int) -> List[Dict[str, Any]]:
    cli, sess = _client_and_session(username, password)
    dat = cli.query_plant_device_view(sess.token, sess.secret, plantid=int(plantid), i18n="pt_BR", lang="pt_BR")

    raw_devices = _flatten_devices(dat)

    norm: List[Dict[str, Any]] = []
    seen = set()

    for d in raw_devices:
        if not isinstance(d, dict):
            continue

        pn = _first_nonempty(d.get("pn"), d.get("PN"))
        devcode = _first_nonempty(d.get("devcode"), d.get("devCode"), d.get("code"))
        devaddr = _first_nonempty(d.get("devaddr"), d.get("devAddr"), d.get("addr"), d.get("address"))
        sn = _first_nonempty(d.get("sn"), d.get("devsn"), d.get("serial_number"), d.get("serial"), d.get("SN"))
        alias = _first_nonempty(d.get("alias"), d.get("name"), d.get("devname"), d.get("title"))

        try:
            devaddr_i = int(devaddr) if devaddr is not None and str(devaddr).strip() != "" else None
        except Exception:
            devaddr_i = None

        if not pn or devcode is None or devaddr_i is None:
            continue

        key = (str(pn), str(devcode), str(devaddr_i), str(sn or ""))
        if key in seen:
            continue
        seen.add(key)

        norm.append(
            {
                "pn": str(pn),
                "devcode": str(devcode),
                "devaddr": int(devaddr_i),
                "sn": str(sn) if sn is not None else "",
                "alias": str(alias) if alias is not None else "",
                "status": d.get("status"),
            }
        )

    return norm


# ----------------------------
# Fetch 1 day (robusto)
# ----------------------------
# CORREÇÃO: incluir "null" (muitos endpoints aceitam) e tratar err=12 como dia vazio.
_ODDEVEN_CANDIDATES = ["odd", "even", "ODD", "EVEN", "0", "1", "", "null"]


def _query_device_data_one_day_paging_safe(
    cli: RenovigiClient,
    sess: RenovigiSession,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: str,
    page: int,
    pagesize: int,
    oddEvenRow: str,
    i18n: str,
    lang: str,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper: se err=12/ERR_NO_RECORD -> retorna None (dia sem dados).
    Qualquer outro erro -> propaga.
    """
    try:
        return cli.query_device_data_one_day_paging(
            token=sess.token,
            secret=sess.secret,
            devaddr=int(devaddr),
            oddEvenRow=str(oddEvenRow),
            pn=str(pn),
            devcode=str(devcode),
            sn=str(sn),
            day_yyyy_mm_dd=str(day),
            page=int(page),
            pagesize=int(pagesize),
            i18n=i18n,
            lang=lang,
        )
    except Exception as exc:
        if _is_no_record_error(exc):
            return None
        raise


def _fetch_one_day_paging(
    cli: RenovigiClient,
    sess: RenovigiSession,
    *,
    pn: str,
    devcode: str,
    devaddr: int,
    sn: str,
    day: str,
    pagesize: int,
    i18n: str,
    lang: str,
) -> Tuple[List[str], List[List[Any]], Dict[str, Any]]:
    debug_attempts: List[Dict[str, Any]] = []

    best_headers: List[str] = []
    best_rows: List[List[Any]] = []

    for oddEvenRow in _ODDEVEN_CANDIDATES:
        for page_base in (0, 1):
            page0 = page_base

            try:
                dat = _query_device_data_one_day_paging_safe(
                    cli,
                    sess,
                    pn=pn,
                    devcode=devcode,
                    devaddr=int(devaddr),
                    sn=str(sn),
                    day=str(day),
                    page=int(page0),
                    pagesize=int(pagesize),
                    oddEvenRow=str(oddEvenRow),
                    i18n=i18n,
                    lang=lang,
                )
            except Exception as exc:
                # erro real: registra e segue tentando outros combos
                debug_attempts.append(
                    {
                        "oddEvenRow": oddEvenRow,
                        "page_base": page_base,
                        "first_page_sent": page0,
                        "error": str(exc),
                    }
                )
                continue

            # ERR_NO_RECORD -> dia vazio, sem necessidade de testar outros combos
            if dat is None:
                debug_day = {
                    "day": day,
                    "picked": None,
                    "attempts": debug_attempts
                    + [
                        {
                            "oddEvenRow": oddEvenRow,
                            "page_base": page_base,
                            "first_page_sent": page0,
                            "note": "ERR_NO_RECORD (dia sem dados)",
                        }
                    ],
                }
                return [], [], debug_day

            headers = _normalize_headers(dat)
            rows = _normalize_rows(dat)

            total = _safe_int(dat.get("total"), 0)
            page_ret = _safe_int(dat.get("page"), page0)
            ps = _safe_int(dat.get("pagesize"), pagesize)

            debug_attempts.append(
                {
                    "oddEvenRow": oddEvenRow,
                    "page_base": page_base,
                    "first_page_sent": page0,
                    "page_returned": page_ret,
                    "total": total,
                    "pagesize_returned": ps,
                    "dat_keys": list(dat.keys())[:30],
                    "title_len": len(headers),
                    "row_type": type(_first_nonempty(dat.get("row"), dat.get("rows"))).__name__,
                    "rows_extracted": len(rows),
                }
            )

            if rows or total > 0:
                all_rows = list(rows)
                if headers and not best_headers:
                    best_headers = headers

                if total > 0 and ps > 0:
                    n_pages = max(1, ceil(total / ps))
                    for pidx in range(1, n_pages):
                        page_to_send = page_base + pidx
                        datp = _query_device_data_one_day_paging_safe(
                            cli,
                            sess,
                            pn=pn,
                            devcode=devcode,
                            devaddr=int(devaddr),
                            sn=str(sn),
                            day=str(day),
                            page=int(page_to_send),
                            pagesize=int(pagesize),
                            oddEvenRow=str(oddEvenRow),
                            i18n=i18n,
                            lang=lang,
                        )
                        if datp is None:
                            break
                        rows_p = _normalize_rows(datp)
                        if rows_p:
                            all_rows.extend(rows_p)

                if len(all_rows) > len(best_rows):
                    best_rows = all_rows
                    if headers:
                        best_headers = headers

                if best_rows:
                    debug_day = {
                        "day": day,
                        "picked": {"oddEvenRow": oddEvenRow, "page_base": page_base},
                        "attempts": debug_attempts,
                    }
                    return best_headers, best_rows, debug_day

    debug_day = {"day": day, "picked": None, "attempts": debug_attempts}
    return best_headers, best_rows, debug_day


# ----------------------------
# Public API: fetch range table
# ----------------------------
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
    cli, sess = _client_and_session(username, password)

    start_dt = date.fromisoformat(start_day)
    end_dt = date.fromisoformat(end_day)
    if end_dt < start_dt:
        raise RenovigiGatewayError("end_day deve ser >= start_day.")

    days_total = (end_dt - start_dt).days + 1
    if days_total > 400:
        raise RenovigiGatewayError(f"Range muito grande ({days_total} dias). Limite de segurança: 400.")

    headers_final: List[str] = []
    rows_final: List[List[Any]] = []

    days_empty = 0
    days_with_data = 0
    first_day_with_data: Optional[str] = None
    last_day_with_data: Optional[str] = None

    sample_days_debug: List[Dict[str, Any]] = []
    meta_days_preview: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    cur = start_dt
    while cur <= end_dt:
        day_s = cur.isoformat()

        try:
            headers, rows, dbg_day = _fetch_one_day_paging(
                cli,
                sess,
                pn=pn,
                devcode=devcode,
                devaddr=int(devaddr),
                sn=str(sn),
                day=day_s,
                pagesize=int(pagesize),
                i18n=i18n,
                lang=lang,
            )
        except Exception as exc:
            # erro real em um dia específico: não aborta o range; registra e segue
            headers, rows, dbg_day = [], [], {"day": day_s, "picked": None, "attempts": [], "error": str(exc)}
            errors.append({"day": day_s, "error": str(exc)})

        if headers and not headers_final:
            headers_final = headers

        if rows:
            rows_final.extend(rows)
            days_with_data += 1
            if first_day_with_data is None:
                first_day_with_data = day_s
            last_day_with_data = day_s
        else:
            days_empty += 1

        if len(meta_days_preview) < 10:
            meta_days_preview.append(
                {
                    "day": day_s,
                    "rows": len(rows),
                    "headers": len(headers),
                    "picked": dbg_day.get("picked"),
                    "error": dbg_day.get("error"),
                }
            )

        if len(sample_days_debug) < 3:
            sample_days_debug.append(dbg_day)

        cur += timedelta(days=1)

    meta = {
        "days_total": days_total,
        "days_empty": days_empty,
        "days_with_data": days_with_data,
        "first_day_with_data": first_day_with_data,
        "last_day_with_data": last_day_with_data,
    }

    debug = {
        "device": {"pn": pn, "devcode": devcode, "devaddr": int(devaddr), "sn": sn},
        "range": {"start_day": start_day, "end_day": end_day},
        "sample_days": sample_days_debug,
        "meta_days_preview": meta_days_preview,
        "errors": errors[:50],
        "note": "ERR_NO_RECORD (err=12) tratado como dia vazio; range não aborta por dia sem dados.",
    }

    return {"headers": headers_final, "rows": rows_final, "meta": meta, "debug": debug}


# ----------------------------
# Compatibilidade: classe opcional
# ----------------------------
@dataclass
class RenovigiGateway:
    """
    Compatibilidade com imports antigos:
      from core.services.renovigi_gateway import RenovigiGateway
    """

    @staticmethod
    def discover_plants(username: str, password: str) -> List[Dict[str, Any]]:
        return discover_plants(username, password)

    @staticmethod
    def discover_devices(username: str, password: str, plantid: int) -> List[Dict[str, Any]]:
        return discover_devices(username, password, plantid)

    @staticmethod
    def fetch_range_table(**kwargs) -> Dict[str, Any]:
        return fetch_range_table(**kwargs)
