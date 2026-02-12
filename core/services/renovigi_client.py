# core/services/renovigi_client.py
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import quote_plus

import requests
from django.conf import settings


class RenovigiError(RuntimeError):
    pass


def _sha1hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest().lower()


def _unwrap_ok(payload: Dict[str, Any]) -> Any:
    """
    Retorna o conteúdo útil da resposta.

    Importante:
    - Alguns OEMs/versões podem não usar 'dat'/'data' como wrapper.
      Nesse caso, fazemos fallback para retornar o payload inteiro,
      para o gateway conseguir fazer deep-search (title/datas em qualquer nível).
    """
    if not isinstance(payload, dict):
        raise RenovigiError(f"Resposta inválida (não dict): {type(payload)}")

    err = payload.get("err")
    if err not in (0, "0", None):
        raise RenovigiError(f"API err={err} desc={payload.get('desc')}")

    if "dat" in payload:
        return payload.get("dat")
    if "data" in payload:
        return payload.get("data")

    # fallback: NÃO devolva {} (isso mata o parser)
    return payload


@dataclass(frozen=True)
class RenovigiSession:
    token: str
    secret: str


class RenovigiClient:
    """
    Cliente para o endpoint público do ShineMonitor (Renovigi OEM).
    """

    def __init__(self):
        self.base_url = settings.RENOVIGI_BASE_URL  # ex: "https://web.shinemonitor.com/public/"
        self.timeout = getattr(settings, "RENOVIGI_HTTP_TIMEOUT", 30.0)

        self.sess = requests.Session()
        self.sess.headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "User-Agent": "Mozilla/5.0",
        })

    # ---------------------------
    # Auth
    # ---------------------------
    def auth(self, usr: str, pwd: str, company_key: str) -> RenovigiSession:
        salt = str(int(time.time() * 1000))
        action = f"&action=auth&usr={quote_plus(usr)}&company-key={quote_plus(company_key)}"
        sign = _sha1hex(salt + _sha1hex(pwd) + action)

        url = f"{self.base_url}?sign={sign}&salt={salt}{action}"
        r = self.sess.get(url, timeout=self.timeout)
        r.raise_for_status()
        dat = _unwrap_ok(r.json())

        if not isinstance(dat, dict):
            raise RenovigiError("Auth OK, mas 'dat' não é dict.")

        token = dat.get("token")
        secret = dat.get("secret")
        if not token or not secret:
            raise RenovigiError("Auth OK, mas não retornou token/secret.")
        return RenovigiSession(token=str(token), secret=str(secret))

    # ---------------------------
    # Signed calls
    # ---------------------------
    def _build_signed_url(self, token: str, secret: str, action_str: str) -> str:
        salt = str(int(time.time() * 1000))
        sign = _sha1hex(salt + secret + token + action_str)
        return f"{self.base_url}?sign={sign}&salt={salt}&token={token}{action_str}"

    def _call_action(self, token: str, secret: str, action: str, params: List[Tuple[str, Any]]) -> Any:
        """
        Monta action_str preservando ordem dos parâmetros (importante para sign).
        Retorna conteúdo útil via _unwrap_ok (pode ser dict, list, etc).
        """
        action_str = f"&action={action}"
        for k, v in params:
            if v is None:
                continue
            action_str += f"&{k}={quote_plus(str(v))}"

        url = self._build_signed_url(token, secret, action_str)
        r = self.sess.get(url, timeout=self.timeout)
        r.raise_for_status()
        return _unwrap_ok(r.json())

    # ---------------------------
    # Data endpoint
    # ---------------------------
    def query_device_data_one_day_paging(
        self,
        token: str,
        secret: str,
        devaddr: int,
        oddEvenRow: str,
        pn: str,
        devcode: str,
        sn: str,
        day_yyyy_mm_dd: str,
        page: int = 0,
        pagesize: int = 50,
        i18n: str = "pt_BR",
        lang: str = "pt_BR",
    ) -> Any:
        params: List[Tuple[str, Any]] = [
            ("devaddr", devaddr),
            ("oddEvenRow", oddEvenRow),
            ("pn", pn),
            ("devcode", devcode),
            ("sn", sn),
            ("date", day_yyyy_mm_dd),
            ("page", page),
            ("pagesize", pagesize),
            ("i18n", i18n),
            ("lang", lang),
        ]
        return self._call_action(token, secret, "queryDeviceDataOneDayPaging", params)

    # ---------------------------
    # Discovery: Plants
    # ---------------------------
    def query_plants(
        self,
        token: str,
        secret: str,
        page: int = 0,
        pagesize: int = 50,
        i18n: str = "pt_BR",
        lang: str = "pt_BR",
    ) -> Any:
        actions = getattr(settings, "RENOVIGI_PLANTS_ACTIONS", None) or [
            "queryPlants",
            "queryPlantsPaging",
            "queryPlantList",
            "queryPlantListPaging",
        ]

        last_err: Optional[Exception] = None
        for act in actions:
            try:
                params = [
                    ("page", page),
                    ("pagesize", pagesize),
                    ("i18n", i18n),
                    ("lang", lang),
                ]
                return self._call_action(token, secret, act, params)
            except Exception as e:
                last_err = e
                continue

        raise RenovigiError(f"Falha ao listar plantas. Último erro: {last_err}")

    # ---------------------------
    # Discovery: Plant -> Devices
    # ---------------------------
    def query_plant_device_view(
        self,
        token: str,
        secret: str,
        plantid: int,
        i18n: str = "pt_BR",
        lang: str = "pt_BR",
    ) -> Any:
        actions = getattr(settings, "RENOVIGI_PLANT_DEVICE_ACTIONS", None) or [
            "queryPlantDeviceView",
            "queryPlantDevice",
            "queryDeviceView",
        ]

        param_variants = [
            [("plantid", plantid), ("i18n", i18n), ("lang", lang)],
            [("pid", plantid), ("i18n", i18n), ("lang", lang)],
        ]

        last_err: Optional[Exception] = None
        for act in actions:
            for params in param_variants:
                try:
                    return self._call_action(token, secret, act, params)
                except Exception as e:
                    last_err = e
                    continue

        raise RenovigiError(f"Falha ao listar devices da planta {plantid}. Último erro: {last_err}")
