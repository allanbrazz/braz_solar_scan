# core/services/growatt_client.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from growattServer import GrowattApi

logger = logging.getLogger(__name__)


class GrowattReadError(Exception):
    """Erro genérico ao ler dados da Growatt."""


class GrowattAuthError(GrowattReadError):
    """Erro de autenticação (login) na Growatt."""


class GrowattClient:
    """
    Wrapper fino em cima da biblioteca growattServer.

    Usa login/senha do Shine/portal, obtém a lista de plantas e
    devolve um snapshot simples da primeira planta encontrada.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.api = GrowattApi()
        self.user_id: str | None = None
        self.login_response: Dict[str, Any] | None = None

    # ---------- Login ----------

    def login(self) -> Dict[str, Any]:
        """
        Faz login na Growatt usando a lib growattServer.
        Lança GrowattAuthError se falhar.
        """
        try:
            resp = self.api.login(self.username, self.password)
        except Exception as exc:  # erro de rede, etc.
            logger.exception("Erro ao contactar servidor Growatt")
            raise GrowattAuthError(f"Erro ao contactar servidor Growatt: {exc}") from exc

        # A lib normalmente retorna algo do tipo:
        # {"success": True, "userId": "...", ...}
        if not isinstance(resp, dict) or not resp.get("success"):
            raise GrowattAuthError(f"Login Growatt falhou: {resp!r}")

        self.login_response = resp
        self.user_id = resp.get("userId") or resp.get("user_id")

        if not self.user_id:
            raise GrowattAuthError(f"Login Growatt não retornou userId: {resp!r}")

        return resp

    # ---------- Plantas ----------

    def _get_plant_list_raw(self) -> Dict[str, Any]:
        """
        Usa o método de listagem de plantas da lib.
        O nome do método pode variar um pouco entre versões, então
        tentamos as duas convenções mais comuns.
        """
        if not self.user_id:
            self.login()

        # Algumas versões usam plant_list, outras plantList
        if hasattr(self.api, "plant_list"):
            func = self.api.plant_list  # type: ignore[attr-defined]
        elif hasattr(self.api, "plantList"):
            func = self.api.plantList  # type: ignore[attr-defined]
        else:
            raise GrowattReadError(
                "A instância GrowattApi não possui métodos plant_list nem plantList. "
                "Verifique a versão da biblioteca growattServer."
            )

        try:
            plants_resp = func(self.user_id)
        except Exception as exc:
            logger.exception("Erro ao obter lista de plantas Growatt")
            raise GrowattReadError(f"Erro ao obter lista de plantas: {exc}") from exc

        if not isinstance(plants_resp, dict) or not plants_resp.get("success", True):
            raise GrowattReadError(f"Resposta inesperada ao listar plantas: {plants_resp!r}")

        return plants_resp

    def get_first_plant(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Retorna (plant_dict, resposta_completa).
        Usa simplesmente a primeira planta da lista do usuário.
        """
        plants_resp = self._get_plant_list_raw()
        data = plants_resp.get("data") or plants_resp.get("plantList") or []

        if not data:
            raise GrowattReadError("Nenhuma planta encontrada na conta Growatt.")

        plant = data[0]
        return plant, plants_resp

    # ---------- Snapshot simplificado ----------

    def get_simple_snapshot(self) -> Dict[str, Any]:
        """
        Devolve um dicionário resumido com alguns campos mais úteis,
        além do JSON bruto da planta.
        """
        plant, plants_raw = self.get_first_plant()

        snapshot: Dict[str, Any] = {
            "plant_id": plant.get("plantId") or plant.get("id"),
            "plant_name": plant.get("plantName") or plant.get("plantName1"),
            # Os nomes exatos dos campos dependem da versão da API/conta,
            # então mantemos tanto o 'currPower' quanto 'power' etc.
            "current_power": plant.get("currPower")
            or plant.get("power")
            or plant.get("pac"),
            "today_energy_kwh": plant.get("todayEnergy")
            or plant.get("todayKwh"),
            "total_energy_kwh": plant.get("totalEnergy")
            or plant.get("totalKwh"),
            "raw_plant": plant,
            "raw_plants_response": plants_raw,
        }
        return snapshot


# ---------- Função de alto nível usada pelas views ----------

def fetch_growatt_plant_data(
    username: str,
    password: str,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Função de conveniência usada nas views.

    - Faz login
    - Pega a primeira planta do usuário
    - Retorna um snapshot simples
    - Se debug=True, inclui informações extras de login/lista.
    """
    client = GrowattClient(username=username, password=password)
    snapshot = client.get_simple_snapshot()

    if debug:
        snapshot["__debug"] = {
            "login_response": client.login_response,
            "plants_raw": snapshot.get("raw_plants_response"),
        }

    return snapshot
