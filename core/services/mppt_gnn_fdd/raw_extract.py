# core/services/mppt_gnn_fdd/raw_extract.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone as dt_tz
from typing import Any, Dict, Optional, Tuple

from django.db.models import Count

from core.models import InverterSample, InverterOperationalData


def _norm_id(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
        return float(x)
    except Exception:
        return None


def _finite_or_none(x: Any) -> Optional[float]:
    v = _to_float(x)
    if v is None:
        return None
    try:
        # evita NaN/inf
        if v != v:
            return None
    except Exception:
        return None
    return v


def _bucket_15m_utc(tsu: datetime) -> datetime:
    if tsu.tzinfo is None:
        tsu = tsu.replace(tzinfo=dt_tz.utc)
    tsu = tsu.astimezone(dt_tz.utc)
    minute = (tsu.minute // 15) * 15
    return tsu.replace(minute=minute, second=0, microsecond=0)


def best_device_key_for_source(source_oper: str, device_keys: list[str]) -> Optional[str]:
    """
    Heurística para mapear source_oper -> device_key quando não batem exatamente.
    """
    if not source_oper or not device_keys:
        return None

    s = _norm_id(source_oper)
    if not s:
        return None

    # 1) match exato normalizado
    for dk in device_keys:
        if _norm_id(dk) == s:
            return dk

    # 2) contains (um contém o outro)
    best = None
    best_score = 0.0
    for dk in device_keys:
        d = _norm_id(dk)
        if not d:
            continue
        if s in d or d in s:
            score = 0.9 * (min(len(s), len(d)) / max(len(s), len(d)))
            if score > best_score:
                best_score = score
                best = dk

    if best is not None:
        return best

    # 3) prefix similarity
    for dk in device_keys:
        d = _norm_id(dk)
        common = 0
        for a, b in zip(s, d):
            if a == b:
                common += 1
            else:
                break
        score = common / max(len(s), len(d), 1)
        if score > best_score:
            best_score = score
            best = dk

    return best


@dataclass
class RawMPPTAgg:
    sum_v: Dict[int, float]
    sum_i: Dict[int, float]
    cnt_v: Dict[int, int]
    cnt_i: Dict[int, int]
    sum_pdc: float
    cnt_pdc: int

    @classmethod
    def new(cls) -> "RawMPPTAgg":
        return cls(sum_v={}, sum_i={}, cnt_v={}, cnt_i={}, sum_pdc=0.0, cnt_pdc=0)

    def add(self, mppt: int, v: Optional[float], i: Optional[float], pdc: Optional[float]) -> None:
        if v is not None:
            self.sum_v[mppt] = self.sum_v.get(mppt, 0.0) + v
            self.cnt_v[mppt] = self.cnt_v.get(mppt, 0) + 1
        if i is not None:
            self.sum_i[mppt] = self.sum_i.get(mppt, 0.0) + i
            self.cnt_i[mppt] = self.cnt_i.get(mppt, 0) + 1
        if pdc is not None:
            self.sum_pdc += pdc
            self.cnt_pdc += 1

    def mean_v(self, mppt: int) -> Optional[float]:
        c = self.cnt_v.get(mppt, 0)
        return (self.sum_v.get(mppt, 0.0) / c) if c > 0 else None

    def mean_i(self, mppt: int) -> Optional[float]:
        c = self.cnt_i.get(mppt, 0)
        return (self.sum_i.get(mppt, 0.0) / c) if c > 0 else None

    def mean_pdc(self) -> Optional[float]:
        return (self.sum_pdc / self.cnt_pdc) if self.cnt_pdc > 0 else None


def _device_key_from_opdata(r: Dict[str, Any]) -> str:
    """
    Device key estável a partir dos campos do InverterOperationalData.
    """
    prov = str(r.get("provedor") or "RENOVIGI")
    pn = str(r.get("pn") or "")
    devcode = str(r.get("devcode") or "")
    devaddr = str(r.get("devaddr") or "")
    sn = str(r.get("sn") or "")
    return f"{prov}|{pn}|{devcode}|{devaddr}|{sn}"


def list_raw_device_keys(*, plant_id: int, start_utc: datetime, end_utc: datetime) -> list[tuple[str, int]]:
    """
    Retorna [(device_key, count)] no range.
    1) tenta InverterSample; se vazio
    2) cai para InverterOperationalData agrupado por device.
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=dt_tz.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=dt_tz.utc)

    # --- 1) InverterSample ---
    rows = (
        InverterSample.objects.filter(plant_id=plant_id, ts__gte=start_utc, ts__lt=end_utc)
        .values("device_key")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    out = [(str(r["device_key"]), int(r["n"])) for r in rows if r.get("device_key")]
    if out:
        return out

    # --- 2) InverterOperationalData ---
    rows2 = (
        InverterOperationalData.objects.filter(plant_id=plant_id, ts_utc__gte=start_utc, ts_utc__lt=end_utc)
        .values("provedor", "pn", "devcode", "devaddr", "sn")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    out2 = []
    for r in rows2:
        dk = f"{r.get('provedor') or 'RENOVIGI'}|{r.get('pn') or ''}|{r.get('devcode') or ''}|{r.get('devaddr') or ''}|{r.get('sn') or ''}"
        out2.append((dk, int(r.get("n") or 0)))
    return out2


def aggregate_raw_mppt_15m_all_devices(
    *,
    plant_id: int,
    start_utc: datetime,
    end_utc: datetime,
) -> Dict[str, Dict[datetime, Dict[str, Any]]]:
    """
    Agrega MPPT em buckets 15min UTC para TODOS os devices disponíveis.
    Fonte:
      - InverterSample se existir no range
      - senão InverterOperationalData.payload (usando extrator existente do projeto)
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=dt_tz.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=dt_tz.utc)

    # --- 1) InverterSample ---
    qs_s = (
        InverterSample.objects.filter(plant_id=plant_id, ts__gte=start_utc, ts__lt=end_utc)
        .values("ts", "device_key", "data")
        .order_by("ts")
    )
    if qs_s.exists():
        aggs: Dict[str, Dict[datetime, RawMPPTAgg]] = {}
        for r in qs_s.iterator(chunk_size=2000):
            tsu = r.get("ts")
            dk = r.get("device_key")
            payload = r.get("data") or {}
            if tsu is None or not dk or not isinstance(payload, dict):
                continue
            if tsu.tzinfo is None:
                tsu = tsu.replace(tzinfo=dt_tz.utc)
            b = _bucket_15m_utc(tsu)

            d_aggs = aggs.setdefault(str(dk), {})
            agg = d_aggs.setdefault(b, RawMPPTAgg.new())

            # aqui só tenta ler chaves já normalizadas; se você usa InverterSample, provavelmente já gravou normalizado
            pdc = _finite_or_none(payload.get("p_dc_w") or payload.get("pdc"))
            for mp in (1, 2, 3, 4):
                v = _finite_or_none(payload.get(f"mppt{mp}_vdc_v"))
                i = _finite_or_none(payload.get(f"mppt{mp}_idc_a"))
                agg.add(mp, v, i, pdc)

        out: Dict[str, Dict[datetime, Dict[str, Any]]] = {}
        for dk, d_aggs in aggs.items():
            dk_out: Dict[datetime, Dict[str, Any]] = {}
            for bts, agg in d_aggs.items():
                row: Dict[str, Any] = {"p_dc_w": agg.mean_pdc()}
                for mp in (1, 2, 3, 4):
                    row[f"mppt{mp}_vdc_v"] = agg.mean_v(mp)
                    row[f"mppt{mp}_idc_a"] = agg.mean_i(mp)
                dk_out[bts] = row
            out[dk] = dk_out
        return out

    # --- 2) InverterOperationalData.payload ---
    # Reutiliza o extrator já existente no teu projeto (Renovigi/ShineMonitor)
    from core.services.series_juntar.timeseries_io import _extract_payload  # noqa: WPS433

    qs_o = (
        InverterOperationalData.objects.filter(plant_id=plant_id, ts_utc__gte=start_utc, ts_utc__lt=end_utc)
        .values("ts_utc", "provedor", "pn", "devcode", "devaddr", "sn", "payload")
        .order_by("ts_utc")
    )

    aggs2: Dict[str, Dict[datetime, RawMPPTAgg]] = {}

    for r in qs_o.iterator(chunk_size=2000):
        tsu = r.get("ts_utc")
        if tsu is None:
            continue
        if tsu.tzinfo is None:
            tsu = tsu.replace(tzinfo=dt_tz.utc)
        b = _bucket_15m_utc(tsu)

        dk = _device_key_from_opdata(r)
        d_aggs = aggs2.setdefault(dk, {})
        agg = d_aggs.setdefault(b, RawMPPTAgg.new())

        prov = str(r.get("provedor") or "RENOVIGI")
        payload = r.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        m = _extract_payload(prov, payload)  # retorna mppt*_v_dc_v / mppt*_i_dc_a

        pdc = _finite_or_none(m.get("p_dc_w"))

        for mp in (1, 2, 3, 4):
            v = _finite_or_none(m.get(f"mppt{mp}_v_dc_v"))
            i = _finite_or_none(m.get(f"mppt{mp}_i_dc_a"))
            agg.add(mp, v, i, pdc)

    out2: Dict[str, Dict[datetime, Dict[str, Any]]] = {}
    for dk, d_aggs in aggs2.items():
        dk_out: Dict[datetime, Dict[str, Any]] = {}
        for bts, agg in d_aggs.items():
            row: Dict[str, Any] = {"p_dc_w": agg.mean_pdc()}
            for mp in (1, 2, 3, 4):
                row[f"mppt{mp}_vdc_v"] = agg.mean_v(mp)
                row[f"mppt{mp}_idc_a"] = agg.mean_i(mp)
            dk_out[bts] = row
        out2[dk] = dk_out

    return out2