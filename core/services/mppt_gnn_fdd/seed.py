# core/services/mppt_gnn_fdd/seed.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo
from django.db.models import Count, Min, Max
from django.utils import timezone

from core.models import PVPlant, PVPlantMergedRecord15m, MPPTDiagnostic15m
from core.services.mppt_gnn_fdd.raw_extract import (
    aggregate_raw_mppt_15m_all_devices,
    list_raw_device_keys,
    best_device_key_for_source,
)


def _plant_tz(plant: PVPlant) -> ZoneInfo:
    tz_name = getattr(plant, "timezone", None) or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _pick_best_source_meteo(plant_id: int, dt0_utc: datetime, dt1_utc: datetime) -> Optional[str]:
    row = (
        PVPlantMergedRecord15m.objects.filter(plant_id=plant_id, ts_utc__gte=dt0_utc, ts_utc__lt=dt1_utc)
        .values("source_meteo")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    return (row or {}).get("source_meteo")


def _list_source_oper(plant_id: int, source_meteo: str, dt0_utc: datetime, dt1_utc: datetime) -> list[str]:
    rows = (
        PVPlantMergedRecord15m.objects.filter(
            plant_id=plant_id,
            source_meteo=source_meteo,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
        )
        .values("source_oper")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    return [r["source_oper"] for r in rows if r.get("source_oper")]


def _bucket_15m_utc(tsu: datetime) -> datetime:
    if tsu.tzinfo is None:
        tsu = tsu.replace(tzinfo=dt_tz.utc)
    tsu = tsu.astimezone(dt_tz.utc)
    minute = (tsu.minute // 15) * 15
    return tsu.replace(minute=minute, second=0, microsecond=0)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class SeedConfig:
    nmax_mppt: int = 4

    # "dia" se irradiância >= daylight_min_wm2 OU potência >= daylight_min_p_w
    daylight_min_wm2: float = 50.0
    daylight_min_p_w: float = 50.0

    # ---- Classe 1: MPPT disconnected (severo)
    disc_gate_wm2: float = 300.0
    i_zero_a: float = 0.10       # “quase zero”
    i_peer_a: float = 0.50       # peer “ativo”
    i_disc_ratio: float = 0.05   # ik <= 5% do peer_max

    # ---- Classe 2: Inverter OFF under sun (trip geral)
    off_gate_wm2: float = 400.0
    off_p_w: float = 25.0        # pac/pdc muito baixo sob sol

    # ---- Classe 3: MPPT imbalance (perda parcial / string degradada)
    imb_gate_wm2: float = 300.0
    imb_peer_a: float = 1.00     # peers razoáveis
    imb_ratio: float = 0.25      # ik <= 25% do peer_median

    allow_agg_fallback: bool = True
    prefer_raw_mppt: bool = True

    # 🔥 importante para não “sumir” por UNIQUE + ignore_conflicts
    replace_any_version: bool = True

    force_top_device_when_unmapped: bool = True

    # default (você vai passar via CLI normalmente)
    model_version: str = "seed_raw_v3"


def seed_mppt_predictions(
    *,
    plant_id: int,
    start: date,
    end: date,
    cfg: SeedConfig = SeedConfig(),
    source_meteo: str | None = None,
    source_oper: str | None = None,
    chunk_size: int = 5000,
) -> dict:
    plant = PVPlant.objects.filter(id=plant_id).first()
    if plant is None:
        raise ValueError("Plant not found")

    tz = _plant_tz(plant)
    d0, d1 = (start, end) if start <= end else (end, start)

    dt0_local = datetime.combine(d0, time.min, tzinfo=tz)
    dt1_local = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz)
    dt0_utc = dt0_local.astimezone(dt_tz.utc)
    dt1_utc = dt1_local.astimezone(dt_tz.utc)

    if source_meteo is None:
        source_meteo = _pick_best_source_meteo(plant_id, dt0_utc, dt1_utc)
    if not source_meteo:
        return {"ok": False, "error": "Não encontrei source_meteo para o range.", "created": 0}

    sources = [source_oper] if source_oper else _list_source_oper(plant_id, source_meteo, dt0_utc, dt1_utc)
    if not sources:
        return {"ok": False, "error": "Não encontrei source_oper para o range.", "created": 0}

    # ---------- RAW (InverterOperationalData payload) ----------
    raw_all: Dict[str, Dict[datetime, Dict[str, Any]]] = {}
    raw_device_keys_counts: list[tuple[str, int]] = []
    raw_device_keys: list[str] = []
    source_to_device: Dict[str, str] = {}
    raw_buckets_with_mppt = 0

    if cfg.prefer_raw_mppt:
        raw_device_keys_counts = list_raw_device_keys(plant_id=plant_id, start_utc=dt0_utc, end_utc=dt1_utc)
        raw_device_keys = [dk for dk, _n in raw_device_keys_counts]
        if raw_device_keys:
            raw_all = aggregate_raw_mppt_15m_all_devices(plant_id=plant_id, start_utc=dt0_utc, end_utc=dt1_utc)
            for dk, mp in raw_all.items():
                for _bts, row in mp.items():
                    if any(row.get(f"mppt{k}_idc_a") is not None for k in (1, 2, 3, 4)):
                        raw_buckets_with_mppt += 1

            if len(raw_device_keys) == 1:
                for s in sources:
                    source_to_device[str(s)] = raw_device_keys[0]
            else:
                for s in sources:
                    dk = best_device_key_for_source(str(s), raw_device_keys)
                    if dk:
                        source_to_device[str(s)] = dk

            if (not source_to_device) and cfg.force_top_device_when_unmapped and raw_device_keys_counts:
                top = raw_device_keys_counts[0][0]
                for s in sources:
                    source_to_device[str(s)] = top

    # ---------- REPLACE intervalo (qualquer versão) ----------
    before_mv = MPPTDiagnostic15m.objects.filter(plant_id=plant_id, model_version=cfg.model_version).count()
    before_any = MPPTDiagnostic15m.objects.filter(plant_id=plant_id).count()

    if cfg.replace_any_version:
        q = MPPTDiagnostic15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
            mppt__gte=1,
            mppt__lte=cfg.nmax_mppt,
            source_oper__in=[str(s) for s in sources],
        )
        deleted, _ = q.delete()
    else:
        deleted = 0

    # ---------- Loop merged ----------
    fields = [
        "ts_utc",
        "source_oper",
        "gti",
        "ghi",
        "p_dc_w",
        "p_ac_w",
        "i_dc_a",
        "temp_air",
    ]

    created_attempt = 0
    buf: List[MPPTDiagnostic15m] = []
    now = timezone.now()

    rows_scanned = 0
    rows_daylight = 0
    rows_have_mppt_current = 0
    rows_have_agg_current = 0
    rows_have_irr = 0
    rows_used_raw = 0

    for src in sources:
        src_key = str(src)

        qs = (
            PVPlantMergedRecord15m.objects.filter(
                plant_id=plant_id,
                source_meteo=source_meteo,
                source_oper=src,
                ts_utc__gte=dt0_utc,
                ts_utc__lt=dt1_utc,
            )
            .order_by("ts_utc")
            .values(*fields)
        )

        dk = source_to_device.get(src_key)
        raw_map = raw_all.get(dk) if dk else None

        for r in qs.iterator(chunk_size=2000):
            rows_scanned += 1

            tsu = r.get("ts_utc")
            if tsu is None:
                continue
            if tsu.tzinfo is None:
                tsu = tsu.replace(tzinfo=dt_tz.utc)

            # irradiância (usa GTI se existir, senão GHI)
            gti = _safe_float(r.get("gti"))
            ghi = _safe_float(r.get("ghi"))
            gpoa_val = gti if gti is not None else ghi
            if gpoa_val is not None:
                rows_have_irr += 1

            # potência
            pdc = _safe_float(r.get("p_dc_w"))
            pac = _safe_float(r.get("p_ac_w"))
            p_val = pdc if pdc is not None else pac

            # daylight
            daylight = False
            if gpoa_val is not None and gpoa_val >= cfg.daylight_min_wm2:
                daylight = True
            elif p_val is not None and p_val >= cfg.daylight_min_p_w:
                daylight = True
            if not daylight:
                continue
            rows_daylight += 1

            # correntes MPPT: prefer RAW
            I_mppt: list[Optional[float]] = [None, None, None, None]
            has_mppt_real = False

            if raw_map is not None:
                b = _bucket_15m_utc(tsu)
                raw_row = raw_map.get(b)
                if raw_row:
                    rows_used_raw += 1
                    for k in (1, 2, 3, 4):
                        I_mppt[k - 1] = _safe_float(raw_row.get(f"mppt{k}_idc_a"))

            has_mppt_real = any(v is not None for v in I_mppt)
            if has_mppt_real:
                rows_have_mppt_current += 1

            # fallback agregado
            i_dc = _safe_float(r.get("i_dc_a"))
            if i_dc is not None:
                rows_have_agg_current += 1

            if not has_mppt_real:
                if not (cfg.allow_agg_fallback and (i_dc is not None)):
                    continue
                I_mppt = [i_dc / float(cfg.nmax_mppt) for _ in range(cfg.nmax_mppt)]
                has_mppt_real = False  # fallback

            # ---------------------------
            # Classes globais (afetam todos MPPTs)
            # ---------------------------
            off_under_sun = (
                (gpoa_val is not None)
                and (gpoa_val >= cfg.off_gate_wm2)
                and (
                    ((pac is not None) and (pac <= cfg.off_p_w))
                    or ((pdc is not None) and (pdc <= cfg.off_p_w))
                )
            )

            # ---------------------------
            # Classificação por MPPT
            # pred_code: 0 normal
            #           1 mppt_disconnected
            #           2 inverter_off_under_sun
            #           3 mppt_imbalance
            # ---------------------------
            for k in range(1, cfg.nmax_mppt + 1):
                ik = I_mppt[k - 1]
                if ik is None:
                    continue

                pred_code = 0
                pred_label = "normal"
                pred_pmax = 0.80 if has_mppt_real else 0.60

                if off_under_sun:
                    pred_code = 2
                    pred_label = "inverter_off_under_sun"
                    pred_pmax = 0.93
                else:
                    peer = [v for j, v in enumerate(I_mppt) if j != (k - 1) and v is not None]
                    peer_max = max(peer) if peer else 0.0
                    peer_sorted = sorted(peer)
                    peer_med = peer_sorted[len(peer_sorted)//2] if peer_sorted else 0.0

                    # 1) disconnected (severo)
                    can_disc = (has_mppt_real is True) and (gpoa_val is not None) and (gpoa_val >= cfg.disc_gate_wm2)
                    if can_disc and (peer_max >= cfg.i_peer_a):
                        if (ik <= cfg.i_zero_a) or (peer_max > 0 and ik <= cfg.i_disc_ratio * peer_max):
                            pred_code = 1
                            pred_label = "mppt_disconnected"
                            pred_pmax = 0.95

                    # 2) imbalance (parcial)
                    if pred_code == 0:
                        can_imb = (has_mppt_real is True) and (gpoa_val is not None) and (gpoa_val >= cfg.imb_gate_wm2)
                        if can_imb and (peer_med >= cfg.imb_peer_a) and (peer_med > 0):
                            if ik <= cfg.imb_ratio * peer_med:
                                pred_code = 3
                                pred_label = "mppt_imbalance"
                                pred_pmax = 0.88

                proba = (
                    {"normal": 1.0 - pred_pmax, pred_label: pred_pmax}
                    if pred_code != 0
                    else {"normal": pred_pmax, "fault": 1.0 - pred_pmax}
                )

                buf.append(
                    MPPTDiagnostic15m(
                        plant_id=plant_id,
                        source_oper=src_key,
                        mppt=k,
                        ts_utc=tsu,
                        model_version=cfg.model_version,
                        pred_code=pred_code,
                        pred_label=pred_label,
                        pred_pmax=pred_pmax,
                        proba=proba,
                        created_at=now,
                        updated_at=now,
                    )
                )

                if len(buf) >= chunk_size:
                    MPPTDiagnostic15m.objects.bulk_create(buf, ignore_conflicts=True)
                    created_attempt += len(buf)
                    buf = []

    if buf:
        MPPTDiagnostic15m.objects.bulk_create(buf, ignore_conflicts=True)
        created_attempt += len(buf)

    after_mv = MPPTDiagnostic15m.objects.filter(plant_id=plant_id, model_version=cfg.model_version).count()
    after_any = MPPTDiagnostic15m.objects.filter(plant_id=plant_id).count()

    avail_mv = MPPTDiagnostic15m.objects.filter(plant_id=plant_id, model_version=cfg.model_version).aggregate(
        ts_min=Min("ts_utc"), ts_max=Max("ts_utc")
    )

    return {
        "ok": True,
        "plant_id": plant_id,
        "model_version": cfg.model_version,
        "source_meteo": source_meteo,
        "sources_count": len(sources),
        "replace_any_version": cfg.replace_any_version,
        "deleted": deleted,
        "created_attempt": created_attempt,
        "count_before_model_version": before_mv,
        "count_after_model_version": after_mv,
        "count_before_any": before_any,
        "count_after_any": after_any,
        "pred_available_min_utc": avail_mv["ts_min"].isoformat() if avail_mv["ts_min"] else None,
        "pred_available_max_utc": avail_mv["ts_max"].isoformat() if avail_mv["ts_max"] else None,
        "rows_scanned": rows_scanned,
        "rows_daylight": rows_daylight,
        "rows_have_irr": rows_have_irr,
        "rows_have_mppt_current": rows_have_mppt_current,
        "rows_have_agg_current": rows_have_agg_current,
        "rows_used_raw": rows_used_raw,
        "raw_device_keys_top": raw_device_keys_counts[:10],
        "raw_buckets_with_mppt": raw_buckets_with_mppt,
        "source_to_device_map": source_to_device,
        "hint": "seed_raw_v3 inclui inverter_off_under_sun e mppt_imbalance além de mppt_disconnected.",
    }