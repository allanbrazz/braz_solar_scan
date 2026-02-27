# core/views/mppt_gnn_fdd.py
from __future__ import annotations

import json
import logging
import traceback
from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, Optional, Tuple, List

from django.conf import settings
from django.db.models import Count, Min, Max
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from zoneinfo import ZoneInfo

from core.views._imports import *  # segue teu padrão
from core.models import PVPlant

logger = logging.getLogger(__name__)

# Import opcional (para não quebrar se ainda não migrou)
try:
    from core.models import MPPTDiagnostic15m  # type: ignore
except Exception:  # pragma: no cover
    MPPTDiagnostic15m = None  # type: ignore

# serviços (opcionais)
try:
    from core.services.mppt_gnn_fdd.infer_pipeline import infer_day_and_persist  # type: ignore
except Exception:  # pragma: no cover
    infer_day_and_persist = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.train_pipeline import train_mppt_gnn_sklearn  # type: ignore
except Exception:  # pragma: no cover
    train_mppt_gnn_sklearn = None  # type: ignore

try:
    from core.services.mppt_gnn_fdd.window_loader import load_daily_window  # type: ignore
except Exception:  # pragma: no cover
    load_daily_window = None  # type: ignore


# ============================================================
# Modelo / classes (MVP aprovado)
# ============================================================
LABEL_BY_CODE: dict[int, str] = {
    0: "normal",
    1: "mppt_disconnected",
    2: "inverter_off_under_sun",
    3: "mppt_imbalance",
    4: "curtailment_clipping",
    5: "meteo_bias",
}

# severidade (para escolher "pior por bin")
SEV_BY_CODE: dict[int, int] = {
    0: 0,
    3: 2,  # imbalance
    1: 3,  # disconnected
    2: 3,  # inverter off
    4: 2,  # curtailment (normalmente não é “crítico” elétrico, mas é evento)
    5: 1,  # meteo_bias (alerta)
}


# ============================================================
# Helpers
# ============================================================
def _plant_tz(plant: PVPlant) -> ZoneInfo:
    tz_name = getattr(plant, "timezone", None) or getattr(settings, "TIME_ZONE", "UTC")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _parse_date(s: Optional[str], default: date) -> date:
    if not s:
        return default
    try:
        return date.fromisoformat(s)
    except Exception:
        return default


def _parse_int(s: Optional[str], default: int, lo: int, hi: int) -> int:
    try:
        v = int(s) if s is not None else default
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "sim", "s"):
        return True
    if v in ("0", "false", "f", "no", "n", "nao", "não"):
        return False
    return default


def _bins_per_day(dt_minutes: int) -> int:
    return int(24 * 60 // dt_minutes)


def _tkey(dt_local: datetime) -> str:
    # chave estável para map em JS
    return dt_local.strftime("%Y-%m-%dT%H:%M")


def _parse_tkey_to_local(tkey: str, tz: ZoneInfo) -> Optional[datetime]:
    """
    tkey esperado: YYYY-MM-DDTHH:MM (local)
    """
    try:
        if " " in tkey and "T" not in tkey:
            tkey = tkey.replace(" ", "T")
        dt = datetime.fromisoformat(tkey)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        return dt
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sev_rank(code: int) -> int:
    return int(SEV_BY_CODE.get(code, 1 if code != 0 else 0))


def _norm_label(code: int, raw_label: Any) -> str:
    s = (str(raw_label or "").strip() if raw_label is not None else "")
    sl = s.lower()
    if not s:
        return LABEL_BY_CODE.get(code, "fault")
    # corrige legado
    if sl == "disconnected":
        return "mppt_disconnected"
    if code != 0 and sl == "normal":
        return LABEL_BY_CODE.get(code, "fault")
    return s


def _json_body(request: HttpRequest) -> dict:
    try:
        raw = request.body.decode("utf-8") if request.body else ""
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _error_json(msg: str, *, trace: Optional[str] = None) -> JsonResponse:
    payload: Dict[str, Any] = {"ok": False, "error": msg}
    if getattr(settings, "DEBUG", False) and trace:
        payload["trace"] = trace
    return JsonResponse(payload, status=200)


# ============================================================
# View: página
# ============================================================
@require_GET
@login_required
def mppt_gnn_fdd_view(request: HttpRequest):
    """
    Página MPPT-GNN (classificação multi-classe).
    O template vai comandar:
      - atualizar heatmap (GET api)
      - buscar dump do ponto (GET dump api)
      - disparar inferência (POST actions api)
    """
    qs = (
        PVPlant.objects.all().order_by("nome")
        if request.user.is_superuser
        else PVPlant.objects.filter(owner=request.user).order_by("nome")
    )
    plants = list(qs)

    d_end = date.today()
    d_start = d_end - timedelta(days=7)

    plant_id = request.GET.get("plant_id") or request.GET.get("pk") or request.GET.get("plant_pk")
    if not plant_id and plants:
        plant_id = str(plants[0].id)

    # defaults para novo modelo
    model_version = request.GET.get("model_version") or "gnn_v1"
    source_oper = request.GET.get("source_oper") or "MPPT_GNN"

    start_q = request.GET.get("start")
    end_q = request.GET.get("end")

    # default start/end baseado no que existe em MPPTDiagnostic15m
    if MPPTDiagnostic15m is not None and plant_id and (not start_q or not end_q):
        try:
            plant_obj = PVPlant.objects.filter(id=int(plant_id)).first()
            if plant_obj:
                tz = _plant_tz(plant_obj)
                q = MPPTDiagnostic15m.objects.filter(plant_id=int(plant_id))
                if model_version:
                    q = q.filter(model_version=model_version)
                if source_oper:
                    # no dataset novo, normalmente é fixo "MPPT_GNN"
                    q = q.filter(source_oper__startswith=str(source_oper))
                agg = q.aggregate(ts_min=Min("ts_utc"), ts_max=Max("ts_utc"))
                if agg["ts_max"]:
                    end_local = agg["ts_max"].astimezone(tz).date()
                    start_local = (end_local - timedelta(days=7))
                    start_q = start_q or start_local.isoformat()
                    end_q = end_q or end_local.isoformat()
        except Exception:
            pass

    return render(
        request,
        "dashboard/mppt_gnn_fdd.html",
        {
            "plants": plants,
            "plant_id": plant_id,
            "start": start_q or d_start.isoformat(),
            "end": end_q or d_end.isoformat(),
            "dt_minutes": int(float(request.GET.get("dt_minutes") or 15)),
            "mppt": int(float(request.GET.get("mppt") or 1)),
            "dt_options": [5, 10, 15, 30, 60],
            "mppt_options": [1, 2, 3, 4],
            "model_version": model_version,
            "source_oper": source_oper,
            # URLs para o template “comandar tudo”
            "api_url": reverse("mppt_gnn_fdd_api"),
            "dump_url": reverse("mppt_gnn_fdd_dump_api"),
            "actions_url": reverse("mppt_gnn_fdd_actions_api"),
        },
    )


# ============================================================
# API 1: Heatmap (rápido) com agregação "pior por bin"
# ============================================================
@require_GET
@login_required
def mppt_gnn_fdd_api(request: HttpRequest) -> JsonResponse:
    """
    Retorna (rápido):
      - grid: [days][bins] 0 none, 1 normal, 2 fault (qualquer pred_code != 0)
      - tkeys: [days][bins] tkey ou null
      - pred_count
      - counts_by_code / counts_by_label (para KPI/legenda)
      - available: listas p/ UI (model_versions/source_opers)

    NÃO faz compute pesado (power_model). O detalhe do ponto vem no dump_api.
    """
    try:
        if MPPTDiagnostic15m is None:
            return _error_json("MPPTDiagnostic15m não está disponível (migrate pendente).")

        plant_id = int(request.GET.get("plant_id") or request.GET.get("plant") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        tz = _plant_tz(plant)

        d_end = _parse_date(request.GET.get("end"), default=date.today())
        d_start = _parse_date(request.GET.get("start"), default=(d_end - timedelta(days=7)))
        if d_start > d_end:
            d_start, d_end = d_end, d_start

        dt_minutes = _parse_int(request.GET.get("dt_minutes"), default=15, lo=5, hi=60)
        mppt = _parse_int(request.GET.get("mppt"), default=1, lo=1, hi=32)

        model_version = (request.GET.get("model_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None

        # range local [start 00:00, end+1 00:00)
        dt0_local = datetime.combine(d_start, time.min, tzinfo=tz)
        dt1_local = datetime.combine(d_end + timedelta(days=1), time.min, tzinfo=tz)
        dt0_utc = dt0_local.astimezone(dt_tz.utc)
        dt1_utc = dt1_local.astimezone(dt_tz.utc)

        bpd = _bins_per_day(dt_minutes)
        days: List[str] = []
        cur = d_start
        while cur <= d_end:
            days.append(cur.isoformat())
            cur = cur + timedelta(days=1)

        grid = [[0 for _ in range(bpd)] for _ in range(len(days))]
        tkeys: List[List[Optional[str]]] = [[None for _ in range(bpd)] for _ in range(len(days))]

        # query
        pred_qs = MPPTDiagnostic15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
            mppt=mppt,
        )
        if model_version:
            pred_qs = pred_qs.filter(model_version=model_version)

        if source_oper:
            if "|" in source_oper:
                pred_qs = pred_qs.filter(source_oper=source_oper)
            else:
                pred_qs = pred_qs.filter(source_oper__startswith=source_oper)

        # estabilidade
        pred_rows = list(
            pred_qs.order_by("ts_utc").values(
                "ts_utc",
                "pred_code",
                "pred_label",
                "pred_pmax",
            )
        )
        pred_count = len(pred_rows)

        # available para UI (sempre, mesmo com pred_count > 0)
        mv_list = list(
            MPPTDiagnostic15m.objects.filter(plant_id=plant_id)
            .values_list("model_version", flat=True)
            .distinct()
            .order_by("model_version")
        )
        so_list = list(
            MPPTDiagnostic15m.objects.filter(plant_id=plant_id)
            .values_list("source_oper", flat=True)
            .distinct()
            .order_by("source_oper")
        )

        if pred_count == 0:
            avail = MPPTDiagnostic15m.objects.filter(plant_id=plant_id).aggregate(ts_min=Min("ts_utc"), ts_max=Max("ts_utc"))
            return JsonResponse(
                {
                    "ok": True,
                    "plant_id": plant_id,
                    "timezone": str(tz),
                    "start": d_start.isoformat(),
                    "end": d_end.isoformat(),
                    "dt_minutes": dt_minutes,
                    "bins_per_day": bpd,
                    "days": days,
                    "grid": grid,
                    "tkeys": tkeys,
                    "pred_count": 0,
                    "available": {
                        "pred_min_utc": avail["ts_min"].isoformat() if avail["ts_min"] else None,
                        "pred_max_utc": avail["ts_max"].isoformat() if avail["ts_max"] else None,
                        "model_versions": mv_list,
                        "source_opers": so_list,
                    },
                    "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
                    "hint": "Sem predições para os filtros atuais.",
                },
                status=200,
            )

        # ---------------------------------------------------------
        # Agregação por bin: escolhe "pior"
        # score: severidade > pmax > proximidade centro do bin
        # ---------------------------------------------------------
        best_score: Dict[Tuple[int, int], float] = {}
        best_row: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for r in pred_rows:
            tsu = r["ts_utc"]
            if isinstance(tsu, str):
                tsu = datetime.fromisoformat(tsu.replace("Z", "+00:00"))
            if tsu.tzinfo is None:
                tsu = tsu.replace(tzinfo=dt_tz.utc)

            ts_local = tsu.astimezone(tz)
            di = (ts_local.date() - d_start).days
            if di < 0 or di >= len(days):
                continue

            minutes = ts_local.hour * 60 + ts_local.minute
            bi = int(minutes // dt_minutes)
            if bi < 0 or bi >= bpd:
                continue

            code = int(r.get("pred_code") or 0)
            sev = _sev_rank(code)
            pmax = _safe_float(r.get("pred_pmax"), 0.0)
            center = bi * dt_minutes + (dt_minutes / 2.0)
            dist = abs(float(minutes) - center)

            score = (sev * 1_000_000_000.0) + (pmax * 1_000_000.0) - dist
            key = (di, bi)

            prev = best_score.get(key)
            if prev is None or score > prev:
                best_score[key] = score
                best_row[key] = {
                    "tsu": tsu,
                    "ts_local": ts_local,
                    "code": code,
                    "label": _norm_label(code, r.get("pred_label")),
                    "pmax": pmax,
                }

        # preencher grid/tkeys e contar classes
        counts_by_code: Dict[str, int] = {}
        counts_by_label: Dict[str, int] = {}

        for (di, bi), info in best_row.items():
            code = int(info["code"])
            label = str(info["label"])
            state = 2 if code != 0 else 1

            grid[di][bi] = state
            tk = _tkey(info["ts_local"])
            tkeys[di][bi] = tk

            kcode = str(code)
            counts_by_code[kcode] = counts_by_code.get(kcode, 0) + 1
            counts_by_label[label] = counts_by_label.get(label, 0) + 1

        return JsonResponse(
            {
                "ok": True,
                "plant_id": plant_id,
                "timezone": str(tz),
                "start": d_start.isoformat(),
                "end": d_end.isoformat(),
                "dt_minutes": dt_minutes,
                "bins_per_day": bpd,
                "days": days,
                "grid": grid,
                "tkeys": tkeys,
                "pred_count": pred_count,
                "counts_by_code": counts_by_code,
                "counts_by_label": counts_by_label,
                "available": {
                    "model_versions": mv_list,
                    "source_opers": so_list,
                },
                "echo": {"model_version": model_version, "source_oper": source_oper, "mppt": mppt},
            },
            status=200,
        )

    except Exception as e:
        logger.exception("mppt_gnn_fdd_api failed")
        return _error_json(str(e), trace=traceback.format_exc())


# ============================================================
# API 2: Dump (detalhe de 1 bin/tkey) — inclui pac_model/mismatch
# ============================================================
@require_GET
@login_required
def mppt_gnn_fdd_dump_api(request: HttpRequest) -> JsonResponse:
    """
    Entrada:
      plant_id, mppt, model_version, source_oper, tkey (local) OU ts_local
    Saída:
      dump completo (pred + vars globais + vars mppt + proba)
    """
    try:
        if MPPTDiagnostic15m is None:
            return _error_json("MPPTDiagnostic15m não está disponível (migrate pendente).")

        plant_id = int(request.GET.get("plant_id") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        tz = _plant_tz(plant)

        mppt = _parse_int(request.GET.get("mppt"), default=1, lo=1, hi=32)

        model_version = (request.GET.get("model_version") or "").strip() or None
        source_oper = (request.GET.get("source_oper") or "").strip() or None

        tkey = (request.GET.get("tkey") or request.GET.get("ts_local") or "").strip()
        if not tkey:
            return _error_json("tkey/ts_local obrigatório (formato YYYY-MM-DDTHH:MM)")

        dt_local = _parse_tkey_to_local(tkey, tz)
        if dt_local is None:
            return _error_json("tkey inválido; esperado YYYY-MM-DDTHH:MM")

        # normaliza para minuto
        dt_local = dt_local.replace(second=0, microsecond=0)
        tsu = dt_local.astimezone(dt_tz.utc)

        q = MPPTDiagnostic15m.objects.filter(
            plant_id=plant_id,
            mppt=mppt,
            ts_utc=tsu,
        )
        if model_version:
            q = q.filter(model_version=model_version)
        if source_oper:
            if "|" in source_oper:
                q = q.filter(source_oper=source_oper)
            else:
                q = q.filter(source_oper__startswith=source_oper)

        row = q.values(
            "ts_utc",
            "source_oper",
            "mppt",
            "model_version",
            "pred_code",
            "pred_label",
            "pred_pmax",
            "proba",
        ).first()

        if not row:
            return JsonResponse(
                {
                    "ok": True,
                    "found": False,
                    "plant_id": plant_id,
                    "mppt": mppt,
                    "tkey": _tkey(dt_local),
                    "echo": {"model_version": model_version, "source_oper": source_oper},
                    "hint": "Sem predição exata para esse ts_utc. (Talvez dt_minutes diferente ou seed não gerou nesse instante.)",
                },
                status=200,
            )

        code = int(row.get("pred_code") or 0)
        label = _norm_label(code, row.get("pred_label"))

        # carrega janela diária para pegar: pac, vdc_total, iac, pac_model, mismatch, G, T e MPPT v/i (se existirem)
        vars_global: Dict[str, Any] = {}
        vars_mppt: Dict[str, Any] = {}

        if load_daily_window is not None:
            day_local = dt_local.date()
            win, ts_grid, meta = load_daily_window(plant_id=plant_id, day_local=day_local, n_mppt=4)
            idx = None
            try:
                idx = {t.astimezone(dt_tz.utc): i for i, t in enumerate(ts_grid)}.get(tsu)
            except Exception:
                idx = None

            if idx is not None:
                # globais
                vars_global = {
                    "pac_w": float(win.pac[idx]) if win.pac is not None else None,
                    "vdc_total_v": float(win.vdc_total[idx]) if win.vdc_total is not None else None,
                    "iac_a": float(win.iac[idx]) if win.iac is not None else None,
                    "pac_model_w": float(win.pac_model[idx]) if win.pac_model is not None else None,
                    "mismatch": float(win.mismatch[idx]) if win.mismatch is not None else None,
                    "g_wm2": float(win.g[idx]) if win.g is not None else None,
                    "t_air_c": float(win.t[idx]) if win.t is not None else None,
                }
                # por MPPT (1..4)
                k = mppt
                if 1 <= k <= win.mppt_vdc.shape[0]:
                    vmp = win.mppt_vdc[k - 1, idx]
                    imp = win.mppt_idc[k - 1, idx]
                    vars_mppt = {
                        "mppt_vdc_v": float(vmp) if vmp == vmp else None,
                        "mppt_idc_a": float(imp) if imp == imp else None,
                        "mppt_pdc_est_w": float(vmp * imp) if (vmp == vmp and imp == imp) else None,
                    }

        dump = {
            "ts_local": dt_local.isoformat(),
            "ts_utc": tsu.isoformat(),
            "plant_id": plant_id,
            "mppt": mppt,
            "source_oper": row.get("source_oper"),
            "model_version": row.get("model_version"),
            "pred": {
                "pred_code": code,
                "pred_label": label,
                "pred_pmax": row.get("pred_pmax"),
                "proba": row.get("proba"),
            },
            "vars_global": vars_global,
            "vars_mppt": vars_mppt,
        }

        return JsonResponse({"ok": True, "found": True, "dump": dump}, status=200)

    except Exception as e:
        logger.exception("mppt_gnn_fdd_dump_api failed")
        return _error_json(str(e), trace=traceback.format_exc())


# ============================================================
# API 3: Actions (inferir / (opcional) treinar)
# ============================================================
@csrf_exempt  # facilita fetch sem CSRF no começo; depois você pode remover e mandar token no template
@require_POST
@login_required
def mppt_gnn_fdd_actions_api(request: HttpRequest) -> JsonResponse:
    """
    POST JSON:
      {
        "action": "infer",
        "plant_id": 2,
        "start": "2023-07-07",
        "end": "2023-07-07",
        "model_version": "gnn_v1",
        "source_oper": "MPPT_GNN",
        "delete_existing": 1
      }

    Retorna resumo por dia.
    """
    try:
        body = _json_body(request)
        action = str(body.get("action") or "").strip().lower()
        if not action:
            return _error_json("action obrigatório (infer | train)")

        plant_id = int(body.get("plant_id") or 0)
        if not plant_id:
            return _error_json("plant_id obrigatório")

        plant = PVPlant.objects.filter(id=plant_id).first()
        if plant is None:
            return _error_json("Plant not found")

        if not (request.user.is_superuser or getattr(plant, "owner_id", None) == request.user.id):
            return _error_json("Forbidden")

        start = _parse_date(str(body.get("start") or ""), default=date.today())
        end = _parse_date(str(body.get("end") or ""), default=start)
        if start > end:
            start, end = end, start

        model_version = str(body.get("model_version") or "gnn_v1").strip()
        source_oper = str(body.get("source_oper") or "MPPT_GNN").strip()
        delete_existing = bool(int(body.get("delete_existing") or 1))

        if action == "infer":
            if infer_day_and_persist is None:
                return _error_json("infer_day_and_persist não disponível. Verifique core/services/mppt_gnn_fdd/infer_pipeline.py")

            cur = start
            outs: List[dict] = []
            while cur <= end:
                out = infer_day_and_persist(
                    plant_id=plant_id,
                    day_local=cur,
                    model_version=model_version,
                    source_oper=source_oper,
                    delete_existing=delete_existing,
                )
                outs.append(out)
                cur = cur + timedelta(days=1)

            return JsonResponse(
                {"ok": True, "action": "infer", "plant_id": plant_id, "model_version": model_version, "days": len(outs), "details": outs},
                status=200,
            )

        if action == "train":
            # opcional (pode demorar)
            if train_mppt_gnn_sklearn is None:
                return _error_json("train_mppt_gnn_sklearn não disponível. Verifique core/services/mppt_gnn_fdd/train_pipeline.py")

            out = train_mppt_gnn_sklearn(
                plant_id=plant_id,
                start=start,
                end=end,
                model_version=model_version,
            )
            return JsonResponse({"ok": True, "action": "train", "result": out}, status=200)

        return _error_json("action inválido (use infer | train)")

    except Exception as e:
        logger.exception("mppt_gnn_fdd_actions_api failed")
        return _error_json(str(e), trace=traceback.format_exc())