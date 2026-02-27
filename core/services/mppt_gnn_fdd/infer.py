from __future__ import annotations

from datetime import date, datetime, timezone as dt_tz
from typing import Dict, List, Tuple

import numpy as np
import torch
from django.utils import timezone
from django.db import transaction

from core.models import PVPlant, MPPTDiagnostic15m
from core.services.mppt_gnn_fdd.config import TrainConfig
from core.services.mppt_gnn_fdd.model import MPPTGNNFDD
from core.services.mppt_gnn_fdd.windowing import iter_graph_windows_from_db
from core.services.mppt_gnn_fdd.io.storage import model_path


def load_model(name: str) -> tuple[MPPTGNNFDD, list[str], dict]:
    path = model_path(name, suffix=".pt")
    ckpt = torch.load(path, map_location="cpu")
    class_names = list(ckpt["class_names"])
    cfg = dict(ckpt.get("cfg") or {})
    fin_ts = 5 if bool(cfg.get("use_meteo", True)) else 3
    model = MPPTGNNFDD(fin_ts=fin_ts, fe=4, n_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, ckpt


@torch.no_grad()
def _predict_batch(model: MPPTGNNFDD, X: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: [B,N,T,F], E:[B,N,N,Fe]
    returns: pred[B,N], pmax[B,N], proba[B,N,C]
    """
    xb = torch.from_numpy(X).float()
    eb = torch.from_numpy(E).float()
    logits = model(xb, eb)
    proba = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = proba.argmax(axis=-1)
    pmax = proba.max(axis=-1)
    return pred, pmax, proba


def bulk_upsert_mppt_preds(
    *,
    plant: PVPlant,
    rows: list[dict],
) -> dict:
    """
    rows: [{source_oper, mppt, ts_utc, model_version, pred_code, pred_label, pred_pmax, proba}, ...]
    """
    if not rows:
        return {"created": 0, "updated": 0}

    keys = [(r["source_oper"], int(r["mppt"]), r["ts_utc"]) for r in rows]
    # chunked fetch
    existing: Dict[tuple[str, int, datetime], MPPTDiagnostic15m] = {}
    chunk = 800
    for i in range(0, len(keys), chunk):
        sub = keys[i:i+chunk]
        # fetch by ts range + filter in python (db can't do tuple-in easily)
        ts_set = list({t for (_, _, t) in sub})
        qs = MPPTDiagnostic15m.objects.filter(plant=plant, ts_utc__in=ts_set)
        for obj in qs:
            existing[(obj.source_oper, int(obj.mppt), obj.ts_utc)] = obj

    to_create: List[MPPTDiagnostic15m] = []
    to_update: List[MPPTDiagnostic15m] = []
    now = timezone.now()

    for r in rows:
        key = (r["source_oper"], int(r["mppt"]), r["ts_utc"])
        obj = existing.get(key)
        is_new = obj is None
        if is_new:
            obj = MPPTDiagnostic15m(
                plant=plant,
                source_oper=key[0],
                mppt=key[1],
                ts_utc=key[2],
            )
            to_create.append(obj)
        else:
            to_update.append(obj)

        obj.model_version = r["model_version"]
        obj.pred_code = int(r["pred_code"])
        obj.pred_label = str(r["pred_label"] or "")
        obj.pred_pmax = float(r["pred_pmax"]) if r.get("pred_pmax") is not None else None
        obj.proba = r.get("proba")
        if hasattr(obj, "updated_at"):
            setattr(obj, "updated_at", now)

    with transaction.atomic():
        if to_create:
            MPPTDiagnostic15m.objects.bulk_create(to_create, ignore_conflicts=True)
        if to_update:
            fields = ["model_version", "pred_code", "pred_label", "pred_pmax", "proba", "updated_at"]
            MPPTDiagnostic15m.objects.bulk_update(to_update, fields=fields)

    return {"created": len(to_create), "updated": len(to_update)}


def infer_and_persist(
    *,
    plant_id: int,
    start: date,
    end: date,
    model_name: str = "mppt_gnn_v1",
    source_oper: str | None = None,
    source_meteo: str | None = None,
) -> dict:
    plant = PVPlant.objects.filter(id=plant_id).select_related("details", "details__module").first()
    if plant is None:
        raise ValueError("Plant not found")

    model, class_names, ckpt = load_model(model_name)
    cfg = TrainConfig(**(ckpt.get("cfg") or {}))
    cfg = TrainConfig(**{**cfg.__dict__, "stride_steps": 1})  # infer a cada 15min

    rows_out: list[dict] = []
    batch: list = []

    for w in iter_graph_windows_from_db(
        plant_id=plant_id, start=start, end=end, cfg=cfg,
        source_oper=source_oper, source_meteo=source_meteo,
        stride_steps=1, mode="infer",
    ):
        batch.append(w)
        if len(batch) >= 64:
            rows_out.extend(_infer_batch_to_rows(batch, model, class_names, model_name))
            batch = []

    if batch:
        rows_out.extend(_infer_batch_to_rows(batch, model, class_names, model_name))

    return bulk_upsert_mppt_preds(plant=plant, rows=rows_out)


def _infer_batch_to_rows(batch, model, class_names, model_version):
    X = np.stack([b.X_ts for b in batch], axis=0).astype(np.float32)    # [B,N,T,F]
    E = np.stack([b.E_attr for b in batch], axis=0).astype(np.float32)  # [B,N,N,Fe]
    pred, pmax, proba = _predict_batch(model, X, E)

    out = []
    for bi, b in enumerate(batch):
        for mppt in range(1, pred.shape[1] + 1):
            code = int(pred[bi, mppt - 1])
            lbl = class_names[code] if 0 <= code < len(class_names) else "unknown"
            out.append(
                {
                    "source_oper": b.source_oper,
                    "mppt": mppt,
                    "ts_utc": b.ts_end_utc,
                    "model_version": model_version,
                    "pred_code": code,
                    "pred_label": lbl,
                    "pred_pmax": float(pmax[bi, mppt - 1]),
                    "proba": {class_names[c]: float(proba[bi, mppt - 1, c]) for c in range(len(class_names))},
                }
            )
    return out