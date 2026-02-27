# core/services/mppt_gnn_fdd/infer_pipeline.py
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone as dt_tz
from typing import Any, Dict, Optional, List

import numpy as np
from django.utils import timezone

from core.models import PVPlant
from core.services.mppt_gnn_fdd.window_loader import load_daily_window
from core.services.mppt_gnn_fdd.features import build_node_features
from core.services.mppt_gnn_fdd.storage import load_artifacts
from core.services.mppt_gnn_fdd.constants import FAULT_LABEL_BY_CODE, N_MPPT_DEFAULT


# opcional (para não quebrar se model ainda não existe no repo)
try:
    from core.models import MPPTDiagnostic15m  # type: ignore
except Exception:  # pragma: no cover
    MPPTDiagnostic15m = None  # type: ignore


def _flatten_node_sequence(X_node: np.ndarray) -> np.ndarray:
    N, T, F = X_node.shape
    return X_node.reshape(N, T * F).astype(np.float32)


def infer_day_and_persist(
    *,
    plant_id: int,
    day_local: date,
    model_version: str,
    source_oper: str = "MPPT_GNN",
    n_mppt: int = N_MPPT_DEFAULT,
    delete_existing: bool = True,
) -> Dict[str, Any]:
    if MPPTDiagnostic15m is None:
        raise RuntimeError("MPPTDiagnostic15m não disponível (migrate pendente).")

    plant = PVPlant.objects.filter(id=plant_id).first()
    if plant is None:
        raise ValueError("Plant not found")

    model, scaler, meta, _art = load_artifacts(model_version=model_version)

    win, ts_grid, _meta = load_daily_window(plant_id=plant_id, day_local=day_local, n_mppt=n_mppt)
    X_node, _fmap = build_node_features(win, scaler)
    X = _flatten_node_sequence(X_node)  # [N, D]

    proba = model.predict_proba(X)  # [N, C]
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("Modelo sem classes_")

    pred_idx = np.argmax(proba, axis=1)
    pred_code = np.asarray(classes, dtype=int)[pred_idx]  # [N]
    pred_pmax = np.max(proba, axis=1)

    now = timezone.now()

    # deletar intervalo do dia (qualquer mppt) para este model_version+source_oper
    if delete_existing:
        dt0_utc = ts_grid[0].astimezone(dt_tz.utc)
        dt1_utc = (ts_grid[-1] + timedelta(minutes=15)).astimezone(dt_tz.utc)
        MPPTDiagnostic15m.objects.filter(
            plant_id=plant_id,
            ts_utc__gte=dt0_utc,
            ts_utc__lt=dt1_utc,
            model_version=model_version,
            source_oper=source_oper,
        ).delete()

    # persiste por timestep (repete a classe do dia por bin)
    bulk: List[Any] = []
    for j, tsu in enumerate(ts_grid):
        for k in range(1, n_mppt + 1):
            code = int(pred_code[k - 1])
            label = FAULT_LABEL_BY_CODE.get(code, "fault")
            pmax = float(pred_pmax[k - 1])

            # proba dict (serializável)
            prob_dict = {str(int(c)): float(proba[k - 1, ii]) for ii, c in enumerate(classes)}

            bulk.append(
                MPPTDiagnostic15m(
                    plant_id=plant_id,
                    source_oper=source_oper,
                    mppt=int(k),
                    ts_utc=tsu.astimezone(dt_tz.utc),
                    model_version=model_version,
                    pred_code=code,
                    pred_label=label,
                    pred_pmax=pmax,
                    proba=prob_dict,
                    created_at=now,
                    updated_at=now,
                )
            )

    MPPTDiagnostic15m.objects.bulk_create(bulk, ignore_conflicts=True)

    return {
        "ok": True,
        "plant_id": plant_id,
        "day_local": day_local.isoformat(),
        "model_version": model_version,
        "source_oper": source_oper,
        "written": len(bulk),
        "pred_by_mppt": {f"mppt{k}": FAULT_LABEL_BY_CODE.get(int(pred_code[k-1]), "fault") for k in range(1, n_mppt+1)},
    }