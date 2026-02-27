from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.services.mppt_gnn_fdd.config import TrainConfig
from core.services.mppt_gnn_fdd.model import MPPTGNNFDD
from core.services.mppt_gnn_fdd.windowing import iter_graph_windows_from_db
from core.services.mppt_gnn_fdd.labeling import IGNORE
from core.services.mppt_gnn_fdd.io.storage import model_path


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_from_db(
    *,
    plant_id: int,
    start,
    end,
    cfg: TrainConfig,
    out_name: str = "mppt_gnn_v1",
    source_oper: str | None = None,
    source_meteo: str | None = None,
) -> Dict[str, float]:
    _set_seed(cfg.seed)

    device = torch.device(cfg.device)
    class_names = list(cfg.class_names)
    C = len(class_names)

    # 1) contar classes (1 pass) -> pesos
    counts = np.zeros((C,), dtype=np.int64)
    n_samples = 0

    for w in iter_graph_windows_from_db(
        plant_id=plant_id, start=start, end=end, cfg=cfg,
        source_oper=source_oper, source_meteo=source_meteo,
        mode="train",
    ):
        y = w.y
        for c in range(C):
            counts[c] += int((y == c).sum())
        n_samples += 1

    # evita div/0
    counts = np.maximum(counts, 1)
    wts = counts.sum() / counts.astype(np.float32)
    class_w = torch.tensor(wts, dtype=torch.float32, device=device)

    # 2) modelo
    fin_ts = 5 if cfg.use_meteo else 3
    fe = 4
    model = MPPTGNNFDD(fin_ts=fin_ts, fe=fe, n_classes=C).to(device)

    crit = nn.CrossEntropyLoss(weight=class_w, ignore_index=IGNORE)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 3) treino (2º pass)
    best_loss = 1e9
    out = model_path(out_name, suffix=".pt")

    def _iter_batches():
        buf = []
        for w in iter_graph_windows_from_db(
            plant_id=plant_id, start=start, end=end, cfg=cfg,
            source_oper=source_oper, source_meteo=source_meteo,
            mode="train",
        ):
            buf.append(w)
            if len(buf) >= cfg.batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    for ep in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0

        for batch in _iter_batches():
            X = torch.from_numpy(np.stack([b.X_ts for b in batch], axis=0)).to(device)          # [B,N,T,F]
            E = torch.from_numpy(np.stack([b.E_attr for b in batch], axis=0)).to(device)        # [B,N,N,Fe]
            y = torch.from_numpy(np.stack([b.y for b in batch], axis=0)).to(device)             # [B,N]

            opt.zero_grad()
            logits = model(X, E)                                                                # [B,N,C]
            loss = crit(logits.reshape(-1, C), y.reshape(-1))
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())
            nb += 1

        ep_loss = ep_loss / max(nb, 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "cfg": asdict(cfg),
                    "model_version": out_name,
                },
                out,
            )

    return {
        "n_samples": float(n_samples),
        "count_normal": float(counts[0]),
        "count_disconnected": float(counts[1] if C > 1 else 0),
        "best_loss": float(best_loss),
        "model_path": float(0.0),  # placeholder (não serialize path em float)
    }