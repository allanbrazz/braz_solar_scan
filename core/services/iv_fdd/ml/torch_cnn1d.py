from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch não está instalado. Use: pip install torch") from e


class Cnn1D(nn.Module):
    """Simple 1D-CNN for IV-curve channels.

    Input expected:
      - X in channels-first format: (B, C, L)
      - C=4 channels (iota, P, dI, dIdV)
      - L = number of resampled points (e.g., 200)

    Architecture:
      conv1d -> relu -> maxpool -> conv1d -> relu -> maxpool -> flatten -> dense -> logits
    """
    def __init__(self, n_classes: int, in_channels: int = 4, kernel_size: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 50, 128),  # assumes L=200 -> after 2 pools -> 50
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    kernel_size: int = 12
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_split: float = 0.2


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_channels_first(X: np.ndarray) -> np.ndarray:
    # X: (N,L,C) -> (N,C,L)
    if X.ndim != 3:
        raise ValueError("X must be (N,L,C)")
    return np.transpose(X, (0, 2, 1))


def train_torch_cnn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_names: list[str],
    cfg: TrainConfig | None = None,
    out_path: Path | str,
) -> Dict[str, float]:
    cfg = cfg or TrainConfig()
    _set_seed(cfg.seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    X = _to_channels_first(X)

    N = X.shape[0]
    n_val = int(N * cfg.val_split)
    idx = np.random.permutation(N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    device = torch.device(cfg.device)
    model = Cnn1D(n_classes=len(class_names), kernel_size=cfg.kernel_size).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def batches(Xb: np.ndarray, yb: np.ndarray, bs: int):
        n = Xb.shape[0]
        order = np.random.permutation(n)
        for i in range(0, n, bs):
            j = order[i : i + bs]
            yield Xb[j], yb[j]

    best_val = -1.0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in batches(Xtr, ytr, cfg.batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            opt.zero_grad()
            logits = model(xb_t)
            loss = crit(logits, yb_t)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            xb_t = torch.from_numpy(Xva).to(device)
            logits = model(xb_t)
            pred = logits.argmax(dim=1).cpu().numpy()
        val_acc = float((pred == yva).mean()) if yva.size else 0.0

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "kernel_size": cfg.kernel_size,
                },
                out_path,
            )

    return {"val_acc_best": float(best_val)}


def load_torch_cnn(path: Path | str) -> Tuple[torch.nn.Module, list[str]]:
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    class_names = list(ckpt["class_names"])
    kernel_size = int(ckpt.get("kernel_size", 12))
    model = Cnn1D(n_classes=len(class_names), kernel_size=kernel_size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names


def predict_torch_cnn(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = _to_channels_first(X)
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device_t)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
    return pred
