from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from core.services.iv_fdd.io.storage import model_path
from core.services.iv_fdd.preprocess.features import load_features_npz
from core.services.iv_fdd.ml.torch_cnn1d import TrainConfig, train_torch_cnn


class Command(BaseCommand):
    help = "Train a simple 1D-CNN (PyTorch) for IV failure classification from features dataset (.npz)"

    def add_arguments(self, parser):
        parser.add_argument("--input", type=str, required=True, help="Path to features .npz")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--batch", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--out", type=str, default="hopwood_cnn_v1")

    def handle(self, *args, **opts):
        X, y, class_names = load_features_npz(Path(opts["input"]))

        cfg = TrainConfig(
            epochs=int(opts["epochs"]),
            batch_size=int(opts["batch"]),
            lr=float(opts["lr"]),
        )

        out = model_path(str(opts["out"]), suffix=".pt")
        metrics = train_torch_cnn(X, y, class_names=class_names, cfg=cfg, out_path=out)
        self.stdout.write(self.style.SUCCESS(f"OK: model saved -> {out} | metrics={metrics}"))
