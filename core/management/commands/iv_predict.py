from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from core.services.iv_fdd.ml.torch_cnn1d import load_torch_cnn, predict_torch_cnn
from core.services.iv_fdd.preprocess.features import load_features_npz


class Command(BaseCommand):
    help = "Run inference with a trained IV 1D-CNN on a features dataset (.npz)"

    def add_arguments(self, parser):
        parser.add_argument("--model", type=str, required=True, help="Path to model .pt")
        parser.add_argument("--input", type=str, required=True, help="Path to features .npz")
        parser.add_argument("--device", type=str, default="cpu")

    def handle(self, *args, **opts):
        model, class_names = load_torch_cnn(Path(opts["model"]))
        X, y, _ = load_features_npz(Path(opts["input"]))

        pred = predict_torch_cnn(model, X, device=str(opts["device"]))
        acc = float((pred == y).mean()) if y.size else 0.0

        self.stdout.write(self.style.SUCCESS(f"OK: accuracy={acc:.4f}"))
        # class-wise
        for i, name in enumerate(class_names):
            mask = (y == i)
            if mask.any():
                a = float((pred[mask] == y[mask]).mean())
                self.stdout.write(f"  - {name}: {a:.4f} (n={int(mask.sum())})")
