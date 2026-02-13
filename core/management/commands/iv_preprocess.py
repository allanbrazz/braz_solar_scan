from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from core.services.iv_fdd.io.storage import dataset_path
from core.services.iv_fdd.sim.iv_generator import load_dataset_npz
from core.services.iv_fdd.preprocess.features import build_input_channels, save_features_npz


class Command(BaseCommand):
    help = "Preprocess synthetic IV dataset into CNN channels (X,y) and save as .npz"

    def add_arguments(self, parser):
        parser.add_argument("--input", type=str, required=True, help="Path to dataset .npz")
        parser.add_argument("--out", type=str, default="hopwood_synth_v1_features")

    def handle(self, *args, **opts):
        inp = Path(opts["input"])
        ds = load_dataset_npz(inp)

        X = build_input_channels(ds["V"], ds["I"])
        y = ds["y"]
        class_names = ds["class_names"]

        out = dataset_path(str(opts["out"]), suffix=".npz")
        save_features_npz(X, y, class_names, out)
        self.stdout.write(self.style.SUCCESS(f"OK: features saved -> {out}"))
