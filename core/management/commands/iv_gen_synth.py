from __future__ import annotations

from django.core.management.base import BaseCommand

from core.models import PVModule
from core.services.power_model.power_model import module_from_pvmodule
from core.services.iv_fdd.io.storage import dataset_path
from core.services.iv_fdd.sim.iv_generator import SynthConfig, generate_synth_dataset, save_dataset_npz


class Command(BaseCommand):
    help = "Generate Hopwood-style fully synthetic IV dataset (.npz) using your 1-diode model baseline."

    def add_arguments(self, parser):
        parser.add_argument("--module-id", type=int, required=True)
        parser.add_argument("--n-samples", type=int, default=20000)
        parser.add_argument("--n-points", type=int, default=200)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--out", type=str, default="hopwood_synth_v1")

    def handle(self, *args, **opts):
        module_id = int(opts["module_id"])
        pvmod = PVModule.objects.get(id=module_id)
        module = module_from_pvmodule(pvmod)

        cfg = SynthConfig(
            n_samples=int(opts["n_samples"]),
            n_points=int(opts["n_points"]),
            seed=int(opts["seed"]),
        )

        ds = generate_synth_dataset(module, cfg=cfg)
        out = dataset_path(str(opts["out"]), suffix=".npz")
        save_dataset_npz(ds, out)
        self.stdout.write(self.style.SUCCESS(f"OK: dataset saved -> {out}"))
