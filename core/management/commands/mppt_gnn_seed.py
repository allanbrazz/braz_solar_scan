from __future__ import annotations

from datetime import date
from django.core.management.base import BaseCommand

from core.services.mppt_gnn_fdd.seed import seed_mppt_predictions, SeedConfig


class Command(BaseCommand):
    help = "Seed MPPTDiagnostic15m with baseline heuristic predictions (normal/disconnected) from merged data."

    def add_arguments(self, p):
        p.add_argument("--plant", type=int, required=True)
        p.add_argument("--start", type=str, required=True)  # YYYY-MM-DD
        p.add_argument("--end", type=str, required=True)    # YYYY-MM-DD
        p.add_argument("--source-meteo", type=str, default="")
        p.add_argument("--source-oper", type=str, default="")
        p.add_argument("--model-version", type=str, default="seed_v0")

        p.add_argument("--daylight-min", type=float, default=50.0)
        p.add_argument("--disc-gate", type=float, default=300.0)
        p.add_argument("--i-zero", type=float, default=0.10)
        p.add_argument("--i-peer", type=float, default=0.50)

    def handle(self, *args, **o):
        cfg = SeedConfig(
            model_version=str(o["model_version"]),
            daylight_min_wm2=float(o["daylight_min"]),
            disc_gate_wm2=float(o["disc_gate"]),
            i_zero_a=float(o["i_zero"]),
            i_peer_a=float(o["i_peer"]),
        )

        res = seed_mppt_predictions(
            plant_id=int(o["plant"]),
            start=date.fromisoformat(o["start"]),
            end=date.fromisoformat(o["end"]),
            cfg=cfg,
            source_meteo=(str(o["source_meteo"]).strip() or None),
            source_oper=(str(o["source_oper"]).strip() or None),
        )
        self.stdout.write(self.style.SUCCESS(str(res)))