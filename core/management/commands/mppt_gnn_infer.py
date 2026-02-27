# core/management/commands/mppt_gnn_infer.py
from __future__ import annotations

from datetime import date, timedelta
from django.core.management.base import BaseCommand

from core.services.mppt_gnn_fdd.infer_pipeline import infer_day_and_persist


class Command(BaseCommand):
    help = "Inferência MPPT-GNN (baseline sklearn) por dia e grava em MPPTDiagnostic15m."

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, required=True)
        parser.add_argument("--start", type=str, required=True)
        parser.add_argument("--end", type=str, required=True)
        parser.add_argument("--model-version", type=str, default="gnn_v1")
        parser.add_argument("--source-oper", type=str, default="MPPT_GNN")
        parser.add_argument("--delete-existing", type=int, default=1)

    def handle(self, *args, **opts):
        plant_id = int(opts["plant"])
        d0 = date.fromisoformat(opts["start"])
        d1 = date.fromisoformat(opts["end"])
        mv = str(opts["model_version"])
        src = str(opts["source_oper"])
        delete_existing = bool(int(opts["delete_existing"]))

        if d0 > d1:
            d0, d1 = d1, d0

        cur = d0
        outs = []
        while cur <= d1:
            out = infer_day_and_persist(
                plant_id=plant_id,
                day_local=cur,
                model_version=mv,
                source_oper=src,
                delete_existing=delete_existing,
            )
            outs.append(out)
            cur = cur + timedelta(days=1)

        self.stdout.write(str({"ok": True, "days": len(outs), "details": outs[:3]}))