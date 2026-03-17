from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from core.services.mppt_gnn_fdd.event_infer_pipeline import infer_events_and_persist


class Command(BaseCommand):
    help = "Inferência MPPT/event-level rule-based alinhada ao pipeline híbrido plant-level."

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, default=None)
        parser.add_argument("--event-id", type=int, action="append", dest="event_ids")
        parser.add_argument("--status", type=str, action="append", dest="statuses")
        parser.add_argument("--model-version", type=str, default="event_rules_v2")
        parser.add_argument("--confidence-threshold", type=float, default=0.60)
        parser.add_argument("--replace-existing", type=int, default=1)

    def handle(self, *args, **opts):
        outs = infer_events_and_persist(
            plant_id=opts.get("plant"),
            event_ids=opts.get("event_ids"),
            statuses=opts.get("statuses") or ["open", "closed", "reviewed"],
            model_version=str(opts["model_version"]),
            confidence_threshold=float(opts["confidence_threshold"]),
            replace_existing=bool(int(opts["replace_existing"])),
        )
        self.stdout.write(json.dumps({"ok": True, "n": len(outs), "details": outs}, ensure_ascii=False, indent=2, default=str))
