from __future__ import annotations

from datetime import date
from django.core.management.base import BaseCommand
from core.services.merged15m_ingest import build_and_persist_merged15m_by_day

class Command(BaseCommand):
    help = "Build + persist merged 15min (pipeline 3b) day-by-day"

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, required=True)
        parser.add_argument("--start", type=str, required=True)  # YYYY-MM-DD
        parser.add_argument("--end", type=str, required=True)    # YYYY-MM-DD (inclusive)
        parser.add_argument("--source-oper", type=str, default="SHINEMONITOR")
        parser.add_argument("--source-meteo", type=str, default="OPENMETEO")
        parser.add_argument("--device-key", type=str, default=None)

    def handle(self, *args, **opts):
        plant_id = int(opts["plant"])
        start_day = date.fromisoformat(opts["start"])
        end_day = date.fromisoformat(opts["end"])

        res = build_and_persist_merged15m_by_day(
            plant_id=plant_id,
            start_day=start_day,
            end_day=end_day,
            source_oper=str(opts["source_oper"]),
            source_meteo=str(opts["source_meteo"]),
            device_key=opts["device_key"] or None,
        )
        self.stdout.write(str(res))
