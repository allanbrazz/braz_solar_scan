from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone as dt_tz
from zoneinfo import ZoneInfo

from django.core.management.base import BaseCommand, CommandError

from core.models import PVPlant
from core.services.fdd.pipeline import run_detection_pipeline


class Command(BaseCommand):
    help = "Roda detector híbrido plant-level (tiers de irradiância + regras) e consolida FaultEvent."

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, required=True)
        parser.add_argument("--start", type=str, required=True, help="Data inicial local YYYY-MM-DD")
        parser.add_argument("--end", type=str, required=True, help="Data final local YYYY-MM-DD (inclusiva)")
        parser.add_argument("--source-oper", type=str, default="")
        parser.add_argument("--source-meteo", type=str, default="")
        parser.add_argument("--detector-version", type=str, default="hybrid_rules_v1")
        parser.add_argument("--delete-existing", type=int, default=1)

    def handle(self, *args, **opts):
        plant = PVPlant.objects.filter(id=int(opts["plant"])).first()
        if plant is None:
            raise CommandError("Plant not found")

        tz_name = getattr(plant, "timezone", None) or "UTC"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")

        try:
            d0 = datetime.fromisoformat(str(opts["start"])).date()
            d1 = datetime.fromisoformat(str(opts["end"])).date()
        except Exception as exc:
            raise CommandError(f"Datas inválidas: {exc}")

        if d0 > d1:
            d0, d1 = d1, d0

        ts_start_utc = datetime.combine(d0, time.min, tzinfo=tz).astimezone(dt_tz.utc)
        ts_end_utc = datetime.combine(d1 + timedelta(days=1), time.min, tzinfo=tz).astimezone(dt_tz.utc)

        out = run_detection_pipeline(
            plant_id=plant.id,
            ts_start_utc=ts_start_utc,
            ts_end_utc=ts_end_utc,
            source_oper=(str(opts["source_oper"]).strip() or None),
            source_meteo=(str(opts["source_meteo"]).strip() or None),
            detector_version=str(opts["detector_version"]),
            delete_existing=bool(int(opts["delete_existing"])),
        )
        self.stdout.write(json.dumps(out, ensure_ascii=False, indent=2, default=str))
