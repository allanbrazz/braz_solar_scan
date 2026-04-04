from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone as dt_tz
from zoneinfo import ZoneInfo

from django.core.management.base import BaseCommand, CommandError

from core.models import PVPlant
from core.services.fdd.dashboard_common import MISMATCH_VERSION_SUMMARY
from core.services.fdd.dashboard_runtime import parse_dashboard_params
from core.services.fdd.validation import compute_validation_report_from_db


class Command(BaseCommand):
    help = "Gera relatório mínimo de validação do FDD mismatch usando GroundTruthEvent + FaultEvent/PlantDiagnostic15m."

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, required=True, dest="plant_id")
        parser.add_argument("--start", required=True, help="YYYY-MM-DD")
        parser.add_argument("--end", required=True, help="YYYY-MM-DD")
        parser.add_argument("--detector-version", default=str(MISMATCH_VERSION_SUMMARY.get("detector_version") or "mismatch_runtime_v1"))
        parser.add_argument("--source-oper", default="")
        parser.add_argument("--source-meteo", default="")
        parser.add_argument("--output-json", default="")

    def handle(self, *args, **opts):
        plant = PVPlant.objects.filter(id=opts["plant_id"]).first()
        if plant is None:
            raise CommandError("Planta não encontrada.")

        tz_name = getattr(plant, "timezone", "UTC") or "UTC"
        params = parse_dashboard_params({"start": opts["start"], "end": opts["end"]}, tz_name)
        report = compute_validation_report_from_db(
            plant_id=plant.id,
            ts_start_utc=params.dt0_utc,
            ts_end_utc=params.dt1_utc,
            detector_version=str(opts.get("detector_version") or "").strip(),
            source_oper=str(opts.get("source_oper") or "").strip(),
            source_meteo=str(opts.get("source_meteo") or "").strip(),
        )

        payload = {
            "ok": True,
            "plant": {"id": plant.id, "nome": getattr(plant, "nome", str(plant.id)), "tz": tz_name},
            "range": {
                "start": opts["start"],
                "end": opts["end"],
                "start_utc": params.dt0_utc.isoformat(),
                "end_utc_excl": params.dt1_utc.isoformat(),
            },
            "filters": {
                "detector_version": opts.get("detector_version"),
                "source_oper": opts.get("source_oper"),
                "source_meteo": opts.get("source_meteo"),
            },
            "validation": report,
        }

        out_path = str(opts.get("output_json") or "").strip()
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self.stdout.write(self.style.SUCCESS(f"Relatório salvo em {out_path}"))
        else:
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
