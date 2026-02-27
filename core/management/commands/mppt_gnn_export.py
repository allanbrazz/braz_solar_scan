# core/management/commands/mppt_gnn_export.py
from django.core.management.base import BaseCommand
from core.services.mppt_gnn_fdd.dataset_export import export_range_to_npz

class Command(BaseCommand):
    def add_arguments(self, p):
        p.add_argument("--plant", type=int, required=True)
        p.add_argument("--start", type=str, required=True)
        p.add_argument("--end", type=str, required=True)
        p.add_argument("--out", type=str, required=True)
    def handle(self, *args, **o):
        export_range_to_npz(plant_id=o["plant"], start=o["start"], end=o["end"], out=o["out"])
        self.stdout.write("ok")