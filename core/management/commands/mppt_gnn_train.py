# core/management/commands/mppt_gnn_train.py
from __future__ import annotations

from datetime import date
from django.core.management.base import BaseCommand

from core.services.mppt_gnn_fdd.train_pipeline import train_mppt_gnn_sklearn
from core.services.mppt_gnn_fdd.dataset import DatasetConfig
from core.services.mppt_gnn_fdd.model_sklearn import SklearnModelConfig


class Command(BaseCommand):
    help = "Treina classificador MPPT-GNN (baseline sklearn) usando fault injection."

    def add_arguments(self, parser):
        parser.add_argument("--plant", type=int, required=True)
        parser.add_argument("--start", type=str, required=True)  # YYYY-MM-DD
        parser.add_argument("--end", type=str, required=True)    # YYYY-MM-DD
        parser.add_argument("--model-version", type=str, default="gnn_v1")

        parser.add_argument("--n-days-max", type=int, default=120)
        parser.add_argument("--aug-per-day", type=int, default=6)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **opts):
        plant_id = int(opts["plant"])
        start = date.fromisoformat(opts["start"])
        end = date.fromisoformat(opts["end"])
        mv = str(opts["model_version"])

        ds_cfg = DatasetConfig(
            n_days_max=int(opts["n_days_max"]),
            aug_per_day=int(opts["aug_per_day"]),
            seed=int(opts["seed"]),
        )

        mlp_cfg = SklearnModelConfig()

        out = train_mppt_gnn_sklearn(
            plant_id=plant_id,
            start=start,
            end=end,
            model_version=mv,
            ds_cfg=ds_cfg,
            mlp_cfg=mlp_cfg,
        )
        self.stdout.write(str(out))