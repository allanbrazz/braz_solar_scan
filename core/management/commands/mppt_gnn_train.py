from __future__ import annotations

from datetime import date
import json
from typing import Tuple

from django.core.management.base import BaseCommand, CommandError

from core.services.mppt_gnn_fdd.train_pipeline import train_mppt_gnn_sklearn
from core.services.mppt_gnn_fdd.dataset import DatasetConfig
from core.services.mppt_gnn_fdd.model_sklearn import SklearnModelConfig


def _parse_hidden_layers(raw: str) -> Tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        raise CommandError("--hidden-layers não pode estar vazio. Ex.: 256,128")
    try:
        vals = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    except Exception as exc:  # pragma: no cover
        raise CommandError(f"--hidden-layers inválido: {raw!r}") from exc
    if not vals or any(v <= 0 for v in vals):
        raise CommandError("--hidden-layers deve conter inteiros positivos. Ex.: 256,128")
    return vals


class Command(BaseCommand):
    help = (
        "Treina classificador MPPT-GNN (baseline sklearn) usando fault injection, "
        "com holdout temporal por dia e persistência completa de meta.json."
    )

    def add_arguments(self, parser):
        # escopo
        parser.add_argument("--plant", type=int, required=True)
        parser.add_argument("--start", type=str, required=True)  # YYYY-MM-DD
        parser.add_argument("--end", type=str, required=True)    # YYYY-MM-DD
        parser.add_argument("--model-version", type=str, default="gnn_v1")

        # dataset / seleção de dias
        parser.add_argument("--n-days-max", type=int, default=180)
        parser.add_argument("--base-days-target", type=int, default=60)
        parser.add_argument("--aug-per-day", type=int, default=6)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--n-mppt", type=int, default=4)

        parser.add_argument("--g-day-gate", type=float, default=120.0)
        parser.add_argument("--pac-gate-w", type=float, default=80.0)
        parser.add_argument("--min-day-points", type=int, default=12)
        parser.add_argument("--pac-model-gate", type=float, default=80.0)
        parser.add_argument("--mm-abs-p90-max-for-normal", type=float, default=0.50)

        parser.add_argument("--keep-most-recent-days", dest="keep_most_recent_days", action="store_true", default=True)
        parser.add_argument("--keep-oldest-days", dest="keep_most_recent_days", action="store_false")

        # mistura de faults sintéticos
        parser.add_argument("--p-disc", type=float, default=0.25)
        parser.add_argument("--p-off", type=float, default=0.20)
        parser.add_argument("--p-imb", type=float, default=0.25)
        parser.add_argument("--p-curt", type=float, default=0.20)
        parser.add_argument("--p-meteo", type=float, default=0.10)

        # MLP / treino
        parser.add_argument("--hidden-layers", type=str, default="256,128")
        parser.add_argument("--alpha", type=float, default=1e-4)
        parser.add_argument("--max-iter", type=int, default=60)
        parser.add_argument("--validation-fraction", type=float, default=0.15)
        parser.add_argument("--n-iter-no-change", type=int, default=8)

        parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=True)
        parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")

        parser.add_argument("--temporal-validation-fraction", type=float, default=0.20)
        parser.add_argument("--min-temporal-validation-days", type=int, default=7)

        parser.add_argument("--balanced-sample-weight", dest="use_balanced_sample_weight", action="store_true", default=True)
        parser.add_argument("--no-balanced-sample-weight", dest="use_balanced_sample_weight", action="store_false")

    def handle(self, *args, **opts):
        try:
            plant_id = int(opts["plant"])
            start = date.fromisoformat(str(opts["start"]))
            end = date.fromisoformat(str(opts["end"]))
            model_version = str(opts["model_version"]).strip()
            if not model_version:
                raise CommandError("--model-version não pode estar vazio.")

            hidden_layers = _parse_hidden_layers(str(opts["hidden_layers"]))

            prob_sum = (
                float(opts["p_disc"]) + float(opts["p_off"]) + float(opts["p_imb"]) +
                float(opts["p_curt"]) + float(opts["p_meteo"])
            )
            if prob_sum <= 0:
                raise CommandError("A soma das probabilidades de fault injection deve ser > 0.")

            ds_cfg = DatasetConfig(
                n_mppt=int(opts["n_mppt"]),
                g_day_gate=float(opts["g_day_gate"]),
                pac_gate_w=float(opts["pac_gate_w"]),
                min_day_points=int(opts["min_day_points"]),
                pac_model_gate=float(opts["pac_model_gate"]),
                mm_abs_p90_max_for_normal=float(opts["mm_abs_p90_max_for_normal"]),
                n_days_max=int(opts["n_days_max"]),
                base_days_target=int(opts["base_days_target"]),
                aug_per_day=int(opts["aug_per_day"]),
                seed=int(opts["seed"]),
                keep_most_recent_days=bool(opts["keep_most_recent_days"]),
                p_disc=float(opts["p_disc"]),
                p_off=float(opts["p_off"]),
                p_imb=float(opts["p_imb"]),
                p_curt=float(opts["p_curt"]),
                p_meteo=float(opts["p_meteo"]),
            )

            mlp_cfg = SklearnModelConfig(
                hidden_layer_sizes=hidden_layers,
                alpha=float(opts["alpha"]),
                max_iter=int(opts["max_iter"]),
                early_stopping=bool(opts["early_stopping"]),
                validation_fraction=float(opts["validation_fraction"]),
                n_iter_no_change=int(opts["n_iter_no_change"]),
                random_state=int(opts["seed"]),
                temporal_validation_fraction=float(opts["temporal_validation_fraction"]),
                min_temporal_validation_days=int(opts["min_temporal_validation_days"]),
                use_balanced_sample_weight=bool(opts["use_balanced_sample_weight"]),
            )

            out = train_mppt_gnn_sklearn(
                plant_id=plant_id,
                start=start,
                end=end,
                model_version=model_version,
                ds_cfg=ds_cfg,
                mlp_cfg=mlp_cfg,
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps({
            "ok": True,
            "request": {
                "plant_id": plant_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "model_version": model_version,
                "dataset": ds_cfg.__dict__,
                "mlp": {
                    **mlp_cfg.__dict__,
                    "hidden_layer_sizes": list(mlp_cfg.hidden_layer_sizes),
                },
            },
            "result": out,
        }, ensure_ascii=False, indent=2))
