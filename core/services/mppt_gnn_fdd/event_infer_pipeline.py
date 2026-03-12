from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from django.db import transaction

from core.models import FaultEvent, FaultEventMPPT
from core.services.mppt_gnn_fdd.event_features import build_event_mppt_features
from core.services.mppt_gnn_fdd.event_loader import load_event_window


RULE_CODE_BY_LABEL = {
    "normal": 0,
    "mppt_disconnected": 1,
    "inverter_off_under_sun": 2,
    "mppt_imbalance": 3,
    "curtailment_clipping": 4,
    "meteo_bias": 5,
    "unknown_fault": 99,
}


def _mk_result(label: str, confidence: float, extra: Optional[dict] = None) -> dict:
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "pred_label": label,
        "pred_code": int(RULE_CODE_BY_LABEL.get(label, 99)),
        "confidence": confidence,
        "novelty_score": 1.0 - confidence,
        "contribution": extra or {},
        "proba": {label: confidence, "unknown_fault": max(0.0, 1.0 - confidence)},
    }


def _rule_predict_one(mppt_feat: dict, plant_summary: dict, prelim_label: str, confidence_threshold: float) -> dict:
    if prelim_label == "curtailment_clipping" or plant_summary.get("clip_frac", 0.0) >= 0.50:
        res = _mk_result("curtailment_clipping", 0.90, {"reason": "clip_frac_high"})
    elif prelim_label == "meteo_bias" and plant_summary.get("mismatch_mean", 0.0) >= 0.20:
        res = _mk_result("meteo_bias", 0.80, {"reason": "positive_residual_bias"})
    elif plant_summary.get("zero_power_frac", 0.0) >= 0.80 and plant_summary.get("g_mean", 0.0) >= 300.0:
        res = _mk_result("inverter_off_under_sun", 0.85, {"reason": "zero_power_under_sun"})
    elif mppt_feat.get("outage_frac", 0.0) >= 0.60 and mppt_feat.get("i_rel_med", 1.0) <= 0.20:
        conf = min(0.95, 0.70 + 0.25 * mppt_feat.get("outage_frac", 0.0))
        res = _mk_result("mppt_disconnected", conf, {"reason": "current_outage_pattern"})
    elif (
        mppt_feat.get("low_i_frac", 0.0) >= 0.60
        and mppt_feat.get("share_low_frac", 0.0) >= 0.50
        and mppt_feat.get("i_rel_med", 1.0) <= 0.65
    ):
        res = _mk_result("mppt_imbalance", 0.76, {"reason": "low_current_vs_peers"})
    elif prelim_label == "meteo_bias":
        res = _mk_result("meteo_bias", 0.65, {"reason": "event_prelabel"})
    else:
        res = _mk_result("normal", 0.55, {"reason": "no_strong_rule"})

    if res["confidence"] < float(confidence_threshold):
        return _mk_result("unknown_fault", res["confidence"], {**res.get("contribution", {}), "open_set": True})
    return res


def _resolve_event_final_label(pred_rows: List[dict], event_prelim: str, plant_summary: dict) -> tuple[str, str, float, float]:
    known = [r for r in pred_rows if r["pred_label"] not in {"normal", "unknown_fault"}]
    if known:
        agg: dict[str, float] = {}
        for r in known:
            agg[r["pred_label"]] = agg.get(r["pred_label"], 0.0) + float(r["confidence"])
        final_label = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        confidence = float(sum(r["confidence"] for r in known) / max(len(known), 1))
        novelty = float(sum(r["novelty_score"] for r in known) / max(len(known), 1))
        return final_label, "known", confidence, novelty

    if event_prelim in {"curtailment_clipping", "meteo_bias", "plant_wide_loss"}:
        mapped = event_prelim
        if mapped == "plant_wide_loss" and plant_summary.get("zero_power_frac", 0.0) >= 0.80:
            mapped = "inverter_off_under_sun"
        return mapped, "known", 0.65, 0.35

    unknowns = [r for r in pred_rows if r["pred_label"] == "unknown_fault"]
    if unknowns:
        confidence = float(sum(r["confidence"] for r in unknowns) / len(unknowns))
        novelty = float(sum(r["novelty_score"] for r in unknowns) / len(unknowns))
        return "unknown_fault", "unknown", confidence, novelty

    return "normal", "known", 0.55, 0.45


def infer_event_and_persist(
    *,
    event_id: int,
    model_version: str = "event_rules_v1",
    confidence_threshold: float = 0.60,
    pre_bins: int = 8,
    post_bins: int = 8,
    n_mppt: int = 4,
    replace_existing: bool = True,
) -> dict:
    event = FaultEvent.objects.filter(id=event_id).first()
    if event is None:
        raise ValueError("FaultEvent não encontrado")

    win, ts_grid, meta = load_event_window(
        event_id=event_id,
        pre_bins=pre_bins,
        post_bins=post_bins,
        n_mppt=n_mppt,
    )
    mppt_feats, plant_summary = build_event_mppt_features(
        win=win,
        ts_grid=ts_grid,
        event_start_utc=meta["event_start_utc"],
        event_end_utc=meta["event_end_utc"],
    )

    pred_rows: List[dict[str, Any]] = []
    for feat in mppt_feats:
        pred = _rule_predict_one(feat, plant_summary, str(event.event_label_prelim or "unknown"), confidence_threshold)
        pred_rows.append(
            {
                "mppt": int(feat["mppt"]),
                "source_oper": meta["source_oper"],
                "model_version": model_version,
                **pred,
                "feature_snapshot": feat,
            }
        )

    final_label, known_vs_unknown, confidence, novelty = _resolve_event_final_label(
        pred_rows,
        str(event.event_label_prelim or "unknown"),
        plant_summary,
    )

    with transaction.atomic():
        if replace_existing:
            FaultEventMPPT.objects.filter(event_id=event_id, model_version=model_version).delete()

        objs = []
        for r in pred_rows:
            contribution = dict(r.get("contribution") or {})
            contribution["features"] = r["feature_snapshot"]
            objs.append(
                FaultEventMPPT(
                    event_id=event_id,
                    source_oper=r["source_oper"],
                    mppt=r["mppt"],
                    model_version=model_version,
                    pred_code=r["pred_code"],
                    pred_label=r["pred_label"],
                    confidence=r["confidence"],
                    novelty_score=r["novelty_score"],
                    contribution=contribution,
                    proba=r["proba"],
                )
            )
        FaultEventMPPT.objects.bulk_create(objs, batch_size=200)

        event.final_label = final_label
        event.known_vs_unknown = known_vs_unknown
        event.confidence = confidence
        event.novelty_score = novelty
        event.meta = {**(event.meta or {}), "plant_summary": plant_summary, "event_model_version": model_version}
        event.save(update_fields=["final_label", "known_vs_unknown", "confidence", "novelty_score", "meta", "updated_at"])

    return {
        "ok": True,
        "event_id": event_id,
        "plant_id": event.plant_id,
        "final_label": final_label,
        "known_vs_unknown": known_vs_unknown,
        "confidence": confidence,
        "novelty_score": novelty,
        "mppt_predictions": [
            {
                "mppt": r["mppt"],
                "pred_label": r["pred_label"],
                "confidence": r["confidence"],
            }
            for r in pred_rows
        ],
    }


def infer_events_and_persist(
    *,
    plant_id: Optional[int] = None,
    event_ids: Optional[Iterable[int]] = None,
    statuses: Optional[Iterable[str]] = None,
    model_version: str = "event_rules_v1",
    confidence_threshold: float = 0.60,
    replace_existing: bool = True,
) -> List[dict]:
    qs = FaultEvent.objects.all().order_by("plant_id", "ts_start_utc")
    if plant_id is not None:
        qs = qs.filter(plant_id=plant_id)
    if event_ids is not None:
        qs = qs.filter(id__in=list(event_ids))
    if statuses:
        qs = qs.filter(status__in=list(statuses))

    outs: List[dict] = []
    for ev in qs:
        outs.append(
            infer_event_and_persist(
                event_id=ev.id,
                model_version=model_version,
                confidence_threshold=confidence_threshold,
                replace_existing=replace_existing,
            )
        )
    return outs