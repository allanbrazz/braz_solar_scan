from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# =============================
# core/services/fdd_mismatch.py
# =============================

# Códigos (simples e estáveis)
CODE_INVALID = 0
CODE_NORMAL = 1
CODE_WARN = 2
CODE_FAULT = 3
CODE_METEO = 4


@dataclass(frozen=True)
class MismatchThresholds:
    """
    Limiarização simples de mismatch.
    mismatch_rel = (Pac_real - Pac_model)/max(|Pac_model|, eps_w)
    """
    gpoa_gate_wm2: float = 180.0
    warn_abs: float = 0.40
    fault_abs: float = 0.90
    meteo_pos_abs: float = 0.25
    shading_std_abs: float = 0.22
    shading_window_points: int = 6
    dt_minutes: float = 15.0
    max_gap_minutes: float = 30.0


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _dominant_label(labels: List[str]) -> str:
    if not labels:
        return "invalid"
    counts: Dict[str, int] = {}
    for s in labels:
        s = (s or "invalid").strip() or "invalid"
        counts[s] = counts.get(s, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _dominant_code(codes: List[int]) -> int:
    if not codes:
        return CODE_INVALID
    counts: Dict[int, int] = {}
    for c in codes:
        counts[int(c)] = counts.get(int(c), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _merge_events(events: List[Dict[str, Any]], *, max_gap_minutes: float) -> List[Dict[str, Any]]:
    if not events:
        return []
    out: List[Dict[str, Any]] = [events[0]]
    for e in events[1:]:
        last = out[-1]
        gap = (e["start_utc"] - last["end_utc"]).total_seconds() / 60.0
        if gap <= max_gap_minutes:
            last["end_utc"] = e["end_utc"]
            last["end_idx"] = e["end_idx"]
            last["n_points"] += e["n_points"]
            last["max_abs_mismatch"] = max(last["max_abs_mismatch"], e["max_abs_mismatch"])
            last["mean_abs_mismatch"] = (
                (last["mean_abs_mismatch"] * (last["n_points"] - e["n_points"]) + e["mean_abs_mismatch"] * e["n_points"])
                / max(last["n_points"], 1)
            )
            last["codes"].extend(e.get("codes", []))
            last["labels"].extend(e.get("labels", []))
        else:
            out.append(e)

    for e in out:
        e["dominant_label"] = _dominant_label(e.get("labels", []))
        e["dominant_code"] = _dominant_code(e.get("codes", []))
        e.pop("codes", None)
        e.pop("labels", None)
    return out


def classify_mismatch_series(
    *,
    times_utc: List[datetime],
    mismatch_rel: List[Optional[float]],
    g_poa_wm2: List[Optional[float]],
    valid: List[bool],
    thresholds: MismatchThresholds,
) -> Dict[str, Any]:
    """
    Retorna:
      - codes/labels por ponto
      - summary: contagens por label/code
      - events: intervalos de anomalia (warn/fault/meteo)
    """
    n = len(times_utc)
    if not (len(mismatch_rel) == len(g_poa_wm2) == len(valid) == n):
        raise ValueError("Listas com tamanhos diferentes (times/mismatch/g_poa/valid).")

    codes: List[int] = [CODE_INVALID] * n
    labels: List[str] = ["invalid"] * n

    for i in range(n):
        v_ok = bool(valid[i])
        g = _safe_float(g_poa_wm2[i])
        m = _safe_float(mismatch_rel[i])

        if (not v_ok) or (g is None) or (g < float(thresholds.gpoa_gate_wm2)) or (m is None):
            codes[i] = CODE_INVALID
            labels[i] = "invalid"
            continue

        a = abs(m)
        if a < float(thresholds.warn_abs):
            codes[i] = CODE_NORMAL
            labels[i] = "normal"
            continue

        # positivo: viés meteo / irradiância subestimada / etc
        if m >= float(thresholds.fault_abs) or m >= float(thresholds.meteo_pos_abs):
            codes[i] = CODE_METEO
            labels[i] = "meteo_error"
            continue

        # negativo forte: perda severa (string desconectada / trip / etc)
        if m <= -float(thresholds.fault_abs):
            codes[i] = CODE_FAULT
            labels[i] = "string_disconnected"
            continue

        # perda moderada (proxy para soiling/degradação etc)
        codes[i] = CODE_WARN
        labels[i] = "degradation"

    counts_by_label: Dict[str, int] = {}
    counts_by_code: Dict[str, int] = {}
    for c, lab in zip(codes, labels):
        counts_by_label[lab] = counts_by_label.get(lab, 0) + 1
        counts_by_code[str(int(c))] = counts_by_code.get(str(int(c)), 0) + 1

    events_raw: List[Dict[str, Any]] = []
    in_evt = False
    cur: Dict[str, Any] = {}
    cur_codes: List[int] = []
    cur_labels: List[str] = []
    cur_abs: List[float] = []

    for i in range(n):
        abnormal = codes[i] not in (CODE_INVALID, CODE_NORMAL)
        if abnormal and not in_evt:
            in_evt = True
            cur = {
                "start_utc": times_utc[i],
                "start_idx": i,
                "end_utc": times_utc[i],
                "end_idx": i,
                "n_points": 0,
                "max_abs_mismatch": 0.0,
                "mean_abs_mismatch": 0.0,
                "codes": [],
                "labels": [],
            }
            cur_codes, cur_labels, cur_abs = [], [], []

        if abnormal and in_evt:
            cur["end_utc"] = times_utc[i]
            cur["end_idx"] = i
            cur["n_points"] += 1
            m = _safe_float(mismatch_rel[i])
            if m is not None:
                cur_abs.append(abs(m))
            cur_codes.append(int(codes[i]))
            cur_labels.append(labels[i])

        if (not abnormal) and in_evt:
            if cur_abs:
                cur["max_abs_mismatch"] = max(cur_abs)
                cur["mean_abs_mismatch"] = sum(cur_abs) / len(cur_abs)
            cur["codes"] = cur_codes
            cur["labels"] = cur_labels
            events_raw.append(cur)
            in_evt = False

    if in_evt:
        if cur_abs:
            cur["max_abs_mismatch"] = max(cur_abs)
            cur["mean_abs_mismatch"] = sum(cur_abs) / len(cur_abs)
        cur["codes"] = cur_codes
        cur["labels"] = cur_labels
        events_raw.append(cur)

    events = _merge_events(events_raw, max_gap_minutes=float(thresholds.max_gap_minutes))

    events_out = []
    for e in events:
        events_out.append(
            {
                "start_utc": e["start_utc"],
                "end_utc": e["end_utc"],
                "start_idx": int(e["start_idx"]),
                "end_idx": int(e["end_idx"]),
                "n_points": int(e["n_points"]),
                "dominant_code": int(e["dominant_code"]),
                "dominant_label": str(e["dominant_label"]),
                "max_abs_mismatch": float(e["max_abs_mismatch"]),
                "mean_abs_mismatch": float(e["mean_abs_mismatch"]),
            }
        )

    return {
        "codes": codes,
        "labels": labels,
        "summary": {
            "counts_by_label": counts_by_label,
            "counts_by_code": counts_by_code,
        },
        "events": events_out,
        "thresholds": asdict(thresholds),
    }
