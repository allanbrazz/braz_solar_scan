from __future__ import annotations

from typing import Optional

from .confidence import channel_confidence
from .contracts import ResidualConfig
from .gates import channel_gate
from .normalization import clip_or_none, safe_abs, safe_rel
from .types import ExpectedElectricalState, ResidualInputRow, ResidualValue


def build_residual_value(*, channel: str, observed: Optional[float], expected: Optional[float], row: ResidualInputRow, expected_state: ExpectedElectricalState, cfg: ResidualConfig) -> ResidualValue:
    gate_ok, status = channel_gate(channel, row, cfg)
    if not expected_state.model_valid or expected is None:
        gate_ok = False
        status = "model_unavailable"

    eps_map = {
        "p_ac": cfg.eps_power_w,
        "p_dc": cfg.eps_power_w,
        "v_dc": cfg.eps_voltage_v,
        "i_dc": cfg.eps_current_a,
    }
    clip_map = {
        "p_ac": cfg.pac_rel_clip,
        "p_dc": cfg.pdc_rel_clip,
        "v_dc": cfg.vdc_rel_clip,
        "i_dc": cfg.idc_rel_clip,
    }

    abs_res = safe_abs(observed, expected) if gate_ok else None
    rel_res = safe_rel(observed, expected, eps_map[channel]) if gate_ok else None
    norm_res = clip_or_none(rel_res, clip_map[channel]) if gate_ok else None
    conf = channel_confidence(row, has_expected=(expected is not None and expected_state.model_valid), channel=channel, cfg=cfg) if gate_ok else 0.0

    notes = list(expected_state.model_notes)
    if not gate_ok and status != "model_unavailable":
        notes.append(f"gate={status}")

    return ResidualValue(
        observed=observed,
        expected=expected,
        abs_residual=abs_res,
        rel_residual=rel_res,
        norm_residual=norm_res,
        valid=bool(gate_ok and norm_res is not None),
        status=status,
        confidence=float(conf),
        notes=notes,
    )
