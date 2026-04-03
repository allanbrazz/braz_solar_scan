from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    from django.conf import settings
except Exception:  # pragma: no cover
    settings = None  # type: ignore

STATE_NONE = 0
STATE_OK = 1
STATE_WARN = 2
STATE_CRIT = 3

STATE_LABELS = {
    STATE_NONE: "Sem diag. útil",
    STATE_OK: "Normal",
    STATE_WARN: "Warn",
    STATE_CRIT: "Falha",
}

STATE_COLORS = {
    STATE_NONE: "#94A3B8",
    STATE_OK: "#46E68C",
    STATE_WARN: "#FFCF5A",
    STATE_CRIT: "#FF5A6E",
}

SITE_COLORS = {
    "navy": "#0F2746",
    "blue": "#17395E",
    "cyan": "#5FB0FF",
    "text": "#1E2D3B",
    "muted": "#425466",
    "line": "#D6E1EE",
    "head": "#EAF2FB",
    "band": "#F8FBFF",
    "warn_bg": "#FFF6D8",
    "danger_bg": "#FFE3E8",
    "ok_bg": "#E8FBF2",
}


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _safe_text(v: Any, default: str = "-") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        return x if isfinite(x) else default
    except Exception:
        return default


def _fmt_float(v: Any, digits: int = 2, default: str = "-") -> str:
    x = _safe_float(v, None)
    return f"{x:.{digits}f}" if x is not None else default


def _fmt_pct(v: Any, digits: int = 1, default: str = "-") -> str:
    x = _safe_float(v, None)
    return f"{100.0 * x:.{digits}f}%" if x is not None else default


def _fmt_int(v: Any, default: str = "-") -> str:
    try:
        if v is None:
            return default
        return str(int(v))
    except Exception:
        return default


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(mean(vals)) if vals else None


def _pct(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def _classify_level(score: Optional[float]) -> str:
    s = _safe_float(score)
    if s is None:
        return "indeterminado"
    if s >= 0.80:
        return "alta"
    if s >= 0.55:
        return "média"
    if s >= 0.30:
        return "baixa"
    return "muito baixa"




def _normalize_heatmap_mode(mode: Any) -> str:
    s = str(mode or "fault_type").strip().lower()
    if s in {"mismatch", "mismatch_power", "power_mismatch"}:
        return "mismatch"
    return "fault_type"


def _display_label_for_report(*, heatmap_mode: str, state: int, label: Any, diagnosis_label: Any) -> str:
    diag = str(diagnosis_label or "").strip()
    base = str(label or diag or "invalid").strip() or "invalid"
    base_norm = base.lower()
    mode = _normalize_heatmap_mode(heatmap_mode)

    if state == STATE_NONE:
        return "invalid" if base_norm in {"ok", "normal", "", "-"} else base
    if state == STATE_OK:
        return "ok" if base_norm in {"normal", "ok", "", "-"} else base
    if state in {STATE_WARN, STATE_CRIT} and base_norm in {"normal", "ok", "invalid", "", "-"}:
        if mode == "mismatch":
            return "severe_power_mismatch" if state == STATE_CRIT else "power_mismatch_warning"
        return "critical_operational_anomaly" if state == STATE_CRIT else "operational_anomaly_warning"
    return base

# ---------------------------------------------------------------------------
# Styles / layout
# ---------------------------------------------------------------------------
def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(SITE_COLORS["navy"]),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#37506A"),
            spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor(SITE_COLORS["blue"]),
            spaceBefore=6,
            spaceAfter=5,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=11.5,
            textColor=colors.HexColor(SITE_COLORS["navy"]),
            spaceBefore=3,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.8,
            textColor=colors.HexColor(SITE_COLORS["text"]),
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.0,
            leading=8.2,
            textColor=colors.HexColor("#5B6C7E"),
            spaceBefore=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=8.5,
            textColor=colors.HexColor(SITE_COLORS["muted"]),
        ),
    }


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def _make_table(
    rows: Sequence[Sequence[Any]],
    widths: Optional[Sequence[float]] = None,
    *,
    font_size: float = 8.0,
    left_cols: Iterable[int] = (0,),
) -> Table:
    tbl = Table(list(rows), colWidths=widths, repeatRows=1)
    left_cols = tuple(left_cols)
    ts = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SITE_COLORS["head"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(SITE_COLORS["blue"])),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(SITE_COLORS["line"])),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ]
    )
    for idx in left_cols:
        ts.add("ALIGN", (idx, 0), (idx, -1), "LEFT")
    for col in range(len(rows[0])):
        if col not in left_cols:
            ts.add("ALIGN", (col, 0), (col, -1), "CENTER")
    for r in range(1, len(rows)):
        if r % 2 == 0:
            ts.add("BACKGROUND", (0, r), (-1, r), colors.HexColor(SITE_COLORS["band"]))
    tbl.setStyle(ts)
    return tbl


def _resolve_logo_path() -> Optional[Path]:
    candidates: List[Path] = []
    if settings is not None:
        base_dir = Path(getattr(settings, "BASE_DIR", "."))
        candidates.extend(
            [
                base_dir / "static" / "img" / "logo_navbar.png",
                base_dir / "static" / "img" / "logo.png",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#506375"))
    canvas.drawString(doc.leftMargin, 11 * mm, "Braz Solar Scan - FDD Mismatch")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 11 * mm, f"Página {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------
def _series(payload: Dict[str, Any], key: str) -> List[Any]:
    series = payload.get("series") or {}
    arr = series.get(key)
    return list(arr) if isinstance(arr, list) else []


def _selected_heatmap_mode(payload: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> str:
    payload_mode = (payload.get("heatmap_mode") or {}).get("selected") or payload.get("display_mode")
    filter_mode = None
    if isinstance(filters, dict):
        filter_mode = filters.get("heatmap_mode") or filters.get("display_mode")
    return _normalize_heatmap_mode(payload_mode or filter_mode or "fault_type")


def _selected_heatmap_label(payload: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> str:
    payload_label = (payload.get("heatmap_mode") or {}).get("selected_label")
    mode = _selected_heatmap_mode(payload, filters)
    if payload_label:
        return _safe_text(payload_label)
    if isinstance(filters, dict):
        raw = filters.get("heatmap_mode") or filters.get("display_mode")
        if raw:
            raw_norm = _normalize_heatmap_mode(raw)
            return "Mismatch" if raw_norm == "mismatch" else "Tipologia de falha"
    return "Mismatch" if mode == "mismatch" else "Tipologia de falha"


def _severity_from_point(code: Any, valid: Any, rca_code_to_sev: Dict[str, str]) -> int:
    try:
        if not bool(valid):
            return STATE_NONE
        severity = rca_code_to_sev.get(str(int(code)), "warn")
    except Exception:
        return STATE_NONE
    return {"ok": STATE_OK, "warn": STATE_WARN, "crit": STATE_CRIT}.get(severity, STATE_NONE)


def _state_from_class_name(cls: Any) -> int:
    s = str(cls or "").strip().lower()
    if s == "ok":
        return STATE_OK
    if s == "warn":
        return STATE_WARN
    if s == "crit":
        return STATE_CRIT
    return STATE_NONE


def _iter_points(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    t_local = _series(payload, "t_local")
    hm_day = _series(payload, "hm_day_local")
    hm_min = _series(payload, "hm_minute_local")
    labels = _series(payload, "label_display") or _series(payload, "diagnosis_label") or _series(payload, "diagnosis_label_display") or _series(payload, "labels") or _series(payload, "rca_label")
    diagnosis_labels = _series(payload, "diagnosis_label") or _series(payload, "diagnosis_label_display") or []
    codes = _series(payload, "codes") or _series(payload, "rca_code")
    valid = _series(payload, "valid") or _series(payload, "valid_period")
    mismatch = _series(payload, "mismatch_rel_raw") or _series(payload, "mismatch_rel")
    pac_real = _series(payload, "p_ac_real_w") or _series(payload, "p_ac_w")
    pac_model = _series(payload, "p_ac_model_w")
    gpoa = _series(payload, "g_poa") or _series(payload, "g_poa_used")
    tcell = _series(payload, "tcell_c")
    data_rel = _series(payload, "data_reliability_score")
    det_conf = _series(payload, "detection_confidence_score")
    diag_conf = _series(payload, "diagnosis_confidence_score")
    diag_lvl = _series(payload, "diagnosis_confidence_level")
    data_lvl = _series(payload, "data_reliability_level")
    det_lvl = _series(payload, "detection_confidence_level")
    domain = _series(payload, "domain_label")
    state_label = _series(payload, "state_label")
    irr_tier = _series(payload, "irradiance_tier")
    direct_grid = _series(payload, "direct_grid_evidence")
    zero_inj = _series(payload, "zero_injection_flag")
    heatmap_cls = _series(payload, "heatmap_class") or _series(payload, "hm_class_selected") or _series(payload, "hm_class")
    heatmap_mode = _selected_heatmap_mode(payload)
    rca_code_to_sev = payload.get("rca_code_to_sev") or {}

    n = max(len(t_local), len(codes), len(valid), len(mismatch), len(heatmap_cls))
    points: List[Dict[str, Any]] = []
    for i in range(n):
        code = codes[i] if i < len(codes) else None
        val = valid[i] if i < len(valid) else None
        if i < len(heatmap_cls) and str(heatmap_cls[i] or "").strip() != "":
            state = _state_from_class_name(heatmap_cls[i])
        else:
            state = _severity_from_point(code, val, rca_code_to_sev)
        raw_label = labels[i] if i < len(labels) else None
        diagnosis_label = diagnosis_labels[i] if i < len(diagnosis_labels) else None
        label = _display_label_for_report(
            heatmap_mode=heatmap_mode,
            state=state,
            label=raw_label,
            diagnosis_label=diagnosis_label,
        )
        points.append(
            {
                "idx": i,
                "ts_local": t_local[i] if i < len(t_local) else None,
                "day": hm_day[i] if i < len(hm_day) else None,
                "minute": hm_min[i] if i < len(hm_min) else None,
                "state": state,
                "label": label,
                "code": code,
                "valid": bool(val) if val is not None else False,
                "mismatch": _safe_float(mismatch[i] if i < len(mismatch) else None),
                "pac_real_w": _safe_float(pac_real[i] if i < len(pac_real) else None),
                "pac_model_w": _safe_float(pac_model[i] if i < len(pac_model) else None),
                "g_poa": _safe_float(gpoa[i] if i < len(gpoa) else None),
                "tcell_c": _safe_float(tcell[i] if i < len(tcell) else None),
                "data_rel": _safe_float(data_rel[i] if i < len(data_rel) else None),
                "det_conf": _safe_float(det_conf[i] if i < len(det_conf) else None),
                "diag_conf": _safe_float(diag_conf[i] if i < len(diag_conf) else None),
                "diag_level": _safe_text(diag_lvl[i] if i < len(diag_lvl) else None, "-"),
                "data_level": _safe_text(data_lvl[i] if i < len(data_lvl) else None, "-"),
                "det_level": _safe_text(det_lvl[i] if i < len(det_lvl) else None, "-"),
                "domain": _safe_text(domain[i] if i < len(domain) else None, "-"),
                "state_label": _safe_text(state_label[i] if i < len(state_label) else None, "-"),
                "irradiance_tier": _safe_text(irr_tier[i] if i < len(irr_tier) else None, "-"),
                "direct_grid": bool(direct_grid[i]) if i < len(direct_grid) else False,
                "zero_injection": bool(zero_inj[i]) if i < len(zero_inj) else False,
                "heatmap_mode": heatmap_mode,
            }
        )
    return points


# ---------------------------------------------------------------------------
# Metrics and interpretation
# ---------------------------------------------------------------------------
def _summarize_points(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    state_counts = Counter(int(p.get("state") or 0) for p in points)
    valid_points = [p for p in points if int(p.get("state") or 0) in {STATE_OK, STATE_WARN, STATE_CRIT}]
    diag_points = [p for p in valid_points if _safe_text(p.get("label"), "invalid").lower() not in {"", "-", "invalid", "normal", "ok"}]
    warn_crit = [p for p in valid_points if int(p.get("state") or 0) in {STATE_WARN, STATE_CRIT}]
    fault_points = [p for p in valid_points if int(p.get("state") or 0) == STATE_CRIT]
    low_data = [p for p in valid_points if (_safe_float(p.get("data_rel"), 0.0) or 0.0) < 0.55]
    low_diag = [p for p in valid_points if (_safe_float(p.get("diag_conf"), 0.0) or 0.0) < 0.55]

    label_counter = Counter(
        _safe_text(p.get("label"), "invalid")
        for p in diag_points
    )
    domain_counter = Counter(
        _safe_text(p.get("domain"), "-")
        for p in diag_points
    )

    worst = sorted(
        warn_crit,
        key=lambda p: (
            int(p.get("state") or 0),
            abs(_safe_float(p.get("mismatch"), 0.0) or 0.0),
            _safe_float(p.get("diag_conf"), 0.0) or 0.0,
        ),
        reverse=True,
    )[:5]

    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in points:
        by_day[str(p.get("day") or "-")].append(p)

    day_scores: List[Tuple[str, int, float, Optional[float], str]] = []
    for day, pts in by_day.items():
        crit = sum(1 for p in pts if int(p.get("state") or 0) == STATE_CRIT)
        warn = sum(1 for p in pts if int(p.get("state") or 0) == STATE_WARN)
        val = [p for p in pts if int(p.get("state") or 0) in {STATE_OK, STATE_WARN, STATE_CRIT}]
        dominant = Counter(_safe_text(p.get("label"), "invalid") for p in val if _safe_text(p.get("label"), "invalid").lower() not in {"invalid", "normal", "ok", "-"}).most_common(1)
        dominant_lbl = dominant[0][0] if dominant else "-"
        avg_diag = _mean([_safe_float(p.get("diag_conf")) for p in val])
        day_scores.append((day, crit * 100 + warn, float(max([abs(_safe_float(p.get("mismatch"), 0.0) or 0.0) for p in val], default=0.0)), avg_diag, dominant_lbl))
    day_scores.sort(key=lambda x: (x[1], x[2], x[3] or 0.0), reverse=True)

    return {
        "n_points": len(points),
        "state_counts": state_counts,
        "valid_points": len(valid_points),
        "warn_crit_points": len(warn_crit),
        "fault_points": len(fault_points),
        "diag_points": len(diag_points),
        "low_data_points": len(low_data),
        "low_diag_points": len(low_diag),
        "data_rel_mean": _mean([_safe_float(p.get("data_rel")) for p in valid_points]),
        "det_conf_mean": _mean([_safe_float(p.get("det_conf")) for p in valid_points]),
        "diag_conf_mean": _mean([_safe_float(p.get("diag_conf")) for p in valid_points]),
        "top_labels": label_counter.most_common(5),
        "top_domains": domain_counter.most_common(5),
        "worst_points": worst,
        "critical_days": day_scores[:12],
        "direct_grid_points": sum(1 for p in diag_points if p.get("direct_grid")),
        "zero_inj_points": sum(1 for p in diag_points if p.get("zero_injection")),
    }


def _build_heatmap_png(points: List[Dict[str, Any]], dt_minutes: int) -> BytesIO:
    if not points:
        img = PILImage.new("RGB", (1000, 220), "white")
        draw = ImageDraw.Draw(img)
        draw.text((40, 100), "Sem dados suficientes para desenhar o mapa de bins.", fill="#2C3E50", font=ImageFont.load_default())
        out = BytesIO()
        img.save(out, format="PNG")
        out.seek(0)
        return out

    days = sorted({str(p["day"]) for p in points if p.get("day")})
    by_day = {d: idx for idx, d in enumerate(days)}
    minutes = [int(p["minute"]) for p in points if p.get("minute") is not None]
    max_minute = max(minutes) if minutes else 23 * 60 + 45
    bins_per_day = int(max_minute // max(1, dt_minutes)) + 1
    grid = [[STATE_NONE for _ in range(bins_per_day)] for _ in range(len(days))]

    for p in points:
        day = p.get("day")
        minute = p.get("minute")
        if day not in by_day or minute is None:
            continue
        di = by_day[day]
        bi = int(int(minute) // max(1, dt_minutes))
        if 0 <= bi < bins_per_day:
            grid[di][bi] = max(grid[di][bi], int(p.get("state") or STATE_NONE))

    cell_w = 7 if bins_per_day <= 48 else 6
    cell_h = 4 if len(days) > 120 else 6
    left_w, top_h, right_pad, bottom_pad = 56, 24, 14, 16
    width = left_w + bins_per_day * cell_w + right_pad
    height = top_h + len(days) * cell_h + bottom_pad
    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, width - 1, height - 1], outline="#C9D7E6", width=1)
    for bi in range(0, bins_per_day + 1):
        x = left_w + bi * cell_w
        draw.line([(x, top_h), (x, height - bottom_pad)], fill="#DCE7F2", width=1)
    for di in range(0, len(days) + 1):
        y = top_h + di * cell_h
        draw.line([(left_w, y), (width - right_pad, y)], fill="#DCE7F2", width=1)

    for di, day in enumerate(days):
        y0 = top_h + di * cell_h
        y1 = y0 + cell_h
        for bi in range(bins_per_day):
            state = grid[di][bi]
            x0 = left_w + bi * cell_w
            x1 = x0 + cell_w
            draw.rectangle([x0, y0, x1, y1], fill=STATE_COLORS.get(state, STATE_COLORS[STATE_NONE]), outline=None)
        if di % (1 if len(days) <= 45 else 7 if len(days) <= 120 else 14) == 0 or di == len(days) - 1:
            draw.text((6, y0 + max(0, cell_h // 2 - 4)), day[5:], fill="#294056", font=font)

    bins_per_hour = max(1, int(round(60 / max(1, dt_minutes))))
    for hour in range(0, 24):
        bi = hour * bins_per_hour
        if bi >= bins_per_day:
            continue
        x = left_w + bi * cell_w + 1
        draw.text((x, 6), f"{hour:02d}:00", fill="#294056", font=font)

    out = BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out


def _build_monthly_rows(points: List[Dict[str, Any]]) -> List[List[Any]]:
    monthly: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"counts": Counter(), "diag": Counter(), "data": [], "det": [], "diagc": []})
    for p in points:
        day = _safe_text(p.get("day"), "-")
        month = day[:7] if len(day) >= 7 else day
        state = int(p.get("state") or 0)
        monthly[month]["counts"][state] += 1
        if state in {STATE_OK, STATE_WARN, STATE_CRIT}:
            lbl = _safe_text(p.get("label"), "invalid")
            if lbl.lower() not in {"invalid", "normal", "ok", "-"}:
                monthly[month]["diag"][lbl] += 1
            for key, bucket in (("data_rel", "data"), ("det_conf", "det"), ("diag_conf", "diagc")):
                v = _safe_float(p.get(key))
                if v is not None:
                    monthly[month][bucket].append(v)

    rows = [["Mês", "Normal", "Warn", "Falha", "Sem diag.", "Rótulo dominante", "Conf. dados", "Conf. detecção", "Conf. diagnóstico"]]
    for month in sorted(monthly.keys()):
        item = monthly[month]
        counts = item["counts"]
        dominant = item["diag"].most_common(1)[0][0] if item["diag"] else "-"
        rows.append([
            month,
            counts.get(STATE_OK, 0),
            counts.get(STATE_WARN, 0),
            counts.get(STATE_CRIT, 0),
            counts.get(STATE_NONE, 0),
            dominant,
            _fmt_pct(_mean(item["data"]), 1),
            _fmt_pct(_mean(item["det"]), 1),
            _fmt_pct(_mean(item["diagc"]), 1),
        ])
    return rows


def _build_critical_days_rows(points: List[Dict[str, Any]], max_rows: int = 18) -> List[List[Any]]:
    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in points:
        by_day[_safe_text(p.get("day"), "-")].append(p)

    ranked: List[Tuple[Any, ...]] = []
    for day, pts in by_day.items():
        valid = [p for p in pts if int(p.get("state") or 0) in {STATE_OK, STATE_WARN, STATE_CRIT}]
        warn = sum(1 for p in pts if int(p.get("state") or 0) == STATE_WARN)
        crit = sum(1 for p in pts if int(p.get("state") or 0) == STATE_CRIT)
        max_mm = max([abs(_safe_float(p.get("mismatch"), 0.0) or 0.0) for p in valid], default=0.0)
        dom = Counter(_safe_text(p.get("label"), "invalid") for p in valid if _safe_text(p.get("label"), "invalid").lower() not in {"invalid", "normal", "ok", "-"}).most_common(1)
        dom_lbl = dom[0][0] if dom else "-"
        avg_diag = _mean([_safe_float(p.get("diag_conf")) for p in valid])
        ranked.append((day, crit, warn, max_mm, dom_lbl, avg_diag))

    ranked.sort(key=lambda x: (x[1], x[2], x[3], x[5] or 0.0), reverse=True)
    rows = [["Data", "Falha", "Warn", "|mismatch|max", "Rótulo dominante", "Conf. diag. média"]]
    for day, crit, warn, max_mm, dom_lbl, avg_diag in ranked[:max_rows]:
        rows.append([day, crit, warn, f"{100.0 * max_mm:.1f}%", dom_lbl, _fmt_pct(avg_diag, 1)])
    return rows


def _build_top_bins_rows(points: List[Dict[str, Any]], max_rows: int = 30) -> List[List[Any]]:
    points = [p for p in points if int(p.get("state") or 0) in {STATE_WARN, STATE_CRIT}]
    points.sort(
        key=lambda p: (
            int(p.get("state") or 0),
            abs(_safe_float(p.get("mismatch"), 0.0) or 0.0),
            _safe_float(p.get("diag_conf"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    rows = [["Instante local", "Estado", "Rótulo", "Mismatch", "Pac real [W]", "Pac modelo [W]", "GPOA [W/m²]", "Conf. diag."]]
    for point in points[:max_rows]:
        rows.append(
            [
                _safe_text(point.get("ts_local")),
                STATE_LABELS.get(int(point.get("state") or 0), "-"),
                _safe_text(point.get("label")),
                _fmt_pct(point.get("mismatch"), 1),
                _fmt_float(point.get("pac_real_w"), 0),
                _fmt_float(point.get("pac_model_w"), 0),
                _fmt_float(point.get("g_poa"), 0),
                _fmt_pct(point.get("diag_conf"), 1),
            ]
        )
    return rows


def _build_executive_bullets(payload: Dict[str, Any], points: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, List[str]]:
    n = int(summary["n_points"])
    valid = int(summary["valid_points"])
    none = int(summary["state_counts"].get(STATE_NONE, 0))
    warn = int(summary["state_counts"].get(STATE_WARN, 0))
    crit = int(summary["state_counts"].get(STATE_CRIT, 0))
    ok = int(summary["state_counts"].get(STATE_OK, 0))

    data_mean = summary["data_rel_mean"]
    det_mean = summary["det_conf_mean"]
    diag_mean = summary["diag_conf_mean"]

    sources = payload.get("sources") or {}
    thresholds = payload.get("thresholds") or {}
    filters = payload.get("range") or {}

    label_txt = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in summary["top_labels"][:3]]) or "sem rótulos diagnósticos dominantes"
    domain_txt = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in summary["top_domains"][:3]]) or "sem domínio predominante"

    worst = summary["worst_points"]
    worst_txt = "não houve bins warn/falha no intervalo"
    if worst:
        wp = worst[0]
        worst_txt = (
            f"o bin mais severo ocorreu em <b>{_safe_text(wp.get('ts_local'))}</b>, com rótulo "
            f"<b>{_safe_text(wp.get('label'))}</b>, mismatch de <b>{_fmt_pct(wp.get('mismatch'), 1)}</b> "
            f"e confiança diagnóstica de <b>{_fmt_pct(wp.get('diag_conf'), 1)}</b>."
        )

    evidence = [
        (
            f"Foram processados <b>{n}</b> bins de 15 min no intervalo. Destes, <b>{valid}</b> ({_fmt_pct(_pct(valid, n), 1)}) "
            f"tiveram base operacional suficiente para classificação em nível de bin, enquanto <b>{none}</b> ({_fmt_pct(_pct(none, n), 1)}) "
            f"permaneceram fora da zona diagnóstica útil do algoritmo."
        ),
        (
            f"Entre os bins válidos, a distribuição observada foi: <b>{ok}</b> normal, <b>{warn}</b> warn e <b>{crit}</b> falha. "
            f"Os rótulos diagnósticos mais recorrentes foram: <b>{label_txt}</b>."
        ),
        worst_txt,
    ]

    inference: List[str] = []
    if crit > 0:
        inference.append(
            f"Houve ocorrência de bins classificados como <b>falha</b> ({crit} ocorrências), o que sustenta a existência de eventos de perda severa e não apenas dispersão estatística do mismatch."
        )
    else:
        inference.append(
            "Não houve predominância de bins em estado de <b>falha</b>; o comportamento do período foi dominado por bins <b>warn</b>, compatível com perdas parciais ou deratings a serem confirmados por evidência complementar."
        )

    if summary["direct_grid_points"] > 0:
        inference.append(
            f"Foram observados <b>{summary['direct_grid_points']}</b> bins com evidência direta de rede, reforçando a interpretação de que parte das ocorrências mais severas está associada a restrições/perturbações elétricas e não apenas a variabilidade meteorológica."
        )
    else:
        inference.append(
            f"O domínio diagnóstico predominante foi <b>{domain_txt}</b>; na ausência de ampla evidência direta de rede, a leitura deve ser tratada como inferência operacional baseada em padrão de resposta do sistema."
        )

    if summary["zero_inj_points"] > 0:
        inference.append(
            f"Foram identificados <b>{summary['zero_inj_points']}</b> bins com indício de zero injection, o que é relevante para diferenciar indisponibilidade/inibição de injeção de simples perda parcial de geração."
        )

    limits = [
        (
            f"A confiabilidade média dos dados no período foi <b>{_fmt_pct(data_mean, 1)}</b> ({_classify_level(data_mean)}), enquanto a confiança média da detecção foi <b>{_fmt_pct(det_mean, 1)}</b> e a do diagnóstico <b>{_fmt_pct(diag_mean, 1)}</b>."
        ),
        (
            f"O uso de <b>{_safe_text(sources.get('source_meteo'))}</b> como fonte meteorológica externa é tecnicamente adequado para supervisão de larga escala, mas não substitui instrumentação local; portanto, diagnósticos finos permanecem condicionados à qualidade do acoplamento entre operativos e meteo."
        ),
        (
            f"Os limiares aplicados no relatório foram warn_abs=<b>{_safe_text(thresholds.get('warn_abs'))}</b>, fault_abs=<b>{_safe_text(thresholds.get('fault_abs'))}</b>, gate GPOA=<b>{_safe_text(thresholds.get('gpoa_gate'))}</b> W/m². Em intervalos longos, a alta parcela de bins fora da zona diagnóstica útil recomenda ler o heatmap como triagem operacional, não como verdade-terreno exaustiva."
        ),
    ]
    return {"evidence": evidence, "inference": inference, "limits": limits}


def _story_bullets(story: List[Any], title: str, bullets: List[str], styles: Dict[str, ParagraphStyle]) -> None:
    story.append(_paragraph(title, styles["subsection"]))
    for txt in bullets:
        story.append(_paragraph(f"• {txt}", styles["body"]))
        story.append(Spacer(1, 1.2 * mm))


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_mismatch_pdf_report(
    *,
    plant_name: str,
    payload: Dict[str, Any],
    filters: Dict[str, Any],
    generated_at_local: str,
    user_label: str = "-",
) -> bytes:
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"FDD Mismatch - {plant_name}",
        author="OpenAI / ChatGPT",
        subject="Mismatch FDD",
    )

    points = _iter_points(payload)
    summary = _summarize_points(points)
    bullets = _build_executive_bullets(payload, points, summary)

    dt_minutes = int((filters.get("dt_minutes") or (payload.get("thresholds") or {}).get("dt_minutes") or 15) or 15)
    heat_png = _build_heatmap_png(points, dt_minutes)

    story: List[Any] = []
    logo = _resolve_logo_path()
    if logo is not None:
        try:
            story.append(Image(str(logo), width=34 * mm, height=10 * mm))
            story.append(Spacer(1, 2 * mm))
        except Exception:
            pass

    story.append(_paragraph(f"Braz Solar Scan - FDD Mismatch - {plant_name}", styles["title"]))
    story.append(
        _paragraph(
            f"Relatório exportado a partir da tela FDD Mismatch. Geração: <b>{_safe_text(generated_at_local)}</b>. Usuário: <b>{_safe_text(user_label)}</b>.",
            styles["subtitle"],
        )
    )

    rng = payload.get("range") or {}
    thresholds = payload.get("thresholds") or {}
    versions = payload.get("versions") or {}
    sources = payload.get("sources") or {}

    # Scope
    story.append(_paragraph("Escopo e parâmetros de execução", styles["section"]))
    scope_rows = [
        ["Campo", "Valor", "Campo", "Valor"],
        ["Planta", plant_name, "Período", f"{_safe_text(rng.get('start'))} -> {_safe_text(rng.get('end'))}"],
        ["Bin [min]", _safe_text(filters.get("dt_minutes")), "Pipeline", _safe_text(filters.get("pipeline") or payload.get("pipeline"))],
        ["Detector", _safe_text(versions.get("detector_version")), "source_meteo", _safe_text(sources.get("source_meteo"))],
        ["Heatmap", _selected_heatmap_label(payload, filters), "política total", _safe_text(sources.get("total_policy"))],
        ["warn_abs", _safe_text(filters.get("warn_abs") or thresholds.get("warn_abs")), "fault_abs", _safe_text(filters.get("fault_abs") or thresholds.get("fault_abs"))],
        ["gpoa_min", _safe_text(filters.get("gpoa_min") or thresholds.get("gpoa_gate")), "pmin_w", _safe_text(filters.get("pmin_w") or thresholds.get("pmin_w"))],
        ["source_oper", _safe_text(filters.get("source_oper")), "Heatmap nota", _safe_text((payload.get("heatmap_mode") or {}).get("selected_note"))],
    ]
    story.append(_make_table(scope_rows, widths=[34 * mm, 76 * mm, 34 * mm, 120 * mm], font_size=8.0, left_cols=(0, 1, 2, 3)))
    story.append(Spacer(1, 4 * mm))

    # Executive KPIs recomputed from points
    story.append(_paragraph("Resumo executivo do período", styles["section"]))
    kpi_rows = [
        ["Indicador", "Valor", "Indicador", "Valor"],
        ["Pontos na série", _fmt_int(summary["n_points"]), "Bins válidos", _fmt_int(summary["valid_points"])],
        ["Normal", _fmt_int(summary["state_counts"].get(STATE_OK)), "Warn", _fmt_int(summary["state_counts"].get(STATE_WARN))],
        ["Falha", _fmt_int(summary["state_counts"].get(STATE_CRIT)), "Sem diag. útil", _fmt_int(summary["state_counts"].get(STATE_NONE))],
        ["Confiab. dados média", _fmt_pct(summary["data_rel_mean"]), "Conf. detecção média", _fmt_pct(summary["det_conf_mean"])],
        ["Conf. diagnóstico média", _fmt_pct(summary["diag_conf_mean"]), "Leitura global", _classify_level(summary["diag_conf_mean"])],
    ]
    story.append(_make_table(kpi_rows, widths=[46 * mm, 40 * mm, 46 * mm, 40 * mm], font_size=8.1, left_cols=(0, 2)))
    story.append(Spacer(1, 4 * mm))

    story.append(_paragraph("Interpretação técnica do período", styles["section"]))
    _story_bullets(story, "1. Evidência observada", bullets["evidence"], styles)
    _story_bullets(story, "2. Inferência diagnóstica", bullets["inference"], styles)
    _story_bullets(story, "3. Limitações e confiabilidade", bullets["limits"], styles)
    story.append(Spacer(1, 3 * mm))

    notes_rows = [["Campo", "Valor"]]
    notes_rows.append(["selected_sources", ", ".join(str(x) for x in (rng.get("selected_sources") or [])) or "-"])
    notes_rows.append(["Topologias diagnósticas dominantes", ", ".join([f"{lbl} ({cnt})" for lbl, cnt in summary["top_domains"][:4]]) or "-"])
    notes_rows.append(["Rótulos diagnósticos dominantes", ", ".join([f"{lbl} ({cnt})" for lbl, cnt in summary["top_labels"][:4]]) or "-"])
    story.append(_make_table(notes_rows, widths=[54 * mm, 198 * mm], font_size=7.8, left_cols=(0, 1)))

    story.append(PageBreak())
    story.append(_paragraph("Heatmap do período", styles["section"]))
    story.append(_paragraph(f"Mapa dia x bin usando o modo de coloração selecionado: <b>{_selected_heatmap_label(payload, filters)}</b>. Cinza = sem diagnóstico útil, verde = normal, âmbar = warn, vermelho = falha.", styles["body"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Image(heat_png, width=255 * mm, height=120 * mm))
    story.append(_paragraph("Em intervalos longos, o heatmap deve ser lido como visão de triagem temporal: ele mostra persistência e sazonalidade dos estados, mas não substitui inspeção por evento nem análise com dados locais de referência.", styles["caption"]))

    story.append(PageBreak())
    story.append(_paragraph("Resumo agregado por mês", styles["section"]))
    story.append(_paragraph("Para intervalos anuais ou extensos, a leitura mensal é mais informativa do que a tabela diária completa. A tabela abaixo resume severidade, rótulo dominante e confiança média por mês.", styles["body"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_make_table(_build_monthly_rows(points), widths=[18 * mm, 18 * mm, 18 * mm, 18 * mm, 22 * mm, 66 * mm, 22 * mm, 24 * mm, 26 * mm], font_size=7.0, left_cols=(0, 5)))

    story.append(Spacer(1, 4 * mm))
    story.append(_paragraph("Dias de maior interesse operacional", styles["section"]))
    story.append(_paragraph("Os dias abaixo foram priorizados por severidade, número de bins warn/falha e magnitude máxima de mismatch. Eles servem como trilha rápida para revisão histórica quando o período analisado é longo.", styles["body"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_make_table(_build_critical_days_rows(points), widths=[24 * mm, 16 * mm, 16 * mm, 24 * mm, 112 * mm, 24 * mm], font_size=7.2, left_cols=(0, 4)))

    top_rows = _build_top_bins_rows(points)
    story.append(PageBreak())
    story.append(_paragraph("Bins de maior interesse diagnóstico", styles["section"]))
    story.append(_paragraph("Lista priorizada pelos bins com maior severidade operacional, maior magnitude de mismatch e maior robustez diagnóstica. Ela separa a evidência observada no tempo da interpretação agregada do período.", styles["body"]))
    if len(top_rows) > 1:
        story.append(Spacer(1, 2 * mm))
        story.append(_make_table(top_rows, widths=[44 * mm, 20 * mm, 52 * mm, 18 * mm, 22 * mm, 24 * mm, 20 * mm, 18 * mm], font_size=7.0, left_cols=(0, 2)))
    else:
        story.append(_paragraph("Não há bins classificados como warn/falha para o intervalo selecionado.", styles["body"]))

    story.append(Spacer(1, 4 * mm))
    story.append(_paragraph("Leitura recomendada do relatório", styles["section"]))
    story.append(_paragraph(
        "<b>Evidência observada</b> corresponde ao que foi efetivamente medido/inferido no bin (potência, mismatch, estado, cobertura e qualidade de dados). "
        "<b>Inferência diagnóstica</b> corresponde ao rótulo atribuído pelo pipeline. "
        "<b>Limitações de confiabilidade</b> delimitam o quanto essa inferência deve ser tomada como hipótese operacional forte, moderada ou fraca.",
        styles["body"],
    ))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    pdf = buf.getvalue()
    buf.close()
    return pdf
