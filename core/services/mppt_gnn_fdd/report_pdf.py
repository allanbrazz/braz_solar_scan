from __future__ import annotations

from io import BytesIO
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


STATE_NONE = 0
STATE_OK = 1
STATE_WARN = 2
STATE_CRIT = 3

STATE_COLORS = {
    STATE_NONE: "#8B95A6",
    STATE_OK: "#46E68C",
    STATE_WARN: "#FFCF5A",
    STATE_CRIT: "#FF5A6E",
}


def _safe_text(v: Any, default: str = "-") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default



def _fmt_float(v: Any, digits: int = 2, default: str = "-") -> str:
    try:
        if v is None:
            return default
        return f"{float(v):.{digits}f}"
    except Exception:
        return default



def _fmt_pct(v: Any, digits: int = 1, default: str = "-") -> str:
    try:
        if v is None:
            return default
        return f"{100.0 * float(v):.{digits}f}%"
    except Exception:
        return default



def _fmt_int(v: Any, default: str = "-") -> str:
    try:
        if v is None:
            return default
        return str(int(v))
    except Exception:
        return default



def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "report_title",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F2746"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "report_subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#37506A"),
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "report_section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#17395E"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "report_body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1E2D3B"),
        ),
        "small": ParagraphStyle(
            "report_small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.2,
            textColor=colors.HexColor("#425466"),
        ),
        "mono": ParagraphStyle(
            "report_mono",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=7.4,
            leading=9.0,
            textColor=colors.HexColor("#203040"),
        ),
        "caption": ParagraphStyle(
            "report_caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.0,
            leading=8.6,
            textColor=colors.HexColor("#5B6C7E"),
            spaceBefore=4,
        ),
    }
    return styles



def _make_table(rows: Sequence[Sequence[Any]], widths: Optional[Sequence[float]] = None, *, header_bg: str = "#EAF2FB", body_bg: str = "#FFFFFF", font_size: float = 8.0, left_cols: Iterable[int] = (0,)) -> Table:
    tbl = Table(list(rows), colWidths=widths, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#163A5B")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(body_bg)),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6E1EE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    for idx in left_cols:
        ts.add("ALIGN", (idx, 0), (idx, -1), "LEFT")
    for col in range(len(rows[0])):
        if col not in set(left_cols):
            ts.add("ALIGN", (col, 0), (col, -1), "CENTER")
    if len(rows) > 2:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                ts.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F8FBFF"))
    tbl.setStyle(ts)
    return tbl



def _build_heatmap_png(payload: Dict[str, Any]) -> BytesIO:
    days = list(payload.get("days") or [])
    grid = list(payload.get("grid") or [])
    bpd = int(payload.get("bins_per_day") or 0)
    dtm = int(payload.get("dt_minutes") or 15)

    if not days or not grid or bpd <= 0:
        img = PILImage.new("RGB", (1000, 220), "white")
        draw = ImageDraw.Draw(img)
        draw.text((40, 100), "Sem dados suficientes para desenhar o mapa de bins.", fill="#2C3E50", font=ImageFont.load_default())
        out = BytesIO()
        img.save(out, format="PNG")
        out.seek(0)
        return out

    cell_w = 7 if bpd <= 48 else 6
    if len(days) > 220:
        cell_h = 3
    elif len(days) > 120:
        cell_h = 4
    else:
        cell_h = 6

    left_w = 56
    top_h = 24
    right_pad = 14
    bottom_pad = 16
    width = left_w + bpd * cell_w + right_pad
    height = top_h + len(days) * cell_h + bottom_pad

    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, width - 1, height - 1], outline="#C9D7E6", width=1)

    for bi in range(0, bpd + 1):
        x = left_w + bi * cell_w
        color = "#DCE7F2" if bi < bpd else "#C9D7E6"
        draw.line([(x, top_h), (x, height - bottom_pad)], fill=color, width=1)
    for di in range(0, len(days) + 1):
        y = top_h + di * cell_h
        color = "#DCE7F2" if di < len(days) else "#C9D7E6"
        draw.line([(left_w, y), (width - right_pad, y)], fill=color, width=1)

    for di, day in enumerate(days):
        row = grid[di] if di < len(grid) else []
        y0 = top_h + di * cell_h
        y1 = y0 + cell_h
        for bi in range(bpd):
            st = int(row[bi]) if bi < len(row) and row[bi] is not None else STATE_NONE
            x0 = left_w + bi * cell_w
            x1 = x0 + cell_w
            draw.rectangle([x0, y0, x1, y1], fill=STATE_COLORS.get(st, STATE_COLORS[STATE_NONE]), outline=None)

    day_step = 1 if len(days) <= 45 else 7 if len(days) <= 120 else 14
    for di, day in enumerate(days):
        if di % day_step != 0 and di != len(days) - 1:
            continue
        y = top_h + di * cell_h + max(0, cell_h // 2 - 4)
        draw.text((6, y), day[5:], fill="#294056", font=font)

    minute_step = 60
    bins_per_hour = max(1, int(round(60 / max(1, dtm))))
    for hour in range(0, 24, max(1, minute_step // 60)):
        bi = hour * bins_per_hour
        if bi >= bpd:
            continue
        x = left_w + bi * cell_w + 1
        draw.text((x, 6), f"{hour:02d}:00", fill="#294056", font=font)

    out = BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out



def _build_daily_rows(payload: Dict[str, Any]) -> List[List[Any]]:
    days = list(payload.get("days") or [])
    grid = list(payload.get("grid") or [])
    labels = list(payload.get("labels") or [])
    rows: List[List[Any]] = [["Data", "Normal", "Warn", "Anômalo", "Sem diag.", "Rótulo dominante"]]
    for di, day in enumerate(days):
        row = grid[di] if di < len(grid) else []
        labs = labels[di] if di < len(labels) else []
        counts = {STATE_NONE: 0, STATE_OK: 0, STATE_WARN: 0, STATE_CRIT: 0}
        label_count: Dict[str, int] = {}
        for bi, st in enumerate(row):
            counts[int(st or 0)] = counts.get(int(st or 0), 0) + 1
            lab = str(labs[bi] or "-") if bi < len(labs) else "-"
            label_count[lab] = label_count.get(lab, 0) + 1
        dominant = max(label_count.items(), key=lambda kv: kv[1])[0] if label_count else "-"
        rows.append([day, counts.get(STATE_OK, 0), counts.get(STATE_WARN, 0), counts.get(STATE_CRIT, 0), counts.get(STATE_NONE, 0), dominant])
    return rows



def _build_confusion_table(advanced: Dict[str, Any]) -> Optional[Table]:
    cm = advanced.get("confusion_matrix")
    labels = advanced.get("labels") or []
    if not cm:
        return None
    headers = ["real\\pred"] + [str(x) for x in labels]
    rows = [headers]
    for idx, row in enumerate(cm):
        rows.append([str(labels[idx]) if idx < len(labels) else str(idx)] + [str(v) for v in row])
    widths = [28 * mm] + [18 * mm] * (len(headers) - 1)
    return _make_table(rows, widths=widths, font_size=7.2, left_cols=(0,))



def _build_classification_report_table(advanced: Dict[str, Any]) -> Optional[Table]:
    rep = advanced.get("classification_report")
    if not isinstance(rep, dict) or not rep:
        return None
    rows = [["Classe", "Precisão", "Recall", "F1", "Suporte"]]
    order = []
    target_names = advanced.get("target_names") or []
    if target_names:
        order.extend([str(x) for x in target_names])
    order.extend([k for k in rep.keys() if k not in order])
    for k in order:
        v = rep.get(k)
        if not isinstance(v, dict):
            continue
        rows.append([
            str(k),
            _fmt_float(v.get("precision"), 3),
            _fmt_float(v.get("recall"), 3),
            _fmt_float(v.get("f1-score"), 3),
            _fmt_int(v.get("support")),
        ])
    widths = [46 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm]
    return _make_table(rows, widths=widths, font_size=7.2, left_cols=(0,))



def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)



def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#506375"))
    canvas.drawString(doc.leftMargin, 11 * mm, "Braz Solar Scan - Síntese operativa MPPT GNN FDD")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 11 * mm, f"Página {doc.page}")
    canvas.restoreState()



def build_mppt_gnn_pdf_report(*, plant_name: str, filters: Dict[str, Any], payload: Dict[str, Any], event_rows: List[Dict[str, Any]], generated_at_local: str, user_label: str = "-") -> bytes:
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"Síntese operativa - {plant_name}",
        author="OpenAI / ChatGPT",
        subject="MPPT GNN FDD",
    )

    story: List[Any] = []
    story.append(_paragraph(f"Síntese operativa - {plant_name}", styles["title"]))
    story.append(_paragraph(
        f"Relatório PDF gerado a partir do dashboard MPPT GNN FDD para o período selecionado pelo usuário. Geração: <b>{_safe_text(generated_at_local)}</b>. Usuário: <b>{_safe_text(user_label)}</b>.",
        styles["subtitle"],
    ))

    versions = payload.get("versions") or {}
    echo = payload.get("echo") or {}
    counts = payload.get("counts_by_state") or {}
    mh = payload.get("model_health") or {}
    advanced = (mh.get("advanced") or {}) if isinstance(mh, dict) else {}

    story.append(_paragraph("Escopo e filtros", styles["section"]))
    scope_rows = [
        ["Campo", "Valor", "Campo", "Valor"],
        ["Planta", plant_name, "Período", f"{_safe_text(payload.get('start'))} -> {_safe_text(payload.get('end'))}"],
        ["Bin [min]", _safe_text(payload.get("dt_minutes")), "view_mode", _safe_text(payload.get("view_mode"))],
        ["Detector", _safe_text(versions.get("detector_version") or echo.get("detector_version")), "Classificador evento", _safe_text(versions.get("event_classifier_version") or echo.get("event_classifier_version"))],
        ["Modelo treinado", _safe_text(versions.get("trained_model_version") or echo.get("trained_model_version")), "MPPT", _safe_text(filters.get("mppt_ui") or filters.get("mppt"))],
        ["source_oper", _safe_text(echo.get("source_oper")), "source_meteo", _safe_text(echo.get("source_meteo"))],
    ]
    story.append(_make_table(scope_rows, widths=[34 * mm, 71 * mm, 34 * mm, 118 * mm], font_size=8.1, left_cols=(0, 1, 2, 3)))
    story.append(Spacer(1, 5 * mm))

    story.append(_paragraph("Síntese operacional do período", styles["section"]))
    summary_rows = [
        ["Indicador", "Valor", "Indicador", "Valor"],
        ["diag_count", _fmt_int(payload.get("pred_count")), "merged_rows_total", _fmt_int(payload.get("merged_rows_total"))],
        ["Operativo normal", _fmt_int(counts.get("ok")), "Warn / atenção", _fmt_int(counts.get("warn"))],
        ["Operativo anômalo", _fmt_int(counts.get("fault")), "Sem diagnóstico", _fmt_int(counts.get("none"))],
        ["diag_rows_valid", _fmt_int(payload.get("diag_rows_valid")), "diag_rows_invalid", _fmt_int(payload.get("diag_rows_invalid"))],
        ["diag_rows_anomaly", _fmt_int(payload.get("diag_rows_anomaly")), "merged_rows_with_oper", _fmt_int(payload.get("merged_rows_with_oper"))],
    ]
    story.append(_make_table(summary_rows, widths=[45 * mm, 50 * mm, 45 * mm, 50 * mm], font_size=8.1, left_cols=(0, 2)))
    story.append(Spacer(1, 5 * mm))

    story.append(_paragraph("Saúde do modelo e bundle treinado", styles["section"]))
    mh_rows = [
        ["Campo", "Valor", "Campo", "Valor"],
        ["Status", _safe_text(mh.get("status_label")), "Fonte das métricas", _safe_text(mh.get("metrics_source_label"))],
        ["Nota", _safe_text(mh.get("status_note")), "Treinado em", _safe_text(mh.get("trained_at_utc"))],
        ["F1 macro validado", _fmt_pct(mh.get("validation_f1_macro")), "Balanced accuracy", _fmt_pct(mh.get("validation_balanced_accuracy"))],
        ["Cobertura validação", _safe_text(mh.get("validation_coverage_label")), "Dias de validação", _fmt_int(mh.get("validation_days"))],
    ]
    story.append(_make_table(mh_rows, widths=[34 * mm, 92 * mm, 34 * mm, 92 * mm], font_size=8.1, left_cols=(0, 1, 2, 3)))

    state_policy = payload.get("state_policy") or {}
    if state_policy:
        story.append(Spacer(1, 4 * mm))
        story.append(_paragraph("Política de estados do heatmap", styles["section"]))
        policy_rows = [["Parâmetro", "Valor"]]
        for k in [
            "name",
            "warn_mismatch_rel",
            "warn_single_rel",
            "warn_min_gpoa_wm2",
            "warn_min_qc_score",
            "warn_allows_interpolated_meteo",
            "event_green_confidence_min",
            "event_warn_confidence_min",
        ]:
            if k in state_policy:
                policy_rows.append([k, _safe_text(state_policy.get(k))])
        story.append(_make_table(policy_rows, widths=[76 * mm, 176 * mm], font_size=7.8, left_cols=(0, 1)))

    heatmap_png = _build_heatmap_png(payload)
    story.append(PageBreak())
    story.append(_paragraph("Mapa de bins do período", styles["section"]))
    story.append(_paragraph(
        "Heatmap dia x bin exportado a partir do estado final mostrado no dashboard MPPT GNN FDD. Cores: cinza = none, verde = normal, âmbar = warn, vermelho = anômalo.",
        styles["body"],
    ))
    heat_img = Image(heatmap_png, width=255 * mm, height=120 * mm)
    story.append(Spacer(1, 2 * mm))
    story.append(heat_img)
    story.append(_paragraph("O mapa acima usa o estado final do bin, já considerando detector plant-level, refinamento event-level e política híbrida de warn quando aplicável.", styles["caption"]))

    daily_rows = _build_daily_rows(payload)
    story.append(PageBreak())
    story.append(_paragraph("Histórico operativo diário", styles["section"]))
    story.append(_paragraph(
        "Cada linha resume, para uma data do período selecionado, quantos bins foram classificados como normal, warn, anômalo ou sem diagnóstico útil.",
        styles["body"],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(_make_table(daily_rows, widths=[28 * mm, 20 * mm, 20 * mm, 22 * mm, 24 * mm, 120 * mm], font_size=7.4, left_cols=(0, 5)))

    story.append(PageBreak())
    story.append(_paragraph("Histórico de eventos operativos", styles["section"]))
    if event_rows:
        evt_rows = [["ID", "Início local", "Fim local", "Status", "Final label", "MPPT/pred", "Conf.", "Dados", "Detec.", "Diag.", "Sev.", "Loss [Wh]"]]
        for r in event_rows:
            evt_rows.append([
                _fmt_int(r.get("event_id")),
                _safe_text(r.get("start_local")),
                _safe_text(r.get("end_local")),
                _safe_text(r.get("status")),
                _safe_text(r.get("final_label")),
                _safe_text(r.get("mppt_pred")),
                _fmt_pct(r.get("confidence")),
                _fmt_pct(r.get("event_data_reliability")),
                _fmt_pct(r.get("event_detection_confidence")),
                _fmt_pct(r.get("event_diagnosis_confidence")),
                _fmt_float(r.get("severity_score"), 2),
                _fmt_float(r.get("energy_loss_wh"), 1),
            ])
        story.append(_make_table(evt_rows, widths=[12 * mm, 27 * mm, 27 * mm, 16 * mm, 31 * mm, 40 * mm, 13 * mm, 13 * mm, 13 * mm, 13 * mm, 14 * mm, 18 * mm], font_size=6.6, left_cols=(1, 2, 3, 4, 5)))
    else:
        story.append(_paragraph("Não há eventos persistidos para os filtros e o intervalo selecionados.", styles["body"]))

    if advanced:
        story.append(PageBreak())
        story.append(_paragraph("Métricas avançadas do modelo treinado", styles["section"]))
        adv_rows = [
            ["Campo", "Valor", "Campo", "Valor"],
            ["Fonte", _safe_text(advanced.get("metrics_source_label") or mh.get("metrics_source_label")), "n_samples", _fmt_int(advanced.get("n_samples"))],
            ["Accuracy", _fmt_pct(advanced.get("accuracy")), "Balanced accuracy", _fmt_pct(advanced.get("balanced_accuracy"))],
            ["F1 macro", _fmt_pct(advanced.get("f1_macro")), "F1 weighted", _fmt_pct(advanced.get("f1_weighted"))],
            ["Log loss", _fmt_float(advanced.get("log_loss"), 4), "Mean confidence", _fmt_float(advanced.get("mean_confidence"), 4)],
        ]
        story.append(_make_table(adv_rows, widths=[35 * mm, 85 * mm, 35 * mm, 85 * mm], font_size=8.0, left_cols=(0, 1, 2, 3)))

        split = advanced.get("dataset_split") or {}
        split_rows = [
            ["Split", "Valor", "Split", "Valor"],
            ["Estratégia", _safe_text(split.get("strategy")), "Train samples", _fmt_int(split.get("train_samples"))],
            ["Validation samples", _fmt_int(split.get("validation_samples")), "Coverage fraction", _fmt_pct(split.get("coverage_fraction"))],
            ["Train day range", f"{_safe_text(split.get('train_day_start'))} -> {_safe_text(split.get('train_day_end'))}", "Validation day range", f"{_safe_text(split.get('validation_day_start'))} -> {_safe_text(split.get('validation_day_end'))}"],
        ]
        story.append(Spacer(1, 4 * mm))
        story.append(_make_table(split_rows, widths=[35 * mm, 85 * mm, 35 * mm, 85 * mm], font_size=8.0, left_cols=(0, 1, 2, 3)))

        cm_tbl = _build_confusion_table(advanced)
        if cm_tbl is not None:
            story.append(Spacer(1, 5 * mm))
            story.append(_paragraph("Matriz de confusão", styles["section"]))
            story.append(cm_tbl)

        rep_tbl = _build_classification_report_table(advanced)
        if rep_tbl is not None:
            story.append(Spacer(1, 5 * mm))
            story.append(_paragraph("Relatório por classe", styles["section"]))
            story.append(rep_tbl)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    pdf = buf.getvalue()
    buf.close()
    return pdf
