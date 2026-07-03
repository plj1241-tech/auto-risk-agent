from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .risk_tools import RISK_LABELS


FONT_NAME = "HYSMyeongJo-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def generate_pdf(report: dict[str, Any]) -> bytes:
    """Generate a Korean five-section analyst report as PDF bytes."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=f"{report['corp_name']} 리스크 분석 리포트",
        author="자동차 부품업계 리스크 분석 에이전트",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "KoreanTitle",
        parent=styles["Title"],
        fontName=FONT_NAME,
        fontSize=20,
        leading=27,
        textColor=colors.HexColor("#0C447C"),
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
        wordWrap="CJK",
    )
    heading = ParagraphStyle(
        "KoreanHeading",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=13,
        leading=19,
        textColor=colors.HexColor("#185FA5"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        wordWrap="CJK",
    )
    body = ParagraphStyle(
        "KoreanBody",
        parent=styles["BodyText"],
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#2B2A28"),
        wordWrap="CJK",
    )
    note = ParagraphStyle(
        "KoreanNote",
        parent=body,
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#5B5A57"),
    )
    header = ParagraphStyle(
        "KoreanTableHeader",
        parent=body,
        fontSize=8.5,
        leading=12,
        textColor=colors.white,
    )

    def table(data: list[list[Any]], widths: list[float]) -> Table:
        converted = []
        for row_index, row in enumerate(data):
            row_style = header if row_index == 0 else body
            converted.append(
                [_paragraph(cell, row_style) if not isinstance(cell, Paragraph) else cell for cell in row]
            )
        result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0C447C")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8D6D0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F3EF")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return result

    sections = report["sections"]
    current = sections["summary"]
    story: list[Any] = [
        _paragraph(f"{report['corp_name']} 리스크 분석 리포트", title),
        _paragraph(
            f"기준연도: {report['year']}년 | Z-score 등급: {current['z_grade']}",
            ParagraphStyle("CenterMeta", parent=body, alignment=TA_CENTER),
        ),
        Spacer(1, 5 * mm),
        _paragraph("1. 요약", heading),
    ]
    metric_rows = [["리스크지표", "값"]] + [
        [RISK_LABELS[name], _fmt(value)] for name, value in current["metrics"].items()
    ]
    story.append(table(metric_rows, [75 * mm, 95 * mm]))
    flags = ", ".join(current["risk_flags"]) or "없음"
    story.extend([Spacer(1, 2 * mm), _paragraph(f"주요 경고: {flags}", body)])

    story.append(_paragraph("2. 시나리오 분석", heading))
    scenario = sections.get("scenario")
    if scenario:
        scenario_rows = [["지표", "기준", "시나리오", "변화"]]
        for item in scenario["predictions"].values():
            scenario_rows.append(
                [item["label"], _fmt(item["baseline"]), _fmt(item["scenario"]), f"{item['delta']:+.2f}"]
            )
        story.append(table(scenario_rows, [55 * mm, 38 * mm, 38 * mm, 39 * mm]))
        story.extend([Spacer(1, 2 * mm), _paragraph(scenario["guardrail"]["message"], note)])
    else:
        story.append(_paragraph("입력된 시나리오가 없습니다.", body))

    story.append(_paragraph("3. 원인 분석", heading))
    causes = sections["causes"]
    cause_rows = [["거시 피처", "값", "SHAP", "방향"]] + [
        [item["label"], _fmt(item["feature_value"]), f"{item['shap_value']:+.3f}", item["direction"]]
        for item in causes["top_macro_effects"]
    ]
    story.append(table(cause_rows, [70 * mm, 35 * mm, 35 * mm, 30 * mm]))
    story.extend([Spacer(1, 2 * mm), _paragraph(causes["validation_note"], note)])

    story.append(_paragraph("4. 동종업계 비교", heading))
    peer_rows = [["리스크지표", "기업값", "업계 중앙값", "건전성 순위"]]
    for name, item in sections["peer_comparison"]["comparisons"].items():
        if item:
            peer_rows.append(
                [
                    RISK_LABELS[name],
                    _fmt(item["value"]),
                    _fmt(item["peer_median"]),
                    f"{item['health_rank']} / {item['peer_count']}",
                ]
            )
    story.append(table(peer_rows, [55 * mm, 40 * mm, 45 * mm, 30 * mm]))

    story.append(_paragraph("5. 점검 제안", heading))
    for index, action in enumerate(sections["actions"], 1):
        story.append(_paragraph(f"{index}. {action}", body))
        story.append(Spacer(1, 1.5 * mm))
    story.extend(
        [
            Spacer(1, 7 * mm),
            _paragraph(
                "본 리포트는 공시 재무데이터 기반 분석 참고자료이며 투자·신용의견이 아닙니다.",
                note,
            ),
        ]
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT_NAME, 8)
        canvas.setFillColor(colors.HexColor("#77756F"))
        canvas.drawString(17 * mm, 10 * mm, "자동차 부품업계 리스크 분석 에이전트")
        canvas.drawRightString(193 * mm, 10 * mm, f"{doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
