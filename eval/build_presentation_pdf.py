"""Build the Sentinel PDF from the reviewed presentation outline.

Run: python -m eval.build_presentation_pdf
"""
from __future__ import annotations

from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation.pdf"
FIGURES = ROOT / "results" / "figures"
FONT_DIR = (Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/"
            "libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype")
PAGE_W, PAGE_H = 960, 540
TOTAL_SLIDES = 11
MARGIN = 54
DARK = colors.HexColor("#172235")
MUTED = colors.HexColor("#4B5563")
ACCENT = colors.HexColor("#315B77")
RULE = colors.HexColor("#D9DEE3")
PALE = colors.HexColor("#F2F5F7")


def setup_fonts() -> None:
    if (FONT_DIR / "LiberationSans-Regular.ttf").is_file():
        regular, bold = FONT_DIR / "LiberationSans-Regular.ttf", FONT_DIR / "LiberationSans-Bold.ttf"
    else:
        bundled = Path(reportlab.__file__).parent / "fonts"
        regular, bold = bundled / "Vera.ttf", bundled / "VeraBd.ttf"
    pdfmetrics.registerFont(TTFont("LiberationSans", str(regular)))
    pdfmetrics.registerFont(TTFont("LiberationSans-Bold", str(bold)))
    pdfmetrics.registerFontFamily("LiberationSans", normal="LiberationSans", bold="LiberationSans-Bold")


def para(text: str, size: float = 15, leading: float | None = None, color=DARK,
         bold: bool = False) -> Paragraph:
    style = ParagraphStyle(
        "slide", fontName="LiberationSans-Bold" if bold else "LiberationSans",
        fontSize=size, leading=leading or size * 1.27, textColor=color,
        spaceAfter=0, allowWidows=0, allowOrphans=0,
    )
    return Paragraph(text, style)


def put(c: canvas.Canvas, text: str, x: float, top: float, width: float,
        size: float = 15, leading: float | None = None, color=DARK,
        bold: bool = False) -> float:
    p = para(text, size, leading, color, bold)
    _, height = p.wrap(width, PAGE_H)
    p.drawOn(c, x, PAGE_H - top - height)
    return top + height


def bullet(c: canvas.Canvas, text: str, top: float, x: float = MARGIN,
           width: float = PAGE_W - 2 * MARGIN, size: float = 15, gap: float = 13) -> float:
    c.setFillColor(ACCENT)
    c.circle(x + 4, PAGE_H - top - 8, 2.4, fill=1, stroke=0)
    return put(c, text, x + 20, top, width - 20, size) + gap


def header(c: canvas.Canvas, number: int, title: str) -> float:
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    put(c, title, MARGIN, 38, PAGE_W - 2 * MARGIN, size=27, bold=True)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.8)
    c.line(MARGIN, PAGE_H - 80, PAGE_W - MARGIN, PAGE_H - 80)
    c.setFillColor(MUTED)
    c.setFont("LiberationSans", 10)
    c.drawRightString(PAGE_W - MARGIN, 24, f"{number} / {TOTAL_SLIDES}")
    return 100


def table(c: canvas.Canvas, data: list[list[str]], col_widths: list[float],
          top: float, size: float = 13, row_pad: float = 7,
          header_rows: int = 1, highlight_rows: tuple[int, ...] = ()) -> float:
    cell = lambda s, bold=False: para(str(s), size, size * 1.14, DARK, bold)
    rows = [[cell(v, i < header_rows or i in highlight_rows) for v in row]
            for i, row in enumerate(data)]
    t = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=header_rows)
    rules = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), row_pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), row_pad),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
    ]
    for index in highlight_rows:
        rules.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#E9F0F4")))
    t.setStyle(TableStyle(rules))
    _, h = t.wrapOn(c, sum(col_widths), PAGE_H)
    t.drawOn(c, MARGIN, PAGE_H - top - h)
    return top + h


def figure(c: canvas.Canvas, path: Path, left: float, top: float,
           width: float, height: float) -> None:
    image = ImageReader(str(path))
    iw, ih = image.getSize()
    scale = min(width / iw, height / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(image, left + (width - w) / 2, PAGE_H - top - (height + h) / 2,
                width=w, height=h, mask="auto")


def finish(c: canvas.Canvas) -> None:
    c.showPage()


def slide_1(c: canvas.Canvas) -> None:
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    put(c, "Sentinel", 72, 106, 816, size=47, bold=True)
    put(c, "Adaptive time-series anomaly detection", 74, 177, 780, size=22, color=ACCENT)
    put(c, "Team April 28  |  Lokesh K  |  Likhit G", 74, 233, 780, size=16)
    put(c, "Challenge 2", 74, 262, 780, size=15, color=MUTED)
    put(c, "Two statistical checks flag unusual events as readings arrive. Each alert includes a short explanation.",
        74, 323, 760, size=19, leading=25)
    put(c, "Event-F1  0.691", 74, 420, 360, size=21, bold=True, color=ACCENT)
    put(c, "2.8 µs per point", 453, 420, 350, size=21, bold=True, color=ACCENT)
    finish(c)


def slide_2(c: canvas.Canvas) -> None:
    y = header(c, 2, "Problem and motivation")
    for text in [
        "A time series is a sequence of readings, such as CPU usage, traffic, or sales.",
        "Sentinel must flag unusual behaviour <b>as each reading arrives</b>, using only earlier readings.",
        "Missed incidents matter. Too many false alerts waste people's attention.",
        "Each result has a score, a NORMAL or ANOMALOUS label, and an explanation.",
    ]:
        y = bullet(c, text, y, size=18, gap=23)
    put(c, "Goal: accurate event alerts with low delay and low runtime.", MARGIN, 409,
        PAGE_W - 2 * MARGIN, size=18, bold=True, color=ACCENT)
    finish(c)


def slide_3(c: canvas.Canvas) -> None:
    y = header(c, 3, "Data and preparation")
    data = [
        ["NAB category", "Files", "Anomaly windows"],
        ["Artificial", "11", "6"], ["AWS CloudWatch", "17", "30"],
        ["Ad Exchange", "6", "14"], ["Known Cause", "7", "19"],
        ["Traffic", "7", "14"], ["Tweets", "10", "33"],
        ["Total", "58", "116"],
    ]
    table(c, data, [350, 130, 210], y, size=13.4, row_pad=5, highlight_rows=(7,))
    x = 770
    put(c, "365,558", x, 118, 150, size=25, bold=True, color=ACCENT)
    put(c, "readings", x, 151, 150, size=13, color=MUTED)
    put(c, "Learning period", x, 219, 150, size=14, bold=True)
    put(c, "First 15% of each file, up to 750 readings. No alerts or scoring during this period.",
        x, 247, 150, size=12.6, leading=17)
    put(c, "Each file stays in its original order; repeated or out-of-order timestamps are handled.",
        MARGIN, 463, 850, size=13, color=MUTED)
    finish(c)


def slide_4(c: canvas.Canvas) -> None:
    header(c, 4, "System design")
    stages = [
        ("1", "Reading arrives", "Time and value"),
        ("2", "Check input", "Clean invalid readings"),
        ("3", "Two detectors", "RobustZ + LaggedResidual"),
        ("4", "Decide", "Adaptive thresholds + new high/low"),
        ("5", "Manage event", "Confirm, join, space alerts"),
        ("6", "Return result", "Score, label, explanation"),
    ]
    y = 119
    for n, title, detail in stages:
        put(c, n, MARGIN + 4, y + 2, 35, size=14, bold=True, color=ACCENT)
        put(c, title, MARGIN + 52, y, 257, size=17, bold=True)
        put(c, detail, MARGIN + 322, y + 1, 480, size=16)
        c.setStrokeColor(RULE)
        c.line(MARGIN + 52, PAGE_H - y - 32, PAGE_W - MARGIN, PAGE_H - y - 32)
        y += 59
    put(c, "Every decision uses the current and earlier readings only.", MARGIN, 477,
        PAGE_W - 2 * MARGIN, size=14.5, bold=True, color=ACCENT)
    finish(c)


def slide_5(c: canvas.Canvas) -> None:
    y = header(c, 5, "Core algorithm")
    data = [
        ["Part", "What it does"],
        ["RobustZ", "Checks whether a reading is far from its usual level, allowing for normal variation."],
        ["LaggedResidual", "Compares it with the average of 8 readings from roughly 256 points earlier."],
        ["Adaptive thresholds", "Each detector sets its own alarm level using high learning-period scores."],
        ["New high / low", "Flags a value beyond the range seen so far by the chosen margin."],
    ]
    y = table(c, data, [235, 617], y, size=15, row_pad=11)
    y += 22
    y = put(c, "RobustZ starts with the median of learning-period readings. It also measures the usual spread, so being above the median alone does not trigger an alert.",
            MARGIN, y, 850, size=15)
    put(c, "Examples from the current default", MARGIN, y + 23, 850,
        size=15, color=ACCENT, bold=True)
    put(c, "<b>RobustZ:</b> CVS tweet count reached 10 versus an expected 0.41. It was far above its usual level.",
        MARGIN, y + 51, 850, size=13.3)
    put(c, "<b>Spatial-extreme flag:</b> Ambient temperature read 82.29 versus the earlier high of 80.52, crossing the new-high margin.",
        MARGIN, y + 79, 850, size=13.3)
    put(c, "Both are inside labelled anomaly windows. The spatial flag is a separate new-range check alongside the two statistical detectors.",
        MARGIN, y + 111, 850, size=11.5, color=MUTED)
    finish(c)


def slide_6(c: canvas.Canvas) -> None:
    y = header(c, 6, "Experiment history and tuning")
    put(c, "We measured each change before choosing one global default.",
        MARGIN, y, 850, size=18, bold=True, color=ACCENT)
    items = [
        "Compared simple baselines and detector signals on NAB.",
        "Searched detector combinations, lag lengths, and window lengths.",
        "Tested the new-high / new-low margin and event grouping rules.",
        "Selected RobustZ + LaggedResidual, an 8-reading older average, a 256-reading lag, and a 5.75% margin.",
    ]
    y = 155
    for number, text in enumerate(items, 1):
        put(c, f"{number}.", MARGIN, y, 45, size=16, bold=True, color=ACCENT)
        y = put(c, text, MARGIN + 43, y, 795, size=17, leading=22) + 24
    put(c, "Validation: rotate which NAB category is held out when selecting settings.",
        MARGIN, 435, 850, size=15, bold=True)
    finish(c)


def slide_7(c: canvas.Canvas) -> None:
    header(c, 7, "Evaluation method")
    put(c, "An alert inside a labelled anomaly window counts as a true positive. An alert outside all windows is a false positive. A window without an alert is a false negative.",
        MARGIN, 101, 850, size=14.2, leading=18)
    data = [
        ["Current default", "Value", "Across 58 files", "Count"],
        ["Precision", "0.731", "True positives", "76"],
        ["Recall", "0.655", "False positives", "28"],
        ["Event-F1", "0.691", "False negatives", "40"],
        ["Official NAB score", "58.80", "Median delay", "100 readings"],
    ]
    table(c, data, [221, 112, 281, 142], 168, size=12.8, row_pad=6, highlight_rows=(3,))
    put(c, "Delay is measured from window start to the first alert, among detected windows only.",
        MARGIN, 348, 850, size=13, color=MUTED)
    compare = [
        ["Model", "Windows found / 116", "Median position"],
        ["Sentinel", "76", "49.7%"], ["CAD OSE", "92", "49.7%"],
        ["Numenta HTM", "96", "49.6%"], ["ARTime", "106", "49.9%"],
        ["Windowed Gaussian", "73", "48.8%"],
    ]
    table(c, compare, [356, 214, 186], 380, size=10.7, row_pad=3.4, highlight_rows=(1,))
    finish(c)


def slide_8(c: canvas.Canvas) -> None:
    header(c, 8, "Event-F1 and inference time")
    figure(c, FIGURES / "event_f1_vs_inference_time.png", 54, 111, 638, 374)
    put(c, "Sentinel default", 713, 141, 193, size=16, bold=True, color=ACCENT)
    put(c, "0.691 event-F1", 713, 176, 193, size=18, bold=True)
    put(c, "2.7 µs per point", 713, 208, 193, size=16)
    put(c, "CAD OSE", 713, 278, 193, size=15, bold=True, color=MUTED)
    put(c, "0.676 event-F1", 713, 310, 193, size=16)
    put(c, "231 µs per point", 713, 340, 193, size=14)
    put(c, "Published models use NAB's stored scores and thresholds. Same-stream runtimes; time axis compressed, with actual µs in labels. A dash means no local runtime.",
        MARGIN, 484, 850, size=11.5, color=MUTED)
    finish(c)


def slide_9(c: canvas.Canvas) -> None:
    header(c, 9, "Precision and recall")
    figure(c, FIGURES / "precision_vs_recall_models.png", 54, 111, 638, 374)
    put(c, "Sentinel default", 713, 141, 193, size=16, bold=True, color=ACCENT)
    put(c, "Precision 0.731", 713, 176, 193, size=17, bold=True)
    put(c, "Recall 0.655", 713, 208, 193, size=16)
    put(c, "CAD OSE", 713, 278, 193, size=15, bold=True, color=MUTED)
    put(c, "Precision 0.590", 713, 310, 193, size=16)
    put(c, "Recall 0.793", 713, 340, 193, size=16)
    put(c, "Published models use NAB's stored scores and thresholds. More windows found can come with more false alerts. A dash means undefined precision.",
        MARGIN, 484, 850, size=11.5, color=MUTED)
    finish(c)


def slide_10(c: canvas.Canvas) -> None:
    header(c, 10, "Four NYC taxi alerts")
    figure(c, FIGURES / "example_nyc_taxi.png", MARGIN, 100, 852, 238)
    data = [
        ["Alert", "Reading", "Older average", "Why LaggedResidual flagged it"],
        ["1 Nov 2014, 13:30", "25,115", "2,602", "Much higher than the earlier level"],
        ["28 Nov 2014, 06:00", "2,870", "24,890", "Much lower than the earlier level"],
        ["25 Dec 2014, 06:00", "2,144", "26,316", "Much lower than the earlier level"],
        ["27 Jan 2015, 06:00", "69", "22,673", "Nearly zero compared with the earlier level"],
    ]
    table(c, data, [173, 105, 146, 426], 353, size=12.2, row_pad=4.2)
    put(c, "Each older average uses 8 readings roughly 256 points earlier. Four of five labelled windows found; no false events.",
        MARGIN, 485, 850, size=11.5, color=MUTED)
    finish(c)


def slide_11(c: canvas.Canvas) -> None:
    y = header(c, 11, "Limits and next steps")
    for text in [
        "<b>40 of 116</b> labelled windows are missed. Changes in daily patterns can be hard to spot when readings remain within the usual range.",
        "The default was selected and measured on public NAB data. Performance on hidden data is unknown.",
        "Next: detect pattern changes while keeping false alerts and runtime low.",
    ]:
        y = bullet(c, text, y, size=17, gap=27)
    put(c, "Conclusion", MARGIN, 359, 850, size=18, bold=True, color=ACCENT)
    put(c, "Two causal statistical checks produce clear alerts at 0.691 event-F1 and about 2.8 µs per point.",
        MARGIN, 395, 830, size=20, leading=27, bold=True)
    finish(c)


def main() -> None:
    setup_fonts()
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    pdf.setTitle("Sentinel - Hackathon Challenge 2")
    pdf.setAuthor("Team April 28")
    for build in (slide_1, slide_2, slide_3, slide_4, slide_5,
                  slide_6, slide_7, slide_8, slide_9, slide_10, slide_11):
        build(pdf)
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
