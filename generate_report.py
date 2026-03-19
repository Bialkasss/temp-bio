"""
generate_report.py  —  QAP Batch Results → PDF
Usage: python3 generate_report.py <results.csv> [output.pdf]
"""
import sys, os, csv
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── palette ─────────────────────────────────────────────────
C_DARK  = colors.HexColor("#0d1b2a")
C_MID   = colors.HexColor("#1b2a3b")
C_ACC   = colors.HexColor("#1f4e79")
C_TEAL  = colors.HexColor("#0e7490")
C_RED   = colors.HexColor("#b91c1c")
C_GREEN = colors.HexColor("#15803d")
C_GOLD  = colors.HexColor("#b45309")
C_GREY  = colors.HexColor("#f1f5f9")
C_WHITE = colors.white
C_LINE  = colors.HexColor("#cbd5e1")

W = A4[0] - 4*cm   # usable width

# ── styles ───────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("T", parent=base["Title"],
        fontSize=20, textColor=C_WHITE, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=2)
    s["subtitle"] = ParagraphStyle("ST", parent=base["Normal"],
        fontSize=9, textColor=colors.HexColor("#94a3b8"),
        alignment=TA_CENTER, spaceAfter=0)
    s["h1"] = ParagraphStyle("H1", parent=base["Heading1"],
        fontSize=13, textColor=C_DARK, fontName="Helvetica-Bold",
        spaceBefore=16, spaceAfter=5)
    s["h2"] = ParagraphStyle("H2", parent=base["Heading2"],
        fontSize=10, textColor=C_TEAL, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=3)
    s["body"] = ParagraphStyle("B", parent=base["Normal"],
        fontSize=8.5, leading=13, textColor=C_DARK)
    s["mono"] = ParagraphStyle("M", parent=base["Normal"],
        fontSize=7.5, fontName="Courier", textColor=C_ACC,
        leading=11, wordWrap="CJK")
    s["label"] = ParagraphStyle("L", parent=base["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#64748b"))
    s["cell"]  = ParagraphStyle("C", parent=base["Normal"],
        fontSize=8, leading=10)
    return s


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── helpers ──────────────────────────────────────────────────
def fmt_f(v, d=4):
    try: return f"{float(v):.{d}f}"
    except: return str(v)

def fmt_i(v):
    try: return f"{int(float(v)):,}"
    except: return str(v)

def gap_color(gap_str):
    try:
        g = float(gap_str)
        if g <= 0:    return C_TEAL
        if g <= 5:    return C_GREEN
        if g <= 20:   return C_GOLD
        return C_RED
    except:
        return C_DARK


# ── banner ───────────────────────────────────────────────────
def header_banner(folder, n_inst, styles):
    banner = Table([[Paragraph("QAP Batch Performance Report", styles["title"])]],
                   colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
    ]))
    sub = Table([[Paragraph(
        f"Folder: <b>{folder}</b> &nbsp;|&nbsp; "
        f"Instances: <b>{n_inst}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["subtitle"])]],colWidths=[W])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_MID),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
    ]))
    return [banner, sub, Spacer(1, 0.5*cm)]


# ── algorithm description ────────────────────────────────────
def algo_section(styles):
    return [
        Paragraph("Algorithm", styles["h1"]),
        Paragraph(
            "<b>Problem:</b> Quadratic Assignment Problem — find permutation π minimising "
            "f(π) = Σ<sub>ij</sub> F[i][j]·D[π[i]][π[j]].",
            styles["body"]),
        Spacer(1,0.15*cm),
        Paragraph(
            "<b>Heuristic:</b> 2-OPT steepest-descent local search. "
            "Each run starts from a fresh Fisher-Yates random permutation. "
            "All unique index pairs (i,j) are evaluated each iteration using "
            "the O(n) delta formula; the best improving swap is applied until "
            "no improvement exists (local optimum). "
            "The starting index of the pair loop is randomised to remove positional bias.",
            styles["body"]),
        Spacer(1,0.15*cm),
        Paragraph(
            "<b>Delta formula O(n):</b> "
            "Δ(r,s) = 2·Σ<sub>k≠r,s</sub>"
            "[(F[k][r]−F[k][s])(D[π[s]][π[k]]−D[π[r]][π[k]])"
            "+(F[r][k]−F[s][k])(D[π[k]][π[s]]−D[π[k]][π[r]])]"
            "+2(F[r][s]−F[s][r])(D[π[s]][π[r]]−D[π[r]][π[s]]).",
            styles["body"]),
        Spacer(1,0.15*cm),
        Paragraph(
            "<b>File format:</b> .dat — first integer n, then n×n flow matrix, "
            "then n×n distance matrix (total 1+2n² values). "
            ".sln — n, known-optimal value, 1-based permutation of length n.",
            styles["body"]),
        HRFlowable(width="100%", thickness=0.5, color=C_LINE, spaceAfter=6),
    ]


# ── overview summary table ───────────────────────────────────
def overview_table(rows, styles):
    header = ["Instance","n","Runs","Best obj","Known opt","Gap %",
              "min t (s)","avg t (s)","max t (s)","Avg swaps","Avg Δ-evals"]
    data   = [header]

    for r in rows:
        gap = r.get("gap_best_pct","N/A")
        try:    gap_s = f"{float(gap):.3f}"
        except: gap_s = "N/A"

        data.append([
            r["instance"],
            r["n"],
            r["runs"],
            fmt_i(r["best_obj"]),
            r.get("known_opt","N/A"),
            gap_s,
            fmt_f(r["min_time_s"],6),
            fmt_f(r["avg_time_s"],6),
            fmt_f(r["max_time_s"],6),
            fmt_f(r["avg_swaps"],1),
            fmt_f(r["avg_deltas"],1),
        ])

    col_w = [3.2,0.8,0.9,2.2,2.0,1.5,2.0,2.0,2.0,1.8,2.0]
    col_w = [x*cm for x in col_w]

    t = Table(data, colWidths=col_w, repeatRows=1)
    style = [
        ("BACKGROUND",    (0,0),(-1,0), C_ACC),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0), 7.5),
        ("ALIGN",         (0,0),(-1,0), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,0), 6),
        ("BOTTOMPADDING", (0,0),(-1,0), 6),
        ("FONTSIZE",      (0,1),(-1,-1), 7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_GREY, C_WHITE]),
        ("ALIGN",         (1,1),(-1,-1), "RIGHT"),
        ("ALIGN",         (0,1),(0,-1), "LEFT"),
        ("TOPPADDING",    (0,1),(-1,-1), 4),
        ("BOTTOMPADDING", (0,1),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
        ("GRID",          (0,0),(-1,-1), 0.4, C_LINE),
        ("LINEABOVE",     (0,0),(-1,0),  1.5, C_TEAL),
        ("LINEBELOW",     (0,-1),(-1,-1),1.0, C_ACC),
    ]
    # colour the gap column per value
    for i, r in enumerate(rows, start=1):
        c = gap_color(r.get("gap_best_pct","N/A"))
        style.append(("TEXTCOLOR", (5,i),(5,i), c))
        style.append(("FONTNAME",  (5,i),(5,i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    return t


# ── per-instance detail card ─────────────────────────────────
def instance_card(r, styles):
    name     = r["instance"]
    has_sln  = r.get("known_opt","N/A") not in ("N/A","")
    perm_str = r.get("best_permutation","").strip('"')

    # timing bar: visual proportion min/avg/max
    try:
        t_min = float(r["min_time_s"])
        t_avg = float(r["avg_time_s"])
        t_max = float(r["max_time_s"])
    except:
        t_min = t_avg = t_max = 0.0

    left_col = [
        Paragraph(f"<b>{name}</b>  (n={r['n']}, {r['runs']} runs)", styles["h2"]),
        Paragraph(
            f"Best obj: <b>{fmt_i(r['best_obj'])}</b> &nbsp; "
            f"Obj range: [{fmt_i(r['min_obj'])}, {fmt_i(r['max_obj'])}] &nbsp; "
            f"Avg: {fmt_f(r['avg_obj'],1)}",
            styles["body"]),
        Spacer(1,0.15*cm),
        Paragraph(
            f"Runtime (s): min={fmt_f(r['min_time_s'],6)} "
            f"/ avg={fmt_f(r['avg_time_s'],6)} "
            f"/ max={fmt_f(r['max_time_s'],6)}",
            styles["body"]),
        Paragraph(
            f"Avg swaps: {fmt_f(r['avg_swaps'],1)} &nbsp;|&nbsp; "
            f"Avg delta-evals: {fmt_f(r['avg_deltas'],1)}",
            styles["body"]),
    ]

    if has_sln:
        gap = r.get("gap_best_pct","N/A")
        try:    gap_s = f"{float(gap):.4f} %"
        except: gap_s = "N/A"
        left_col.append(Spacer(1,0.1*cm))
        left_col.append(Paragraph(
            f"Known optimum: <b>{fmt_i(r['known_opt'])}</b> &nbsp; "
            f"Gap (best found): <b>{gap_s}</b>",
            styles["body"]))

    left_col.append(Spacer(1,0.15*cm))
    left_col.append(Paragraph("Best permutation (1-based):", styles["label"]))
    left_col.append(Paragraph(f"[ {perm_str} ]", styles["mono"]))

    card_data = [[left_col]]
    card = Table(card_data, colWidths=[W])
    card.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_GREY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("LINEABOVE",     (0,0),(-1,0),  2, C_TEAL),
        ("LINEBELOW",     (0,-1),(-1,-1),0.5, C_LINE),
    ]))
    return card


# ── build PDF ────────────────────────────────────────────────
def build_pdf(csv_path, pdf_path):
    rows = load_csv(csv_path)
    if not rows:
        print("No data in CSV."); return

    folder = os.path.dirname(os.path.abspath(csv_path))
    styles = build_styles()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.5*cm, bottomMargin=2*cm,
                            title="QAP Batch Report")
    story = []

    story.extend(header_banner(folder, len(rows), styles))
    story.extend(algo_section(styles))

    # Overview table
    story.append(Paragraph(f"Overview — {len(rows)} Instance(s)", styles["h1"]))
    story.append(Paragraph(
        "Gap % = (best_found − known_opt) / |known_opt| × 100. "
        "Negative gap means the heuristic found a better value than the .sln reference "
        "(the reference may not be the true optimum for synthetic instances).",
        styles["label"]))
    story.append(Spacer(1,0.3*cm))
    story.append(overview_table(rows, styles))
    story.append(Spacer(1,0.6*cm))

    # Per-instance detail cards
    story.append(Paragraph("Per-Instance Details", styles["h1"]))
    story.append(Spacer(1,0.2*cm))
    for r in rows:
        story.append(KeepTogether([instance_card(r, styles), Spacer(1,0.35*cm)]))

    doc.build(story)
    print(f"PDF saved: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_report.py <results.csv> [output.pdf]")
        sys.exit(1)
    csv_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) >= 3 else \
               os.path.splitext(csv_path)[0] + "_report.pdf"
    build_pdf(csv_path, pdf_path)