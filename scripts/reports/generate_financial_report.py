"""Generate source-reconciled financial reports, charts, and persistent run traces.

python -m scripts.reports.generate_financial_report --profile configs/financial_reports/apple_2024.json
"""
import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import pymupdf

from scripts.observability.trace_log import TraceLog, artifact, emit

from scripts.common.paths import ROOT  # noqa: E402 - repo root


def normalize(text):
    return " ".join(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, default=str) + "\n")


def parse_row(text, label, count, occurrence=0):
    """Fail if a row is ambiguous or incomplete; never collect the next row's values."""
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if normalize(line) == normalize(label)]
    if occurrence >= len(matches):
        raise ValueError(f"Missing financial row: {label}")
    start = matches[occurrence]
    values, evidence = [], [lines[start]]
    for line in lines[start + 1:]:
        evidence.append(line)
        clean = line.replace("\ufffd", "").replace("\u00a0", "").strip()
        if not clean or clean == "$" or "%" in clean:
            continue
        if not re.fullmatch(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", clean):
            break
        value = Decimal(clean.replace(",", "").strip("()"))
        if clean.startswith("("):
            value = -value
        values.append(value)
        if len(values) == count:
            return values, "\n".join(evidence)
    raise ValueError(f"Incomplete financial row: {label}; expected {count}, got {len(values)}")


def retrieve_financials(profile, source):
    groups, notes = {}, []
    with pymupdf.open(source) as pdf:
        for group, specification in profile["groups"].items():
            candidates = [i for i, p in enumerate(pdf) if specification["heading"] in p.get_text()]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing financial statement: {group}")
            page = candidates[0]
            text = pdf[page].get_text()
            rows = {}
            for key, selector in specification["rows"].items():
                selector = {"label": selector} if isinstance(selector, str) else selector
                values, quote = parse_row(text, selector["label"], specification["count"], selector.get("occurrence", 0))
                rows[key] = {"values": values, "label": selector["label"], "quote": quote,
                             "pdf_page": page + 1, "years": profile["years"][:specification["count"]]}
                emit("retrieval.row", group=group, row_id=key, label=selector["label"], pdf_page=page + 1,
                     values=values, years=rows[key]["years"], source_quote_sha256=hashlib.sha256(quote.encode()).hexdigest())
            groups[group] = rows
            emit("retrieval.statement", group=group, source=str(source), pdf_page=page + 1,
                 row_count=len(rows), row_ids=list(rows), evidence_sha256=hashlib.sha256(text.encode()).hexdigest())
        for note in profile.get("notes", []):
            text = pdf[note["pdf_page"] - 1].get_text(sort=True)
            if normalize(note["anchor"]) not in normalize(text):
                raise ValueError(f"Missing narrative anchor: {note['title']}")
            notes.append(note)
            emit("retrieval.narrative", title=note["title"], pdf_page=note["pdf_page"], anchor=note["anchor"])
    return groups, notes


def analyze(groups):
    def v(group, row, i=0):
        return groups[group][row]["values"][i]
    checks = []
    for i in range(3):
        checks.extend([
            (f"income_gross_{i}", v("income", "revenue", i) - v("income", "cost_of_sales", i) == v("income", "gross_profit", i)),
            (f"income_operating_{i}", v("income", "gross_profit", i) - v("income", "opex", i) == v("income", "operating_income", i)),
            (f"income_net_{i}", v("income", "pretax_income", i) - v("income", "tax", i) == v("income", "net_income", i)),
            (f"product_total_{i}", sum(row["values"][i] for row in groups["products"].values()) == v("income", "revenue", i)),
            (f"geographic_total_{i}", sum(row["values"][i] for row in groups["geography"].values()) == v("income", "revenue", i)),
        ])
    for i in range(2):
        checks.append((f"balance_equation_{i}", v("balance", "assets", i) == v("balance", "liabilities", i) + v("balance", "equity", i)))
    emit("validation.reconciliation", checks=[{"name": n, "passed": b} for n, b in checks])
    if not all(passed for _, passed in checks):
        raise ValueError("Financial reconciliation failed")
    ratios = {}
    for i, year in enumerate(groups["income"]["revenue"]["years"]):
        revenue = v("income", "revenue", i)
        ratios[year] = {
            "Gross margin": v("income", "gross_profit", i) / revenue * 100,
            "Operating margin": v("income", "operating_income", i) / revenue * 100,
            "Net margin": v("income", "net_income", i) / revenue * 100,
            "Effective tax rate": v("income", "tax", i) / v("income", "pretax_income", i) * 100,
            "Free cash flow": v("cashflow", "operating_cash", i) + v("cashflow", "capex", i),
            "Cash conversion": v("cashflow", "operating_cash", i) / v("income", "net_income", i),
        }
    for i, year in enumerate(groups["balance"]["assets"]["years"]):
        debt = sum(v("balance", name, i) for name in ("commercial_paper", "current_debt", "noncurrent_debt"))
        ratios[year].update({
            "Current ratio": v("balance", "current_assets", i) / v("balance", "current_liabilities", i),
            "Working capital": v("balance", "current_assets", i) - v("balance", "current_liabilities", i),
            "Debt carrying value": debt,
            "Cash plus securities": sum(v("balance", name, i) for name in ("cash", "current_securities", "noncurrent_securities")),
            "Debt / equity": debt / v("balance", "equity", i),
        })
    emit("analysis.metrics", ratios=ratios, arithmetic="decimal", checks_passed=len(checks))
    return {"checks": checks, "ratios": ratios}


def number(value, decimals=0):
    return f"{value:,.{decimals}f}"


def growth(now, prior):
    return (now / prior - 1) * 100 if prior else None


def make_charts(profile, groups, analysis, directory):
    # ReportLab vector charts avoid an optional matplotlib runtime dependency.
    from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, PolyLine
    from reportlab.graphics import renderSVG
    from reportlab.lib.colors import HexColor
    navy, teal, gold, muted = [HexColor(c) for c in ("#142D4E", "#008A8B", "#D69238", "#53657A")]
    paths = {}
    def text(d, x, y, value, size=10, color=None, anchor="start"):
        d.add(String(x, y, str(value), fontName="Helvetica", fontSize=size,
                     fillColor=color or navy, textAnchor=anchor))
    def save(name, drawing):
        svg = directory / f"{name}.svg"
        svg.write_text(renderSVG.drawToString(drawing), encoding="utf-8")
        path = directory / f"{name}.png"
        with pymupdf.open(stream=svg.read_bytes(), filetype="svg") as vector:
            with pymupdf.open(stream=vector.convert_to_pdf(), filetype="pdf") as document:
                document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(path)
        paths[name] = path
        emit("chart.generated", chart=name, vector_path=str(svg.resolve()), **artifact(path))
    years = list(reversed(profile["years"]))
    def grouped(name, series, maximum):
        d=Drawing(660, 215)
        left, bottom, height, width = 55, 35, 140, 570
        text(d, left, 199, profile["currency"] + " billions", 10, muted)
        for tick in range(5):
            value=maximum*tick/4; y=bottom+height*tick/4
            d.add(Line(left,y,left+width,y,strokeColor=HexColor("#E5ECF1"),strokeWidth=.6))
            text(d,left-9,y-3,f"{value:g}",9,muted,"end")
        for k,(label,values,color) in enumerate(series):
            for i,value in enumerate(values):
                x=left+48+i*190+k*30; h=float(value)/1000/maximum*height
                d.add(Rect(x,bottom,25,h,fillColor=color,strokeColor=None))
                text(d,x+12.5,bottom+h+5,f"{float(value)/1000:.1f}",9,navy,"middle")
            lx=left+k*185
            d.add(Rect(lx,184,9,9,fillColor=color,strokeColor=None));text(d,lx+14,184,label,9)
        for i,year in enumerate(years):
            text(d,left+48+i*190+(len(series)-1)*15+12,17,year,10,muted,"middle")
        save(name,d)
    grouped("earnings", [("Revenue",list(reversed(groups["income"]["revenue"]["values"])),navy),
                         ("Net income",list(reversed(groups["income"]["net_income"]["values"])),teal)], 450)
    for group in ("products","geography"):
        rows=sorted(groups[group].items(),key=lambda item:item[1]["values"][0],reverse=True)
        d=Drawing(660,240);maximum=max(float(row["values"][0]) for _,row in rows)*1.12
        text(d,220,222,f"FY{profile['year']} revenue, {profile['currency']} billions",10,muted)
        for i,(key,row) in enumerate(rows):
            y=184-i*35; length=float(row["values"][0])/maximum*370
            text(d,208,y+6,key,10,navy,"end")
            d.add(Rect(220,y,length,22,fillColor=teal if key=="Services" else navy,strokeColor=None))
            text(d,225+length,y+6,f"{float(row['values'][0])/1000:.1f}",10)
        save(group,d)
    d=Drawing(660,220)
    left,bottom,width,height=55,35,570,135
    for tick in (20,30,40,50):
        y=bottom+(tick-18)/34*height
        d.add(Line(left,y,left+width,y,strokeColor=HexColor("#E5ECF1"),strokeWidth=.6))
        text(d,left-10,y-3,f"{tick}%",9,muted,"end")
    for k,(metric,color) in enumerate([("Gross margin",navy),("Operating margin",teal),("Net margin",gold)]):
        points=[]
        for i,year in enumerate(years):
            value=float(analysis["ratios"][year][metric]);x=left+45+i*235;y=bottom+(value-18)/34*height
            points.extend([x,y]);d.add(Circle(x,y,3,fillColor=color,strokeColor=None))
            text(d,x,y+9,f"{value:.1f}%",9,color,"middle")
        d.add(PolyLine(points,strokeColor=color,strokeWidth=2))
        x=left+k*185;d.add(Rect(x,195,9,9,fillColor=color,strokeColor=None));text(d,x+14,195,metric,10)
    for i,year in enumerate(years):text(d,left+45+i*235,17,year,10,muted,"middle")
    save("margins",d)
    grouped("cashflow",[("Operating cash",list(reversed(groups["cashflow"]["operating_cash"]["values"])),navy),
                        ("Capital expenditure",[-v for v in reversed(groups["cashflow"]["capex"]["values"])],gold),
                        ("Free cash flow",[analysis["ratios"][y]["Free cash flow"] for y in years],teal)],150)
    return paths


def build_sections(profile, groups, analysis):
    y = profile["year"]; ratios = analysis["ratios"]; latest = ratios[y]
    income = groups["income"]; revenue = income["revenue"]["values"]
    sections = []
    def table(group, rows=None, share=False):
        data = groups[group]; rows = rows or list(data)
        years = next(iter(data.values()))["years"]
        headers = ["Financial measure"] + [str(v) for v in years] + ([f"FY{y} mix"] if share else [])
        display = {"diluted_eps": "Diluted EPS (USD/share)", "gross_profit": "Gross profit",
                   "current_securities": "Marketable securities - current", "noncurrent_securities": "Marketable securities - non-current",
                   "current_debt": "Term debt - current", "noncurrent_debt": "Term debt - non-current"}
        result = [headers]
        for key in rows:
            row = data[key]; decimal = 2 if key == "diluted_eps" else 0
            result.append([display.get(key, row["label"])] + [number(v, decimal) for v in row["values"]] +
                          ([number(row["values"][0] / revenue[0] * 100, 1) + "%"] if share else []))
        return result
    notes = {n["title"]: n for n in profile.get("notes", [])}
    def note(title):
        n = notes.get(title)
        return n["text"] + f" [Source PDF p.{n['pdf_page']}]" if n else "Not provided in this source profile."
    sections.append({"title": "Executive financial review", "table": table("income", ["revenue", "gross_profit", "operating_income", "net_income", "diluted_eps"]),
        "paragraphs": [note("Business model"),
            f"Revenue grew {number(growth(revenue[0], revenue[1]), 2)}% year over year. Operating income grew {number(growth(income['operating_income']['values'][0], income['operating_income']['values'][1]), 2)}%, while reported net income changed {number(growth(income['net_income']['values'][0], income['net_income']['values'][1]), 2)}%. [Calculated from source PDF p.{income['revenue']['pdf_page']}]",
            note("Tax comparability"), note("Fiscal comparability")], "chart": "earnings", "sources": sorted({income["revenue"]["pdf_page"], *[n["pdf_page"] for n in profile.get("notes", []) if n["title"] in {"Fiscal comparability", "Tax comparability"}]})})
    margin_table = [["Ratio"] + [str(year) for year in profile["years"]]] + [[name] + [number(ratios[year][name], 2) + "%" for year in profile["years"]] for name in ("Gross margin", "Operating margin", "Net margin", "Effective tax rate")]
    sections.append({"title": "Earnings quality and profitability", "table": table("income", ["cost_of_sales", "research", "sga", "opex", "pretax_income", "tax"]),
        "extra_table": margin_table, "paragraphs": [note("Margin drivers"), "Margins are calculated from unrounded reported amounts. Profit growth and cash generation are assessed separately; no adjusted earnings or forecast is substituted for GAAP results."], "chart": "margins", "sources": sorted({income["revenue"]["pdf_page"], *[n["pdf_page"] for n in profile.get("notes", []) if n["title"] in {"Margin drivers", "Tax comparability"}]})})
    for group, title, relevant_note in [("products", "Product and Services revenue mix", "Services growth"), ("geography", "Geographic revenue and concentration", "Greater China")]:
        changes = [["Category", f"FY{y} YoY growth"]] + [[name, number(growth(row["values"][0], row["values"][1]), 2) + "%"] for name, row in groups[group].items()]
        sections.append({"title": title, "table": table(group, share=True), "extra_table": changes,
                         "paragraphs": [note(relevant_note), "Mix is each category's revenue divided by consolidated revenue. Product categories are distinct from geographic reportable segments."],
                         "chart": group, "sources": [next(iter(groups[group].values()))["pdf_page"]]})
    balance_rows = ["cash", "current_securities", "noncurrent_securities", "current_assets", "assets", "current_liabilities", "liabilities", "equity", "commercial_paper", "current_debt", "noncurrent_debt"]
    balance_metrics = [["Calculated measure"] + [str(year) for year in profile["years"][:2]]] + [[name] + [number(ratios[year][name], 2 if name in {"Current ratio", "Debt / equity"} else 0) + ("x" if name in {"Current ratio", "Debt / equity"} else "") for year in profile["years"][:2]] for name in ("Current ratio", "Working capital", "Debt carrying value", "Cash plus securities", "Debt / equity")]
    sections.append({"title": "Balance sheet, liquidity and leverage", "table": table("balance", balance_rows), "extra_table": balance_metrics,
        "paragraphs": [f"Working capital was {number(latest['Working capital'])} million and the current ratio was {number(latest['Current ratio'], 2)}x. This is a snapshot, not proof of a liquidity crisis or assurance of future solvency.",
            note("Restricted funds"), "Debt is commercial paper plus current and non-current term debt at carrying value; lease liabilities are excluded. Cash plus securities includes current and non-current marketable securities; it is not presented as unrestricted cash."], "sources": sorted({groups["balance"]["assets"]["pdf_page"], *[n["pdf_page"] for n in profile.get("notes", []) if n["title"] == "Restricted funds"]})})
    cf_table = table("cashflow", ["operating_cash", "capex", "investing_cash", "dividends", "buybacks", "financing_cash"])
    cf_table.append(["Free cash flow (calculated)"] + [number(ratios[year]["Free cash flow"]) for year in profile["years"]])
    sections.append({"title": "Cash flow and capital allocation", "table": cf_table,
        "paragraphs": [f"Free cash flow was {number(latest['Free cash flow'])} million, calculated as operating cash flow less cash capital expenditure. Operating cash flow / net income was {number(latest['Cash conversion'], 2)}x. Free cash flow is a report-defined non-GAAP measure, not a separate GAAP statement line.",
                       note("Capital allocation"), "Negative amounts in this table denote cash outflows. Repurchases are the cash-flow statement amounts; authorizations and declared dividends are not treated as cash paid."], "chart": "cashflow", "sources": sorted({groups["cashflow"]["operating_cash"]["pdf_page"], *[n["pdf_page"] for n in profile.get("notes", []) if n["title"] == "Capital allocation"]})})
    watchpoints = [["Historical financial watchpoint", f"FY{y}"],
                  ["Net income growth vs prior year", number(growth(income["net_income"]["values"][0], income["net_income"]["values"][1]), 2) + "%"],
                  ["Change in effective tax rate", number(latest["Effective tax rate"] - ratios[y-1]["Effective tax rate"], 2) + " percentage points"],
                  ["Current assets / current liabilities", number(latest["Current ratio"], 2) + "x"],
                  ["Cash dividends + buybacks / free cash flow", number(-(groups["cashflow"]["dividends"]["values"][0] + groups["cashflow"]["buybacks"]["values"][0]) / latest["Free cash flow"] * 100, 2) + "%"]]
    sections.append({"title": "Risk, accounting and interpretation", "table": watchpoints, "paragraphs": [note("Risk context"), note("Fiscal comparability"), note("Tax comparability"), note("Restricted funds"),
        "This report describes historical performance through the stated fiscal year-end. It includes no live share price, valuation multiple, price target, investment recommendation or unsupported projection. Missing or ambiguous values halt numerical report generation rather than being filled by a language model."], "sources": sorted({n["pdf_page"] for n in profile.get("notes", []) if n["title"] in {"Risk context", "Fiscal comparability", "Tax comparability", "Restricted funds"}})})
    return sections


def markdown_report(profile, sections, charts, trace, analysis):
    lines = [f"# {profile['company']} - FY{profile['year']} financial report", "",
             f"Fiscal year ended {profile['period_end']}. Historical figures in {profile['currency']} {profile['unit']}, except EPS and ratios.", "",
             f"Thread ID: `{trace.thread_id}`  \nTrace ID: `{trace.trace_id}`", "",
             "Source: company annual report already present in the workspace; not a current-period market update.", ""]
    for section in sections:
        lines.extend(["## " + section["title"], ""])
        for key in ("table", "extra_table"):
            if key in section:
                table = section[key]; lines.append("| " + " | ".join(table[0]) + " |")
                lines.append("| " + " | ".join("---" for _ in table[0]) + " |")
                lines.extend("| " + " | ".join(row) + " |" for row in table[1:]); lines.append("")
        lines.extend(p + "\n" for p in section["paragraphs"])
        if section.get("chart"):
            lines.extend([f"![{section['title']}](charts/{charts[section['chart']].name})", ""])
        lines.extend(["Source PDF pages: " + ", ".join(str(p) for p in section["sources"]) + ".", ""])
    lines.extend(["## Calculation definitions", "", "Growth = (current / prior - 1) x 100. Margin = relevant profit / revenue x 100. Effective tax rate = income-tax provision / pretax income x 100. Current ratio = current assets / current liabilities. Working capital = current assets - current liabilities. Debt/equity = defined debt carrying value / equity. Cash conversion = operating cash / net income. Free cash flow = operating cash - absolute cash capital expenditure.", "",
                  f"All {len(analysis['checks'])} source arithmetic checks passed. These checks do not constitute an audit opinion.", "",
                  "## Source and audit trail", "", f"Annual report: {profile['source_url']}", "", f"Source SHA-256: `{profile['source_sha256']}`", "",
                  "PDF page numbers are one-based file pages; printed Form 10-K page numbers may differ. The JSON evidence file preserves every extracted row, source page and exact source passage. Generated chart values use the same validated dataset as the tables.", ""])
    return "\n".join(lines)


def create_pdf(path, profile, sections, charts, trace, analysis):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    navy, teal, muted = colors.HexColor("#142D4E"), colors.HexColor("#008A8B"), colors.HexColor("#53657A")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=navy, spaceAfter=13))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.2, leading=13.5, textColor=navy, spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="Helvetica", fontSize=8, leading=11, textColor=muted, spaceAfter=7))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8.4, leading=11, textColor=navy))
    story = []
    def para(text, style="ReportBody"):
        return Paragraph(escape(text), styles[style])
    def table(rows):
        width = 7.1 * inch
        columns = len(rows[0]); widths = [width * .43] + [width * .57 / (columns - 1)] * (columns - 1)
        numeric = ParagraphStyle(name="NumberCell", parent=styles["Cell"], alignment=TA_RIGHT)
        header_left = ParagraphStyle(name="HeaderLeft", parent=styles["Cell"], textColor=colors.white, fontName="Helvetica-Bold")
        header_right = ParagraphStyle(name="HeaderRight", parent=header_left, alignment=TA_RIGHT)
        data = [[Paragraph(escape(cell), header_left if col == 0 else header_right) for col, cell in enumerate(rows[0])]]
        data += [[Paragraph(escape(cell), styles["Cell"] if col == 0 else numeric) for col, cell in enumerate(row)] for row in rows[1:]]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F5F8")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                               ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#D3DFE7"))]))
        return t
    for index, section in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.append(para(f"{profile['ticker']} / FY{profile['year']} / {index+1:02d}", "ReportSmall"))
        story.append(para(section["title"] if index else profile["company"] + "\nFinancial review", "ReportTitle"))
        if index == 0:
            story.append(para(f"Fiscal year ended {profile['period_end']} | FY{profile['years'][-1]}-FY{profile['year']} comparison", "ReportBody"))
        story.append(para(f"{profile['currency']} millions, except EPS, percentages and ratios. Historical source: FY{profile['year']} annual report.", "ReportSmall"))
        for key in ("table", "extra_table"):
            if key in section:
                story.extend([table(section[key]), Spacer(1, 10)])
        if section.get("chart"):
            image = Image(str(charts[section["chart"]]))
            image.drawHeight *= (7.1 * inch / image.drawWidth)
            image.drawWidth = 7.1 * inch
            story.extend([image, Spacer(1, 8)])
        for text in section["paragraphs"]:
            story.append(para(text))
        story.append(para("Source PDF pages: " + ", ".join(map(str, section["sources"])) + ".", "ReportSmall"))
    story.extend([PageBreak(), para("Methodology and audit trail", "ReportTitle")])
    for text in [f"Company: {profile['company']} ({profile['ticker']}). Reporting period: {profile['period_end']}.",
                 f"Reported statement amounts are {profile['currency']} millions. Diluted EPS is {profile['currency']} per share. Negative cash-flow amounts are outflows. Ratios and growth rates are calculated from the underlying unrounded amounts, then rounded for display.",
                 "Growth = (current / prior - 1) x 100. Margins = profit / revenue x 100. Effective tax rate = tax provision / pretax income x 100. Current ratio = current assets / current liabilities. Working capital = current assets - current liabilities.",
                 "Free cash flow = operating cash flow - absolute capital expenditure. Debt carrying value = commercial paper + current term debt + non-current term debt. Debt/equity excludes leases. Cash conversion = operating cash / net income.",
                 f"All {len(analysis['checks'])} arithmetic reconciliation checks passed: gross profit, operating profit, net income, product totals, geographic totals and the balance-sheet equation. The generated report is not an independent audit.",
                 "Source page references use one-based PDF file pages. See each section for its statement page and evidence.json for every row. Printed annual-report page numbers may differ.",
                 "No language model supplies the statement numbers or performs the calculations. The standard run logs the LLM stage as skipped. Optional RAG evidence, if requested, is a separate supplement and cannot overwrite the validated tables.",
                 f"This report uses the specified FY{profile['year']} source from the local corpus. It does not assert that this is the latest publicly available filing.",
                 "Source: " + profile["source_url"], "Source SHA-256: " + profile["source_sha256"],
                 "Thread ID: " + trace.thread_id, "Trace ID: " + trace.trace_id,
                 "The companion Markdown, evidence JSON, chart files, manifest and JSONL logs share this trace ID. A successful run is not a promise that future tools, models or data sources cannot fail."]:
        story.append(para(text))
    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(teal); canvas.setLineWidth(1.2)
        canvas.line(42, 755, 554, 755)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(muted)
        canvas.drawString(42, 27, f"{profile['ticker']} FY{profile['year']} | Historical financial analysis")
        canvas.drawRightString(554, 27, f"{trace.trace_id[:8]} | {doc.page}")
        canvas.restoreState()
    temp = path.with_name(path.name + ".tmp")
    doc = SimpleDocTemplate(str(temp), pagesize=(612, 792), rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=45,
                            title=f"{profile['company']} FY{profile['year']} financial report", author="Source-grounded financial report workflow")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    temp.replace(path)


def generate(profile_path, *, output_root=None, thread_id=None, provider=None, rag=None):
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if profile["unit"] != "millions" or profile["years"] != [profile["year"] - i for i in range(3)]:
        raise ValueError("Profile requires three descending consecutive fiscal years and amounts in millions")
    if not re.fullmatch(r"[A-Z0-9.-]{1,16}", profile["ticker"]):
        raise ValueError("Invalid ticker")
    identifier = str(uuid4())
    directory = Path(output_root or ROOT / "output/pdf") / f"{profile['ticker']}_{profile['year']}" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    trace = TraceLog(directory / "logs", thread_id=thread_id, trace_id=identifier)
    manifest = {"status": "started", "thread_id": trace.thread_id, "trace_id": trace.trace_id,
                "company": profile["company"], "year": profile["year"], "log_path": str(trace.path.resolve())}
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    with trace.activate():
        emit("report.started", company=profile["company"], ticker=profile["ticker"], fiscal_year=profile["year"],
             generated_at_ist=datetime.now(ZoneInfo("Asia/Kolkata")).isoformat())
        try:
            with trace.span("source_validation_agent"):
                source = ROOT / profile["source"]
                source_info = artifact(source)
                if source_info["sha256"] != profile["source_sha256"]:
                    raise ValueError("Source checksum mismatch")
                emit("source.validated", **source_info)
            with trace.span("financial_retrieval_agent"):
                groups, notes = retrieve_financials(profile, source)
                write_json(directory / "evidence.json", {"source": source_info, "groups": groups, "notes": notes})
            with trace.span("reconciliation_and_analysis_agent"):
                analysis = analyze(groups)
                write_json(directory / "analysis.json", analysis)
                sections = build_sections(profile, groups, analysis)
            with trace.span("rag_supplement_agent"):
                if provider:
                    try:
                        if rag is None:
                            raise ValueError("A configured RAG instance is required for the supplement")
                        result = rag.answer("financial_report_qa", "What financial risks and liquidity constraints are disclosed?",
                                            provider=provider, ticker=profile["ticker"], year=profile["year"],
                                            timeout_seconds=60, idempotency_key=identifier)
                        if not isinstance(result, dict) or not isinstance(result.get("evidence", []), list):
                            raise ValueError("Invalid RAG supplement result")
                    except Exception as exc:
                        result = {"final_status": "tool_error", "evidence": [], "error_type": type(exc).__name__,
                                  "trace_id": trace.trace_id}
                        emit("rag.supplement_failed", error_type=type(exc).__name__, action="continue_validated_numeric_report")
                    write_json(directory / "rag_supplement.json", result)
                    emit("rag.supplement", final_status=result.get("final_status"), evidence_count=len(result.get("evidence", [])))
                else:
                    emit("llm.skipped", reason="deterministic_numeric_report; no model needed")
            with trace.span("chart_agent"):
                chart_dir = directory / "charts"; chart_dir.mkdir()
                charts = make_charts(profile, groups, analysis, chart_dir)
            with trace.span("report_writer_agent"):
                stem = f"{profile['ticker'].lower()}_fy{profile['year']}_financial_report"
                md, pdf = directory / f"{stem}.md", directory / f"{stem}.pdf"
                atomic_text(md, markdown_report(profile, sections, charts, trace, analysis))
                create_pdf(pdf, profile, sections, charts, trace, analysis)
                emit("report.file_written", **artifact(pdf))
            with trace.span("artifact_validation_agent"):
                previews = directory / "preview"; previews.mkdir()
                with pymupdf.open(pdf) as document:
                    if len(document) < 8 or any(not page.get_text().strip() for page in document):
                        raise ValueError("Report contains missing or blank pages")
                    for i, page in enumerate(document):
                        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).save(previews / f"page_{i+1:02}.png")
                    page_count = len(document)
                emit("artifact.validated", pdf_pages=page_count, visual_review="pending", **artifact(pdf))
            manifest.update(status="generated_pending_visual_review", pdf_pages=page_count,
                            artifacts=[artifact(p) for p in [pdf, md, directory / "evidence.json", directory / "analysis.json",
                                                            *sorted(chart_dir.glob("*")),
                                                            *sorted(directory.glob("rag_supplement.json"))]],
                            checks_passed=len(analysis["checks"]), visual_review="pending")
            write_json(manifest_path, manifest)
            emit("report.generated", manifest=str(manifest_path.resolve()), status=manifest["status"])
        except Exception as exc:
            manifest.update(status="failed", error_type=type(exc).__name__)
            write_json(manifest_path, manifest)
            emit("report.failed", error_type=type(exc).__name__)
            raise
    return manifest_path


def record_visual_review(manifest_path):
    """Record an external render review, only after a person/agent inspected every page."""
    import threading
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "generated_pending_visual_review":
        raise ValueError("Only a generated report awaiting review can be finalized")
    for item in manifest["artifacts"]:
        if artifact(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Artifact changed since generation")
    if len(list((manifest_path.parent / "preview").glob("page_*.png"))) != manifest["pdf_pages"]:
        raise ValueError("Missing render previews")
    # Resume the single-writer local log after the generator process has exited.
    log_path = Path(manifest["log_path"])
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if any(r["trace_id"] != manifest["trace_id"] or r["thread_id"] != manifest["thread_id"] for r in rows):
        raise ValueError("Trace identity mismatch")
    sequence = max(r["sequence"] for r in rows)
    with log_path.open("a", encoding="utf-8") as stream:
        for index, event in enumerate(("artifact.visual_review_passed", "report.completed"), 1):
            row = {"timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
                   "sequence": sequence + index, "event": event, "trace_id": manifest["trace_id"],
                   "thread_id": manifest["thread_id"], "span_id": None, "worker_thread_id": threading.get_ident(),
                   "review": "all_rendered_pages_inspected", "pdf_pages": manifest["pdf_pages"]}
            stream.write(json.dumps(row) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    manifest.update(status="completed", visual_review="passed")
    write_json(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/financial_reports/apple_2024.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thread-id")
    parser.add_argument("--provider", choices=["extractive", "ollama", "llamacpp", "transformers"])
    parser.add_argument("--record-visual-review", type=Path,
                        help="Record an already completed inspection of all rendered pages")
    args = parser.parse_args()
    if args.record_visual_review:
        print(record_visual_review(args.record_visual_review).resolve())
        return
    if args.provider:
        from scripts.rag.multimodal_rag import LocalRAG
        rag = LocalRAG()
        try:
            path = generate(args.profile, output_root=args.output, thread_id=args.thread_id, provider=args.provider, rag=rag)
        finally:
            rag.close()
    else:
        path = generate(args.profile, output_root=args.output, thread_id=args.thread_id)
    print(path.resolve())


if __name__ == "__main__":
    main()
