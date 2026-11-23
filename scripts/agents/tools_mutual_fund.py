"""Fund tools of the agent (design section 6.1): ``fund_lookup`` runs fixed, parameterised SQL over
``nport_mutual_fund.sqlite3`` and returns rows as citable evidence; ``company_figures`` caches verified company numbers.

Never free-form SQL: every query here is fixed and takes only bound parameters chosen by code (series, accession,
ticker, a row limit). Each returned row is an evidence item with a stable ID (``nport_<accession digits>_<row>``), the
row as text and ``source_type: nport``, so the critic's citation check, figure check and support judge work on it
unchanged. Percentages are of the fund's net assets, as N-PORT reports them.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.common.companies import ALIASES, NAMES, SECTOR_OF
from scripts.common.paths import ROOT
from scripts.rag import nport_mutual_fund as nport

SOURCE_TYPE = "nport"
MANIFEST_GLOBS = ("data/raw/financial_report_qa/annual_reports/*/pdf_manifest.jsonl",
                  "data/raw/financial_report_qa/edgar/pdf_manifest.jsonl")
GENERIC_NAME_WORDS = {"fund", "funds", "index", "the", "trust", "portfolio", "shares", "etf", "inc", "series", "investor",
                      "admiral", "institutional", "class", "of", "and", "&", "s"}


# Figures cached by an older row reader are not reused: reader 1 cached Walmart's net sales (706,413) as its revenue
# (713,163) and one column for two years of JPMorgan.
READER_VERSION = 2


def usd_millions(value: float | None) -> str:
    return "n/a" if value is None else f"USD {value / 1e6:,.1f} million"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def months_of(period_end: str) -> list[str]:
    """The three months an N-PORT report covers, oldest first: '2026-06-30' -> ['April 2026', 'May 2026', 'June 2026']."""
    try:
        end = datetime.strptime(period_end, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return ["month 1", "month 2", "month 3"]
    out = []
    for back in (2, 1, 0):
        month, year = end.month - back, end.year
        if month <= 0:
            month, year = month + 12, year - 1
        out.append(date(year, month, 1).strftime("%B %Y"))
    return out


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9&]+", (text or "").lower().replace("s&p", "sp"))


class FundStore:
    """Read-only access to the fund database (and the company-figure cache, written by the fund agent)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else nport.default_db()

    def available(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with self.read() as db:
                return db.execute("SELECT COUNT(*) FROM filings").fetchone()[0] > 0
        except sqlite3.Error:
            return False

    def read(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    # ------------------------------------------------------------------ which fund
    def funds(self) -> list[dict[str, Any]]:
        with self.read() as db:
            rows = db.execute("SELECT f.series_id, f.series_name, f.family, f.registrant, f.cik, g.accession, g.period_end, "
                              "g.net_assets, g.filing_date FROM funds f JOIN filings g USING (series_id) "
                              "ORDER BY f.series_name, g.period_end").fetchall()
        out: dict[str, dict] = {}
        for series_id, name, family, registrant, cik, accession, period_end, net_assets, filing_date in rows:
            fund = out.setdefault(series_id, {"series_id": series_id, "series_name": name, "family": family,
                                              "registrant": registrant, "cik": cik, "filings": []})
            fund["filings"].append({"accession": accession, "period_end": period_end, "net_assets": net_assets,
                                    "filing_date": filing_date})
        return list(out.values())

    def resolve(self, query: str) -> dict[str, Any]:
        """Funds the question names. A fund matches when the question has every distinctive word of its name, or
        the family plus at least one distinctive word ('Vanguard 500' -> Vanguard 500 Index Fund). Only the family
        ('Vanguard fund') makes every fund of the family a candidate, so the agent can ask which one."""
        words = set(tokens(query))
        fund_word = bool(words & {"fund", "funds", "etf", "etfs", "index", "portfolio"})
        matches, family_only = [], []
        for fund in self.funds():
            family = set(tokens(fund["family"]))
            name = [w for w in tokens(fund["series_name"]) if w not in GENERIC_NAME_WORDS]
            distinctive = [w for w in name if w not in family]
            hit = [w for w in distinctive if w in words]
            # A bare number ('500 stores') is not a fund: the full name also needs the family or a fund word.
            if distinctive and len(hit) == len(distinctive) and (fund_word or (family and family <= words)):
                matches.append((len(hit) + 1, fund))
            elif hit and family and family <= words:
                matches.append((len(hit), fund))
            elif family and family <= words:
                family_only.append(fund)
        matches.sort(key=lambda m: -m[0])
        best = [f for score, f in matches if score == matches[0][0]] if matches else []
        return {"matches": best, "family_only": family_only}

    # ------------------------------------------------------------------ fixed queries
    def filing(self, series_id: str, year: int | None = None) -> dict[str, Any] | None:
        """The fund's latest filing (in ``year`` when given)."""
        sql = ("SELECT g.accession, g.period_end, g.filing_date, g.net_assets, g.total_assets, g.total_liabilities, "
               "g.holdings_count, g.form, f.series_name, f.registrant, f.family, f.cik, f.series_id FROM filings g "
               "JOIN funds f USING (series_id) WHERE g.series_id = ?")
        params: list[Any] = [series_id]
        if year:
            sql += " AND substr(g.period_end, 1, 4) = ?"
            params.append(str(year))
        with self.read() as db:
            row = db.execute(sql + " ORDER BY g.period_end DESC, g.filing_date DESC LIMIT 1", params).fetchone()
        keys = ("accession", "period_end", "filing_date", "net_assets", "total_assets", "total_liabilities",
                "holdings_count", "form", "series_name", "registrant", "family", "cik", "series_id")
        return dict(zip(keys, row)) if row else None

    def top_holdings(self, accession: str, limit: int = 10) -> list[dict[str, Any]]:
        """Equity holdings, long, one row per company (share classes summed): mapped companies by ticker, the rest
        by issuer CUSIP-6 (or name)."""
        with self.read() as db:
            rows = db.execute(
                "SELECT COALESCE(l.corpus_ticker, NULLIF(substr(h.cusip, 1, 6), '000000'), h.issuer_name) AS company, "
                "l.corpus_ticker, MIN(h.issuer_name), SUM(h.pct_of_net_assets) AS p, SUM(h.value_usd), COUNT(*), "
                "MIN(CAST(h.holding_id AS INTEGER)), GROUP_CONCAT(DISTINCT l.method) "
                "FROM holdings h LEFT JOIN holding_link l USING (accession, holding_id) WHERE h.accession = ? "
                "AND h.asset_cat = 'EC' AND COALESCE(h.payoff, 'Long') = 'Long' GROUP BY company ORDER BY p DESC LIMIT ?",
                (accession, int(limit))).fetchall()
        return [{"rank": i, "ticker": t, "issuer_name": n, "pct_of_net_assets": p, "value_usd": v, "lines": k,
                 "holding_id": str(h), "method": m} for i, (_, t, n, p, v, k, h, m) in enumerate(rows, start=1)]

    def exposure(self, accession: str, ticker: str) -> dict[str, Any] | None:
        with self.read() as db:
            row = db.execute("SELECT pct_of_net_assets, value_usd, lines FROM fund_exposure WHERE accession = ? "
                             "AND corpus_ticker = ?", (accession, ticker)).fetchone()
        return {"pct_of_net_assets": row[0], "value_usd": row[1], "lines": row[2]} if row else None

    def holders(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Funds holding a company in their latest loaded filing, largest weight first."""
        with self.read() as db:
            rows = db.execute(
                "SELECT f.series_name, e.series_id, e.accession, e.period_end, e.pct_of_net_assets, e.value_usd "
                "FROM fund_exposure e JOIN funds f USING (series_id) WHERE e.corpus_ticker = ? AND e.period_end = "
                "(SELECT MAX(period_end) FROM filings g WHERE g.series_id = e.series_id) "
                "ORDER BY e.pct_of_net_assets DESC LIMIT ?", (ticker, int(limit))).fetchall()
        return [{"series_name": n, "series_id": s, "accession": a, "period_end": d, "pct_of_net_assets": p,
                 "value_usd": v} for n, s, a, d, p, v in rows]

    def flows(self, accession: str) -> list[dict[str, Any]]:
        with self.read() as db:
            rows = db.execute("SELECT month, sales, reinvestments, redemptions FROM flows WHERE accession = ? "
                              "ORDER BY month", (accession,)).fetchall()
        return [{"month": m, "sales": s, "reinvestments": r, "redemptions": d} for m, s, r, d in rows]

    def returns(self, accession: str) -> dict[str, list[tuple[int, float]]]:
        with self.read() as db:
            rows = db.execute("SELECT class_id, month, total_return FROM monthly_returns WHERE accession = ? "
                              "ORDER BY class_id, month", (accession,)).fetchall()
        out: dict[str, list] = defaultdict(list)
        for class_id, month, value in rows:
            out[class_id].append((month, value))
        return dict(out)

    def sector_mix(self, accession: str) -> list[dict[str, Any]]:
        with self.read() as db:
            return nport.sector_mix(db, accession)

    def coverage(self, accession: str) -> dict[str, Any]:
        with self.read() as db:
            return nport.coverage(db, accession)

    # ------------------------------------------------------------------ company_figures cache
    def cached_figure(self, ticker: str, fiscal_year: int, metric: str) -> dict[str, Any] | None:
        metric = f"{metric} (reader {READER_VERSION})"
        try:
            with self.read() as db:
                row = db.execute("SELECT value, value_text, unit, chunk_id, page, method FROM company_figures WHERE "
                                 "ticker = ? AND fiscal_year = ? AND metric = ?", (ticker, fiscal_year, metric)).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        return {"year": fiscal_year, "value": row[0], "value_text": row[1], "unit": row[2], "chunk_id": row[3],
                "page": row[4], "method": row[5], "status": "ok", "cached": True}

    def save_figure(self, ticker: str, fiscal_year: int, metric: str, point: dict[str, Any]) -> bool:
        """Cache a verified figure (status ok, with its chunk). A read-only database is not an error."""
        if point.get("status") != "ok" or point.get("value") is None or not point.get("chunk_id"):
            return False
        metric = f"{metric} (reader {READER_VERSION})"
        try:
            with sqlite3.connect(self.path) as db:
                db.execute("INSERT OR REPLACE INTO company_figures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (ticker, fiscal_year, metric, point["value"], point.get("value_text"), point.get("unit"),
                            point["chunk_id"], point.get("page"), point.get("method"),
                            datetime.now(timezone.utc).isoformat(timespec="seconds")))
            return True
        except sqlite3.Error:
            return False


# ---------------------------------------------------------------------------- annual report available at a date
@lru_cache(maxsize=1)
def manifest_filings() -> dict[str, list[tuple[str, int, str]]]:
    """ticker -> [(filing_date, fiscal year, doc_id)] from the corpus manifests."""
    out: dict[str, list] = defaultdict(list)
    for pattern in MANIFEST_GLOBS:
        for path in ROOT.glob(pattern):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ticker") and row.get("year") and row.get("filing_date"):
                    out[str(row["ticker"]).upper()].append((str(row["filing_date"]), int(row["year"]),
                                                            str(row.get("document_id") or "")))
    return dict(out)


def report_for(ticker: str, holdings_date: str, indexed_years: list[int] | None = None) -> dict[str, Any] | None:
    """The company's latest annual report filed on or before the holdings date (design 4.2: no look-ahead).
    Without a manifest row (a server holding only an index) the latest indexed fiscal year before the holdings year."""
    rows = [r for r in manifest_filings().get(ticker, []) if r[0] <= holdings_date]
    if indexed_years is not None:
        rows = [r for r in rows if r[1] in indexed_years]
    if rows:
        filing_date, year, doc_id = max(rows)
        return {"year": year, "filing_date": filing_date, "doc_id": doc_id, "method": "manifest"}
    years = [y for y in (indexed_years or []) if y < int(holdings_date[:4])]
    if years and not manifest_filings().get(ticker):
        return {"year": max(years), "filing_date": None, "doc_id": f"edgar_ars_{ticker}_{max(years)}",
                "method": "indexed_year_before_holdings"}
    return None


# ---------------------------------------------------------------------------- rows as evidence
def evidence(filing: dict[str, Any], row: str, text: str, ticker: str | None = None) -> dict[str, Any]:
    accession = filing["accession"]
    chunk_id = f"nport_{nport.compact(accession)}_{row}"
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(filing['cik'])}/{nport.compact(accession)}/primary_doc.xml"
           if str(filing.get("cik") or "").strip().isdigit() else None)
    title = f"{filing['series_name'].title()} Form N-PORT, holdings as of {filing['period_end']}"
    return {"chunk_id": chunk_id, "text": text, "preview": text[:240], "source_type": SOURCE_TYPE, "ticker": ticker,
            "year": int(str(filing["period_end"])[:4]), "page": None, "title": title, "source_url": url,
            "doc_id": f"nport_{nport.compact(accession)}", "metadata": {"chunk_id": chunk_id, "accession": accession,
                                                                         "series_id": filing.get("series_id")}}


def fund_name(filing: dict[str, Any]) -> str:
    return filing["series_name"].title()


def facts_row(filing: dict[str, Any]) -> dict[str, Any]:
    text = (f"{fund_name(filing)} (series {filing['series_id']}), Form {filing.get('form') or 'NPORT-P'} with holdings "
            f"as of {filing['period_end']}, accession {filing['accession']}"
            + (f", filed {filing['filing_date']}" if filing.get("filing_date") else "") +
            f": net assets {usd_millions(filing['net_assets'])} ({(filing['net_assets'] or 0) / 1e9:,.1f} billion); "
            f"total assets {usd_millions(filing['total_assets'])}; total liabilities "
            f"{usd_millions(filing['total_liabilities'])}; {filing['holdings_count']} holdings reported.")
    return evidence(filing, "fund", text)


def holding_rows(filing: dict[str, Any], holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for h in holdings:
        who = f"{NAMES[h['ticker']]} ({h['ticker']})" if h.get("ticker") in NAMES else f"{h['issuer_name']} (not in the corpus)"
        text = (f"{fund_name(filing)}, holdings as of {filing['period_end']}: rank {h['rank']} equity holding {who}, "
                f"{pct(h['pct_of_net_assets'])} of net assets, value {usd_millions(h['value_usd'])}"
                + (f", {h['lines']} share classes or lines summed" if h["lines"] > 1 else "") + ".")
        rows.append(evidence(filing, f"h{h['holding_id']}", text, h.get("ticker")))
    return rows


def coverage_row(filing: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    text = (f"{fund_name(filing)}, holdings as of {filing['period_end']}: equity holdings are "
            f"{pct(info['equity_pct_of_net_assets'])} of net assets; {info['corpus_companies']} of them are companies "
            f"with annual reports in the corpus, together {pct(info['corpus_pct_of_net_assets'])} of net assets.")
    return evidence(filing, "coverage", text)


def sector_rows(filing: dict[str, Any], mix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in mix:
        largest = ", ".join(f"{NAMES.get(t, t)} {pct(p)}" for t, p in r["top"])
        text = (f"{fund_name(filing)}, holdings as of {filing['period_end']}: sector {r['sector']} is "
                f"{pct(r['pct_of_net_assets'])} of net assets across {r['companies']} companies"
                + (f" (largest: {largest})" if largest else "") + ".")
        rows.append(evidence(filing, f"sector_{slug(r['sector'])}", text))
    return rows


def flow_row(filing: dict[str, Any], flows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not flows:
        return None
    names = months_of(filing["period_end"])
    parts = [f"{names[f['month'] - 1]}: sales {usd_millions(f['sales'])}, reinvestments {usd_millions(f['reinvestments'])}, "
             f"redemptions {usd_millions(f['redemptions'])}" for f in flows]
    return evidence(filing, "flows", f"{fund_name(filing)} monthly flows for the quarter ending {filing['period_end']}: "
                    + "; ".join(parts) + ".")


def return_rows(filing: dict[str, Any], returns: dict[str, list[tuple[int, float]]]) -> list[dict[str, Any]]:
    names = months_of(filing["period_end"])
    return [evidence(filing, f"returns_{slug(class_id)}", f"{fund_name(filing)} share class {class_id} monthly total "
                     "returns: " + ", ".join(f"{names[m - 1]} {value:.2f}%" for m, value in values) + ".")
            for class_id, values in returns.items()]


def exposure_row(filing: dict[str, Any], ticker: str, found: dict[str, Any] | None) -> dict[str, Any]:
    company = f"{NAMES.get(ticker, ticker)} ({ticker})"
    if not found:
        text = (f"{fund_name(filing)}, holdings as of {filing['period_end']}: no equity holding of {company} is "
                "reported in this filing.")
    else:
        text = (f"{fund_name(filing)}, holdings as of {filing['period_end']}: {company} is "
                f"{pct(found['pct_of_net_assets'])} of net assets, value {usd_millions(found['value_usd'])}"
                + (f", {found['lines']} share classes or lines summed" if found["lines"] > 1 else "") + ".")
    return evidence(filing, f"x_{slug(ticker)}", text, ticker)


def holder_rows(ticker: str, holders: list[dict[str, Any]], store: FundStore) -> list[dict[str, Any]]:
    rows = []
    for rank, h in enumerate(holders, start=1):
        filing = store.filing(h["series_id"]) or {}
        filing = {**filing, "accession": h["accession"], "period_end": h["period_end"]}
        text = (f"{h['series_name'].title()}, holdings as of {h['period_end']}: {NAMES.get(ticker, ticker)} ({ticker}) is "
                f"{pct(h['pct_of_net_assets'])} of net assets, value {usd_millions(h['value_usd'])} "
                f"(rank {rank} of the loaded funds by weight).")
        rows.append(evidence(filing, f"x_{slug(ticker)}", text, ticker))
    return rows


def alias_words(ticker: str) -> list[str]:
    return [NAMES.get(ticker, "").lower(), *[a.lower() for a in ALIASES.get(ticker, [])]]


def sector_of(ticker: str | None) -> str | None:
    return SECTOR_OF.get(ticker or "")
