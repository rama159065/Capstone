"""Mutual fund holdings (SEC Form N-PORT) -> SQLite ``nport_mutual_fund.sqlite3``, each holding mapped to a corpus company.

Design: ``docs/mutual_fund_rag_design.md`` (sections 4 and 5). Holdings are exact numbers that are summed and ranked,
so they live in SQL tables, never in the vector index; ``cards_mutual_fund`` embeds one short card per fund and period
so search can find a fund by name.

Sources (``download_mutual_fund`` fetches both):

* one ``NPORT-P`` filing's ``primary_doc.xml`` (a few MB) - ``load --xml <file or folder>``;
* the SEC quarterly bulk ZIP (about 440 MB, 30 TSV tables) - ``load --tsv-zip <zip> --series S000002839 ...`` or
  ``--name-like "500 INDEX"`` or ``--top 50``. The ZIP is streamed: the 950 MB holdings table is never held in memory,
  only the rows of the chosen funds' latest filings are kept.

Tables (``SCHEMA``): ``funds``, ``filings``, ``flows``, ``monthly_returns``, ``holdings``, ``security_map`` (identifier ->
corpus ticker, learned), ``holding_link`` (holding -> corpus ticker + method), the ``fund_exposure`` view (equity, long,
share classes summed per company) and ``company_figures`` (verified company numbers cached by the fund agent).

Holdings date: ``filings.period_end`` is the date the holdings are reported for (XML ``repPdDate``, TSV
``REPORT_DATE``). ``REPORT_ENDING_PERIOD`` / ``repPdEnd`` is the fund's fiscal year end and is kept as ``fiscal_year_end``.

Mapping order (method recorded on every link): 1 ticker from the identifiers, 2 issuer CUSIP-6 (all share classes and
securities of one issuer), 3 ISIN, 4 issuer LEI, 5 cleaned issuer name against ``companies.py`` names and aliases
(equity of corporate issuers only; listed for review). Identifiers matched once are learned, so later filings map by
identifier. Unmatched holdings stay unmatched and count as the fund's "not covered" share.

    python -m scripts.rag.nport_mutual_fund load --xml <RAG_HOME>/indexes/mutual_fund/raw_mutual_fund
    python -m scripts.rag.nport_mutual_fund load --tsv-zip 2026q3_nport.zip --name-like "500 INDEX" --top 20
    python -m scripts.rag.nport_mutual_fund status
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from scripts.common.companies import ALIASES, NAMES, SECTOR_OF, normalize_ticker

NS = {"n": "http://www.sec.gov/edgar/nport"}
DB_NAME = "nport_mutual_fund.sqlite3"
RAW_DIR = "raw_mutual_fund"
MANIFEST_NAME = "manifest_mutual_fund.jsonl"
EQUITY = "EC"
NAME_REVIEW_CONFIDENCE = 0.6

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS funds (series_id TEXT PRIMARY KEY, series_name TEXT, registrant TEXT, cik TEXT,
    series_lei TEXT, family TEXT);
CREATE TABLE IF NOT EXISTS filings (accession TEXT PRIMARY KEY, series_id TEXT, period_end TEXT, fiscal_year_end TEXT,
    filing_date TEXT, form TEXT, net_assets REAL, total_assets REAL, total_liabilities REAL, source TEXT,
    holdings_count INTEGER);
CREATE TABLE IF NOT EXISTS flows (accession TEXT, month INTEGER, sales REAL, reinvestments REAL, redemptions REAL,
    PRIMARY KEY (accession, month));
CREATE TABLE IF NOT EXISTS monthly_returns (accession TEXT, class_id TEXT, month INTEGER, total_return REAL,
    PRIMARY KEY (accession, class_id, month));
CREATE TABLE IF NOT EXISTS holdings (accession TEXT, holding_id TEXT, issuer_name TEXT, title TEXT, cusip TEXT,
    isin TEXT, ticker_raw TEXT, lei TEXT, value_usd REAL, pct_of_net_assets REAL, asset_cat TEXT, issuer_type TEXT,
    country TEXT, payoff TEXT, PRIMARY KEY (accession, holding_id));
CREATE TABLE IF NOT EXISTS security_map (kind TEXT, key TEXT, corpus_ticker TEXT, method TEXT, confidence REAL,
    PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS holding_link (accession TEXT, holding_id TEXT, corpus_ticker TEXT, method TEXT,
    PRIMARY KEY (accession, holding_id));
CREATE TABLE IF NOT EXISTS company_figures (ticker TEXT, fiscal_year INTEGER, metric TEXT, value REAL, value_text TEXT,
    unit TEXT, chunk_id TEXT, page INTEGER, method TEXT, verified_at TEXT, PRIMARY KEY (ticker, fiscal_year, metric));
CREATE INDEX IF NOT EXISTS holdings_cusip ON holdings (cusip);
CREATE INDEX IF NOT EXISTS link_ticker ON holding_link (corpus_ticker);
CREATE INDEX IF NOT EXISTS filings_series ON filings (series_id, period_end);
CREATE VIEW IF NOT EXISTS fund_exposure AS
    SELECT f.series_id, f.period_end, f.accession, l.corpus_ticker,
           SUM(h.pct_of_net_assets) AS pct_of_net_assets, SUM(h.value_usd) AS value_usd, COUNT(*) AS lines
    FROM holding_link l JOIN holdings h USING (accession, holding_id) JOIN filings f USING (accession)
    WHERE h.asset_cat = 'EC' AND COALESCE(h.payoff, 'Long') = 'Long' AND l.corpus_ticker IS NOT NULL
    GROUP BY f.accession, l.corpus_ticker;
"""

FAMILIES = ("Vanguard", "Fidelity", "iShares", "SPDR", "Schwab", "American Funds", "T. Rowe Price", "BlackRock",
            "Invesco", "JPMorgan", "Dimensional", "State Street", "Franklin", "Nuveen", "PIMCO", "Columbia", "MFS",
            "Janus Henderson", "Putnam", "Dodge & Cox", "Northern Trust", "Principal", "Hartford", "Lord Abbett",
            "Capital Group", "Goldman Sachs", "Morgan Stanley", "Allspring", "Federated Hermes", "Victory", "Invesco QQQ")
STOP_ALIASES = {"southern", "general", "american", "united", "first", "national", "international", "global"}
LEGAL_WORDS = (r"\b(inc|incorporated|corp|corporation|co|company|cos|ltd|limited|plc|holdings?|group|the|class [a-c]|"
               r"cl [a-c]|new|de|n v|nv|na|sa|ag|com|common|stock|shs|ord|reit|lp|llc)\b")


# ---------------------------------------------------------------------------- small helpers
def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def default_dir() -> Path:
    from scripts.common.storage import layout
    return layout().mutual_fund


def default_db() -> Path:
    return default_dir() / DB_NAME


def to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text not in ("", "N/A", "None") else None
    except (TypeError, ValueError):
        return None


def iso_date(value: Any) -> str | None:
    """'2026-06-30' or '30-JUN-2026' (bulk data sets) -> '2026-06-30'."""
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text or None


def compact(accession: str) -> str:
    """'0000036405-26-000473' -> '000003640526000473': evidence IDs without dashes, so no part reads as a figure."""
    return re.sub(r"\D", "", accession or "")


def family_of(*names: str) -> str:
    text = " ".join(n for n in names if n).lower()
    for family in sorted(FAMILIES, key=len, reverse=True):
        if family.lower() in text:
            return family
    first = next((n for n in names if n), "").split()
    return first[0].title() if first else ""


def norm(text: str | None) -> str:
    text = re.sub(r"/[a-z]{2}\b", " ", (text or "").lower())             # state of incorporation: 'Inc/MD', 'Bancorp/IN'
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(LEGAL_WORDS, " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------- parsing
def parse_xml(data: bytes, accession: str | None = None, filing_date: str | None = None,
              source: str = "xml") -> dict[str, Any]:
    """One ``NPORT-P`` ``primary_doc.xml`` -> fund, filing, flows, monthly returns and holdings."""
    root = ET.fromstring(data)

    def text(path, node=root):
        return (node.findtext(path, namespaces=NS) or "").strip()

    form = text(".//n:headerData/n:submissionType") or "NPORT-P"
    series_id = text(".//n:genInfo/n:seriesId") or text(".//n:seriesClassInfo/n:seriesId")
    registrant = text(".//n:genInfo/n:regName")
    series_name = text(".//n:genInfo/n:seriesName")
    fund = {"series_id": series_id, "series_name": series_name, "registrant": registrant,
            "cik": text(".//n:genInfo/n:regCik"), "series_lei": text(".//n:genInfo/n:seriesLei"),
            "family": family_of(series_name, registrant)}
    filing = {"accession": accession or "", "series_id": series_id, "period_end": iso_date(text(".//n:genInfo/n:repPdDate")),
              "fiscal_year_end": iso_date(text(".//n:genInfo/n:repPdEnd")), "filing_date": iso_date(filing_date),
              "form": form, "net_assets": to_float(text(".//n:fundInfo/n:netAssets")),
              "total_assets": to_float(text(".//n:fundInfo/n:totAssets")),
              "total_liabilities": to_float(text(".//n:fundInfo/n:totLiabs")), "source": source}
    flows = []
    for month in (1, 2, 3):
        node = root.find(f".//n:fundInfo/n:mon{month}Flow", NS)
        if node is not None:
            flows.append({"month": month, "sales": to_float(node.get("sales")),
                          "reinvestments": to_float(node.get("reinvestment")),
                          "redemptions": to_float(node.get("redemption"))})
    returns = []
    for node in root.iterfind(".//n:returnInfo/n:monthlyTotReturns/n:monthlyTotReturn", NS):
        for month in (1, 2, 3):
            value = to_float(node.get(f"rtn{month}"))
            if value is not None:
                returns.append({"class_id": node.get("classId") or "", "month": month, "total_return": value})
    holdings = []
    for index, sec in enumerate(root.iterfind(".//n:invstOrSecs/n:invstOrSec", NS), start=1):
        ids = sec.find("n:identifiers", NS)
        isin = ids.find("n:isin", NS) if ids is not None else None
        ticker = ids.find("n:ticker", NS) if ids is not None else None
        holdings.append({
            "holding_id": str(index), "issuer_name": text("n:name", sec), "title": text("n:title", sec),
            "cusip": text("n:cusip", sec), "isin": isin.get("value") if isin is not None else "",
            "ticker_raw": ticker.get("value") if ticker is not None else "", "lei": text("n:lei", sec),
            "value_usd": to_float(text("n:valUSD", sec)), "pct_of_net_assets": to_float(text("n:pctVal", sec)),
            "asset_cat": text("n:assetCat", sec) or ("OTHER" if sec.find("n:assetConditional", NS) is not None else ""),
            "issuer_type": text("n:issuerCat", sec), "country": text("n:invCountry", sec),
            "payoff": text("n:payoffProfile", sec)})
    return {"fund": fund, "filing": filing, "flows": flows, "returns": returns, "holdings": holdings}


def tsv_rows(archive: zipfile.ZipFile, table: str) -> Iterable[dict[str, str]]:
    """Rows of one bulk-data table, streamed (the holdings table is about 950 MB unpacked)."""
    name = next((n for n in archive.namelist() if n.split("/")[-1].upper() == f"{table}.TSV"), None)
    if name is None:
        raise SystemExit(f"{table}.tsv is not in the archive")
    with archive.open(name) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""),
                            delimiter="\t", quoting=csv.QUOTE_NONE)
        header = [h.strip().upper() for h in next(reader)]
        for row in reader:
            yield dict(zip(header, row))


def select_tsv_filings(archive: zipfile.ZipFile, series: Iterable[str] = (), name_like: Iterable[str] = (),
                       top: int | None = None) -> dict[str, dict]:
    """The latest filing of each chosen fund series: accession -> fund info row (+ submission and registrant)."""
    submissions = {r["ACCESSION_NUMBER"]: r for r in tsv_rows(archive, "SUBMISSION")}
    series = {s.upper() for s in series}
    patterns = [p.lower() for p in name_like]
    latest: dict[str, dict] = {}
    for row in tsv_rows(archive, "FUND_REPORTED_INFO"):
        sub = submissions.get(row["ACCESSION_NUMBER"])
        if not sub or not row.get("SERIES_ID"):
            continue
        chosen = (row["SERIES_ID"].upper() in series or any(p in row.get("SERIES_NAME", "").lower() for p in patterns)
                  or (not series and not patterns))
        if not chosen:
            continue
        date = iso_date(sub.get("REPORT_DATE")) or ""
        current = latest.get(row["SERIES_ID"])
        if current is None or date > current["_date"] or (date == current["_date"] and
                                                           sub.get("FILING_DATE", "") > current["_sub"].get("FILING_DATE", "")):
            latest[row["SERIES_ID"]] = {**row, "_date": date, "_sub": sub}
    rows = sorted(latest.values(), key=lambda r: -(to_float(r.get("NET_ASSETS")) or 0))
    if top:
        rows = rows[:top]
    chosen = {r["ACCESSION_NUMBER"]: r for r in rows}
    for reg in tsv_rows(archive, "REGISTRANT"):
        if reg["ACCESSION_NUMBER"] in chosen:
            chosen[reg["ACCESSION_NUMBER"]]["_reg"] = reg
    return chosen


def parse_tsv_zip(path: Path | str, series: Iterable[str] = (), name_like: Iterable[str] = (),
                  top: int | None = None) -> list[dict[str, Any]]:
    """The chosen funds' latest filings from a quarterly bulk ZIP, in ``parse_xml``'s shape."""
    with zipfile.ZipFile(path) as archive:
        chosen = select_tsv_filings(archive, series, name_like, top)
        parsed: dict[str, dict] = {}
        for accession, row in chosen.items():
            sub, reg = row["_sub"], row.get("_reg") or {}
            fund = {"series_id": row["SERIES_ID"], "series_name": row.get("SERIES_NAME", ""),
                    "registrant": reg.get("REGISTRANT_NAME", ""), "cik": reg.get("CIK", ""),
                    "series_lei": row.get("SERIES_LEI", ""),
                    "family": family_of(row.get("SERIES_NAME", ""), reg.get("REGISTRANT_NAME", ""))}
            filing = {"accession": accession, "series_id": row["SERIES_ID"], "period_end": iso_date(sub.get("REPORT_DATE")),
                      "fiscal_year_end": iso_date(sub.get("REPORT_ENDING_PERIOD")),
                      "filing_date": iso_date(sub.get("FILING_DATE")), "form": sub.get("SUB_TYPE") or "NPORT-P",
                      "net_assets": to_float(row.get("NET_ASSETS")), "total_assets": to_float(row.get("TOTAL_ASSETS")),
                      "total_liabilities": to_float(row.get("TOTAL_LIABILITIES")), "source": f"tsv:{Path(path).name}"}
            flows = [{"month": m, "sales": to_float(row.get(f"SALES_FLOW_MON{m}")),
                      "reinvestments": to_float(row.get(f"REINVESTMENT_FLOW_MON{m}")),
                      "redemptions": to_float(row.get(f"REDEMPTION_FLOW_MON{m}"))} for m in (1, 2, 3)]
            parsed[accession] = {"fund": fund, "filing": filing, "flows": flows, "returns": [], "holdings": []}
        for row in tsv_rows(archive, "MONTHLY_TOTAL_RETURN"):
            if row["ACCESSION_NUMBER"] in parsed:
                for m in (1, 2, 3):
                    value = to_float(row.get(f"MONTHLY_TOTAL_RETURN{m}"))
                    if value is not None:
                        parsed[row["ACCESSION_NUMBER"]]["returns"].append(
                            {"class_id": row.get("CLASS_ID", ""), "month": m, "total_return": value})
        by_holding: dict[str, dict] = {}
        for row in tsv_rows(archive, "FUND_REPORTED_HOLDING"):
            accession = row["ACCESSION_NUMBER"]
            if accession not in parsed:
                continue
            holding = {"holding_id": row["HOLDING_ID"], "issuer_name": row.get("ISSUER_NAME", ""),
                       "title": row.get("ISSUER_TITLE", ""), "cusip": row.get("ISSUER_CUSIP", ""), "isin": "",
                       "ticker_raw": "", "lei": row.get("ISSUER_LEI", ""), "value_usd": to_float(row.get("CURRENCY_VALUE")),
                       "pct_of_net_assets": to_float(row.get("PERCENTAGE")), "asset_cat": row.get("ASSET_CAT", ""),
                       "issuer_type": row.get("ISSUER_TYPE", ""), "country": row.get("INVESTMENT_COUNTRY", ""),
                       "payoff": row.get("PAYOFF_PROFILE", "")}
            parsed[accession]["holdings"].append(holding)
            by_holding[holding["holding_id"]] = holding
        for row in tsv_rows(archive, "IDENTIFIERS"):
            holding = by_holding.get(row.get("HOLDING_ID", ""))
            if holding is not None:
                holding["isin"] = holding["isin"] or row.get("IDENTIFIER_ISIN", "")
                holding["ticker_raw"] = holding["ticker_raw"] or row.get("IDENTIFIER_TICKER", "")
    return list(parsed.values())


# ---------------------------------------------------------------------------- storing
def store(db: sqlite3.Connection, parsed: dict[str, Any]) -> str:
    """Insert (or replace) one filing and its rows. Returns the accession."""
    fund, filing = parsed["fund"], parsed["filing"]
    if not filing.get("accession"):
        raise ValueError("a filing needs its accession number (pass it, or name the file <accession>.xml)")
    if not fund.get("series_id") or not filing.get("period_end"):
        raise ValueError(f"{filing['accession']}: no series ID or report date; not an NPORT-P filing?")
    accession = filing["accession"]
    db.execute("INSERT OR REPLACE INTO funds VALUES (:series_id, :series_name, :registrant, :cik, :series_lei, :family)", fund)
    for table in ("flows", "monthly_returns", "holdings", "holding_link"):
        db.execute(f"DELETE FROM {table} WHERE accession = ?", (accession,))
    db.execute("INSERT OR REPLACE INTO filings VALUES (:accession, :series_id, :period_end, :fiscal_year_end, :filing_date, "
               ":form, :net_assets, :total_assets, :total_liabilities, :source, :holdings_count)",
               {**filing, "holdings_count": len(parsed["holdings"])})
    db.executemany("INSERT OR REPLACE INTO flows VALUES (?, ?, ?, ?, ?)",
                   [(accession, f["month"], f["sales"], f["reinvestments"], f["redemptions"]) for f in parsed["flows"]])
    db.executemany("INSERT OR REPLACE INTO monthly_returns VALUES (?, ?, ?, ?)",
                   [(accession, r["class_id"], r["month"], r["total_return"]) for r in parsed["returns"]])
    db.executemany("INSERT OR REPLACE INTO holdings VALUES (:accession, :holding_id, :issuer_name, :title, :cusip, :isin, "
                   ":ticker_raw, :lei, :value_usd, :pct_of_net_assets, :asset_cat, :issuer_type, :country, :payoff)",
                   [{**h, "accession": accession} for h in parsed["holdings"]])
    return accession


# ---------------------------------------------------------------------------- mapping holdings to corpus companies
# Legal names that differ from every alias: BlackRock's parent was renamed "BlackRock Funding" in its 2024 reorganisation.
NAME_OVERRIDES = {"blackrock funding": "BLK"}


def name_index() -> dict[str, str]:
    by_name: dict[str, str] = dict(NAME_OVERRIDES)
    for ticker, aliases in ALIASES.items():
        for alias in [NAMES.get(ticker, ""), *aliases]:
            if norm(alias):
                by_name.setdefault(norm(alias), ticker)
    return by_name


# Words that may follow or precede a company's short name without making it another company ("Costco Wholesale",
# "Walt Disney"); anything else ("Apple Hospitality", "Portland General Electric", "Antero Midstream") is refused.
GENERIC_WORDS = frozenset("""wholesale international worldwide financial services group bancorp bancshares brands industries
technologies technology systems communications entertainment pharmaceuticals therapeutics motors stores platforms
enterprises partners solutions labs laboratories networks software semiconductor semiconductors products companies
global america american us usa and of mfg manufacturing foods health healthcare insurance bank banks walt interactive
energy resources trust chemicals cruises cruise scientific mexican grill bancorporation corporation""".split())


def match_name(name: str, by_name: dict[str, str]) -> str | None:
    """Exact cleaned name; else a whole-word alias inside it ('costco' in 'costco wholesale') or a 3-letter prefix
    ('tjx cos') whose other words are all generic; else an acronym ('international business machines' -> ibm).
    Common words alone never match ('norfolk southern' is not Southern Company), and a distinguishing word blocks a
    match ('apple hospitality reit' is not Apple)."""
    cleaned = norm(name)
    if not cleaned:
        return None
    if cleaned in by_name:
        return by_name[cleaned]
    words = cleaned.split()
    initials = "".join(w[0] for w in words)
    best = None
    for alias, ticker in by_name.items():
        inside = (len(alias) >= 4 and alias not in STOP_ALIASES
                  and re.search(rf"(^| ){re.escape(alias)}( |$)", cleaned))
        prefix = len(alias) == 3 and cleaned.startswith(alias + " ")
        if (inside or prefix) and not set(cleaned.replace(alias, " ", 1).split()) <= GENERIC_WORDS:
            inside = prefix = False
        acronym = (len(alias) >= 3 and " " not in alias and initials == alias and len(words) >= 3
                   and min(map(len, words)) >= 4)
        if (inside or prefix or acronym) and (best is None or len(alias) > len(best[0])):
            best = (alias, ticker)
    return best[1] if best else None


MISSING = {"", "N/A", "NA", "NONE", "NULL", "0", "000000000", "000000000000"}


def ident(value: str | None) -> str | None:
    """An identifier, or None for the placeholders filers use ('N/A', zeros)."""
    text = (value or "").strip().upper()
    return None if text in MISSING or set(text) <= {"0"} else text


def cusip6(cusip: str | None) -> str | None:
    value = ident(cusip)
    return value[:6] if value and len(value) >= 6 and value[:6] != "000000" else None


def map_holdings(db: sqlite3.Connection, accessions: Iterable[str] | None = None) -> dict[str, int]:
    """Link holdings to corpus tickers: identifiers first (learned in ``security_map``), names last."""
    learned = {(kind, key): ticker for kind, key, ticker in db.execute("SELECT kind, key, corpus_ticker FROM security_map")}
    by_name = name_index()
    where = ""
    params: list[str] = []
    if accessions is not None:
        accessions = list(accessions)
        where = f" WHERE accession IN ({','.join('?' * len(accessions))})"
        params = accessions
    rows = db.execute("SELECT accession, holding_id, issuer_name, cusip, isin, ticker_raw, lei, asset_cat, issuer_type "
                      f"FROM holdings{where}", params).fetchall()

    def learn(kind, key, ticker, method, confidence):
        if key and (kind, key) not in learned:
            learned[(kind, key)] = ticker
            db.execute("INSERT OR IGNORE INTO security_map VALUES (?, ?, ?, ?, ?)", (kind, key, ticker, method, confidence))

    def teach(row, ticker, method, confidence):
        learn("cusip6", cusip6(row[3]), ticker, method, confidence)
        learn("isin", ident(row[4]), ticker, method, confidence)
        learn("lei", ident(row[6]), ticker, method, confidence)

    links: dict[tuple[str, str], tuple[str | None, str]] = {}
    for row in rows:                                          # 1: tickers reported with the holding
        ticker = normalize_ticker(row[5]) if row[5] else ""
        if ticker in NAMES:
            links[(row[0], row[1])] = (ticker, "ticker")
            teach(row, ticker, "ticker", 1.0)
    for row in rows:                                          # 2-4: identifiers learned so far; 5: names
        key = (row[0], row[1])
        if key in links:
            continue
        found = next(((learned[(kind, value)], kind) for kind, value in
                      (("cusip6", cusip6(row[3])), ("isin", ident(row[4])), ("lei", ident(row[6])))
                      if value and (kind, value) in learned), None)
        if found:
            links[key] = found
            continue
        if row[7] == EQUITY and (row[8] or "CORP") == "CORP":
            ticker = match_name(row[2], by_name)
            if ticker:
                links[key] = (ticker, "name")
                teach(row, ticker, "name", NAME_REVIEW_CONFIDENCE)
                continue
        links[key] = (None, "unmatched")
    db.executemany("INSERT OR REPLACE INTO holding_link VALUES (?, ?, ?, ?)",
                   [(a, h, t, m) for (a, h), (t, m) in links.items()])
    counts: dict[str, int] = defaultdict(int)
    for ticker, method in links.values():
        counts[method] += 1
    return dict(counts)


def coverage(db: sqlite3.Connection, accession: str) -> dict[str, Any]:
    """Share of the fund's equity in corpus companies, by mapping method."""
    total = db.execute("SELECT COALESCE(SUM(pct_of_net_assets), 0), COUNT(*) FROM holdings WHERE accession = ? "
                       "AND asset_cat = 'EC' AND COALESCE(payoff, 'Long') = 'Long'", (accession,)).fetchone()
    by_method = {method: {"holdings": n, "pct_of_net_assets": round(pct or 0, 4)} for method, n, pct in db.execute(
        "SELECT l.method, COUNT(*), SUM(h.pct_of_net_assets) FROM holding_link l JOIN holdings h USING (accession, holding_id) "
        "WHERE l.accession = ? AND h.asset_cat = 'EC' AND COALESCE(h.payoff, 'Long') = 'Long' GROUP BY l.method", (accession,))}
    covered = db.execute("SELECT COUNT(*), COALESCE(SUM(pct_of_net_assets), 0) FROM fund_exposure WHERE accession = ?",
                         (accession,)).fetchone()
    return {"equity_pct_of_net_assets": round(total[0], 4), "equity_holdings": total[1], "corpus_companies": covered[0],
            "corpus_pct_of_net_assets": round(covered[1], 4), "by_method": by_method}


def sector_mix(db: sqlite3.Connection, accession: str) -> list[dict[str, Any]]:
    """Equity weight per corpus sector (companies outside the corpus are 'Not in corpus')."""
    out: dict[str, dict] = defaultdict(lambda: {"pct_of_net_assets": 0.0, "companies": 0, "top": []})
    for ticker, pct in db.execute("SELECT corpus_ticker, pct_of_net_assets FROM fund_exposure WHERE accession = ? "
                                  "ORDER BY pct_of_net_assets DESC", (accession,)):
        row = out[SECTOR_OF.get(ticker) or "Unclassified"]
        row["pct_of_net_assets"] += pct or 0
        row["companies"] += 1
        if len(row["top"]) < 3:
            row["top"].append((ticker, pct))
    info = coverage(db, accession)
    rest = info["equity_pct_of_net_assets"] - info["corpus_pct_of_net_assets"]
    rows = [{"sector": sector, **values} for sector, values in out.items()]
    rows.sort(key=lambda r: -r["pct_of_net_assets"])
    if rest > 0.005:
        rows.append({"sector": "Not in corpus", "pct_of_net_assets": rest,
                     "companies": info["equity_holdings"] - sum(r["companies"] for r in rows), "top": []})
    return rows


# ---------------------------------------------------------------------------- loading
def read_manifest(folder: Path) -> dict[str, dict]:
    path = Path(folder) / MANIFEST_NAME
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["accession"]: row for row in rows if row.get("accession")}


def xml_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.xml")) if path.is_dir() else [path]
    return files


def accession_from(path: Path) -> str | None:
    match = re.search(r"\d{10}-\d{2}-\d{6}", path.name)
    return match.group(0) if match else None


def load(args) -> dict[str, Any]:
    db_path = Path(args.db or default_db())
    db = connect(db_path)
    loaded: list[str] = []
    with db:
        for path in xml_files(args.xml or []):
            meta = read_manifest(path.parent).get(accession_from(path) or "", {})
            parsed = parse_xml(path.read_bytes(), accession=accession_from(path), filing_date=meta.get("filing_date"),
                               source=f"xml:{path.name}")
            loaded.append(store(db, parsed))
        if args.tsv_zip:
            for parsed in parse_tsv_zip(args.tsv_zip, args.series or (), args.name_like or (), args.top):
                loaded.append(store(db, parsed))
        methods = map_holdings(db, loaded) if loaded else {}
    summary = {"db": str(db_path), "filings_loaded": len(loaded), "links": methods,
               "funds": db.execute("SELECT COUNT(*) FROM funds").fetchone()[0],
               "filings": db.execute("SELECT COUNT(*) FROM filings").fetchone()[0],
               "coverage": {a: coverage(db, a) for a in loaded[:20]}}
    db.close()
    return summary


def status(args) -> dict[str, Any]:
    db_path = Path(args.db or default_db())
    if not db_path.exists():
        return {"db": str(db_path), "exists": False}
    db = sqlite3.connect(db_path)
    rows = db.execute("SELECT f.series_name, f.family, g.period_end, g.accession, g.net_assets, g.holdings_count "
                      "FROM filings g JOIN funds f USING (series_id) ORDER BY g.net_assets DESC").fetchall()
    out = {"db": str(db_path), "exists": True, "filings": [
        {"fund": r[0], "family": r[1], "period_end": r[2], "accession": r[3],
         "net_assets_usd_bn": round((r[4] or 0) / 1e9, 1), "holdings": r[5], **coverage(db, r[3])} for r in rows]}
    db.close()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("load", help="parse N-PORT XML files and/or a quarterly bulk ZIP into the SQLite tables")
    one.add_argument("--db", help=f"default: <RAG_HOME>/indexes/mutual_fund/{DB_NAME}")
    one.add_argument("--xml", nargs="*", help="primary_doc.xml files or folders (accession in the file name)")
    one.add_argument("--tsv-zip", help="SEC N-PORT quarterly data set ZIP (e.g. 2026q3_nport.zip)")
    one.add_argument("--series", nargs="*", help="series IDs to take from the ZIP (S000002839 ...)")
    one.add_argument("--name-like", nargs="*", help="series name fragments to take from the ZIP ('500 INDEX')")
    one.add_argument("--top", type=int, help="only the N largest chosen funds by net assets")
    two = sub.add_parser("status", help="funds, periods and corpus coverage in the database")
    two.add_argument("--db")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "load" and not (args.xml or args.tsv_zip):
        raise SystemExit("load needs --xml and/or --tsv-zip")
    result = load(args) if args.command == "load" else status(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
