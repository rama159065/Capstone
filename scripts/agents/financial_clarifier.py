"""Clarify incomplete or high-level financial questions before any agent runs.

A chat turn is parsed into slots (intent, companies, fiscal years, metric). When a
slot that the intent needs is missing, ambiguous or not in the corpus, the API
asks for it, offering the options the corpus actually has, and remembers the
partial request under a ``conversation_id``. Only a complete request reaches
the agents; a full report additionally needs an explicit confirmation.

Questions the filings can't answer are said so plainly, with what is available:
a company or year we hold no filings for, or a measure a bank doesn't report.
``question_scope`` gives the agent the same decision for callers that skip the
clarifier, and the companies its citations must belong to.

Deterministic on purpose: no LLM call, so behaviour is predictable, testable and
free. Guard-worthy messages (advice, predictions, prompt injection) are passed
straight through so the agent's guardrails refuse them; we never ask follow-up
questions about them.

| Intent | Needs | Example |
|---|---|---|
| figure | company, year, metric | "What was Apple's revenue in FY2024?" |
| comparison | two or more companies, year, metric | "Compare Visa and Mastercard net income 2025" |
| trend | company, metric (years default to all available) | "How has NVIDIA's revenue changed?" |
| sector_trend | one sector, metric (years default to the sector's indexed years) | "How did healthcare revenue trend from 2022 to 2025?" |
| explain | company (year defaults to the latest, stated as an assumption) | "Why did Chevron's earnings fall?" |
| report | company, year, then a yes/no confirmation | "Generate a financial report for Apple 2024" |
| overview | the user picks one of the above | "Tell me about Microsoft" |
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from scripts.common.paths import ROOT  # noqa: E402 - repo root
from scripts.common.companies import (  # noqa: E402
    ALIASES as COMPANY_ALIASES, AMBIGUOUS_TICKERS, BANKING, NAMES as COMPANY_NAMES, SECTOR_NAMES,
    SECTOR_OF as COMPANY_SECTORS, fiscal_label, normalize_ticker)
MANIFESTS = (ROOT / "data/raw/financial_report_qa/edgar/pdf_manifest.jsonl",
             ROOT / "data/raw/financial_report_qa/pdf_manifest.jsonl",
             ROOT / "data/raw/financial_report_qa/annual_reports/pdf_manifest.jsonl")   # sector corpus, once downloaded
REPORT_PROFILES = ROOT / "configs/financial_reports"
MAX_CLARIFICATION_TURNS = 3

# Names, aliases and sectors of every company in the corpora (the 13 sectors of the sector corpus plan).
NAMES, ALIASES, SECTOR_OF = COMPANY_NAMES, COMPANY_ALIASES, COMPANY_SECTORS
SECTORS = {sector: " ".join(sorted(t for t, s in SECTOR_OF.items() if s == sector)) for sector in SECTOR_NAMES}
# Banks' income statements have no cost of revenue (JPM, BAC and WFC filings carry no XBRL value for these measures).
# Visa and Mastercard aren't here: they are payment networks and do report operating income.
BANKS = {t for t, sector in SECTOR_OF.items() if sector == BANKING} | {"GS", "MS"}
NOT_REPORTED_BY_BANKS = {"gross profit", "gross margin", "operating income", "operating margin"}
BANK_MEASURES = ["Net revenue", "Net income", "Net interest income"]
METRICS = {  # canonical name -> phrases, matched longest first
    "diluted EPS": ["diluted eps", "diluted earnings per share", "earnings per share", "eps"],
    "free cash flow": ["free cash flow", "fcf"],
    "operating cash flow": ["operating cash flow", "cash from operations", "cash provided by operating activities"],
    "capital expenditures": ["capital expenditures", "capital expenditure", "capex"],
    "cash and cash equivalents": ["cash and cash equivalents", "cash position", "cash balance"],
    "operating margin": ["operating margin"], "gross margin": ["gross margin"], "net margin": ["net margin", "profit margin"],
    "operating income": ["operating income", "operating profit", "ebit"],
    "gross profit": ["gross profit"],
    "net income": ["net income", "net earnings", "net profit", "profit", "earnings"],
    "net interest income": ["net interest income"],
    "deposits": ["total deposits", "deposits"],
    "revenue": ["net revenue", "revenue", "revenues", "net sales", "total sales", "sales", "top line", "turnover"],
    "total assets": ["total assets", "assets"], "total liabilities": ["total liabilities", "liabilities"],
    "long-term debt": ["long-term debt", "long term debt", "total debt", "debt"],
    "shareholders' equity": ["shareholders' equity", "stockholders' equity", "shareholders equity", "equity"],
    "dividends paid": ["dividends paid", "dividends", "dividend"],
    "share repurchases": ["share repurchases", "share buybacks", "buybacks", "repurchases"],
    "research and development": ["research and development", "r&d"],
    "employees": ["employees", "headcount", "workforce"],
    "auditor": ["auditor", "audit firm"],
    "segment revenue": ["segment revenue", "revenue by segment", "segments", "segment"],
}
PASSTHROUGH = re.compile(  # guardrail territory: never ask follow-ups, let the agent refuse
    r"\b(should i (buy|sell|invest)|buy or sell|price target|predict|forecast|will .{0,40}\b(rise|fall|beat|reach)|"
    r"next (year|quarter)'?s? (earnings|revenue|price)|what will|will\b.{0,60}\bnext (fiscal )?(year|quarter)|"
    r"ignore (all |your )?(previous |prior )?instructions|system prompt|api key|password|poem|song|joke|"
    r"home address|personal (address|phone|email)|non-public|insider|safest|riskiest|where to (keep|put|invest))\b", re.I)
BENCHMARK = re.compile(r"\b(peers?|peer[- ]group|median|average|benchmark|industry|percentile|prior year|previous year|"
                       r"last year|year earlier)\b", re.I)
LOOKUP = re.compile(r"^\s*(what|which|who|when|where|how (much|many)|according|did|does|do|is|are|was|were|list|show)\b",
                    re.I)   # a specific question: answer it from the documents instead of offering the overview menu
REPORT = re.compile(r"\b(generate|create|prepare|write|build|produce|make|give me)\b.{0,60}\b(report|analysis|summary)\b|"
                    r"\b(full|complete|detailed|comprehensive)\s+(financial\s+)?(report|analysis)\b", re.I)
COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|relative to|which (company|of)|higher|lower|more than|less than)\b", re.I)
TREND = re.compile(r"\b(trend|over (the )?(last|past)|over time|year over year|yoy|history|historical|growth|grew|"
                   r"change[ds]?|evolv\w*)\b", re.I)
EXPLAIN = re.compile(r"^\s*(why|how (did|does|has|is)|what (drove|caused|explains|are the (main )?(risks?|drivers))|"
                     r"explain|describe|summari[sz]e what)\b|\b(risk factors?|strategy|outlook discussion|drivers?)\b", re.I)
YES = re.compile(r"^\s*(y|yes|yeah|yep|ok|okay|sure|confirm(ed)?|go ahead|do it|generate( it)?|proceed)\s*[.!]*\s*$", re.I)
NO = re.compile(r"^\s*(n|no|nope|cancel|stop|never ?mind)\s*[.!]*\s*$", re.I)
CONTINUE = re.compile(r"^\s*(continue|skip|just answer|answer anyway|as is|run it)\b", re.I)
FOLLOW_UP = re.compile(r"^\s*(and|also|what about|how about|same for|now|then)\b", re.I)
FOLLOW_UP_FILLER = frozenset("""a about also and company fiscal for fy how in is it its now of on please same that the
    their them then these they this those to from through until up was were what with year years s""".split())
RESET = re.compile(r"^\s*(new question|start over|restart|reset)\b", re.I)
LATEST = re.compile(r"\b(latest|most recent|last (fiscal )?year|this year|current year|recent)\b", re.I)
ALL_YEARS = re.compile(r"\b(all( years)?|every year|all available)\b", re.I)
ALONE = re.compile(r"\b(alone|only)\s*[.!]*\s*$", re.I)
NAME_RUN = r"[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*)*(?![A-Za-z0-9])"
UNKNOWN_NAME = re.compile(  # Intel's, Reliance Industries', revenue of Contoso, did Tata Motors, and Acme Holdings
    rf"\b({NAME_RUN})'(?:s\b|(?=\s|$))"
    rf"|\b(?:of|for|at|by|from|about|did|does|has|had|and|vs\.?|versus|with|against)\s+({NAME_RUN})"
    rf"|\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*)*\s+(?:Inc|Corp|Corporation|Ltd|Limited|LLC|PLC|Industries|Motors|Holdings|Group))\b")
NOT_A_NAME = frozenset("""a about according all also an and annual api april are ars at august board by cagr capex cash ceo cfo
    chart company compare comparison companies debt december did do does earnings ebit edgar eps equity february fcf
    financial fiscal for form from fy gaap give gross how in income is item january july june latest list march margin
    may me net non-gaap note november october of on operating or page please profit report reports revenue revenues roa
    roe sales sec segment september show statement statements table tell the their total us usa u.s. usd vs was we were
    what when where which who why with xbrl year years""".split())
SECTOR_TREND_YEARS = 5     # same as financial_trend.MAX_TREND_YEARS
INTENT_CHOICES = {"Key figures": "figure", "Explain performance": "explain",
                  "Trend over the years": "trend", "Full financial report": "report"}


def corpus_manifests():
    """Manifests that describe the corpus: the repo's, plus ``RAG_CORPUS_DIR``'s when a server mounts the PDFs elsewhere."""
    paths = list(MANIFESTS)
    if os.getenv("RAG_CORPUS_DIR"):
        paths.append(Path(os.environ["RAG_CORPUS_DIR"]) / "pdf_manifest.jsonl")
    return paths


def index_state_files():
    """Ingest state of the indexes this process serves. They list the filings that are searchable even
    when the PDFs and their manifests stay on Drive (a server that only restored the index)."""
    return [Path(os.environ[env]) / name for env, name in (("RAG_INGEST_DIR", "ingest.sqlite3"),
                                                          ("RAG_VLM_DIR", "vlm_ingest.sqlite3")) if os.getenv(env)]


def indexed_filings(path):
    """(ticker, year) of every filing whose text is in the index described by an ingest state file."""
    import sqlite3
    try:
        with closing(sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(filings)")}
            # text stage of pipeline 2 (Qdrant); pipeline 1 (Chroma) records its one stage as "vlm"
            where = "status='complete'" + (" AND stage IN ('text', 'vlm')" if "stage" in columns else "")
            doc_ids = [row[0] for row in db.execute(f"SELECT DISTINCT doc_id FROM filings WHERE {where}")]
    except sqlite3.Error:
        return []
    found = []
    for doc_id in doc_ids:
        match = re.fullmatch(r"edgar_ars_(.+)_(\d{4})", doc_id or "")
        if match:
            found.append((match.group(1).upper(), int(match.group(2))))
    return found


def load_directory(paths=None):
    """ticker -> sorted fiscal years with a filing in the corpus, as the company names them (``companies.fiscal_label``).

    By default the filings the served indexes hold, because only those can be answered: a manifest also lists
    filings that are downloaded but not (yet) ingested. Without an index state file (Chroma default) the corpus
    manifests decide. ``COVERAGE_SOURCE=corpus`` always adds the manifests. Explicit ``paths`` read only those
    manifests.
    """
    years: dict[str, set[int]] = {}
    if paths is None:
        for state in index_state_files():
            if state.exists():
                for ticker, year in indexed_filings(state):
                    years.setdefault(ticker, set()).add(fiscal_label(ticker, year))
        if years and os.getenv("COVERAGE_SOURCE", "index").lower() != "corpus":
            return {ticker: sorted(values) for ticker, values in sorted(years.items())}
    for path in (corpus_manifests() if paths is None else paths):
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    ticker = row["ticker"].upper()
                    years.setdefault(ticker, set()).add(fiscal_label(ticker, int(row["year"])))
    return {ticker: sorted(values) for ticker, values in sorted(years.items())}


def name(ticker):
    return NAMES.get(ticker, ticker)


def join_names(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@dataclass
class Slots:
    intent: str | None = None
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    latest: bool = False
    all_years: bool = False
    metric: str | None = None
    confirmed: bool | None = None
    sectors: list[str] = field(default_factory=list)     # sector_trend: the sectors named (one is needed)


@dataclass
class Conversation:
    conversation_id: str
    original_query: str = ""
    slots: Slots = field(default_factory=Slots)
    pending: list[str] = field(default_factory=list)       # slots asked in the last turn
    turns: int = 0
    assumptions: list[str] = field(default_factory=list)
    unknown_companies: list[str] = field(default_factory=list)
    banks_without_metric: list[str] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


class Clarifier:
    def __init__(self, directory=None, profiles_dir=REPORT_PROFILES):
        self.directory = directory if directory is not None else load_directory()
        self.profiles_dir = Path(profiles_dir)

    # ------------------------------------------------------------------ parsing
    def companies_in(self, text):
        found = []                                                  # (start, end, ticker)
        lower = text.lower()
        for ticker, aliases in ALIASES.items():
            for alias in aliases:
                match = re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower)
                if match:
                    found.append((match.start(), match.end(), ticker))
                    break
        # Upper-case tickers (AAPL, NVDA, BRK-B / BRK.B). Single letters and word-like tickers (AR, ICE, NOW)
        # only as $V or "ticker V".
        for match in re.finditer(r"(?:\$|\bticker\s+)?\b([A-Z]{1,5}(?:[.-][A-Z])?)\b", text):
            ticker = normalize_ticker(match.group(1))
            explicit = match.group(0)[0] in "$t"
            if (ticker in self.directory or ticker in NAMES) and (
                    explicit or (len(ticker) > 1 and ticker not in AMBIGUOUS_TICKERS)):
                found.append((match.start(1), match.end(1), ticker))
        # The longest name wins: "GE Vernova" is not also GE, "Cheniere Energy Partners" not also Cheniere.
        found = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0])
                                             and o[2] != f[2] for o in found)]
        ordered = []
        for _, _, ticker in sorted(found):
            if ticker not in ordered:
                ordered.append(ticker)
        return ordered

    @staticmethod
    def years_in(text):
        years = [int(y) for y in re.findall(r"(?<!\d)(20[0-4]\d)(?!\d)", text)]           # 2024, FY2024
        years += [2000 + int(y) for y in re.findall(r"\bfy\s?'?(\d{2})(?!\d)", text, re.I)]  # FY24
        # "2022-2025", "FY2022 to FY2025", "fiscal 2023 to fiscal 2026", "between 2022 and 2025"
        span = (re.search(r"(?<!\d)(20[0-4]\d)\s*(?:-|–|—|to|through|thru|until|till)\s*(?:the\s+)?(?:fiscal\s*)?(?:years?\s*)?"
                          r"(?:fy\s?)?(20[0-4]\d)(?!\d)", text, re.I)
                or re.search(r"\bbetween\s+(?:fiscal\s*)?(?:years?\s*)?(?:fy\s?)?(20[0-4]\d)\s+and\s+(?:fiscal\s*)?(?:years?\s*)?"
                             r"(?:fy\s?)?(20[0-4]\d)(?!\d)", text, re.I))
        if span:
            years += list(range(int(span.group(1)), int(span.group(2)) + 1))
        return sorted(set(years))

    @staticmethod
    def metric_in(text):
        lower = text.lower()
        best = None
        for metric, phrases in METRICS.items():
            for phrase in phrases:
                match = re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lower)
                if match and (best is None or len(phrase) > best[1]):
                    best = (metric, len(phrase))
        return best[0] if best else None

    def subject_in(self, text):
        """The words that narrow the measure to a segment, product or region ("data center" revenue), or None."""
        from scripts.agents.financial_checks import subject
        from scripts.agents.financial_sector_trend import SECTOR_ALIASES
        metric = self.metric_in(text)
        aliases = [a for words in ALIASES.values() for a in words] + list(NAMES.values())
        return subject(text, metric, METRICS.get(metric, []), aliases,
                       [p for phrases in SECTOR_ALIASES.values() for p in phrases], self.companies_in(text))

    @staticmethod
    def sectors_in(text):
        from scripts.agents.financial_sector_trend import sectors_in
        return sectors_in(text)

    def intent_in(self, text, companies, years):
        cleaned = re.sub(r"\bannual reports?\b|\bform (ars|10-?k)\b", " ", text, flags=re.I)
        if REPORT.search(cleaned):
            return "report"
        # A sector and no company: the sector's trend (one measure across its companies, year by year).
        if not companies and self.sectors_in(cleaned) and (TREND.search(cleaned) or len(years) >= 2):
            return "sector_trend"
        # One company compared with a benchmark (peers, median, prior year) is a question about that company;
        # "compare X" with no benchmark still needs a second company, so it stays a comparison.
        explicit = re.search(r"\b(compare|versus|vs\.?)\b", cleaned, re.I) and not BENCHMARK.search(cleaned)
        if len(companies) >= 2 or (not companies and COMPARE.search(cleaned)) or (len(companies) == 1 and explicit):
            return "comparison"
        if TREND.search(cleaned) or (len(years) >= 2 and len(companies) == 1):
            return "trend"
        if EXPLAIN.search(cleaned):
            return "explain"
        if self.metric_in(cleaned):
            return "figure"
        if companies and len(cleaned.split()) >= 6 and LOOKUP.match(cleaned):
            return "explain"            # e.g. Chevron's production or reserves: not a figure we have a slot for
        return None

    @staticmethod
    def unknown_names(text):
        """Company-like names we have no alias for ("Intel's revenue", "revenue of Reliance Industries")."""
        from scripts.agents.financial_sector_trend import SECTOR_ALIASES
        words = {a for aliases in ALIASES.values() for a in aliases} | {p for ps in SECTOR_ALIASES.values() for p in ps}
        for alias in sorted(words, key=len, reverse=True):
            text = re.sub(rf"(?<![A-Za-z]){re.escape(alias)}(?:'s|')?(?![A-Za-z])", " ", text, flags=re.I)
        found = []
        for match in UNKNOWN_NAME.finditer(text):
            words = next(group for group in match.groups() if group).split()
            while words and words[0].lower().rstrip(".") in NOT_A_NAME:
                words.pop(0)
            while words and words[-1].lower().rstrip(".") in NOT_A_NAME:
                words.pop()
            candidate = " ".join(words)
            if len(candidate) > 1 and candidate not in found:
                found.append(candidate)
        return found

    def parse(self, text):
        companies = self.companies_in(text)
        years = self.years_in(text)
        return Slots(intent=self.intent_in(text, companies, years), companies=companies, years=years,
                     latest=bool(LATEST.search(text)), all_years=bool(ALL_YEARS.search(text)),
                     metric=self.metric_in(text), sectors=self.sectors_in(text))

    # ------------------------------------------------------------------ dialogue
    def step(self, conversation: Conversation, message: str, *, ticker=None, year=None) -> dict[str, Any]:
        """Advance one user turn; returns needs_clarification, ready, cancelled or passthrough."""
        conversation.updated = time.time()
        text = message.strip()
        if PASSTHROUGH.search(text):
            conversation.pending = []
            return self.result(conversation, "passthrough", action="answer", resolved_query=text,
                               ticker=None, year=None, original=text)
        if self.fund_question(text):
            # A fund question needs a fund, not a company and year: the fund agent resolves the fund and period itself
            # and says which fund it needs when the question is ambiguous. Only when fund data is loaded.
            self.restart(conversation, "")      # a later "what about net income?" must not go back to an older company
            return self.result(conversation, "passthrough", action="answer", resolved_query=text,
                               ticker=None, year=None, original=text, route_hint="fund")
        action = self.action_route(text)
        if action:
            # Shareholder returns and dividend income plan their own companies, year and holdings, and say what they
            # can't cover; a follow-up question would only drop the sector or the share counts.
            self.restart(conversation, "")
            return self.result(conversation, "passthrough", action="answer", resolved_query=text,
                               ticker=ticker, year=year, original=text, route_hint=action)
        if RESET.match(text):
            self.restart(conversation, "")
            return self.ask(conversation, "Sure, what would you like to know?", [])
        if conversation.pending == ["confirm"]:
            if YES.match(text):
                conversation.slots.confirmed = True
                return self.ready(conversation)
            if NO.match(text):
                conversation.pending = []
                conversation.slots.confirmed = None
                return self.result(conversation, "cancelled", message="Report cancelled. Ask another question any time.")

        parsed = self.parse(text)
        choice = next((intent for label, intent in INTENT_CHOICES.items() if text.lower().startswith(label.lower())), None)
        slots = conversation.slots
        new_subject = bool(parsed.companies) and set(parsed.companies) != set(slots.companies)
        # A reply to our question is short ("FY2024", "Net income"); a full sentence naming a measure, or a
        # different company plus a measure/year, is a new request.
        # After an answer, only "and for 2023?" / "what about Visa?" or a short fragment continues the old request.
        follow_up = bool(FOLLOW_UP.match(text)) or len(text.split()) <= 3
        # Mid-clarification, a sentence about a company we did not ask for is a change of subject.
        switch = (new_subject and not follow_up and not {"company", "companies"} & set(conversation.pending))
        fresh = ((not conversation.pending and not follow_up) or switch
                 or parsed.intent is not None and (not conversation.pending or (new_subject and bool(parsed.metric or parsed.years))
                                                   or (len(text.split()) >= 5 and bool(parsed.metric))))
        another = follow_up and self.another_measure(conversation, text, parsed)
        if another:     # "what about net income?": same companies, years and kind of question, another measure
            fresh = False
            parsed.intent = ("figure" if slots.intent not in {"comparison", "trend", "sector_trend"}
                             or (slots.intent == "trend" and len(parsed.years) == 1) else slots.intent)
        if not conversation.original_query or fresh:
            self.restart(conversation, text)
            slots = conversation.slots
            parsed.companies = parsed.companies or ([ticker.upper()] if ticker else [])
            parsed.years = parsed.years or ([int(year)] if year else [])
        # Merge: what the user just said fills or replaces the matching slot; other slots carry over,
        # so a follow-up such as "and for 2023?" keeps the company and the measure.
        # A short answer to our question ("Revenue" when asked for the measure) fills a slot; it is not a new intent.
        answering = bool(conversation.pending) and follow_up and not parsed.companies and slots.intent is not None
        slots.intent = choice or (slots.intent if answering else parsed.intent) or slots.intent
        if parsed.companies:
            slots.companies = (list(dict.fromkeys(slots.companies + parsed.companies))
                               if slots.intent == "comparison" else parsed.companies)
        if parsed.years:
            slots.years, slots.latest, slots.all_years = parsed.years, False, False
        slots.latest = (slots.latest or parsed.latest) and not parsed.years
        slots.all_years = slots.all_years or parsed.all_years
        slots.metric = parsed.metric or slots.metric
        slots.sectors = parsed.sectors or slots.sectors
        covered = [c for c in slots.companies if c in self.directory]
        if len(covered) == 1 and "companies" in conversation.pending and ALONE.search(text):
            slots.intent, slots.companies = "figure", covered

        missing = self.missing(conversation)
        if not missing or (missing != ["confirm"] and (CONTINUE.match(text) or conversation.turns >= MAX_CLARIFICATION_TURNS)):
            return self.ready(conversation, forced=bool(missing))
        conversation.turns += 1
        return self.ask(conversation, None, missing)

    def another_measure(self, conversation, text, parsed):
        """True when the message only names another measure (and perhaps years) for the companies or sector already
        being discussed. Any other word (a company we don't know, a segment, "growth") makes it a new question, so it is
        never answered for the previous company."""
        slots = conversation.slots
        if not parsed.metric or parsed.companies or parsed.sectors or not (slots.companies or slots.sectors):
            return False
        if self.subject_in(conversation.original_query):    # "U.S. Personal Banking net income": the segment isn't a slot
            return False
        rest = text.lower().replace("’", "'")
        for phrase in sorted(METRICS[parsed.metric], key=len, reverse=True):
            rest = re.sub(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", " ", rest)
        rest = re.sub(r"\b(?:fy\s?)?20[0-4]\d\b|\bfy\s?'?\d{2}\b", " ", rest)
        return set(re.findall(r"[^\W_]+", rest)) <= FOLLOW_UP_FILLER

    @staticmethod
    def fund_question(text):
        from scripts.agents.financial_router import fund_question
        return fund_question(text)

    @staticmethod
    def action_route(text):
        """'dividend_income' or 'shareholder_returns' when the question's words ask for that agent, else None."""
        from scripts.agents.financial_dividend_income import holdings_signal
        from scripts.agents.financial_shareholder_returns import returns_signal
        if REPORT.search(text):
            return None
        return "dividend_income" if holdings_signal(text) else "shareholder_returns" if returns_signal(text) else None

    def restart(self, conversation, text):
        conversation.__dict__.update(Conversation(conversation.conversation_id, original_query=text).__dict__)

    def uncovered(self, slots, query):
        """Names of the companies the question asks about that we hold no filings for."""
        directory = self.known(slots)
        covered = [c for c in slots.companies if c in directory]
        names = [name(c) for c in slots.companies if c not in directory]
        if len(covered) < (2 if slots.intent == "comparison" else 1):
            names += [n for n in self.unknown_names(query) if n not in names]
        return names

    def banks_without(self, slots):
        """Covered banks that don't report the requested measure."""
        if slots.intent not in {"figure", "comparison", "trend"} or slots.metric not in NOT_REPORTED_BY_BANKS:
            return []
        return [c for c in slots.companies if c in BANKS and c in self.directory]

    def missing(self, conversation):
        slots, missing = conversation.slots, []
        if slots.intent == "sector_trend":
            conversation.unknown_companies, conversation.banks_without_metric = [], []
            if len(slots.sectors) != 1:
                missing.append("sector")
            if not slots.metric or self.bank_sector_without(slots):
                missing.append("metric")
            return missing
        if slots.intent is None:
            missing.append("intent")
        directory = self.known(slots)
        covered = [c for c in slots.companies if c in directory]
        conversation.unknown_companies = self.uncovered(slots, conversation.original_query)
        conversation.banks_without_metric = self.banks_without(slots)
        if len(covered) < (2 if slots.intent == "comparison" else 1):
            missing.append("companies" if slots.intent == "comparison" else "company")
        elif slots.intent in {"figure", "comparison", "report"}:
            available = self.common_years(covered, directory)
            if slots.latest and available:
                slots.years, slots.latest = [available[-1]], False
            if not slots.years or any(y not in available for y in slots.years):
                missing.append("year")
        if slots.intent in {"figure", "comparison", "trend"} and (not slots.metric or conversation.banks_without_metric):
            missing.append("metric")
        if slots.intent == "report" and not missing and slots.confirmed is not True:
            missing.append("confirm")
        return missing

    @staticmethod
    def bank_sector_without(slots):
        """A banking sector trend asked for a measure banks don't report."""
        return slots.sectors == [BANKING] and slots.metric in NOT_REPORTED_BY_BANKS

    def sector_members(self, sector):
        return sorted(t for t in self.directory if SECTOR_OF.get(t) == sector)

    def sector_years(self, sector):
        return sorted({y for t in self.sector_members(sector) for y in self.directory.get(t, [])})

    def common_years(self, companies, directory=None):
        directory = self.directory if directory is None else directory
        sets = [set(directory.get(c, [])) for c in companies]
        return sorted(set.intersection(*sets)) if sets else []

    def similar_companies(self, slots, limit=3):
        """Covered companies in the same sector as a company we don't have."""
        sectors = {SECTOR_OF[c] for c in slots.companies if c not in self.directory and c in SECTOR_OF}
        return [t for t in self.directory if SECTOR_OF.get(t) in sectors and t not in slots.companies][:limit]

    def company_not_available(self, slots, unknown):
        """The reply and machine-readable reason when we hold no filings for a company."""
        covered = [c for c in slots.companies if c in self.directory]
        similar = self.similar_companies(slots)
        if covered:
            text = (f"I don't have filings for {join_names(unknown)}, so I can't make this comparison. "
                    f"I can answer for {join_names(name(c) for c in covered)} alone.")
        else:
            text = f"I don't have filings for {join_names(unknown)}, so I can't answer this from my documents."
        if similar:
            text += f" Similar companies I do have: {join_names(name(c) for c in similar)}."
        return {"status": "not_available", "reason": "company_not_in_filings", "missing": unknown,
                "similar": similar, "message": text}

    @staticmethod
    def metric_not_applicable(slots, banks):
        """The reply and machine-readable reason when a bank doesn't report the measure."""
        verb = "doesn't" if len(banks) == 1 else "don't"
        text = (f"{join_names(name(c) for c in banks)} {verb} report {slots.metric}, because a bank's income statement "
                "has no such line. I won't calculate a substitute. "
                f"I can give {BANK_MEASURES[0].lower()}, {BANK_MEASURES[1].lower()} or {BANK_MEASURES[2].lower()} instead.")
        return {"status": "not_available", "reason": "metric_not_applicable", "metric": slots.metric,
                "companies": banks, "alternatives": BANK_MEASURES, "message": text}

    def scope(self, query, ticker=None):
        """What the agent needs when the clarifier was skipped or overridden.

        Returns the companies citations must belong to, and why the filings can't answer the question (or None).
        An empty directory means coverage is unknown, so nothing is decided.
        """
        if not self.directory:
            return {"expected": [], "coverage": None}
        slots = self.parse(query)
        if slots.intent == "sector_trend" and not ticker:
            # Citations must come from the sector's companies; the sector agent says what it can't cover.
            return {"expected": self.sector_members(slots.sectors[0]) if len(slots.sectors) == 1 else [],
                    "coverage": None}
        if not slots.companies and ticker:
            slots.companies = [ticker.upper()]
        covered = [c for c in slots.companies if c in self.directory]
        unknown = self.uncovered(slots, query)
        coverage = None
        if unknown:
            coverage = self.company_not_available(slots, unknown)
        elif self.banks_without(slots):
            coverage = self.metric_not_applicable(slots, self.banks_without(slots))
        elif slots.intent == "figure" and len(covered) == 1 and len(slots.years) == 1:
            available = self.directory[covered[0]]
            if slots.years[0] not in available:
                coverage = {"status": "not_available", "reason": "year_not_in_filings", "missing": [f"FY{slots.years[0]}"],
                            "available": available, "message": (
                                f"I don't have {name(covered[0])}'s FY{slots.years[0]} report. "
                                f"I have {', '.join(f'FY{y}' for y in available)}.")}
        return {"expected": covered, "coverage": coverage}

    def ask(self, conversation, message, missing):
        conversation.pending = missing
        slots, questions, coverage = conversation.slots, [], None
        directory = self.known(slots)
        covered = [c for c in slots.companies if c in directory]
        for slot in missing:
            if slot == "intent":
                subject = name(covered[0]) if covered else "the company"
                questions.append({"slot": "intent", "question": f"What would you like to know about {subject}?",
                                  "options": list(INTENT_CHOICES)})
            elif slot in {"company", "companies"}:
                ask = "Which companies should I compare (two or more)?" if slot == "companies" else "Which company?"
                prefix, first = "", []
                if conversation.unknown_companies:
                    coverage = self.company_not_available(slots, conversation.unknown_companies)
                    prefix = coverage["message"] + " "
                    ask = (f"Which company should I compare with {join_names(name(c) for c in covered)} instead?"
                           if covered else "Which company should I use instead?")
                    first = coverage["similar"]
                alone = [f"{name(covered[0])} only"] if slot == "companies" and len(covered) == 1 and prefix else []
                others = [t for t in self.directory if t not in first and t not in covered]
                questions.append({"slot": slot, "question": prefix + ask,
                                  "options": alone + [f"{name(t)} ({t})" for t in first + others]})
            elif slot == "year":
                available = self.common_years(covered, directory)
                outside = [y for y in slots.years if y not in available]
                prefix = ""
                if outside:
                    prefix = (f"FY{', FY'.join(map(str, outside))} is not in the filings I have"
                              + (f"; the latest is FY{available[-1]}. " if available else ". "))
                    coverage = {"status": "not_available", "reason": "year_not_in_filings",
                                "missing": [f"FY{y}" for y in outside], "available": available}
                who = " and ".join(name(c) for c in covered)
                questions.append({"slot": "year", "question": f"{prefix}Which fiscal year for {who}?",
                                  "options": [f"FY{y}" for y in reversed(available)] + ["Latest"]})
            elif slot == "sector":
                named = slots.sectors
                ask = f"One sector at a time: which of {join_names(named)}?" if len(named) > 1 else "Which sector?"
                options = [s for s in (named if len(named) > 1 else SECTOR_NAMES) if len(self.sector_members(s)) >= 2]
                questions.append({"slot": "sector", "question": ask, "options": options})
            elif slot == "metric" and slots.intent == "sector_trend" and self.bank_sector_without(slots):
                questions.append({"slot": "metric", "question": f"Banks don't report {slots.metric}. Which measure?",
                                  "options": list(BANK_MEASURES)})
            elif slot == "metric" and conversation.banks_without_metric:
                coverage = self.metric_not_applicable(slots, conversation.banks_without_metric)
                questions.append({"slot": "metric", "question": coverage["message"] + " Which measure?",
                                  "options": list(BANK_MEASURES)})
            elif slot == "metric":
                questions.append({"slot": "metric", "question": "Which measure?",
                                  "options": ["Revenue", "Net income", "Diluted EPS", "Operating margin",
                                              "Free cash flow", "Cash and cash equivalents", "Total debt"]})
            elif slot == "confirm":
                spec = self.report_spec(conversation)
                note = "" if spec["profile"] else (" Note: an automated report profile exists only for "
                                                   f"{', '.join(self.available_reports()) or 'no company'} so far.")
                questions.append({"slot": "confirm", "question": f"Generate the full financial report for {name(spec['ticker'])} "
                                  f"FY{spec['year']}? It takes a few minutes.{note}", "options": ["Yes", "No"]})
        text = message or " ".join(q["question"] for q in questions)
        return self.result(conversation, "needs_clarification", message=text, questions=questions, coverage=coverage)

    def ready(self, conversation, forced=False):
        """Fill stated defaults and hand over to the agent (answer) or the report generator (report)."""
        conversation.pending = []
        slots = conversation.slots
        if slots.intent == "sector_trend" and len(slots.sectors) == 1:
            available = self.sector_years(slots.sectors[0])
            if not slots.years and available:
                slots.years = available[-SECTOR_TREND_YEARS:]
                conversation.assumptions.append(
                    f"Using the sector's indexed fiscal years: FY{slots.years[0]}–FY{slots.years[-1]}.")
            if forced:
                conversation.assumptions.append("Some details were not given; answering the question as asked.")
            return self.result(conversation, "ready", action="answer", resolved_query=self.compose(conversation),
                               ticker=None, year=None, sector=slots.sectors[0], original=conversation.original_query)
        directory = self.known(slots)
        covered = [c for c in slots.companies if c in directory]
        available = self.common_years(covered, directory) if covered else []
        if not slots.years and available:
            if slots.intent == "trend" or slots.all_years:
                slots.years = available
                conversation.assumptions.append(f"Using every fiscal year in the corpus: FY{available[0]}–FY{available[-1]}.")
            else:
                slots.years = [available[-1]]
                conversation.assumptions.append(f"Using the latest fiscal year in the corpus: FY{available[-1]}.")
        if slots.intent == "report" and slots.confirmed is True:
            return self.result(conversation, "ready", action="report", report=self.report_spec(conversation))
        query = self.compose(conversation) if covered else conversation.original_query
        single = len(covered) == 1 and len(slots.years) == 1
        if forced:
            conversation.assumptions.append("Some details were not given; answering the question as asked.")
        return self.result(conversation, "ready", action="answer", resolved_query=query,
                           ticker=covered[0] if len(covered) == 1 else None,
                           year=slots.years[0] if single else None, original=conversation.original_query)

    def compose(self, conversation):
        """The user's own question when it already said everything; otherwise one built from the slots."""
        slots, original = conversation.slots, conversation.original_query
        said = self.parse(original)
        if ((said.intent, said.companies, said.years, said.metric, said.sectors)
                == (slots.intent, slots.companies, slots.years, slots.metric, slots.sectors)):
            return original
        names = [name(c) for c in slots.companies if c in self.directory]
        span = slots.years if len(slots.years) < 2 else [slots.years[0], slots.years[-1]]
        years = "–".join(f"FY{y}" for y in span)
        metric = slots.metric or "key financial results"
        if slots.intent == "comparison":
            return f"Compare {metric} for {' and '.join(names)} in {years}."
        if slots.intent == "sector_trend":
            return f"How did the {slots.sectors[0]} sector's {metric} trend across {years}?"
        if slots.intent == "trend":
            return f"How did {names[0]}'s {metric} change across {years}?"
        if slots.intent == "figure":
            return f"What was {names[0]}'s {metric} for {years}?"
        return f"{original} ({', '.join(names)}, {years})"

    def report_spec(self, conversation):
        slots = conversation.slots
        directory = self.known(slots)
        ticker = next((c for c in slots.companies if c in directory), None)
        year = slots.years[-1] if slots.years else None
        return {"ticker": ticker, "year": year, "profile": self.profile_for(ticker, year)}

    def profile_for(self, ticker, year):
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("ticker") == ticker and int(data.get("year", 0)) == year:
                return str(path)
        return None

    def report_directory(self):
        """Ticker -> fiscal years with a report profile. A report reads its own source PDF, not the index."""
        out = {}
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.setdefault(data["ticker"], set()).add(int(data["year"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return {ticker: sorted(years) for ticker, years in out.items()}

    def known(self, slots):
        """The companies and years a request can use: the indexed filings, plus the report profiles for a report."""
        if slots.intent != "report":
            return self.directory
        merged = {ticker: list(years) for ticker, years in self.directory.items()}
        for ticker, years in self.report_directory().items():
            merged[ticker] = sorted(set(merged.get(ticker, [])) | set(years))
        return merged

    def available_reports(self):
        out = []
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(f"{name(data['ticker'])} FY{data['year']}")
            except (OSError, ValueError, KeyError):
                continue
        return out

    @staticmethod
    def result(conversation, status, **extra):
        payload = {"status": status, "conversation_id": conversation.conversation_id,
                   "slots": asdict(conversation.slots), "assumptions": list(conversation.assumptions),
                   "clarification_turns": conversation.turns, **extra}
        payload.setdefault("questions", [])
        return payload


@lru_cache(maxsize=1)
def default_clarifier():
    return Clarifier()


def question_scope(query, ticker=None):
    """Companies the citations must belong to, and why the filings can't answer the question (or None)."""
    return default_clarifier().scope(query, ticker)


class ConversationStore:
    """Conversation state with a TTL: in memory (default), or in SQLite when ``path`` is given.

    SQLite (``CONVERSATION_DB`` in the API) keeps a conversation across API restarts and
    between worker processes, so a deployment can restart or scale without losing a
    half-answered clarification. Only the slots are stored, never answers or evidence.
    """

    def __init__(self, ttl_seconds=1800, max_conversations=10_000, path=None):
        self.ttl, self.max = ttl_seconds, max_conversations
        self.items: dict[str, Conversation] = {}
        self.lock = threading.Lock()
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
                db.execute("CREATE TABLE IF NOT EXISTS conversations "
                           "(conversation_id TEXT PRIMARY KEY, updated REAL NOT NULL, state TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        """One short-lived connection per call: committed on success, always closed."""
        import sqlite3
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _load(state: str) -> Conversation:
        data = json.loads(state)
        return Conversation(**{**data, "slots": Slots(**data.get("slots") or {})})

    def get(self, conversation_id=None):
        if self.path:
            return self._get_sqlite(conversation_id)
        with self.lock:
            now = time.time()
            for key in [k for k, c in self.items.items() if now - c.updated > self.ttl]:
                del self.items[key]
            if conversation_id and conversation_id in self.items:
                return self.items[conversation_id]
            if len(self.items) >= self.max:
                del self.items[min(self.items, key=lambda k: self.items[k].updated)]
            conversation = Conversation(conversation_id or str(uuid.uuid4()))
            self.items[conversation.conversation_id] = conversation
            return conversation

    def _get_sqlite(self, conversation_id):
        with self.lock, self._connect() as db:
            db.execute("DELETE FROM conversations WHERE updated < ?", (time.time() - self.ttl,))
            if conversation_id:
                row = db.execute("SELECT state FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
                if row:
                    return self._load(row[0])
            count = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            if count >= self.max:
                db.execute("DELETE FROM conversations WHERE conversation_id IN "
                           "(SELECT conversation_id FROM conversations ORDER BY updated LIMIT ?)", (count - self.max + 1,))
            conversation = Conversation(conversation_id or str(uuid.uuid4()))
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?)",
                       (conversation.conversation_id, conversation.updated, json.dumps(asdict(conversation))))
            return conversation

    def save(self, conversation):
        """Persist a conversation after a turn changed it (in memory the object is already the stored one)."""
        if not self.path:
            return
        conversation.updated = time.time()
        with self.lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?)",
                       (conversation.conversation_id, conversation.updated, json.dumps(asdict(conversation))))

    def drop(self, conversation_id):
        with self.lock:
            self.items.pop(conversation_id, None)
            if self.path:
                with self._connect() as db:
                    db.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
