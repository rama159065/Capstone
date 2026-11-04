"""Clarify incomplete or high-level financial questions before any agent runs.

A chat turn is parsed into slots (intent, companies, fiscal years, metric). When a
slot that the intent needs is missing, ambiguous or not in the corpus, the API
asks for it, offering the options the corpus actually has, and remembers the
partial request under a ``conversation_id``. Only a complete request reaches
the agents; a full report additionally needs an explicit confirmation.

Deterministic on purpose: no LLM call, so behaviour is predictable, testable and
free. Guard-worthy messages (advice, predictions, prompt injection) are passed
straight through so the agent's guardrails refuse them; we never ask follow-up
questions about them.

| Intent | Needs | Example |
|---|---|---|
| figure | company, year, metric | "What was Apple's revenue in FY2024?" |
| comparison | two or more companies, year, metric | "Compare Visa and Mastercard net income 2025" |
| trend | company, metric (years default to all available) | "How has NVIDIA's revenue changed?" |
| explain | company (year defaults to the latest, stated as an assumption) | "Why did Chevron's earnings fall?" |
| report | company, year, then a yes/no confirmation | "Generate a financial report for Apple 2024" |
| overview | the user picks one of the above | "Tell me about Microsoft" |
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from scripts.common.paths import ROOT  # noqa: E402 - repo root
MANIFESTS = (ROOT / "data/raw/financial_report_qa/edgar/pdf_manifest.jsonl",
             ROOT / "data/raw/financial_report_qa/pdf_manifest.jsonl")
REPORT_PROFILES = ROOT / "configs/financial_reports"
MAX_CLARIFICATION_TURNS = 3

NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
         "AVGO": "Broadcom", "JPM": "JPMorgan Chase", "WMT": "Walmart", "V": "Visa", "MA": "Mastercard",
         "COST": "Costco", "PG": "Procter & Gamble", "NFLX": "Netflix", "JNJ": "Johnson & Johnson",
         "ABBV": "AbbVie", "BAC": "Bank of America", "KO": "Coca-Cola", "AMD": "AMD", "CVX": "Chevron",
         "PEP": "PepsiCo", "TMO": "Thermo Fisher", "ADBE": "Adobe", "WFC": "Wells Fargo", "PFE": "Pfizer",
         "HD": "Home Depot", "TSLA": "Tesla", "CRM": "Salesforce", "XOM": "Exxon Mobil", "LLY": "Eli Lilly"}
ALIASES = {"AAPL": ["apple"], "MSFT": ["microsoft"], "NVDA": ["nvidia"], "AMZN": ["amazon"],
           "GOOGL": ["alphabet", "google"], "AVGO": ["broadcom"], "JPM": ["jpmorgan", "jp morgan", "chase"],
           "WMT": ["walmart", "wal-mart"], "V": ["visa"], "MA": ["mastercard"], "COST": ["costco"],
           "PG": ["procter & gamble", "procter and gamble", "p&g"], "NFLX": ["netflix"],
           "JNJ": ["johnson & johnson", "johnson and johnson", "j&j"], "ABBV": ["abbvie"],
           "BAC": ["bank of america", "bofa"], "KO": ["coca-cola", "coca cola", "coke"],
           "AMD": ["advanced micro devices"], "CVX": ["chevron"], "PEP": ["pepsico", "pepsi"],
           "TMO": ["thermo fisher"], "ADBE": ["adobe"], "WFC": ["wells fargo"], "PFE": ["pfizer"],
           "HD": ["home depot"], "TSLA": ["tesla"], "CRM": ["salesforce"], "XOM": ["exxon"], "LLY": ["eli lilly", "lilly"]}
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
    "revenue": ["revenue", "revenues", "net sales", "total sales", "sales", "top line", "turnover"],
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
RESET = re.compile(r"^\s*(new question|start over|restart|reset)\b", re.I)
LATEST = re.compile(r"\b(latest|most recent|last (fiscal )?year|this year|current year|recent)\b", re.I)
ALL_YEARS = re.compile(r"\b(all( years)?|every year|all available)\b", re.I)
INTENT_CHOICES = {"Key figures": "figure", "Explain performance": "explain",
                  "Trend over the years": "trend", "Full financial report": "report"}


def load_directory(paths=MANIFESTS):
    """ticker -> sorted fiscal years with a filing in the corpus."""
    years: dict[str, set[int]] = {}
    for path in paths:
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    row = json.loads(line)
                    years.setdefault(row["ticker"].upper(), set()).add(int(row["year"]))
    return {ticker: sorted(values) for ticker, values in sorted(years.items())}


def name(ticker):
    return NAMES.get(ticker, ticker)


@dataclass
class Slots:
    intent: str | None = None
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    latest: bool = False
    all_years: bool = False
    metric: str | None = None
    confirmed: bool | None = None


@dataclass
class Conversation:
    conversation_id: str
    original_query: str = ""
    slots: Slots = field(default_factory=Slots)
    pending: list[str] = field(default_factory=list)       # slots asked in the last turn
    turns: int = 0
    assumptions: list[str] = field(default_factory=list)
    unknown_companies: list[str] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


class Clarifier:
    def __init__(self, directory=None, profiles_dir=REPORT_PROFILES):
        self.directory = directory if directory is not None else load_directory()
        self.profiles_dir = Path(profiles_dir)

    # ------------------------------------------------------------------ parsing
    def companies_in(self, text):
        found = []
        lower = " " + text.lower() + " "
        for ticker, aliases in ALIASES.items():
            for alias in aliases:
                if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower):
                    found.append((lower.index(alias), ticker))
                    break
        # Upper-case tickers (AAPL, NVDA); single letters only as $V or "ticker V".
        for match in re.finditer(r"(?:\$|\bticker\s+)?\b([A-Z]{1,5})\b", text):
            ticker = match.group(1)
            if ticker in self.directory and (len(ticker) > 1 or match.group(0)[0] in "$t"):
                found.append((match.start(), ticker))
        ordered = []
        for _, ticker in sorted(found):
            if ticker not in ordered:
                ordered.append(ticker)
        return ordered

    @staticmethod
    def years_in(text):
        years = [int(y) for y in re.findall(r"(?<!\d)(20[0-4]\d)(?!\d)", text)]           # 2024, FY2024
        years += [2000 + int(y) for y in re.findall(r"\bfy\s?'?(\d{2})(?!\d)", text, re.I)]  # FY24
        span = re.search(r"(?<!\d)(20[0-4]\d)\s*(?:-|–|to|through|until)\s*(?:fy\s?)?(20[0-4]\d)(?!\d)", text, re.I)
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

    def intent_in(self, text, companies, years):
        cleaned = re.sub(r"\bannual reports?\b|\bform (ars|10-?k)\b", " ", text, flags=re.I)
        if REPORT.search(cleaned):
            return "report"
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
        """Possessive company-like names we have no alias for ("Intel's revenue")."""
        known = {alias for aliases in ALIASES.values() for alias in aliases}
        return [n for n in re.findall(r"\b([A-Z][A-Za-z&.\-]+)'s\b", text) if n.lower() not in known]

    def parse(self, text):
        companies = self.companies_in(text)
        years = self.years_in(text)
        return Slots(intent=self.intent_in(text, companies, years), companies=companies, years=years,
                     latest=bool(LATEST.search(text)), all_years=bool(ALL_YEARS.search(text)),
                     metric=self.metric_in(text))

    # ------------------------------------------------------------------ dialogue
    def step(self, conversation: Conversation, message: str, *, ticker=None, year=None) -> dict[str, Any]:
        """Advance one user turn; returns needs_clarification, ready, cancelled or passthrough."""
        conversation.updated = time.time()
        text = message.strip()
        if PASSTHROUGH.search(text):
            conversation.pending = []
            return self.result(conversation, "passthrough", action="answer", resolved_query=text,
                               ticker=None, year=None, original=text)
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
        if not conversation.original_query or fresh:
            self.restart(conversation, text)
            slots = conversation.slots
            parsed.companies = parsed.companies or ([ticker.upper()] if ticker else [])
            parsed.years = parsed.years or ([int(year)] if year else [])
        # Merge: what the user just said fills or replaces the matching slot; other slots carry over,
        # so a follow-up such as "and for 2023?" keeps the company and the measure.
        slots.intent = choice or parsed.intent or slots.intent
        if parsed.companies:
            slots.companies = (list(dict.fromkeys(slots.companies + parsed.companies))
                               if slots.intent == "comparison" else parsed.companies)
        if parsed.years:
            slots.years, slots.latest, slots.all_years = parsed.years, False, False
        slots.latest = (slots.latest or parsed.latest) and not parsed.years
        slots.all_years = slots.all_years or parsed.all_years
        slots.metric = parsed.metric or slots.metric

        missing = self.missing(conversation)
        if not missing or (missing != ["confirm"] and (CONTINUE.match(text) or conversation.turns >= MAX_CLARIFICATION_TURNS)):
            return self.ready(conversation, forced=bool(missing))
        conversation.turns += 1
        return self.ask(conversation, None, missing)

    def restart(self, conversation, text):
        conversation.__dict__.update(Conversation(conversation.conversation_id, original_query=text).__dict__)

    def missing(self, conversation):
        slots, missing = conversation.slots, []
        if slots.intent is None:
            missing.append("intent")
        covered = [c for c in slots.companies if c in self.directory]
        conversation.unknown_companies = ([c for c in slots.companies if c not in self.directory]
                                          or (self.unknown_names(conversation.original_query) if not covered else []))
        if len(covered) < (2 if slots.intent == "comparison" else 1):
            missing.append("companies" if slots.intent == "comparison" else "company")
        elif slots.intent in {"figure", "comparison", "report"}:
            available = self.common_years(covered)
            if slots.latest and available:
                slots.years, slots.latest = [available[-1]], False
            if not slots.years or any(y not in available for y in slots.years):
                missing.append("year")
        if slots.intent in {"figure", "comparison", "trend"} and not slots.metric:
            missing.append("metric")
        if slots.intent == "report" and not missing and slots.confirmed is not True:
            missing.append("confirm")
        return missing

    def common_years(self, companies):
        sets = [set(self.directory.get(c, [])) for c in companies]
        return sorted(set.intersection(*sets)) if sets else []

    def ask(self, conversation, message, missing):
        conversation.pending = missing
        slots, questions = conversation.slots, []
        covered = [c for c in slots.companies if c in self.directory]
        for slot in missing:
            if slot == "intent":
                subject = name(covered[0]) if covered else "the company"
                questions.append({"slot": "intent", "question": f"What would you like to know about {subject}?",
                                  "options": list(INTENT_CHOICES)})
            elif slot in {"company", "companies"}:
                prefix = (f"{', '.join(conversation.unknown_companies)} is not in the corpus. "
                          if conversation.unknown_companies else "")
                ask = "Which companies should I compare (two or more)?" if slot == "companies" else "Which company?"
                questions.append({"slot": slot, "question": prefix + ask,
                                  "options": [f"{name(t)} ({t})" for t in self.directory]})
            elif slot == "year":
                available = self.common_years(covered)
                outside = [y for y in slots.years if y not in available]
                prefix = (f"FY{', FY'.join(map(str, outside))} is not in the filings I have. " if outside else "")
                who = " and ".join(name(c) for c in covered)
                questions.append({"slot": "year", "question": f"{prefix}Which fiscal year for {who}?",
                                  "options": [f"FY{y}" for y in reversed(available)] + ["Latest"]})
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
        return self.result(conversation, "needs_clarification", message=text, questions=questions)

    def ready(self, conversation, forced=False):
        """Fill stated defaults and hand over to the agent (answer) or the report generator (report)."""
        conversation.pending = []
        slots = conversation.slots
        covered = [c for c in slots.companies if c in self.directory]
        available = self.common_years(covered) if covered else []
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
        if (said.intent, said.companies, said.years, said.metric) == (slots.intent, slots.companies, slots.years, slots.metric):
            return original
        names = [name(c) for c in slots.companies if c in self.directory]
        span = slots.years if len(slots.years) < 2 else [slots.years[0], slots.years[-1]]
        years = "–".join(f"FY{y}" for y in span)
        metric = slots.metric or "key financial results"
        if slots.intent == "comparison":
            return f"Compare {metric} for {' and '.join(names)} in {years}."
        if slots.intent == "trend":
            return f"How did {names[0]}'s {metric} change across {years}?"
        if slots.intent == "figure":
            return f"What was {names[0]}'s {metric} for {years}?"
        return f"{original} ({', '.join(names)}, {years})"

    def report_spec(self, conversation):
        slots = conversation.slots
        ticker = next((c for c in slots.companies if c in self.directory), None)
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


class ConversationStore:
    """In-memory, per-process conversation state with a TTL (like the API rate limiter)."""

    def __init__(self, ttl_seconds=1800, max_conversations=10_000):
        self.ttl, self.max = ttl_seconds, max_conversations
        self.items: dict[str, Conversation] = {}
        self.lock = threading.Lock()

    def get(self, conversation_id=None):
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

    def drop(self, conversation_id):
        with self.lock:
            self.items.pop(conversation_id, None)
