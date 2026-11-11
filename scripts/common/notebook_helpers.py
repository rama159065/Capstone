"""Helpers shared by the notebooks in notebooks/ (they run on Colab or locally from the repo root).

Everything here wraps existing modules: the EDGAR manifest and PDF download (rag.financial_ingest_qdrant_db),
the chat templates and scorer (evaluation.prompt_baselines), the SEC XBRL lookup
(evaluation.build_financial_eval_set) and the frozen evaluation set.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.common.paths import ROOT
from scripts.evaluation import build_financial_eval_set as evalset
from scripts.evaluation import prompt_baselines as pb
from scripts.rag import financial_ingest_qdrant_db as fi

IN_COLAB = "google.colab" in sys.modules
MANIFEST = fi.MANIFEST
PDF_DIR = MANIFEST.parent / "pdfs"
EVAL_SET = ROOT / "data/evaluation/financial_report_qa/eval_set_v1.jsonl"
SEC_CACHE = ROOT / "data/processed/financial_report_qa/sec_cache"
TEMPLATES = ["zero_shot", "structured_few_shot", "chain_of_thought"]


# ----------------------------------------------------------------------------- corpus

def pick_filings(doc_ids: list[str]) -> list[dict]:
    """Manifest rows for e.g. ["edgar_ars_AMD_2025", ...] (or short forms "AMD_2025")."""
    wanted = [d if d.startswith("edgar_ars_") else f"edgar_ars_{d}" for d in doc_ids]
    rows = {r["document_id"]: r for r in fi.read_manifest(MANIFEST)}
    missing = [d for d in wanted if d not in rows]
    if missing:
        raise KeyError(f"not in the manifest: {missing}")
    return [rows[d] for d in wanted]


def fetch_pdfs(rows: list[dict], pdf_dir: Path = PDF_DIR, user_agent: str | None = None) -> dict[str, Path]:
    """Local PDFs for the rows, downloading missing ones from EDGAR (SHA-256 checked)."""
    user_agent = user_agent or os.getenv("SEC_USER_AGENT")
    out = {}
    for row in rows:
        out[row["document_id"]] = fi.resolve_pdf(row, pdf_dir, download=True, user_agent=user_agent)
    return out


def eval_items(doc_ids: list[str], categories: tuple[str, ...] | None = None, general: bool = True) -> list[dict]:
    """Frozen eval items tied to these filings, plus (optionally) refusal and not-disclosed items."""
    wanted = {d if d.startswith("edgar_ars_") else f"edgar_ars_{d}" for d in doc_ids}
    items = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").split("\n") if line.strip()]
    chosen = []
    for item in items:
        docs = {s.get("doc_id") for s in item.get("sources", []) if s.get("doc_id")}
        if (docs & wanted) or (general and not docs and item["category"] in {"refusal", "not_disclosed"}):
            if categories is None or item["category"] in categories:
                chosen.append(item)
    return chosen


# ----------------------------------------------------------------------------- LLM clients

class TransformersChat:
    """Small instruct model on the Colab GPU, as a stand-in when the provided endpoint is not configured.

    Track B grades the provided hosted <=8B endpoint; use this only to exercise the notebooks.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu").eval()

    def chat(self, system: str, user: str, *, temperature: float, max_tokens: int, seed: int) -> dict:
        import time
        import torch
        torch.manual_seed(seed)
        prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": system},
                                                     {"role": "user", "content": user}],
                                                    tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                         temperature=temperature or None, top_p=0.9 if temperature else None)
        new = output[0][inputs["input_ids"].shape[1]:]
        return {"text": self.tokenizer.decode(new, skip_special_tokens=True), "latency_s": time.perf_counter() - started,
                "prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(len(new))}


def make_llm(provider: str = "auto"):
    """('hosted' | 'transformers' | 'mock' | 'auto') -> (client with .chat(...), description)."""
    if provider == "auto":
        provider = ("hosted" if all(os.getenv(k) for k in ("LLM_API_BASE", "LLM_MODEL")) else
                    "transformers" if IN_COLAB and _gpu() else "mock")
    if provider == "hosted":
        return pb.ChatClient(os.environ["LLM_API_BASE"], os.getenv("LLM_API_KEY", ""), os.environ["LLM_MODEL"]), \
            f"hosted endpoint {os.environ['LLM_MODEL']} (the Track B model)"
    if provider == "transformers":
        model = os.getenv("NB_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        return TransformersChat(model), f"local stand-in {model} (not the graded endpoint)"
    return pb.MockClient(), "mock client: deterministic, offline; checks the pipeline, not model quality"


def _gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- external tool: SEC XBRL

METRICS = {label: (kind, concepts) for label, kind, concepts in evalset.CONCEPTS}
METRIC_ALIASES = {  # clarifier / user wording -> eval-set XBRL label
    "diluted eps": "diluted earnings per share", "eps": "diluted earnings per share",
    "net income": "net income attributable to the company", "operating cash flow": "net cash provided by operating activities",
    "research and development": "research and development expense", "r&d": "research and development expense",
    "shareholders' equity": "total stockholders' equity", "stockholders' equity": "total stockholders' equity",
    "long-term debt": "long-term debt (non-current)", "share repurchases": "cash paid for share repurchases",
    "buybacks": "cash paid for share repurchases", "sales": "revenue", "net sales": "revenue",
}


def metric_label(metric: str) -> str | None:
    metric = METRIC_ALIASES.get(metric.lower().strip(), metric.lower().strip())
    return next((m for m in METRICS if metric == m), None) or next((m for m in METRICS if metric in m or m in metric), None)


def companyfacts(cik: str, user_agent: str | None = None) -> dict:
    user_agent = user_agent or os.getenv("SEC_USER_AGENT")
    if not user_agent or "@" not in user_agent:
        raise ValueError("Set SEC_USER_AGENT to 'name email@domain' (SEC requires a contact address)")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    return json.loads(evalset.get(url, user_agent, SEC_CACHE / f"companyfacts_{int(cik):010d}.json"))


def get_xbrl_fact(ticker: str, metric: str, fiscal_year: int, user_agent: str | None = None) -> dict | None:
    """Exact figure from the company's 10-K XBRL (the external source), as an evidence record."""
    row = next((r for r in fi.read_manifest(MANIFEST) if r["ticker"].upper() == ticker.upper()), None)
    label = metric_label(metric)
    if row is None or label is None:
        return None
    found = evalset.annual_fact(companyfacts(row["cik"], user_agent), METRICS[label][1], int(fiscal_year))
    if not found:
        return None
    concept, unit, fact, prior = found
    scale = "USD per share" if unit.endswith("/shares") else "USD"
    text = (f"SEC XBRL (10-K, accession {fact['accn']}): {row['company']} {label}, period ending {fact['end']} "
            f"(company label fiscal {fact['fy']}): {fact['val']:,} {scale}"
            + (f"; prior period ending {prior['end']}: {prior['val']:,} {scale}" if prior else "") + ".")
    return {"chunk_id": "c1", "doc_id": f"xbrl:{ticker}:{concept}", "page": 0, "text": text,
            "value": fact["val"], "unit": unit, "period_end": fact["end"], "concept": f"us-gaap:{concept}"}


def xbrl_item(ticker: str, metric: str, fiscal_year: int, company: str | None = None) -> dict | None:
    """An eval-set-style question whose gold answer is the exact XBRL figure (for filings the frozen set lacks)."""
    fact = get_xbrl_fact(ticker, metric, fiscal_year)
    if fact is None:
        return None
    label = metric_label(metric)
    per_share = fact["unit"].endswith("/shares")
    gold = f"${fact['value']:,.2f} per share" if per_share else f"${fact['value'] / 1e6:,.0f} million"
    return {"id": f"XBRL-{ticker}-{fiscal_year}-{label.split()[0]}", "category": "single_figure",
            "question": f"What was {company or ticker}'s {label} for fiscal {fiscal_year}?",
            "gold_answer": f"{gold} (period ending {fact['period_end']}; {fact['concept']})", "answerability": "answerable",
            "gold_value": {"value": fact["value"], "unit": "USD/shares" if per_share else "USD",
                           "period_end": fact["period_end"]},
            "sources": [{"type": "xbrl_fact", "concept": fact["concept"]}, {"type": "document", "doc_id": f"edgar_ars_{ticker}_{fiscal_year}"}]}


# ----------------------------------------------------------------------------- templates and scoring

def label_hits(hits: list[dict]) -> list[dict]:
    """Agent/RAG hits -> prompt_baselines evidence records labelled [c1]..[cN]."""
    out = []
    for rank, hit in enumerate(hits, 1):
        meta = hit.get("metadata", hit)
        out.append({"chunk_id": f"c{rank}", "source_chunk_id": hit.get("chunk_id"),
                    "doc_id": meta.get("doc_id") or hit.get("doc_id"), "page": hit.get("page"),
                    "text": hit.get("text", "")})
    return out


def run_template(client, strategy: str, question: str, hits: list[dict], seed: int = 1000) -> dict:
    """One single-prompt call with a Week 1 chat template; returns text, parsed JSON and usage."""
    user = f"Evidence:\n{pb.evidence_block(hits)}\n\nQuestion: {question}\n\n{pb.SUFFIX[strategy]}"
    reply = client.chat(pb.SYSTEM, user, temperature=0.0, max_tokens=pb.MAX_TOKENS[strategy], seed=seed)
    return {**reply, "strategy": strategy, "parsed": pb.parse_json(reply["text"])}


def score(item: dict, reply: dict, hits: list[dict]) -> dict:
    return pb.score(item, reply["text"], reply["parsed"], hits, reply["strategy"])


# ----------------------------------------------------------------------------- display and fakes

def show(obj: Any) -> None:
    try:
        from IPython.display import display
        display(obj)
    except Exception:  # noqa: BLE001 - plain Python run
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


def table(rows: list[dict], columns: list[str] | None = None):
    import pandas as pd
    frame = pd.DataFrame(rows)
    return frame[[c for c in columns if c in frame]] if columns else frame


class HashColPali:
    """Deterministic fake ColPali for machines without a GPU: exercises Qdrant plumbing only, not retrieval quality."""
    signature = fi.colpali_signature("hash-colpali-fake")
    dtype_name = "fake"

    def embed_images(self, images):
        out = []
        for image in images:
            rng = np.random.default_rng(int(hashlib.sha256(image.tobytes()).hexdigest()[:8], 16))
            out.append((fi.l2(rng.normal(size=(1030, 128))), np.arange(1030) < 1024))
        return out

    def embed_query(self, text):
        rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        return fi.l2(rng.normal(size=(16, 128)))
