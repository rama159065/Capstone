"""Pipeline 2 (Chroma only): table, chart and picture pages are read by OCR or a VLM, then chunks go to Chroma.

The two financial ingestion pipelines over the same manifests (one PDF file, or the
sector corpus folder of scripts/downloads/download_annual_report_corpus.py):

* Pipeline 1, ``scripts.rag.financial_ingest_qdrant_db``: ColPali page vectors in Qdrant
  ``visual_pages``. Mean-pooled rows and columns are kept in RAM for the first stage,
  and the full 1030 x 128 multivectors are kept on disk for the MaxSim rerank.
  Pages are found from their pixels.
* Pipeline 2 (this module), **Chroma only**: pages are turned into text first. Pages with a
  table, chart or picture (or every page, with ``--vlm-pages all``) are rendered and read by
  the page reader chosen with ``--vlm`` (OCR, a local Qwen VLM, or an OpenAI-compatible API,
  self-hosted or paid): tables become Markdown tables, and charts become one line per
  labelled value. Plain narrative pages keep their accurate, free text layer. The text is
  chunked (table headers are repeated on every piece), embedded and written to the Chroma
  collection ``annual_reports_vlm``. A BM25 index of the same chunks (SQLite FTS5 over
  number-aware terms, ``<root>/chroma/bm25.sqlite3``) is fused with the dense results by RRF,
  so exact figures such as ``391,035`` or ``$6.08`` are matched as whole terms.

Every chunk carries the filing payload of pipeline 1 (ticker, fiscal year, sector,
page, citable ``chunk_id``), ``text_source`` = vlm / native_pdf / ocr, and
````unverified_numbers``: figures the VLM wrote that the page's own text layer does
not print. A VLM can misread digits, so these are flagged, never trusted silently.
``number_check`` = no_text_layer marks pages (scanned or font-encoded) that had nothing
to check against: that is where the VLM adds the most, and its numbers are the least checked.

Transcriptions are cached per page in ``<cache-dir>/<document_id>.jsonl``, and
finished filings are recorded in ``<root>/vlm_ingest.sqlite3``. A Colab disconnect
loses at most the page in progress. Put ``--cache-dir`` on Drive.

Page readers (``--vlm``):
  ocr           RapidOCR (CPU, free) plus PyMuPDF's native table extraction as Markdown. Reads printed text,
                labels and data labels in charts and images; no understanding of chart layout.
  transformers  Qwen3-VL in-process (default Qwen/Qwen3-VL-4B-Instruct; Qwen3-VL-2B / -8B with --vlm-model;
                4-bit on a CUDA GPU when bitsandbytes is installed; needs transformers >= 4.57). Colab T4/L4.
  openai        any OpenAI-compatible vision endpoint (RAG_VISION_API_BASE, RAG_VISION_MODEL,
                RAG_VISION_API_KEY): self-hosted Qwen-VL on vLLM / llama.cpp / Ollama, or a paid API
                (OpenAI, Google Gemini's OpenAI endpoint, OpenRouter, ...). HTTP 429/503 are retried after
                RAG_VISION_RETRY_DELAYS seconds (default 2,4,8). Paid APIs bill per page: cap the spend with
                --vlm-max-pages-per-pdf and --vlm-pages tables_charts.
  mock          copies the text layer; offline smoke runs and tests.
  routed        a reader per page, chosen by the page audit rules (scripts/rag/page_audit.route):
                  none -> skipped; pdfParser -> text layer; pdfParser+colpali -> text layer + native tables;
                  ocr -> RapidOCR; qwen3VL -> Qwen3-VL (--qwen-reader), figures checked against the text layer;
                  paidVL -> the fallback chain: OCR the page, Qwen3-VL transcribes it, every figure is checked
                  against the OCR text (the text layer is unusable on these pages); only when more than
                  --paid-threshold of the figures are unconfirmed (or OCR found too little text to check) does the
                  page go to the paid model (RAG_PAID_VISION_API_BASE / _MODEL / _API_KEY). The reading with fewer
                  unconfirmed figures is kept; its unconfirmed figures stay in unverified_numbers. Without a paid
                  model configured the Qwen3-VL reading is kept, flagged.
Every reader's output is cached per page (routed: per reader), so switching readers re-reads pages but a rerun
never does, and changing the paid model or the threshold reuses the OCR and Qwen3-VL readings.

    python -m scripts.rag.financial_ingest_chroma_db ingest --manifest data/raw/financial_report_qa/annual_reports --sectors banking --vlm ocr
    python -m scripts.rag.financial_ingest_chroma_db ingest --manifest data/raw/financial_report_qa/annual_reports --sectors banking --vlm transformers
    python -m scripts.rag.financial_ingest_chroma_db ingest --vlm openai --tickers CVX --max-pages-per-pdf 20      # local llama.cpp
    # paid API, e.g. OpenAI:
    #   RAG_VISION_API_BASE=https://api.openai.com/v1 RAG_VISION_MODEL=gpt-4o-mini RAG_VISION_API_KEY=sk-...
    python -m scripts.rag.financial_ingest_chroma_db ingest --vlm openai --root indexes/financial_vlm_gpt4o --vlm-pages tables_charts
    # one reader per page, paid model only where Qwen3-VL disagrees with OCR:
    #   RAG_PAID_VISION_API_BASE=https://api.openai.com/v1 RAG_PAID_VISION_MODEL=gpt-4o RAG_PAID_VISION_API_KEY=sk-...
    python -m scripts.rag.financial_ingest_chroma_db ingest --vlm routed --qwen-reader transformers --paid-threshold 0.1
    python -m scripts.rag.financial_ingest_chroma_db query "net interest income" --sector banking
    python -m scripts.rag.financial_ingest_chroma_db status
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.request

import pymupdf

from scripts.common.companies import normalize_ticker
from scripts.common.paths import ROOT
from scripts.common.sector_corpus import parse_sectors
from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import hybrid

DEFAULT_ROOT = Path(os.getenv("RAG_VLM_DIR", ROOT / "indexes/financial_vlm"))
CHROMA_COLLECTION = "annual_reports_vlm"
PROMPT_VERSION = 1
VLM_PROMPT = """Transcribe this annual report page into Markdown. Copy every number exactly as printed.
- Tables: a Markdown table with the printed column headers. Keep the unit line (e.g. "in millions"). Keep every row.
- Charts: state the chart title, then one line per labelled value saying which series, bar or year it belongs to, using the legend. Never estimate an unlabelled point.
- Text: copy headings and paragraphs. Ignore photographs, logos and QR codes.
Output only the transcription."""
MAX_PIXELS = 1280 * 28 * 28            # ~1.0 MP per page: ~980 visual tokens for Qwen3-VL (32 x 32 px each)
RENDER_DPI = 150
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,}")   # 1,234 / 6.08 / 2025; not "31,"


# ----------------------------------------------------------------------------- VLM backends

def render(pdf_page, dpi=RENDER_DPI, max_pixels=MAX_PIXELS):
    """Page -> RGB image, shrunk to the VLM's pixel budget."""
    from PIL import Image
    pixmap = pdf_page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image


class MockVLM:
    """Returns the page's text layer: the pipeline end to end without a model."""
    signature = "mock-v1"

    def transcribe(self, image, page):
        return fi.clean(page.get_text(sort=True))


class OcrReader:
    """RapidOCR on the page image plus the page's native tables as Markdown (CPU, no model download).

    OCR copies printed characters: table cells, chart titles, axis and data labels. It does not know which
    label belongs to which bar, so a chart with no printed values gives labels only.
    """

    def __init__(self, scale=2.0, min_confidence=.6, ocr=None):
        self.scale, self.min_confidence = scale, min_confidence
        self._ocr = ocr
        self.signature = ocr_reader_signature(scale, min_confidence)

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = fi.PageOCR(scale=self.scale, min_confidence=self.min_confidence)
        return self._ocr

    def transcribe(self, image, page):
        try:
            tables = [t for t in map(fi.table_markdown, page.find_tables().tables) if t]
        except Exception:
            tables = []
        parts = ["\n".join(lines) for lines in tables]
        text = self.ocr.read(page).strip()
        return "\n\n".join([*parts, text] if text else parts)


def ocr_reader_signature(scale=2.0, min_confidence=.6):
    return f"ocr:{fi.ocr_signature(scale, min_confidence)}+native_tables"


class TransformersVLM:
    def __init__(self, model="Qwen/Qwen3-VL-4B-Instruct", max_new_tokens=1200):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.max_new_tokens = torch, max_new_tokens
        options = {"low_cpu_mem_usage": True}
        quant = "none"
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                options.update(quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True), device_map={"": 0})
                quant = "nf4"
            except ImportError:
                options.update(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map={"": 0})
        else:
            options.update(dtype=torch.float32)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, **options).eval()
        self.signature, self.quantization = f"transformers:{model}:{max_new_tokens}", quant

    def transcribe(self, image, page):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": VLM_PROMPT}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


class OpenAIVLM:
    """Any OpenAI-compatible /chat/completions endpoint that accepts image_url parts."""

    def __init__(self, base=None, model=None, max_tokens=1200, timeout=300, api_key=None):
        self.base = (base or os.getenv("RAG_VISION_API_BASE", "http://127.0.0.1:8081/v1")).rstrip("/")
        self.model = model or os.getenv("RAG_VISION_MODEL", "local-qwen3-vl")
        self.api_key = api_key if api_key is not None else os.getenv("RAG_VISION_API_KEY", "")
        self.max_tokens, self.timeout = max_tokens, timeout
        self.signature = f"openai:{self.model}:{max_tokens}"
        self.delays = [float(x) for x in os.getenv("RAG_VISION_RETRY_DELAYS", "2,4,8").split(",") if x.strip()]

    def transcribe(self, image, page):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}},
                   {"type": "text", "text": VLM_PROMPT}]
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}],
                           "temperature": 0, "max_tokens": self.max_tokens, "stream": False}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base + "/chat/completions", data=body, headers=headers)
        for attempt in range(len(self.delays) + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return (json.load(response)["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as exc:
                # Rate limits and overload are explicit rejections, so a retry can't double-bill a page.
                if exc.code not in {429, 503} or attempt >= len(self.delays):
                    raise
                try:
                    wait = max(self.delays[attempt], float(exc.headers.get("Retry-After") or 0))
                except (TypeError, ValueError):
                    wait = self.delays[attempt]
                time.sleep(wait)


DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen3-VL-4B-Instruct"   # 2B (fast, T4) / 4B (T4 in 4-bit) / 8B (L4, A100)


def make_qwen(args):
    """The Qwen3-VL reader of --vlm routed (--qwen-reader): in-process, an OpenAI-compatible endpoint, or mock."""
    kind = getattr(args, "qwen_reader", "transformers")
    if kind == "mock":
        return MockVLM()
    if kind == "openai":
        return OpenAIVLM(model=args.vlm_model, max_tokens=args.max_new_tokens)
    return TransformersVLM(args.vlm_model or DEFAULT_TRANSFORMERS_MODEL, args.max_new_tokens)


def qwen_signature(args):
    kind = getattr(args, "qwen_reader", "transformers")
    if kind == "mock":
        return MockVLM.signature
    if kind == "openai":
        return OpenAIVLM(model=args.vlm_model, max_tokens=args.max_new_tokens).signature
    return f"transformers:{args.vlm_model or DEFAULT_TRANSFORMERS_MODEL}:{args.max_new_tokens}"


def paid_reader(max_tokens=1200):
    """The paid fallback reader from RAG_PAID_VISION_API_BASE / _MODEL / _API_KEY, or None when not configured."""
    base, model = os.getenv("RAG_PAID_VISION_API_BASE"), os.getenv("RAG_PAID_VISION_MODEL")
    if not (base and model):
        return None
    return OpenAIVLM(base=base, model=model, max_tokens=max_tokens, api_key=os.getenv("RAG_PAID_VISION_API_KEY", ""))


def make_vlm(args):
    if args.vlm == "mock":
        return MockVLM()
    if args.vlm == "routed":
        return RoutedReader(qwen_factory=lambda: make_qwen(args), qwen_signature=qwen_signature(args),
                            paid=paid_reader(args.max_new_tokens), threshold=args.paid_threshold,
                            paid_limit=args.paid_max_pages_per_pdf)
    if args.vlm == "ocr":
        return OcrReader()
    if args.vlm == "openai":
        return OpenAIVLM(model=args.vlm_model, max_tokens=args.max_new_tokens)
    return TransformersVLM(args.vlm_model or DEFAULT_TRANSFORMERS_MODEL, args.max_new_tokens)


def vlm_signature(args):
    """Known before the model loads, so finished filings are skipped without loading it."""
    if args.vlm == "mock":
        return MockVLM.signature
    if args.vlm == "routed":
        paid = paid_reader(args.max_new_tokens)
        return (f"routed-v{ROUTE_VERSION}:qwen={qwen_signature(args)}:ocr={ocr_reader_signature()}:"
                f"paid={paid.signature if paid else None}:threshold={args.paid_threshold}:limit={args.paid_max_pages_per_pdf}")
    if args.vlm == "ocr":
        return ocr_reader_signature()
    if args.vlm == "openai":
        return OpenAIVLM(model=args.vlm_model, max_tokens=args.max_new_tokens).signature
    return f"transformers:{args.vlm_model or DEFAULT_TRANSFORMERS_MODEL}:{args.max_new_tokens}"


# ----------------------------------------------------------------------------- transcription cache

def strip_fences(markdown):
    """Models often wrap the answer in ```markdown fences."""
    return re.sub(r"^```[a-z]*\s*\n|\n?```\s*$", "", markdown.strip()).strip()


def unverified_numbers(markdown, native_text):
    """Figures the VLM wrote that the page's text layer doesn't print.

    None when the page has no usable text layer (scanned, outlined or font-encoded text): nothing to check against.
    """
    if len(re.findall(r"[a-z]{3,}", native_text.lower())) < 15:
        return None
    printed = {n.replace(",", "") for n in NUMBER.findall(native_text)}
    return sorted({n for n in NUMBER.findall(markdown) if n.replace(",", "") not in printed})


class TranscriptCache:
    """<dir>/<document_id>.jsonl, one line per page; a line cut by a disconnect is ignored."""

    def __init__(self, directory, signature):
        self.dir, self.signature = Path(directory), signature
        self.dir.mkdir(parents=True, exist_ok=True)

    def load(self, doc_id):
        path, pages = self.dir / f"{doc_id}.jsonl", {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("signature") == self.signature and row.get("prompt") == PROMPT_VERSION:
                    pages[int(row["page"])] = row
        return pages

    def add(self, doc_id, row):
        with (self.dir / f"{doc_id}.jsonl").open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({**row, "signature": self.signature, "prompt": PROMPT_VERSION}, ensure_ascii=False) + "\n")


def vlm_targets(pages, mode, limit=0):
    targets = fi.visual_targets(pages, mode)               # the same page selection as ColPali in pipeline 1
    return targets[:limit] if limit else targets


ROUTE_VERSION = 1


class RoutedReader:
    """One reader per page (page_audit.route), with the paidVL fallback: OCR -> Qwen3-VL -> paid model.

    The Qwen3-VL model loads only when the first page needs it. ``paid`` is None when no paid model is configured.
    """

    def __init__(self, qwen_factory, qwen_signature, ocr=None, paid=None, threshold=.10, paid_limit=0):
        self._qwen_factory, self._qwen, self.qwen_signature = qwen_factory, None, qwen_signature
        self.ocr = ocr or OcrReader()
        self.paid, self.threshold, self.paid_limit = paid, threshold, paid_limit
        self.signature = (f"routed-v{ROUTE_VERSION}:qwen={qwen_signature}:ocr={self.ocr.signature}:"
                          f"paid={paid.signature if paid else None}:threshold={threshold}:limit={paid_limit}")

    @property
    def qwen(self):
        if self._qwen is None:
            self._qwen = self._qwen_factory()
        return self._qwen

    def readers(self):
        """name -> (reader signature, reader getter): each reading is cached under its own signature."""
        out = {"ocr": (self.ocr.signature, lambda: self.ocr), "qwen3VL": (self.qwen_signature, lambda: self.qwen)}
        if self.paid:
            out["paidVL"] = (self.paid.signature, lambda: self.paid)
        return out


def cross_check(markdown, reference):
    """Figures in ``markdown`` that ``reference`` doesn't print, and their share of all figures.

    share None: the reference has too little text to check against.
    """
    unverified = unverified_numbers(markdown, reference)
    figures = {n.replace(",", "") for n in NUMBER.findall(markdown)}
    if unverified is None:
        return {"unverified": [], "figures": len(figures), "share": None}
    return {"unverified": unverified, "figures": len(figures), "share": len(unverified) / len(figures) if figures else 0.0}


def routed_filing(reader, cache, row, pdf, pages, limit=0):
    """Page number -> reading for --vlm routed. Raw readings are cached per reader; the routing is recomputed."""
    from scripts.rag import page_audit
    doc_id = row["document_id"]
    caches = {name: TranscriptCache(cache.dir, sig) for name, (sig, _) in reader.readers().items()}
    loaded = {name: c.load(doc_id) for name, c in caches.items()}
    getters = {name: get for name, (_, get) in reader.readers().items()}
    entries, model_pages, paid_pages = {}, 0, 0

    with pymupdf.open(pdf) as document:
        def read(name, page):
            if page["page"] not in loaded[name]:
                started, pdf_page = time.time(), document[page["page"] - 1]
                image = None if name == "ocr" else render(pdf_page)
                text = strip_fences(getters[name]().transcribe(image, pdf_page))
                row_ = {"page": page["page"], "markdown": text, "reader": name, "seconds": round(time.time() - started, 1)}
                caches[name].add(doc_id, row_)
                loaded[name][page["page"]] = row_
                print(json.dumps({"doc_id": doc_id, "page": page["page"], "reader": name, "chars": len(text),
                                  "seconds": row_["seconds"]}), flush=True)
            return loaded[name][page["page"]]["markdown"]

        for page in pages:
            route = page_audit.route(page)
            number = page["page"]
            base = {"page": number, "route": route}
            if route == "none":
                entries[number] = {**base, "skip": True}
            elif route == "pdfParser":
                continue                                            # native text, chunked by vlm_records
            elif route == "pdfParser+colpali":
                tables = ["\n".join(lines) for lines in page.get("tables") or []]
                body = "\n\n".join([*tables, fi.readable_text(page)[0]])
                entries[number] = {**base, "markdown": body, "source": "native_pdf", "reader": "pdfParser",
                                   "unverified_numbers": [], "number_check": "text_layer"}
            elif route == "ocr" or (limit and model_pages >= limit):
                entries[number] = {**base, "markdown": read("ocr", page), "source": "ocr", "reader": "ocr",
                                   "unverified_numbers": [], "number_check": "ocr_only"}
            elif route == "qwen3VL":
                model_pages += 1
                markdown = read("qwen3VL", page)
                check = cross_check(markdown, "" if page["garbled"] else page["text"])
                entries[number] = {**base, "markdown": markdown, "source": "vlm", "reader": "qwen3VL",
                                   "unverified_numbers": check["unverified"],
                                   "number_check": "text_layer" if check["share"] is not None else "no_text_layer"}
            else:                                                   # paidVL: the fallback chain
                model_pages += 1
                reference = read("ocr", page)
                qwen_text = read("qwen3VL", page)
                qwen_check = cross_check(qwen_text, reference)
                needs_more = qwen_check["share"] is None or qwen_check["share"] > reader.threshold
                can_pay = reader.paid is not None and (not reader.paid_limit or paid_pages < reader.paid_limit)
                chosen, check, used = qwen_text, qwen_check, "qwen3VL"
                if needs_more and can_pay:
                    paid_pages += 1
                    paid_text = read("paidVL", page)
                    paid_check = cross_check(paid_text, reference)
                    if (paid_check["share"] or 0) <= (qwen_check["share"] if qwen_check["share"] is not None else 1):
                        chosen, check, used = paid_text, paid_check, "paidVL"
                entries[number] = {**base, "markdown": chosen, "source": "vlm", "reader": used,
                                   "unverified_numbers": check["unverified"],
                                   "number_check": "ocr" if check["share"] is not None else "no_reference",
                                   "qwen_unverified_share": qwen_check["share"],
                                   "escalated": used == "paidVL",
                                   "escalation": ("not_needed" if not needs_more else "paid" if used == "paidVL"
                                                  else "paid_not_better" if needs_more and can_pay
                                                  else "no_paid_model" if reader.paid is None else "paid_limit")}
    return entries


def transcribe_filing(vlm, cache, row, pdf, pages, mode, limit=0):
    """Page number -> cached transcription row, transcribing what is missing."""
    if isinstance(vlm, RoutedReader):
        return routed_filing(vlm, cache, row, pdf, pages, limit)
    done = cache.load(row["document_id"])
    todo = [p for p in vlm_targets(pages, mode, limit) if p["page"] not in done]
    if todo:
        with pymupdf.open(pdf) as document:
            for page in todo:
                started = time.time()
                pdf_page = document[page["page"] - 1]
                markdown = strip_fences(vlm.transcribe(render(pdf_page), pdf_page))
                unverified = unverified_numbers(markdown, "" if page["garbled"] else page["text"])
                entry = {"page": page["page"], "markdown": markdown, "unverified_numbers": unverified or [],
                         "number_check": "text_layer" if unverified is not None else "no_text_layer",
                         "seconds": round(time.time() - started, 1)}
                cache.add(row["document_id"], entry)
                done[page["page"]] = entry
                print(json.dumps({"doc_id": row["document_id"], "page": page["page"], "chars": len(markdown),
                                  "number_check": entry["number_check"], "unverified": len(entry["unverified_numbers"]),
                                  "seconds": entry["seconds"]}), flush=True)
    return done


# ----------------------------------------------------------------------------- chunks

def markdown_parts(markdown):
    """('text', chunk) and ('table', chunk) parts; a table keeps its header and the line before it on every piece."""
    lines, prose, parts, i = markdown.splitlines(), [], [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: ") and "-" in lines[i + 1]:
            header, i = lines[i:i + 2], i + 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            title = next((p for p in reversed(prose) if p.strip()), "")[:200]
            for piece in fi.split_table(header + rows):
                parts.append(("table", (title + "\n" if title else "") + "\n".join(piece)))
        else:
            prose.append(lines[i])
            i += 1
    parts += [("text", chunk) for chunk in fi.windows("\n".join(prose)) if len(chunk) >= 40]
    return parts


def vlm_records(row, pages, transcripts, vlm_signature):
    """(id, payload, text to embed) for every chunk of the filing: VLM Markdown where transcribed, else the text layer."""
    base, records = fi.filing_payload(row), []
    for page in pages:
        entry = transcripts.get(page["page"])
        if entry and entry.get("skip"):
            continue                                                # routed: nothing to read on this page
        if entry:
            body = entry["markdown"]
            source = entry.get("source") or ("ocr" if vlm_signature.startswith("ocr:") else "vlm")
        else:
            body, source = fi.readable_text(page)
        if not body.strip():
            continue
        common = {**base, "page": page["page"], "section": page["section"], "page_type": page["page_type"],
                  "scale": page["scale"], "text_source": source, "pipeline": "vlm",
                  "vlm_model": vlm_signature if entry else None,
                  "unverified_numbers": entry["unverified_numbers"] if entry else [],
                  "number_check": entry.get("number_check", "text_layer") if entry else None,
                  "page_route": entry.get("route") if entry else None,
                  "reader": entry.get("reader") if entry else "pdfParser",
                  "escalation": entry.get("escalation") if entry else None}
        for index, (kind, chunk) in enumerate(markdown_parts(body) if entry else [("text", c) for c in fi.windows(body)]):
            if entry is None and len(chunk) < 40:
                continue
            payload = {**common, "chunk_id": fi.chunk_name(row["document_id"], page["page"], f"vlm{index}"),
                       "chunk_index": index, "content_type": kind, "text": chunk,
                       "unverified_in_chunk": [n for n in common["unverified_numbers"] if n in chunk]}
            identifier = fi.point_id(row["document_id"], page["page"], "vlm", index)
            records.append((identifier, payload, fi.context_line(payload) + "\n" + chunk))
    return records


# ----------------------------------------------------------------------------- vector stores

class Bm25Index:
    """BM25 over the same chunks as the Chroma collection: SQLite FTS5 on number-aware terms.

    Chroma's local store has dense search only. Terms come from ``financial_ingest_qdrant_db.terms`` (pipeline 1's
    sparse vectors), so ``$93,736`` is indexed as ``$93,736``, ``93,736`` and ``93736``. The file sits in the
    Chroma folder, so backups and restores of that folder carry it.
    """

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(terms, chunk_id UNINDEXED, "
                        "doc_id UNINDEXED, ticker UNINDEXED, fiscal_year UNINDEXED, sector_slug UNINDEXED, "
                        "tokenize = \"unicode61 tokenchars '$,.%'\")")

    def replace(self, doc_id, records):
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", [
                (" ".join(fi.terms(text)), payload["chunk_id"], doc_id, payload.get("ticker"),
                 payload.get("fiscal_year"), payload.get("sector_slug")) for identifier, payload, text in records])

    def search(self, query, top_k, ticker=None, year=None, sectors=()):
        words = sorted(set(fi.terms(query)))
        if not words:
            return []
        sql = "SELECT chunk_id FROM chunks WHERE chunks MATCH ?"
        params = [" OR ".join('"' + w.replace('"', '""') + '"' for w in words)]
        if ticker:
            sql, params = sql + " AND ticker = ?", params + [ticker.upper()]
        if year:
            sql, params = sql + " AND fiscal_year = ?", params + [int(year)]
        if sectors:
            sql += f" AND sector_slug IN ({','.join('?' * len(sectors))})"
            params += list(sectors)
        return [row[0] for row in self.db.execute(sql + " ORDER BY bm25(chunks) LIMIT ?", params + [top_k])]

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


class ChromaStore:
    def __init__(self, root, collection=CHROMA_COLLECTION):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(Path(root) / "chroma"))
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.bm25 = Bm25Index(Path(root) / "chroma" / "bm25.sqlite3")

    @staticmethod
    def metadata(payload):
        """Chroma metadata holds scalars only: lists become JSON, None is dropped."""
        out = {}
        for key, value in payload.items():
            if key == "text" or value is None:
                continue
            out[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        return out

    def replace(self, doc_id, records, vectors):
        self.collection.delete(where={"doc_id": doc_id})
        for start in range(0, len(records), 256):
            part, vecs = records[start:start + 256], vectors[start:start + 256]
            self.collection.upsert(ids=[r[0] for r in part], embeddings=vecs, documents=[r[1]["text"] for r in part],
                                   metadatas=[self.metadata(r[1]) for r in part])
        self.bm25.replace(doc_id, records)

    def search(self, vector, query, top_k, ticker=None, year=None, sector=None, candidates=50, mode="hybrid"):
        """Dense (Chroma) + BM25 (FTS5) rankings fused by ``hybrid.rrf_fuse``: [(rrf score, payload, ranks)].
        ``mode``: hybrid (default), dense or bm25."""
        conditions = [{"ticker": ticker.upper()}] if ticker else []
        conditions += [{"fiscal_year": year}] if year else []
        sectors = sorted(parse_sectors(sector) or [])
        conditions += [{"sector_slug": sectors[0]} if len(sectors) == 1 else {"sector_slug": {"$in": sectors}}] if sectors else []
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions} if conditions else None
        found, rankings = {}, {}
        if hybrid.uses(mode, "dense"):
            result = self.collection.query(query_embeddings=[vector], n_results=max(top_k, candidates), where=where,
                                           include=["documents", "metadatas"])
            found = {meta["chunk_id"]: {**meta, "text": text} for text, meta in
                     zip(result["documents"][0], result["metadatas"][0])}
            rankings["dense"] = list(found)
        if hybrid.uses(mode, "bm25"):
            rankings["bm25"] = self.bm25.search(query, max(top_k, candidates), ticker, year, sectors)
            missing = [c for c in rankings["bm25"] if c not in found]
            if missing:
                extra = self.collection.get(where={"chunk_id": {"$in": missing}}, include=["documents", "metadatas"])
                found.update({meta["chunk_id"]: {**meta, "text": text}
                              for text, meta in zip(extra["documents"], extra["metadatas"])})
        fused = [e for e in hybrid.rrf_fuse(rankings) if e["id"] in found][:top_k]
        return [(e["rrf"], found[e["id"]], e["ranks"]) for e in fused]

    def count(self):
        return self.collection.count()


def open_store(args, dim):
    """Pipeline 2 writes to Chroma only; Qdrant is pipeline 1's database (scripts/rag/financial_ingest_qdrant_db.py)."""
    return ChromaStore(args.root)


# ----------------------------------------------------------------------------- ingest

class VlmIngest:
    def __init__(self, args, *, vlm=None, embedder=None):
        self.args, self.root = args, Path(args.root)
        if args.reset:
            reset(args)
        self.state = fi.State(self.root / "vlm_ingest.sqlite3")
        self.embedder = embedder or fi.Embedder(args.embedding_provider, args.embedding_model)
        self.state.check("embedding", self.embedder.signature)
        self.state.check("store", args.store)
        self._vlm = vlm
        self.vlm_signature = vlm.signature if vlm else vlm_signature(args)
        self.cache = TranscriptCache(args.cache_dir or self.root / "vlm_pages", self.vlm_signature)
        self.store = open_store(args, self.embedder.dim)
        self.pipeline = json.dumps({"analysis": fi.ANALYSIS_VERSION, "max_pages": args.max_pages_per_pdf,
                                    "embedding": self.embedder.signature, "vlm": self.vlm_signature,
                                    "prompt": PROMPT_VERSION, "pages": args.vlm_pages, "limit": args.vlm_max_pages_per_pdf,
                                    "chunks": [fi.CHUNK_CHARS, fi.OVERLAP_CHARS, fi.TABLE_CHARS]}, sort_keys=True)

    @property
    def vlm(self):
        if self._vlm is None:
            self._vlm = make_vlm(self.args)
        return self._vlm

    def run(self):
        rows = fi.read_manifest(self.args.manifest, self.args.sectors)
        if self.args.tickers:
            wanted = {normalize_ticker(t) for t in self.args.tickers.split(",")}
            rows = [r for r in rows if normalize_ticker(r["ticker"]) in wanted]
        rows = fi.in_preset(rows, getattr(self.args, "corpus_preset", "full"))
        rows = rows[:self.args.limit_filings] if self.args.limit_filings else rows
        todo = [r for r in rows if self.args.force or not self.state.done(r["document_id"], "vlm", self.pipeline)]
        print(f"{len(rows)} filings selected, {len(todo)} to ingest with {self.vlm_signature} -> {self.args.store}", flush=True)
        summary = Counter()
        for row in todo:
            summary[self.ingest(row)] += 1
        return dict(summary)

    def ingest(self, row):
        doc_id, started = row["document_id"], time.time()
        self.state.mark(doc_id, "vlm", "processing", self.pipeline)
        try:
            pdf = fi.resolve_pdf(row, self.args.pdf_dir, self.args.download_missing, self.args.user_agent)
            pages = fi.load_pages(self.root, row, pdf, self.args.max_pages_per_pdf)
            transcripts = transcribe_filing(self.vlm, self.cache, row, pdf, pages, self.args.vlm_pages,
                                            self.args.vlm_max_pages_per_pdf)
            records = vlm_records(row, pages, transcripts, self.vlm_signature)
            vectors = self.embedder([text for _, _, text in records]) if records else []
            self.store.replace(doc_id, records, vectors)
        except Exception as exc:
            self.state.mark(doc_id, "vlm", "failed", self.pipeline, seconds=time.time() - started, error=repr(exc))
            print(json.dumps({"doc_id": doc_id, "status": "failed", "error": repr(exc)}), flush=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return "failed"
        seconds = time.time() - started
        self.state.mark(doc_id, "vlm", "complete", self.pipeline, len(records), seconds)
        flagged = sum(1 for e in transcripts.values() if e.get("unverified_numbers"))
        print(json.dumps({"doc_id": doc_id, "sector": row.get("sector"), "status": "complete", "pages": len(pages),
                          "vlm_pages": len(transcripts), "pages_with_unverified_numbers": flagged,
                          "chunks": len(records), "seconds": round(seconds, 1)}), flush=True)
        return "complete"


def reset(args):
    root = Path(args.root)
    import chromadb
    client = chromadb.PersistentClient(path=str(root / "chroma"))
    if CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(CHROMA_COLLECTION)
    bm25 = Bm25Index(root / "chroma" / "bm25.sqlite3")
    with bm25.db:
        bm25.db.execute("DELETE FROM chunks")
    bm25.db.close()
    (root / "vlm_ingest.sqlite3").unlink(missing_ok=True)
    print("vlm collection and state reset (page analysis and transcription caches kept)", flush=True)


# ----------------------------------------------------------------------------- retrieval

def retrieve(query, top_k=5, ticker=None, year=None, sector=None, *, root=DEFAULT_ROOT, store="chroma", embedder=None,
             mode="hybrid"):
    """Hits in the agents' evidence format (chunk_id, text, ticker, year, sector, page, ...)."""
    root = Path(root)
    if embedder is None:
        embedder = fi.recorded_embedder(argparse.Namespace(root=root, embedding_provider="hashing", embedding_model=None),
                                        state_name="vlm_ingest.sqlite3")
    if store != "chroma":
        raise ValueError("pipeline 2 serves from Chroma only; Qdrant is pipeline 1 (RAG_BACKEND=financial_qdrant)")
    backend = ChromaStore(root)
    hits = []
    vector = embedder([query])[0] if hybrid.uses(mode, "dense") else None
    for score, p, ranks in backend.search(vector, query, top_k, ticker, year, sector, mode=mode):
        unverified = p.get("unverified_in_chunk") or []
        hits.append({"chunk_id": p["chunk_id"], "text": p.get("text", ""), "title": p.get("title"),
                     "source_type": p.get("source_type"), "ticker": p.get("ticker"), "year": p.get("fiscal_year"),
                     "sector": p.get("sector"), "page": p.get("page"), "source_url": p.get("source_url"),
                     "local_path": p.get("pdf_name"), "section": p.get("section"), "scale": p.get("scale"),
                     "text_source": p.get("text_source"), "content_type": p.get("content_type"),
                     "unverified_numbers": json.loads(unverified) if isinstance(unverified, str) else unverified,
                     "preview": re.sub(r"\s+", " ", p.get("text", ""))[:500], "score": float(score),
                     "retrieval_paths": ["vlm_text"], "retrieval_ranks": ranks, "metadata": dict(p)})
    return hits


_serving: dict = {}


def retrieve_for_agent(query, top_k=5, ticker=None, year=None, sector=None, mode="hybrid"):
    """Evidence for the financial agents (RAG_BACKEND=financial_vlm) from the index under RAG_VLM_DIR.

    Uses the Chroma index and the embedder recorded when it was built, loaded once per process.
    Hits carry rerank_score/rrf_score like the other backends.
    """
    root = Path(os.getenv("RAG_VLM_DIR", DEFAULT_ROOT))
    store = "chroma"
    key = (str(root), store)
    if _serving.get("key") != key:
        embedder = fi.recorded_embedder(argparse.Namespace(root=root, embedding_provider="hashing", embedding_model=None),
                                        state_name="vlm_ingest.sqlite3")
        _serving.clear()
        _serving.update(key=key, embedder=embedder)
    hits = retrieve(query, top_k, ticker, year, sector, root=root, store=store, embedder=_serving["embedder"], mode=mode)
    for rank, hit in enumerate(hits, 1):
        hit.update(rrf_score=1 / (60 + rank), rerank_score=hit["score"], image_path=None)
    return hits


def status(args):
    state = Path(args.root) / "vlm_ingest.sqlite3"
    report = {}
    if state.exists():
        import sqlite3
        with sqlite3.connect(state) as db:
            report["filings"] = [dict(zip(("status", "filings", "chunks", "seconds"), r)) for r in db.execute(
                "SELECT status, COUNT(*), SUM(points), SUM(seconds) FROM filings GROUP BY status")]
            report["failed"] = [dict(zip(("doc_id", "error"), r)) for r in db.execute(
                "SELECT doc_id, error FROM filings WHERE status='failed'")]
            report["settings"] = dict(db.execute("SELECT key, value FROM settings"))
    cache = Path(args.cache_dir or Path(args.root) / "vlm_pages")
    pages = [json.loads(line) for f in cache.glob("*.jsonl") for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    report["transcribed_pages"] = len(pages)
    report["pages_with_unverified_numbers"] = sum(1 for p in pages if p.get("unverified_numbers"))
    report["readings_by_reader"] = dict(Counter(p.get("reader", "single") for p in pages))
    report["pages_without_text_layer_to_check"] = sum(1 for p in pages if p.get("number_check") == "no_text_layer")
    report["vlm_seconds_per_page"] = round(sum(p.get("seconds", 0) for p in pages) / len(pages), 1) if pages else None
    print(json.dumps(report, indent=2))
    return report


def query(args):
    started = time.time()
    hits = retrieve(args.text, args.top_k, args.ticker, args.year, args.sector, root=args.root, store=args.store)
    for hit in hits:
        print(json.dumps({"score": round(hit["score"], 4), "chunk_id": hit["chunk_id"], "sector": hit["sector"],
                          "page": hit["page"], "source": hit["text_source"], "type": hit["content_type"],
                          "unverified": hit["unverified_numbers"][:5], "preview": hit["preview"][:160]}))
    print(f"{len(hits)} hits in {time.time() - started:.2f}s")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="state, page analysis cache and the Chroma index")
    common.add_argument("--store", choices=["chroma"], default="chroma",
                        help="Chroma only; Qdrant is pipeline 1 (scripts.rag.financial_ingest_qdrant_db)")
    common.add_argument("--cache-dir", type=Path, help="VLM transcriptions (default <root>/vlm_pages); put it on Drive")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common])
    ingest.add_argument("--manifest", type=Path, default=fi.MANIFEST,
                        help="a pdf_manifest.jsonl, or a sector corpus folder (reads <dir>/<sector>/pdf_manifest.jsonl)")
    ingest.add_argument("--pdf-dir", type=Path, help="where the PDFs are (default: next to the manifest)")
    ingest.add_argument("--sectors", help="comma-separated sector names or slugs")
    ingest.add_argument("--tickers", help="comma-separated subset")
    ingest.add_argument("--corpus-preset", choices=["full", "min500"], default="full",
                        help="full = every filing in the manifest; min500 = only the 509-filing variant's filings")
    ingest.add_argument("--limit-filings", type=int)
    ingest.add_argument("--max-pages-per-pdf", type=int, help="smoke runs only")
    ingest.add_argument("--download-missing", action="store_true")
    ingest.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"))
    ingest.add_argument("--vlm", choices=["routed", "ocr", "transformers", "openai", "mock"], default="transformers",
                        help="page reader: routed = one reader per page from the page audit rules, with the paidVL "
                             "fallback (OCR -> Qwen3-VL -> paid model); ocr = RapidOCR + native tables (CPU); "
                             "transformers = local Qwen3-VL; openai = OpenAI-compatible endpoint (self-hosted Qwen-VL "
                             "or a paid API); mock = text layer")
    ingest.add_argument("--qwen-reader", choices=["transformers", "openai", "mock"], default="transformers",
                        help="routed: how Qwen3-VL runs (in-process, or an endpoint at RAG_VISION_API_BASE)")
    ingest.add_argument("--paid-threshold", type=float, default=0.10,
                        help="routed: send a paidVL page to the paid model when more than this share of Qwen3-VL's "
                             "figures is not in the page's OCR text")
    ingest.add_argument("--paid-max-pages-per-pdf", type=int, default=0, help="routed: cap paid calls per filing (0 = no cap)")
    ingest.add_argument("--vlm-model", help=f"HF model for transformers (default {DEFAULT_TRANSFORMERS_MODEL}); "
                                            "served model name for openai (default RAG_VISION_MODEL)")
    ingest.add_argument("--vlm-pages", choices=["visual", "tables_charts_images", "tables_charts", "all"], default="visual",
                        help="visual: tables, charts, pictures and unreadable pages (default); "
                             "tables_charts_images: no unreadable pages; tables_charts: the cheapest with a paid API")
    ingest.add_argument("--vlm-max-pages-per-pdf", type=int, default=0, help="0 = every target page")
    ingest.add_argument("--max-new-tokens", type=int, default=1200)
    ingest.add_argument("--embedding-provider", choices=["local", "hosted", "hashing"], default="local")
    ingest.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ingest.add_argument("--force", action="store_true", help="redo finished filings (transcriptions stay cached)")
    ingest.add_argument("--reset", action="store_true", help="drop the collection and state first")

    sub.add_parser("status", parents=[common])
    search = sub.add_parser("query", parents=[common])
    search.add_argument("text")
    search.add_argument("--ticker")
    search.add_argument("--year", type=int)
    search.add_argument("--sector")
    search.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        VlmIngest(args).run()
    elif args.command == "status":
        status(args)
    else:
        query(args)


if __name__ == "__main__":
    main()
