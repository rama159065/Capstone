"""One chunk schema for both RAG indexes: Qdrant ``annual_reports_text`` and Chroma ``annual_reports_vlm``.

Both ingestion pipelines build every stored chunk with ``chunk_payload`` and both turn search results into agent
evidence with ``evidence_hit``, so a chunk carries the same fields whichever database holds it, and the agent, the API
and the critic see the same keys from either backend.

Storage differences are encodings only: Chroma metadata accepts scalars, so its list fields are stored as JSON strings
and ``None`` values are left out; ``normalize`` turns a Chroma metadata dict back into the full schema.

Fields (``FIELDS``):

    filing      domain, doc_id, filing_id, doc_kind, ticker, company, cik, sector, sector_slug, sic, year, fiscal_year,
                accession_number, filing_date, source_url, source_type, content_sha256, pdf_name, title, doc_title
    page        page, section, page_type, scale, garbled, visual_rich, page_route
    chunk       chunk_id, point_id, chunk_index, content_type (text | table | page_description), text,
                table_markdown (list)
    provenance  pipeline (qdrant_db | chroma_db), text_source (native_pdf | ocr | vlm),
                reader (pdfParser | ocr | qwen3VL | paidVL | openaiVL | mock), reader_model, escalation,
                number_check (text_layer | ocr | ocr_only | no_text_layer | no_reference),
                unverified_numbers (list, the page's), unverified_in_chunk (list, this chunk's)
"""
from __future__ import annotations

import json
import re

FILING_FIELDS = ("domain", "doc_id", "filing_id", "doc_kind", "ticker", "company", "cik", "sector", "sector_slug", "sic",
                 "year", "fiscal_year", "accession_number", "filing_date", "source_url", "source_type", "content_sha256",
                 "pdf_name", "title", "doc_title")
PAGE_FIELDS = ("page", "section", "page_type", "scale", "garbled", "visual_rich", "page_route")
CHUNK_FIELDS = ("chunk_id", "point_id", "chunk_index", "content_type", "text", "table_markdown")
PROVENANCE_FIELDS = ("pipeline", "text_source", "reader", "reader_model", "escalation", "number_check",
                     "unverified_numbers", "unverified_in_chunk")
FIELDS = FILING_FIELDS + PAGE_FIELDS + CHUNK_FIELDS + PROVENANCE_FIELDS
LIST_FIELDS = ("table_markdown", "unverified_numbers", "unverified_in_chunk")

EVIDENCE_KEYS = ("chunk_id", "text", "preview", "title", "company", "ticker", "year", "sector", "page", "section", "scale",
                 "source_type", "source_url", "local_path", "image_path", "content_type", "text_source", "reader",
                 "unverified_numbers", "table_markdown", "score", "rrf_score", "rerank_score", "retrieval_paths",
                 "retrieval_ranks", "metadata")


def chunk_payload(filing: dict, page: dict, *, chunk_id: str, point_id: str, chunk_index: int, content_type: str,
                  text: str, pipeline: str, text_source: str, reader: str, reader_model: str | None = None,
                  page_route: str | None = None, escalation: str | None = None, number_check: str | None = None,
                  unverified_numbers=(), table_markdown=()) -> dict:
    """Every field of ``FIELDS``, in order. ``filing`` is financial_ingest_qdrant_db.filing_payload(row)."""
    unverified = list(unverified_numbers or [])
    payload = {name: filing.get(name) for name in FILING_FIELDS}
    payload.update(page=page["page"], section=page.get("section"), page_type=page.get("page_type"),
                   scale=page.get("scale"), garbled=bool(page.get("garbled")), visual_rich=bool(page.get("visual")),
                   page_route=page_route)
    payload.update(chunk_id=chunk_id, point_id=str(point_id), chunk_index=chunk_index, content_type=content_type,
                   text=text, table_markdown=list(table_markdown or []))
    payload.update(pipeline=pipeline, text_source=text_source, reader=reader, reader_model=reader_model,
                   escalation=escalation, number_check=number_check, unverified_numbers=unverified,
                   unverified_in_chunk=[n for n in unverified if n in text])
    return payload


def normalize(stored: dict) -> dict:
    """A stored payload or Chroma metadata dict -> the full schema (lists decoded, missing fields None or [])."""
    out = dict(stored)
    for name in FIELDS:
        value = out.get(name)
        if name in LIST_FIELDS:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [value]
            out[name] = list(value or [])
        else:
            out[name] = value
    return out


def evidence_hit(stored: dict, *, rrf_score: float, rerank_score: float | None = None, paths=("text",),
                 ranks: dict | None = None) -> dict:
    """A stored chunk -> the agent's evidence dict; the same keys from Qdrant and Chroma (``EVIDENCE_KEYS``)."""
    p = normalize(stored)
    text = p.get("text") or ""
    rerank = rrf_score if rerank_score is None else rerank_score
    return {"chunk_id": p["chunk_id"], "text": text, "preview": re.sub(r"\s+", " ", text)[:500],
            "title": p["title"], "company": p["company"], "ticker": p["ticker"], "year": p["fiscal_year"],
            "sector": p["sector"], "page": p["page"], "section": p["section"], "scale": p["scale"],
            "source_type": p["source_type"], "source_url": p["source_url"], "local_path": p["pdf_name"],
            "image_path": None, "content_type": p["content_type"], "text_source": p["text_source"],
            "reader": p["reader"], "unverified_numbers": p["unverified_in_chunk"], "table_markdown": p["table_markdown"],
            "score": float(rerank), "rrf_score": float(rrf_score), "rerank_score": float(rerank),
            "retrieval_paths": list(paths), "retrieval_ranks": dict(ranks or {}), "metadata": p}
