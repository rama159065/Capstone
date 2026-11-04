"""Verify financial-report PDFs and index them into Chroma for RAG.

Recommended capstone mode:
    set EMBEDDING_API_BASE=https://your-hosted-endpoint/v1
    set EMBEDDING_API_KEY=...
    set EMBEDDING_MODEL=your-embedding-model
    python scripts\\rag\\index_financial_report_pdfs.py --embedding-provider hosted

Local smoke-test mode, no network/model download:
    python scripts\\rag\\index_financial_report_pdfs.py --embedding-provider hashing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_MANIFEST = Path("data/raw/financial_report_qa/pdf_manifest.jsonl")
DEFAULT_CHUNKS_OUT = Path("data/processed/financial_report_qa/pdf_chunks.jsonl")
DEFAULT_CHART_IMAGES_DIR = Path("data/processed/financial_report_qa/chart_images")
DEFAULT_INDEX_DIR = Path("indexes/chroma/financial_report_qa_pdfs")
DEFAULT_COLLECTION = "financial_report_qa_pdfs"


@dataclass(frozen=True)
class PdfReport:
    document_id: str
    title: str
    ticker: str
    company: str
    year: int
    source_url: str
    local_path: Path
    content_sha256: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, object]


class HashingEmbeddings:
    """Deterministic local embedding fallback for indexing smoke tests.

    This is not a substitute for the hosted embedding endpoint in the final
    capstone evaluation, but it lets the pipeline build and query a Chroma DB
    without downloading a model.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype("float32").toarray().tolist()


class HostedEmbeddings:
    """OpenAI-compatible /v1/embeddings client."""

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        request_payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/embeddings",
            data=request_payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "capstone-rag-indexer/0.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = sorted(payload["data"], key=lambda item: item["index"])
        return [row["embedding"] for row in rows]


def read_manifest(path: Path) -> list[PdfReport]:
    reports = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reports.append(
                PdfReport(
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    ticker=str(row["ticker"]),
                    company=str(row["company"]),
                    year=int(row["year"]),
                    source_url=str(row["source_url"]),
                    local_path=Path(row["local_path"]),
                    content_sha256=str(row["content_sha256"]),
                )
            )
    return reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def verify_and_extract(
    report: PdfReport,
    max_pages_per_pdf: int | None = None,
) -> tuple[list[tuple[int, str]], dict[str, object]]:
    if not report.local_path.exists():
        raise FileNotFoundError(report.local_path)
    header = report.local_path.read_bytes()[:4]
    if header != b"%PDF":
        raise ValueError(f"{report.local_path} does not start with %PDF")
    actual_hash = sha256_file(report.local_path)
    if actual_hash != report.content_sha256:
        raise ValueError(f"{report.local_path} SHA-256 mismatch")

    reader = PdfReader(str(report.local_path))
    encrypted = bool(reader.is_encrypted)
    if encrypted:
        reader.decrypt("")

    pages = []
    pages_to_read = reader.pages[:max_pages_per_pdf] if max_pages_per_pdf else reader.pages
    for index, page in enumerate(pages_to_read, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((index, text))

    stats = {
        "document_id": report.document_id,
        "ticker": report.ticker,
        "year": report.year,
        "path": str(report.local_path),
        "page_count": len(reader.pages),
        "pages_read": len(pages_to_read),
        "text_pages": len(pages),
        "encrypted": encrypted,
        "bytes": report.local_path.stat().st_size,
        "sha256": actual_hash,
    }
    return pages, stats


def build_chunks(
    report: PdfReport,
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    records = []
    for page_number, page_text in pages:
        for chunk_index, text in enumerate(splitter.split_text(page_text)):
            text = normalize_text(text)
            if len(text) < 80:
                continue
            chunk_id = f"{report.document_id}_p{page_number:04d}_c{chunk_index:03d}"
            metadata = {
                "domain": "financial_report_qa",
                "source_type": "annual_report_pdf",
                "document_id": report.document_id,
                "title": report.title,
                "ticker": report.ticker,
                "company": report.company,
                "year": report.year,
                "page": page_number,
                "source_url": report.source_url,
                "local_path": str(report.local_path),
                "chunk_id": chunk_id,
            }
            records.append(ChunkRecord(chunk_id, report.document_id, text, metadata))
    return records


def image_extension(image_name: str) -> str:
    suffix = Path(image_name).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif"}:
        return "jpg" if suffix == "jpeg" else suffix
    return "png"


def page_context(page_text: str, max_chars: int = 900) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    useful_lines = [
        line for line in lines
        if len(line) >= 20 or re.search(r"\d", line)
    ]
    context = " ".join(useful_lines[:18]) or " ".join(lines[:18])
    return normalize_text(context)[:max_chars]


def clean_table_cell(value: object) -> str:
    if value is None:
        return ""
    return normalize_text(str(value).replace("\n", " "))


def table_to_markdown(rows: list[list[object]], max_cell_chars: int = 240) -> str:
    cleaned_rows = [
        [clean_table_cell(cell)[:max_cell_chars] for cell in row]
        for row in rows
    ]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    if not cleaned_rows:
        return ""

    max_cols = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    if not any(header):
        header = [f"Column {index + 1}" for index in range(max_cols)]
    separator = ["---"] * max_cols
    markdown_rows = [header, separator, *body]
    return "\n".join(
        "| " + " | ".join(cell.replace("|", "/") for cell in row) + " |"
        for row in markdown_rows
    )


def extract_table_summary_chunks(
    report: PdfReport,
    max_pages_per_pdf: int | None,
    min_rows: int,
    max_tables_per_page: int,
    max_tables_per_pdf: int,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    try:
        with pdfplumber.open(str(report.local_path)) as pdf:
            pages = pdf.pages[:max_pages_per_pdf] if max_pages_per_pdf else pdf.pages
            for page_index, page in enumerate(pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                saved_on_page = 0
                for table_index, table in enumerate(tables):
                    cleaned_table = [
                        row for row in table
                        if row and any(clean_table_cell(cell) for cell in row)
                    ]
                    if len(cleaned_table) < min_rows:
                        continue
                    markdown = table_to_markdown(cleaned_table)
                    if not markdown:
                        continue
                    row_count = len(cleaned_table)
                    column_count = max(len(row) for row in cleaned_table)
                    table_id = (
                        f"{report.document_id}_p{page_index:04d}_"
                        f"table{table_index:03d}"
                    )
                    text = normalize_text(
                        "\n".join(
                            [
                                f"Table extracted from {report.company} annual report {report.year}.",
                                f"Ticker: {report.ticker}.",
                                f"Page: {page_index}.",
                                f"Rows: {row_count}. Columns: {column_count}.",
                                "Table text:",
                                markdown,
                            ]
                        )
                    )
                    metadata = {
                        "domain": "financial_report_qa",
                        "source_type": "table_summary",
                        "document_id": report.document_id,
                        "title": report.title,
                        "ticker": report.ticker,
                        "company": report.company,
                        "year": report.year,
                        "page": page_index,
                        "source_url": report.source_url,
                        "local_path": str(report.local_path),
                        "table_index": table_index,
                        "table_rows": row_count,
                        "table_columns": column_count,
                        "chunk_id": table_id,
                    }
                    records.append(ChunkRecord(table_id, report.document_id, text, metadata))
                    saved_on_page += 1
                    if saved_on_page >= max_tables_per_page:
                        break
                    if len(records) >= max_tables_per_pdf:
                        return records
    except Exception:
        return records
    return records


def run_tesseract_ocr(image_path: Path, ocr_command: str, timeout: int) -> tuple[str, str]:
    executable = shutil.which(ocr_command) or (
        str(Path(ocr_command)) if Path(ocr_command).exists() else None
    )
    if not executable:
        return "", "unavailable"
    try:
        completed = subprocess.run(
            [executable, str(image_path), "stdout", "--psm", "6"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception:
        return "", "error"
    if completed.returncode != 0:
        return "", "error"
    return normalize_text(completed.stdout), "ok"


def extract_chart_summary_chunks(
    report: PdfReport,
    pages: list[tuple[int, str]],
    chart_images_dir: Path,
    max_pages_per_pdf: int | None,
    min_width: int,
    min_height: int,
    max_images_per_page: int,
    max_images_per_pdf: int,
    enable_ocr: bool,
    ocr_command: str,
    ocr_timeout: int,
) -> list[ChunkRecord]:
    """Extract large embedded PDF images as chart/figure retrieval hooks.

    This intentionally stores text summaries plus image paths in Chroma, not
    image embeddings. It keeps the RAG stack simple while preserving a route to
    the original visual evidence for later vision-model review.
    """
    reader = PdfReader(str(report.local_path))
    if reader.is_encrypted:
        reader.decrypt("")

    page_text = dict(pages)
    chart_images_dir.mkdir(parents=True, exist_ok=True)
    pages_to_scan = reader.pages[:max_pages_per_pdf] if max_pages_per_pdf else reader.pages
    records: list[ChunkRecord] = []
    seen_hashes: set[str] = set()

    for page_number, page in enumerate(pages_to_scan, start=1):
        saved_on_page = 0
        try:
            page_images = list(page.images)
        except Exception:
            continue
        for image_index, image_file in enumerate(page_images):
            try:
                width, height = image_file.image.size
            except Exception:
                continue
            if width < min_width or height < min_height:
                continue
            image_hash = hashlib.sha256(image_file.data).hexdigest()
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)

            extension = image_extension(image_file.name)
            image_id = (
                f"{report.document_id}_p{page_number:04d}_"
                f"img{image_index:03d}_{image_hash[:10]}"
            )
            image_path = chart_images_dir / f"{image_id}.{extension}"
            try:
                image_path.write_bytes(image_file.data)
            except Exception:
                continue

            ocr_text = ""
            ocr_status = "disabled"
            if enable_ocr:
                ocr_text, ocr_status = run_tesseract_ocr(
                    image_path=image_path,
                    ocr_command=ocr_command,
                    timeout=ocr_timeout,
                )

            context = page_context(page_text.get(page_number, ""))
            ocr_lines = []
            if enable_ocr:
                ocr_lines = [
                    f"OCR status: {ocr_status}.",
                    "OCR text:",
                    ocr_text or "No OCR text was extracted.",
                ]
            text = normalize_text(
                "\n".join(
                    [
                        f"Chart/figure summary for {report.company} annual report {report.year}.",
                        f"Ticker: {report.ticker}.",
                        f"Source PDF: {report.local_path}.",
                        f"Page: {page_number}.",
                        f"Image path: {image_path}.",
                        f"Image size: {width}x{height}.",
                        "Nearby page text for retrieval context:",
                        context or "No nearby page text was extracted.",
                        *ocr_lines,
                    ]
                )
            )
            chunk_id = f"{image_id}_chart_summary"
            metadata = {
                "domain": "financial_report_qa",
                "source_type": "chart_summary",
                "document_id": report.document_id,
                "title": report.title,
                "ticker": report.ticker,
                "company": report.company,
                "year": report.year,
                "page": page_number,
                "source_url": report.source_url,
                "local_path": str(report.local_path),
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "image_sha256": image_hash,
                "ocr_enabled": enable_ocr,
                "ocr_status": ocr_status,
                "ocr_text_chars": len(ocr_text),
                "chunk_id": chunk_id,
            }
            records.append(ChunkRecord(chunk_id, report.document_id, text, metadata))
            saved_on_page += 1
            if saved_on_page >= max_images_per_page:
                break
            if len(records) >= max_images_per_pdf:
                return records
    return records


def batched(items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_embedder(provider: str, dimensions: int):
    if provider == "hashing":
        return HashingEmbeddings(dimensions=dimensions)
    if provider == "hosted":
        api_base = os.environ.get("EMBEDDING_API_BASE")
        api_key = os.environ.get("EMBEDDING_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL")
        missing = [
            name for name, value in {
                "EMBEDDING_API_BASE": api_base,
                "EMBEDDING_API_KEY": api_key,
                "EMBEDDING_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing hosted embedding env var(s): {', '.join(missing)}")
        return HostedEmbeddings(api_base=api_base, api_key=api_key, model=model)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_query_filter(
    source_type: str | None,
    ticker: str | None,
    year: int | None,
) -> dict[str, object] | None:
    filters: list[dict[str, object]] = []
    if source_type:
        filters.append({"source_type": source_type})
    if ticker:
        filters.append({"ticker": ticker.upper()})
    if year:
        filters.append({"year": year})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]*", text.lower())


def load_chunk_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def metadata_matches_filter(
    metadata: dict[str, object],
    source_type: str | None,
    ticker: str | None,
    year: int | None,
) -> bool:
    if source_type and metadata.get("source_type") != source_type:
        return False
    if ticker and str(metadata.get("ticker", "")).upper() != ticker.upper():
        return False
    if year and metadata.get("year") != year:
        return False
    return True


def bm25_search(
    chunk_rows: list[dict[str, object]],
    query: str,
    top_k: int,
    source_type: str | None = None,
    ticker: str | None = None,
    year: int | None = None,
) -> list[dict[str, object]]:
    filtered_rows = [
        row for row in chunk_rows
        if metadata_matches_filter(
            row.get("metadata", {}),
            source_type=source_type,
            ticker=ticker,
            year=year,
        )
    ]
    if not filtered_rows:
        return []
    tokenized_corpus = [tokenize_for_bm25(str(row.get("text", ""))) for row in filtered_rows]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize_for_bm25(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    hits = []
    for rank, (index, score) in enumerate(ranked[:top_k], start=1):
        row = filtered_rows[index]
        metadata = row.get("metadata", {})
        hits.append(
            {
                "rank": rank,
                "chunk_id": row.get("chunk_id"),
                "bm25_score": float(score),
                "text": row.get("text", ""),
                "metadata": metadata,
            }
        )
    return hits


def hit_from_chunk(
    chunk_id: str,
    text: str,
    metadata: dict[str, object],
    rank: int,
    distance: float | None = None,
    bm25_score: float | None = None,
    rrf_score: float | None = None,
    rerank_score: float | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "distance": distance,
        "bm25_score": bm25_score,
        "rrf_score": rrf_score,
        "rerank_score": rerank_score,
        "document_id": metadata.get("document_id"),
        "source_type": metadata.get("source_type"),
        "title": metadata.get("title"),
        "ticker": metadata.get("ticker"),
        "company": metadata.get("company"),
        "year": metadata.get("year"),
        "page": metadata.get("page"),
        "source_url": metadata.get("source_url"),
        "local_path": metadata.get("local_path"),
        "image_path": metadata.get("image_path"),
        "text": text,
        "preview": normalize_text(text)[:500],
        "metadata": metadata,
    }


def retrieve_top_k(
    collection,
    embedder,
    query: str,
    top_k: int,
    source_type: str | None = None,
    ticker: str | None = None,
    year: int | None = None,
) -> list[dict[str, object]]:
    query_embedding = embedder.embed_documents([query])[0]
    where = build_query_filter(source_type=source_type, ticker=ticker, year=year)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for rank, chunk_id in enumerate(ids, start=1):
        text = documents[rank - 1] or ""
        metadata = metadatas[rank - 1] or {}
        hits.append(
            hit_from_chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                rank=rank,
                distance=distances[rank - 1] if rank - 1 < len(distances) else None,
            )
        )
    return hits


def rrf_score(rank: int, rrf_k: int) -> float:
    return 1.0 / (rrf_k + rank)


def min_max_normalize(value: float | None, max_value: float | None) -> float:
    if value is None or max_value is None or max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def lexical_overlap_score(query: str, text: str) -> float:
    query_terms = {term for term in tokenize_for_bm25(query) if len(term) > 2}
    if not query_terms:
        return 0.0
    text_terms = set(tokenize_for_bm25(text))
    return len(query_terms & text_terms) / len(query_terms)


def rerank_fused_hits(
    query: str,
    fused_hits: list[dict[str, object]],
    reranker: str,
    top_k: int,
) -> list[dict[str, object]]:
    if reranker == "none":
        reranked = fused_hits
    elif reranker == "simple":
        max_bm25 = max(
            [hit.get("bm25_score") or 0.0 for hit in fused_hits],
            default=0.0,
        )
        for hit in fused_hits:
            dense_component = 0.0
            if hit.get("distance") is not None:
                dense_component = 1.0 / (1.0 + float(hit["distance"]))
            bm25_component = min_max_normalize(hit.get("bm25_score"), max_bm25)
            overlap_component = lexical_overlap_score(query, str(hit.get("text", "")))
            hit["rerank_score"] = (
                float(hit.get("rrf_score") or 0.0)
                + 0.20 * overlap_component
                + 0.10 * bm25_component
                + 0.05 * dense_component
            )
        reranked = sorted(
            fused_hits,
            key=lambda hit: (
                hit.get("rerank_score") or 0.0,
                hit.get("rrf_score") or 0.0,
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"Unsupported reranker: {reranker}")

    for rank, hit in enumerate(reranked[:top_k], start=1):
        hit["rank"] = rank
    return reranked[:top_k]


def retrieve_hybrid_top_k(
    collection,
    embedder,
    chunk_rows: list[dict[str, object]],
    query: str,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    rrf_k: int,
    reranker: str,
    source_type: str | None = None,
    ticker: str | None = None,
    year: int | None = None,
) -> list[dict[str, object]]:
    dense_hits = retrieve_top_k(
        collection=collection,
        embedder=embedder,
        query=query,
        top_k=dense_top_k,
        source_type=source_type,
        ticker=ticker,
        year=year,
    )
    lexical_hits = bm25_search(
        chunk_rows=chunk_rows,
        query=query,
        top_k=bm25_top_k,
        source_type=source_type,
        ticker=ticker,
        year=year,
    )

    fused: dict[str, dict[str, object]] = {}
    for dense_rank, hit in enumerate(dense_hits, start=1):
        chunk_id = str(hit["chunk_id"])
        fused[chunk_id] = dict(hit)
        fused[chunk_id]["dense_rank"] = dense_rank
        fused[chunk_id]["rrf_score"] = rrf_score(dense_rank, rrf_k)

    for bm25_rank, hit in enumerate(lexical_hits, start=1):
        chunk_id = str(hit["chunk_id"])
        if chunk_id not in fused:
            metadata = hit.get("metadata", {})
            fused[chunk_id] = hit_from_chunk(
                chunk_id=chunk_id,
                text=str(hit.get("text", "")),
                metadata=metadata,
                rank=bm25_rank,
            )
            fused[chunk_id]["rrf_score"] = 0.0
        fused[chunk_id]["bm25_rank"] = bm25_rank
        fused[chunk_id]["bm25_score"] = hit.get("bm25_score")
        fused[chunk_id]["rrf_score"] = (
            float(fused[chunk_id].get("rrf_score") or 0.0)
            + rrf_score(bm25_rank, rrf_k)
        )

    fused_hits = sorted(
        fused.values(),
        key=lambda hit: hit.get("rrf_score") or 0.0,
        reverse=True,
    )
    return rerank_fused_hits(query, fused_hits, reranker=reranker, top_k=top_k)


def print_retrieval_results(
    query: str,
    hits: list[dict[str, object]],
    retrieval_mode: str,
) -> None:
    payload = {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "top_k": len(hits),
        "hits": hits,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunks-out", type=Path, default=DEFAULT_CHUNKS_OUT)
    parser.add_argument("--chart-images-dir", type=Path, default=DEFAULT_CHART_IMAGES_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-provider", choices=["hosted", "hashing"], default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-pdfs", type=int)
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        help="Optional cap for faster first-pass indexing of large annual reports.",
    )
    parser.add_argument(
        "--include-chart-summaries",
        action="store_true",
        help="Extract large embedded images and index chart_summary chunks with image paths.",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        help="Extract tables with pdfplumber and index table_summary chunks.",
    )
    parser.add_argument("--table-min-rows", type=int, default=2)
    parser.add_argument("--max-tables-per-page", type=int, default=3)
    parser.add_argument("--max-tables-per-pdf", type=int, default=40)
    parser.add_argument("--chart-min-width", type=int, default=150)
    parser.add_argument("--chart-min-height", type=int, default=100)
    parser.add_argument("--max-chart-images-per-page", type=int, default=1)
    parser.add_argument("--max-chart-images-per-pdf", type=int, default=12)
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Run local Tesseract OCR on extracted chart images when the binary is installed.",
    )
    parser.add_argument("--ocr-command", default="tesseract")
    parser.add_argument("--ocr-timeout", type=int, default=20)
    parser.add_argument("--query", help="Run top-k retrieval after indexing, or with --query-only.")
    parser.add_argument("--query-only", action="store_true", help="Query the existing Chroma index without rebuilding it.")
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker", choices=["simple", "none"], default="simple")
    parser.add_argument("--filter-source-type", help="Optional metadata filter, e.g. annual_report_pdf or chart_summary.")
    parser.add_argument("--filter-ticker", help="Optional ticker metadata filter, e.g. AAPL.")
    parser.add_argument("--filter-year", type=int, help="Optional annual report year metadata filter.")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = chromadb.PersistentClient(path=str(args.index_dir))
    embedder = get_embedder(args.embedding_provider, args.hash_dimensions)

    if args.query_only:
        if not args.query:
            raise SystemExit("--query-only requires --query")
        collection = client.get_collection(args.collection)
        if args.retrieval_mode == "dense":
            hits = retrieve_top_k(
                collection=collection,
                embedder=embedder,
                query=args.query,
                top_k=args.top_k,
                source_type=args.filter_source_type,
                ticker=args.filter_ticker,
                year=args.filter_year,
            )
        else:
            chunk_rows = load_chunk_rows(args.chunks_out)
            hits = retrieve_hybrid_top_k(
                collection=collection,
                embedder=embedder,
                chunk_rows=chunk_rows,
                query=args.query,
                top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=args.rrf_k,
                reranker=args.reranker,
                source_type=args.filter_source_type,
                ticker=args.filter_ticker,
                year=args.filter_year,
            )
        print_retrieval_results(args.query, hits, retrieval_mode=args.retrieval_mode)
        return 0

    reports = read_manifest(args.manifest)
    if args.limit_pdfs:
        reports = reports[:args.limit_pdfs]
    if not reports:
        raise SystemExit(f"No PDF reports found in {args.manifest}")

    verification_rows = []
    chunks: list[ChunkRecord] = []
    if args.reset and args.include_chart_summaries and args.chart_images_dir.exists():
        shutil.rmtree(args.chart_images_dir)
    for report in reports:
        print(f"Verifying/extracting {report.ticker} {report.year}: {report.local_path}")
        pages, stats = verify_and_extract(report, max_pages_per_pdf=args.max_pages_per_pdf)
        verification_rows.append(stats)
        chunks.extend(build_chunks(report, pages, args.chunk_size, args.chunk_overlap))
        if args.include_tables:
            table_chunks = extract_table_summary_chunks(
                report=report,
                max_pages_per_pdf=args.max_pages_per_pdf,
                min_rows=args.table_min_rows,
                max_tables_per_page=args.max_tables_per_page,
                max_tables_per_pdf=args.max_tables_per_pdf,
            )
            chunks.extend(table_chunks)
            stats["table_summary_chunks"] = len(table_chunks)
        if args.include_chart_summaries:
            chart_chunks = extract_chart_summary_chunks(
                report=report,
                pages=pages,
                chart_images_dir=args.chart_images_dir,
                max_pages_per_pdf=args.max_pages_per_pdf,
                min_width=args.chart_min_width,
                min_height=args.chart_min_height,
                max_images_per_page=args.max_chart_images_per_page,
                max_images_per_pdf=args.max_chart_images_per_pdf,
                enable_ocr=args.enable_ocr,
                ocr_command=args.ocr_command,
                ocr_timeout=args.ocr_timeout,
            )
            chunks.extend(chart_chunks)
            stats["chart_summary_chunks"] = len(chart_chunks)

    write_jsonl(
        args.chunks_out,
        [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    )
    verify_path = args.chunks_out.with_name("pdf_verification.jsonl")
    write_jsonl(verify_path, verification_rows)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "domain": "financial_report_qa"},
    )

    indexed = 0
    started = time.time()
    for batch in batched(chunks, args.batch_size):
        embeddings = embedder.embed_documents([chunk.text for chunk in batch])
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(chunks)} chunks")

    summary = {
        "pdf_reports": len(reports),
        "verified_reports": len(verification_rows),
        "chunks": len(chunks),
        "chart_summary_chunks": sum(
            1 for chunk in chunks
            if chunk.metadata.get("source_type") == "chart_summary"
        ),
        "table_summary_chunks": sum(
            1 for chunk in chunks
            if chunk.metadata.get("source_type") == "table_summary"
        ),
        "ocr_enabled": args.enable_ocr,
        "ocr_status_counts": {
            status: sum(
                1 for chunk in chunks
                if chunk.metadata.get("source_type") == "chart_summary"
                and chunk.metadata.get("ocr_status") == status
            )
            for status in sorted(
                {
                    str(chunk.metadata.get("ocr_status"))
                    for chunk in chunks
                    if chunk.metadata.get("source_type") == "chart_summary"
                    and chunk.metadata.get("ocr_status") is not None
                }
            )
        },
        "chart_images_dir": str(args.chart_images_dir),
        "collection": args.collection,
        "index_dir": str(args.index_dir),
        "chunks_out": str(args.chunks_out),
        "verification_out": str(verify_path),
        "embedding_provider": args.embedding_provider,
        "elapsed_seconds": round(time.time() - started, 2),
        "collection_count": collection.count(),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    summary_path = args.index_dir / "index_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.query:
        if args.retrieval_mode == "dense":
            hits = retrieve_top_k(
                collection=collection,
                embedder=embedder,
                query=args.query,
                top_k=args.top_k,
                source_type=args.filter_source_type,
                ticker=args.filter_ticker,
                year=args.filter_year,
            )
        else:
            chunk_rows = load_chunk_rows(args.chunks_out)
            hits = retrieve_hybrid_top_k(
                collection=collection,
                embedder=embedder,
                chunk_rows=chunk_rows,
                query=args.query,
                top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=args.rrf_k,
                reranker=args.reranker,
                source_type=args.filter_source_type,
                ticker=args.filter_ticker,
                year=args.filter_year,
            )
        print_retrieval_results(args.query, hits, retrieval_mode=args.retrieval_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
