"""Ingest EDGAR annual-report PDFs into Qdrant for the financial copilot (built for Colab).

Implements the PDF part of docs/financial_rag_final_design.md (sections 3-7):

* ``docs_text``: native PDF text and tables, cut into chunks, with a dense
  vector and a number-aware sparse vector ("bm25", IDF modifier) that keeps
  ``391,035`` / ``391035`` / ``$6.08`` / ``2.1%`` as whole terms. Pages whose
  text layer is garbled are OCR'd (RapidOCR) or left to the visual path; their
  broken text is never indexed.
* ``visual_pages``: ColPali full multivectors (1030 x 128, float16, on disk,
  no HNSW) used only for reranking, mean-pooled rows and columns (38 x 128
  each, special tokens kept, in RAM, HNSW) for the first stage, and a text
  ``summary`` vector. Numbers never pass through these vectors: they stay as
  exact strings in the payload.

Each filing goes through two stages, ``text`` (CPU) and ``visual`` (GPU). Both
are recorded in ``<root>/ingest.sqlite3``, so a rerun after a Colab disconnect
skips finished work. After each batch the collections can be snapshotted to
Drive (``--snapshot-dir``) and restored in the next session (``restore``).
The 10-K HTML / XBRL facts DB (store A of the design) is a separate step.

    python -m scripts.rag.financial_ingest ingest --qdrant-url http://127.0.0.1:6333 \
        --pdf-dir /content/drive/MyDrive/capstone_rag/pdfs --snapshot-dir /content/drive/MyDrive/capstone_rag/snapshots
    python -m scripts.rag.financial_ingest status --qdrant-url http://127.0.0.1:6333
    python -m scripts.rag.financial_ingest query "Chevron net oil-equivalent production" --ticker CVX --visual
    python -m scripts.rag.financial_ingest restore --qdrant-url http://127.0.0.1:6333 --snapshot-dir ...

Without ``--qdrant-url`` (or ``QDRANT_URL``) it uses embedded Qdrant under
``--root``: fine for smoke runs, too slow for the corpus. The Colab session is
in notebooks/colab_financial_ingest.ipynb.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pymupdf
from qdrant_client import QdrantClient, models

from scripts.rag.multimodal_rag import dense, term_id

from scripts.common.paths import ROOT  # noqa: E402 - repo root
DOMAIN = "financial_report_qa"
MANIFEST = ROOT / "data/raw/financial_report_qa/edgar/pdf_manifest.jsonl"
DEFAULT_ROOT = Path(os.getenv("RAG_INGEST_DIR", ROOT / "indexes/financial_qdrant"))
TEXT, VISUAL = "docs_text", "visual_pages"
GRID = 32                          # ColPali v1.3: 32 x 32 image patches + special tokens
INDEXING_THRESHOLD = 10_000        # KB; Qdrant's default, restored after bulk upload
ANALYSIS_VERSION = 1
CHUNK_CHARS, OVERLAP_CHARS, TABLE_CHARS = 1200, 200, 3000
RENDER_SCALE = 1.5                 # same scale as the pooling evaluation
STOPWORDS = frozenset("the of and to in a is for on that by with as are be this from or at an our we".split())
SCALE = re.compile(r"\b(?:in|amounts in|dollars in)\s+(thousands|millions|billions)\b[^)\n]{0,60}", re.I)
ITEM = re.compile(r"^\s*item\s+(1a|1b|1c|7a|9a|9b|9c|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16)\b", re.I | re.M)
TERM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[a-z]+\d*")
KEYWORD_FIELDS = ("domain", "doc_id", "filing_id", "section", "content_type", "page_type", "doc_kind")
INTEGER_FIELDS = ("fiscal_year", "year", "page")


# ----------------------------------------------------------------------------- terms and chunks

def terms(text):
    """Lower-case words plus numbers kept whole, each number also without $, % and commas."""
    out = []
    for token in TERM.findall(text.lower()):
        if token[0].isalpha():
            out.append(token)
            continue
        token = token.rstrip(",")
        bare = token.strip("$%")
        out.extend({token, bare, bare.replace(",", "")})
    return out


def sparse_vector(text, query=False):
    counts = Counter(term_id(t) for t in terms(text))
    length = sum(counts.values()) or 1
    items = sorted(counts.items())
    # BM25 term saturation with a fixed reference length; Qdrant applies corpus IDF.
    return models.SparseVector(indices=[k for k, _ in items], values=[
        1.0 if query else c * 2.2 / (c + 1.2 * (.25 + .75 * length / 250)) for _, c in items])


def clean(text):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def windows(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    text, start = clean(text), 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            end = cut if cut > start + size // 2 else end
        yield text[start:end].strip()
        if end >= len(text):
            break
        space = text.find(" ", end - overlap, end)
        start = space + 1 if space != -1 else end


def split_table(lines, budget=TABLE_CHARS):
    header, part, size = lines[:2], [], 0
    for line in lines[2:]:
        if part and size + len(line) > budget:
            yield header + part
            part, size = [], 0
        part.append(line)
        size += len(line) + 1
    yield header + part


def chunk_name(doc_id, page, part):
    """Citable chunk id in the agents' convention (extract_citations matches annual_report_*)."""
    return f"annual_report_{doc_id}_p{page:04d}_{part}"


def point_id(*parts):
    return str(uuid5(NAMESPACE_URL, ":".join(map(str, ("fin",) + parts))))


# ----------------------------------------------------------------------------- page analysis (CPU)

def table_markdown(table):
    rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in row] for row in table.extract()]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    return [" | ".join(rows[0]), " | ".join(["---"] * width)] + [" | ".join(r) for r in rows[1:]]


def analyse_page(page):
    text = page.get_text(sort=True).strip()
    words = re.findall(r"[a-z]+", text.lower())
    garbled = (len(words) >= 40 and sum(w in STOPWORDS for w in words) / len(words) < .05) or text.count("�") > 5
    try:
        tables = [t for t in map(table_markdown, page.find_tables().tables) if t]
    except Exception:
        tables = []
    area = page.rect.get_area() or 1
    shapes = sum(1 for d in page.get_drawings() if d["rect"].width * d["rect"].height > 4)
    image_share = max(((pymupdf.Rect(i["bbox"]) & page.rect).get_area() / area for i in page.get_image_info()), default=0)
    kind = ("table" if tables else "chart" if shapes > 60 else "image" if image_share > .08
            else "outlined_text" if not words and shapes > 4 else "blank" if not words else "text")
    scale = SCALE.search(text)
    return {"text": text, "garbled": garbled, "tables": tables, "page_type": kind,
            "visual": kind in {"table", "chart", "image", "outlined_text"} or (garbled and kind != "blank"),
            "scale": scale.group(0).strip(" ()") if scale else None,
            "headings": [h.lower() for h in ITEM.findall(text)],
            "shapes": shapes, "image_share": round(image_share, 3)}


def analyse_pdf(path, max_pages=None):
    pages, section = [], "front"
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            if max_pages and number > max_pages:
                break
            info = analyse_page(page)
            heads = info.pop("headings")
            if len(set(heads)) >= 4:           # a table of contents lists every item
                info["section"] = "contents"
            else:
                info["section"] = f"item_{heads[0]}" if heads else section
                section = f"item_{heads[-1]}" if heads else section
            info["page"] = number
            pages.append(info)
    return pages


def readable_text(page):
    """Text that may be indexed: native if the layer is clean, otherwise OCR (may be empty)."""
    if page["garbled"] or page["page_type"] == "outlined_text":
        return page.get("ocr_text") or "", "ocr"
    return page["text"], "native_pdf"


# ----------------------------------------------------------------------------- models

class Embedder:
    """Dense text vectors: hashing (offline), local sentence-transformers, or a hosted OpenAI-compatible endpoint."""

    def __init__(self, provider="local", model="sentence-transformers/all-MiniLM-L6-v2", batch=64):
        self.provider, self.batch = provider, batch
        if provider == "hashing":
            self.dim, self.signature = 384, "hashing-384-v1"
        elif provider == "local":
            import torch
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
            self.dim = self.model.get_sentence_embedding_dimension()
            self.signature = f"local:{model}:{self.dim}"
        elif provider == "hosted":
            missing = [k for k in ("EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL") if not os.getenv(k)]
            if missing:
                raise RuntimeError(f"Missing hosted embedding env var(s): {', '.join(missing)}")
            self.base, self.key, self.model = (os.environ[k] for k in ("EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"))
            self.dim = len(self._hosted(["dimension probe"])[0])
            self.signature = f"hosted:{self.model}:{self.dim}"
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def __call__(self, texts):
        if self.provider == "hashing":
            return [dense(t) for t in texts]
        out = []
        for start in range(0, len(texts), self.batch):
            part = texts[start:start + self.batch]
            out += (self.model.encode(part, batch_size=self.batch, normalize_embeddings=True).tolist()
                    if self.provider == "local" else self._hosted(part))
        return out

    def _hosted(self, texts):
        body = json.dumps({"model": self.model, "input": [t[:8000] for t in texts]}).encode()
        for attempt in range(4):
            request = urllib.request.Request(self.base.rstrip("/") + "/embeddings", data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)["data"]
                return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
            except urllib.error.HTTPError as exc:   # retry only explicit overload, as in resilience.py
                if exc.code not in (429, 503) or attempt == 3:
                    raise
                time.sleep(2 ** attempt * 2)


def colpali_signature(model_name):
    return json.dumps({"model": model_name, "query_prefix": "Query: ", "grid": GRID,
                       "pooling": "mean_rows+mean_cols+specials", "render_scale": RENDER_SCALE}, sort_keys=True)


class ColPali:
    """vidore/colpali-v1.3-hf through transformers; returns full multivectors plus the image-token mask."""

    def __init__(self, model_name="vidore/colpali-v1.3-hf", dtype="auto"):
        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if self.device == "cuda" else "float32"
        self.dtype_name, self.signature = dtype, colpali_signature(model_name)
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.processor.query_prefix = "Query: "     # matches rag/colpali_retrieval.py
        self.model = ColPaliForRetrieval.from_pretrained(model_name, dtype=getattr(torch, dtype), device_map=self.device,
                                                         attn_implementation="sdpa").eval()
        vlm = getattr(self.model.config, "vlm_config", self.model.config)
        self.image_token = getattr(vlm, "image_token_id", None) or getattr(vlm, "image_token_index")

    def embed_images(self, images):
        import torch
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False).embeddings.float()
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite ColPali output; on a T4 (no bfloat16) rerun with --colpali-dtype float32")
        result = []
        for row, ids, attention in zip(out, inputs["input_ids"], inputs["attention_mask"]):
            keep = attention.bool()
            result.append((torch.nn.functional.normalize(row[keep], dim=-1).cpu().numpy(),
                           (ids[keep] == self.image_token).cpu().numpy()))
        return result

    def embed_query(self, text):
        import torch
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False).embeddings[0].float()
        return torch.nn.functional.normalize(out[inputs["attention_mask"][0].bool()], dim=-1).cpu().numpy()


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pool_rows_cols(vectors, is_image):
    """Mean of each patch row and each patch column (32 each) plus the unpooled special tokens."""
    vectors = np.asarray(vectors, dtype=np.float32)
    patches, specials = vectors[is_image], vectors[~is_image]
    if len(patches) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} image patches, got {len(patches)}")
    grid = patches.reshape(GRID, GRID, -1)
    return l2(np.concatenate([grid.mean(axis=1), specials])), l2(np.concatenate([grid.mean(axis=0), specials]))


class PageOCR:
    def __init__(self, cuda=False, scale=2.0, min_confidence=.6):
        from rapidocr import RapidOCR
        self.engine = RapidOCR(params={"Global.log_level": "warning", "EngineConfig.onnxruntime.use_cuda": cuda})
        self.scale, self.min_confidence = scale, min_confidence
        self.signature = ocr_signature(scale, min_confidence)

    def read(self, page):
        from PIL import Image
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(self.scale, self.scale), alpha=False)
        output = self.engine(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        if output.txts is None:
            return ""
        return "\n".join(t for t, s in zip(output.txts, output.scores) if float(s) >= self.min_confidence)


def ocr_signature(scale=2.0, min_confidence=.6):
    return f"rapidocr-full-page:{scale}:{min_confidence}"


# ----------------------------------------------------------------------------- state

class State:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.db = path, sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS filings (doc_id TEXT, stage TEXT, status TEXT, pipeline TEXT, points INTEGER,
                                                seconds REAL, error TEXT, updated_at TEXT, PRIMARY KEY (doc_id, stage));
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);""")

    def check(self, key, value):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row and row[0] != value:
            raise ValueError(f"{key} differs from the existing index ({row[0]} != {value}). "
                             "Use a new --root with fresh collections, or --reset.")
        self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))
        self.db.commit()

    def done(self, doc_id, stage, pipeline):
        row = self.db.execute("SELECT status, pipeline FROM filings WHERE doc_id=? AND stage=?", (doc_id, stage)).fetchone()
        return row == ("complete", pipeline)

    def mark(self, doc_id, stage, status, pipeline, points=0, seconds=0.0, error=None):
        self.db.execute("INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?)", (
            doc_id, stage, status, pipeline, points, round(seconds, 1), error, datetime.now(timezone.utc).isoformat()))
        self.db.commit()

    def backup(self, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(dest) as target:
            self.db.backup(target)


# ----------------------------------------------------------------------------- Qdrant

def connect(url=None, root=DEFAULT_ROOT, grpc=True, timeout=300):
    url = url or os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=grpc, timeout=timeout)
    print("No --qdrant-url: using embedded Qdrant (smoke runs only).", flush=True)
    return QdrantClient(path=str(Path(root) / "qdrant_embedded"))


def is_server(client):
    from qdrant_client.local.qdrant_local import QdrantLocal
    return not isinstance(client._client, QdrantLocal)


def ensure_collections(client, dim, visual=True):
    multi = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    cosine = models.Distance.COSINE
    if not client.collection_exists(TEXT):
        client.create_collection(TEXT, on_disk_payload=True, vectors_config={
            "dense": models.VectorParams(size=dim, distance=cosine, on_disk=False,
                                         hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    if visual and not client.collection_exists(VISUAL):
        pooled = models.VectorParams(size=128, distance=cosine, on_disk=False,
                                     datatype=models.Datatype.FLOAT16, multivector_config=multi)
        client.create_collection(VISUAL, on_disk_payload=True, vectors_config={
            # Full 1030 x 128: rerank only, so no HNSW graph (m=0), memory-mapped from disk.
            "colpali": models.VectorParams(size=128, distance=cosine, on_disk=True, datatype=models.Datatype.FLOAT16,
                                           multivector_config=multi, hnsw_config=models.HnswConfigDiff(m=0)),
            "mean_rows": pooled, "mean_cols": pooled,
            "summary": models.VectorParams(size=dim, distance=cosine, on_disk=False)})
    for name, vector in ((TEXT, "dense"), (VISUAL, "summary")):
        if not client.collection_exists(name):
            continue
        info = client.get_collection(name)
        size = info.config.params.vectors[vector].size
        if size != dim:
            raise ValueError(f"{name}.{vector} has dimension {size}, the embedder gives {dim}. Use a new --root/--reset.")
        if name == VISUAL and "mean_rows" not in info.config.params.vectors:
            raise ValueError("visual_pages exists without pooled vectors (created by LocalRAG?). Use --reset.")
        for field in KEYWORD_FIELDS + ("ticker",) + INTEGER_FIELDS:
            if field in (info.payload_schema or {}):
                continue
            schema = (models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True) if field == "ticker"
                      else models.PayloadSchemaType.INTEGER if field in INTEGER_FIELDS else models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(name, field, field_schema=schema, wait=True)


def set_bulk(client, collections, on):
    """Pause HNSW building during upload (indexing_threshold=0) and restore it afterwards."""
    if not is_server(client):
        return
    for name in collections:
        if client.collection_exists(name):
            client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0 if on else INDEXING_THRESHOLD))


def wait_green(client, collections, timeout=1800):
    deadline = time.time() + timeout
    for name in collections:
        while client.collection_exists(name) and client.get_collection(name).status != models.CollectionStatus.GREEN:
            if time.time() > deadline:
                print(f"{name}: still optimising after {timeout}s; snapshotting anyway", flush=True)
                return
            time.sleep(3)


def doc_filter(doc_id):
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def query_filter(ticker=None, year=None):
    conditions = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
                  for k, v in (("ticker", ticker and ticker.upper()), ("fiscal_year", year)) if v]
    return models.Filter(must=conditions) if conditions else None


def rest_url(url):
    return (url or os.getenv("QDRANT_URL") or "").rstrip("/").replace(":6334", ":6333")


def snapshot(client, url, collection, dest, keep=1):
    """Create a server snapshot, stream it to dest (e.g. Drive), verify its checksum, prune older copies."""
    description = client.create_snapshot(collection, wait=True)
    dest.mkdir(parents=True, exist_ok=True)
    part, digest = dest / (description.name + ".part"), hashlib.sha256()
    headers = {"api-key": os.environ["QDRANT_API_KEY"]} if os.getenv("QDRANT_API_KEY") else {}
    request = urllib.request.Request(f"{rest_url(url)}/collections/{collection}/snapshots/{description.name}", headers=headers)
    with urllib.request.urlopen(request, timeout=3600) as response, part.open("wb") as handle:
        while block := response.read(8 << 20):
            digest.update(block)
            handle.write(block)
    if description.checksum and digest.hexdigest() != description.checksum:
        part.unlink()
        raise RuntimeError(f"Snapshot checksum mismatch for {collection}")
    final = dest / description.name
    part.replace(final)
    (dest / (description.name + ".json")).write_text(json.dumps({
        "collection": collection, "name": description.name, "sha256": digest.hexdigest(),
        "bytes": final.stat().st_size, "created_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    client.delete_snapshot(collection, description.name, wait=True)     # free local disk
    for old in sorted(dest.glob(f"{collection}-*.snapshot"), key=lambda p: p.stat().st_mtime)[:-keep]:
        old.unlink()
        old.with_name(old.name + ".json").unlink(missing_ok=True)
    return final


def restore(args):
    client = connect(args.qdrant_url, args.root, not args.no_grpc, args.timeout)
    if not is_server(client):
        raise SystemExit("restore needs a Qdrant server (--qdrant-url)")
    source, root = Path(args.snapshot_dir), Path(args.root)
    for collection in (TEXT, VISUAL):
        files = sorted(source.glob(f"{collection}-*.snapshot"), key=lambda p: p.stat().st_mtime)
        if not files:
            print(f"{collection}: no snapshot in {source}")
            continue
        meta = files[-1].with_name(files[-1].name + ".json")
        checksum = json.loads(meta.read_text(encoding="utf-8"))["sha256"] if meta.exists() else None
        # The server may only read snapshots from its own snapshots directory, so copy there first.
        local = Path(args.server_snapshots_path or root / "snapshots") / collection / files[-1].name
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[-1], local)
        client.recover_snapshot(collection, location=local.resolve().as_uri(), checksum=checksum,
                                priority=models.SnapshotPriority.SNAPSHOT, wait=True)
        local.unlink()
        print(f"{collection}: restored {files[-1].name}, {client.count(collection, exact=True).count} points")
    for name in ("ingest.sqlite3", "pages"):
        src, dst = source / name, root / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    print(f"state restored to {root}")


# ----------------------------------------------------------------------------- ingest

def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def resolve_pdf(row, pdf_dir, download=False, user_agent=None):
    path = Path(pdf_dir) / Path(row["local_path"].replace("\\", "/")).name
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing (copy it from Drive or pass --download-missing)")
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires --user-agent with a contact e-mail")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(row["source_url"], headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=300) as response:
            path.with_suffix(".part").write_bytes(response.read())
        path.with_suffix(".part").replace(path)
        time.sleep(0.15)                                    # stay under SEC's 10 requests/s
    if row.get("content_sha256") and sha256_file(path) != row["content_sha256"]:
        raise ValueError(f"{path.name}: SHA-256 differs from the manifest")
    return path


def filing_payload(row):
    year = int(row["year"])
    return {"domain": DOMAIN, "doc_id": row["document_id"], "filing_id": f"{row['ticker']}_FY{year}",
            "doc_kind": "ARS", "ticker": row["ticker"].upper(), "company": row.get("company"), "cik": row.get("cik"),
            "year": year, "fiscal_year": year, "accession_number": row.get("accession_number"),
            "filing_date": row.get("filing_date"), "source_url": row.get("source_url"),
            "source_type": row.get("source_type", "annual_report_pdf"), "content_sha256": row.get("content_sha256"),
            "pdf_name": Path(row["local_path"].replace("\\", "/")).name,
            "title": row.get("title"), "doc_title": row.get("title")}


def context_line(payload):
    return (f"{payload.get('company') or payload['ticker']} ({payload['ticker']}) fiscal {payload['fiscal_year']} "
            f"annual report, {payload['section'].replace('_', ' ')}, page {payload['page']}")


def text_records(row, pages):
    """(id, payload, text to embed) for every prose window and table part of the filing."""
    base, records = filing_payload(row), []
    for page in pages:
        body, source = readable_text(page)
        if not body.strip():
            continue
        common = {**base, "page": page["page"], "section": page["section"], "page_type": page["page_type"],
                  "visual_rich": page["visual"], "garbled": page["garbled"], "scale": page["scale"], "text_source": source}
        parts = [("text", i, chunk, None) for i, chunk in enumerate(windows(body))]
        if source == "native_pdf":
            for t, lines in enumerate(page["tables"]):
                for j, part in enumerate(split_table(lines)):
                    units = f" ({page['scale']})" if page["scale"] else ""
                    parts.append((f"table{t}", j, f"Table, page {page['page']}{units}\n" + "\n".join(part), part))
        for index, (kind, i, chunk, table) in enumerate(parts):
            identifier = point_id(row["document_id"], page["page"], kind, i)
            payload = {**common, "chunk_id": chunk_name(row["document_id"], page["page"], f"{kind}{i}"),
                       "point_id": identifier, "chunk_index": index,
                       "content_type": "table" if table else "text", "text": chunk,
                       "table_markdown": ["\n".join(table)] if table else []}
            records.append((identifier, payload, context_line(payload) + "\n" + chunk))
    return records


def load_pages(root, row, pdf, max_pages, analysed=None):
    cache = Path(root) / "pages" / f"{row['document_id']}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if (data.get("version"), data.get("max_pages"), data.get("sha256")) == (ANALYSIS_VERSION, max_pages, row.get("content_sha256")):
            return data["pages"]
    pages = analysed if analysed is not None else analyse_pdf(pdf, max_pages)
    save_pages(root, row, max_pages, pages)
    return pages


def save_pages(root, row, max_pages, pages):
    cache = Path(root) / "pages" / f"{row['document_id']}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": ANALYSIS_VERSION, "max_pages": max_pages, "sha256": row.get("content_sha256"),
                               "pages": pages}), encoding="utf-8")
    tmp.replace(cache)


def apply_ocr(ocr, pdf, pages, limit):
    """OCR unreadable pages (garbled or outlined text), front section first, up to `limit` per filing (0 = all)."""
    unreadable = [p for p in pages if p["garbled"] or p["page_type"] == "outlined_text"]
    done = sum(p.get("ocr_signature") == ocr.signature for p in unreadable)
    todo = sorted((p for p in unreadable if p.get("ocr_signature") != ocr.signature),
                  key=lambda p: (p["section"] != "front", p["page"]))
    if limit:
        todo = todo[:max(0, limit - done)]
    with pymupdf.open(pdf) as document:
        for page in todo:
            page["ocr_text"] = ocr.read(document[page["page"] - 1])
            page["ocr_signature"] = ocr.signature
    return len(todo)


class Ingest:
    def __init__(self, args, *, embedder=None, colpali=None, client=None, ocr=None):
        self.args, self.root = args, Path(args.root)
        self.client = client or connect(args.qdrant_url, args.root, not args.no_grpc, args.timeout)
        if args.reset:
            reset(args.root, self.client)
        self.state = State(self.root / "ingest.sqlite3")
        self.embedder = embedder or Embedder(args.embedding_provider, args.embedding_model)
        self.state.check("embedding", self.embedder.signature)
        self.stages = [s for s in ("text", "visual") if s in args.stages.split(",")]
        self._colpali, self._ocr = colpali, ocr
        self.colpali_sig = colpali.signature if colpali else colpali_signature(args.colpali_model)
        if "visual" in self.stages:
            self.state.check("colpali", self.colpali_sig)
        self.ocr_sig = None if args.ocr == "none" else (ocr.signature if ocr else ocr_signature())
        ensure_collections(self.client, self.embedder.dim, visual="visual" in self.stages)
        common = {"analysis": ANALYSIS_VERSION, "max_pages": args.max_pages_per_pdf, "embedding": self.embedder.signature}
        self.pipeline = {
            "text": json.dumps({**common, "chunk": [CHUNK_CHARS, OVERLAP_CHARS, TABLE_CHARS], "terms": "num-v1", "ids": "v2",
                                "ocr": self.ocr_sig, "ocr_limit": args.ocr_max_pages_per_filing}, sort_keys=True),
            "visual": json.dumps({**common, "colpali": self.colpali_sig, "pages": args.visual_pages, "ids": "v2",
                                  "ocr": self.ocr_sig, "ocr_limit": args.ocr_max_pages_per_filing}, sort_keys=True)}

    @property
    def colpali(self):
        if self._colpali is None:
            self._colpali = ColPali(self.args.colpali_model, self.args.colpali_dtype)
        return self._colpali

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = PageOCR(cuda=self.args.ocr_cuda)
        return self._ocr

    def pending(self, row):
        return [s for s in self.stages if self.args.force or not self.state.done(row["document_id"], s, self.pipeline[s])]

    def run(self):
        rows = read_manifest(self.args.manifest)
        if self.args.tickers:
            wanted = {t.strip().upper() for t in self.args.tickers.split(",")}
            rows = [r for r in rows if r["ticker"].upper() in wanted]
        rows = rows[:self.args.limit_filings] if self.args.limit_filings else rows
        todo = [r for r in rows if self.pending(r)]
        print(f"{len(rows)} filings selected, {len(todo)} with pending stages {self.stages}", flush=True)
        summary = Counter()
        for start in range(0, len(todo), self.args.batch_size):
            batch = todo[start:start + self.args.batch_size]
            free = shutil.disk_usage(self.root).free / 2**30
            if free < 10:
                print(f"warning: only {free:.1f} GB free under {self.root}", flush=True)
            collections = [TEXT] + ([VISUAL] if "visual" in self.stages else [])
            set_bulk(self.client, collections, True)
            try:
                for row, pdf, pages in self.prepare(batch):
                    for stage in self.pending(row):
                        summary[self.run_stage(stage, row, pdf, pages)] += 1
            finally:
                set_bulk(self.client, collections, False)
            if self.args.snapshot_dir and is_server(self.client):
                wait_green(self.client, collections)
                for name in collections:
                    print(f"snapshot -> {snapshot(self.client, self.args.qdrant_url, name, Path(self.args.snapshot_dir), self.args.keep_snapshots)}", flush=True)
                self.state.backup(Path(self.args.snapshot_dir) / "ingest.sqlite3")
                shutil.copytree(self.root / "pages", Path(self.args.snapshot_dir) / "pages", dirs_exist_ok=True)
            print(f"batch {start // self.args.batch_size + 1}: {dict(summary)}", flush=True)
        return dict(summary)

    def prepare(self, batch):
        """Resolve and verify PDFs, then analyse uncached filings in parallel worker processes."""
        ready = []
        for row in batch:
            try:
                ready.append((row, resolve_pdf(row, self.args.pdf_dir, self.args.download_missing, self.args.user_agent)))
            except Exception as exc:
                for stage in self.pending(row):
                    self.state.mark(row["document_id"], stage, "failed", self.pipeline[stage], error=str(exc))
                print(json.dumps({"doc_id": row["document_id"], "status": "failed", "error": str(exc)}), flush=True)
        cached = {row["document_id"] for row, _ in ready
                  if (self.root / "pages" / f"{row['document_id']}.json").exists()}
        fresh = [(row, pdf) for row, pdf in ready if row["document_id"] not in cached]
        analysed = {}
        if fresh and self.args.workers > 1:
            with ProcessPoolExecutor(self.args.workers) as pool:
                futures = {row["document_id"]: pool.submit(analyse_pdf, str(pdf), self.args.max_pages_per_pdf) for row, pdf in fresh}
                for doc_id, future in futures.items():
                    try:
                        analysed[doc_id] = future.result()
                    except Exception as exc:
                        print(json.dumps({"doc_id": doc_id, "analysis_error": str(exc)}), flush=True)
        for row, pdf in ready:
            try:
                yield row, pdf, load_pages(self.root, row, pdf, self.args.max_pages_per_pdf, analysed.get(row["document_id"]))
            except Exception as exc:
                for stage in self.pending(row):
                    self.state.mark(row["document_id"], stage, "failed", self.pipeline[stage], error=str(exc))
                print(json.dumps({"doc_id": row["document_id"], "status": "failed", "error": str(exc)}), flush=True)

    def run_stage(self, stage, row, pdf, pages):
        doc_id, started = row["document_id"], time.time()
        self.state.mark(doc_id, stage, "processing", self.pipeline[stage])
        try:
            if self.ocr_sig and any(p["garbled"] or p["page_type"] == "outlined_text" for p in pages):
                ocred = apply_ocr(self.ocr, pdf, pages, self.args.ocr_max_pages_per_filing)
                if ocred:
                    save_pages(self.root, row, self.args.max_pages_per_pdf, pages)
            points = self.write_text(row, pages) if stage == "text" else self.write_visual(row, pdf, pages)
        except Exception as exc:
            self.state.mark(doc_id, stage, "failed", self.pipeline[stage], seconds=time.time() - started, error=repr(exc))
            print(json.dumps({"doc_id": doc_id, "stage": stage, "status": "failed", "error": repr(exc)}), flush=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return "failed"
        seconds = time.time() - started
        self.state.mark(doc_id, stage, "complete", self.pipeline[stage], points, seconds)
        print(json.dumps({"doc_id": doc_id, "stage": stage, "status": "complete", "pages": len(pages),
                          "points": points, "seconds": round(seconds, 1)}), flush=True)
        return "complete"

    def write_text(self, row, pages):
        records = text_records(row, pages)
        self.client.delete(TEXT, models.FilterSelector(filter=doc_filter(row["document_id"])), wait=True)
        for start in range(0, len(records), 128):
            part = records[start:start + 128]
            vectors = self.embedder([text for _, _, text in part])
            self.client.upsert(TEXT, wait=True, points=[models.PointStruct(id=i, payload=p, vector={"dense": v, "bm25": sparse_vector(t)})
                                       for (i, p, t), v in zip(part, vectors)])
        return len(records)

    def write_visual(self, row, pdf, pages):
        from PIL import Image
        targets = [p for p in pages if p["page_type"] != "blank"] if self.args.visual_pages == "all" else [p for p in pages if p["visual"]]
        self.client.delete(VISUAL, models.FilterSelector(filter=doc_filter(row["document_id"])), wait=True)
        base = filing_payload(row)
        with pymupdf.open(pdf) as document:
            for start in range(0, len(targets), self.args.colpali_batch):
                part = targets[start:start + self.args.colpali_batch]
                images = []
                for page in part:
                    pixmap = document[page["page"] - 1].get_pixmap(matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)
                    images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
                encoded = self.colpali.embed_images(images)
                payloads = []
                for page in part:
                    text, source = readable_text(page)
                    identifier = point_id(row["document_id"], page["page"], "visual")
                    payloads.append({**base, "page": page["page"], "section": page["section"], "page_type": page["page_type"],
                                     "content_type": page["page_type"], "garbled": page["garbled"], "scale": page["scale"],
                                     "text_source": source, "text": clean(text)[:4000],
                                     "table_markdown": ["\n".join(t) for t in page["tables"]] if source == "native_pdf" else [],
                                     "chunk_id": chunk_name(row["document_id"], page["page"], "visual"),
                                     "visual_point_id": identifier, "visual_rich": True,
                                     "colpali_model": self.colpali_sig, "colpali_dtype": getattr(self.colpali, "dtype_name", None)})
                summaries = self.embedder([context_line(p) + "\n" + p["text"][:2000] for p in payloads])
                points = []
                for payload, summary, (full, is_image) in zip(payloads, summaries, encoded):
                    rows_, cols_ = pool_rows_cols(full, is_image)
                    payload["colpali_patches"] = len(full)
                    points.append(models.PointStruct(id=payload["visual_point_id"], payload=payload, vector={
                        "colpali": np.asarray(full, dtype=np.float16).tolist(),
                        "mean_rows": rows_.astype(np.float16).tolist(), "mean_cols": cols_.astype(np.float16).tolist(),
                        "summary": summary}))
                self.client.upsert(VISUAL, points=points, wait=True)
        return len(targets)


def reset(root, client):
    for name in (TEXT, VISUAL):
        if client.collection_exists(name):
            client.delete_collection(name)
    Path(root, "ingest.sqlite3").unlink(missing_ok=True)
    print("collections and ingest state reset (page analysis cache kept)", flush=True)


# ----------------------------------------------------------------------------- search (verification and serving)

def search_text(client, embedder, query, top_k=10, flt=None, prefetch=50):
    """Dense + number-aware sparse prefetch fused with Qdrant's native RRF."""
    return client.query_points(TEXT, prefetch=[
        models.Prefetch(query=embedder([query])[0], using="dense", filter=flt, limit=prefetch),
        models.Prefetch(query=sparse_vector(query, query=True), using="bm25", filter=flt, limit=prefetch)],
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=top_k, with_payload=True).points


def search_visual(client, colpali, embedder, query, top_k=20, flt=None, prefetch=200):
    """Pooled rows/cols + summary prefetch, then exact MaxSim over the full on-disk vectors."""
    multivector = np.asarray(colpali.embed_query(query), dtype=np.float32).tolist()
    return client.query_points(VISUAL, prefetch=[
        models.Prefetch(query=multivector, using="mean_rows", filter=flt, limit=prefetch),
        models.Prefetch(query=multivector, using="mean_cols", filter=flt, limit=prefetch),
        models.Prefetch(query=embedder([query])[0], using="summary", filter=flt, limit=prefetch)],
        query=multivector, using="colpali", query_filter=flt, limit=top_k, with_payload=True).points


_serving: dict = {}
_serving_lock = threading.RLock()


def overlap(query, text):
    """Share of the query's (number-aware) terms that appear in the text."""
    wanted = set(terms(query)) - {"the", "of", "and", "what", "was", "for", "in", "a", "to", "is"}
    return len(wanted & set(terms(text))) / len(wanted) if wanted else 0.0


def retrieve_for_agent(query, top_k=5, ticker=None, year=None, candidates=30):
    """Evidence for the financial agents from docs_text (+ visual_pages when RAG_FIN_VISUAL=1).

    Hybrid dense + sparse with Qdrant RRF, then a rerank by RRF + term overlap (numbers kept whole),
    or by the cross-encoder when RAG_RERANKER_MODEL is set.
    Hits use the agent's evidence format, so citation checks work unchanged.
    """
    with _serving_lock:
        if "client" not in _serving:
            root = Path(os.getenv("RAG_INGEST_DIR", DEFAULT_ROOT))
            args = argparse.Namespace(root=root, embedding_provider="hashing", embedding_model=None)
            _serving.update(client=connect(os.getenv("QDRANT_URL"), root), embedder=recorded_embedder(args))
        client, embedder = _serving["client"], _serving["embedder"]
        flt = query_filter(ticker, year)
        fused = {}
        for rank, hit in enumerate(search_text(client, embedder, query, max(candidates, top_k), flt), 1):
            fused[hit.payload["chunk_id"]] = {"payload": hit.payload, "rrf": 1 / (60 + rank), "paths": ["text"]}
        if os.getenv("RAG_FIN_VISUAL") == "1" and client.collection_exists(VISUAL):
            if "colpali" not in _serving:
                _serving["colpali"] = ColPali(os.getenv("RAG_FIN_COLPALI_MODEL", "vidore/colpali-v1.3-hf"))
            for rank, hit in enumerate(search_visual(client, _serving["colpali"], embedder, query, 20, flt), 1):
                entry = fused.setdefault(hit.payload["chunk_id"], {"payload": hit.payload, "rrf": 0.0, "paths": []})
                entry["rrf"] += 1 / (60 + rank)
                entry["paths"].append("visual")
    hits = []
    for entry in fused.values():
        p = entry["payload"]
        hits.append({"chunk_id": p["chunk_id"], "text": p.get("text", ""), "title": p.get("title"),
                     "source_type": p.get("source_type"), "ticker": p.get("ticker"), "year": p.get("fiscal_year"),
                     "page": p.get("page"), "source_url": p.get("source_url"), "local_path": p.get("pdf_name"),
                     "image_path": None, "section": p.get("section"), "scale": p.get("scale"),
                     "preview": re.sub(r"\s+", " ", p.get("text", ""))[:500],
                     "rrf_score": entry["rrf"], "rerank_score": entry["rrf"] + 0.02 * overlap(query, p.get("text", "")),
                     "retrieval_paths": entry["paths"], "metadata": dict(p)})
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return cross_encoder_rerank(query, hits, top_k) if os.getenv("RAG_RERANKER_MODEL") else hits[:top_k]


def cross_encoder_rerank(query, hits, top_k):
    """Order text hits by cross-encoder score; on failure keep the overlap order and say so.

    ColPali pages are found from pixels, so their text can't judge them: the best one keeps a slot, unscored.
    """
    from scripts.common.local_models import reranker
    text_hits = [h for h in hits if "text" in h["retrieval_paths"]]
    visual_hits = [h for h in hits if "text" not in h["retrieval_paths"]][:1]
    try:
        scores = reranker(os.environ["RAG_RERANKER_MODEL"]).predict([(query, h["text"]) for h in text_hits]) if text_hits else []
        if len(scores) != len(text_hits) or not all(np.isfinite(float(s)) for s in scores):
            raise ValueError("Invalid reranker scores")
    except Exception:
        for hit in hits:
            hit["retrieval_warnings"] = ["reranker_unavailable"]
        return hits[:top_k]
    for hit, score in zip(text_hits, scores):
        hit["rerank_score"] = float(score)
        hit["reranked"] = True
    text_hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return text_hits[:max(top_k - len(visual_hits), 0)] + visual_hits[:top_k]


def status(args):
    client = connect(args.qdrant_url, args.root, not args.no_grpc, args.timeout)
    state = Path(args.root) / "ingest.sqlite3"
    report = {"collections": {n: client.count(n, exact=True).count for n in (TEXT, VISUAL) if client.collection_exists(n)}}
    if state.exists():
        with sqlite3.connect(state) as db:
            report["stages"] = [dict(zip(("stage", "status", "filings", "points", "seconds"), r)) for r in db.execute(
                "SELECT stage, status, COUNT(*), SUM(points), SUM(seconds) FROM filings GROUP BY stage, status ORDER BY stage, status")]
            report["failed"] = [dict(zip(("doc_id", "stage", "error"), r)) for r in db.execute(
                "SELECT doc_id, stage, error FROM filings WHERE status='failed'")]
            report["settings"] = dict(db.execute("SELECT key, value FROM settings"))
    print(json.dumps(report, indent=2))
    return report


def recorded_embedder(args):
    """The embedder the index was built with (from ingest.sqlite3), else the command-line choice."""
    state = Path(args.root) / "ingest.sqlite3"
    if state.exists():
        with sqlite3.connect(state) as db:
            row = db.execute("SELECT value FROM settings WHERE key='embedding'").fetchone()
        if row:
            provider, _, rest = row[0].partition(":")
            if provider.startswith("hashing"):
                return Embedder("hashing")
            if provider == "local":
                return Embedder("local", rest.rsplit(":", 1)[0])
            return Embedder("hosted")
    return Embedder(args.embedding_provider, args.embedding_model)


def query(args):
    client = connect(args.qdrant_url, args.root, not args.no_grpc, args.timeout)
    embedder = recorded_embedder(args)
    flt = query_filter(args.ticker, args.year)
    started = time.time()
    hits = [("text", h) for h in search_text(client, embedder, args.text, args.top_k, flt)]
    if args.visual:
        hits += [("visual", h) for h in search_visual(client, ColPali(args.colpali_model, args.colpali_dtype),
                                                      embedder, args.text, args.top_k, flt)]
    for path, hit in hits:
        p = hit.payload
        print(json.dumps({"path": path, "score": round(hit.score, 4), "doc_id": p["doc_id"], "page": p["page"],
                          "section": p.get("section"), "type": p.get("content_type"), "scale": p.get("scale"),
                          "preview": re.sub(r"\s+", " ", p.get("text", ""))[:160]}))
    print(f"{len(hits)} hits in {time.time() - started:.2f}s (including query encoding)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="local working dir (state, page cache, embedded Qdrant)")
    common.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    common.add_argument("--no-grpc", action="store_true", help="use REST instead of gRPC for uploads")
    common.add_argument("--timeout", type=int, default=300)
    common.add_argument("--embedding-provider", choices=["local", "hosted", "hashing"], default="local")
    common.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    common.add_argument("--colpali-model", default="vidore/colpali-v1.3-hf")
    common.add_argument("--colpali-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common], help="write PDFs into docs_text and visual_pages")
    ingest.add_argument("--manifest", type=Path, default=MANIFEST)
    ingest.add_argument("--pdf-dir", type=Path, default=MANIFEST.parent / "pdfs")
    ingest.add_argument("--download-missing", action="store_true", help="fetch missing PDFs from EDGAR (SHA-256 checked)")
    ingest.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"))
    ingest.add_argument("--tickers", help="comma-separated subset")
    ingest.add_argument("--limit-filings", type=int)
    ingest.add_argument("--max-pages-per-pdf", type=int, help="smoke runs only")
    ingest.add_argument("--stages", default="text,visual", help="text (CPU), visual (GPU) or both")
    ingest.add_argument("--visual-pages", choices=["rule", "all"], default="rule")
    ingest.add_argument("--colpali-batch", type=int, default=4)
    ingest.add_argument("--ocr", choices=["garbled", "none"], default="garbled")
    ingest.add_argument("--ocr-cuda", action="store_true", help="RapidOCR on GPU (needs onnxruntime-gpu)")
    ingest.add_argument("--ocr-max-pages-per-filing", type=int, default=20, help="0 = every unreadable page")
    ingest.add_argument("--batch-size", type=int, default=10, help="filings between snapshots")
    ingest.add_argument("--workers", type=int, default=2, help="processes for page analysis")
    ingest.add_argument("--snapshot-dir", type=Path, help="e.g. a Drive folder; snapshot after every batch")
    ingest.add_argument("--keep-snapshots", type=int, default=1)
    ingest.add_argument("--force", action="store_true", help="redo finished filings")
    ingest.add_argument("--reset", action="store_true", help="drop both collections and the ingest state first")

    sub.add_parser("status", parents=[common])
    search = sub.add_parser("query", parents=[common])
    search.add_argument("text")
    search.add_argument("--ticker")
    search.add_argument("--year", type=int)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--visual", action="store_true", help="also run the ColPali path (loads the model)")

    back = sub.add_parser("restore", parents=[common], help="load the latest snapshots and state from --snapshot-dir")
    back.add_argument("--snapshot-dir", type=Path, required=True)
    back.add_argument("--server-snapshots-path", type=Path, help="the server's QDRANT__STORAGE__SNAPSHOTS_PATH (default <root>/snapshots)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        Ingest(args).run()
    elif args.command == "status":
        status(args)
    elif args.command == "query":
        query(args)
    else:
        restore(args)


if __name__ == "__main__":
    main()
