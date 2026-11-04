"""Measure ColPali mean/max pooling (1030 -> 38 vectors per page) on a Colab GPU.

Standalone: it does not import the repo, so it can be copied to Colab alone.
It encodes real annual-report pages once, derives every pooled variant from
the same full multivectors, and reports what each variant costs and loses:

* retrieval quality per content type (chart, table, image, text) against
  labelled queries, for pooled-only search and for pooled prefetch + full
  MaxSim rerank;
* agreement with full MaxSim (does the prefetch keep the pages full MaxSim
  would rank first?), compared with the current text-summary prefetch;
* storage per page and Qdrant upsert/query latency for the recommended
  collection layout.

Colab (T4 or better):

    !pip -q install "transformers>=4.57,<5" "qdrant-client>=1.12,<2" pymupdf sentence-transformers scikit-learn
    !mkdir -p /content/qdrant && cd /content/qdrant && \
      curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
    !cd /content/qdrant && nohup ./qdrant > qdrant.log 2>&1 &
    !python colab_colpali_pooling_eval.py --manifest pdf_manifest.jsonl --docs 10 \
        --user-agent "capstone/0.1 you@example.com" --qdrant-url http://127.0.0.1:6333

Results go to --out (results.json, results.md). Encoded vectors are cached
there, so a rerun after a Colab disconnect skips the GPU step.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GRID = 32                      # ColPali v1.3: 448x448 input, 14 px patches -> 32x32 patch grid
VARIANTS = {                   # name -> description used in the report
    "full": "all 1024 patches + 6 special tokens (1030)",
    "mean_rows": "mean of each patch row (32) + specials = 38",
    "mean_cols": "mean of each patch column (32) + specials = 38",
    "max_rows": "element-wise max of each patch row (32) + specials = 38",
    "rows_cols": "mean rows + mean columns (64) + specials = 70",
    "mean_2x2": "mean of 2x2 patch blocks (256) + specials = 262",
    "hier_3": "hierarchical (Ward) clustering, pool factor 3 (341) + specials = 347",
    "global_mean": "one mean vector + specials = 7",
}


# ----------------------------------------------------------------------------- pooling

def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pool(vectors: np.ndarray, is_image: np.ndarray, method: str) -> np.ndarray:
    """Pool one page's multivector. Specials (non-image tokens) are always kept unpooled."""
    vectors = vectors.astype(np.float32)
    patches, specials = vectors[is_image], vectors[~is_image]
    if method == "full":
        return l2(vectors)
    if patches.shape[0] != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} image patches, got {patches.shape[0]}")
    grid = patches.reshape(GRID, GRID, -1)          # row-major: [row, column, dim]
    if method == "mean_rows":
        pooled = grid.mean(axis=1)
    elif method == "mean_cols":
        pooled = grid.mean(axis=0)
    elif method == "max_rows":
        pooled = grid.max(axis=1)
    elif method == "rows_cols":
        pooled = np.concatenate([grid.mean(axis=1), grid.mean(axis=0)])
    elif method == "mean_2x2":
        pooled = grid.reshape(GRID // 2, 2, GRID // 2, 2, -1).mean(axis=(1, 3)).reshape(-1, grid.shape[-1])
    elif method == "hier_3":
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=patches.shape[0] // 3, linkage="ward").fit_predict(l2(patches))
        pooled = np.stack([patches[labels == k].mean(axis=0) for k in range(labels.max() + 1)])
    elif method == "global_mean":
        pooled = patches.mean(axis=0, keepdims=True)
    else:
        raise ValueError(method)
    return l2(np.concatenate([l2(pooled), specials]))


# ----------------------------------------------------------------------------- pages

def download_pdfs(manifest: Path, docs: int, pdf_dir: Path, user_agent: str) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked, seen = [], set()
    for row in rows:                                 # one report per company first, for variety
        if row["ticker"] not in seen:
            picked.append(row); seen.add(row["ticker"])
    picked = (picked + [r for r in rows if r not in picked])[:docs]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for row in picked:
        path = pdf_dir / Path(row["local_path"].replace("\\", "/")).name
        if not path.exists():
            request = urllib.request.Request(row["source_url"], headers={"User-Agent": user_agent})
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())
            time.sleep(0.2)
        row["pdf"] = str(path)
    return picked


def classify(page) -> tuple[str, bool]:
    """Content kind for reporting, plus the repo's ingest rule for visual_rich pages."""
    text = page.get_text(sort=True).strip()
    try:
        tables = len(page.find_tables().tables)
    except Exception:
        tables = 0
    drawings = len(page.get_drawings())
    area = page.rect.get_area()
    big_image = any(page.get_image_bbox(img).get_area() > 0.15 * area for img in page.get_images(full=True))
    kind = "table" if tables else "chart" if drawings > 60 else "image" if big_image else "text"
    visual_rule = bool(tables or page.get_images() or drawings > 4 or not text)   # multimodal_rag.py:329
    return kind, visual_rule


def render_pages(reports: list[dict], out: Path, max_pages: int | None) -> list[dict]:
    import pymupdf
    pages, image_dir = [], out / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        with pymupdf.open(report["pdf"]) as pdf:
            for number, page in enumerate(pdf, start=1):
                if max_pages and number > max_pages:
                    break
                kind, visual_rule = classify(page)
                png = image_dir / f"{report['document_id']}_p{number:04d}.png"
                if not png.exists():
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(png)   # same scale as ingest
                pages.append({"key": f"{report['document_id']}:{number}", "doc_id": report["document_id"],
                              "ticker": report["ticker"], "page": number, "kind": kind,
                              "visual_rule": visual_rule, "png": str(png), "text": page.get_text().strip()})
    return pages


# ----------------------------------------------------------------------------- queries

def auto_queries(pages: list[dict], limit: int) -> list[dict]:
    """One query per page from a heading-like line that occurs on no other page.

    These favour text matching; add hand-written visual queries with --queries-csv.
    """
    def candidates(text):
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            words = line.split()
            letters = sum(ch.isalpha() for ch in line)
            if 4 <= len(words) <= 12 and letters >= 0.7 * len(line.replace(" ", "")) and not line.isupper():
                yield line
    counts = Counter(line.lower() for page in pages for line in set(candidates(page["text"])))
    queries, text_pages = [], 0
    for page in pages:
        if page["kind"] == "text":
            text_pages += 1
            if text_pages % 3:                             # every third text page, so visual pages dominate
                continue
        line = next((c for c in candidates(page["text"]) if counts[c.lower()] == 1), None)
        if line:
            queries.append({"query": f"{page['ticker']} {line}", "target": page["key"],
                            "kind": page["kind"], "source": "auto"})
    rng = np.random.default_rng(7)
    rng.shuffle(queries)
    return queries[:limit]


def csv_queries(path: Path, pages: list[dict]) -> list[dict]:
    import csv
    kinds = {page["key"]: page["kind"] for page in pages}
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):           # columns: query, doc_id, page
            key = f"{row['doc_id']}:{int(row['page'])}"
            if key in kinds:
                rows.append({"query": row["query"], "target": key, "kind": kinds[key], "source": "manual"})
    return rows


# ----------------------------------------------------------------------------- encoding

def load_model(model_name: str):
    import torch
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = ColPaliProcessor.from_pretrained(model_name)
    processor.query_prefix = "Query: "               # matches scripts/domain_common/colpali_retrieval.py
    model = ColPaliForRetrieval.from_pretrained(model_name, dtype=dtype, device_map="cuda",
                                                attn_implementation="sdpa").eval()
    vlm = getattr(model.config, "vlm_config", model.config)
    image_token = getattr(vlm, "image_token_id", None) or getattr(vlm, "image_token_index")
    return processor, model, image_token


def encode_pages(pages, processor, model, image_token, batch_size, cache: Path):
    import torch
    from PIL import Image
    if cache.exists():
        data = np.load(cache)
        if len(data["vectors"]) == len(pages):
            return data["vectors"], data["is_image"], json.loads(str(data["timing"]))
        print("cached vectors do not match the selected pages; re-encoding", flush=True)
    vectors, masks, started = [], [], time.time()
    for start in range(0, len(pages), batch_size):
        batch = pages[start:start + batch_size]
        images = [Image.open(page["png"]).convert("RGB") for page in batch]
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            out = model(**inputs, use_cache=False).embeddings.float()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite ColPali output; rerun on a bf16-capable GPU or in float32")
        for row, ids, attention in zip(out, inputs["input_ids"], inputs["attention_mask"]):
            keep = attention.bool()
            vectors.append(torch.nn.functional.normalize(row[keep], dim=-1).cpu().numpy().astype(np.float16))
            masks.append((ids[keep] == image_token).cpu().numpy())
        print(f"encoded {start + len(batch)}/{len(pages)} pages", flush=True)
    seconds = time.time() - started
    timing = {"pages": len(pages), "seconds": round(seconds, 1), "pages_per_second": round(len(pages) / seconds, 2),
              "gpu": torch.cuda.get_device_name(0), "dtype": str(model.dtype), "batch_size": batch_size}
    vectors, masks = np.stack(vectors), np.stack(masks)
    np.savez(cache, vectors=vectors, is_image=masks, timing=json.dumps(timing))
    return vectors, masks, timing


def encode_queries(queries, processor, model) -> list[np.ndarray]:
    import torch
    encoded = []
    for item in queries:
        inputs = processor(text=[item["query"]], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, use_cache=False).embeddings[0].float()
        out = out[inputs["attention_mask"][0].bool()]
        encoded.append(torch.nn.functional.normalize(out, dim=-1).cpu().numpy())
    return encoded


# ----------------------------------------------------------------------------- scoring

def maxsim_matrix(query_vectors: list[np.ndarray], docs: list[np.ndarray], chunk: int = 256) -> np.ndarray:
    """scores[q, p] = sum over query tokens of max over page vectors (exact MaxSim), on GPU if present."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    width = max(doc.shape[0] for doc in docs)
    padded = np.zeros((len(docs), width, docs[0].shape[1]), dtype=np.float16)
    mask = np.zeros((len(docs), width), dtype=bool)
    for i, doc in enumerate(docs):
        padded[i, :len(doc)], mask[i, :len(doc)] = doc, True
    scores = np.zeros((len(query_vectors), len(docs)), dtype=np.float32)
    for start in range(0, len(docs), chunk):
        block = torch.from_numpy(padded[start:start + chunk]).to(device, dtype)
        block_mask = torch.from_numpy(mask[start:start + chunk]).to(device)
        for qi, query in enumerate(query_vectors):
            q = torch.from_numpy(query).to(device, dtype)
            sims = torch.einsum("ld,pnd->pln", q, block).float()
            sims = sims.masked_fill(~block_mask[:, None, :], float("-inf"))
            scores[qi, start:start + chunk] = sims.max(dim=2).values.sum(dim=1).cpu().numpy()
    return scores


def ranks(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1)


def evaluate(name, order, targets, kinds, full_order, full_scores, prefetch_sizes=(20, 50, 100, 200)):
    """Recall/MRR for a ranking, its agreement with full MaxSim, and prefetch -> full rerank.

    vs_full_top20@200 repeats Qdrant's published measurement: prefetch 200, rerank with full
    ColPali, compare the top 20 with exact full-ColPali top 20 (graded relevance 20..1 for NDCG).
    """
    position = np.array([int(np.where(order[i] == t)[0][0]) for i, t in enumerate(targets)])
    result = {"variant": name, "recall@1": float((position < 1).mean()), "recall@5": float((position < 5).mean()),
              "recall@10": float((position < 10).mean()), "mrr": float((1.0 / (position + 1)).mean())}
    result["by_kind_recall@5"] = {k: float((position[kinds == k] < 5).mean()) for k in sorted(set(kinds))}
    full_top5 = full_order[:, :5]
    for n in prefetch_sizes:
        kept = [set(order[i, :n]) for i in range(len(order))]
        result[f"keeps_full_top5@{n}"] = float(np.mean([len(kept[i] & set(full_top5[i])) / 5 for i in range(len(order))]))
        reranked = []
        for i in range(len(order)):
            candidates = order[i, :n]
            best = candidates[np.argsort(-full_scores[i, candidates])][:5]
            reranked.append(targets[i] in best)
        result[f"rerank_recall@5_from_{n}"] = float(np.mean(reranked))
    recall20, ndcg20 = [], []
    for i in range(len(order)):
        candidates = order[i, :200]
        top20 = candidates[np.argsort(-full_scores[i, candidates])][:20]
        truth = {doc: 20 - rank for rank, doc in enumerate(full_order[i, :20])}
        recall20.append(len(set(top20) & set(truth)) / len(truth))
        dcg = sum((2 ** truth.get(doc, 0) - 1) / np.log2(rank + 2) for rank, doc in enumerate(top20))
        ideal = sum((2 ** rel - 1) / np.log2(rank + 2) for rank, rel in enumerate(sorted(truth.values(), reverse=True)))
        ndcg20.append(dcg / ideal)
    result["vs_full_top20@200_recall"] = float(np.mean(recall20))
    result["vs_full_top20@200_ndcg"] = float(np.mean(ndcg20))
    return result


def text_summary_order(pages, queries, model_name: str) -> np.ndarray:
    """The current production prefetch: dense text embedding of the page text (summary vector)."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name, device="cuda")
    page_vectors = encoder.encode([page["text"][:4000] or "empty page" for page in pages], normalize_embeddings=True)
    query_vectors = encoder.encode([q["query"] for q in queries], normalize_embeddings=True)
    return ranks(query_vectors @ page_vectors.T), encoder, page_vectors


def fuse(*orders: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion of several candidate rankings (as Qdrant fuses prefetches)."""
    fused = np.zeros(orders[0].shape, dtype=np.float64)
    for order in orders:
        rank_of = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        rank_of[rows, order] = np.arange(order.shape[1])[None, :]
        fused += 1.0 / (k + 1 + rank_of)
    return ranks(fused)


# ----------------------------------------------------------------------------- qdrant

def qdrant_benchmark(url, pages, pooled, page_text_vectors, query_vectors, query_text_vectors, targets,
                     top_k=20, prefetch_limit=200):
    from qdrant_client import QdrantClient, models
    client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url, prefer_grpc=True, timeout=300)
    name = "pooling_eval"
    if client.collection_exists(name):
        client.delete_collection(name)

    def multi(hnsw: bool, on_disk: bool):
        return models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=on_disk,
                                   datatype=models.Datatype.FLOAT16,
                                   multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                                   hnsw_config=None if hnsw else models.HnswConfigDiff(m=0))
    client.create_collection(name, vectors_config={
        "original": multi(hnsw=False, on_disk=True),     # rerank only: no HNSW graph, kept on disk
        "mean_rows": multi(hnsw=True, on_disk=False),    # first stage: small, indexed, in RAM
        "mean_cols": multi(hnsw=True, on_disk=False),
        "summary": models.VectorParams(size=page_text_vectors.shape[1], distance=models.Distance.COSINE)})
    started = time.time()
    for start in range(0, len(pages), 16):
        client.upsert(name, wait=True, points=[models.PointStruct(id=i, payload={"key": pages[i]["key"]}, vector={
            "original": pooled["full"][i].tolist(), "mean_rows": pooled["mean_rows"][i].tolist(),
            "mean_cols": pooled["mean_cols"][i].tolist(), "summary": page_text_vectors[i].tolist()})
            for i in range(start, min(start + 16, len(pages)))])
    upsert_seconds = time.time() - started

    exact = models.SearchParams(exact=True)

    def run(mode, q, t):
        if mode == "full_exact":              # ground truth: every page scored with all 1030 vectors
            return client.query_points(name, query=q, using="original", limit=top_k, search_params=exact).points
        if mode == "pooled_only":
            return client.query_points(name, query=q, using="mean_rows", limit=top_k).points
        if mode == "current_text_prefetch":   # multimodal_rag.retrieve_visual today
            return client.query_points(name, prefetch=models.Prefetch(query=t, using="summary", limit=max(30, top_k)),
                                       query=q, using="original", limit=top_k).points
        if mode == "qdrant_rows_prefetch":    # Qdrant blog: mean rows top 200 -> full rerank top 20
            return client.query_points(name, prefetch=models.Prefetch(query=q, using="mean_rows", limit=prefetch_limit),
                                       query=q, using="original", limit=top_k).points
        return client.query_points(name, prefetch=[
            models.Prefetch(query=q, using="mean_rows", limit=prefetch_limit),
            models.Prefetch(query=q, using="mean_cols", limit=prefetch_limit),
            models.Prefetch(query=t, using="summary", limit=prefetch_limit)], query=q, using="original", limit=top_k).points

    results = {"points": len(pages), "upsert_seconds": round(upsert_seconds, 1), "top_k": top_k,
               "prefetch_limit": prefetch_limit}
    truth = {}
    for mode in ("full_exact", "pooled_only", "current_text_prefetch", "qdrant_rows_prefetch", "recommended_three_way"):
        hits, overlap, times = 0, [], []
        for qi, (qv, tv, target) in enumerate(zip(query_vectors, query_text_vectors, targets)):
            started = time.perf_counter()
            ids = [p.id for p in run(mode, qv.tolist(), tv.tolist())]
            times.append(time.perf_counter() - started)
            if mode == "full_exact":
                truth[qi] = set(ids)
            hits += target in ids[:5]
            overlap.append(len(set(ids) & truth[qi]) / max(1, len(truth[qi])))
        results[mode] = {"label_recall@5": round(hits / len(targets), 3),
                         f"recall@{top_k}_vs_full_exact": round(float(np.mean(overlap)), 3),
                         "first_query_ms": round(1000 * times[0], 1),
                         "p50_ms": round(1000 * np.median(times), 1), "p95_ms": round(1000 * np.percentile(times, 95), 1)}
    return results


# ----------------------------------------------------------------------------- report

def write_report(out: Path, report: dict) -> None:
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# ColPali pooling evaluation", "",
             f"Pages: {report['pages']} ({report['kinds']}); queries: {report['queries']}; "
             f"encoding: {report['encoding']}", "",
             "| Variant | Vectors/page | KB/page (f16) | R@1 | R@5 | MRR | Keeps full top-5 @100 | "
             "Rerank R@5 from 100 | Rerank R@5 from 200 | Top-20 recall vs full (200) | Top-20 NDCG vs full (200) | "
             "R@5 chart | R@5 table | R@5 image | R@5 text |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["variants"]:
        kind = row["by_kind_recall@5"]
        lines.append(f"| {row['variant']} | {row.get('vectors_per_page', '-')} | {row.get('kb_per_page', '-')} | "
                     f"{row['recall@1']:.3f} | {row['recall@5']:.3f} | {row['mrr']:.3f} | {row['keeps_full_top5@100']:.3f} | "
                     f"{row['rerank_recall@5_from_100']:.3f} | {row['rerank_recall@5_from_200']:.3f} | "
                     f"{row['vs_full_top20@200_recall']:.3f} | {row['vs_full_top20@200_ndcg']:.3f} | "
                     + " | ".join(f"{kind.get(k, float('nan')):.3f}" for k in ("chart", "table", "image", "text")) + " |")
    if report.get("qdrant"):
        lines += ["", "## Qdrant", "", "```json", json.dumps(report["qdrant"], indent=2), "```"]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="edgar/pdf_manifest.jsonl from the repo")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/content/pdfs"))
    parser.add_argument("--out", type=Path, default=Path("/content/pooling_eval"))
    parser.add_argument("--docs", type=int, default=10)
    parser.add_argument("--max-pages-per-doc", type=int, default=None)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--queries-csv", type=Path, help="Hand-written queries: query,doc_id,page")
    parser.add_argument("--model", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--qdrant-url", help="Qdrant server for the latency benchmark (skipped if omitted; "
                        "':memory:' checks correctness only)")
    parser.add_argument("--user-agent", default="capstone-colpali-eval/0.1 (set --user-agent with your email)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reports = download_pdfs(args.manifest, args.docs, args.pdf_dir, args.user_agent)
    pages = render_pages(reports, args.out, args.max_pages_per_doc)
    kinds_count = dict(Counter(page["kind"] for page in pages))
    print(f"{len(pages)} pages {kinds_count}; repo visual rule marks "
          f"{100 * np.mean([p['visual_rule'] for p in pages]):.0f}%", flush=True)

    processor, model, image_token = load_model(args.model)
    vectors, is_image, timing = encode_pages(pages, processor, model, image_token, args.batch_size,
                                             args.out / "full_vectors.npz")
    queries = auto_queries(pages, args.queries) + (csv_queries(args.queries_csv, pages) if args.queries_csv else [])
    key_index = {page["key"]: i for i, page in enumerate(pages)}
    targets = np.array([key_index[q["target"]] for q in queries])
    kinds = np.array([q["kind"] for q in queries])
    query_vectors = encode_queries(queries, processor, model)

    pooled, pool_seconds = {}, {}
    for name in args.variants:
        started = time.time()
        pooled[name] = [pool(vectors[i], is_image[i], name).astype(np.float16) for i in range(len(pages))]
        pool_seconds[name] = round(1000 * (time.time() - started) / len(pages), 2)

    scores = {name: maxsim_matrix(query_vectors, pooled[name]) for name in args.variants}
    full_scores = scores["full"]
    full_order = ranks(full_scores)
    rows = []
    for name in args.variants:
        row = evaluate(name, ranks(scores[name]), targets, kinds, full_order, full_scores)
        row["vectors_per_page"] = int(np.mean([len(p) for p in pooled[name]]))
        row["kb_per_page"] = round(row["vectors_per_page"] * 128 * 2 / 1024, 1)
        row["pool_ms_per_page"] = pool_seconds[name]
        rows.append(row)

    text_order, text_encoder, page_text_vectors = text_summary_order(pages, queries, args.text_model)
    rows.append(evaluate("text_summary (current prefetch)", text_order, targets, kinds, full_order, full_scores))
    if {"mean_rows", "mean_cols"} <= set(args.variants):
        rows.append(evaluate("fused rows+cols+text (recommended prefetch)",
                             fuse(ranks(scores["mean_rows"]), ranks(scores["mean_cols"]), text_order),
                             targets, kinds, full_order, full_scores))

    report = {"pages": len(pages), "kinds": kinds_count, "queries": len(queries),
              "query_sources": dict(Counter(q["source"] for q in queries)), "encoding": timing,
              "visual_rule_share": float(np.mean([p["visual_rule"] for p in pages])), "variants": rows}
    if args.qdrant_url and {"full", "mean_rows", "mean_cols"} <= set(args.variants):
        query_text_vectors = text_encoder.encode([q["query"] for q in queries], normalize_embeddings=True)
        report["qdrant"] = qdrant_benchmark(args.qdrant_url, pages, pooled, page_text_vectors,
                                            query_vectors, query_text_vectors, targets.tolist())
    write_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
