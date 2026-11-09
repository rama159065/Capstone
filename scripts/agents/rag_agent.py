"""Generic PDF RAG + guarded agent runtime for Track B domains."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer

from scripts.common.agent_limits import STEP_LIMIT_ANSWER, graph_config, max_steps
from scripts.guardrails import checks as guard_checks, patterns as guard_patterns
from scripts.prompts import domain as prompts
from scripts.observability.langsmith_utils import (
    add_langsmith_args,
    maybe_enable_from_args,
    run_traced,
)


@dataclass(frozen=True)
class DomainConfig:
    domain: str
    display_name: str
    description: str
    intent_terms: tuple[str, ...]
    review_terms: tuple[str, ...]
    source_types: tuple[str, ...]
    default_query: str


@dataclass(frozen=True)
class PdfDocument:
    document_id: str
    title: str
    source_name: str
    source_url: str
    source_type: str
    local_path: Path
    content_sha256: str


class MockResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class MockDomainLLM:
    def __init__(self, domain_config: DomainConfig) -> None:
        self.domain_config = domain_config

    def invoke(self, messages: list[Any]) -> MockResponse:
        joined = "\n".join(getattr(message, "content", str(message)) for message in messages)
        lowered = joined.lower()
        chunk_ids = [
            chunk_id
            for chunk_id in re.findall(r"chunk_id:\s*([A-Za-z0-9_.=-]+)", joined)
            if "EXAMPLE" not in chunk_id
        ]
        citations = sorted(set(chunk_ids))[:3]
        if "plain-text classifier" in lowered:
            query_match = re.search(r"Message:\s*(.*?)\n\nReturn JSON", joined, flags=re.DOTALL)
            query = query_match.group(1).strip() if query_match else joined
            query_lower = query.lower()
            blocked_terms = {"<script", "```", "select *", "ignore previous"}
            return MockResponse(
                json.dumps(
                    {
                        "is_plain_text": not any(term in query_lower for term in blocked_terms),
                        "domain_intent": any(term in query_lower for term in self.domain_config.intent_terms),
                        "risk_level": "low",
                        "reason": "Mock classifier decision.",
                        "safe_query": query,
                    }
                )
            )
        if "plan json" in lowered:
            return MockResponse(
                json.dumps(
                    {
                        "intent": self.domain_config.domain,
                        "tools": ["hybrid_search"],
                        "source_type": None,
                        "needs_human_review": False,
                    }
                )
            )
        return MockResponse(
            "Mock answer generated from retrieved evidence. "
            f"Citations: {', '.join(citations) if citations else 'none'}."
        )


class HostedChatLLM:
    def __init__(self, *, api_base: str, api_key: str, model: str, temperature: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def invoke(self, messages):
        from scripts.common.resilience import Calls, fingerprint
        calls = Calls(float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45")))
        key = fingerprint({"endpoint": self.api_base, "model": self.model,
                           "credential": hashlib.sha256(self.api_key.encode()).hexdigest(),
                           "temperature": self.temperature,
                           "messages": [str(getattr(m, "content", m)) + str(type(m)) for m in messages]})
        return calls.run("hosted_model", lambda: self._invoke(messages), key=key, retry=True)

    def _invoke(self, messages: list[Any]) -> MockResponse:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [self._message_to_dict(message) for message in messages],
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request.add_header("Idempotency-Key", hashlib.sha256(request.data).hexdigest())
        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45"))) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise
        data = json.loads(raw)
        return MockResponse(str(data["choices"][0]["message"]["content"]))

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, str]:
        role = "system" if isinstance(message, SystemMessage) else "user"
        if getattr(message, "type", None) == "ai":
            role = "assistant"
        return {"role": role, "content": str(getattr(message, "content", message))}


class HashingEmbeddings:
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
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.api_base}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [row["embedding"] for row in sorted(payload["data"], key=lambda item: item["index"])]


# Guardrail patterns and checks live in scripts/guardrails; the domain agents use the domain pattern set.
PROMPT_INJECTION_PATTERNS = guard_patterns.DOMAIN_PROMPT_INJECTION_PATTERNS
CODE_OR_TOOL_PATTERNS = guard_patterns.DOMAIN_CODE_OR_TOOL_PATTERNS
SECRET_PATTERNS = guard_patterns.SECRET_PATTERNS
SENSITIVE_DATA_PATTERNS = guard_patterns.SENSITIVE_DATA_PATTERNS


def domain_paths(domain: str) -> dict[str, Path]:
    return {
        "manifest": Path(f"data/raw/{domain}/pdf_manifest.jsonl"),
        "chunks": Path(f"data/processed/{domain}/pdf_chunks.jsonl"),
        "index": Path(f"indexes/chroma/{domain}_pdfs"),
        "html": Path(f"output/{domain}_agent_result.html"),
    }


def collection_name(domain: str) -> str:
    return f"{domain}_pdfs"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_manifest(path: Path) -> list[PdfDocument]:
    docs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                PdfDocument(
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    source_name=str(row["source_name"]),
                    source_url=str(row["source_url"]),
                    source_type=str(row["source_type"]),
                    local_path=Path(row["local_path"]),
                    content_sha256=str(row["content_sha256"]),
                )
            )
    return docs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf_pages(doc: PdfDocument, max_pages: int | None = None) -> list[tuple[int, str]]:
    if sha256_file(doc.local_path) != doc.content_sha256:
        raise ValueError(f"SHA mismatch for {doc.local_path}")
    reader = PdfReader(str(doc.local_path))
    if reader.is_encrypted:
        reader.decrypt("")
    pages_to_read = reader.pages[:max_pages] if max_pages else reader.pages
    pages = []
    for page_number, page in enumerate(pages_to_read, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def build_pdf_chunks(
    domain_config: DomainConfig,
    docs: list[PdfDocument],
    chunk_size: int,
    chunk_overlap: int,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        for page, text in extract_pdf_pages(doc, max_pages=max_pages):
            for chunk_index, chunk_text in enumerate(splitter.split_text(text)):
                chunk_text = normalize_text(chunk_text)
                if len(chunk_text) < 80:
                    continue
                chunk_id = f"{doc.document_id}_p{page:04d}_c{chunk_index:03d}"
                metadata = {
                    "domain": domain_config.domain,
                    "source_type": doc.source_type,
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "source_name": doc.source_name,
                    "source_url": doc.source_url,
                    "local_path": str(doc.local_path),
                    "page": page,
                    "chunk_id": chunk_id,
                }
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc.document_id,
                        "text": chunk_text,
                        "metadata": metadata,
                    }
                )
    return chunks


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_embedder(provider: str, dimensions: int):
    if provider == "hashing":
        return HashingEmbeddings(dimensions)
    if provider == "hosted":
        missing = [
            name for name in ("EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(f"Missing hosted embedding env var(s): {', '.join(missing)}")
        return HostedEmbeddings(
            api_base=str(os.environ["EMBEDDING_API_BASE"]),
            api_key=str(os.environ["EMBEDDING_API_KEY"]),
            model=str(os.environ["EMBEDDING_MODEL"]),
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def get_llm(provider: str, domain_config: DomainConfig, temperature: float = 0.0):
    if provider == "mock":
        return MockDomainLLM(domain_config)
    if provider == "hosted":
        missing = [
            name for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(f"Missing hosted LLM env var(s): {', '.join(missing)}")
        return HostedChatLLM(
            api_base=str(os.environ["LLM_API_BASE"]),
            api_key=str(os.environ["LLM_API_KEY"]),
            model=str(os.environ["LLM_MODEL"]),
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def index_domain_pdfs(
    domain_config: DomainConfig,
    *,
    embedding_provider: str,
    hash_dimensions: int,
    chunk_size: int,
    chunk_overlap: int,
    max_pages: int | None,
    reset: bool,
    batch_size: int,
) -> dict[str, Any]:
    paths = domain_paths(domain_config.domain)
    docs = read_pdf_manifest(paths["manifest"])
    chunks = build_pdf_chunks(domain_config, docs, chunk_size, chunk_overlap, max_pages)
    write_jsonl(paths["chunks"], chunks)

    client = chromadb.PersistentClient(path=str(paths["index"]))
    if reset:
        try:
            client.delete_collection(collection_name(domain_config.domain))
        except Exception:
            pass
    collection = client.get_or_create_collection(
        collection_name(domain_config.domain),
        metadata={"hnsw:space": "cosine", "domain": domain_config.domain},
    )
    embedder = get_embedder(embedding_provider, hash_dimensions)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = embedder.embed_documents([row["text"] for row in batch])
        collection.upsert(
            ids=[row["chunk_id"] for row in batch],
            documents=[row["text"] for row in batch],
            metadatas=[row["metadata"] for row in batch],
            embeddings=embeddings,
        )
    summary = {
        "domain": domain_config.domain,
        "pdfs": len(docs),
        "chunks": len(chunks),
        "collection": collection_name(domain_config.domain),
        "collection_count": collection.count(),
        "chunks_path": str(paths["chunks"]),
        "index_path": str(paths["index"]),
    }
    paths["index"].mkdir(parents=True, exist_ok=True)
    (paths["index"] / "index_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]*", text.lower())


def bm25_search(rows: list[dict[str, Any]], query: str, top_k: int, source_type: str | None) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows
        if not source_type or row.get("metadata", {}).get("source_type") == source_type
    ]
    if not filtered:
        return []
    bm25 = BM25Okapi([tokenize(str(row.get("text", ""))) for row in filtered])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    hits = []
    for rank, (index, score) in enumerate(ranked[:top_k], start=1):
        row = filtered[index]
        hits.append(
            {
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": row["metadata"],
                "bm25_score": float(score),
            }
        )
    return hits


def dense_search(
    domain_config: DomainConfig,
    query: str,
    top_k: int,
    embedding_provider: str,
    hash_dimensions: int,
    source_type: str | None,
) -> list[dict[str, Any]]:
    paths = domain_paths(domain_config.domain)
    client = chromadb.PersistentClient(path=str(paths["index"]))
    collection = client.get_collection(collection_name(domain_config.domain))
    where = {"source_type": source_type} if source_type else None
    embedder = get_embedder(embedding_provider, hash_dimensions)
    result = collection.query(
        query_embeddings=[embedder.embed_documents([query])[0]],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for rank, chunk_id in enumerate(result.get("ids", [[]])[0], start=1):
        metadata = result.get("metadatas", [[]])[0][rank - 1] or {}
        text = result.get("documents", [[]])[0][rank - 1] or ""
        hits.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": result.get("distances", [[]])[0][rank - 1],
            }
        )
    return hits


def rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def lexical_overlap(query: str, text: str) -> float:
    terms = {term for term in tokenize(query) if len(term) > 2}
    if not terms:
        return 0.0
    return len(terms & set(tokenize(text))) / len(terms)


def retrieve_hybrid(
    domain_config: DomainConfig,
    *,
    query: str,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    embedding_provider: str,
    hash_dimensions: int,
    source_type: str | None,
) -> list[dict[str, Any]]:
    if os.getenv("RAG_BACKEND") == "qdrant":
        from scripts.rag.multimodal_rag import retrieve_local
        return retrieve_local(domain_config.domain, query, top_k, source_type=source_type)
    rows = read_jsonl(domain_paths(domain_config.domain)["chunks"])
    dense_hits = dense_search(
        domain_config, query, dense_top_k, embedding_provider, hash_dimensions, source_type
    )
    bm25_hits = bm25_search(rows, query, bm25_top_k, source_type)
    fused: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        fused[str(hit["chunk_id"])] = dict(hit) | {"rrf_score": rrf(rank)}
    for rank, hit in enumerate(bm25_hits, start=1):
        chunk_id = str(hit["chunk_id"])
        fused.setdefault(chunk_id, dict(hit) | {"rrf_score": 0.0})
        fused[chunk_id]["bm25_score"] = hit.get("bm25_score")
        fused[chunk_id]["rrf_score"] = float(fused[chunk_id].get("rrf_score") or 0.0) + rrf(rank)
    hits = list(fused.values())
    for hit in hits:
        hit["rerank_score"] = float(hit.get("rrf_score") or 0.0) + 0.2 * lexical_overlap(query, hit.get("text", ""))
    hits = sorted(hits, key=lambda item: item.get("rerank_score") or 0.0, reverse=True)
    for rank, hit in enumerate(hits[:top_k], start=1):
        hit["rank"] = rank
    return hits[:top_k]


canonicalize = guard_checks.canonicalize
regex_labels = guard_checks.regex_labels
redact_sensitive = guard_checks.redact


def input_guard(message: str) -> dict[str, Any]:
    return guard_checks.input_guard(message, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS)


def invoke_prompt(llm, system: str, human: str) -> str:
    return str(llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content)


def safe_json_loads(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return fallback


def plain_text_classifier(llm, domain_config: DomainConfig, message: str) -> dict[str, Any]:
    text = invoke_prompt(llm, prompts.CLASSIFIER_SYSTEM.format(display_name=domain_config.display_name),
                         prompts.CLASSIFIER_HUMAN.format(message=message))
    fallback = {
        "is_plain_text": True,
        "domain_intent": any(term in message.lower() for term in domain_config.intent_terms),
        "risk_level": "low",
        "reason": "Fallback classifier decision.",
        "safe_query": message,
    }
    parsed = safe_json_loads(text, fallback)
    parsed["allowed"] = bool(parsed.get("is_plain_text")) and bool(parsed.get("domain_intent")) and str(
        parsed.get("risk_level", "medium")
    ).lower() not in {"medium", "high", "critical"}
    parsed["safe_query"] = canonicalize(str(parsed.get("safe_query") or message))
    return parsed


def evidence_guard(hits: list[dict[str, Any]]) -> dict[str, Any]:
    result = guard_checks.evidence_guard(hits, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS)
    return {"safe": result["safe"], "flagged_chunks": result["flagged_chunks"]}


def output_content_filter(answer: str, domain_config: DomainConfig) -> dict[str, Any]:
    return guard_checks.output_filter(answer, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS,
                                      [(domain_config.review_terms, "domain_policy_sensitive_output")])


def evidence_context(hits: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    used = 0
    for hit in hits:
        meta = hit.get("metadata", {})
        block = "\n".join(
            [
                "---",
                f"chunk_id: {hit.get('chunk_id')}",
                f"title: {meta.get('title')}",
                f"source_type: {meta.get('source_type')}",
                f"page: {meta.get('page')}",
                f"source_url: {meta.get('source_url')}",
                "text:",
                str(hit.get("text", "")),
            ]
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def validate_citations(answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {str(hit["chunk_id"]) for hit in hits}
    cited = {chunk_id for chunk_id in allowed if chunk_id in answer}
    return {
        "valid": bool(cited) and cited <= allowed,
        "cited_chunk_ids": sorted(cited),
        "missing_chunk_ids": [],
        "allowed_chunk_ids": sorted(allowed),
    }


def summarize_evidence(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for hit in hits:
        meta = hit.get("metadata", {})
        rows.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "title": meta.get("title"),
                "source_type": meta.get("source_type"),
                "page": meta.get("page"),
                "source_url": meta.get("source_url"),
                "preview": normalize_text(str(hit.get("text", "")))[:500],
                "rank": hit.get("rank"),
                "score": hit.get("rerank_score"),
            }
        )
    return rows


def human_review_router(
    domain_config: DomainConfig,
    query: str,
    answer: str,
    hits: list[dict[str, Any]],
    citation_check: dict[str, Any],
    security: dict[str, Any],
    evidence_check: dict[str, Any],
    output_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lowered = f"{query} {answer}".lower()
    triggers = []
    if any(term in lowered for term in domain_config.review_terms):
        triggers.append("domain_high_impact_decision")
    if not hits:
        triggers.append("low_retrieval_confidence")
    if not citation_check.get("valid"):
        triggers.append("citation_failure")
    if security.get("decision") == "review":
        triggers.append("security_sensitive_input")
    if not evidence_check.get("safe", True):
        triggers.append("retrieved_evidence_prompt_injection")
    if output_filter and output_filter.get("decision") == "review":
        triggers.append("output_content_filter_review")
    queue_id = hashlib.sha256(f"{domain_config.domain}|{query}|{','.join(triggers)}".encode("utf-8")).hexdigest()[:12]
    return {
        "required": bool(triggers),
        "status": "pending_human_review" if triggers else "auto_approved",
        "queue_id": f"human_review_{queue_id}" if triggers else None,
        "triggers": sorted(set(triggers)),
    }


class DomainAgentState(TypedDict, total=False):
    query: str
    safe_query: str
    final_status: str
    answer: str
    security: dict[str, Any]
    plain_text: dict[str, Any]
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    evidence_guard: dict[str, Any]
    output_filter: dict[str, Any]
    citation_check: dict[str, Any]
    human_review: dict[str, Any]


def run_domain_agent(
    domain_config: DomainConfig,
    *,
    query: str,
    llm_provider: str,
    embedding_provider: str,
    hash_dimensions: int,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    source_type: str | None,
    max_context_chars: int,
    enable_guardrails: bool,
    enable_human_review: bool,
) -> dict[str, Any]:
    from scripts.common.resilience import Calls, CallFailure
    calls = Calls(float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "60")))
    started = time.time()
    llm = get_llm(llm_provider, domain_config)

    def security_node(state: DomainAgentState) -> DomainAgentState:
        security = input_guard(state["query"]) if enable_guardrails else {
            "decision": "allow",
            "risk_labels": [],
            "sanitized_query": state["query"],
        }
        if security["decision"] == "block":
            return {
                "security": security,
                "final_status": "refused",
                "answer": "Request blocked by security guardrails.",
                "evidence": [],
            }
        return {"security": security, "safe_query": str(security["sanitized_query"])}

    def plain_text_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        plain_text = plain_text_classifier(llm, domain_config, state["safe_query"]) if enable_guardrails else {
            "allowed": True,
            "safe_query": state["safe_query"],
        }
        if not plain_text.get("allowed"):
            return {
                "plain_text": plain_text,
                "final_status": "refused",
                "answer": "Request is not a safe plain-text question for this domain.",
                "evidence": [],
            }
        return {"plain_text": plain_text, "safe_query": str(plain_text["safe_query"])}

    def planner_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        plan = safe_json_loads(
            invoke_prompt(llm, prompts.PLANNER_SYSTEM.format(display_name=domain_config.display_name),
                          prompts.PLANNER_HUMAN.format(query=state["safe_query"])),
            {"intent": domain_config.domain, "tools": ["hybrid_search"], "source_type": source_type},
        )
        return {"plan": plan}

    def retrieval_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        plan = state.get("plan", {})
        hits = retrieve_hybrid(
            domain_config,
            query=state["safe_query"],
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            embedding_provider=embedding_provider,
            hash_dimensions=hash_dimensions,
            source_type=plan.get("source_type") or source_type,
        )
        return {"evidence": hits}

    def evidence_guard_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        hits = state.get("evidence", [])
        evidence_check = evidence_guard(hits) if enable_guardrails else {"safe": True, "flagged_chunks": []}
        if not evidence_check["safe"]:
            return {
                "evidence_guard": evidence_check,
                "final_status": "refused",
                "answer": "Retrieved evidence contained unsafe instructions.",
            }
        return {"evidence_guard": evidence_check}

    def answer_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        hits = state.get("evidence", [])
        if not hits:
            return {"answer": "No answer found in the available evidence.",
                    "final_status": "insufficient_evidence"}
        if llm_provider == "hosted":
            records = [{"source_id": h["chunk_id"], "text": h.get("text", "")} for h in hits]
            # Send only whole evidence records that fit the configured context budget.
            selected_records, used = [], 0
            for record in records:
                size = len(json.dumps(record))
                if used + size <= max_context_chars:
                    selected_records.append(record)
                    used += size
            if not selected_records:
                return {"answer": "No answer found within the context budget.", "final_status": "insufficient_evidence"}
            raw = invoke_prompt(llm, prompts.EVIDENCE_SELECTION_SYSTEM,
                                json.dumps({"question": state["safe_query"], "evidence": selected_records}))
            try:
                rows = json.loads(raw)["selections"]
                sources = {r["source_id"]: r["text"] for r in selected_records}
                if not isinstance(rows, list) or not rows:
                    raise ValueError("No selections")
                if any(not isinstance(r.get("quote"), str) or not r["quote"].strip()
                       or r["quote"] not in sources.get(r.get("source_id"), "") for r in rows):
                    raise ValueError("Unsupported selection")
                return {"answer": "Evidence excerpts:\n" + "\n".join(
                    r["quote"] + " [" + r["source_id"] + "]" for r in rows), "final_status": "evidence_found"}
            except (ValueError, TypeError, KeyError, AttributeError):
                return {"answer": "No answer found that could be verified against the evidence.", "final_status": "insufficient_evidence"}
        answer = invoke_prompt(llm, prompts.ANSWER_SYSTEM.format(display_name=domain_config.display_name),
                               prompts.ANSWER_HUMAN.format(query=state["safe_query"],
                                                           context=evidence_context(hits, max_context_chars)))
        return {"answer": answer}

    def output_filter_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {}
        answer = str(state.get("answer", ""))
        output_check = output_content_filter(answer, domain_config) if enable_guardrails else {
            "decision": "allow",
            "risk_labels": [],
            "redacted_preview": answer[:800],
        }
        if output_check["decision"] == "block":
            return {
                "output_filter": output_check,
                "final_status": "refused",
                "answer": "Generated answer was blocked by the output content filter.",
            }
        return {"output_filter": output_check}

    def citation_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {
                "citation_check": {
                    "valid": False,
                    "cited_chunk_ids": [],
                    "missing_chunk_ids": [],
                    "allowed_chunk_ids": [str(hit.get("chunk_id")) for hit in state.get("evidence", [])],
                }
            }
        return {"citation_check": validate_citations(str(state.get("answer", "")), state.get("evidence", []))}

    def governance_node(state: DomainAgentState) -> DomainAgentState:
        if state.get("final_status") == "refused":
            return {
                "human_review": {"required": False, "status": "not_required", "triggers": []},
                "final_status": "refused",
            }
        human_review = human_review_router(
            domain_config,
            state.get("safe_query", query),
            str(state.get("answer", "")),
            state.get("evidence", []),
            state.get("citation_check", {}),
            state.get("security", {}),
            state.get("evidence_guard", {}),
            state.get("output_filter"),
        ) if enable_human_review else {"required": False, "status": "disabled", "triggers": []}
        return {
            "human_review": human_review,
            "final_status": human_review["status"] if human_review.get("required") else state.get("final_status", "answered"),
        }

    def continue_or_end(state: DomainAgentState) -> Literal["continue", "end"]:
        return "end" if state.get("final_status") == "refused" else "continue"

    def bounded(name, fn):
        def node(state):
            if state.get("final_status") in {"tool_error", "insufficient_evidence"}:
                return {}
            try:
                return calls.run(name, lambda: fn(state))
            except CallFailure:
                return {"final_status": "tool_error", "answer": "No answer found because a required service failed or timed out."}
        return node

    graph = StateGraph(DomainAgentState)
    graph.add_node("security_guard_step", bounded("security_guard_step", security_node))
    graph.add_node("plain_text_classifier_step", bounded("plain_text_classifier_step", plain_text_node))
    graph.add_node("planner_step", bounded("planner_step", planner_node))
    graph.add_node("hybrid_retriever_step", bounded("hybrid_retriever_step", retrieval_node))
    graph.add_node("evidence_guard_step", bounded("evidence_guard_step", evidence_guard_node))
    graph.add_node("answer_agent_step", bounded("answer_agent_step", answer_node))
    graph.add_node("output_filter_step", bounded("output_filter_step", output_filter_node))
    graph.add_node("citation_critic_step", bounded("citation_critic_step", citation_node))
    graph.add_node("governance_step", bounded("governance_step", governance_node))
    graph.set_entry_point("security_guard_step")
    graph.add_conditional_edges("security_guard_step", continue_or_end, {"continue": "plain_text_classifier_step", "end": "citation_critic_step"})
    graph.add_conditional_edges("plain_text_classifier_step", continue_or_end, {"continue": "planner_step", "end": "citation_critic_step"})
    graph.add_edge("planner_step", "hybrid_retriever_step")
    graph.add_edge("hybrid_retriever_step", "evidence_guard_step")
    graph.add_conditional_edges("evidence_guard_step", continue_or_end, {"continue": "answer_agent_step", "end": "citation_critic_step"})
    graph.add_edge("answer_agent_step", "output_filter_step")
    graph.add_conditional_edges("output_filter_step", continue_or_end, {"continue": "citation_critic_step", "end": "citation_critic_step"})
    graph.add_edge("citation_critic_step", "governance_step")
    graph.add_edge("governance_step", END)
    try:
        final_state = graph.compile().invoke({"query": query}, config=graph_config())
    except GraphRecursionError:
        final_state = {"final_status": "step_limit", "answer": STEP_LIMIT_ANSWER,
                       "citation_check": {"valid": False, "reason": f"step limit of {max_steps()} reached"}}

    hits = final_state.get("evidence", [])
    return {
        "domain": domain_config.domain,
        "display_name": domain_config.display_name,
        "final_status": final_state.get("final_status", "answered"),
        "query": query,
        "safe_query": final_state.get("safe_query", query),
        "answer": final_state.get("answer"),
        "security": final_state.get("security"),
        "plain_text": final_state.get("plain_text"),
        "plan": final_state.get("plan"),
        "evidence_guard": final_state.get("evidence_guard"),
        "output_filter": final_state.get("output_filter"),
        "citation_check": final_state.get("citation_check"),
        "human_review": final_state.get("human_review"),
        "evidence": summarize_evidence(hits),
        "orchestration": {
            "engine": "langgraph",
            "nodes": [
                "security_guard",
                "plain_text_classifier",
                "planner",
                "hybrid_retriever",
                "evidence_guard",
                "answer_agent",
                "output_filter",
                "citation_critic",
                "governance",
            ],
        },
        "trace_id": calls.trace_id,
        "trace": calls.events,
        "grounding": {"method": "verbatim_quotes" if llm_provider == "hosted" else "mock", "semantic_answer_verified": False},
        "elapsed_seconds": round(time.time() - started, 3),
    }


def render_html_result(result: dict[str, Any]) -> str:
    evidence_items = "\n".join(
        f"<li><strong>{html.escape(str(row.get('chunk_id')))}</strong> "
        f"{html.escape(str(row.get('title')))} p.{html.escape(str(row.get('page')))}"
        f"<pre>{html.escape(str(row.get('preview')))}</pre></li>"
        for row in result.get("evidence", [])
    )
    guardrail_json = json.dumps(
        {
            "security": result.get("security"),
            "plain_text": result.get("plain_text"),
            "evidence_guard": result.get("evidence_guard"),
            "output_filter": result.get("output_filter"),
        },
        indent=2,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(str(result.get('display_name', 'Domain Agent')))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #eef; }}
    pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 12px; border-radius: 4px; }}
    section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>{html.escape(str(result.get('display_name', 'Domain Agent')))}</h1>
  <p class="status">{html.escape(str(result.get('final_status')))}</p>
  <section><h2>Answer</h2><p>{html.escape(str(result.get('answer')))}</p></section>
  <section><h2>Citations</h2><pre>{html.escape(json.dumps(result.get('citation_check'), indent=2))}</pre></section>
  <section><h2>Guardrails</h2><pre>{html.escape(guardrail_json)}</pre></section>
  <section><h2>Human Review</h2><pre>{html.escape(json.dumps(result.get('human_review'), indent=2))}</pre></section>
  <section><h2>Evidence</h2><ol>{evidence_items}</ol></section>
</body>
</html>"""


def add_common_agent_args(parser: argparse.ArgumentParser, domain_config: DomainConfig) -> None:
    parser.add_argument("--query", default=domain_config.default_query)
    parser.add_argument("--llm-provider", choices=["mock", "hosted"], default="mock")
    parser.add_argument("--embedding-provider", choices=["hashing", "hosted"], default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--source-type")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--disable-guardrails", action="store_true")
    parser.add_argument("--disable-human-review", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--html-out", type=Path)


def agent_cli(domain_config: DomainConfig) -> int:
    parser = argparse.ArgumentParser(description=f"{domain_config.display_name} guarded RAG agent")
    add_common_agent_args(parser, domain_config)
    add_langsmith_args(parser, default_project=f"capstone-{domain_config.domain}")
    args = parser.parse_args()
    langsmith_status = maybe_enable_from_args(args)
    result = run_traced(
        f"{domain_config.domain}.agent",
        run_domain_agent,
        domain_config,
        run_type="chain",
        tags=[
            "capstone",
            domain_config.domain,
            args.llm_provider,
            args.embedding_provider,
            *args.langsmith_tag,
        ],
        metadata={
            "domain": domain_config.domain,
            "top_k": args.top_k,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "source_type": args.source_type,
            "guardrails_enabled": not args.disable_guardrails,
            "human_review_enabled": not args.disable_human_review,
        },
        query=args.query,
        llm_provider=args.llm_provider,
        embedding_provider=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        source_type=args.source_type,
        max_context_chars=args.max_context_chars,
        enable_guardrails=not args.disable_guardrails,
        enable_human_review=not args.disable_human_review,
    )
    result["langsmith"] = langsmith_status
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.html_out:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(render_html_result(result), encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    return 0


def index_cli(domain_config: DomainConfig) -> int:
    parser = argparse.ArgumentParser(description=f"Index {domain_config.display_name} PDFs")
    parser.add_argument("--embedding-provider", choices=["hashing", "hosted"], default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    parser.add_argument("--max-pages-per-pdf", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--query")
    args = parser.parse_args()
    summary = index_domain_pdfs(
        domain_config,
        embedding_provider=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_pages=args.max_pages_per_pdf,
        reset=args.reset,
        batch_size=args.batch_size,
    )
    sys.stdout.buffer.write((json.dumps(summary, indent=2) + "\n").encode("utf-8", errors="replace"))
    if args.query:
        hits = retrieve_hybrid(
            domain_config,
            query=args.query,
            top_k=5,
            dense_top_k=30,
            bm25_top_k=30,
            embedding_provider=args.embedding_provider,
            hash_dimensions=args.hash_dimensions,
            source_type=None,
        )
        sys.stdout.buffer.write((json.dumps(summarize_evidence(hits), indent=2) + "\n").encode("utf-8", errors="replace"))
    return 0


def streamlit_app(domain_config: DomainConfig) -> None:
    import streamlit as st

    st.set_page_config(page_title=domain_config.display_name, layout="wide")
    st.title(domain_config.display_name)
    st.caption(domain_config.description)
    with st.sidebar:
        llm_provider = st.selectbox("LLM provider", ["mock", "hosted"])
        embedding_provider = st.selectbox("Embedding provider", ["hashing", "hosted"])
        top_k = st.slider("Top k", 1, 12, 5)
        source_type = st.selectbox("Source type", ["", *domain_config.source_types])
        guardrails = st.toggle("Guardrails", True)
        hitl = st.toggle("Human review routing", True)
    query = st.chat_input(domain_config.default_query)
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Running guarded RAG agent..."):
                result = run_domain_agent(
                    domain_config,
                    query=query,
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    hash_dimensions=384,
                    top_k=top_k,
                    dense_top_k=30,
                    bm25_top_k=30,
                    source_type=source_type or None,
                    max_context_chars=12000,
                    enable_guardrails=guardrails,
                    enable_human_review=hitl,
                )
            st.write(result["answer"])
            st.info(f"Status: {result['final_status']}")
            with st.expander("Citations and evidence", expanded=True):
                st.json({"citation_check": result.get("citation_check"), "evidence": result.get("evidence")})
            with st.expander("Guardrails and HITL"):
                st.json(
                    {
                        "security": result.get("security"),
                        "plain_text": result.get("plain_text"),
                        "evidence_guard": result.get("evidence_guard"),
                        "human_review": result.get("human_review"),
                    }
                )
            with st.expander("HTML result"):
                st.components.v1.html(render_html_result(result), height=600, scrolling=True)


def ensure_repo_root_on_path(file_path: str) -> None:
    root = next(p for p in Path(file_path).resolve().parents if (p / "pyproject.toml").exists())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

