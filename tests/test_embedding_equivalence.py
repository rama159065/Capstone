"""Pipeline 1 embeds with bge-small-en-v1.5, locally or through the shared API: one model, served either way."""
import json
import sqlite3

import pytest

from scripts.common import runtime_profile
from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import financial_ingest_chroma_db as vi
from tests.test_financial_ingest_chroma_db import args_for, corpus  # noqa: F401  (Chroma corpus fixture)

LOCAL_SIG, HOSTED_SIG, MINILM_SIG = ("local:BAAI/bge-small-en-v1.5:384", "hosted:bge-small-en-v1.5:384",
                                     "local:sentence-transformers/all-MiniLM-L6-v2:384")


class FakeEmbedder(fi.Embedder):
    """Deterministic 384-d vectors under any signature (hashing underneath): no model, no network."""

    def __init__(self, signature):
        self.provider, self.batch, self.dim, self.signature = "hashing", 64, 384, signature


def test_portable_signature():
    assert fi.portable_signature(LOCAL_SIG) == fi.portable_signature(HOSTED_SIG) == "bge-small-en-v1.5:384"
    assert fi.portable_signature(MINILM_SIG) == MINILM_SIG                       # not interchangeable: unchanged
    assert fi.portable_signature("hosted:text-embedding-3-small:1536") == "hosted:text-embedding-3-small:1536"
    assert fi.portable_signature("hashing-384-v1") == "hashing-384-v1"


def test_pipeline_1_defaults_to_bge_small(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    args = vi.build_parser().parse_args(["ingest"])
    assert args.embedding_model == "BAAI/bge-small-en-v1.5" and args.embedding_provider == "local"
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hosted")
    assert vi.build_parser().parse_args(["ingest"]).embedding_provider == "hosted"
    assert fi.build_parser().parse_args(["ingest"]).embedding_model == "sentence-transformers/all-MiniLM-L6-v2"  # pipeline 2


def test_an_index_built_locally_continues_hosted_without_redoing_filings(tmp_path, corpus):
    first = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=FakeEmbedder(LOCAL_SIG))
    assert first.run() == {"complete": 1}
    again = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=FakeEmbedder(HOSTED_SIG))
    assert again.run() == {}                                                     # same model: the filing stays done
    with sqlite3.connect(tmp_path / "chroma" / "vlm_ingest.sqlite3") as db:
        assert db.execute("SELECT value FROM settings WHERE key='embedding'").fetchone() == (LOCAL_SIG,)
    with pytest.raises(ValueError, match="embedding differs"):                     # a different model is refused
        vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=FakeEmbedder(MINILM_SIG))


def test_an_older_minilm_index_keeps_its_model_and_its_done_filings(tmp_path, corpus):
    vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=FakeEmbedder(MINILM_SIG)).run()
    with sqlite3.connect(tmp_path / "chroma" / "vlm_ingest.sqlite3") as db:
        pipeline = json.loads(db.execute("SELECT pipeline FROM filings").fetchone()[0])
    assert pipeline["embedding"] == MINILM_SIG                                   # unchanged marker: resumes as before
    with pytest.raises(ValueError, match="embedding differs"):
        vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=FakeEmbedder(LOCAL_SIG))


@pytest.fixture
def providers(monkeypatch, tmp_path):
    """Embedder construction recorded; hosted or local can be made to fail."""
    made, broken = [], set()

    def fake(provider="local", model=None, batch=64):
        made.append((provider, model))
        if provider in broken:
            raise RuntimeError(f"{provider} unavailable")
        return FakeEmbedder({"local": f"local:{model}:384", "hosted": HOSTED_SIG}.get(provider, "hashing-384-v1"))

    monkeypatch.setattr(fi, "Embedder", fake)
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)

    def index(signature):
        root = tmp_path / signature.replace(":", "_").replace("/", "_")
        if not root.exists():
            root.mkdir()
            with sqlite3.connect(root / "vlm_ingest.sqlite3") as db:
                db.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
                db.execute("INSERT INTO settings VALUES ('embedding', ?)", (signature,))
        return fi.recorded_embedder(type("A", (), {"root": root, "embedding_provider": "hashing",
                                                   "embedding_model": None})(), state_name="vlm_ingest.sqlite3")
    return made, broken, index, monkeypatch


def test_serving_uses_the_preferred_provider_for_a_bge_index(providers):
    made, broken, index, monkeypatch = providers
    assert index(LOCAL_SIG).signature == LOCAL_SIG and made[-1] == ("local", "BAAI/bge-small-en-v1.5")
    assert index(HOSTED_SIG).signature == HOSTED_SIG                             # built hosted: served hosted
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hosted")
    assert index(LOCAL_SIG).signature == HOSTED_SIG                              # built locally, served by the API
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local")
    assert index(HOSTED_SIG).signature == LOCAL_SIG                              # built by the API, served locally


def test_serving_falls_back_to_the_equivalent_provider(providers, capsys):
    made, broken, index, monkeypatch = providers
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hosted")
    broken.add("hosted")                                                         # no key or API unreachable
    assert index(LOCAL_SIG).signature == LOCAL_SIG and "embedding provider fallback" in capsys.readouterr().out
    broken.add("local")
    with pytest.raises(RuntimeError, match="no embedder for bge-small-en-v1.5"):
        index(LOCAL_SIG)


def test_a_minilm_index_is_served_with_minilm_whatever_the_preference(providers):
    made, broken, index, monkeypatch = providers
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hosted")
    assert index(MINILM_SIG).signature == MINILM_SIG and made[-1] == ("local", "sentence-transformers/all-MiniLM-L6-v2")


def test_hosted_defaults_to_the_shared_api(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.payload

    def urlopen(request, timeout, context=None):
        calls.append(request.full_url)
        n = len(json.loads(request.data)["input"])
        return Response(json.dumps({"data": [{"index": i, "embedding": [0.0] * 384} for i in range(n)]}).encode())

    monkeypatch.setattr(fi.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fi.json, "load", lambda response: json.loads(response.read()))
    for key in ("EMBEDDING_API_BASE", "EMBEDDING_MODEL", "EMBEDDING_CA_CERT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        fi.Embedder("hosted")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_TOKENIZER", "missing/tokenizer-for-test")
    embedder = fi.Embedder("hosted")
    assert embedder.signature == "hosted:bge-small-en-v1.5:384" and calls == ["https://103.42.50.41:1035/v1/embeddings"]


def test_deployed_profiles_prefer_the_shared_api(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_HOME", str(tmp_path))
    assert runtime_profile.defaults_for("gcp")["RAG_EMBEDDING_PROVIDER"] == "hosted"
    assert runtime_profile.defaults_for("server")["RAG_EMBEDDING_PROVIDER"] == "hosted"
    assert "RAG_EMBEDDING_PROVIDER" not in runtime_profile.defaults_for("local")


def test_colab_settings_pick_the_model_per_pipeline(tmp_path):
    from scripts.rag import colab_pipeline as cp
    assert cp.Settings(pipeline="vlm_chroma", drive_root=tmp_path, work=tmp_path).embedding_model == "BAAI/bge-small-en-v1.5"
    assert (cp.Settings(pipeline="colpali_qdrant", drive_root=tmp_path, work=tmp_path).embedding_model
            == "sentence-transformers/all-MiniLM-L6-v2")


class RecordingEmbedder(FakeEmbedder):
    def __init__(self, signature):
        super().__init__(signature)
        self.inputs = []

    def __call__(self, texts):
        self.inputs += list(texts)
        return super().__call__(texts)


def test_reembed_copies_the_index_with_new_vectors_and_resumable_state(tmp_path, corpus):
    old_root = tmp_path / "chroma"
    old = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "mock"), embedder=RecordingEmbedder(MINILM_SIG))
    old.run()
    original_inputs = list(old.embedder.inputs)
    new_root = tmp_path / "bge"
    bge = RecordingEmbedder(LOCAL_SIG)
    args = vi.build_parser().parse_args(["reembed", "--from-root", str(old_root), "--root", str(new_root)])
    summary = vi.reembed(args, embedder=bge)
    assert summary["counts_match"] and summary["new_count"] == old.store.count() > 0
    assert sorted(bge.inputs) == sorted(original_inputs)                       # same text as at ingest, new model
    copy = vi.ChromaStore(new_root).collection.get(include=["documents", "metadatas"])
    first = old.store.collection.get(include=["documents", "metadatas"])
    assert sorted(zip(copy["ids"], copy["documents"])) == sorted(zip(first["ids"], first["documents"]))
    with sqlite3.connect(new_root / "vlm_ingest.sqlite3") as db:
        assert db.execute("SELECT value FROM settings WHERE key='embedding'").fetchone() == (LOCAL_SIG,)
    assert vi.reembed(args, embedder=bge)["reembedded_filings"] == 0           # resumable: nothing left
    resumed = vi.VlmIngest(args_for(tmp_path / "x", corpus, "chroma", "--vlm", "mock", "--root", str(new_root)),
                           embedder=FakeEmbedder(HOSTED_SIG))
    assert resumed.run() == {}                                                 # ingest sees the filing as done
    with pytest.raises(SystemExit):
        vi.reembed(vi.build_parser().parse_args(["reembed", "--from-root", str(old_root), "--root", str(old_root)]),
                   embedder=bge)
