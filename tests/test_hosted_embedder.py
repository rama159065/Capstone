"""The hosted embedding client: private CA, the server's batch limit and the 512-token cap. No network."""
import io
import json
import ssl

import pytest

from scripts.common import runtime_profile, storage
from scripts.rag import financial_ingest_qdrant_db as fi


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    """A fake OpenAI-compatible /embeddings endpoint that records every request."""
    calls = []

    def urlopen(request, timeout, context=None):
        body = json.loads(request.data)
        calls.append({"url": request.full_url, "inputs": body["input"], "context": context,
                      "auth": request.get_header("Authorization")})
        data = [{"index": i, "embedding": [float(len(text)), 0.0, 1.0]} for i, text in reversed(list(enumerate(body["input"])))]
        return Response(json.dumps({"data": data}).encode())

    monkeypatch.setattr(fi.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("EMBEDDING_API_BASE", "https://embeddings.example.invalid/v1/")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-small-en-v1.5")
    for key in ("EMBEDDING_CA_CERT", "EMBEDDING_MAX_BATCH", "EMBEDDING_MAX_TOKENS", "EMBEDDING_TOKENIZER"):
        monkeypatch.delenv(key, raising=False)
    return calls


class WordTokenizer:
    """One token per word, [CLS] and [SEP] added: enough to check where the trimmer cuts."""

    def num_special_tokens_to_add(self):
        return 2

    def __call__(self, text, add_special_tokens, return_offsets_mapping, truncation, verbose):
        import re
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


@pytest.fixture
def words(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, name: WordTokenizer()))


def test_requests_hold_at_most_16_texts_and_vectors_keep_their_order(server, words):
    embedder = fi.Embedder("hosted")
    assert embedder.batch == 16 and embedder.dim == 3 and embedder.signature == "hosted:bge-small-en-v1.5:3"
    texts = [f"chunk {'x' * i}" for i in range(40)]
    vectors = embedder(texts)
    sizes = [len(c["inputs"]) for c in server[1:]]                    # after the dimension probe
    assert sizes == [16, 16, 8] and [v[0] for v in vectors] == [float(len(t)) for t in texts]
    assert server[0]["url"] == "https://embeddings.example.invalid/v1/embeddings" and server[0]["auth"] == "Bearer sk-test"
    with pytest.raises(ValueError, match="limit is 16"):
        embedder._hosted(["a"] * 17)


def test_batch_limit_comes_from_the_environment(server, words, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MAX_BATCH", "4")
    embedder = fi.Embedder("hosted")
    embedder(["t"] * 10)
    assert [len(c["inputs"]) for c in server[1:]] == [4, 4, 2]


def test_texts_are_cut_to_512_tokens_at_a_token_boundary(server, words):
    embedder = fi.Embedder("hosted")
    long_text = " ".join(f"w{i}" for i in range(900))
    embedder([long_text, "short text"])
    sent = server[-1]["inputs"]
    assert len(sent[0].split()) == 510 and long_text.startswith(sent[0]) and sent[0].endswith("w509")
    assert sent[1] == "short text" and embedder.trimmer.trimmed == 1 and embedder.trimmer.name == "BAAI/bge-small-en-v1.5"


def test_without_the_tokenizer_texts_are_cut_by_characters(server, monkeypatch):
    import transformers

    def offline(cls, name):
        raise OSError("not cached")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(offline))
    embedder = fi.Embedder("hosted")
    embedder(["y" * 5000])
    assert server[-1]["inputs"] == ["y" * 1200] and embedder.trimmer.tokenizer is None


def test_the_private_ca_is_used_for_every_request(server, words, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_CA_CERT", str(tmp_path / "missing.crt"))
    with pytest.raises(RuntimeError, match="EMBEDDING_CA_CERT not found"):
        fi.Embedder("hosted")
    contexts = []
    monkeypatch.setattr(ssl, "create_default_context", lambda cafile: contexts.append(cafile) or "ctx")
    cert = tmp_path / "embedding_ca.crt"
    cert.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_CA_CERT", str(cert))
    fi.Embedder("hosted")(["a", "b"])
    assert contexts == [str(cert)] and {c["context"] for c in server} == {"ctx"}
    monkeypatch.delenv("EMBEDDING_CA_CERT")
    assert fi.embedding_ssl_context() is None                          # public endpoints: the system's CAs


def test_wrong_vector_count_fails_closed(server, words, monkeypatch):
    embedder = fi.Embedder("hosted")
    monkeypatch.setattr(fi.urllib.request, "urlopen", lambda request, timeout, context=None: Response(
        json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}).encode()))
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        embedder._hosted(["a", "b"])


def test_runtime_profile_points_at_the_embedding_ca_in_secrets(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "secrets").mkdir(parents=True)
    monkeypatch.setenv("RAG_HOME", str(home))
    assert "EMBEDDING_CA_CERT" not in runtime_profile.defaults_for("local")
    (home / "secrets" / storage.EMBEDDING_CA_CERT).write_text("pem", encoding="utf-8")
    assert runtime_profile.defaults_for("local")["EMBEDDING_CA_CERT"] == str(home / "secrets" / "embedding_ca.crt")
