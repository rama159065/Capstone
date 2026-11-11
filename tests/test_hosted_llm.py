import json
import ssl

import pytest

from scripts.common import hosted_llm


def test_clean_content_strips_qwen3_think_blocks():
    assert hosted_llm.clean_content("<think>step 1\nstep 2</think>\n\nNet income was $93.7B [c1].") == \
        "Net income was $93.7B [c1]."
    assert hosted_llm.clean_content("<think>cut off by max_tokens") == ""
    assert hosted_llm.clean_content(None) == ""
    assert hosted_llm.clean_content("plain") == "plain"


def test_payload_extras_disables_thinking_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    assert hosted_llm.payload_extras() == {"max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setenv("LLM_ENABLE_THINKING", "1")
    assert hosted_llm.payload_extras(max_tokens=64) == {"max_tokens": 64}


def test_missing_ca_cert_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_CA_CERT", "no/such/dir/missing-ca.crt")
    with pytest.raises(RuntimeError, match="LLM_CA_CERT not found"):
        hosted_llm.ssl_context()
    monkeypatch.delenv("LLM_CA_CERT")
    assert hosted_llm.ssl_context() is None


def test_chat_sends_openai_request_and_reads_content(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "<think>x</think>RAG: retrieve, ground, answer."}}]}).encode()

    def fake_open(request, timeout):
        sent.update(url=request.full_url, auth=request.get_header("Authorization"), body=json.loads(request.data))
        return Response()

    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    monkeypatch.setattr(hosted_llm, "open_url", fake_open)
    text = hosted_llm.chat([{"role": "user", "content": "Explain RAG in three steps."}],
                           api_base="https://103.42.50.41:1034/v1/", api_key="k", model="Qwen/Qwen3-8B", max_tokens=256)
    assert text == "RAG: retrieve, ground, answer."
    assert sent["url"] == "https://103.42.50.41:1034/v1/chat/completions"
    assert sent["auth"] == "Bearer k"
    assert sent["body"]["model"] == "Qwen/Qwen3-8B" and sent["body"]["max_tokens"] == 256
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def _http_error(code):
    import email.message
    import urllib.error
    return urllib.error.HTTPError("https://x/v1/chat/completions", code, "err", email.message.Message(), None)


def test_open_url_backs_off_on_429_then_succeeds(monkeypatch):
    calls, waits = [], []
    monkeypatch.setenv("LLM_RETRY_DELAYS", "2,4,8")
    monkeypatch.setattr(hosted_llm.time, "sleep", waits.append)

    def fake(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429 if len(calls) == 1 else 503)
        return "ok"

    monkeypatch.setattr(hosted_llm, "_urlopen", fake)
    assert hosted_llm.open_url(object(), 5) == "ok"
    assert waits == [2.0, 4.0]


def test_open_url_stops_after_retry_limit_and_never_retries_other_errors(monkeypatch):
    waits = []
    monkeypatch.setenv("LLM_RETRY_DELAYS", "2,4,8")
    monkeypatch.setattr(hosted_llm.time, "sleep", waits.append)
    monkeypatch.setattr(hosted_llm, "_urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(503)))
    with pytest.raises(RuntimeError, match="HTTP 503 after 3 retries"):
        hosted_llm.open_url(object(), 5)
    assert waits == [2.0, 4.0, 8.0]

    import urllib.error
    waits.clear()
    monkeypatch.setattr(hosted_llm, "_urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(401)))
    with pytest.raises(urllib.error.HTTPError):
        hosted_llm.open_url(object(), 5)
    assert waits == []
