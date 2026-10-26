import base64
import json

from scripts.domain_common import local_models


def test_llamacpp_serializes_actual_image_bytes(tmp_path, monkeypatch):
    path = tmp_path / "page.png"
    path.write_bytes(b"test image bytes")
    seen = {}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"choices":[{"message":{"content":"The chart increases."}}]}'
    def urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["url"] = request.full_url
        return Response()
    monkeypatch.setattr(local_models.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("RAG_VISION_API_BASE", "http://127.0.0.1:8081/v1")
    assert local_models.llamacpp("Describe the trend", [path]) == "The chart increases."
    assert seen["url"].endswith("/v1/chat/completions")
    parts = seen["body"]["messages"][0]["content"]
    assert parts[0]["text"] == "Describe the trend"
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",")[1]) == path.read_bytes()
