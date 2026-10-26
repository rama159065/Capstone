from io import BytesIO
import hashlib
from types import SimpleNamespace

import pytest
import requests

from scripts import download_model_files as downloader


def test_download_resumes_verified_chunks_and_checks_final_hash(tmp_path, monkeypatch):
    content = b"model data" * (1024 * 1024)
    checksum = hashlib.sha256(content).hexdigest()
    metadata = SimpleNamespace(rfilename="weights.bin", size=len(content), lfs=SimpleNamespace(sha256=checksum))
    info = SimpleNamespace(siblings=[metadata], sha="revision")
    monkeypatch.setattr(downloader, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **k: info))
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    monkeypatch.setenv("RAG_DOWNLOAD_WORKERS", "1")
    state = {"fail_second": True, "requested": []}

    class Response:
        status_code = 206
        def __init__(self, url, start, end):
            self.url = url
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
            self.raw = BytesIO(content[start:end+1])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            state["requested"].append(start)
            if start and state["fail_second"]:
                raise requests.ConnectionError("temporary failure")
            return Response(url, start, end)

    monkeypatch.setattr(downloader.requests, "Session", Session)
    with pytest.raises(requests.ConnectionError):
        downloader.download("test/repo", "weights.bin", tmp_path)
    assert not (tmp_path / "weights.bin").exists()
    state["fail_second"] = False
    state["requested"].clear()
    target = downloader.download("test/repo", "weights.bin", tmp_path)
    assert target.read_bytes() == content
    assert 0 not in state["requested"]
    state["requested"].clear()
    assert downloader.download("test/repo", "weights.bin", tmp_path) == target
    assert state["requested"] == []
