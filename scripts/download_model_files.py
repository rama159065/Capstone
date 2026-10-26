"""Verified, ranged HTTPS downloader for large local model files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from threading import Lock, local
import time

import requests
from huggingface_hub import HfApi


def download(repo, filename, folder):
    info = HfApi().model_info(repo, files_metadata=True, timeout=30)
    metadata = next(f for f in info.siblings if f.rfilename == filename)
    target = Path(folder) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = metadata.lfs.sha256
    def valid(path):
        if not path.exists() or path.stat().st_size != metadata.size:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == checksum
    if valid(target):
        return target
    part = target.with_suffix(target.suffix + ".download")
    block = 8 * 1024 * 1024
    ranges = [(i, min(i + block, metadata.size)-1) for i in range(0, metadata.size, block)]
    lock = Lock()
    progress = target.with_suffix(target.suffix + ".progress.json")
    import json
    manifest = {"sha256": checksum, "chunks": {}}
    if part.exists() and progress.exists():
        previous = json.loads(progress.read_text())
        if previous.get("sha256") == checksum and part.stat().st_size == metadata.size:
            manifest = previous
    sessions = local()
    resolved_urls = {}
    with part.open("r+b" if part.exists() else "w+b") as stream:
        stream.truncate(metadata.size)
        completed = set()
        for start, end in ranges:
            digest = manifest["chunks"].get(str(start))
            if digest:
                stream.seek(start)
                if hashlib.sha256(stream.read(end-start+1)).hexdigest() == digest:
                    completed.add(start)
        manifest["chunks"] = {k: v for k, v in manifest["chunks"].items() if int(k) in completed}
        done = len(completed)
        print(f"{filename}: resuming {done}/{len(ranges)} verified chunks", flush=True)
        def fetch(bounds):
            nonlocal done
            start, end = bounds
            if not hasattr(sessions, "http"):
                sessions.http = requests.Session()
            for attempt in range(8):
                try:
                    url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{filename}?download=true&part={start}"
                    request_url = resolved_urls.get("cdn", url)
                    with sessions.http.get(request_url, headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 90), stream=True) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                            resolved_urls.pop("cdn", None)
                            raise ValueError("Server returned an invalid byte range")
                        if response.url != url:
                            resolved_urls["cdn"] = response.url
                        content = response.raw.read(end-start+2)
                        if len(content) != end-start+1:
                            raise ValueError("Server returned an invalid byte range length")
                    with lock:
                        stream.seek(start)
                        stream.write(content)
                        stream.flush()
                        manifest["chunks"][str(start)] = hashlib.sha256(content).hexdigest()
                        temporary = progress.with_suffix(".tmp")
                        temporary.write_text(json.dumps(manifest), encoding="utf-8")
                        temporary.replace(progress)
                        done += 1
                        if done % 10 == 0 or done == len(ranges):
                            print(f"{filename}: {done}/{len(ranges)} chunks", flush=True)
                    return
                except (requests.RequestException, ValueError):
                    resolved_urls.pop("cdn", None)
                    if attempt == 7:
                        raise
                    time.sleep(min(attempt + 1, 5))
        with ThreadPoolExecutor(max_workers=int(os.getenv("RAG_DOWNLOAD_WORKERS", "4"))) as pool:
            list(pool.map(fetch, [bounds for bounds in ranges if bounds[0] not in completed]))
    if not valid(part):
        raise ValueError("Downloaded model failed SHA-256 verification")
    part.replace(target)
    print(f"SHA-256 verified: {target}", flush=True)
    return target


if __name__ == "__main__":
    from scripts.domain_common.multimodal_rag import ROOT
    for name in ["Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"]:
        download("Qwen/Qwen3-VL-2B-Instruct-GGUF", name, ROOT / "indexes/models/Qwen3-VL-2B")
