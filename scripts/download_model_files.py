"""Verified, ranged HTTPS downloader for large local model files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Lock
import time

import requests
from huggingface_hub import HfApi


def download(repo, filename, folder):
    info = HfApi().model_info(repo, files_metadata=True)
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
    with part.open("w+b") as stream:
        stream.truncate(metadata.size)
        done = 0
        def fetch(bounds):
            nonlocal done
            start, end = bounds
            for attempt in range(4):
                try:
                    url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{filename}?download=true&part={start}"
                    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90))
                    response.raise_for_status()
                    if response.status_code != 206 or len(response.content) != end-start+1:
                        raise ValueError("Server returned an invalid byte range")
                    with lock:
                        stream.seek(start)
                        stream.write(response.content)
                        done += 1
                        if done % 10 == 0 or done == len(ranges):
                            print(f"{filename}: {done}/{len(ranges)} chunks", flush=True)
                    return
                except (requests.RequestException, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, ranges))
    if not valid(part):
        raise ValueError("Downloaded model failed SHA-256 verification")
    part.replace(target)
    print(f"SHA-256 verified: {target}", flush=True)
    return target


if __name__ == "__main__":
    from scripts.domain_common.multimodal_rag import ROOT
    for name in ["Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"]:
        download("Qwen/Qwen3-VL-2B-Instruct-GGUF", name, ROOT / "indexes/models/Qwen3-VL-2B")
