"""Prepare, run and validate the complete small-corpus local multimodal profile.

Examples: python -m scripts.run_local_multimodal prepare
          python -m scripts.run_local_multimodal evaluate
          python -m scripts.run_local_multimodal serve
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import zipfile

from scripts.domain_common.multimodal_rag import DOMAINS, ROOT

MODELS = ROOT / "indexes/models"
RUNTIME = ROOT / "indexes/runtime/llama.cpp"
LLAMA_RELEASE = "b11900"


def configure():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("RAG_LOCAL_DIR", str(ROOT / "indexes/multimodal_full"))
    os.environ.setdefault("RAG_EMBEDDING_MODEL", str(MODELS / "all-MiniLM-L6-v2"))
    os.environ.setdefault("RAG_RERANKER_MODEL", str(MODELS / "ms-marco-MiniLM-L-6-v2"))
    os.environ.setdefault("RAG_VISION_API_BASE", "http://127.0.0.1:8081/v1")
    os.environ.setdefault("RAG_VISION_MODEL_ID", "Qwen3-VL-2B-Instruct-Q4_K_M+Q8-projector")
    os.environ.setdefault("RAG_BACKEND", "qdrant")


def prepare():
    from huggingface_hub import snapshot_download
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    specs = [
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", ["*.json", "*.safetensors", "*.txt", "1_Pooling/*"]),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2", ["*.json", "*.safetensors", "*.txt"]),
    ]
    from scripts.download_model_files import download
    for filename in ["Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"]:
        download("Qwen/Qwen3-VL-2B-Instruct-GGUF", filename, MODELS / "Qwen3-VL-2B")
    for repo, folder, patterns in specs:
        snapshot_download(repo, local_dir=MODELS / folder, allow_patterns=patterns, ignore_patterns=["onnx/*", "openvino/*"])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / f"llama-{LLAMA_RELEASE}-bin-win-cpu-x64.zip"
    if not (RUNTIME / "llama-server.exe").exists():
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{archive.name}"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as package:
            package.extractall(RUNTIME)
    print("Models and runtime ready under indexes/.")


@contextmanager
def vision_server():
    model = MODELS / "Qwen3-VL-2B/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    projector = MODELS / "Qwen3-VL-2B/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    executable = RUNTIME / "llama-server.exe"
    if not all(p.exists() for p in [model, projector, executable]):
        raise FileNotFoundError("Run python -m scripts.run_local_multimodal prepare first")
    base = os.environ["RAG_VISION_API_BASE"].removesuffix("/v1")
    health = base + "/health"
    # This launcher owns its child service; fail clearly if the chosen port is occupied.
    from urllib.parse import urlparse
    import socket
    url = urlparse(base)
    if url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Managed vision service must bind to localhost")
    port = url.port or 8081
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop that service or choose another RAG_VISION_API_BASE.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(executable), "-m", str(model), "--mmproj", str(projector),
                                    "--host", "127.0.0.1", "--port", str(port), "-c", "8192",
                                    "-t", os.getenv("RAG_CPU_THREADS", "4"), "-np", "1",
                                    "--alias", "local-qwen3-vl", "--no-warmup"],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Vision server exited; inspect indexes/runtime/llama.cpp/server.log")
                try:
                    with urllib.request.urlopen(health, timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(.5)
            else:
                raise TimeoutError("Vision server did not become ready within 180 seconds")
            print("Local Qwen3-VL vision service ready.", flush=True)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "ingest", "evaluate", "serve", "query"])
    parser.add_argument("--input", type=Path, default=ROOT / "data/samples/multimodal")
    parser.add_argument("--domain", choices=DOMAINS, default="financial_report_qa")
    parser.add_argument("--query", default="Which quarter has the highest revenue in the chart?")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    configure()
    with vision_server():
        if args.command == "serve":
            import uvicorn
            from scripts.domain_common import multimodal_rag as mm
            try:
                uvicorn.run("scripts.api_app:app", host="127.0.0.1", port=8000, workers=1)
            finally:
                if mm._instance is not None:
                    mm._instance.close()
                    mm._instance = None
        elif args.command == "evaluate":
            from scripts.evaluate_multimodal_local import main as evaluate
            evaluate()
        else:
            from scripts.domain_common.multimodal_rag import LocalRAG
            rag = LocalRAG()
            try:
                if args.command == "ingest":
                    for domain in DOMAINS:
                        for path in sorted((args.input / domain).glob("*.pdf")):
                            print(json.dumps(rag.ingest(domain, path, force=args.force)), flush=True)
                else:
                    print(json.dumps(rag.answer(args.domain, args.query, top_k=1, provider="llamacpp"), indent=2))
            finally:
                rag.close()


if __name__ == "__main__":
    main()
