"""Lazy, CPU-capable learned models for the local multimodal pipeline."""
from functools import lru_cache
from pathlib import Path
import os
import base64
import json
import hashlib
import threading
import urllib.request

# One model load at a time per process. transformers builds a model on the "meta" device and patches torch globally
# while it does; two loads in parallel threads (the API's warm-up and a first request) can leave that patch in place,
# and every later load in the process then fails with "Cannot copy out of meta tensor".
LOAD_LOCK = threading.RLock()


@lru_cache(maxsize=2)
def text_model(name):
    import torch
    from sentence_transformers import SentenceTransformer
    with LOAD_LOCK:
        torch.set_num_threads(int(os.getenv("RAG_CPU_THREADS", "4")))
        return SentenceTransformer(name, device=os.getenv("RAG_DEVICE", "cpu"))


@lru_cache(maxsize=1)
def reranker(name):
    import torch
    from sentence_transformers import CrossEncoder
    with LOAD_LOCK:
        torch.set_num_threads(int(os.getenv("RAG_CPU_THREADS", "4")))
        return CrossEncoder(name, device=os.getenv("RAG_DEVICE", "cpu"))


@lru_cache(maxsize=1)
def vision_model(name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    with LOAD_LOCK:
        torch.set_num_threads(int(os.getenv("RAG_CPU_THREADS", "4")))
        processor = AutoProcessor.from_pretrained(name)
        model = AutoModelForImageTextToText.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True)
        model.to(os.getenv("RAG_DEVICE", "cpu")).eval()
    return processor, model


def generate(prompt, images=(), model=None, max_new_tokens=768):
    import torch
    from PIL import Image
    name = model or os.getenv("RAG_TRANSFORMERS_MODEL")
    if not name:
        raise ValueError("Configure RAG_TRANSFORMERS_MODEL with a vision model path or repository")
    processor, network = vision_model(name)
    pictures = []
    for path in images:
        with Image.open(path) as picture:
            pictures.append(picture.convert("RGB"))
    content = [{"type": "image"} for _ in pictures] + [{"type": "text", "text": prompt}]
    chat = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inputs = processor(text=chat, images=pictures or None, return_tensors="pt").to(network.device)
    with torch.inference_mode():
        output = network.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  repetition_penalty=1.1)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def llamacpp(prompt, images=()):
    content = [{"type": "text", "text": prompt}]
    for path in images:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
    body = {"model": "local-qwen3-vl", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 768, "stream": False}
    endpoint = os.getenv("RAG_VISION_API_BASE", "http://127.0.0.1:8081/v1").rstrip("/")
    request = urllib.request.Request(endpoint + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    request.add_header("Idempotency-Key", hashlib.sha256(request.data).hexdigest())
    with urllib.request.urlopen(request, timeout=float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45"))) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()
