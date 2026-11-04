"""Official Transformers ColPali image/query multivectors, never pooled vectors."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


class ColPaliEncoder:
    dimensions = 128

    def __init__(self, model, device=None, quantization=None):
        self.model_path = str(Path(model).resolve()) if Path(model).exists() else model
        self.device = device or os.getenv("RAG_COLPALI_DEVICE", "cpu")
        self.quantization = quantization or os.getenv("RAG_COLPALI_QUANTIZATION", "4bit")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("RAG_COLPALI_QUANTIZATION must be none, 4bit, or 8bit")
        configuration = {"model": self.model_path, "dimensions": 128,
                         "quantization": self.quantization, "query_prefix": "Query: ", "version": 1}
        self.signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()

    def _encode(self, *, image=None, query=None):
        import torch
        from PIL import Image
        processor, model = load_colpali(self.model_path, self.device, self.quantization)
        if image is not None:
            with Image.open(image) as source:
                inputs = processor(images=[source.convert("RGB")], return_tensors="pt")
        else:
            inputs = processor(text=[query], return_tensors="pt")
        inputs = inputs.to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            vectors = model(**inputs, use_cache=False).embeddings[0].float()
        mask = inputs.get("attention_mask")
        if mask is not None:
            vectors = vectors[mask[0].bool()]
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions or not torch.isfinite(vectors).all():
            raise ValueError("ColPali returned invalid multivectors")
        return vectors.cpu().tolist()

    def embed_image(self, image):
        return self._encode(image=image)

    def embed_query(self, query):
        return self._encode(query=query)


@lru_cache(maxsize=1)
def load_colpali(model_path, device, quantization):
    import torch
    from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor
    torch.set_num_threads(int(os.getenv("RAG_CPU_THREADS", "4")))
    kwargs = {"dtype": torch.bfloat16, "device_map": device,
              "attn_implementation": "sdpa", "low_cpu_mem_usage": True}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["embedding_proj_layer", "lm_head"])
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
            llm_int8_skip_modules=["embedding_proj_layer", "lm_head"])
    processor = ColPaliProcessor.from_pretrained(model_path)
    # This checkpoint's trained query prefix differs from the processor default.
    processor.query_prefix = "Query: "
    model = ColPaliForRetrieval.from_pretrained(model_path, **kwargs).eval()
    if model.config.embedding_dim != 128:
        raise ValueError("Expected the official 128-dimensional ColPali projection")
    return processor, model
