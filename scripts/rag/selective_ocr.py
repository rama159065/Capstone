"""Selective CPU OCR with source coordinates; native PDF text stays authoritative."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import math
import os
import re

import pymupdf as fitz


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


@lru_cache(maxsize=1)
def engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Selective OCR requires rapidocr and onnxruntime; install project dependencies or set RAG_OCR_ENABLED=0") from exc
    return RapidOCR(params={"Global.log_level": "warning", "Global.text_score": 0.0,
        "EngineConfig.onnxruntime.intra_op_num_threads": int(os.getenv("RAG_OCR_THREADS", "2")),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1})


@dataclass
class OCRResult:
    mode: str = "none"
    text: str = ""
    blocks: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    rejected_lines: int = 0
    engine: str = "rapidocr"

    def payload(self):
        return {"ocr_applied": self.mode != "none", "ocr_mode": self.mode,
                "ocr_text": self.text, "ocr_blocks": self.blocks, "ocr_regions": self.regions,
                "ocr_engine": self.engine, "ocr_rejected_lines": self.rejected_lines,
                "ocr_coordinate_space": "unrotated_pdf_points"}


class SelectiveOCR:
    def __init__(self, recognizer=None):
        self.recognizer = recognizer
        self.enabled = os.getenv("RAG_OCR_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.scale = float(os.getenv("RAG_OCR_SCALE", "2"))
        self.confidence = float(os.getenv("RAG_OCR_MIN_CONFIDENCE", "0.6"))
        if not 0 <= self.confidence <= 1 or not .5 <= self.scale <= 4:
            raise ValueError("OCR confidence must be 0..1 and render scale must be 0.5..4")
        try:
            package = version("rapidocr")
        except PackageNotFoundError:
            package = "not-installed"
        settings = {"version": 1, "enabled": self.enabled, "engine": "rapidocr", "package": package,
                    "scale": self.scale, "min_confidence": self.confidence}
        self.signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def regions_for(page):
        # Image information uses unrotated PDF coordinates, like get_text('words').
        bounds = fitz.Rect(0, 0, page.cropbox.width, page.cropbox.height)
        words = page.get_text("words")
        regions = []
        for image in page.get_image_info():
            rect = fitz.Rect(image["bbox"]) & bounds
            if rect.is_empty or rect.get_area() < bounds.get_area() * .005 or image["width"] < 80 or image["height"] < 30:
                continue
            inside = [w for w in words if rect.contains(fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2))]
            native = " ".join(w[4] for w in inside)
            covered = sum((fitz.Rect(w[:4]) & rect).get_area() for w in inside) / rect.get_area()
            if len(normalized(native)) >= 40 and len(inside) >= 12 and covered >= .03 and "\ufffd" not in native:
                continue  # Image already has a usable native/OCR text layer.
            regions.append(rect)
        # Prefer the largest crop when one image is contained in another.
        unique = []
        for rect in sorted(regions, key=lambda r: r.get_area(), reverse=True):
            if not any((rect & prior).get_area() / rect.get_area() >= .9 for prior in unique):
                unique.append(rect)
        native_text = page.get_text(sort=True)
        sparse_native = len(normalized(native_text)) < 40 or "\ufffd" in native_text
        large_scan = any(r.get_area() / bounds.get_area() >= .65 for r in unique)
        outlined_text = not normalized(native_text) and not page.get_image_info() and len(page.get_drawings()) > 4
        if (large_scan and sparse_native) or outlined_text or (page.rotation and unique):
            return "full_page", [page.rect]
        return ("image_regions", unique) if unique else ("none", [])

    def extract(self, page):
        if not self.enabled:
            return OCRResult(engine="disabled")
        mode, regions = self.regions_for(page)
        result = OCRResult(mode=mode)
        if not regions:
            return result
        from PIL import Image
        native = " " + normalized(page.get_text(sort=True)) + " "
        texts = []
        for rect in regions:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(self.scale, self.scale), clip=rect, alpha=False)
            picture = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            output = (self.recognizer or engine())(picture)
            result.regions.append(list(rect))
            if output.txts is None:
                continue
            for box, text, score in zip(output.boxes, output.txts, output.scores):
                text, score = str(text).strip(), float(score)
                if not text or not math.isfinite(score) or not self.confidence <= score <= 1:
                    result.rejected_lines += 1
                    continue
                points = []
                for x, y in box:
                    point = fitz.Point(rect.x0 + float(x)/self.scale, rect.y0 + float(y)/self.scale)
                    if page.rotation:
                        point = point * page.derotation_matrix
                    points.append([round(point.x, 3), round(point.y, 3)])
                new_text = bool(normalized(text)) and (" " + normalized(text) + " ") not in native
                result.blocks.append({"text": text, "confidence": score, "polygon": points,
                                      "included_in_text": new_text})
                if new_text:
                    texts.append(text)
        result.text = "\n".join(texts)
        return result


def save_artifact(path, base_text, result, signature):
    path.write_text(json.dumps({"signature": signature, "base_text": base_text,
                               "ocr": asdict(result)}, indent=2), encoding="utf-8")
