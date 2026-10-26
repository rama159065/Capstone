"""Add selective OCR text to the existing learned-embedding Qdrant index."""
import argparse
import json
import os

from scripts.domain_common.multimodal_rag import DOMAINS, LocalRAG, ROOT


def configure_ocr():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("RAG_LOCAL_DIR", str(ROOT / "indexes/multimodal_full"))
    os.environ.setdefault("RAG_EMBEDDING_MODEL", str(ROOT / "indexes/models/all-MiniLM-L6-v2"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_ocr()
    rag = LocalRAG()
    try:
        print(json.dumps(rag.backfill_ocr(args.domain, force=args.force), indent=2))
    finally:
        rag.close()


if __name__ == "__main__":
    main()
