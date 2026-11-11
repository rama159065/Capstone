"""Backfill actual ColPali multivectors into the existing local Qdrant database."""
import argparse
import json
import os

from scripts.rag.multimodal_rag import DOMAINS, LocalRAG, ROOT


def configure_colpali():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("RAG_LOCAL_DIR", str(ROOT / "indexes/multimodal_full"))
    os.environ.setdefault("RAG_EMBEDDING_MODEL", str(ROOT / "indexes/models/all-MiniLM-L6-v2"))
    os.environ.setdefault("RAG_RERANKER_MODEL", str(ROOT / "indexes/models/ms-marco-MiniLM-L-6-v2"))
    os.environ.setdefault("RAG_COLPALI_MODEL", str(ROOT / "indexes/models/colpali-v1.3-hf"))
    os.environ.setdefault("RAG_COLPALI_QUANTIZATION", "4bit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["backfill", "query", "status"])
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--query", default="Which quarter has the highest revenue in the chart?")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_colpali()
    rag = LocalRAG()
    try:
        if args.command == "backfill":
            result = rag.backfill_visual(args.domain, force=args.force)
        elif args.command == "query":
            result = rag.retrieve_visual(args.domain or "financial_report_qa", args.query)
        else:
            result = {"storage": str(rag.root), "colpali_signature": rag.colpali.signature,
                      "collections": {name: rag.client.count(name, exact=True).count
                                      for name in ["docs_text", "visual_pages"]},
                      "colpali_dimensions": 128, "comparator": "max_sim"}
        print(json.dumps(result, indent=2))
    finally:
        rag.close()


if __name__ == "__main__":
    main()
