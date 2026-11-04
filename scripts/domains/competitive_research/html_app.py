import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.rag_agent import streamlit_app  # noqa: E402
from scripts.domains.competitive_research.domain_config import DOMAIN_CONFIG  # noqa: E402

if __name__ == "__main__":
    streamlit_app(DOMAIN_CONFIG)
