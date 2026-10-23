import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.domain_common.rag_agent import agent_cli  # noqa: E402
from scripts.incident_postmortem.domain_config import DOMAIN_CONFIG  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(agent_cli(DOMAIN_CONFIG))
