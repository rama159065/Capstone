"""Write the runtime requirements for the Docker image from pyproject.toml.

torch and torchvision are installed first from the CPU-only wheel index (the PyPI Linux
wheels pull in about 3 GB of CUDA libraries the server never uses), and JupyterLab is a
notebook tool the server doesn't need. Everything else is installed as pinned in
pyproject.toml.

    python deploy/requirements_from_pyproject.py pyproject.toml > requirements-docker.txt
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

SKIP = {"torch", "torchvision", "jupyterlab"}


def requirements(pyproject: Path) -> list[str]:
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    return [dep for dep in deps if re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0].strip().lower() not in SKIP]


if __name__ == "__main__":
    print("\n".join(requirements(Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml"))))
