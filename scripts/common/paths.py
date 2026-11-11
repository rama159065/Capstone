"""Repository paths, independent of where a module sits under scripts/."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
