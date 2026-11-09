"""Zip the project's code for Colab, so notebooks run without cloning the (private) repo.

    python -m scripts.common.colab_bundle                 # -> output/colab/capstone_code.zip
    python -m scripts.common.colab_bundle --out D:/x.zip --list

Upload the zip to Google Drive as ``MyDrive/capstone_rag/capstone_code.zip``. The setup cell
of every notebook that imports ``scripts`` (02-05, colab_financial_ingest, colab_rag_pipelines)
unpacks it to ``/content/Capstone`` and puts that folder on ``sys.path``. A notebook only
unpacks again when the zip on Drive is a different build (``BUNDLE_INFO.json``), so
rerunning setup cells costs nothing.

What goes in: the files git tracks plus new files it doesn't ignore (so uncommitted work is
included), limited to code and the small data the code and tests read: ``scripts/``,
``tests/``, ``configs/``, the project files, Python helpers under ``notebooks/``, manifests
and plans (``data/raw/**/*.jsonl``), ``data/evaluation/``, ``data/samples/`` and the PDFs
the report profiles in ``configs/`` name. Corpus PDFs, indexes, models and outputs stay
out; the notebooks download those.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

from scripts.common.paths import ROOT

DEFAULT_OUT = ROOT / "output/colab/capstone_code.zip"
INFO_NAME = "BUNDLE_INFO.json"
TOP_FILES = {"pyproject.toml", "uv.lock", "README.md", "CLAUDE.md"}
FOLDERS = ("scripts/", "tests/", "configs/", "data/evaluation/", "data/samples/")


def included(path: str) -> bool:
    p = PurePosixPath(path)
    if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
        return False
    if path in TOP_FILES or path.startswith(FOLDERS):
        return True
    if path.startswith("notebooks/"):
        return p.suffix == ".py"                       # helpers notebooks import; the .ipynb files are opened separately
    return path.startswith("data/raw/") and p.suffix == ".jsonl"     # corpus plans and manifests, not the PDFs


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def candidate_files() -> list[str]:
    """Tracked files plus untracked files that .gitignore doesn't exclude (forward slashes)."""
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    return sorted({f for f in listed if f and (ROOT / f).is_file()})


def bundle_info(files: list[str]) -> dict:
    try:
        commit, branch = git("rev-parse", "--short", "HEAD").strip(), git("rev-parse", "--abbrev-ref", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        commit, branch, dirty = None, None, None
    return {"built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "commit": commit,
            "branch": branch, "uncommitted_changes": dirty, "files": len(files)}


def profile_sources() -> set[str]:
    """PDFs the report profiles read (configs/financial_reports/*.json "source"), e.g. the Apple 2024 report."""
    found = set()
    for profile in (ROOT / "configs").rglob("*.json"):
        try:
            source = json.loads(profile.read_text(encoding="utf-8")).get("source")
        except (ValueError, AttributeError):
            continue
        if isinstance(source, str) and (ROOT / source).is_file():
            found.add(source.replace("\\", "/"))
    return found


def build(out: Path = DEFAULT_OUT, files: list[str] | None = None) -> dict:
    extra = profile_sources() if files is None else set()
    files = sorted({f for f in (files if files is not None else candidate_files()) if included(f) or f in extra})
    missing = [f for f in ("scripts/rag/colab_pipeline.py", "pyproject.toml") if f not in files]
    if missing:
        raise SystemExit(f"bundle would miss {missing}; run from the repo")
    info = bundle_info(files)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in files:
            archive.write(ROOT / name, name)
        archive.writestr(INFO_NAME, json.dumps(info, indent=2))
    tmp.replace(out)
    return {**info, "zip": str(out), "mb": round(out.stat().st_size / 1e6, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list", action="store_true", help="print the files that go in")
    args = parser.parse_args(argv)
    result = build(args.out)
    if args.list:
        with zipfile.ZipFile(args.out) as archive:
            print("\n".join(archive.namelist()))
    print(json.dumps(result, indent=2))
    print("Upload it to Google Drive as MyDrive/capstone_rag/capstone_code.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
