"""Sector-partitioned annual-report corpus: folder layout and manifest reading.

The corpus written by ``scripts/downloads/download_annual_report_corpus.py``::

    <corpus>/
        corpus_plan.jsonl          what to fetch (sector, ticker, year, EDGAR URL)
        pdf_manifest.jsonl         every downloaded PDF, all sectors
        download_summary.json      counts and failures of the last run
        banking/
            pdf_manifest.jsonl     this sector's PDFs only
            EDGAR_ARS_JPM_2025.pdf
        oil_and_gas/
            ...

A manifest's ``local_path`` is relative to the folder of the manifest that lists
it (``banking/EDGAR_ARS_JPM_2025.pdf`` in the combined manifest,
``EDGAR_ARS_JPM_2025.pdf`` in the sector one), so the corpus can be moved to
Google Drive or another machine as a whole. Older manifests that hold
repo-relative paths (``data\\raw\\...``) still resolve.

Standard library (and scripts.common.companies) only, so the downloader also runs on a bare Colab runtime.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.common.companies import sector_of

MANIFEST_NAME = "pdf_manifest.jsonl"
PLAN_NAME = "corpus_plan.jsonl"


def sector_slug(sector: str | None) -> str:
    """Folder and filter key for a sector: "Oil and gas" -> "oil_and_gas"."""
    slug = re.sub(r"[^a-z0-9]+", "_", (sector or "").strip().lower()).strip("_")
    return slug or "unknown"


def parse_sectors(value: str | list[str] | None) -> set[str] | None:
    """Comma-separated sector names or slugs -> set of slugs (None = all sectors)."""
    if not value:
        return None
    items = value.split(",") if isinstance(value, str) else value
    return {sector_slug(item) for item in items if item.strip()} or None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_files(path: Path, sectors: set[str] | None = None) -> list[Path]:
    """The manifests to read for `path`: the file itself, or a corpus folder's sector manifests."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: no such manifest or corpus folder")
    found = sorted(path.glob(f"*/{MANIFEST_NAME}"))
    if found:                                             # sector corpus: an absent sector reads as empty
        return [p for p in found if not sectors or p.parent.name in sectors]
    combined = path / MANIFEST_NAME                       # flat corpus folder (e.g. edgar/)
    if combined.exists():
        return [combined]
    raise FileNotFoundError(f"{path}: no {MANIFEST_NAME} in it or in its sector folders")


def read_manifest(path: Path, sectors: str | set[str] | None = None) -> list[dict]:
    """Rows of a manifest file or of a sector corpus folder, optionally only some sectors.

    Each row gets ``sector`` / ``sector_slug`` (from the ticker, scripts/common/companies.py, when the
    manifest has none, e.g. the older EDGAR and AnnualReports.com manifests; else ``unknown``) and
    ``_manifest_dir``, the folder its ``local_path`` is relative to.
    """
    wanted = parse_sectors(sectors) if not isinstance(sectors, set) else sectors
    rows = []
    for manifest in manifest_files(path, wanted):
        for row in read_jsonl(manifest):
            row["sector"] = row.get("sector") or sector_of(row.get("ticker")) or "unknown"
            row["sector_slug"] = row.get("sector_slug") or sector_slug(row["sector"])
            if wanted and row["sector_slug"] not in wanted:
                continue
            row["_manifest_dir"] = str(manifest.parent)
            rows.append(row)
    return rows


def resolve_local_path(row: dict, root: Path | None = None) -> Path:
    """Where the row's PDF is: next to its manifest, else (older manifests) relative to the repo root or cwd."""
    rel = Path(str(row["local_path"]).replace("\\", "/"))
    if rel.is_absolute():
        return rel
    candidates = []
    if row.get("_manifest_dir"):
        candidates.append(Path(row["_manifest_dir"]) / rel)
    if root is not None:
        candidates.append(Path(root) / rel)
    candidates.append(rel)
    return next((c for c in candidates if c.exists()), candidates[0])
