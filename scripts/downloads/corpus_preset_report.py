"""HTML list of the filings a corpus preset downloads, sector by sector.

    python -m scripts.downloads.corpus_preset_report --preset min500 --out docs/corpus_min500.html
    python -m scripts.downloads.corpus_preset_report --preset full --out docs/corpus_full.html

Each filing shows ticker, company, fiscal year, filing date, size, whether the frozen eval set
uses it, and why the preset includes it. The page has a search box (no external scripts).
"""
from __future__ import annotations

import argparse
import html
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from scripts.common.sector_corpus import read_jsonl, sector_slug
from scripts.downloads import download_annual_report_corpus as dl

CSS = """:root{--bg:#fff;--fg:#1f2933;--muted:#52606d;--line:#cbd5e1;--head:#eff6ff;--zebra:#f8fafc;--accent:#2563eb;--h2:#1e3a8a;--tag:#dcfce7;--tagfg:#166534;--tag2:#fef3c7;--tag2fg:#92400e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f172a;--fg:#e2e8f0;--muted:#94a3b8;--line:#334155;--head:#1e293b;--zebra:#111c33;--accent:#60a5fa;--h2:#93c5fd;--tag:#14532d;--tagfg:#bbf7d0;--tag2:#78350f;--tag2fg:#fde68a}}
body{font-family:Segoe UI,Arial,sans-serif;max-width:1150px;margin:2rem auto;padding:0 16px;color:var(--fg);background:var(--bg);line-height:1.5}
h1{border-bottom:3px solid var(--accent);padding-bottom:.3rem}h2{margin-top:2rem;color:var(--h2);border-bottom:1px solid var(--line)}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.88rem}
th,td{border:1px solid var(--line);padding:.35rem .55rem;text-align:left;vertical-align:top}th{background:var(--head);position:sticky;top:0}
tr:nth-child(even) td{background:var(--zebra)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:0 .4rem;border-radius:4px;font-size:.78rem;background:var(--tag);color:var(--tagfg)}
.tag.add{background:var(--tag2);color:var(--tag2fg)}.muted{color:var(--muted)}a{color:var(--accent)}
input{font:inherit;padding:.4rem .6rem;width:100%;max-width:420px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}"""

SCRIPT = """<script>
const box=document.getElementById('q');
box.addEventListener('input',()=>{const q=box.value.trim().toLowerCase();
document.querySelectorAll('tbody.filings tr').forEach(r=>{r.style.display=!q||r.textContent.toLowerCase().includes(q)?'':'none';});
document.querySelectorAll('section.sector').forEach(s=>{const shown=[...s.querySelectorAll('tbody.filings tr')].some(r=>r.style.display!=='none');s.style.display=shown?'':'none';});});
</script>"""


def eval_filings() -> set[str]:
    text = dl.EVAL_SET.read_text(encoding="utf-8") if dl.EVAL_SET.exists() else ""
    return set(re.findall(r"edgar_ars_[A-Za-z.-]+_\d{4}", text))


def reasons(plan: list[dict], preset: str) -> dict[str, str]:
    """document_id -> why the preset includes it."""
    if preset == "full":
        return {dl.document_id(r): "planned" for r in plan}
    per_sector, top = Counter(), set()
    for row in sorted(plan, key=lambda r: (sector_slug(r["sector"]), r.get("rank", 0), r["ticker"], -int(r["year"]))):
        slug = sector_slug(row["sector"])
        if per_sector[slug] < dl.MIN500_PER_SECTOR:
            per_sector[slug] += 1
            top.add(dl.document_id(row))
    return {dl.document_id(r): ("top 41 in sector" if dl.document_id(r) in top else "eval-set company")
            for r in dl.preset_rows(plan, preset)}


def render(preset: str, plan_path: Path) -> str:
    plan = read_jsonl(plan_path)
    rows = dl.preset_rows(plan, preset)
    why, used = reasons(plan, preset), eval_filings()
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sector[row["sector"]].append(row)
    total_bytes = sum(r.get("bytes") or 0 for r in rows)
    e = html.escape

    summary = []
    for sector in sorted(by_sector):
        items = by_sector[sector]
        years = Counter(int(r["year"]) for r in items)
        summary.append(
            f"<tr><td><a href='#{sector_slug(sector)}'>{e(sector)}</a></td><td class='num'>{len(items)}</td>"
            f"<td class='num'>{len({r['ticker'] for r in items})}</td>"
            f"<td class='num'>{sum(r.get('bytes') or 0 for r in items) / 1e6:,.0f}</td>"
            f"<td class='num'>{sum(1 for r in items if dl.document_id(r) in used)}</td>"
            f"<td>{', '.join(f'{y}: {n}' for y, n in sorted(years.items()))}</td></tr>")

    sections = []
    for sector in sorted(by_sector):
        items = sorted(by_sector[sector], key=lambda r: (r.get("rank", 0), r["ticker"], -int(r["year"])))
        body = []
        for i, r in enumerate(items, 1):
            doc = dl.document_id(r)
            reason = why.get(doc, "")
            tag = "tag add" if reason == "eval-set company" else "tag"
            body.append(
                f"<tr><td class='num'>{i}</td><td><strong>{e(r['ticker'])}</strong></td><td>{e(r['company'])}</td>"
                f"<td class='num'>{r['year']}</td><td>{e(str(r.get('filed') or ''))}</td>"
                f"<td class='num'>{(r.get('bytes') or 0) / 1e6:,.1f}</td>"
                f"<td>{'✓' if doc in used else ''}</td><td><span class='{tag}'>{e(reason)}</span></td>"
                f"<td><a href='{e(r['url'])}' rel='noopener'>EDGAR</a></td></tr>")
        sections.append(
            f"<section class='sector' id='{sector_slug(sector)}'><h2>{e(sector)} "
            f"<span class='muted'>({len(items)} filings, {len({r['ticker'] for r in items})} companies)</span></h2>"
            "<div class='wrap'><table><thead><tr><th class='num'>#</th><th>Ticker</th><th>Company</th>"
            "<th class='num'>Fiscal year</th><th>Filed</th><th class='num'>MB</th><th>In eval set</th>"
            "<th>Why included</th><th>Source</th></tr></thead><tbody class='filings'>"
            + "".join(body) + "</tbody></table></div></section>")

    title = {"min500": "min500 corpus: filings to download", "full": "Full corpus: filings to download"}[preset]
    note = ("The 41 largest filings in each sector (companies by size, newest year first), plus every filing of the "
            "companies the frozen eval set asks about, so no eval question loses its source."
            if preset == "min500" else "Every filing in the corpus plan.")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><style>{CSS}</style></head><body>
<h1>{e(title)}</h1>
<p><strong>{len(rows)} Form ARS PDFs</strong> from <strong>{len({r['ticker'] for r in rows})} companies</strong> in
<strong>{len(by_sector)} sectors</strong>, {total_bytes / 1e9:.2f} GB, from SEC EDGAR. {e(note)}</p>
<p class="muted">Preset <code>{e(preset)}</code> of <code>{e(str(plan_path.name))}</code> ({len(plan)} planned filings).
Generated {date.today().isoformat()} by <code>python -m scripts.downloads.corpus_preset_report --preset {e(preset)}</code>.
Download with <code>CORPUS = "{e(preset)}"</code> in <code>colab_rag_pipelines.ipynb</code>, or
<code>python -m scripts.downloads.download_annual_report_corpus --preset {e(preset)}</code>.
✓ = the frozen evaluation set (eval_set_v1) cites this filing.</p>
<h2>Summary by sector</h2>
<div class="wrap"><table><thead><tr><th>Sector</th><th class="num">Filings</th><th class="num">Companies</th>
<th class="num">MB</th><th class="num">Eval-set filings</th><th>Filings per fiscal year</th></tr></thead><tbody>
{''.join(summary)}
<tr><td><strong>Total</strong></td><td class="num"><strong>{len(rows)}</strong></td>
<td class="num"><strong>{len({r['ticker'] for r in rows})}</strong></td>
<td class="num"><strong>{total_bytes / 1e6:,.0f}</strong></td>
<td class="num"><strong>{sum(1 for r in rows if dl.document_id(r) in used)}</strong></td><td></td></tr>
</tbody></table></div>
<p><label for="q">Search filings (ticker, company, year): </label><br><input id="q" type="search" placeholder="e.g. JPM, Costco, 2024"></p>
{''.join(sections)}
{SCRIPT}
</body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=dl.PRESETS, default="min500")
    parser.add_argument("--plan", type=Path, default=dl.DEFAULT_DIR / dl.PLAN_NAME)
    parser.add_argument("--out", type=Path, default=None, help="default: docs/corpus_<preset>.html")
    args = parser.parse_args(argv)
    out = args.out or dl.ROOT / "docs" / f"corpus_{args.preset}.html"
    out.write_text(render(args.preset, args.plan), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
