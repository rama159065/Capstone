import json

from scripts.evaluation import vlm_page_probe as probe


def test_figures_match_without_separators_and_hits_match_their_page():
    assert probe.contains_all("Net income | $ 3,097 | 1,382", ["3097", "1,382"])
    assert probe.contains_all("Efficiency ratio 46 %", ["46%"]) and not probe.contains_all("4.6%", ["46%"])
    case = {"doc_id": "edgar_ars_C_2025", "page": 41}
    assert probe.on_page({"chunk_id": "annual_report_edgar_ars_C_2025_p0041_vlm0"}, case)
    assert not probe.on_page({"chunk_id": "annual_report_edgar_ars_C_2025_p0410_vlm0"}, case)


def test_cases_file_is_complete():
    cases = [json.loads(line) for line in probe.CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({c["id"] for c in cases}) == len(cases) >= 10
    for c in cases:
        assert c["kind"] in {"gemini", "qwen", "control", "regression"} and c["expected"] and c["query"]


def test_compare_marks_fixed_and_lost(tmp_path, capsys):
    row = lambda i, ok: {"id": i, "kind": "gemini", "value_found": ok, "reader": "paidVL" if ok else None}
    (tmp_path / "b.json").write_text(json.dumps({"rows": [row("a", False), row("b", True)]}))
    (tmp_path / "a.json").write_text(json.dumps({"rows": [row("a", True), row("b", False)]}))
    probe.compare(tmp_path / "b.json", tmp_path / "a.json")
    out = capsys.readouterr().out
    assert "FIXED" in out.splitlines()[1] and "LOST" in out.splitlines()[2]
