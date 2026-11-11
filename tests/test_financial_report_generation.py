from decimal import Decimal
import json
from pathlib import Path

import pymupdf
import pytest

from scripts.reports.generate_financial_report import ROOT, analyze, generate, parse_row, retrieve_financials
from scripts.common.resilience import Calls
from scripts.observability.trace_log import TraceLog, current_trace_id, emit
from scripts.observability.monitor_report_trace import summarize

PROFILE = ROOT / "configs/financial_reports/apple_2024.json"


def test_financial_row_parser_rejects_missing_values():
    assert parse_row("Net income\n$\n93,736\n96,995\n99,803", "Net income", 3)[0] == [Decimal(93736), Decimal(96995), Decimal(99803)]
    assert parse_row("Outflow\n(9,447)\n(10,959)", "Outflow", 2)[0] == [Decimal(-9447), Decimal(-10959)]
    with pytest.raises(ValueError, match="Incomplete"):
        parse_row("Net income\n93,736\nOther row\n123\n456", "Net income", 3)


def test_actual_apple_statement_values_and_reconciliation():
    profile = json.loads(PROFILE.read_text())
    groups, _ = retrieve_financials(profile, ROOT / profile["source"])
    assert groups["income"]["revenue"]["values"] == list(map(Decimal, [391035, 383285, 394328]))
    assert groups["income"]["net_income"]["pdf_page"] == 32
    analysis = analyze(groups)
    assert len(analysis["checks"]) == 17
    assert analysis["ratios"][2024]["Free cash flow"] == 108807
    assert analysis["ratios"][2024]["Debt carrying value"] == 106629
    groups["income"]["revenue"]["values"][0] += 1
    with pytest.raises(ValueError, match="reconciliation"):
        analyze(groups)


def test_root_trace_propagates_to_nested_worker_calls(tmp_path):
    trace = TraceLog(tmp_path)
    with trace.activate():
        with trace.span("report_agent"):
            calls = Calls(2)
            def node():
                assert current_trace_id() == trace.trace_id
                nested = Calls(1)
                assert nested.trace_id == trace.trace_id
                return nested.run("llm", lambda: emit("llm.test", provider="test"))
            calls.run("retrieval", node)
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert {r["trace_id"] for r in rows} == {trace.trace_id}
    assert {r["thread_id"] for r in rows} == {trace.thread_id}
    assert [r["sequence"] for r in rows] == list(range(1, len(rows) + 1))
    assert len({r["worker_thread_id"] for r in rows}) > 1
    assert any(r["event"] == "llm.test" for r in rows)


def test_corrupted_source_is_logged_and_no_report_is_written(tmp_path):
    profile = json.loads(PROFILE.read_text())
    profile["source_sha256"] = "0" * 64
    path = tmp_path / "bad_profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="checksum"):
        generate(path, output_root=tmp_path / "output")
    manifests = list((tmp_path / "output").glob("**/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    events = [json.loads(line)["event"] for line in Path(manifest["log_path"]).read_text().splitlines()]
    assert "span.failed" in events and "report.failed" in events
    assert not list((tmp_path / "output").glob("**/*.pdf"))


def test_complete_report_artifacts_and_trace_monitor(tmp_path):
    manifest_path = generate(PROFILE, output_root=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "generated_pending_visual_review"
    assert manifest["checks_passed"] == 17
    pdf = next(manifest_path.parent.glob("*.pdf"))
    with pymupdf.open(pdf) as document:
        assert len(document) == 8
        text = "\n".join(p.get_text() for p in document)
        assert "391,035" in text and "93,736" in text and "108,807" in text
        assert manifest["trace_id"] in text and manifest["thread_id"] in text
    assert len(list((manifest_path.parent / "charts").glob("*.png"))) == 5
    monitor = summarize(tmp_path, trace_id=manifest["trace_id"])
    assert monitor["count"] == 1
    assert len(monitor["runs"][0]["agents"]) == 7
    assert monitor["runs"][0]["failures"] == []
    assert monitor["runs"][0]["event_counts"]["llm.skipped"] == 1


def test_rag_model_and_result_logs_share_report_trace(tmp_path, monkeypatch):
    from scripts.agents.safe_answer import answer
    from scripts.rag import multimodal_rag as mm
    class RAG:
        root = tmp_path
        pipeline = "test"
        reranker_model = None
        def retrieve(self, *args, **kwargs):
            return [{"chunk_id": "source-1", "doc_id": "apple", "title": "Annual report", "page": 32,
                     "text": "Revenue was 391035 million.", "page_context": "Revenue was 391035 million."}]
    monkeypatch.setattr(mm, "ollama", lambda *a: json.dumps({
        "selections": [{"source_id": "source-1", "quote": "Revenue was 391035 million."}], "unanswered": []}))
    trace = TraceLog(tmp_path / "logs")
    with trace.activate():
        result = answer(RAG(), "financial_report_qa", "What was revenue?", provider="ollama", ticker="AAPL", year=2024)
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert result["trace_id"] == trace.trace_id
    assert {row["trace_id"] for row in rows} == {trace.trace_id}
    assert {"retrieval.result", "llm.request", "llm.validated", "answer.completed"} <= {row["event"] for row in rows}
    assert len([row for row in rows if row["event"] == "call.completed"]) == 2


def test_bad_optional_supplement_preserves_numeric_report(tmp_path):
    class BadRAG:
        def answer(self, *a, **kw):
            return None
    path = generate(PROFILE, output_root=tmp_path, provider="ollama", rag=BadRAG())
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "generated_pending_visual_review"
    assert manifest["checks_passed"] == 17
    supplement = json.loads((path.parent / "rag_supplement.json").read_text())
    assert supplement["final_status"] == "tool_error"
    assert next(path.parent.glob("*.pdf")).is_file()
