"""Near-miss checks, fiscal-year names, derived figures and the LLM client's context limit. No network or models."""
from __future__ import annotations

import io
import urllib.error

import pytest

from scripts.agents import financial_checks as checks
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_prompting_agent as agent
from scripts.agents import financial_sector_trend as sector
from scripts.agents import financial_tools as tools
from scripts.common import companies, hosted_llm


def hit(chunk, text, ticker="X", year=2025, page=1, **extra):
    return {"chunk_id": chunk, "text": text, "ticker": ticker, "year": year, "page": page, **extra}


def problems(query, answer, cited, evidence=None, subject=None):
    return checks.problems(query, answer, cited, evidence or cited, subject, sector.parent_only)


# ------------------------------------------------------------------------------- what the question is about

@pytest.mark.parametrize("question, subject", [
    ("What was Citi's U.S. Personal Banking net income in 2025 compared with 2024?", "U.S. Personal Banking"),
    ("How did NVIDIA's data center revenue trend from 2023 to 2026?", "data center"),
    ("What was Amazon's AWS operating income in 2024?", "AWS"),
    ("What was Costco's membership fee revenue in 2025?", "membership fee"),
    ("What was Apple's total net revenue in fiscal 2024?", None),
    ("Compare Costco and Walmart revenue in 2025", None),
    ("JPMorgan or Bank of America, higher net income in FY2025?", None),
    ("How did healthcare revenue trend from 2022 to 2025?", None),
    ("How did banking sector net income trend from 2022 to 2025?", None),
    ("Who was Adobe's independent auditor for fiscal year 2025?", None),
])
def test_subject_is_what_narrows_the_measure(question, subject):
    assert fc.Clarifier(directory={}).subject_in(question) == subject


def test_a_segment_figure_must_come_from_a_page_naming_the_segment():
    services = hit("c_2024_p0030_vlm2", "SERVICES\nNet income $ 6,483 $ 4,635", page=30)
    uspb = hit("c_2025_p0041_vlm4", "Net income | 2025 | $ 3,097\nNet income | 2024 | $ 1,382", page=41)
    title = hit("c_2025_p0041_vlm0", "U.S. PERSONAL BANKING | In millions of dollars", page=41)
    question = "What was Citi's U.S. Personal Banking net income in 2025?"
    notes = problems(question, "It was 6,483 million [c_2024_p0030_vlm2].", [services], [services, uspb, title],
                     "U.S. Personal Banking")
    assert notes and "c_2025_p0041_vlm4" in notes[0]
    assert not problems(question, "It was 3,097 million [c_2025_p0041_vlm4].", [uspb], [services, uspb, title],
                        "U.S. Personal Banking")


# ------------------------------------------------------------------------------- region, product, axis, term, parent

CHEVRON = hit("cvx_p0041", "### Selected Operating Data\n| | 2022 | 2021 |\n| **U.S. Upstream** | | |\n"
              "| Net Crude Oil and Natural Gas Liquids Production (MBPD) | 888 | 858 |\n"
              "| Net Natural Gas Production (MMCFPD) | 1,758 | 1,689 |", page=41)
CHART = hit("cvx_p0035", "Thousands of barrels per day\n| Year | Value |\n| 22 | 1,440 |", page=35)


@pytest.mark.parametrize("question, answer, cited, wrong", [
    ("What was Chevron's net crude oil production in 2022?", "888 thousand barrels per day.", CHEVRON, True),
    ("What was Chevron's net crude oil production in 2022?", "1,440 thousand barrels per day.", CHART, False),
    ("What was Chevron's U.S. net natural gas production in 2022?", "1,758 MMCFPD.", CHEVRON, False),
    ("What net natural gas production did Chevron report?", "1,758 MMCFPD.", CHEVRON, True),
])
def test_a_region_or_product_the_question_does_not_ask_for(question, answer, cited, wrong):
    assert bool(problems(question, answer, [cited])) is wrong


def test_a_paragraph_continued_from_the_chunk_above_keeps_its_heading():
    prose = hit("cvx_p0037", "Net natural gas production averaged 1.76 billion cubic feet per day in 2022.", page=37)
    question = "What net natural gas production did Chevron report in its 2022 annual-report chart?"
    above = "Net oil-equivalent production was 2,999.\nU.S. Upstream\n\nU.S. upstream reported earnings of $12.6 billion."
    wrong = checks.problems(question, "1.76 billion cubic feet per day", [prose], [prose], None, sector.parent_only,
                            lambda h: above)
    assert wrong and "U.S." in wrong[0]


def test_an_extra_breakdown_after_the_right_figure_is_fine():
    assert not problems("What was Chevron's net crude oil production in 2022?",
                        "1,440 thousand barrels per day; the U.S. row is 888.", [CHART, CHEVRON])


def test_a_chart_axis_label_is_not_a_value():
    page = hit("nvda_p0060", "Revenue (in billions)\n$250\n$229.70\n$200\n$150\n$113.27\n$100\n$54.03\n$50\n$40.93\n$0")
    question = "What revenue does the peer-comparison bar chart show for NVIDIA?"
    assert "axis label" in problems(question, "NVIDIA's revenue is $250 billion.", [page])[0]
    assert not problems(question, "NVIDIA's revenue is $113.27 billion.", [page])


def test_a_definition_cites_the_chunk_that_defines_the_term():
    question = "How does Bank of America define conduct risk and who must follow the Code of Conduct?"
    assert checks.defined_term(question) == "conduct risk"
    ethics = hit("bac_p0059", "Employees follow the Code of Conduct.")
    defining = hit("bac_p0088", "Conduct risk is\nthe risk of improper actions, behaviors or practices.")
    assert "bac_p0088" in problems(question, "It is compliance risk.", [ethics], [ethics, defining])[0]
    assert not problems(question, "The risk of improper actions.", [defining], [ethics, defining])


def test_a_parent_company_statement_is_not_the_companys():
    parent = hit("bac_p0212", "Parent Company Only Condensed Balance Sheet\nTotal assets 559,733 527,902")
    assert problems("What were Bank of America's total assets?", "559,733 million.", [parent])
    assert not problems("What were the parent company's total assets?", "559,733 million.", [parent])


def test_companies_are_not_ranked_without_a_figure_of_each():
    walmart, depot = hit("wmt", "Total assets 284,668", ticker="WMT"), hit("hd", "Total assets 105,095", ticker="HD")
    pair = {"WMT": "Walmart", "HD": "Home Depot"}
    answer = "Walmart had higher total assets, 284,668."
    assert "Home Depot" in checks.ranked_without_figure(answer, pair, [walmart])[0]
    assert not checks.ranked_without_figure(answer, pair, [walmart, depot])
    assert not checks.ranked_without_figure("Only Walmart's figure is in the evidence, so no comparison can be made.", pair, [walmart])


# ------------------------------------------------------------------------------- charts read by a vision model

def test_values_the_chart_does_not_label_are_left_out():
    copied = "Thousands of barrels per day\n| Year | Value |\n|---|---|\n| 20 | 1,440 |\n| 21 | 1,440 |\n| 22 | 1,440 |"
    cleaned = checks.clean_chart_text(copied)
    assert "| 22 | 1,440 |" in cleaned and "| 20 |" not in cleaned and "| 21 |" not in cleaned
    real = "| Year | Value |\n|---|---|\n| 20 | 1,440 |\n| 21 | 1,512 |\n| 22 | 1,633 |"
    assert checks.clean_chart_text(real) == real
    zeros = "| | 20 | 21 |\n|---|---|---|\n| Other | 0 | 0 |\n| Fuel oil | 0 | 0 |\n| Jet fuel | 0 | 0 |"
    assert "Jet fuel" not in checks.clean_chart_text(zeros)
    hits = checks.clean_chart_tables([hit("a", copied, content_type="table", reader="qwen3VL"),
                                      hit("b", copied, content_type="table", reader="pdfParser")])
    assert "| 20 |" not in hits[0]["text"] and hits[1]["text"] == copied      # printed tables are never touched


def test_chart_tables_take_their_titles_only_when_the_units_line_up():
    from scripts.rag import financial_ingest_chroma_db as fi
    def table(n, unit, value):
        return {"chunk_id": f"t{n}", "content_type": "table", "reader": "qwen3VL", "text": f"{unit}\n| Year | Value |\n| 22 | {value} |"}
    tables = [table(0, "Thousands of barrels per day", "1,440"), table(1, "Millions of cubic feet per day", "7,677")]
    text = {"chunk_id": "x", "content_type": "text", "reader": "qwen3VL", "text": (
        "## Net crude oil production\nThousands of barrels per day\n\nLegend:\n- Asia\n\n"
        "## Net natural gas production\nMillions of cubic feet per day\n")}
    assert fi.chart_titles([*tables, text]) == {"t0": "Net crude oil production", "t1": "Net natural gas production"}
    assert fi.chart_titles([*reversed(tables), text]) == {}
    named = hit("t1", "Chart: Net natural gas production\n| 22 | 7,677 |", chart_title="Net natural gas production")
    assert not problems("What net natural gas production did Chevron report?", "7,677", [named])
    assert problems("What net crude oil production did Chevron report?", "7,677", [named])


# ------------------------------------------------------------------------------- derived figures

def test_changes_the_answer_computes_are_recomputed_and_the_formula_constant_is_not_a_figure():
    cited = [hit("c", "Dividends 12,296 in 2025 and 8,553 in 2020")]
    answer = "From 8,553 to 12,296, up 3,743, or (12,296 - 8,553) / 8,553 × 100 = 43.76%."
    numeric = tools.numeric_grounding(answer, cited, [])
    assert numeric["grounded"], numeric
    assert tools.numeric_grounding("From 8,553 to 12,296, up 4,000 (51.00%).", cited, [])["ungrounded"] == ["4,000", "51.00"]
    assert tools.numeric_grounding("The rate was 100.5%.", cited, [])["ungrounded"] == ["100.5"]
    fixed, records = tools.correct_changes("From 8,553 to 12,296, up 43.77%.", {8553.0, 12296.0})
    assert fixed == "From 8,553 to 12,296, up 43.76%." and records[0]["value"] == 43.76
    assert tools.correct_changes("From 8,553 to 12,296, up 51.20%.", {8553.0, 12296.0})[1] == []          # not close
    assert tools.exact_changes("From 8,553 to 12,296, up 43.77%.", {8553.0, 12296.0}, ["43.77"]) == [
        "(12,296 - 8,553) / 8,553 * 100 = 43.76%, not 43.77%"]
    assert tools.numeric_grounding("From 8,553 to 12,296, up 43.7%.", cited, [])["grounded"]                 # cut off
    assert tools.numeric_grounding("From 8,553 to 12,296, up 43.77%.", cited, [])["ungrounded"] == ["43.77"]   # wrong


def test_a_direction_claim_follows_from_the_answers_own_figures():
    known = {1440.0, 1497.0, 1560.0, 1827.0}
    answer = "1,440 in 2022, 1,497 in 2023, 1,560 in 2024 and 1,827 in 2025."
    assert tools.direction_follows("This indicates a consistent upward trend from 2022 to 2025", answer, known)
    assert not tools.direction_follows("This indicates a consistent downward trend", answer, known)
    assert not tools.direction_follows("An upward trend of 12.5%", answer, known)
    assert not tools.direction_follows("A steady upward trend", "1,440, then 1,827, then 1,560.", known)


# ------------------------------------------------------------------------------- statement rows

def test_total_revenue_wins_over_net_sales_unless_net_sales_is_asked():
    text = ("CONSOLIDATED STATEMENTS OF INCOME\n(in millions)\n2025 2024 2023\nNet sales 269,912 249,625 237,710\n"
            "Membership fees 5,323 4,828 4,580\nTotal revenue 275,235 254,453 242,290")
    hits = [hit("cost", text)]
    assert sector.label_match(hits, "revenue", 2025)["value"] == "275,235"
    labels = sector.labels_for("revenue", "How much did Costco net sales grow?")
    assert sector.label_match(hits, "revenue", 2025, labels)["value"] == "269,912"


def test_net_operating_revenues_and_written_scales_are_read():
    text = "CONSOLIDATED STATEMENTS OF INCOME\nYear Ended December 31, 2023 2022 2021\nNet Operating Revenues $ 45,754 $ 43,004 $ 38,655"
    row = sector.label_match([hit("ko", text, scale="In millions except per share data")], "revenue", 2023)
    assert row["value"] == "45,754" and row["unit"] == "millions"
    assert sector.scale_of_text("(MILLIONS, EXCEPT PER COMMON SHARE DATA)") == "millions"
    assert sector.scale_of_text("A note about millions of people") == ""


def test_segment_tables_and_change_columns_are_not_the_statement():
    text = ("Table 8.1: Domestic Card Business Results\n(Dollars in millions) 2024 2023 2022 2023 2022\n"
            "Net interest income $ 20,881 $ 18,610 $ 15,616 12% 19%")
    assert sector.label_match([hit("cof", text)], "net interest income", 2024) is None


def test_net_income_attributable_to_the_company_is_preferred():
    text = ("CONSOLIDATED STATEMENTS OF INCOME\n(in millions)\n2025 2024 2023\nNet income 5,878 5,067 11,172\n"
            "Net income attributable to MPC 4,047 3,445 9,681")
    assert sector.label_match([hit("mpc", text)], "net income", 2025)["value"] == "4,047"


def test_equity_and_operating_cash_flow_rows_are_read():
    balance = "CONSOLIDATED BALANCE SHEETS\n(in millions)\n2025 2024\nTotal liabilities 40,000 38,000\nTotal stockholders\u2019 equity 59,142 59,646"
    assert sector.label_match([hit("crm", balance)], "shareholders' equity", 2025)["value"] == "59,142"
    cash = "CONSOLIDATED STATEMENTS OF CASH FLOWS\n(in millions)\n2025 2024\nNet Cash Provided by Operating Activities 16,472 21,468"
    assert sector.label_match([hit("mrk", cash)], "operating cash flow", 2025)["value"] == "16,472"


def test_a_small_cell_is_still_a_column():
    text = ("CONSOLIDATED STATEMENTS OF OPERATIONS\n(in millions)\n2025 2024 2023\n"
            "Operating income . . . . . . . . 3,694 1,900 401\nNet income 3,000 1,641 854")
    assert sector.label_match([hit("amd", text)], "operating income", 2025)["value"] == "3,694"


def test_a_figure_no_line_names_is_not_the_measure():
    equity = hit("crm", "Balance at January 31, 2024 971 59,646\nOther comprehensive loss (12)\nBalance at January 31, 2025 962 61,173")
    phrases = ["total stockholders' equity", "stockholders' equity"]
    assert checks.off_measure("$59,646 million", [equity], phrases) == ["59,646"]
    prose = hit("hd", "Our fiscal 2025 net earnings\nwere $14.2 billion.")
    assert not checks.off_measure("$14.2 billion", [prose], ["net earnings"])


def test_a_page_readers_code_fence_is_not_a_code_payload():
    page = hit("cvx", "# Returns\n```\n300\n250\n```\nCash returned", reader="qwen3VL", content_type="text")
    assert "```" not in checks.clean_chart_tables([page])[0]["text"]
    printed = hit("x", "```\nrm -rf\n```", reader="pdfParser", content_type="text")
    assert "```" in checks.clean_chart_tables([printed])[0]["text"]


def test_a_row_spelled_one_year_per_line_is_read():
    text = ("CONSOLIDATED STATEMENTS OF OPERATIONS | (in millions of U.S. dollars)\nTotal revenues | 2022 | 43,166\n"
            "Total revenues | 2021 | 40,963")
    row = sector.label_match([hit("cb", text)], "revenue", 2022)
    assert row["value"] == "43,166" and row["columns"] == {2022: "43,166", 2021: "40,963"}


def test_a_subsidiarys_statement_or_a_note_is_not_the_companys():
    ohio = ("DUKE ENERGY OHIO, INC.\nCONSOLIDATED BALANCE SHEETS\n(in millions)\n2025 2024\nTotal current assets 900 800\n"
            "Total assets 13,446 12,768")
    assert sector.label_match([hit("duk", ohio, ticker="DUK")], "total assets", 2025) is None
    parent = ohio.replace("OHIO, INC.", "CORPORATION").replace("13,446", "191,000")
    assert sector.label_match([hit("duk", parent, ticker="DUK")], "total assets", 2025)["value"] == "191,000"
    note = "The acquisition (in thousands):\n2025 2024\nReal property 1,200,000 900,000\nTotal assets 1,524,609 1,100,000"
    assert sector.label_match([hit("well", note, ticker="WELL")], "total assets", 2025) is None


def test_units_agree_when_a_neighbouring_filing_prints_the_same_figure():
    points = [{"year": 2022, "status": "ok", "value": 82156.0, "unit": "thousands", "columns": {2022: "82,156"}},
              {"year": 2023, "status": "ok", "value": 58574.0, "unit": "millions", "columns": {2023: "58,574", 2022: "82,156"}},
              {"year": 2024, "status": "ok", "value": 56953.0, "unit": "", "columns": {2024: "56,953", 2023: "58,574", 2022: "82,156"}},
              {"year": 2025, "status": "ok", "value": 61548.0, "unit": "millions", "columns": {2025: "61,548", 2024: "56,953"}}]
    assert [p["unit"] for p in sector.reconcile_units(points)][2:] == ["millions", "millions"]
    page = "Millions of Dollars\n2022 2021\nTotal Revenues and Other Income 82,156 48,349\nshares in thousands"
    assert sector.scale_of_text(page.split("\nshares")[0]) == "millions"


def test_an_llm_figure_must_sit_in_its_years_column():
    by_country = hit("eog", "United States Trinidad International Total\n2022\nRevenues $ 22,486 $ 310 $ 1,200 $ 22,796")
    assert not sector.row_position_ok({"chunk_id": "eog", "value_text": "22,796"}, [by_country], 2022)
    prose = hit("amgn", "Amgen's total revenues were 33,424 million in fiscal 2024.")
    assert sector.row_position_ok({"chunk_id": "amgn", "value_text": "33,424"}, [prose], 2024)
    row = hit("vlo", "2025 2024 2023\nRevenues 122,687 129,881 144,766")
    assert sector.row_position_ok({"chunk_id": "vlo", "value_text": "129,881"}, [row], 2024)


def test_statement_pages_are_found_by_title_not_by_a_mention():
    meta = {"doc_id": "edgar_ars_FOXA_2023"}
    title = hit("t", "FOX CORPORATION\nCONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME", page=72, metadata=meta)
    note = hit("n", "The amounts recognized in the consolidated statements of operations are shown below.", page=90, metadata=meta)
    assert sector.statement_pages([note, title], "revenue") == [("edgar_ars_FOXA_2023", 72), ("edgar_ars_FOXA_2023", 71)]


# ------------------------------------------------------------------------------- fiscal-year names

def test_home_depot_and_lowes_name_the_year_that_starts_the_fiscal_year():
    assert companies.filing_year("HD", 2025) == 2026 and companies.fiscal_label("HD", 2026) == 2025
    assert companies.filing_year("WMT", 2025) == 2025 and companies.filing_year("HD", None) is None


def test_searches_use_the_filing_year_and_hits_show_the_companys_name_for_it(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "_retrieve_evidence", lambda config: seen.append(config.year) or [
        {"chunk_id": "a", "ticker": "HD", "year": 2026, "text": "Fiscal 2025"}])
    config = agent.AgentConfig(query="q", ticker="HD", year=2025, source_type=None, top_k=4, dense_top_k=4, bm25_top_k=4,
                               retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                               max_context_chars=6000, self_consistency_samples=1)
    assert agent.retrieve_evidence(config)[0]["year"] == 2025 and seen == [2026]


def test_a_year_the_question_says_ends_in_is_the_filing_year():
    clarifier = fc.Clarifier(directory={"HD": [2023, 2024, 2025], "WMT": [2024, 2025, 2026]})
    plan = tools.plan_targets("Which company had higher total assets in its fiscal year ending in 2026: Home Depot or "
                              "Walmart?", None, None, clarifier)
    assert plan == [{"ticker": "HD", "year": 2025}, {"ticker": "WMT", "year": 2026}]


# ------------------------------------------------------------------------------- classifier and LLM client

def test_a_clean_question_naming_a_company_and_measure_is_financial_whatever_the_llm_says(monkeypatch):
    monkeypatch.setattr(agent, "invoke_prompt", lambda *a: '{"is_plain_text": true, "financial_intent": false, "risk_level": "low"}')
    assert agent.llm_plain_text_classifier(None, "Compare Costco and Walmart revenue in 2025")["allowed"]
    assert not agent.llm_plain_text_classifier(None, "What is the weather in Paris?")["allowed"]


def test_a_prompt_over_the_context_limit_is_cut_and_sent_again(monkeypatch):
    sent = []

    def post(self, messages):
        sent.append(sum(len(m["content"]) for m in messages))
        if len(sent) < 3:
            raise urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(b"This model's maximum context length is 8192"))
        return agent.MockResponse("ok")

    monkeypatch.setattr(agent.HostedChatLLM, "_post", post)
    llm = agent.HostedChatLLM(api_base="http://x", api_key="k", model="m", temperature=0)
    assert llm._invoke([{"role": "user", "content": "Q " + "e" * 9000 + " Return JSON"}]).content == "ok"
    assert sent[0] > sent[1] > sent[2]
    cut = hosted_llm.shrink_messages([{"role": "user", "content": "start" + "e" * 1000 + "END"}])[0]["content"]
    assert cut.startswith("start") and cut.endswith("END")


def test_other_http_errors_are_not_retried_with_a_shorter_prompt(monkeypatch):
    calls = []

    def post(self, messages):
        calls.append(1)
        raise urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(b"unknown model"))

    monkeypatch.setattr(agent.HostedChatLLM, "_post", post)
    with pytest.raises(urllib.error.HTTPError):
        agent.HostedChatLLM(api_base="http://x", api_key="k", model="m", temperature=0)._invoke([{"role": "user", "content": "q"}])
    assert len(calls) == 1
