"""N-PORT loader: XML and bulk-TSV parsing, identifier-first mapping to corpus companies, exposure and coverage."""
from __future__ import annotations

import sqlite3

from scripts.rag import nport_mutual_fund as nport
from tests.data_mutual_fund import ACC_500, ACC_TOTAL, HOLDINGS_500, SERIES_500, load_funds, nport_xml, tsv_zip


def links(db_path, accession):
    with sqlite3.connect(db_path) as db:
        return {name: (ticker, method) for name, ticker, method in db.execute(
            "SELECT h.issuer_name, l.corpus_ticker, l.method FROM holding_link l JOIN holdings h USING (accession, "
            "holding_id) WHERE accession = ?", (accession,))}


def test_parse_xml_reads_fund_filing_flows_returns_and_holdings():
    parsed = nport.parse_xml(nport_xml(SERIES_500, "VANGUARD 500 INDEX FUND", "2026-06-30", 1e12, HOLDINGS_500),
                             accession=ACC_500, filing_date="2026-08-27")
    assert parsed["fund"] == {"series_id": SERIES_500, "series_name": "VANGUARD 500 INDEX FUND",
                              "registrant": "VANGUARD INDEX FUNDS", "cik": "0000036405",
                              "series_lei": "12WZ1W76P8QD4VJ6OB47", "family": "Vanguard"}
    filing = parsed["filing"]
    assert (filing["period_end"], filing["fiscal_year_end"], filing["filing_date"]) == ("2026-06-30", "2026-12-31", "2026-08-27")
    assert filing["net_assets"] == 1e12 and len(parsed["holdings"]) == len(HOLDINGS_500)
    assert parsed["flows"][0] == {"month": 1, "sales": 40632647042.06, "reinvestments": 5915.33, "redemptions": 12174698544.92}
    assert [r["total_return"] for r in parsed["returns"]] == [10.49, 5.26, -0.95]
    msft = next(h for h in parsed["holdings"] if h["issuer_name"] == "Microsoft Corp")
    assert (msft["ticker_raw"], msft["cusip"], msft["asset_cat"], msft["pct_of_net_assets"]) == ("MSFT", "594918104", "EC", 4.3)


def test_holdings_map_by_identifier_first_and_never_guess(tmp_path):
    db = load_funds(tmp_path)
    found = links(db, ACC_500)
    assert found["Microsoft Corp"] == ("MSFT", "ticker")                         # reported ticker
    assert found["NVIDIA Corp"] == ("NVDA", "name")                              # no identifier known yet: name
    assert found["Alphabet Inc-Cl A"][0] == found["Alphabet Inc-Cl C"][0] == "GOOGL"   # share classes, one company
    assert found["Microsoft Corp 3.5% 02/12/2035"] == ("MSFT", "cusip6")         # issuer CUSIP-6 learned from MSFT
    assert found["Apple Hospitality REIT Inc"] == (None, "unmatched")            # not Apple
    assert found["Acme Widgets Ltd"] == (None, "unmatched")
    assert links(db, ACC_TOTAL)["NVIDIA Corp"] == ("NVDA", "cusip6")              # learned in the first filing
    with sqlite3.connect(db) as conn:
        exposure = dict(conn.execute("SELECT corpus_ticker, pct_of_net_assets FROM fund_exposure WHERE accession = ?",
                                     (ACC_500,)))
    assert exposure["GOOGL"] == 3.25 + 2.59 and exposure["MSFT"] == 4.3          # debt is not ownership
    assert "ESU6" not in exposure


def test_coverage_and_sector_mix(tmp_path):
    db = load_funds(tmp_path)
    with sqlite3.connect(db) as conn:
        info = nport.coverage(conn, ACC_500)
        mix = nport.sector_mix(conn, ACC_500)
    assert info["corpus_companies"] == 5
    assert round(info["corpus_pct_of_net_assets"], 4) == round(7.5 + 3.25 + 2.59 + 4.3 + 0.8 + 0.5, 4)
    assert round(info["equity_pct_of_net_assets"], 4) == round(info["corpus_pct_of_net_assets"] + 1.0 + 0.005, 4)
    assert mix[-1]["sector"] == "Not in corpus" and round(mix[-1]["pct_of_net_assets"], 3) == 1.005
    assert round(sum(r["pct_of_net_assets"] for r in mix), 4) == round(info["equity_pct_of_net_assets"], 4)


def test_reload_replaces_a_filing_instead_of_duplicating(tmp_path):
    db = load_funds(tmp_path)
    nport.main(["load", "--db", str(db), "--xml", str(tmp_path / "raw_mutual_fund")])
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM holdings WHERE accession = ?", (ACC_500,)).fetchone()[0] == len(HOLDINGS_500)
        assert conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 2


def test_bulk_zip_is_streamed_for_the_chosen_series_only(tmp_path):
    archive = tsv_zip(tmp_path / "2026q3_nport_mutual_fund.zip")
    parsed = nport.parse_tsv_zip(archive, name_like=["500 INDEX"])
    assert len(parsed) == 1
    filing = parsed[0]
    assert filing["fund"]["series_id"] == SERIES_500 and filing["fund"]["registrant"] == "VANGUARD INDEX FUNDS"
    assert (filing["filing"]["period_end"], filing["filing"]["fiscal_year_end"]) == ("2026-06-30", "2026-12-31")
    assert filing["filing"]["filing_date"] == "2026-08-27"
    assert {h["issuer_name"]: h["ticker_raw"] for h in filing["holdings"]} == {"NVIDIA Corp": "", "Microsoft Corp": "MSFT"}
    db = tmp_path / "bulk.sqlite3"
    nport.main(["load", "--db", str(db), "--tsv-zip", str(archive), "--series", SERIES_500])
    assert links(db, ACC_500)["Microsoft Corp"] == ("MSFT", "ticker")
    assert nport.parse_tsv_zip(archive, top=1)[0]["fund"]["series_id"] == SERIES_500     # largest by net assets


def test_name_matching_refuses_other_companies_with_the_same_words():
    by_name = nport.name_index()
    assert nport.match_name("Costco Wholesale Corp", by_name) == "COST"
    assert nport.match_name("Marriott International Inc/MD", by_name) == "MAR"
    assert nport.match_name("Apple Hospitality REIT Inc", by_name) is None
    assert nport.match_name("Portland General Electric Co", by_name) is None
    assert nport.match_name("Norfolk Southern Corp", by_name) != "SO"
    assert nport.ident("N/A") is None and nport.cusip6("000000000") is None
