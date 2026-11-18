"""Synthetic N-PORT filings (same structure as EDGAR's ``primary_doc.xml``) and a loaded fund database for the
mutual fund tests. No network, models or downloaded data."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

ACC_500 = "0000036405-26-000473"
ACC_TOTAL = "0000036405-26-000480"
SERIES_500 = "S000002839"
SERIES_TOTAL = "S000002848"

# name, cusip, isin, ticker, assetCat, pct, value USD, issuerCat, payoff
HOLDINGS_500 = [
    ("NVIDIA Corp", "67066G104", "US67066G1040", None, "EC", 7.5, 75e9, "CORP", "Long"),
    ("Alphabet Inc-Cl A", "02079K305", "US02079K3059", None, "EC", 3.25, 32.5e9, "CORP", "Long"),
    ("Alphabet Inc-Cl C", "02079K107", "US02079K1079", None, "EC", 2.59, 25.9e9, "CORP", "Long"),
    ("Microsoft Corp", "594918104", "US5949181045", "MSFT", "EC", 4.3, 43e9, "CORP", "Long"),
    ("Acme Widgets Ltd", "G00000101", "KYG000001011", None, "EC", 1.0, 10e9, "CORP", "Long"),
    ("AbbVie Inc", "00287Y109", "US00287Y1091", None, "EC", 0.8, 8e9, "CORP", "Long"),
    ("Apple Hospitality REIT Inc", "03784Y200", "US03784Y2000", None, "EC", 0.005, 50e6, "CORP", "Long"),
    ("Amgen Inc", "031162100", "US0311621009", None, "EC", 0.5, 5e9, "CORP", "Long"),
    ("Microsoft Corp 3.5% 02/12/2035", "594918BB9", "N/A", None, "DBT", 0.1, 1e9, "CORP", "Long"),
    ("S&P500 EMINI FUT SEP26", "N/A", "N/A", "ESU6", "DE", 0.05, 5e8, "CORP", "Long"),
]
HOLDINGS_TOTAL = [
    ("NVIDIA Corp", "67066G104", "US67066G1040", None, "EC", 6.0, 120e9, "CORP", "Long"),
    ("Microsoft Corp", "594918104", "US5949181045", None, "EC", 4.0, 80e9, "CORP", "Long"),
    ("Amgen Inc", "031162100", "US0311621009", None, "EC", 0.3, 6e9, "CORP", "Long"),
]


def nport_xml(series_id: str, series_name: str, period: str, net_assets: float, holdings: list[tuple]) -> bytes:
    rows = []
    for name, cusip, isin, ticker, cat, pct, value, issuer, payoff in holdings:
        ids = f'<isin value="{isin}"/>' if isin != "N/A" else ""
        ids += f'<ticker value="{ticker}"/>' if ticker else ""
        rows.append(f"""<invstOrSec><name>{name.replace('&', '&amp;')}</name><lei>N/A</lei><title>{name[:16].replace('&', '&amp;')}</title>
<cusip>{cusip}</cusip><identifiers>{ids}</identifiers><balance>1000</balance><units>NS</units><curCd>USD</curCd>
<valUSD>{value}</valUSD><pctVal>{pct}</pctVal><payoffProfile>{payoff}</payoffProfile><assetCat>{cat}</assetCat>
<issuerCat>{issuer}</issuerCat><invCountry>US</invCountry></invstOrSec>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?><edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
<headerData><submissionType>NPORT-P</submissionType><filerInfo><seriesClassInfo><seriesId>{series_id}</seriesId>
<classId>C000007774</classId></seriesClassInfo></filerInfo></headerData>
<formData><genInfo><regName>VANGUARD INDEX FUNDS</regName><regCik>0000036405</regCik><seriesName>{series_name}</seriesName>
<seriesId>{series_id}</seriesId><seriesLei>12WZ1W76P8QD4VJ6OB47</seriesLei><repPdEnd>2026-12-31</repPdEnd>
<repPdDate>{period}</repPdDate></genInfo>
<fundInfo><totAssets>{net_assets * 1.001}</totAssets><totLiabs>{net_assets * 0.001}</totLiabs><netAssets>{net_assets}</netAssets>
<returnInfo><monthlyTotReturns><monthlyTotReturn classId="C000007774" rtn1="10.49" rtn2="5.26" rtn3="-0.95"/></monthlyTotReturns></returnInfo>
<mon1Flow redemption="12174698544.92" reinvestment="5915.33" sales="40632647042.06"/>
<mon2Flow redemption="11000000000" reinvestment="6000" sales="30000000000"/>
<mon3Flow redemption="10000000000" reinvestment="7000" sales="20000000000"/>
</fundInfo><invstOrSecs>{''.join(rows)}</invstOrSecs></formData></edgarSubmission>""".encode()


def write_raw(folder: Path) -> Path:
    """Two filings in a raw folder, named as the downloader names them, with the manifest of filing dates."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"nport_{ACC_500}_mutual_fund.xml").write_bytes(
        nport_xml(SERIES_500, "VANGUARD 500 INDEX FUND", "2026-06-30", 1_000_000_000_000, HOLDINGS_500))
    (folder / f"nport_{ACC_TOTAL}_mutual_fund.xml").write_bytes(
        nport_xml(SERIES_TOTAL, "VANGUARD TOTAL STOCK MARKET INDEX FUND", "2026-06-30", 2_000_000_000_000, HOLDINGS_TOTAL))
    (folder / "manifest_mutual_fund.jsonl").write_text(
        f'{{"accession": "{ACC_500}", "filing_date": "2026-08-27"}}\n'
        f'{{"accession": "{ACC_TOTAL}", "filing_date": "2026-08-28"}}\n', encoding="utf-8")
    return folder


def tsv_zip(path: Path) -> Path:
    """A miniature quarterly data set: the six tables the loader reads, two funds, real column names."""
    tables = {
        "SUBMISSION": ["ACCESSION_NUMBER\tFILING_DATE\tFILE_NUM\tSUB_TYPE\tREPORT_ENDING_PERIOD\tREPORT_DATE\tIS_LAST_FILING",
                       f"{ACC_500}\t27-AUG-2026\t\tNPORT-P\t31-DEC-2026\t30-JUN-2026\tN",
                       "0000000001-26-000001\t01-AUG-2026\t\tNPORT-P\t31-DEC-2026\t30-JUN-2026\tN"],
        "REGISTRANT": ["ACCESSION_NUMBER\tCIK\tREGISTRANT_NAME\tFILE_NUM\tLEI",
                       f"{ACC_500}\t0000036405\tVANGUARD INDEX FUNDS\t811-02652\t549300G6KNDK44WUN559",
                       "0000000001-26-000001\t0000000001\tTiny Bond Trust\t811-00001\tN/A"],
        "FUND_REPORTED_INFO": ["ACCESSION_NUMBER\tSERIES_NAME\tSERIES_ID\tSERIES_LEI\tTOTAL_ASSETS\tTOTAL_LIABILITIES\t"
                               "NET_ASSETS\tSALES_FLOW_MON1\tREINVESTMENT_FLOW_MON1\tREDEMPTION_FLOW_MON1",
                               f"{ACC_500}\tVANGUARD 500 INDEX FUND\t{SERIES_500}\tX\t1001\t1\t1000000000000\t5\t1\t2",
                               "0000000001-26-000001\tTiny Bond Fund\tS000000001\tY\t11\t1\t10\t0\t0\t0"],
        "MONTHLY_TOTAL_RETURN": ["ACCESSION_NUMBER\tMONTHLY_TOTAL_RETURN_ID\tCLASS_ID\tMONTHLY_TOTAL_RETURN1\t"
                                 "MONTHLY_TOTAL_RETURN2\tMONTHLY_TOTAL_RETURN3",
                                 f"{ACC_500}\t1\tC000007774\t10.49\t5.26\t-0.95"],
        "FUND_REPORTED_HOLDING": ["ACCESSION_NUMBER\tHOLDING_ID\tISSUER_NAME\tISSUER_LEI\tISSUER_TITLE\tISSUER_CUSIP\tBALANCE\t"
                                  "UNIT\tOTHER_UNIT_DESC\tCURRENCY_CODE\tCURRENCY_VALUE\tEXCHANGE_RATE\tPERCENTAGE\t"
                                  "PAYOFF_PROFILE\tASSET_CAT\tOTHER_ASSET\tISSUER_TYPE\tOTHER_ISSUER\tINVESTMENT_COUNTRY\t"
                                  "IS_RESTRICTED_SECURITY\tFAIR_VALUE_LEVEL\tDERIVATIVE_CAT",
                                  f"{ACC_500}\t101\tNVIDIA Corp\tN/A\tNVIDIA\t67066G104\t1\tNS\t\tUSD\t75000000000\t\t7.5\tLong\tEC\t\tCORP\t\tUS\tN\t1\t",
                                  f"{ACC_500}\t102\tMicrosoft Corp\tN/A\tMSFT\t594918104\t1\tNS\t\tUSD\t43000000000\t\t4.3\tLong\tEC\t\tCORP\t\tUS\tN\t1\t",
                                  "0000000001-26-000001\t900\tSome Bond\tN/A\tB\t000000000\t1\tPA\t\tUSD\t10\t\t100\tLong\tDBT\t\tCORP\t\tUS\tN\t2\t"],
        "IDENTIFIERS": ["HOLDING_ID\tIDENTIFIERS_ID\tIDENTIFIER_ISIN\tIDENTIFIER_TICKER\tOTHER_IDENTIFIER\tOTHER_IDENTIFIER_DESC",
                        "102\t1\tUS5949181045\tMSFT\t\t"],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(f"{name}.tsv", "\n".join(lines) + "\n")
    path.write_bytes(buffer.getvalue())
    return path


def load_funds(folder: Path) -> Path:
    """Load both synthetic filings into ``<folder>/nport_mutual_fund.sqlite3``; returns the database path."""
    from scripts.rag import nport_mutual_fund as nport
    raw = write_raw(folder / "raw_mutual_fund")
    db = folder / "nport_mutual_fund.sqlite3"
    nport.main(["load", "--db", str(db), "--xml", str(raw)])
    return db
