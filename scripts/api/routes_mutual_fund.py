"""Read-only mutual fund endpoints (``/mutual_fund/...``), mounted by ``api_app``. Questions go through the chat routes
(router -> fund agent); these list what is loaded and return rows exactly, for the UI and for checking answers.
The API's auth middleware and rate limiter cover them like every other route."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from scripts.agents.tools_mutual_fund import FundStore
from scripts.common.companies import NAMES, normalize_ticker

router = APIRouter(prefix="/mutual_fund", tags=["Mutual funds"])


def store() -> FundStore:
    found = FundStore()
    if not found.available():
        raise HTTPException(status_code=404, detail="No mutual fund data is loaded (scripts.rag.nport_mutual_fund load).")
    return found


@router.get("/funds", summary="Loaded funds and their N-PORT filings")
def list_funds() -> dict[str, Any]:
    funds = store().funds()
    return {"count": len(funds), "funds": funds}


@router.get("/holdings", summary="Top equity holdings of one fund filing (share classes summed)")
def holdings(series_id: str = Query(..., max_length=20), year: int | None = Query(None, ge=2019, le=2100),
             limit: int = Query(25, ge=1, le=500)) -> dict[str, Any]:
    funds = store()
    filing = funds.filing(series_id, year)
    if not filing:
        raise HTTPException(status_code=404, detail=f"No filing for series {series_id}" + (f" in {year}" if year else ""))
    return {"filing": filing, "coverage": funds.coverage(filing["accession"]),
            "holdings": funds.top_holdings(filing["accession"], limit),
            "sector_mix": funds.sector_mix(filing["accession"])}


@router.get("/holders/{ticker}", summary="Loaded funds holding a corpus company, largest weight first")
def holders(ticker: str, limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    if symbol not in NAMES:
        raise HTTPException(status_code=404, detail=f"{ticker} is not a corpus company")
    return {"ticker": symbol, "company": NAMES[symbol], "holders": store().holders(symbol, limit)}


@router.get("/search", summary="Fund cards nearest a query (bge-small-en-v1.5, collection fund_cards_mutual_fund)")
def search(q: str = Query(..., min_length=2, max_length=300), top_k: int = Query(3, ge=1, le=10)) -> dict[str, Any]:
    store()
    from scripts.rag import cards_mutual_fund as cards
    return {"query": q, "hits": cards.search(q, top_k)}
