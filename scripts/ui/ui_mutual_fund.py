"""Streamlit view of a fund answer (``mutual_fund`` in the chat result): the holdings table, sector mix, exposure or
holders, the linked company figures with weighted growth, and the coverage line. Used by ``api_streamlit_app``."""
from __future__ import annotations

from typing import Any

KIND_TITLES = {"top_holdings": "Top holdings", "sector_mix": "Sector mix", "exposure": "Exposure",
               "holders": "Funds holding", "facts": "Fund facts", "flows": "Monthly flows",
               "returns": "Monthly returns", "holdings_growth": "Holdings' growth", "holdings_text": "Holdings' reports"}


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value:.2f}%"


def render_mutual_fund(table: dict[str, Any]) -> None:
    if not table:
        return
    import streamlit as st                    # at call time: the app's streamlit (tests swap the module)
    kind = table.get("kind") or ""
    if kind == "holders":
        title = f"{KIND_TITLES[kind]} {table.get('company')} ({table.get('ticker')})"
    else:
        title = f"{KIND_TITLES.get(kind, kind)}: {table.get('fund')} · holdings as of {table.get('period_end')}"
    with st.expander(title, expanded=True):
        if kind != "holders":
            coverage = table.get("coverage") or {}
            net = table.get("net_assets")
            st.caption(f"Form N-PORT, accession {table.get('accession')}"
                       + (f", filed {table['filing_date']}" if table.get("filing_date") else "")
                       + (f" · net assets ${net / 1e9:,.1f} billion" if net else "")
                       + (f" · {_pct(coverage.get('corpus_pct_of_net_assets'))} of net assets in "
                          f"{coverage.get('corpus_companies')} corpus companies" if coverage else ""))
        if table.get("rows"):
            st.dataframe(table["rows"], width="stretch")
        if table.get("sector_mix"):
            st.dataframe(table["sector_mix"], width="stretch")
            st.bar_chart([{"sector": r["sector"], "% of net assets": r["% of net assets"]} for r in table["sector_mix"]],
                         x="sector", y="% of net assets")
        companies = table.get("companies") or []
        if kind == "holdings_growth" and companies:
            st.dataframe([{"company": c["ticker"], "weight": _pct(c.get("weight")), "report": f"FY{c['year']}",
                           **{f"FY{p['year']}": p.get("value_text") if p.get("status") == "ok" else p.get("status")
                              for p in c.get("points") or []},
                           "growth": _pct(c.get("growth_pct"))} for c in companies], width="stretch")
            st.caption(f"Weighted growth {_pct(table.get('weighted_growth_pct'))} over "
                       f"{_pct(table.get('covered_weight'))} of the {_pct(table.get('asked_weight'))} of net assets asked "
                       "about. Each figure is read from the annual report the company filed by the holdings date.")
        elif kind == "holdings_text" and companies:
            st.caption("Annual reports searched: " + ", ".join(f"{c['ticker']} FY{c['year']}" for c in companies))
        if table.get("excluded"):
            st.caption("Not used: " + "; ".join(f"{e['company']} ({e['reason']})" for e in table["excluded"]))
        st.caption("Holdings are as reported on Form N-PORT for the date shown and change over time. Facts only, not "
                   "investment advice.")
