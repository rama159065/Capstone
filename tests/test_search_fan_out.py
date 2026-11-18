"""Parallel search fan-out: searches of one step run side by side, results stay in target order, LLM calls stay serial,
and a run gives the same evidence and figures with 1 worker or 4."""
from __future__ import annotations

import threading
import time

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_tools as tools
from tests.test_sector_trend import DIRECTORY, QUERY, config, hit


def test_fan_out_keeps_order_runs_first_alone_and_overlaps(monkeypatch):
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    running, peak, first_done = [0], [0], threading.Event()
    lock = threading.Lock()

    def work(item):
        if item != 0:
            assert first_done.is_set()                  # the first item ran alone (lazy models are loaded once)
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        if item == 0:
            first_done.set()
        return item * 10

    results, log = tools.fan_out(work, list(range(9)), workers=4)
    assert results == [i * 10 for i in range(9)]
    assert log["tool"] == "search_fan_out" and log["searches"] == 9 and log["workers"] == 4
    assert peak[0] == 4


def test_fan_out_raises_like_a_loop():
    def work(item):
        if item == 3:
            raise ValueError("search failed")
        return item

    with pytest.raises(ValueError, match="search failed"):
        tools.fan_out(work, list(range(6)), workers=3)


def test_workers_setting_and_embedded_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("RAG_BACKEND", "financial_vlm")
    monkeypatch.delenv("RAG_SEARCH_WORKERS", raising=False)
    assert tools.search_workers() == 4
    monkeypatch.setenv("RAG_SEARCH_WORKERS", "1")
    assert tools.search_workers() == 1
    monkeypatch.setenv("RAG_SEARCH_WORKERS", "50")
    assert tools.search_workers() == tools.MAX_SEARCH_WORKERS
    monkeypatch.setenv("RAG_BACKEND", "financial_qdrant")                    # embedded Qdrant: one client, sequential
    assert tools.search_workers() == 1
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    assert tools.search_workers() == tools.MAX_SEARCH_WORKERS


def run_sector(monkeypatch, workers):
    monkeypatch.setenv("RAG_SEARCH_WORKERS", str(workers))
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    for key in ("RAG_FIN_VISUAL", "RAG_BACKEND", "RAG_SUPPORT_CHECK", "RAG_SECTOR_MAX_COMPANIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    threads = set()

    def fake(run_config):
        threads.add(threading.get_ident())
        time.sleep(0.02)
        return [hit(run_config.ticker, run_config.year)]

    monkeypatch.setattr(agent, "retrieve_evidence", fake)
    result = agent.financial_graph(agent.MockLLM(), config(QUERY), "sector_trend_graph")
    return result, threads


def test_sector_run_is_the_same_with_one_worker_or_four(monkeypatch):
    serial, serial_threads = run_sector(monkeypatch, 1)
    parallel, parallel_threads = run_sector(monkeypatch, 4)
    assert len(serial_threads) == 1 and len(parallel_threads) > 1
    assert [h["chunk_id"] for h in parallel["evidence"]] == [h["chunk_id"] for h in serial["evidence"]]
    assert parallel["sector_trend"]["rows"] == serial["sector_trend"]["rows"]
    assert parallel["agent_run"]["path"] == serial["agent_run"]["path"]
    strip = lambda calls: [{k: v for k, v in c.items() if k != "ms"} for c in calls]
    assert strip(parallel["agent_run"]["tool_calls"]) == strip(serial["agent_run"]["tool_calls"])
    assert serial["agent_run"]["search_fan_out"] == []
    assert parallel["agent_run"]["search_fan_out"][0]["workers"] == 4
