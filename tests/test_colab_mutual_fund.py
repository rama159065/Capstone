"""Mutual fund data travels to Colab and GCP: pack the local folder into a zip, unpack it from Drive into the Colab
RAG_HOME on ``restore``, export it to the bucket, and the deploy scripts pick it up."""
from __future__ import annotations

import sqlite3
import zipfile

from scripts.common.paths import ROOT
from scripts.rag import colab_pipeline as cp
from tests.data_mutual_fund import load_funds


def local_funds(tmp_path):
    folder = tmp_path / "local" / "mutual_fund"
    load_funds(folder)
    (folder / "chroma_mutual_fund").mkdir()
    (folder / "chroma_mutual_fund" / "chroma.sqlite3").write_bytes(b"cards")
    return folder


def holdings(db):
    with sqlite3.connect(db) as con:
        return con.execute("select count(*) from holdings").fetchone()[0]


def test_pack_restore_and_export_mutual_fund_data(tmp_path):
    source = local_funds(tmp_path)
    drive, work = tmp_path / "drive", tmp_path / "work"
    archive = cp.pack_mutual_fund(source, drive / cp.FUND_ZIP)
    names = zipfile.ZipFile(archive).namelist()
    assert cp.FUND_DB in names and "chroma_mutual_fund/chroma.sqlite3" in names
    assert any(n.startswith("raw_mutual_fund/") for n in names)

    s = cp.Settings(pipeline="vlm_chroma", work=work, drive_root=drive)
    assert cp.restore_mutual_fund(s) is True
    live = work / "indexes" / "mutual_fund"
    assert holdings(live / cp.FUND_DB) == holdings(source / cp.FUND_DB) > 0
    assert (live / "chroma_mutual_fund" / "chroma.sqlite3").read_bytes() == b"cards"
    assert (drive / "indexes" / "mutual_fund" / cp.FUND_DB).exists()       # unpacked on Drive once, for export
    assert cp.restore_mutual_fund(s) is False                               # already in place: untouched

    commands = cp.export_to_gcs(s, "capstone-rag-demo", dry_run=True)
    assert commands[-1][-2:] == [str(drive / "indexes" / "mutual_fund"), "gs://capstone-rag-demo/indexes/mutual_fund"]


def test_restore_without_fund_data_leaves_the_route_off(tmp_path, capsys):
    s = cp.Settings(pipeline="vlm_chroma", work=tmp_path / "work", drive_root=tmp_path / "drive")
    assert cp.restore_mutual_fund(s) is False
    assert "fund route stays off" in capsys.readouterr().out
    assert all("mutual_fund" not in c[-1] for c in cp.export_to_gcs(s, "b", dry_run=True))


def test_deploy_scripts_bring_the_fund_folder():
    sync = (ROOT / "deploy" / "sync_from_gcs.sh").read_text(encoding="utf-8")
    restore = (ROOT / "deploy" / "restore_index.sh").read_text(encoding="utf-8")
    assert '"indexes/mutual_fund"' in sync
    assert "/snapshots/mutual_fund_data.zip" in restore and "nport_mutual_fund.sqlite3" in restore
