import json
import zipfile

from scripts.common import colab_bundle as cb


def test_only_code_and_small_data_go_in():
    keep = ["scripts/rag/colab_pipeline.py", "tests/test_x.py", "configs/financial_reports/apple_2024.json",
            "pyproject.toml", "uv.lock", "notebooks/rag_architectures/eval_set_cvx_2024.py",
            "data/raw/financial_report_qa/annual_reports/corpus_plan.jsonl", "data/evaluation/financial_report_qa/eval_set_v1.jsonl",
            "data/samples/multimodal/hr_policy/x.pdf"]
    drop = ["scripts/rag/__pycache__/x.cpython-312.pyc", "notebooks/colab_rag_pipelines.ipynb",
            "data/raw/financial_report_qa/pdfs/x.pdf", "data/raw/hr_policy/documents/x.json", "docs/x.html",
            "Annual_Report_Copilot_TrackB_Mentor_Deck.pptx"]
    assert all(cb.included(p) for p in keep)
    assert not any(cb.included(p) for p in drop)


def test_build_writes_a_stamped_zip_of_the_selected_files(tmp_path):
    out = tmp_path / "code.zip"
    result = cb.build(out, files=["scripts/rag/colab_pipeline.py", "pyproject.toml", "docs/annual_report_corpus_plan.html"])
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        info = json.loads(archive.read(cb.INFO_NAME))
    assert sorted(names) == [cb.INFO_NAME, "pyproject.toml", "scripts/rag/colab_pipeline.py"]
    assert info["files"] == result["files"] == 2 and info["built_at"]
    assert not out.with_name("code.zip.part").exists()
