"""The Colab notebook the user runs: every code cell is valid Python and cell 1 holds the settings of the real run."""
import json
import re

from scripts.common.paths import ROOT

NOTEBOOK = ROOT / "notebooks" / "colab_ingest_chroma_db.ipynb"


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_every_code_cell_compiles_without_its_shell_lines():
    for source in cells():
        python = "\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%"))
                           else line for line in source.splitlines())
        compile(python, source.splitlines()[0], "exec")


def test_cell_one_defaults_are_the_real_run():
    settings = cells()[0]
    expected = {"SMOKE_TEST": "False", "READER": '"routed"', "QWEN_MODEL": '"Qwen/Qwen3-VL-8B-Instruct"',
                "QWEN_PRECISION": '"auto"', "PAID_VISION_MODEL": '"gemini-3.8-flash"', "PAID_REASONING_EFFORT": '"low"',
                "PAID_FIRST": "True", "PAID_THRESHOLD": "0.10", "PAID_MAX_TOKENS": "4096", "DESCRIBE_VISUALS": "True",
                "PAID_CONCURRENCY": "4", "CORPUS": '"min500"'}
    for name, value in expected.items():
        assert re.search(rf"^{name} = {re.escape(value)}\s", settings, re.M), name
    assert "PAID_VISION_API_KEY" not in re.sub(r"#.*", "", settings)       # the key is a Colab secret, never a setting


def test_cell_order_and_titles():
    titles = [source.splitlines()[0] for source in cells()]
    order = [t.split(".")[0].replace("# @title ", "") for t in titles]
    assert order[:9] == ["1", "2", "3", "3b", "3c", "4", "5", "6", "6b"] and "6c" in order
