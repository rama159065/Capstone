"""Build notebooks/08_week1_prompt_strategies.ipynb (Track B Week 1: single agent, single prompt).

    python notebooks/build_week1_notebook.py

The notebook drives scripts/evaluation/prompt_baselines.py and scripts/evaluation/failure_replay.py, so the code
under test is the repo's; the notebook only sets up, runs and shows. It works on the PC (repo venv) and on Colab
(code zip or clone, the Chroma index restored from Drive, the provided LLM endpoint from Colab secrets).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("08_week1_prompt_strategies.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = r"""
# 08 · Week 1: single agent, single prompt — five prompting strategies, and how the small model breaks

**Track B, Week 1 requirement:**

> Single-agent, single-prompt attempt. Systematic comparison of at least three prompting strategies (structured output
> with few-shot, chain-of-thought or self-consistency, decomposition prompting). Small models break in specific ways —
> the team must document how theirs breaks.

This notebook runs that comparison on the **frozen evaluation set** (86 questions, `data/evaluation/financial_report_qa/eval_set_v1.jsonl`):

| Step | What it shows |
|---|---|
| 1. The five templates | `zero_shot`, `structured_few_shot`, `chain_of_thought`, `self_consistency` (3 samples, majority vote), `decomposition`: one system prompt, one call per answer (three for self-consistency) |
| 2. Evidence, four ways | **none** (prompt only, no documents: **the Week 1 single-prompt baseline**, the model answers from memory), **rag** (the capstone's own retrieval: company × year filters, Chroma hybrid search on the 509-filing index), plus two references: **bm25** (naive keyword search) and **oracle** (the gold document: the ceiling for prompting) |
| 3. Run | the same prompts and model on each evidence set; resumable |
| 4. Results | accuracy per strategy and category, valid JSON, valid citations, unsupported numbers, cost |
| 5. How it breaks | the failure taxonomy with a verbatim example of each failure |
| 6. What the capstone fixes | which prompt-only failures capstone retrieval turns into correct answers, and which the agent's checks (policy refusal, coverage, company scope, number grounding) stop before release |

Code: [scripts/evaluation/prompt_baselines.py](../scripts/evaluation/prompt_baselines.py), [scripts/evaluation/failure_replay.py](../scripts/evaluation/failure_replay.py).
Write-ups: [docs/week1_prompt_baselines.md](../docs/week1_prompt_baselines.md) and [docs/week1_failure_scenarios.md](../docs/week1_failure_scenarios.md).

**Where it runs**
- **PC:** the repo's venv. The local LLM stand-in is llama.cpp + Qwen3-VL-2B (`python -m scripts.rag.run_local_multimodal prepare` downloads it); about 20–40 s per call on CPU, so a full context (602 calls) takes hours. The run resumes where it stopped.
- **Colab:** the provided Track B endpoint (Colab secrets `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`), the Chroma index restored from Drive for **rag**, and `week1_baselines.zip` on Drive for **bm25/oracle** and the recorded runs (make it on the PC with `python -m scripts.evaluation.prompt_baselines --pack`).
"""

SETTINGS = r"""
# @title 1. Settings
LLM = "auto"  # @param ["auto", "hosted", "local", "mock"]
# auto: the provided endpoint when LLM_API_BASE and LLM_MODEL are set (env or Colab secrets), else mock.
# local: the PC's llama.cpp Qwen3-VL-2B stand-in. mock: checks the pipeline, says nothing about a model.
CONTEXTS = "none,rag,bm25,oracle"  # @param {type:"string"}
# none = prompt only (Week 1 baseline); rag = capstone retrieval; bm25 and oracle are references
STRATEGIES = "zero_shot,structured_few_shot,chain_of_thought,self_consistency,decomposition"  # @param {type:"string"}
LIMIT = 0  # @param {type:"integer"}
# 0 = all 86 questions; e.g. 10 for a quick pass (a run with a limit is a separate run id)
RUN_LLM = True  # @param {type:"boolean"}
# False: only show the recorded runs and the retrieval measures (no model calls)
RUN_PREFIX = ""  # @param {type:"string"}
# "" = hosted_<model> / local_qwen3vl2b / mock; runs are output/prompt_baselines/<prefix>_<context>
CHROMA_DIR = ""  # @param {type:"string"}
# PC only: the Chroma index for the rag context ("" = RAG_VLM_DIR or the runtime profile's chroma_db)
DRIVE_ROOT = "/content/drive/MyDrive/capstone_rag"
CODE_ZIP = "/content/drive/MyDrive/capstone_rag/capstone_code.zip"
WEEK1_ZIP = "/content/drive/MyDrive/capstone_rag/week1_baselines.zip"
REPO = "https://github.com/rama159065/Capstone.git"
BRANCH = "financial-report-only"
"""

SETUP = r"""
# @title 2. Setup: code, packages, evidence inputs, LLM
import json, os, subprocess, sys, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    root = Path("/content/Capstone")
    if Path(CODE_ZIP).exists():
        with zipfile.ZipFile(CODE_ZIP) as archive:
            info = archive.read("BUNDLE_INFO.json")
            stamp = root / "BUNDLE_INFO.json"
            if not stamp.exists() or stamp.read_bytes() != info:
                archive.extractall(root)
    elif not root.exists():
        token = None
        try:
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            pass
        url = REPO.replace("https://", f"https://{token}@") if token else REPO
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, str(root)], check=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    pin = ["-c", "deploy/colab_constraints.txt"] if Path("deploy/colab_constraints.txt").exists() else []
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pin, "chromadb==1.5.9", "rank-bm25",
                           "pymupdf>=1.24,<2", "sentence-transformers>=3", "langgraph>=0.2", "langchain-core>=0.3",
                           "langsmith", "pydantic>=2"])
    if Path(WEEK1_ZIP).exists():                       # BM25 chunks + recorded runs from the PC (--pack)
        with zipfile.ZipFile(WEEK1_ZIP) as archive:
            archive.extractall(root)
        print("unpacked", WEEK1_ZIP)
    for key in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL", "LLM_CA_CERT"):
        try:
            value = userdata.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = value
else:
    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    os.chdir(root)
    sys.path.insert(0, str(root))

from scripts.evaluation import prompt_baselines as pb
from scripts.evaluation import failure_replay as fr

# Chroma index for the rag context
if IN_COLAB:
    from scripts.rag import colab_pipeline as cp
    s = cp.Settings(pipeline="vlm_chroma", corpus_preset="min500", drive_root=Path(DRIVE_ROOT))
    if "rag" in CONTEXTS:
        cp.restore(s)                                    # Drive copy -> local disk (skipped when already there)
    os.environ.update(cp.serving_env(s))
else:
    from scripts.common import runtime_profile
    if CHROMA_DIR:
        os.environ["RAG_VLM_DIR"] = CHROMA_DIR
    runtime_profile.apply_profile()
    os.environ["RAG_BACKEND"] = "financial_vlm"

contexts = [c.strip() for c in CONTEXTS.split(",") if c.strip()]
strategies = [x.strip() for x in STRATEGIES.split(",") if x.strip()]
ready = {"none": True, "bm25": pb.CHUNKS.exists() or pb.MANIFEST.exists(), "oracle": pb.CHUNKS.exists() or pb.MANIFEST.exists(),
         "rag": (Path(os.environ.get("RAG_VLM_DIR", "")) / "vlm_ingest.sqlite3").exists()}
for c in contexts:
    print(f"{c:7s}", "ready" if ready.get(c) else "MISSING INPUT: " + (
        "the Chroma index (RAG_VLM_DIR)" if c == "rag" else f"{pb.CHUNKS} (week1_baselines.zip) or the EDGAR PDFs"))
contexts = [c for c in contexts if ready.get(c)]

provider = LLM
if provider == "auto":
    provider = "hosted" if os.getenv("LLM_API_BASE") and os.getenv("LLM_MODEL") else "mock"
prefix = RUN_PREFIX or {"hosted": "hosted_" + os.getenv("LLM_MODEL", "model").split("/")[-1].lower().replace(".", ""),
                        "local": "local_qwen3vl2b", "mock": "mock"}[provider]
if LIMIT:
    prefix += f"_n{LIMIT}"
runs = {c: f"{prefix}_{c}" for c in contexts}
print("LLM:", provider, os.getenv("LLM_MODEL", "") if provider == "hosted" else "", "| runs:", runs)
"""

TEMPLATES_MD = r"""
## 1. The five templates (one agent, one prompt)

Within a context every strategy gets the same system prompt and the same evidence; only the instruction after the
question changes. Self-consistency is chain-of-thought sampled at temperature 0.7 with a majority vote.

- **none (the Week 1 baseline):** question + prompt only, no documents, no citations. The model answers from memory.
- **rag / bm25 / oracle:** the same five strategies with five evidence chunks (`[c1]`…`[c5]`) and citations.
"""

TEMPLATES = r"""
print("PROMPT ONLY (none) SYSTEM:\n" + pb.SYSTEM_PROMPT_ONLY + "\n")
for name in pb.STRATEGIES:
    print(f"===== none / {name}  (max tokens {pb.MAX_TOKENS[name]})\n{pb.SUFFIX_PROMPT_ONLY[name]}\n")
print("\nWITH EVIDENCE (rag, bm25, oracle) SYSTEM:\n" + pb.SYSTEM + "\n")
for name in pb.STRATEGIES:
    print(f"===== evidence / {name}  (max tokens {pb.MAX_TOKENS[name]})\n{pb.SUFFIX[name]}\n")
"""

RETRIEVAL_MD = r"""
## 2. Evidence per context (no model calls)

How often the five chunks the model sees hold the **gold document** and the **gold figure**. A prompt can only be
right when the evidence is: this is the ceiling for every strategy. `none` has no evidence by design (0 everywhere).
"""

RETRIEVAL = r"""
import pandas as pd
from IPython.display import Markdown, display

for c, run_id in runs.items():     # fetches only the questions not yet in retrieval.jsonl
    pb.main(["--context", c, "--run-id", run_id, "--retrieval-only"] + (["--limit", str(LIMIT)] if LIMIT else []))
measures = {c: pb.retrieval_measures(pb.OUT_ROOT / r) for c, r in runs.items()}
table = pd.DataFrame({c: {"questions with a document answer": m["retrievable_questions"],
                          "gold document in the 5 chunks": m["gold_doc_in_top5"],
                          "gold figure in the evidence": m["gold_value_in_evidence"]} for c, m in measures.items()})
display(table)
display(pd.DataFrame({c: {cat: f"{v['gold_doc']}/{v['questions']} doc, {v['gold_value']} value"
                          for cat, v in m["by_category"].items()} for c, m in measures.items()}))
"""

RUN_MD = r"""
## 3. Run every strategy on every evidence set

Resumable: a stopped run continues where it stopped (`responses.jsonl`). With `RUN_LLM = False` this only rescores.
"""

RUN = r"""
for c, run_id in runs.items():
    args = ["--context", c, "--run-id", run_id, "--strategies", *strategies]
    args += ["--limit", str(LIMIT)] if LIMIT else []
    if not RUN_LLM:
        if (pb.OUT_ROOT / run_id / "responses.jsonl").exists():
            pb.main(args + ["--report-only"])
        continue
    if provider == "mock":
        args += ["--llm-provider", "mock"]
    elif provider == "local":
        args += ["--local-llm"]
    print(f"\n######## {run_id}")
    pb.main(args)
"""

RESULTS_MD = r"""
## 4. Results

`none` is the Week 1 single-prompt attempt (prompt only). `rag` is the same prompts with the capstone's retrieval, so
the difference between `none` and `rag` is what RAG adds. `oracle` (the right evidence) is the ceiling and `bm25` is
naive retrieval, for reference.
"""

RESULTS = r"""
summaries = {c: json.loads((pb.OUT_ROOT / r / "summary.json").read_text(encoding="utf-8"))
             for c, r in runs.items() if (pb.OUT_ROOT / r / "summary.json").exists()}
pct = lambda x: None if x is None else round(100 * x)
display(Markdown("**Accuracy (%)**"))
display(pd.DataFrame({c: {st: pct(v["accuracy"]) for st, v in m["strategies"].items()} for c, m in summaries.items()}))
for c, m in summaries.items():
    display(Markdown(f"**{c}: by category and output reliability (%)**"))
    display(pd.DataFrame({st: {**{k: pct(x) for k, x in v["by_category"].items()},
                               "valid JSON": pct(v["schema_valid"]), "valid citations": pct(v["citation_valid_answerable"]),
                               "unsupported numbers": pct(v["unsupported_number_rate"]),
                               "calls/q": v["calls_per_question"], "tokens/q": round(v["tokens_per_question"])}
                          for st, v in m["strategies"].items()}))
if len(summaries) > 1:
    order = [runs[c] for c in ("none", "bm25", "rag", "oracle") if c in summaries]
    display(Markdown(pb.compare(order)))
"""

BREAKS_MD = r"""
## 5. How the small model breaks

Every incorrect answer has one failure label. Below: the count per strategy, then one verbatim answer per label
(`retrieval_miss` = the gold figure wasn't in the evidence; `copied_few_shot_example` = the worked example leaked into
the answer; `truncated_output` = ran out of tokens before the final JSON; see the write-up for every label).
"""

BREAKS = r"""
for c, r in runs.items():
    path = pb.OUT_ROOT / r / "scores.jsonl"
    if not path.exists():
        continue
    rows = pd.DataFrame(pb.read_jsonl(path))
    wrong = rows[~rows["correct"]]
    display(Markdown(f"### {c}: failure labels (incorrect answers)"))
    display(wrong.pivot_table(index="failure", columns="strategy", values="id", aggfunc="count", fill_value=0))
    items = {i["id"]: i for i in pb.read_jsonl(pb.EVAL_SET)}
    for label, group in wrong.groupby("failure"):
        row = group.iloc[0]
        print(f"--- {label} | {row['strategy']} | {row['id']}: {items[row['id']]['question']}\n"
              f"    gold:   {items[row['id']]['gold_answer'][:160]}\n    answer: {row['answer'][-300:]!r}\n")
"""

REPLAY_MD = r"""
## 6. What the capstone fixes

Two kinds of fix, measured separately:

1. **Retrieval** (`rag` vs `bm25` above): the planner's company × year filters and Chroma hybrid search put the right
   filing in front of the model, so `retrieval_miss` answers become correct ones.
2. **Checks before release** (below): every recorded wrong answer is replayed through the agent's deterministic
   layers, using the functions the LangGraph flow runs. A caught answer becomes a safe outcome (refused,
   not available, withheld, or sent to human review), not a correct one.
"""

REPLAY = r"""
replayed = [run for run in runs.values() if (pb.OUT_ROOT / run / "scores.jsonl").exists()]
if replayed:
    fr.main(replayed)
    display(Markdown((pb.OUT_ROOT / "failure_replay.md").read_text(encoding="utf-8")))
"""

CLOSE_MD = r"""
## 7. What is left

Failures neither retrieval nor the checks remove, and the agent feature that addresses each
(details and measured numbers: [docs/week1_failure_scenarios.md](../docs/week1_failure_scenarios.md)):

| Still breaks | Why | Capstone answer |
|---|---|---|
| Over-refusal (declines when the figure is in the evidence) | the model is cautious with unfamiliar table layouts | the evidence guard broadens the search once for a target with no evidence; the critic can send the answer back once |
| Wrong row or concept (e.g. dividends vs distributed earnings) | the figure is real, just the wrong line | statement-row reading in code (trend and sector agents), support check by the LLM judge |
| Arithmetic and rounding | the model rounds or mis-divides | `CALC[...]` → the calculator, operands must appear in the evidence |
| Chart-only values | the figure is printed only in a chart image | pipeline 1 transcribes chart pages (Qwen3-VL / paid model) into text chunks |
| Truncated reasoning | long chain-of-thought hits the token limit | the graph's answer prompt asks for a cited answer, not step-by-step reasoning |
| Fiscal years ending in January or February | "fiscal 2025" can mean the year ending early 2025 or early 2026 | the agent follows the filing's own label; the eval gold sometimes takes the other year |
"""


def build() -> dict:
    cells = [md(INTRO), code(SETTINGS), code(SETUP), md(TEMPLATES_MD), code(TEMPLATES), md(RETRIEVAL_MD),
             code(RETRIEVAL), md(RUN_MD), code(RUN), md(RESULTS_MD), code(RESULTS), md(BREAKS_MD), code(BREAKS),
             md(REPLAY_MD), code(REPLAY), md(CLOSE_MD)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT)
