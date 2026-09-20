"""Run the paper's original RAG classification prompt against existing corpora.

This is intentionally a small, isolated runner for the first follow-up
experiment.  It does not rebuild splits, augmentation, embeddings, or prompt
optimization.  It reuses the existing RAG indexes and evaluates both corpus
variants with the report's original RAG prompt and top_k=1.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common as C  # noqa: E402


def load_rebuild_module():
    path = SRC / "12_rebuild_experiment.py"
    spec = importlib.util.spec_from_file_location("rebuild_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_rebuild_module()

RUN = "PAPER_RAG_TOPK1_EXISTING"
TOP_K = 1
CORPORA = ("noaug", "aug")
REPORT_PATH = Path(
    r"C:\Users\wuc120\OneDrive\Lab Cooperative Workspace\Paper\彥宗論文Revise\R1\CHBR-D-26-01681_R1.docx"
)

# This is the runtime RAG prompt quoted in the report's reproducibility text
# and Table VIII prompt description.  Keep the single-line construction and
# label spelling exactly as reported.
REPORT_RAG_TEMPLATE = (
    "Classify the text into one of {valid_labels}. "
    "Below is some related reference content that might help you classify the new text: "
    "{reference_context}. "
    "Now classify this text: {text}. "
    "Please only output one of the following labels: {valid_labels}. "
    "Do not output anything else."
)
REPORT_VALID_LABELS = "[Normal, Depression, Anxiety, Bipolar]"


def run_dir() -> Path:
    path = C.RUNS / RUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_inputs(out: Path) -> None:
    """Copy immutable input snapshots into this run for reproducibility."""
    splits = out / "splits"
    corpus = out / "corpus"
    splits.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)

    for name in ("train.csv", "val_search.csv", "val_confirm.csv", "test.csv", "split_audit.json"):
        source = C.DATA_SPLITS / name
        if source.exists():
            shutil.copy2(source, splits / name)
    for name in ("noaug", "aug"):
        for suffix in (".index", "_docs.json", "_meta.json"):
            source = C.ROOT / "data" / "corpus" / f"{name}{suffix}"
            if not source.exists():
                C.die(f"找不到既有 RAG 輸入：{source}")
            shutil.copy2(source, corpus / source.name)


def load_test(out: Path) -> pd.DataFrame:
    path = out / "splits" / "test.csv"
    if not path.exists():
        C.die(f"找不到 test split：{path}")
    test = pd.read_csv(path)
    required = {"id", "statement", "label_id", "status"}
    missing = required - set(test.columns)
    if missing:
        C.die(f"test split 缺少欄位：{sorted(missing)}")
    return test


def render_prompt(reference_context: str, text: str) -> str:
    return (
        REPORT_RAG_TEMPLATE
        .replace("{valid_labels}", REPORT_VALID_LABELS)
        .replace("{reference_context}", reference_context)
        .replace("{text}", text)
    )


def corpus_meta(out: Path, name: str) -> dict:
    path = out / "corpus" / f"{name}_meta.json"
    meta = C.load_json(path)
    index_path = out / "corpus" / f"{name}.index"
    docs_path = out / "corpus" / f"{name}_docs.json"
    return {
        **meta,
        "meta_sha256": C.sha256_file(path),
        "index_sha256_actual": C.sha256_file(index_path),
        "docs_sha256_actual": C.sha256_file(docs_path),
    }


def write_manifest(out: Path, test: pd.DataFrame) -> None:
    manifest = {
        "run": RUN,
        "experiment": "paper_original_rag_prompt_topk1_existing_corpora",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": R.git_revision(),
        "model": R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "report_path": str(REPORT_PATH),
        "report_path_exists": REPORT_PATH.exists(),
        "prompt_source": "report runtime RAG prompt / Table VIII original prompt",
        "prompt_template": REPORT_RAG_TEMPLATE,
        "valid_labels_text": REPORT_VALID_LABELS,
        "prompt_sha256": C.sha256_text(REPORT_RAG_TEMPLATE),
        "top_k": TOP_K,
        "embedding_model": R.EMBED_MODEL,
        "test": {
            "path": "splits/test.csv",
            "n": int(len(test)),
            "sha256": C.sha256_file(out / "splits" / "test.csv"),
            "label_counts": test["status"].value_counts().to_dict(),
        },
        "corpora": {name: corpus_meta(out, name) for name in CORPORA},
        "no_tpe": True,
        "no_augmentation_generation": True,
    }
    C.save_json(out / "run_manifest.json", manifest)


def evaluate_one(out: Path, test: pd.DataFrame, corpus_name: str) -> list[dict]:
    rag = R.RebuildRagIndex(RUN, corpus_name)
    # RebuildRagIndex resolves its run directory from common.RUNS, which is
    # the same run directory used here.  Its search method asserts TOP_K == 1.
    pred_path = out / f"predictions_rag_{corpus_name}_paper_topk1.jsonl"
    retrieval_path = out / f"{corpus_name}_retrieval_audit.jsonl"
    done = {rec["id"] for rec in C.read_jsonl(pred_path)}
    model_digest = R.probe_llm_identity().get("digest")

    for row in test.itertuples(index=False):
        if row.id in done:
            continue
        docs, ids, sims = rag.search(row.statement)
        if len(docs) != 1 or len(ids) != 1 or len(sims) != 1:
            C.die(f"{corpus_name} 檢索結果不是 top_k=1：id={row.id}")
        prompt = render_prompt(docs[0], row.statement)
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = {
            "condition": f"rag_{corpus_name}_paper_topk1",
            "condition_description": (
                f"{corpus_name} RAG；論文 Table VIII 原始分類 prompt；top_k=1"
            ),
            "id": row.id,
            "true_label_id": int(row.label_id),
            "true_label": row.status,
            "prompt_id": "paper_table_viii_rag",
            "prompt_sha256": C.sha256_text(REPORT_RAG_TEMPLATE),
            "prompt_template": REPORT_RAG_TEMPLATE,
            "rendered_prompt": prompt,
            "retrieved_ids": ids,
            "retrieved_docs": docs,
            "retrieved_labels": [R.label_from_corpus_doc(doc) for doc in docs],
            "retrieved_similarities": sims,
            "top_k": TOP_K,
            "raw_response": response["raw_response"],
            "pred_label_id": pred,
            "pred_label": C.ID_TO_LABEL.get(pred) if pred is not None else None,
            "parse_reason": reason,
            "invalid_reason": reason if pred is None else None,
            "invalid": pred is None,
            "correct": bool(pred is not None and pred == int(row.label_id)),
            "elapsed_s": round(response["elapsed_s"], 3),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "token_count": (response.get("prompt_eval_count") or 0)
            + (response.get("eval_count") or 0),
            "model_digest": model_digest,
        }
        C.append_jsonl(pred_path, record)
        C.append_jsonl(
            retrieval_path,
            {
                "condition": record["condition"],
                "id": row.id,
                "retrieved_ids": ids,
                "retrieved_docs": docs,
                "retrieved_labels": record["retrieved_labels"],
                "similarities": sims,
                "top_k": TOP_K,
            },
        )
        C.log(f"{corpus_name}: {len(done) + 1:,}/{len(test):,}，id={row.id}")
        done.add(row.id)

    records = list(C.read_jsonl(pred_path))
    if len(records) != len(test) or len({r["id"] for r in records}) != len(records):
        C.die(f"{corpus_name} 預測筆數不完整：{len(records)} / {len(test)}")
    return records


def metrics(out: Path, records: list[dict], corpus_name: str) -> dict:
    y_true = [int(r["true_label_id"]) for r in records]
    y_pred = [r["pred_label_id"] for r in records]
    invalid = sum(pred is None for pred in y_pred)
    strict_correct = sum(pred is not None and pred == true for true, pred in zip(y_true, y_pred))
    valid_pairs = [(true, pred) for true, pred in zip(y_true, y_pred) if pred is not None]
    valid_correct = sum(true == pred for true, pred in valid_pairs)
    result = {
        "run": RUN,
        "condition": f"rag_{corpus_name}_paper_topk1",
        "condition_description": f"{corpus_name} RAG；論文 Table VIII 原始分類 prompt；top_k=1",
        "n": len(records),
        "invalid_count": invalid,
        "invalid_rate": invalid / len(records) if records else 0.0,
        "strict_accuracy": strict_correct / len(records) if records else 0.0,
        "valid_only_accuracy": valid_correct / len(valid_pairs) if valid_pairs else None,
        "correct_count": strict_correct,
        "prompt_id": "paper_table_viii_rag",
        "prompt_sha256": C.sha256_text(REPORT_RAG_TEMPLATE),
        "top_k": TOP_K,
        "test_split_sha256": C.sha256_file(out / "splits" / "test.csv"),
        "corpus_meta_sha256": C.sha256_file(out / "corpus" / f"{corpus_name}_meta.json"),
    }
    C.save_json(out / f"metrics_rag_{corpus_name}_paper_topk1.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    args = parser.parse_args()
    if args.run != RUN:
        C.die(f"此 runner 固定使用 run={RUN}，收到 {args.run}")

    out = run_dir()
    setup_inputs(out)
    test = load_test(out)
    write_manifest(out, test)
    rows = []
    for corpus_name in CORPORA:
        records = evaluate_one(out, test, corpus_name)
        rows.append(metrics(out, records, corpus_name))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    C.save_json(
        out / "completion.json",
        {
            "run": RUN,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "conditions": [row["condition"] for row in rows],
            "top_k": TOP_K,
            "n_test_each": int(len(test)),
        },
    )
    C.log("實驗完成")
    for row in rows:
        C.log(
            row["condition"],
            f"strict_accuracy={row['strict_accuracy']:.4f}",
            f"invalid={row['invalid_count']}",
        )


if __name__ == "__main__":
    main()
