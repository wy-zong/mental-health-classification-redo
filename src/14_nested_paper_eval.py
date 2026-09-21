"""Evaluate nested incremental corpora on the nested 1,998-row test split.

This runner is isolated from the historical PAPER_* runs. It uses the
report prompts, fixed top_k=1 for RAG, the nested test split, and the nested
incremental noaug/aug indexes. Prediction and retrieval records are resumable.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
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
MODEL_DIGEST = R.probe_llm_identity().get("digest")

RUN = "NESTED_70_10_20_INCREMENTAL"
TOP_K = 1
CORPORA = ("noaug", "aug")
REPORT_DOC_NAME = "CHBR-D-26-01681_R1.docx"
REPORT_VALID_LABELS = "[Normal, Depression, Anxiety, Bipolar]"
REPORT_NORAG_TEMPLATE = (
    "Classify the text into one of {valid_labels}. "
    "Now classify this text: {text}. "
    "Please only output one of the following labels: {valid_labels}. "
    "Do not output anything else."
)
REPORT_RAG_TEMPLATE = (
    "Classify the text into one of {valid_labels}. "
    "Below is some related reference content that might help you classify the new text: "
    "{reference_context}. "
    "Now classify this text: {text}. "
    "Please only output one of the following labels: {valid_labels}. "
    "Do not output anything else."
)


def run_dir() -> Path:
    path = C.RUNS / RUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_test(out: Path) -> pd.DataFrame:
    path = out / "splits" / "test.csv"
    if not path.exists():
        C.die(f"missing nested test split: {path}")
    test = pd.read_csv(path)
    required = {"id", "statement", "label_id", "status"}
    missing = required - set(test.columns)
    if missing:
        C.die(f"nested test missing columns: {sorted(missing)}")
    if len(test) != 1998:
        C.die(f"nested test row count is {len(test)}, expected 1998")
    return test


def load_corpus_meta(out: Path, name: str) -> dict:
    meta_path = out / "corpus" / f"{name}_meta.json"
    meta = C.load_json(meta_path)
    expected = {"noaug": 6994, "aug": 18095}[name]
    if meta.get("n_docs") != expected or meta.get("top_k") != TOP_K:
        C.die(f"{name} corpus metadata failed nested validation")
    return {
        **meta,
        "meta_sha256": C.sha256_file(meta_path),
        "index_sha256_actual": C.sha256_file(out / "corpus" / f"{name}.index"),
        "docs_sha256_actual": C.sha256_file(out / "corpus" / f"{name}_docs.json"),
    }


def render(template: str, text: str, reference_context: str | None = None) -> str:
    prompt = template.replace("{valid_labels}", REPORT_VALID_LABELS).replace("{text}", text)
    if reference_context is not None:
        prompt = prompt.replace("{reference_context}", reference_context)
    return prompt


def base_record(row, condition: str, description: str, prompt_id: str,
                template: str, prompt: str, response: dict, pred, reason: str) -> dict:
    return {
        "run": RUN,
        "condition": condition,
        "condition_description": description,
        "id": row.id,
        "true_label_id": int(row.label_id),
        "true_label": row.status,
        "prompt_id": prompt_id,
        "prompt_sha256": C.sha256_text(template),
        "prompt_template": template,
        "rendered_prompt": prompt,
        "top_k": None,
        "retrieval_used": False,
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
        "model_digest": MODEL_DIGEST,
    }


def check_complete(path: Path, test: pd.DataFrame) -> list[dict]:
    records = list(C.read_jsonl(path))
    ids = [str(r["id"]) for r in records]
    expected = [str(x) for x in test["id"]]
    if len(records) != len(test) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        C.die(f"incomplete prediction file: {path.name} {len(records)} / {len(test)}")
    return records


def evaluate_norag(out: Path, test: pd.DataFrame) -> list[dict]:
    path = out / "predictions_nested_norag_paper_exact.jsonl"
    done = {str(r["id"]) for r in C.read_jsonl(path)}
    for row in test.itertuples(index=False):
        if str(row.id) in done:
            continue
        prompt = render(REPORT_NORAG_TEMPLATE, str(row.statement))
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = base_record(
            row,
            "nested_norag_paper_exact",
            "nested test LLM-only; original Table VIII classification prompt; no RAG",
            "paper_table_viii_norag_exact",
            REPORT_NORAG_TEMPLATE,
            prompt,
            response,
            pred,
            reason,
        )
        C.append_jsonl(path, record)
        done.add(str(row.id))
        C.log(f"norag: {len(done):,}/{len(test):,}, id={row.id}")
    records = check_complete(path, test)
    save_metrics(out, records, "nested_norag_paper_exact", "paper_table_viii_norag_exact", None)
    return records


def evaluate_rag_one(out: Path, test: pd.DataFrame, corpus_name: str) -> list[dict]:
    rag = R.RebuildRagIndex(RUN, corpus_name)
    pred_path = out / f"predictions_nested_rag_{corpus_name}_paper_topk1.jsonl"
    retrieval_path = out / f"nested_{corpus_name}_retrieval_audit.jsonl"
    done = {str(r["id"]) for r in C.read_jsonl(pred_path)}
    condition = f"nested_rag_{corpus_name}_paper_topk1"
    description = f"nested {corpus_name} RAG; original Table VIII prompt; top_k=1"
    for row in test.itertuples(index=False):
        if str(row.id) in done:
            continue
        docs, ids, sims = rag.search(str(row.statement))
        if len(docs) != TOP_K or len(ids) != TOP_K or len(sims) != TOP_K:
            C.die(f"{corpus_name} retrieval is not top_k=1: id={row.id}")
        prompt = render(REPORT_RAG_TEMPLATE, str(row.statement), docs[0])
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = base_record(
            row,
            condition,
            description,
            "paper_table_viii_rag",
            REPORT_RAG_TEMPLATE,
            prompt,
            response,
            pred,
            reason,
        )
        record.update({
            "top_k": TOP_K,
            "retrieval_used": True,
            "retrieved_ids": ids,
            "retrieved_docs": docs,
            "retrieved_labels": [R.label_from_corpus_doc(doc) for doc in docs],
            "retrieved_similarities": sims,
        })
        C.append_jsonl(pred_path, record)
        C.append_jsonl(retrieval_path, {
            "run": RUN,
            "condition": condition,
            "id": row.id,
            "retrieved_ids": ids,
            "retrieved_docs": docs,
            "retrieved_labels": record["retrieved_labels"],
            "similarities": sims,
            "top_k": TOP_K,
        })
        done.add(str(row.id))
        C.log(f"{corpus_name}: {len(done):,}/{len(test):,}, id={row.id}")
    records = check_complete(pred_path, test)
    save_metrics(out, records, condition, "paper_table_viii_rag", TOP_K)
    return records


def save_metrics(out: Path, records: list[dict], condition: str,
                 prompt_id: str, top_k: int | None) -> dict:
    n = len(records)
    invalid = sum(bool(r["invalid"]) for r in records)
    correct = sum(bool(r["correct"]) for r in records)
    valid = [r for r in records if not r["invalid"]]
    valid_correct = sum(bool(r["correct"]) for r in valid)
    template = REPORT_RAG_TEMPLATE if top_k is not None else REPORT_NORAG_TEMPLATE
    result = {
        "run": RUN,
        "condition": condition,
        "n": n,
        "correct_count": correct,
        "invalid_count": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "strict_accuracy": correct / n if n else 0.0,
        "valid_only_accuracy": valid_correct / len(valid) if valid else None,
        "prompt_id": prompt_id,
        "prompt_sha256": C.sha256_text(template),
        "top_k": top_k,
        "test_split_sha256": C.sha256_file(out / "splits" / "test.csv"),
        "model_digest": MODEL_DIGEST,
    }
    C.save_json(out / f"metrics_{condition}.json", result)
    return result


def write_manifest(out: Path, test: pd.DataFrame) -> None:
    C.save_json(out / "nested_eval_manifest.json", {
        "run": RUN,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": R.git_revision(),
        "model": R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "report_document": REPORT_DOC_NAME,
        "test": {
            "path": "splits/test.csv",
            "n": int(len(test)),
            "sha256": C.sha256_file(out / "splits" / "test.csv"),
            "label_counts": test["status"].value_counts().to_dict(),
        },
        "top_k": TOP_K,
        "prompts": {
            "norag": {
                "id": "paper_table_viii_norag_exact",
                "sha256": C.sha256_text(REPORT_NORAG_TEMPLATE),
                "template": REPORT_NORAG_TEMPLATE,
            },
            "rag": {
                "id": "paper_table_viii_rag",
                "sha256": C.sha256_text(REPORT_RAG_TEMPLATE),
                "template": REPORT_RAG_TEMPLATE,
            },
        },
        "corpora": {name: load_corpus_meta(out, name) for name in CORPORA},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["norag", "rag", "all"], default="all")
    args = parser.parse_args()
    out = run_dir()
    test = load_test(out)
    write_manifest(out, test)
    if args.stage in {"norag", "all"}:
        evaluate_norag(out, test)
    if args.stage in {"rag", "all"}:
        for name in CORPORA:
            evaluate_rag_one(out, test, name)
    rows = []
    for path in sorted(out.glob("metrics_nested_*.json")):
        rows.append(C.load_json(path))
    pd.DataFrame(rows).to_csv(out / "nested_eval_summary.csv", index=False, encoding="utf-8-sig")
    C.save_json(out / "nested_eval_completion.json", {
        "run": RUN,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "conditions": [row["condition"] for row in rows],
        "top_k": TOP_K,
        "n_test": len(test),
    })
    C.log("nested test evaluation complete")
    for row in rows:
        C.log(row["condition"], f"strict_accuracy={row['strict_accuracy']:.4f}", f"invalid={row['invalid_count']}")


if __name__ == "__main__":
    main()
