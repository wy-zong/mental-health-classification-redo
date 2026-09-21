"""Run the user-specified prompt experiment on the nested incremental RAG.

This is a separate, resumable experiment under the nested run.  It keeps the
historical PAPER_* evaluation rules: fixed Ollama generation settings, the
shared strict label parser, invalid responses counted as errors, complete raw
responses, retrieval records, metrics, confusion matrices, and a paired table.
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
RUN = "NESTED_70_10_20_INCREMENTAL"
BASE_OUT = C.RUNS / RUN
OUT = BASE_OUT / "CUSTOM_PROMPT_EVAL"
TOP_K = 1
MODEL_DIGEST = R.probe_llm_identity().get("digest")
VALID_LABELS = "[Normal, Depression, Anxiety, Bipolar]"

# The first template is intentionally preserved exactly as supplied by the
# user, including the trailing "r" after "else.r".
PROMPTS = {
    "optimized_rag": {
        "prompt_id": "user_optimized_rag_prompt_exact",
        "template": (
            "Classify the text into one of [Normal, Depression, Anxiety, Bipolar].\n"
            "Below is some related reference content that might help you classify the new text: "
            "{reference_context}.\n"
            "Now classify this text: {text}.\n"
            "Please only output one of the following labels: [Normal, Depression, Anxiety, Bipolar]. "
            "Do not output anything else.r"
        ),
        "uses_rag": True,
        "description": "RAG；使用者指定的最佳化 prompt；top_k=1",
    },
    "base_rag": {
        "prompt_id": "user_base_rag_prompt",
        "template": (
            "Classify the text into one of [Normal, Depression, Anxiety, Bipolar].\n"
            "You can refer to the following reference content for categorization: {reference_context}.\n"
            "Now classify this text: {text}.\n"
            "output only one of the following labels: [Normal, Depression, Anxiety, Bipolar]."
        ),
        "uses_rag": True,
        "description": "RAG；使用者指定的 base prompt；top_k=1",
    },
    "base_norag": {
        "prompt_id": "user_base_llm_only_prompt",
        "template": (
            "Classify the text into one of [Normal, Depression, Anxiety, Bipolar].\n"
            "Now classify this text: {text}.\n"
            "output only one of the following labels: [Normal, Depression, Anxiety, Bipolar]."
        ),
        "uses_rag": False,
        "description": "LLM-only；使用者指定的 base prompt；無 RAG",
    },
}

CONDITIONS = (
    ("rag_noaug_optimized", "optimized_rag", "noaug"),
    ("rag_aug_optimized", "optimized_rag", "aug"),
    ("rag_noaug_base", "base_rag", "noaug"),
    ("rag_aug_base", "base_rag", "aug"),
    ("norag_base", "base_norag", None),
)


def render(template: str, text: str, reference_context: str | None = None) -> str:
    prompt = template.replace("{text}", text)
    if reference_context is not None:
        prompt = prompt.replace("{reference_context}", reference_context)
    if "{" in prompt or "}" in prompt:
        C.die("rendered prompt contains an unreplaced placeholder")
    return prompt


def load_test() -> pd.DataFrame:
    path = BASE_OUT / "splits" / "test.csv"
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


def load_rag(corpus_name: str):
    rag = R.RebuildRagIndex(RUN, corpus_name)
    if rag.meta.get("top_k") != TOP_K:
        C.die(f"{corpus_name} corpus top_k is not {TOP_K}")
    return rag


def record_for(row, condition: str, prompt_spec: dict, prompt: str,
               response: dict, pred: int | None, reason: str,
               corpus_name: str | None, docs, ids, sims) -> dict:
    return {
        "run": RUN,
        "experiment": "CUSTOM_PROMPT_EVAL",
        "condition": condition,
        "condition_description": prompt_spec["description"],
        "id": row.id,
        "true_label_id": int(row.label_id),
        "true_label": row.status,
        "prompt_id": prompt_spec["prompt_id"],
        "prompt_sha256": C.sha256_text(prompt_spec["template"]),
        "prompt_template": prompt_spec["template"],
        "rendered_prompt": prompt,
        "corpus": corpus_name,
        "top_k": TOP_K if corpus_name else None,
        "retrieval_used": bool(corpus_name),
        "retrieved_ids": ids,
        "retrieved_docs": docs,
        "retrieved_labels": [R.label_from_corpus_doc(doc) for doc in docs] if docs else None,
        "retrieved_similarities": sims,
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


def evaluate_condition(test: pd.DataFrame, condition: str, prompt_name: str,
                       corpus_name: str | None, rag_cache: dict) -> list[dict]:
    spec = PROMPTS[prompt_name]
    path = OUT / f"predictions_{condition}.jsonl"
    done = {str(r["id"]) for r in C.read_jsonl(path)}
    rag = None
    if corpus_name:
        rag = rag_cache.setdefault(corpus_name, load_rag(corpus_name))

    for row in test.itertuples(index=False):
        if str(row.id) in done:
            continue
        docs = ids = sims = None
        reference_context = None
        if rag:
            docs, ids, sims = rag.search(str(row.statement))
            if len(docs) != TOP_K or len(ids) != TOP_K or len(sims) != TOP_K:
                C.die(f"{corpus_name} retrieval is not top_k=1: id={row.id}")
            reference_context = docs[0]
        prompt = render(spec["template"], str(row.statement), reference_context)
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = record_for(
            row, condition, spec, prompt, response, pred, reason,
            corpus_name, docs, ids, sims,
        )
        C.append_jsonl(path, record)
        done.add(str(row.id))
        C.log(f"{condition}: {len(done):,}/{len(test):,}, id={row.id}")

    records = check_complete(path, test)
    if corpus_name:
        audit_path = OUT / f"retrieval_audit_{condition}.jsonl"
        if audit_path.exists():
            audit_path.unlink()
        for rec in records:
            C.append_jsonl(audit_path, {
                "run": RUN,
                "experiment": "CUSTOM_PROMPT_EVAL",
                "condition": condition,
                "id": rec["id"],
                "retrieved_ids": rec.get("retrieved_ids"),
                "retrieved_docs": rec.get("retrieved_docs"),
                "retrieved_labels": rec.get("retrieved_labels"),
                "similarities": rec.get("retrieved_similarities"),
                "top_k": rec.get("top_k"),
            })
    save_metrics(condition, records, spec, corpus_name)
    save_confusion(condition, records)
    return records


def save_metrics(condition: str, records: list[dict], spec: dict,
                 corpus_name: str | None) -> dict:
    n = len(records)
    invalid = sum(bool(r["invalid"]) for r in records)
    correct = sum(bool(r["correct"]) for r in records)
    valid = [r for r in records if not r["invalid"]]
    valid_correct = sum(bool(r["correct"]) for r in valid)
    result = {
        "run": RUN,
        "experiment": "CUSTOM_PROMPT_EVAL",
        "condition": condition,
        "corpus": corpus_name,
        "n": n,
        "correct_count": correct,
        "invalid_count": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "strict_accuracy": correct / n if n else 0.0,
        "valid_only_accuracy": valid_correct / len(valid) if valid else None,
        "prompt_id": spec["prompt_id"],
        "prompt_sha256": C.sha256_text(spec["template"]),
        "prompt_template": spec["template"],
        "top_k": TOP_K if corpus_name else None,
        "test_split_sha256": C.sha256_file(BASE_OUT / "splits" / "test.csv"),
        "model_digest": MODEL_DIGEST,
    }
    C.save_json(OUT / f"metrics_{condition}.json", result)
    return result


def save_confusion(condition: str, records: list[dict]) -> None:
    labels = ["Normal", "Depression", "Anxiety", "Bipolar", "INVALID"]
    rows = []
    for rec in records:
        rows.append({
            "true_label": rec["true_label"],
            "pred_label": rec["pred_label"] if rec["pred_label"] else "INVALID",
        })
    table = pd.crosstab(
        pd.Series([r["true_label"] for r in rows], name="true_label"),
        pd.Series([r["pred_label"] for r in rows], name="pred_label"),
    ).reindex(index=labels[:4], columns=labels, fill_value=0)
    table.to_csv(OUT / f"confusion_{condition}.csv", encoding="utf-8-sig")


def write_manifest(test: pd.DataFrame) -> None:
    corpora = {}
    for name in ("noaug", "aug"):
        meta_path = BASE_OUT / "corpus" / f"{name}_meta.json"
        corpora[name] = {
            "meta": C.load_json(meta_path),
            "meta_sha256": C.sha256_file(meta_path),
            "index_sha256_actual": C.sha256_file(BASE_OUT / "corpus" / f"{name}.index"),
            "docs_sha256_actual": C.sha256_file(BASE_OUT / "corpus" / f"{name}_docs.json"),
        }
    C.save_json(OUT / "custom_prompt_eval_manifest.json", {
        "run": RUN,
        "experiment": "CUSTOM_PROMPT_EVAL",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": R.git_revision(),
        "model": R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "test": {
            "path": "../splits/test.csv",
            "n": int(len(test)),
            "sha256": C.sha256_file(BASE_OUT / "splits" / "test.csv"),
            "label_counts": test["status"].value_counts().to_dict(),
        },
        "top_k": TOP_K,
        "prompt_source": "user-confirmed prompts in the current request",
        "prompts": {
            name: {
                "prompt_id": spec["prompt_id"],
                "sha256": C.sha256_text(spec["template"]),
                "template": spec["template"],
                "uses_rag": spec["uses_rag"],
            }
            for name, spec in PROMPTS.items()
        },
        "conditions": [
            {"condition": c, "prompt": p, "corpus": corpus}
            for c, p, corpus in CONDITIONS
        ],
        "corpora": corpora,
        "tpe": False,
    })


def save_paired(records_by_condition: dict[str, list[dict]]) -> None:
    by_id = {}
    for condition, records in records_by_condition.items():
        for rec in records:
            row = by_id.setdefault(str(rec["id"]), {
                "id": rec["id"], "true_label": rec["true_label"],
            })
            row[f"{condition}_pred"] = rec["pred_label"] or "INVALID"
            row[f"{condition}_invalid"] = rec["invalid"]
            row[f"{condition}_correct"] = rec["correct"]
    pd.DataFrame(sorted(by_id.values(), key=lambda x: str(x["id"]))).to_csv(
        OUT / "paired_comparison.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["all"] + [c[0] for c in CONDITIONS], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    test = load_test()
    write_manifest(test)
    selected = CONDITIONS if args.condition == "all" else tuple(
        item for item in CONDITIONS if item[0] == args.condition
    )
    rag_cache = {}
    records_by_condition = {}
    for condition, prompt_name, corpus_name in selected:
        records_by_condition[condition] = evaluate_condition(
            test, condition, prompt_name, corpus_name, rag_cache
        )

    all_metrics = []
    for condition, prompt_name, corpus_name in CONDITIONS:
        metric_path = OUT / f"metrics_{condition}.json"
        if metric_path.exists():
            all_metrics.append(C.load_json(metric_path))
    pd.DataFrame(all_metrics).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    available = {}
    for condition, _, _ in CONDITIONS:
        path = OUT / f"predictions_{condition}.jsonl"
        if path.exists():
            available[condition] = check_complete(path, test)
    if len(available) == len(CONDITIONS):
        save_paired(available)
        C.save_json(OUT / "completion.json", {
            "run": RUN,
            "experiment": "CUSTOM_PROMPT_EVAL",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "conditions": [c[0] for c in CONDITIONS],
            "top_k": TOP_K,
            "n_test": len(test),
        })
        C.log("custom prompt experiment complete")
        for metric in all_metrics:
            C.log(metric["condition"], f"strict_accuracy={metric['strict_accuracy']:.4f}",
                  f"invalid={metric['invalid_count']}")


if __name__ == "__main__":
    main()
