"""Run the user-confirmed custom LLM-only classification prompt."""
from __future__ import annotations

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
import run_paper_norag_experiment as base  # noqa: E402


RUN = "PAPER_NORAG_PROMPT2"
REQUESTED_TOP_K = 1
PROMPT_ID = "user_confirmed_custom_norag_prompt"
VALID_LABELS = "[Normal, Depression, Anxiety, Bipolar]"

# Preserve the user's two-segment prompt exactly, including the two placeholders.
TEMPLATE = (
    "Classify the {text} into one of {valid_labels}. "
    "Please only output one of the following labels: {valid_labels}. "
)


def out_dir() -> Path:
    path = C.RUNS / RUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_test(out: Path) -> pd.DataFrame:
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    source = C.DATA_SPLITS / "test.csv"
    destination = split_dir / "test.csv"
    shutil.copy2(source, destination)
    return pd.read_csv(destination)


def render(text: str) -> str:
    return TEMPLATE.replace("{text}", text).replace("{valid_labels}", VALID_LABELS)


def write_manifest(out: Path, test: pd.DataFrame) -> None:
    C.save_json(out / "run_manifest.json", {
        "run": RUN,
        "experiment": "user_confirmed_custom_llm_only_prompt",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": base.R.git_revision(),
        "model": base.R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "prompt_source": "user-confirmed prompt",
        "prompt_template": TEMPLATE,
        "valid_labels_text": VALID_LABELS,
        "prompt_sha256": C.sha256_text(TEMPLATE),
        "prompt_id": PROMPT_ID,
        "requested_top_k": REQUESTED_TOP_K,
        "top_k": None,
        "retrieval_used": False,
        "tpe_used": False,
        "test": {
            "path": "splits/test.csv",
            "n": int(len(test)),
            "sha256": C.sha256_file(out / "splits" / "test.csv"),
            "label_counts": test["status"].value_counts().to_dict(),
        },
    })


def evaluate(out: Path, test: pd.DataFrame) -> list[dict]:
    path = out / "predictions_norag_prompt2.jsonl"
    done = {record["id"] for record in C.read_jsonl(path)}
    digest = base.R.probe_llm_identity().get("digest")
    for row in test.itertuples(index=False):
        if row.id in done:
            continue
        prompt = render(row.statement)
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = {
            "condition": "norag_prompt2",
            "condition_description": "LLM-only；使用者確認的兩段式 prompt；無 RAG、無 top_k",
            "id": row.id,
            "true_label_id": int(row.label_id),
            "true_label": row.status,
            "prompt_id": PROMPT_ID,
            "prompt_sha256": C.sha256_text(TEMPLATE),
            "prompt_template": TEMPLATE,
            "rendered_prompt": prompt,
            "requested_top_k": REQUESTED_TOP_K,
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
            "model_digest": digest,
        }
        C.append_jsonl(path, record)
        done.add(row.id)
        C.log(f"{len(done):,}/{len(test):,}，id={row.id}")
    records = list(C.read_jsonl(path))
    if len(records) != len(test) or len({r["id"] for r in records}) != len(records):
        C.die(f"預測筆數不完整：{len(records)} / {len(test)}")
    return records


def save_metrics(out: Path, records: list[dict]) -> dict:
    n = len(records)
    invalid = sum(bool(r["invalid"]) for r in records)
    correct = sum(bool(r["correct"]) for r in records)
    valid = [r for r in records if not r["invalid"]]
    valid_correct = sum(bool(r["correct"]) for r in valid)
    metrics = {
        "run": RUN,
        "condition": "norag_prompt2",
        "condition_description": "LLM-only；使用者確認的兩段式 prompt；無 RAG、無 top_k",
        "n": n,
        "correct_count": correct,
        "invalid_count": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "strict_accuracy": correct / n if n else 0.0,
        "valid_only_accuracy": valid_correct / len(valid) if valid else None,
        "prompt_id": PROMPT_ID,
        "prompt_sha256": C.sha256_text(TEMPLATE),
        "requested_top_k": REQUESTED_TOP_K,
        "top_k": None,
        "retrieval_used": False,
        "test_split_sha256": C.sha256_file(out / "splits" / "test.csv"),
    }
    C.save_json(out / "metrics_norag_prompt2.json", metrics)
    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    out = out_dir()
    test = setup_test(out)
    write_manifest(out, test)
    records = evaluate(out, test)
    metrics = save_metrics(out, records)
    C.save_json(out / "completion.json", {
        "run": RUN,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "n_test": len(test),
        "requested_top_k": REQUESTED_TOP_K,
        "retrieval_used": False,
    })
    C.log("實驗完成", f"strict_accuracy={metrics['strict_accuracy']:.4f}", f"invalid={metrics['invalid_count']}")


if __name__ == "__main__":
    main()
