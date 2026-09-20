"""Run the report's original LLM-only classification prompt."""
from __future__ import annotations

import argparse
import importlib.util
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
RUN = "PAPER_NORAG_EXACT"
REQUESTED_TOP_K = 1
REPORT_PATH = Path(
    r"C:\Users\wuc120\OneDrive\Lab Cooperative Workspace\Paper\彥宗論文Revise\R1\CHBR-D-26-01681_R1.docx"
)
REPORT_VALID_LABELS = "[Normal, Depression, Anxiety, Bipolar]"
REPORT_NORAG_TEMPLATE = (
    "Classify the text into one of {valid_labels}. "
    "Now classify this text: {text}. "
    "Please only output one of the following labels: {valid_labels}. "
    "Do not output anything else."
)


def run_dir() -> Path:
    path = C.RUNS / RUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_test(out: Path) -> pd.DataFrame:
    splits = out / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    source = C.DATA_SPLITS / "test.csv"
    if not source.exists():
        C.die(f"找不到 test split：{source}")
    destination = splits / "test.csv"
    shutil.copy2(source, destination)
    test = pd.read_csv(destination)
    required = {"id", "statement", "label_id", "status"}
    missing = required - set(test.columns)
    if missing:
        C.die(f"test split 缺少欄位：{sorted(missing)}")
    return test


def render_prompt(text: str) -> str:
    return (
        REPORT_NORAG_TEMPLATE
        .replace("{valid_labels}", REPORT_VALID_LABELS)
        .replace("{text}", text)
    )


def write_manifest(out: Path, test: pd.DataFrame) -> None:
    C.save_json(out / "run_manifest.json", {
        "run": RUN,
        "experiment": "paper_original_llm_only_prompt",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": R.git_revision(),
        "model": R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "report_path": str(REPORT_PATH),
        "report_path_exists": REPORT_PATH.exists(),
        "prompt_source": "report runtime Llama-only prompt / Table VIII original prompt",
        "prompt_template": REPORT_NORAG_TEMPLATE,
        "valid_labels_text": REPORT_VALID_LABELS,
        "prompt_sha256": C.sha256_text(REPORT_NORAG_TEMPLATE),
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
    path = out / "predictions_norag_paper_exact.jsonl"
    done = {record["id"] for record in C.read_jsonl(path)}
    model_digest = R.probe_llm_identity().get("digest")
    for row in test.itertuples(index=False):
        if row.id in done:
            continue
        prompt = render_prompt(row.statement)
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        record = {
            "condition": "norag_paper_exact",
            "condition_description": "LLM-only；論文 Table VIII 原始分類 prompt；無 RAG、無 top_k",
            "id": row.id,
            "true_label_id": int(row.label_id),
            "true_label": row.status,
            "prompt_id": "paper_table_viii_norag_exact",
            "prompt_sha256": C.sha256_text(REPORT_NORAG_TEMPLATE),
            "prompt_template": REPORT_NORAG_TEMPLATE,
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
            "model_digest": model_digest,
        }
        C.append_jsonl(path, record)
        C.log(f"{len(done) + 1:,}/{len(test):,}，id={row.id}")
        done.add(row.id)
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
        "condition": "norag_paper_exact",
        "condition_description": "LLM-only；論文 Table VIII 原始分類 prompt；無 RAG、無 top_k",
        "n": n,
        "correct_count": correct,
        "invalid_count": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "strict_accuracy": correct / n if n else 0.0,
        "valid_only_accuracy": valid_correct / len(valid) if valid else None,
        "prompt_id": "paper_table_viii_norag_exact",
        "prompt_sha256": C.sha256_text(REPORT_NORAG_TEMPLATE),
        "requested_top_k": REQUESTED_TOP_K,
        "top_k": None,
        "retrieval_used": False,
        "test_split_sha256": C.sha256_file(out / "splits" / "test.csv"),
    }
    C.save_json(out / "metrics_norag_paper_exact.json", metrics)
    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    args = parser.parse_args()
    if args.run != RUN:
        C.die(f"此 runner 固定使用 run={RUN}，收到 {args.run}")
    out = run_dir()
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
