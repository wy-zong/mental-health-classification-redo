"""Plot confusion matrices for the PAPER_* experiments.

This is intentionally a standalone plotting program.  It reads the immutable
prediction JSONL files produced by the experiment runners and does not call the
LLM, rebuild the RAG index, or recompute predictions.

The plotting code follows the original project's ``prompt最佳化.py`` function
``evaluate_confusion_matrix``:

* ``ConfusionMatrixDisplay``;
* ``cmap="Blues"``;
* integer cell values;
* 8 x 6 inch figure;
* ``INVALID`` as an additional predicted-label category;
* 45-degree x-axis tick labels.

The original function calls ``plt.show()``.  Batch generation defaults to a
non-interactive backend so that all figures can be produced reproducibly; use
``--show`` when an interactive display is desired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


VALID_LABELS = ["Normal", "Depression", "Anxiety", "Bipolar"]
PLOT_LABELS = VALID_LABELS + ["INVALID"]


EXPERIMENTS = [
    {
        "key": "paper_norag_exact",
        "name": "PAPER_NORAG_EXACT",
        "prediction_file": Path("runs/PAPER_NORAG_EXACT/predictions_norag_paper_exact.jsonl"),
        "title": "PAPER_NORAG_EXACT - Table VIII prompt (LLM-only)",
    },
    {
        "key": "paper_norag_prompt2",
        "name": "PAPER_NORAG_PROMPT2",
        "prediction_file": Path("runs/PAPER_NORAG_PROMPT2/predictions_norag_prompt2.jsonl"),
        "title": "PAPER_NORAG_PROMPT2 - custom prompt (LLM-only)",
    },
    {
        "key": "paper_rag_noaug",
        "name": "PAPER_RAG_TOPK1_EXISTING/noaug",
        "prediction_file": Path(
            "runs/PAPER_RAG_TOPK1_EXISTING/predictions_rag_noaug_paper_topk1.jsonl"
        ),
        "title": "PAPER_RAG_TOPK1_EXISTING - noaug RAG (top_k=1)",
    },
    {
        "key": "paper_rag_aug",
        "name": "PAPER_RAG_TOPK1_EXISTING/aug",
        "prediction_file": Path(
            "runs/PAPER_RAG_TOPK1_EXISTING/predictions_rag_aug_paper_topk1.jsonl"
        ),
        "title": "PAPER_RAG_TOPK1_EXISTING - augmented RAG (top_k=1)",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def plotted_prediction(record: dict[str, Any]) -> str:
    """Map every non-valid prediction to the original code's INVALID bucket."""

    prediction = record.get("pred_label")
    if prediction in VALID_LABELS:
        return prediction
    return "INVALID"


def plot_one(experiment: dict[str, Any], root: Path, output_dir: Path, show: bool) -> dict[str, Any]:
    prediction_path = root / experiment["prediction_file"]
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    records = read_jsonl(prediction_path)
    if not records:
        raise ValueError(f"Prediction file is empty: {prediction_path}")

    y_true = [record.get("true_label") for record in records]
    if any(label not in VALID_LABELS for label in y_true):
        invalid_true = sorted({label for label in y_true if label not in VALID_LABELS})
        raise ValueError(f"Unexpected true labels in {prediction_path}: {invalid_true}")
    y_pred = [plotted_prediction(record) for record in records]

    # This matches the original plotting implementation: invalid responses are
    # represented by an additional predicted-label column.
    cm = confusion_matrix(y_true, y_pred, labels=PLOT_LABELS)
    strict_accuracy = float(accuracy_score(y_true, y_pred))
    invalid_count = int(sum(prediction == "INVALID" for prediction in y_pred))
    valid_count = len(records) - invalid_count
    valid_only_accuracy = (
        float(sum(t == p for t, p in zip(y_true, y_pred) if p != "INVALID") / valid_count)
        if valid_count
        else 0.0
    )

    output_stem = f"confusion_{experiment['key']}"
    image_path = output_dir / f"{output_stem}.png"
    matrix_path = output_dir / f"{output_stem}.csv"

    matrix_frame = pd.DataFrame(cm, index=PLOT_LABELS, columns=PLOT_LABELS)
    matrix_frame.index.name = "True Label"
    matrix_frame.to_csv(matrix_path, encoding="utf-8-sig")

    # Keep the original prompt最佳化.py drawing settings.
    plt.rcParams.update({"font.size": 15})
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PLOT_LABELS)
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title(experiment["title"], fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    per_class_accuracy = {}
    for index, label in enumerate(PLOT_LABELS):
        row_total = int(cm[index].sum())
        per_class_accuracy[label] = float(cm[index, index] / row_total) if row_total else 0.0

    return {
        "key": experiment["key"],
        "experiment": experiment["name"],
        "title": experiment["title"],
        "prediction_file": str(experiment["prediction_file"]).replace("/", "\\"),
        "prediction_file_sha256": sha256_file(prediction_path),
        "n": len(records),
        "strict_accuracy": strict_accuracy,
        "correct": int(sum(t == p for t, p in zip(y_true, y_pred))),
        "invalid": invalid_count,
        "invalid_rate": invalid_count / len(records),
        "valid_only_accuracy": valid_only_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix_csv": matrix_path.name,
        "figure": image_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing runs/ (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/PAPER_CONFUSION_MATRICES)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after each figure; omitted for batch/headless execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "runs/PAPER_CONFUSION_MATRICES").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [plot_one(experiment, root, output_dir, args.show) for experiment in EXPERIMENTS]

    summary_path = output_dir / "summary.csv"
    summary_fields = [
        "key",
        "experiment",
        "n",
        "strict_accuracy",
        "correct",
        "invalid",
        "invalid_rate",
        "valid_only_accuracy",
        "figure",
        "confusion_matrix_csv",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in summary_fields})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/plot_paper_confusion_matrices.py",
        "source_plot_function": "C:/Users/wuc120/Code/prompt最佳化.py::evaluate_confusion_matrix",
        "plot_method": {
            "display": "sklearn.metrics.ConfusionMatrixDisplay",
            "cmap": "Blues",
            "values_format": "d",
            "figure_size_inches": [8, 6],
            "labels": PLOT_LABELS,
            "x_label": "Predicted Label",
            "y_label": "True Label",
            "x_tick_rotation": 45,
            "invalid_policy": "Any prediction not in the four valid labels is plotted as INVALID.",
            "interactive_show": bool(args.show),
        },
        "experiments": summaries,
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"output_dir": str(output_dir), "experiments": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
