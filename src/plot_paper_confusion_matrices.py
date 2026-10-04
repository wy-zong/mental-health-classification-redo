"""Plot confusion matrices for the PAPER_* experiments and the 15_ C1-C5 set.

``--set paper`` (default) reproduces the PAPER_* figures below unchanged;
``--set custom15`` draws the 15_ (CUSTOM_PROMPT_EVAL) C1-C5 results as one
row-normalized five-panel figure (see ``plot_custom15``).

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


# 15_ (CUSTOM_PROMPT_EVAL) conditions, coded as in the paper's Table VI.
CUSTOM15_DIR = Path("runs/NESTED_70_10_20_INCREMENTAL/CUSTOM_PROMPT_EVAL")
CUSTOM15_EXPERIMENTS = [
    {"code": "C1", "condition": "norag_base", "title": "C1  LLM only"},
    {"code": "C2", "condition": "rag_noaug_base", "title": "C2  RAG"},
    {"code": "C3", "condition": "rag_aug_base", "title": "C3  RAG + DA"},
    {"code": "C4", "condition": "rag_noaug_optimized", "title": "C4  RAG + PO"},
    {"code": "C5", "condition": "rag_aug_optimized", "title": "C5  RAG + PO + DA"},
]
CUSTOM15_N = 1998


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


def load_custom15(root: Path) -> list[tuple[dict[str, Any], Path, list[str], list[str]]]:
    """Read C1-C5 predictions and check they are the same 1,998 test samples."""

    loaded = []
    reference_ids: list[str] | None = None
    for experiment in CUSTOM15_EXPERIMENTS:
        path = root / CUSTOM15_DIR / f"predictions_{experiment['condition']}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        records = read_jsonl(path)
        if len(records) != CUSTOM15_N:
            raise ValueError(f"{path} has {len(records)} records, expected {CUSTOM15_N}")
        if any(r.get("experiment") != "CUSTOM_PROMPT_EVAL" or
               r.get("condition") != experiment["condition"] for r in records):
            raise ValueError(f"{path} is not a CUSTOM_PROMPT_EVAL/{experiment['condition']} file")
        records.sort(key=lambda r: str(r["id"]))
        ids = [str(r["id"]) for r in records]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{path} does not cover the same test ids as C1")
        y_true = [r.get("true_label") for r in records]
        if any(label not in VALID_LABELS for label in y_true):
            raise ValueError(f"Unexpected true labels in {path}")
        loaded.append((experiment, path, y_true, [plotted_prediction(r) for r in records]))
    return loaded


def plot_custom15(root: Path, output_dir: Path, show: bool) -> dict[str, Any]:
    """Five row-normalized confusion matrices side by side on a shared 0-1 scale.

    Rows are true labels (4 classes); columns are predicted labels including
    INVALID, so each row sums to 1 and the INVALID column shows the per-class
    invalid-response rate.
    """

    loaded = load_custom15(root)
    summaries = []
    normalized = []
    for experiment, path, y_true, y_pred in loaded:
        cm = confusion_matrix(y_true, y_pred, labels=PLOT_LABELS)[: len(VALID_LABELS)]
        row_sum = cm.sum(axis=1, keepdims=True)
        rates = cm / row_sum
        normalized.append(rates)

        stem = f"confusion_{experiment['code']}_{experiment['condition']}"
        counts = pd.DataFrame(cm, index=VALID_LABELS, columns=PLOT_LABELS)
        counts.index.name = "True Label"
        counts.to_csv(output_dir / f"{stem}_counts.csv", encoding="utf-8-sig")
        rownorm = pd.DataFrame(rates, index=VALID_LABELS, columns=PLOT_LABELS)
        rownorm.index.name = "True Label"
        rownorm.to_csv(output_dir / f"{stem}_rownorm.csv", encoding="utf-8-sig")

        summaries.append({
            "code": experiment["code"],
            "condition": experiment["condition"],
            "prediction_file": str(path.relative_to(root)).replace("\\", "/"),
            "prediction_file_sha256": sha256_file(path),
            "n": len(y_true),
            "strict_accuracy": float(accuracy_score(y_true, y_pred)),
            "invalid": int(sum(p == "INVALID" for p in y_pred)),
            "per_class_recall": {lab: float(rates[i, i]) for i, lab in enumerate(VALID_LABELS)},
            "per_class_invalid_rate": {
                lab: float(rates[i, len(VALID_LABELS)]) for i, lab in enumerate(VALID_LABELS)
            },
            "counts_csv": f"{stem}_counts.csv",
            "rownorm_csv": f"{stem}_rownorm.csv",
        })

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.2 * len(loaded), 4.6), sharey=True,
                             constrained_layout=True)
    image = None
    for ax, (experiment, _, _, _), rates in zip(axes, loaded, normalized):
        image = ax.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                ax.text(j, i, f"{rates[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if rates[i, j] > 0.5 else "black")
        ax.set_title(experiment["title"], fontsize=12)
        ax.set_xticks(range(len(PLOT_LABELS)), PLOT_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(VALID_LABELS)), VALID_LABELS)
        ax.set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Proportion of true-label row")

    figure_stem = "confusion_rownorm_C1-C5"
    fig.savefig(output_dir / f"{figure_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{figure_stem}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/plot_paper_confusion_matrices.py --set custom15",
        "source_experiment": "src/15_nested_custom_prompt_eval.py (CUSTOM_PROMPT_EVAL)",
        "plot_method": {
            "normalization": "row (each true-label row sums to 1)",
            "color_scale": "Blues, shared vmin=0, vmax=1, one colorbar",
            "rows": "True Label (4 classes)",
            "columns": "Predicted Label, including INVALID",
            "invalid_policy": "Any prediction not in the four valid labels is plotted as INVALID.",
            "figures": [f"{figure_stem}.png", f"{figure_stem}.pdf"],
            "interactive_show": bool(show),
        },
        "experiments": summaries,
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"output_dir": str(output_dir), "experiments": summaries}


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
        help="Output directory (default: runs/PAPER_CONFUSION_MATRICES for --set paper, "
             "runs/NESTED_70_10_20_INCREMENTAL/CUSTOM_PROMPT_EVAL/figures for --set custom15)",
    )
    parser.add_argument(
        "--set",
        dest="experiment_set",
        choices=["paper", "custom15"],
        default="paper",
        help="paper: the PAPER_* runs (original per-figure count plots); "
             "custom15: 15_ C1-C5 as one row-normalized five-panel figure",
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
    if args.experiment_set == "custom15":
        output_dir = (args.output_dir or root / CUSTOM15_DIR / "figures").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result = plot_custom15(root, output_dir, args.show)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

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
