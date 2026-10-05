"""Plot confusion matrices for the PAPER_* experiments and the main-eval C1-C5 set.

``--set paper`` (default) reproduces the PAPER_* figures below unchanged;
``--set custom15`` draws the main-experiment C1-C5 results as one
row-normalized five-panel figure (see ``plot_custom15``).  ``--profile`` picks
the result set (see ``custom_eval_common``):

* ``v1``: 15_ (CUSTOM_PROMPT_EVAL).  Its figures are a frozen record, so
  ``--output-dir`` is required and may not point inside CUSTOM_PROMPT_EVAL/.
* ``v2``: 27_ (MAIN_EVAL_AUGV2, aug_v2 corpus, selected top-k).  Output defaults
  to MAIN_EVAL_AUGV2/figures/.  Panel titles are "code + display name" (the same
  code means a different condition in v1 and v2), drawn with a CJK font.

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
from matplotlib import font_manager
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


# 15_ (CUSTOM_PROMPT_EVAL, v1) conditions, coded as in the paper's Table VI.
CUSTOM15_EXPERIMENTS = [
    {"code": "C1", "condition": "norag_base", "title": "C1  LLM only"},
    {"code": "C2", "condition": "rag_noaug_base", "title": "C2  RAG"},
    {"code": "C3", "condition": "rag_aug_base", "title": "C3  RAG + DA"},
    {"code": "C4", "condition": "rag_noaug_optimized", "title": "C4  RAG + PO"},
    {"code": "C5", "condition": "rag_aug_optimized", "title": "C5  RAG + PO + DA"},
]
CUSTOM15_N = 1998
CJK_FONT = "Microsoft JhengHei"


def load_profile(name: str):
    """custom_eval_common is imported lazily so that ``--set paper`` stays standalone."""
    import sys

    src = str(Path(__file__).resolve().parent)
    if src not in sys.path:
        sys.path.insert(0, src)
    import custom_eval_common as E

    return E, E.get_profile(name)


def custom_experiments(profile) -> list[dict[str, Any]]:
    """v1 keeps the original panel titles; other profiles use code + display name."""
    if profile.name == "v1":
        if [e["condition"] for e in CUSTOM15_EXPERIMENTS] != list(profile.codes.values()):
            raise ValueError("CUSTOM15_EXPERIMENTS no longer matches the v1 profile")
        return CUSTOM15_EXPERIMENTS
    return [{"code": code, "condition": cond, "title": profile.label(code)}
            for code, cond in profile.codes.items()]


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


def load_custom15(profile) -> list[tuple[dict[str, Any], Path, list[str], list[str]]]:
    """Read C1-C5 predictions and check they are the same 1,998 test samples."""

    loaded = []
    reference_ids: list[str] | None = None
    for experiment in custom_experiments(profile):
        path = profile.in_dir / f"predictions_{experiment['condition']}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        records = read_jsonl(path)
        if len(records) != CUSTOM15_N:
            raise ValueError(f"{path} has {len(records)} records, expected {CUSTOM15_N}")
        if any(r.get("experiment") != profile.experiment or
               r.get("condition") != experiment["condition"] for r in records):
            raise ValueError(f"{path} is not a {profile.experiment}/{experiment['condition']} file")
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


def plot_custom15(root: Path, output_dir: Path, show: bool, profile) -> dict[str, Any]:
    """Five row-normalized confusion matrices side by side on a shared 0-1 scale.

    Rows are true labels (4 classes); columns are predicted labels including
    INVALID, so each row sums to 1 and the INVALID column shows the per-class
    invalid-response rate.
    """

    loaded = load_custom15(profile)
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
        if profile.name != "v1":
            summaries[-1].update({"display_name": profile.display(experiment["code"]),
                                  "title": experiment["title"]})

    plt.rcParams.update({"font.size": 11})
    if profile.name != "v1":
        # Display names are Chinese; fail rather than render tofu boxes.
        if CJK_FONT not in {f.name for f in font_manager.fontManager.ttflist}:
            raise RuntimeError(f"CJK font {CJK_FONT!r} not found; cannot draw the panel titles")
        plt.rcParams.update({"font.family": [CJK_FONT, "DejaVu Sans"]})
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
        "script": "src/plot_paper_confusion_matrices.py --set custom15"
                  + ("" if profile.name == "v1" else f" --profile {profile.name}"),
        "source_experiment": f"{profile.source_script} ({profile.experiment})",
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
    if profile.name != "v1":
        manifest.update({"profile": profile.name, "top_k": profile.top_k,
                         "panel_title_font": CJK_FONT})
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
             "MAIN_EVAL_AUGV2/figures for --set custom15 --profile v2; required for "
             "--profile v1, whose CUSTOM_PROMPT_EVAL/figures is frozen)",
    )
    parser.add_argument(
        "--set",
        dest="experiment_set",
        choices=["paper", "custom15"],
        default="paper",
        help="paper: the PAPER_* runs (original per-figure count plots); "
             "custom15: main-experiment C1-C5 as one row-normalized five-panel figure",
    )
    parser.add_argument(
        "--profile",
        choices=["v1", "v2"],
        default="v1",
        help="--set custom15 only: v1 = 15_ (CUSTOM_PROMPT_EVAL), v2 = 27_ (MAIN_EVAL_AUGV2)",
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
        E, profile = load_profile(args.profile)
        frozen = E.V1_OUT.resolve()
        if args.output_dir is None:
            if profile.name == "v1":
                raise SystemExit("v1 (CUSTOM_PROMPT_EVAL/) is a frozen record: "
                                 "--profile v1 requires --output-dir")
            output_dir = (profile.in_dir / "figures").resolve()
        else:
            output_dir = args.output_dir.resolve()
        if profile.name == "v1" and (output_dir == frozen or frozen in output_dir.parents):
            raise SystemExit(f"--output-dir {output_dir} is inside the frozen v1 directory {frozen}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = plot_custom15(root, output_dir, args.show, profile)
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
