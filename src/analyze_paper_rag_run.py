"""Create durable comparison artifacts for PAPER_RAG_TOPK1_EXISTING."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "PAPER_RAG_TOPK1_EXISTING"
LABELS = ["Normal", "Depression", "Anxiety", "Bipolar"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    by_corpus = {
        name: read_jsonl(RUN / f"predictions_rag_{name}_paper_topk1.jsonl")
        for name in ("noaug", "aug")
    }
    for corpus, records in by_corpus.items():
        rows = []
        for record in records:
            pred = record["pred_label"] if record["pred_label"] is not None else "INVALID"
            rows.append({"true_label": record["true_label"], "pred_label": pred})
        frame = pd.DataFrame(rows)
        matrix = pd.crosstab(
            frame["true_label"], frame["pred_label"],
            dropna=False,
        ).reindex(index=LABELS, columns=LABELS + ["INVALID"], fill_value=0)
        matrix.to_csv(RUN / f"confusion_rag_{corpus}_paper_topk1.csv", encoding="utf-8-sig")

        invalid = Counter(
            r["invalid_reason"] for r in records if r["invalid"]
        )
        pd.DataFrame(
            [{"invalid_reason": reason, "count": count} for reason, count in sorted(invalid.items())]
        ).to_csv(RUN / f"invalid_reasons_rag_{corpus}_paper_topk1.csv", index=False, encoding="utf-8-sig")

        retrieved = []
        for record in records:
            retrieved_label = (record.get("retrieved_labels") or [None])[0]
            similarity = (record.get("retrieved_similarities") or [None])[0]
            retrieved.append({
                "id": record["id"],
                "true_label": record["true_label"],
                "retrieved_label": retrieved_label,
                "retrieved_label_correct": retrieved_label == record["true_label"],
                "retrieved_id": (record.get("retrieved_ids") or [None])[0],
                "similarity": similarity,
                "retrieved_is_aug": str((record.get("retrieved_ids") or [""])[0]).startswith("aug_"),
            })
        retrieved_frame = pd.DataFrame(retrieved)
        retrieved_frame.to_csv(RUN / f"retrieval_analysis_rag_{corpus}_paper_topk1.csv", index=False, encoding="utf-8-sig")

    noaug = {r["id"]: r for r in by_corpus["noaug"]}
    aug = {r["id"]: r for r in by_corpus["aug"]}
    paired = []
    for sample_id in sorted(set(noaug) & set(aug)):
        n = noaug[sample_id]
        a = aug[sample_id]
        n_correct = bool(n["correct"])
        a_correct = bool(a["correct"])
        n_invalid = bool(n["invalid"])
        a_invalid = bool(a["invalid"])
        paired.append({
            "id": sample_id,
            "true_label": n["true_label"],
            "noaug_pred": n["pred_label"] or "INVALID",
            "aug_pred": a["pred_label"] or "INVALID",
            "noaug_correct": n_correct,
            "aug_correct": a_correct,
            "noaug_invalid": n_invalid,
            "aug_invalid": a_invalid,
            "outcome": (
                "both_correct" if n_correct and a_correct else
                "aug_only_correct" if a_correct and not n_correct else
                "noaug_only_correct" if n_correct and not a_correct else
                "both_wrong"
            ),
            "invalid_transition": f"{'invalid' if n_invalid else 'valid'}_to_{'invalid' if a_invalid else 'valid'}",
            "retrieved_id_noaug": (n.get("retrieved_ids") or [None])[0],
            "retrieved_id_aug": (a.get("retrieved_ids") or [None])[0],
            "similarity_noaug": (n.get("retrieved_similarities") or [None])[0],
            "similarity_aug": (a.get("retrieved_similarities") or [None])[0],
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(RUN / "paired_comparison_rag_paper_topk1.csv", index=False, encoding="utf-8-sig")

    def avg(records: list[dict], key: str) -> float | None:
        values = [r[key] for r in records if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    analysis = {
        "run": "PAPER_RAG_TOPK1_EXISTING",
        "n_paired": len(paired),
        "same_prompt": True,
        "top_k": 1,
        "corpus_comparison": {
            corpus: {
                "n": len(records),
                "correct": sum(bool(r["correct"]) for r in records),
                "invalid": sum(bool(r["invalid"]) for r in records),
                "retrieved_label_correct": sum(
                    bool((r.get("retrieved_labels") or [None])[0] == r["true_label"])
                    for r in records
                ),
                "retrieved_label_accuracy": sum(
                    bool((r.get("retrieved_labels") or [None])[0] == r["true_label"])
                    for r in records
                ) / len(records),
                "mean_retrieval_similarity": avg(
                    [{"value": (r.get("retrieved_similarities") or [None])[0]} for r in records],
                    "value",
                ),
                "invalid_reasons": dict(Counter(
                    r["invalid_reason"] for r in records if r["invalid"]
                )),
            }
            for corpus, records in by_corpus.items()
        },
        "paired_outcomes": ({str(k): int(v) for k, v in paired_frame["outcome"].value_counts().items()} if len(paired_frame) else {}),
        "invalid_transitions": ({str(k): int(v) for k, v in paired_frame["invalid_transition"].value_counts().items()} if len(paired_frame) else {}),
    }
    (RUN / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
