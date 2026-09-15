"""從逐筆預測計算指標，兩種 invalid 分母慣例並陳。

原始實驗在這裡有兩個問題：`test.py` 把無效回應排除在分母外，`prompt最佳化.py` 卻納入
分母，兩支腳本算出來的數字不可比 —— 這正是審稿意見 R2C23 抓到的「不同分母」。

這裡兩種慣例一律並陳，不預設哪一種才對：

    strict   無效回應計為答錯，分母 = 全部樣本
             （使用者角度：模型沒給出可用答案就是失敗）
    valid_only  無效回應排除，分母 = 可解析的樣本
             （模型能力角度：只看它有作答時答得如何）

兩者都報告，並附上無效率本身，讀者才能自行判斷。

用法：
    python 08_metrics.py --run MAIN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

INVALID_TOKEN = "INVALID"


def score(y_true: list[int], y_pred: list[int | None], convention: str) -> dict:
    """計算單一慣例下的指標。

    strict：把無效回應標成一個不在標籤集內的符號。sklearn 會把它計為該真實類別的
    false negative，但不會成為任何類別的 false positive —— 這正是我們要的語意。
    """
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    if convention == "valid_only":
        pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
        if not pairs:
            return {"n": 0}
        yt = [t for t, _ in pairs]
        yp = [p for _, p in pairs]
        labels = list(range(len(C.LABELS)))
        names = C.LABELS
    else:
        yt = [C.ID_TO_LABEL[t] for t in y_true]
        yp = [C.ID_TO_LABEL[p] if p is not None else INVALID_TOKEN for p in y_pred]
        labels = C.LABELS
        names = C.LABELS

    acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
    macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    p, r, f, sup = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0)

    return {
        "n": len(yt),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            name: {"precision": round(float(p[i]), 4),
                   "recall": round(float(r[i]), 4),
                   "f1": round(float(f[i]), 4),
                   "support": int(sup[i])}
            for i, name in enumerate(names)
        },
    }


def confusion(y_true: list[int], y_pred: list[int | None]) -> dict:
    cols = C.LABELS + [INVALID_TOKEN]
    m = pd.DataFrame(0, index=C.LABELS, columns=cols)
    for t, p in zip(y_true, y_pred):
        m.loc[C.ID_TO_LABEL[t], C.ID_TO_LABEL[p] if p is not None else INVALID_TOKEN] += 1
    return {row: m.loc[row].to_dict() for row in m.index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN")
    args = ap.parse_args()

    run_dir = C.RUNS / args.run
    preds_dir = run_dir / "preds"
    if not preds_dir.exists():
        C.die(f"找不到 {preds_dir}，尚未執行任何條件")

    run_cond = C.load_module("06_run_condition")
    out: dict = {"run": args.run, "conditions": {}}

    for name in run_cond.CONDITIONS:
        path = preds_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        records = list(C.read_jsonl(path))
        if not records:
            continue

        y_true = [r["true_label_id"] for r in records]
        y_pred = [r["pred_label_id"] for r in records]
        n_invalid = sum(p is None for p in y_pred)

        lat = [r["elapsed_s"] for r in records if r.get("elapsed_s") is not None]
        ptok = [r["prompt_eval_count"] for r in records if r.get("prompt_eval_count")]
        otok = [r["eval_count"] for r in records if r.get("eval_count")]

        reasons: dict[str, int] = {}
        for r in records:
            if r["pred_label_id"] is None:
                reasons[r.get("parse_reason", "?")] = reasons.get(r.get("parse_reason", "?"), 0) + 1

        entry = {
            "n": len(records),
            "invalid_count": n_invalid,
            "invalid_rate": round(n_invalid / len(records), 4),
            "invalid_reasons": reasons,
            "strict": score(y_true, y_pred, "strict"),
            "valid_only": score(y_true, y_pred, "valid_only"),
            "confusion_strict": confusion(y_true, y_pred),
            "cost": {
                "total_seconds": round(float(np.sum(lat)), 1) if lat else None,
                "mean_seconds": round(float(np.mean(lat)), 3) if lat else None,
                "mean_prompt_tokens": round(float(np.mean(ptok)), 1) if ptok else None,
                "mean_output_tokens": round(float(np.mean(otok)), 1) if otok else None,
            },
        }
        out["conditions"][name] = entry

        C.log(f"{name:4s}  n={entry['n']:,}  "
              f"acc(strict)={entry['strict']['accuracy']:.4f}  "
              f"macroF1={entry['strict']['macro_f1']:.4f}  "
              f"invalid={entry['invalid_rate']:.4f}  "
              f"{entry['cost']['mean_seconds']}s/筆")

    if not out["conditions"]:
        C.die("沒有任何條件的預測檔可供計算")

    path = run_dir / "metrics.json"
    C.save_json(path, out)
    C.log(f"已寫入 {path}")


if __name__ == "__main__":
    main()
