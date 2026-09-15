"""條件之間的配對統計檢定。

原始實驗從未保存逐筆預測，導致審稿人要求的配對檢定（McNemar、bootstrap CI）
在數學上無法事後補算 —— 這一項本身就足以構成重跑的理由。本流程每一筆預測都落地，
因此這些檢定都是實打實算出來的。

* **McNemar 精確檢定**：配對設計下比較兩個條件，只看「一個對、另一個錯」的不一致對。
  用 binomtest 做精確檢定，不用卡方近似（不一致對很少時近似不可靠）。
* **Holm 校正**：要做的成對比較不只一組，不校正會膨脹偽陽性。
* **Paired bootstrap**：準確率差的信賴區間，重抽樣時兩個條件抽同一組索引，
  保留配對結構。

正確性一律以 strict 慣例認定（無效回應計為答錯）—— 使用者拿不到可用答案就是失敗。

用法：
    python 09_stats.py --run MAIN [--bootstrap 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# 要檢定的成對比較，每一組都對應一個明確的研究問題。
COMPARISONS = [
    ("S0",  "S1",  "prompt 最佳化本身的效果（無 RAG 干擾）"),
    ("S0",  "S2a", "RAG 本身的效果（指令固定為基準）"),
    ("S1",  "S2b", "已最佳化指令再加上 RAG 的增益"),
    ("S2a", "S2b", "最佳化在有 RAG 時還剩多少作用"),
    ("S2a", "S3a", "擴增語料庫的純效果（指令固定）"),
    ("S2b", "S3b", "擴增語料庫在最佳化指令下的效果"),
    ("S3a", "S4",  "針對擴增設定重搜指令的效果"),
    ("S3b", "S4",  "指令最佳化能否跨設定遷移 vs 重新搜尋"),
    ("S0",  "S4",  "完整管線相對於樸素基準的總增益"),
]


def load_correct(run_dir: Path, name: str) -> dict[str, bool] | None:
    path = run_dir / "preds" / f"{name}.jsonl"
    if not path.exists():
        return None
    out = {}
    for rec in C.read_jsonl(path):
        pred = rec["pred_label_id"]
        out[rec["id"]] = bool(pred is not None and pred == rec["true_label_id"])
    return out or None


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """精確 McNemar 檢定。a、b 是同一組樣本在兩個條件下的正確與否。"""
    from scipy.stats import binomtest

    n01 = int(np.sum(~a & b))   # a 錯 b 對
    n10 = int(np.sum(a & ~b))   # a 對 b 錯
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0,
                "note": "兩個條件的對錯完全一致"}
    res = binomtest(n10, n_disc, 0.5)
    return {"n01": n01, "n10": n10, "n_discordant": n_disc,
            "p_value": float(res.pvalue)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int = 42) -> dict:
    """準確率差（b − a）的 paired bootstrap 信賴區間。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)          # 兩個條件抽同一組索引，保留配對
        diffs[i] = b[idx].mean() - a[idx].mean()
    return {
        "mean_diff": round(float(b.mean() - a.mean()), 5),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 5),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 5),
        "n_boot": n_boot,
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni 逐步校正，回傳校正後 p 值（含單調性處理）。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)      # 保證非遞減
        adjusted[key] = round(running, 6)
    return adjusted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    run_dir = C.RUNS / args.run
    correct = {}
    run_cond = C.load_module("06_run_condition")
    for name in run_cond.CONDITIONS:
        c = load_correct(run_dir, name)
        if c:
            correct[name] = c
    if len(correct) < 2:
        C.die(f"至少需要兩個條件的預測檔，目前只有：{sorted(correct)}")

    C.log(f"可用條件：{sorted(correct)}")

    results: dict[str, dict] = {}
    pvals: dict[str, float] = {}

    for left, right, question in COMPARISONS:
        if left not in correct or right not in correct:
            continue
        shared = sorted(set(correct[left]) & set(correct[right]))
        if not shared:
            continue
        a = np.array([correct[left][i] for i in shared])
        b = np.array([correct[right][i] for i in shared])

        key = f"{left}_vs_{right}"
        mc = mcnemar(a, b)
        results[key] = {
            "question": question,
            "n_paired": len(shared),
            f"accuracy_{left}": round(float(a.mean()), 4),
            f"accuracy_{right}": round(float(b.mean()), 4),
            "delta_pp": round(float((b.mean() - a.mean()) * 100), 3),
            "mcnemar": mc,
            "bootstrap": paired_bootstrap(a, b, args.bootstrap),
        }
        pvals[key] = mc["p_value"]

    adjusted = holm(pvals)
    for key, p_adj in adjusted.items():
        results[key]["mcnemar"]["p_value_holm"] = p_adj
        results[key]["significant_after_holm"] = bool(p_adj < 0.05)

    out = {
        "run": args.run,
        "correctness_convention": "strict（無效回應計為答錯）",
        "multiple_comparison_correction": "Holm–Bonferroni",
        "n_comparisons": len(results),
        "comparisons": results,
    }
    C.save_json(run_dir / "stats.json", out)

    C.log("")
    for key, r in results.items():
        sig = "顯著" if r.get("significant_after_holm") else "不顯著"
        ci = r["bootstrap"]
        C.log(f"{key:12s} {r['delta_pp']:+7.2f}pp  "
              f"CI95[{ci['ci95_low'] * 100:+.2f},{ci['ci95_high'] * 100:+.2f}]pp  "
              f"p_holm={r['mcnemar']['p_value_holm']:.3g}  {sig}")
        C.log(f"             {r['question']}")
    C.log(f"已寫入 {run_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
