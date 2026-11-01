"""Issue #4、#2：重切與 baseline 的彙整（離線，不呼叫 LLM）。

LLM（v2、k = 1）
  4 組切分：主切分（MAIN_EVAL_AUGV2_K1/）＋ RESPLITS/resplit_seed{0,1,2}/MAIN_EVAL_K1/。
  * 各切分 C1–C5 的 accuracy、macro-F1、weighted-F1、無效率（含 bootstrap 95% CI），
    以及 4 組的平均 ± SD、最小、最大
  * 各切分 10 組配對：Δ（含 CI）、n01／n10、exact McNemar、Holm（每組切分內 10 組校正，與 17_ 相同）
  * 各配對跨切分：Δ accuracy 的平均 ± SD、Δ > 0 的切分數、Holm 後顯著的切分數
baseline（07_ --split-dir）
  主切分 BASELINES/（seed 42–44）、各重切 RESPLITS/<name>/BASELINES/（seed 42）
  * 各切分、模型、版本、seed 的指標（含 bootstrap 95% CI）
  * 各模型 × 版本：主切分 3 seed 的平均 ± SD；4 組切分（seed 42）的平均 ± SD
  * 各切分 baseline（seed 42）vs C4、C5：Δ（含 CI）與 exact McNemar（LLM 的 INVALID 算錯；每組切分內 Holm）
  * 成本：訓練／推論時間、DistilBERT 峰值 GPU 記憶體，對照 LLM 的 cost_latency.csv

數字慣例同 17_：無效回覆算錯、F1 只平均四個有效類別、Δ = 後者 − 前者、
相對錯誤率下降 = (err_前 − err_後) / err_前 × 100。
bootstrap 95% CI 也同 17_：每組切分各自 10,000 次重抽（seed 42，percentile 2.5／97.5），
同一切分內所有 LLM 條件與 baseline 共用同一組重抽索引，配對的 Δ CI 用同一組索引。
主切分的點估計、CI、McNemar 必須與 MAIN_EVAL_AUGV2_K1/stats_conditions.csv、
stats_paired.csv 相同（本腳本會檢查）。

輸出：runs/NESTED_70_10_20_INCREMENTAL/RESPLITS/summary/

用法
  python src/31_resplit_baseline_summary.py              # 全部切分都要完成
  python src/31_resplit_baseline_summary.py --partial    # 跳過尚未完成的切分或 baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

STATS09 = C.load_module("09_stats")

BASE = E.BASE_OUT
K1 = E.V2_K1_OUT
RESPLITS = BASE / "RESPLITS"
OUT = RESPLITS / "summary"
SPLITS = ("exp15", "resplit_seed0", "resplit_seed1", "resplit_seed2")
CODES = E.V2_CODES                                   # C1–C5 → 條件 ID
PAIRS = list(combinations(CODES, 2))
MODELS = ("tfidf", "minilm", "distilbert")
VARIANTS = ("noaug", "aug_v2")
MAIN_SEEDS = (42, 43, 44)
CMP_SEED = 42
COMPARE_LLM = ("C4", "C5")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "invalid_rate")


def llm_dir(split: str) -> Path:
    return K1 if split == "exp15" else RESPLITS / split / "MAIN_EVAL_K1"


def baseline_dir(split: str) -> Path:
    return BASE / "BASELINES" if split == "exp15" else RESPLITS / split / "BASELINES"


def split_dir(split: str) -> Path:
    return BASE / "splits" if split == "exp15" else RESPLITS / split / "splits"


def read_text_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_truth(split: str) -> pd.Series:
    df = read_text_csv(split_dir(split) / "test.csv")
    return pd.Series(df["label_id"].astype(int).to_numpy(), index=df["id"])


# ---------------------------------------------------------------- LLM

def load_llm(split: str, truth: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    d = llm_dir(split)
    if not (d / "completion.json").exists():
        return None
    out = {}
    for code, cond in CODES.items():
        recs = list(C.read_jsonl(d / f"predictions_{cond}.jsonl"))
        by_id = {r["id"]: r for r in recs}
        if len(recs) != len(truth) or set(by_id) != set(truth.index):
            C.die(f"{split} {cond}：逐筆預測的 id 與 test.csv 不同")
        ordered = [by_id[i] for i in truth.index]
        y_true, y_pred = E.encode(ordered)
        if not np.array_equal(y_true, truth.to_numpy()):
            C.die(f"{split} {cond}：true_label 與 test.csv 不同")
        out[code] = (y_true, y_pred)
    return out


def boot_metrics(enc: dict, idx: np.ndarray) -> dict[str, dict]:
    """每個條件（或 baseline）在同一組重抽索引下的 bootstrap 指標陣列。"""
    return {key: E.metrics_from_confusion(E.boot_confusions(y, p, idx)) for key, (y, p) in enc.items()}


def with_ci(point: dict, boot: dict, keys: tuple[str, ...]) -> dict:
    row = {}
    for k in keys:
        lo, hi = E.ci95(boot[k])
        row.update({k: float(point[k]), f"{k}_ci_low": lo, f"{k}_ci_high": hi})
    return row


def llm_conditions(split: str, enc: dict, bm: dict) -> list[dict]:
    rows = []
    for code, (y, p) in enc.items():
        m = E.metrics_from_confusion(E.confusion(y, p))
        rows.append({"split": split, "code": code, "condition": CODES[code], "n": int(len(y)),
                     "invalid_count": int((p == E.INVALID).sum()),
                     **with_ci(m, bm[code], METRICS)})
    return rows


def paired_rows(split: str, enc: dict, bm: dict, pairs: list[tuple[str, str]],
                family: str) -> list[dict]:
    """pairs 的每組 (a, b)：Δ = b − a 與配對 bootstrap CI、exact McNemar；同一 family 內 Holm。"""
    rows = []
    for a, b in pairs:
        (ya, pa), (yb, pb) = enc[a], enc[b]
        ma, mb = E.metrics_from_confusion(E.confusion(ya, pa)), E.metrics_from_confusion(E.confusion(yb, pb))
        ca, cb = pa == ya, pb == yb
        mc = STATS09.mcnemar(ca, cb)
        err_a, err_b = 1 - float(ma["accuracy"]), 1 - float(mb["accuracy"])
        row = {"split": split, "family": family, "first": a, "second": b,
               "accuracy_first": float(ma["accuracy"]), "accuracy_second": float(mb["accuracy"])}
        for k in ("accuracy", "macro_f1", "weighted_f1"):
            lo, hi = E.ci95(bm[b][k] - bm[a][k])
            row.update({f"delta_{k}": float(mb[k] - ma[k]), f"delta_{k}_ci_low": lo, f"delta_{k}_ci_high": hi})
        row.update({"n01": mc["n01"], "n10": mc["n10"], "mcnemar_p": mc["p_value"],
                    "relative_error_reduction_pct": (err_a - err_b) / err_a * 100 if err_a else float("nan")})
        rows.append(row)
    adj = E.holm({f"{r['first']}|{r['second']}": r["mcnemar_p"] for r in rows})
    for r in rows:
        r["mcnemar_p_holm"] = adj[f"{r['first']}|{r['second']}"]
        r["significant_after_holm"] = bool(r["mcnemar_p_holm"] < 0.05)
    return rows


def check_exp15(cond_rows: list[dict], pair_rows: list[dict]) -> list[str]:
    """主切分必須與 17_ 的 stats_conditions.csv／stats_paired.csv 相同（含 bootstrap CI）。"""
    problems = []
    ref = pd.read_csv(K1 / "stats_conditions.csv", encoding="utf-8-sig").set_index("code")
    for r in cond_rows:
        for k in [f"{m}{s}" for m in METRICS for s in ("", "_ci_low", "_ci_high")]:
            if abs(r[k] - ref.loc[r["code"], k]) > 1e-12:
                problems.append(f"exp15 {r['code']} {k}：{r[k]} vs 17_ {ref.loc[r['code'], k]}")
    refp = pd.read_csv(K1 / "stats_paired.csv", encoding="utf-8-sig").set_index(["first", "second"])
    for r in pair_rows:
        x = refp.loc[(r["first"], r["second"])]
        for k in [f"delta_{m}{s}" for m in ("accuracy", "macro_f1", "weighted_f1")
                  for s in ("", "_ci_low", "_ci_high")] + ["mcnemar_p", "mcnemar_p_holm"]:
            if abs(r[k] - x[k]) > 1e-12 * max(1.0, abs(x[k])):
                problems.append(f"exp15 {r['first']}→{r['second']} {k}：{r[k]} vs 17_ {x[k]}")
        if int(r["n01"]) != int(x["n01"]) or int(r["n10"]) != int(x["n10"]):
            problems.append(f"exp15 {r['first']}→{r['second']} n01/n10 不同")
    return problems


def describe(values: pd.Series) -> dict:
    v = values.astype(float)
    return {"n_splits": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "min": float(v.min()), "max": float(v.max())}


# ---------------------------------------------------------------- baseline

def load_baseline(split: str, truth: pd.Series, model: str, variant: str, seed: int):
    path = baseline_dir(split) / "predictions" / f"{model}_{variant}_seed{seed}.jsonl"
    if not path.exists():
        return None
    by_id = {r["id"]: r for r in C.read_jsonl(path)}
    if set(by_id) != set(truth.index):
        C.die(f"{path}：id 與 test.csv 不同")
    y_pred = np.array([int(by_id[i]["pred_label_id"]) for i in truth.index], dtype=np.int64)
    y_true = np.array([int(by_id[i]["true_label_id"]) for i in truth.index], dtype=np.int64)
    if not np.array_equal(y_true, truth.to_numpy()):
        C.die(f"{path}：true_label_id 與 test.csv 不同")
    return y_true, y_pred


def baseline_runs(split: str) -> dict:
    """baselines.json 中各模型、版本、seed 的時間與記憶體紀錄。"""
    res = C.load_json(baseline_dir(split) / "baselines.json")
    out = {}
    for variant, bucket in res.get("variants", {}).items():
        for model in MODELS:
            for r in bucket.get(model, {}).get("runs", []):
                out[(model, variant, r["seed"])] = {**r, "train_rows": bucket.get("train_rows")}
    return out


# ---------------------------------------------------------------- 輸出

def fmt(x: float, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def markdown(llm_agg: pd.DataFrame, pair_agg: pd.DataFrame, base_agg: pd.DataFrame,
             vs: pd.DataFrame, done_splits: list[str]) -> str:
    lines = [f"# 重切與 baseline 彙整（v2、k = 1）", "",
             f"切分：{'、'.join(done_splits)}（n = {len(done_splits)}）。無效回覆算錯。", ""]
    if len(llm_agg):
        lines += ["## LLM：各條件跨切分", "",
                  "| 條件 | accuracy 平均 ± SD [min, max] | macro-F1 平均 ± SD | weighted-F1 平均 ± SD | 無效率 平均 ± SD |",
                  "|---|---|---|---|---|"]
        for code in CODES:
            g = llm_agg[llm_agg["code"] == code].set_index("metric")
            if g.empty:
                continue
            a = g.loc["accuracy"]
            lines.append(f"| {code}　{E.DISPLAY_NAMES[CODES[code]]} | {fmt(a['mean'])} ± {fmt(a['sd'])} "
                         f"[{fmt(a['min'])}, {fmt(a['max'])}] | "
                         + " | ".join(f"{fmt(g.loc[k, 'mean'])} ± {fmt(g.loc[k, 'sd'])}"
                                      for k in ("macro_f1", "weighted_f1", "invalid_rate")) + " |")
        lines += ["", "## LLM：10 組配對跨切分（accuracy）", "",
                  "| 配對 | Δ accuracy 平均 ± SD [min, max] | Δ > 0 的切分數 | Holm 後顯著的切分數 |",
                  "|---|---|---|---|"]
        for _, r in pair_agg.iterrows():
            lines.append(f"| {r['first']}→{r['second']} | {fmt(r['delta_accuracy_mean'])} ± "
                         f"{fmt(r['delta_accuracy_sd'])} [{fmt(r['delta_accuracy_min'])}, "
                         f"{fmt(r['delta_accuracy_max'])}] | {r['n_positive']}／{r['n_splits']} | "
                         f"{r['n_significant']}／{r['n_splits']} |")
    if len(base_agg):
        lines += ["", "## baseline", "",
                  "| 模型 | 版本 | 主切分 accuracy（3 seed 平均 ± SD） | 主切分 macro-F1 | "
                  "跨切分 accuracy（seed 42，平均 ± SD） | 跨切分 macro-F1 | 切分數 |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in base_agg.iterrows():
            lines.append(f"| {r['model']} | {r['variant']} | {fmt(r['main_accuracy_mean'])} ± "
                         f"{fmt(r['main_accuracy_sd'])} | {fmt(r['main_macro_f1_mean'])} ± "
                         f"{fmt(r['main_macro_f1_sd'])} | {fmt(r['splits_accuracy_mean'])} ± "
                         f"{fmt(r['splits_accuracy_sd'])} | {fmt(r['splits_macro_f1_mean'])} ± "
                         f"{fmt(r['splits_macro_f1_sd'])} | {r['n_splits']} |")
    if len(vs):
        lines += ["", "## baseline（seed 42）vs LLM C4、C5（各切分 exact McNemar，切分內 Holm）", "",
                  "| 切分 | 比較 | baseline acc | LLM acc | Δ（LLM − baseline） | n01／n10 | Holm p |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in vs.iterrows():
            lines.append(f"| {r['split']} | {r['first']} vs {r['second']} | {fmt(r['accuracy_first'])} | "
                         f"{fmt(r['accuracy_second'])} | {fmt(r['delta_accuracy'])} | "
                         f"{r['n01']}／{r['n10']} | {r['mcnemar_p_holm']:.2e} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true", help="跳過尚未完成的切分或 baseline")
    ap.add_argument("--bootstrap", type=int, default=E.N_BOOT, help="bootstrap 次數（同 17_）")
    args = ap.parse_args()

    truths = {s: test_truth(s) for s in SPLITS if (split_dir(s) / "test.csv").exists()}
    missing = [s for s in SPLITS if s not in truths]
    if missing and not args.partial:
        C.die(f"缺少切分（先跑 30_）：{missing}")

    problems, notes = [], []
    boot_idx = {s: E.bootstrap_indices(len(t), args.bootstrap, E.SEED) for s, t in truths.items()}
    cond_rows, pair_rows, llm_enc, llm_bm = [], [], {}, {}
    for split, truth in truths.items():
        enc = load_llm(split, truth)
        if enc is None:
            if not args.partial:
                C.die(f"{split} 的 LLM 結果尚未完成：{llm_dir(split)}")
            notes.append(f"{split}：LLM 未完成，略過")
            continue
        llm_enc[split], llm_bm[split] = enc, boot_metrics(enc, boot_idx[split])
        c = llm_conditions(split, enc, llm_bm[split])
        p = paired_rows(split, enc, llm_bm[split], PAIRS, "llm_10_pairs")
        if split == "exp15":
            problems += check_exp15(c, p)
        cond_rows += c
        pair_rows += p
    cond_df, pair_df = pd.DataFrame(cond_rows), pd.DataFrame(pair_rows)

    llm_agg = pd.DataFrame([
        {"code": code, "metric": k, **describe(g[k])}
        for code, g in (cond_df.groupby("code", sort=False) if len(cond_df) else []) for k in METRICS])
    pair_agg = pd.DataFrame([
        {"first": a, "second": b, "n_splits": int(len(g)),
         **{f"delta_accuracy_{k}": v for k, v in describe(g["delta_accuracy"]).items() if k != "n_splits"},
         **{f"delta_macro_f1_{k}": v for k, v in describe(g["delta_macro_f1"]).items() if k != "n_splits"},
         "n_positive": int((g["delta_accuracy"] > 0).sum()),
         "n_significant": int(g["significant_after_holm"].sum()),
         "splits_significant": "、".join(g.loc[g["significant_after_holm"], "split"])}
        for (a, b), g in (pair_df.groupby(["first", "second"], sort=False) if len(pair_df) else [])])

    base_rows, vs_rows, cost_rows = [], [], []
    for split, truth in truths.items():
        runs = baseline_runs(split)
        seeds = MAIN_SEEDS if split == "exp15" else (CMP_SEED,)
        enc_b, bm_b = {}, {}
        for model in MODELS:
            for variant in VARIANTS:
                for seed in seeds:
                    got = load_baseline(split, truth, model, variant, seed)
                    if got is None:
                        notes.append(f"{split}：缺 {model} {variant} seed {seed}")
                        continue
                    m = E.metrics_from_confusion(E.confusion(*got))
                    bmb = boot_metrics({"b": got}, boot_idx[split])["b"]
                    r = runs.get((model, variant, seed), {})
                    base_rows.append({"split": split, "model": model, "variant": variant, "seed": seed,
                                      **with_ci(m, bmb, ("accuracy", "macro_f1", "weighted_f1"))})
                    cost_rows.append({"split": split, "model": model, "variant": variant, "seed": seed,
                                      "train_rows": r.get("train_rows"), "train_seconds": r.get("train_seconds"),
                                      "inference_seconds": r.get("inference_seconds"),
                                      "inference_ms_per_query": (r["inference_seconds"] / len(truth) * 1000
                                                                 if r.get("inference_seconds") is not None else None),
                                      "device": r.get("device"),
                                      "peak_gpu_memory_allocated_mib": r.get("peak_gpu_memory_allocated_mib")})
                    if seed == CMP_SEED:
                        enc_b[f"{model}:{variant}"] = got
                        bm_b[f"{model}:{variant}"] = bmb
        if split in llm_enc and enc_b:
            both = {**enc_b, **{c: llm_enc[split][c] for c in COMPARE_LLM}}
            bm_both = {**bm_b, **{c: llm_bm[split][c] for c in COMPARE_LLM}}
            vs_rows += paired_rows(split, both, bm_both, [(b, c) for b in enc_b for c in COMPARE_LLM],
                                   "baseline_vs_llm")
    if any(n.startswith(tuple(SPLITS)) and "缺" in n for n in notes) and not args.partial:
        C.die("baseline 結果不完整：" + "；".join(n for n in notes if "缺" in n))
    base_df, vs_df, cost_df = pd.DataFrame(base_rows), pd.DataFrame(vs_rows), pd.DataFrame(cost_rows)

    base_agg_rows = []
    if len(base_df):
        for (model, variant), g in base_df.groupby(["model", "variant"], sort=False):
            main = g[g["split"] == "exp15"]
            cmp_ = g[g["seed"] == CMP_SEED]
            dm_a, dm_f = describe(main["accuracy"]) if len(main) else {}, describe(main["macro_f1"]) if len(main) else {}
            ds_a, ds_f = describe(cmp_["accuracy"]), describe(cmp_["macro_f1"])
            base_agg_rows.append({"model": model, "variant": variant,
                                  "main_n_seeds": int(len(main)),
                                  "main_accuracy_mean": dm_a.get("mean"), "main_accuracy_sd": dm_a.get("sd"),
                                  "main_macro_f1_mean": dm_f.get("mean"), "main_macro_f1_sd": dm_f.get("sd"),
                                  "n_splits": ds_a["n_splits"],
                                  "splits_accuracy_mean": ds_a["mean"], "splits_accuracy_sd": ds_a["sd"],
                                  "splits_accuracy_min": ds_a["min"], "splits_accuracy_max": ds_a["max"],
                                  "splits_macro_f1_mean": ds_f["mean"], "splits_macro_f1_sd": ds_f["sd"]})
    base_agg = pd.DataFrame(base_agg_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"llm_conditions_by_split": cond_df, "llm_conditions_aggregate": llm_agg,
              "llm_paired_by_split": pair_df, "llm_paired_aggregate": pair_agg,
              "baseline_by_split": base_df, "baseline_aggregate": base_agg,
              "baseline_vs_llm": vs_df, "baseline_cost": cost_df}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    llm_cost = pd.read_csv(K1 / "cost_latency.csv", encoding="utf-8-sig")
    llm_cost.to_csv(OUT / "llm_cost_exp15.csv", index=False, encoding="utf-8-sig")
    done = sorted(llm_enc, key=SPLITS.index)
    (OUT / "tables.md").write_text(markdown(llm_agg, pair_agg, base_agg, vs_df, done), encoding="utf-8")
    C.save_json(OUT / "summary.json", {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "llm_splits_done": done,
        "conventions": {"invalid": "counted as incorrect", "f1": "mean over the four valid labels",
                        "delta": "second minus first", "mcnemar": "exact (09_stats.mcnemar)",
                        "holm": "within each split: 10 LLM pairs; baseline-vs-LLM pairs as a separate family",
                        "sd": "sample SD (ddof = 1) across splits",
                        "bootstrap": {"n_boot": args.bootstrap, "seed": E.SEED, "ci": "percentile 2.5/97.5",
                                      "scheme": "per split; indices shared by all LLM conditions and "
                                                "baselines in that split (same as 17_)"}},
        "exp15_matches_17_stats": not problems,
        "problems": problems, "notes": notes,
        "tables": {k: f"{k}.csv" for k in tables},
    })
    for n in notes:
        C.log(f"  註：{n}")
    if problems:
        C.die("主切分與 17_ 不一致：" + "；".join(problems[:5]))
    C.log(f"完成：{OUT}（LLM 切分 {done}；baseline 紀錄 {len(base_df)} 筆）")


if __name__ == "__main__":
    main()
