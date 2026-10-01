"""15_（CUSTOM_PROMPT_EVAL）的統計結果：論文 Table IV／VI／VII 與無效回覆分析。

只讀 15_ 已落地的逐筆預測，不呼叫 LLM。輸出到 CUSTOM_PROMPT_EVAL/：

Table VI／VII（issue #7）
* stats_conditions.csv     各條件 accuracy、macro-F1、weighted-F1、invalid rate ＋ bootstrap 95% CI
* stats_paired.csv         10 組配對：Δacc、ΔmacroF1（含 CI）、n01／n10、exact McNemar、Holm、相對錯誤率下降
* stats_per_class.csv      各類 precision／recall／F1 與 macro／weighted 平均（Table IV 格式）
* stats_paper_check.csv    與論文 Table VI／VII 逐格比對

無效回覆（issue #8）
* stats_invalid.csv          invalid rate 與 CI；valid-only accuracy／macro-F1（敏感度分析）
* stats_invalid_paired.csv   「是否無效」的 exact McNemar ＋ Holm
* stats_invalid_reasons.csv  各條件的 invalid_reason 分佈

全部彙整於 stats.json。慣例：無效回應計為答錯；差值一律為後者減前者；
bootstrap 10,000 次、seed 42，配對時兩個條件共用同一組重抽索引。

用法：
    python 17_custom_prompt_stats.py [--bootstrap 10000]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

STATS09 = C.load_module("09_stats")

# 論文 R1 稿 Table VI：(accuracy, CI, macro-F1, CI, invalid)。數字以字串保存，
# 比對時依字串的位數決定四捨五入容許誤差。
PAPER_TABLE_VI = {
    "C1": ("0.5395", ("0.5175", "0.5611"), "0.6225", ("0.6010", "0.6435"), "551"),
    "C2": ("0.6176", ("0.5961", "0.6396"), "0.6830", ("0.6623", "0.7031"), "437"),
    "C3": ("0.6136", ("0.5921", "0.6356"), "0.6729", ("0.6524", "0.6932"), "396"),
    "C4": ("0.7558", ("0.7367", "0.7743"), "0.7720", ("0.7536", "0.7897"), "100"),
    "C5": ("0.7533", ("0.7347", "0.7718"), "0.7680", ("0.7494", "0.7857"), "89"),
}
# Table VII：(Δacc, CI, ΔF1, CI, n01, n10, McNemar p, Holm p, Err %)。
PAPER_TABLE_VII = {
    ("C1", "C2"): ("0.0781", ("0.0546", "0.1016"), "0.0604", ("0.0384", "0.0828"),
                   "372", "216", "1.28e-10", "5.13e-10", "17"),
    ("C1", "C3"): ("0.0741", ("0.0511", "0.0971"), "0.0504", ("0.0289", "0.0721"),
                   "355", "207", "4.49e-10", "1.35e-09", "16"),
    ("C1", "C4"): ("0.2162", ("0.1927", "0.2402"), "0.1494", ("0.1261", "0.1733"),
                   "559", "127", "1.70e-65", "1.70e-64", "47"),
    ("C1", "C5"): ("0.2137", ("0.1902", "0.2377"), "0.1454", ("0.1227", "0.1691"),
                   "556", "129", "5.25e-64", "4.73e-63", "46"),
    ("C2", "C3"): ("-0.0040", ("-0.0180", "0.0100"), "-0.0101", ("-0.0239", "0.0036"),
                   "98", "106", "0.624", "1.000", "-1"),
    ("C2", "C4"): ("0.1381", ("0.1186", "0.1577"), "0.0890", ("0.0706", "0.1071"),
                   "354", "78", "4.56e-43", "3.19e-42", "36"),
    ("C2", "C5"): ("0.1356", ("0.1156", "0.1557"), "0.0850", ("0.0661", "0.1039"),
                   "370", "99", "7.65e-38", "3.83e-37", "36"),
    ("C3", "C4"): ("0.1421", ("0.1216", "0.1622"), "0.0991", ("0.0798", "0.1186"),
                   "374", "90", "3.47e-42", "2.08e-41", "37"),
    ("C3", "C5"): ("0.1396", ("0.1206", "0.1587"), "0.0950", ("0.0772", "0.1125"),
                   "345", "66", "1.12e-46", "8.96e-46", "36"),
    ("C4", "C5"): ("-0.0025", ("-0.0140", "0.0095"), "-0.0040", ("-0.0157", "0.0078"),
                   "71", "76", "0.742", "1.000", "-1"),
}


def tolerance(paper: str) -> float:
    """論文數字的四捨五入半格：'0.5395' → 5e-5；'1.28e-10' → 5e-13；'551' → 0.5。"""
    if "e" in paper.lower():
        mantissa, exp = paper.lower().split("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return 0.5 * 10 ** (int(exp) - decimals)
    decimals = len(paper.split(".")[1]) if "." in paper else 0
    return 0.5 * 10 ** (-decimals)


def check_row(table: str, row: str, column: str, paper: str, ours: float) -> dict:
    diff = abs(float(ours) - float(paper))
    return {
        "table": table, "row": row, "column": column,
        "paper": paper, "ours": ours, "abs_diff": diff,
        "rounding_tolerance": tolerance(paper),
        "match": bool(diff <= tolerance(paper) * (1 + 1e-9)),
    }


def paper_check(cond_rows: pd.DataFrame, paired_rows: pd.DataFrame) -> pd.DataFrame:
    checks = []
    cond = cond_rows.set_index("code")
    for code, (acc, acc_ci, f1, f1_ci, inv) in PAPER_TABLE_VI.items():
        r = cond.loc[code]
        checks += [
            check_row("VI", code, "accuracy", acc, r.accuracy),
            check_row("VI", code, "accuracy_ci_low", acc_ci[0], r.accuracy_ci_low),
            check_row("VI", code, "accuracy_ci_high", acc_ci[1], r.accuracy_ci_high),
            check_row("VI", code, "macro_f1", f1, r.macro_f1),
            check_row("VI", code, "macro_f1_ci_low", f1_ci[0], r.macro_f1_ci_low),
            check_row("VI", code, "macro_f1_ci_high", f1_ci[1], r.macro_f1_ci_high),
            check_row("VI", code, "invalid_count", inv, r.invalid_count),
        ]
    paired = paired_rows.set_index(["first", "second"])
    for (a, b), vals in PAPER_TABLE_VII.items():
        r = paired.loc[(a, b)]
        name = f"{a} vs {b}"
        d_acc, acc_ci, d_f1, f1_ci, n01, n10, p, p_holm, err = vals
        checks += [
            check_row("VII", name, "delta_accuracy", d_acc, r.delta_accuracy),
            check_row("VII", name, "delta_accuracy_ci_low", acc_ci[0], r.delta_accuracy_ci_low),
            check_row("VII", name, "delta_accuracy_ci_high", acc_ci[1], r.delta_accuracy_ci_high),
            check_row("VII", name, "delta_macro_f1", d_f1, r.delta_macro_f1),
            check_row("VII", name, "delta_macro_f1_ci_low", f1_ci[0], r.delta_macro_f1_ci_low),
            check_row("VII", name, "delta_macro_f1_ci_high", f1_ci[1], r.delta_macro_f1_ci_high),
            check_row("VII", name, "n01", n01, r.n01),
            check_row("VII", name, "n10", n10, r.n10),
            check_row("VII", name, "mcnemar_p", p, r.mcnemar_p),
            check_row("VII", name, "mcnemar_p_holm", p_holm, r.mcnemar_p_holm),
            check_row("VII", name, "relative_error_reduction_pct", err,
                      r.relative_error_reduction_pct),
        ]
    return pd.DataFrame(checks)


def condition_table(enc: dict, boot_cm: dict) -> pd.DataFrame:
    rows = []
    for code, (y_true, y_pred) in enc.items():
        cm = E.confusion(y_true, y_pred)
        point = E.metrics_from_confusion(cm)
        boot = E.metrics_from_confusion(boot_cm[code])
        row = {"code": code, "condition": E.CODES[code], "description": E.DESCRIPTIONS[code],
               "n": int(len(y_true)), "correct_count": int(np.trace(cm[:, :4])),
               "invalid_count": int(cm[:, E.INVALID].sum())}
        for key in ("accuracy", "macro_f1", "weighted_f1", "invalid_rate"):
            lo, hi = E.ci95(boot[key])
            row.update({key: float(point[key]), f"{key}_ci_low": lo, f"{key}_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(enc: dict, boot_cm: dict) -> pd.DataFrame:
    rows, pvals = [], {}
    for a, b in E.PAIRS:
        ya, pa = enc[a]
        yb, pb = enc[b]
        ca, cb = pa == ya, pb == yb
        ma = E.metrics_from_confusion(E.confusion(ya, pa))
        mb = E.metrics_from_confusion(E.confusion(yb, pb))
        bma = E.metrics_from_confusion(boot_cm[a])
        bmb = E.metrics_from_confusion(boot_cm[b])
        mc = STATS09.mcnemar(ca, cb)
        err_a, err_b = 1 - ma["accuracy"], 1 - mb["accuracy"]
        row = {"first": a, "second": b,
               "comparison": f"{E.CODES[a]} -> {E.CODES[b]}",
               "accuracy_first": float(ma["accuracy"]), "accuracy_second": float(mb["accuracy"])}
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            lo, hi = E.ci95(bmb[key] - bma[key])
            row.update({f"delta_{key}": float(mb[key] - ma[key]),
                        f"delta_{key}_ci_low": lo, f"delta_{key}_ci_high": hi})
        row.update({"n01": mc["n01"], "n10": mc["n10"], "n_discordant": mc["n_discordant"],
                    "mcnemar_p": mc["p_value"],
                    "relative_error_reduction_pct": float((err_a - err_b) / err_a * 100)})
        rows.append(row)
        pvals[f"{a}_{b}"] = mc["p_value"]
    adjusted = E.holm(pvals)
    for row in rows:
        row["mcnemar_p_holm"] = adjusted[f"{row['first']}_{row['second']}"]
        row["significant_after_holm"] = bool(row["mcnemar_p_holm"] < 0.05)
    return pd.DataFrame(rows)


def per_class_table(enc: dict) -> pd.DataFrame:
    frames = []
    for code, (y_true, y_pred) in enc.items():
        df = E.per_class_report(y_true, y_pred)
        df.insert(0, "condition", E.CODES[code])
        df.insert(0, "code", code)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def invalid_tables(preds: dict, enc: dict, boot_cm: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for code, (y_true, y_pred) in enc.items():
        cm = E.confusion(y_true, y_pred)
        valid = y_pred != E.INVALID
        v_true, v_pred = y_true[valid], y_pred[valid]
        # valid-only：只在有效回覆上計分，F1 的支持度也只算有效回覆。
        v_metrics = E.metrics_from_confusion(E.confusion(v_true, v_pred))
        inv_lo, inv_hi = E.ci95(E.metrics_from_confusion(boot_cm[code])["invalid_rate"])
        rows.append({
            "code": code, "condition": E.CODES[code], "n": int(len(y_true)),
            "invalid_count": int((~valid).sum()), "valid_count": int(valid.sum()),
            "invalid_rate": float((~valid).mean()),
            "invalid_rate_ci_low": inv_lo, "invalid_rate_ci_high": inv_hi,
            "strict_accuracy": float(E.metrics_from_confusion(cm)["accuracy"]),
            "strict_macro_f1": float(E.metrics_from_confusion(cm)["macro_f1"]),
            "valid_only_accuracy": float(v_metrics["accuracy"]),
            "valid_only_macro_f1": float(v_metrics["macro_f1"]),
        })

    paired, pvals = [], {}
    for a, b in E.PAIRS:
        # 以「回覆有效」當作成功：n01 = 前者無效、後者有效（後者改善）。
        va = enc[a][1] != E.INVALID
        vb = enc[b][1] != E.INVALID
        mc = STATS09.mcnemar(va, vb)
        paired.append({"first": a, "second": b,
                       "comparison": f"{E.CODES[a]} -> {E.CODES[b]}",
                       "invalid_rate_first": float((~va).mean()),
                       "invalid_rate_second": float((~vb).mean()),
                       "delta_invalid_rate": float((~vb).mean() - (~va).mean()),
                       "only_first_invalid": mc["n01"], "only_second_invalid": mc["n10"],
                       "mcnemar_p": mc["p_value"]})
        pvals[f"{a}_{b}"] = mc["p_value"]
    adjusted = E.holm(pvals)
    for row in paired:
        row["mcnemar_p_holm"] = adjusted[f"{row['first']}_{row['second']}"]
        row["significant_after_holm"] = bool(row["mcnemar_p_holm"] < 0.05)

    reasons = []
    for code, records in preds.items():
        counts = pd.Series([r["invalid_reason"] for r in records if r["invalid"]]).value_counts()
        for reason, count in counts.items():
            reasons.append({"code": code, "condition": E.CODES[code], "invalid_reason": reason,
                            "count": int(count), "share_of_invalid": float(count / counts.sum()),
                            "share_of_all": float(count / len(records))})
    return pd.DataFrame(rows), pd.DataFrame(paired), pd.DataFrame(reasons)


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(E.OUT / name, index=False, encoding="utf-8-sig")
    C.log(f"已寫入 {E.OUT / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=E.N_BOOT)
    args = ap.parse_args()

    preds = E.load_predictions()
    enc = {code: E.encode(records) for code, records in preds.items()}
    for y_true, y_pred in enc.values():
        E.check_against_sklearn(y_true, y_pred)

    idx = E.bootstrap_indices(E.N_TEST, args.bootstrap, E.SEED)
    boot_cm = {code: E.boot_confusions(y, p, idx) for code, (y, p) in enc.items()}

    cond = condition_table(enc, boot_cm)
    paired = paired_table(enc, boot_cm)
    per_class = per_class_table(enc)
    inv, inv_paired, inv_reasons = invalid_tables(preds, enc, boot_cm)
    check = paper_check(cond, paired)

    save_csv(cond, "stats_conditions.csv")
    save_csv(paired, "stats_paired.csv")
    save_csv(per_class, "stats_per_class.csv")
    save_csv(check, "stats_paper_check.csv")
    save_csv(inv, "stats_invalid.csv")
    save_csv(inv_paired, "stats_invalid_paired.csv")
    save_csv(inv_reasons, "stats_invalid_reasons.csv")

    def records(df):
        return [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                 for k, v in row.items()} for row in df.to_dict("records")]

    C.save_json(E.OUT / "stats.json", {
        "run": E.RUN,
        "experiment": "CUSTOM_PROMPT_EVAL",
        "script": "src/17_custom_prompt_stats.py",
        "codes": E.CODES,
        "conventions": {
            "correctness": "strict（無效回應計為答錯，分母一律 1998）",
            "f1": "四個有效類別上平均；INVALID 視為預測錯誤，不是第五類",
            "difference": "後者減前者",
            "bootstrap": {"n_boot": args.bootstrap, "seed": E.SEED,
                          "scheme": "numpy default_rng(seed)，每次 rng.integers(0, n, n)；"
                                    "所有條件與配對共用同一組重抽索引",
                          "ci": "percentile 2.5 / 97.5"},
            "mcnemar": "exact（scipy binomtest，09_stats.mcnemar）",
            "multiple_comparison": "Holm–Bonferroni，10 組（accuracy 與 invalid 各自校正）",
            "relative_error_reduction": "(err_first − err_second) / err_first × 100",
            "valid_only": "只在有效回覆上計分（敏感度分析）",
        },
        "inputs": {
            "test_split_sha256": C.sha256_file(E.TEST_PATH),
            "predictions_sha256": {
                code: C.sha256_file(E.OUT / f"predictions_{cond_name}.jsonl")
                for code, cond_name in E.CODES.items()
            },
        },
        "paper_check_summary": {
            "cells": int(len(check)),
            "matched": int(check["match"].sum()),
            "mismatched": records(check[~check["match"]]),
        },
        "conditions": records(cond),
        "paired": records(paired),
        "per_class": records(per_class),
        "invalid": records(inv),
        "invalid_paired": records(inv_paired),
        "invalid_reasons": records(inv_reasons),
    })
    C.log(f"已寫入 {E.OUT / 'stats.json'}")

    C.log("")
    for r in cond.itertuples():
        C.log(f"{r.code} acc={r.accuracy:.4f} [{r.accuracy_ci_low:.4f},{r.accuracy_ci_high:.4f}] "
              f"macroF1={r.macro_f1:.4f} [{r.macro_f1_ci_low:.4f},{r.macro_f1_ci_high:.4f}] "
              f"wF1={r.weighted_f1:.4f} invalid={r.invalid_count}")
    for r in paired.itertuples():
        C.log(f"{r.first}->{r.second} Δacc={r.delta_accuracy:+.4f} "
              f"[{r.delta_accuracy_ci_low:+.4f},{r.delta_accuracy_ci_high:+.4f}] "
              f"{r.n01}/{r.n10} p={r.mcnemar_p:.3g} holm={r.mcnemar_p_holm:.3g} "
              f"err={r.relative_error_reduction_pct:.1f}%")
    for r in inv_paired.itertuples():
        C.log(f"invalid {r.first}->{r.second} {r.only_first_invalid}/{r.only_second_invalid} "
              f"p={r.mcnemar_p:.3g} holm={r.mcnemar_p_holm:.3g}")
    C.log(f"論文比對：{int(check['match'].sum())}/{len(check)} 格吻合")
    for r in check[~check["match"]].itertuples():
        C.log(f"  不符 Table {r.table} {r.row} {r.column}: paper={r.paper} ours={r.ours:.6g}")


if __name__ == "__main__":
    main()
