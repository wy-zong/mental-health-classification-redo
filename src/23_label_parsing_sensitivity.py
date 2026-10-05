"""標籤解析規則的敏感度分析（issue #13）。

論文 Experimental Results 寫的是：回覆「scanned case-insensitively in the fixed order
Normal, Depression, Anxiety, and Bipolar; the first matching label was returned」。
15_ 實際用的是 common.parse_label：先比對數字，再比對標籤文字且只接受唯一命中，
多個標籤同時命中算 INVALID（multi_label）。

v2（27_）沿用 common.parse_label。這支腳本只讀已落地的 raw_response，不呼叫 LLM：

1. 先用 common.parse_label 重新解析，確認與存檔的 pred_label_id、parse_reason 逐筆相同；
2. 以論文描述的規則（README 記錄的原始實作：不分大小寫的子字串比對，依固定順序取
   第一個命中，不比對數字）重新解析；
3. 兩種規則下 C1–C5 的 accuracy、macro-F1、weighted-F1、invalid rate（bootstrap 95% CI），
   同一條件內兩規則的差異（exact McNemar），以及用 first-match 重算的 Table VII。

輸出到 <輸出目錄>/label_parsing/（v2 預設 MAIN_EVAL_AUGV2/；v1 的 CUSTOM_PROMPT_EVAL/ 是凍結的
存檔，必須以 --out-dir 另外指定；與論文 Table VI 的點估計比對只在 v1 做）：
* parse_rule_conditions.csv          兩種規則 × C1–C5 的指標與 CI
* parse_rule_within_condition.csv    同一條件內 first-match 減 strict 的差異、McNemar
* parse_rule_paired.csv              兩種規則下的 Table VII 10 組配對，標出 Holm 顯著性是否翻轉
* parse_rule_changed_rows.jsonl      兩種規則結果不同的逐筆樣本
* summary.json

慣例同 17_：無效回應計為答錯；差值一律為後者減前者；bootstrap 10,000 次、seed 42，
所有條件與兩種規則共用同一組重抽索引。

用法：
    python 23_label_parsing_sensitivity.py --profile v2 [--bootstrap 10000]
    python 23_label_parsing_sensitivity.py --profile v1 --out-dir 目錄
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

S17 = C.load_module("17_custom_prompt_stats")
STATS09 = S17.STATS09

RULES = ("strict", "first_match")
RULE_DESCRIPTIONS = {
    "strict": "common.parse_label（15_ 實際使用）：先比對數字；標籤文字以 \\b 字界、"
              "不分大小寫比對，只接受唯一命中，多重命中為 INVALID(multi_label)",
    "first_match": "論文描述／README 記錄的原始實作：for label in [Normal, Depression, "
                   "Anxiety, Bipolar]: if label.lower() in raw.lower() → 取第一個命中；"
                   "子字串比對、不比對數字",
}


def parse_first_match(raw: str | None) -> tuple[int | None, str]:
    """論文描述的解析規則。"""
    if raw is None or not raw.strip():
        return None, "empty"
    text = raw.lower()
    for label in C.LABELS:
        if label.lower() in text:
            return C.LABEL_TO_ID[label], "first_match"
    return None, "no_label"


def verify_strict(P: E.Profile, preds: dict[str, list[dict]]) -> None:
    """用 common.parse_label 重新解析，必須與存檔的結果逐筆相同。"""
    for code, records in preds.items():
        bad = [r["id"] for r in records
               if C.parse_label(r["raw_response"]) != (r["pred_label_id"], r["parse_reason"])]
        if bad:
            C.die(f"{code}：{len(bad)} 筆以 common.parse_label 重新解析後與存檔不符，"
                  f"例如 {bad[:5]}")
    C.log(f"strict 規則重新解析：C1–C5 全部與 {P.experiment} 存檔一致")


def encode_first_match(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y_true = np.array([C.LABEL_TO_ID[r["true_label"]] for r in records], dtype=np.int64)
    parsed = [parse_first_match(r["raw_response"]) for r in records]
    y_pred = np.array([E.INVALID if p is None else p for p, _ in parsed], dtype=np.int64)
    return y_true, y_pred, [reason for _, reason in parsed]


def label_name(i: int) -> str:
    return "INVALID" if i == E.INVALID else C.ID_TO_LABEL[int(i)]


def within_condition(P: E.Profile, enc: dict, boot: dict) -> pd.DataFrame:
    rows = []
    for code in P.codes:
        y, ps = enc["strict"][code]
        _, pf = enc["first_match"][code]
        ms = E.metrics_from_confusion(E.confusion(y, ps))
        mf = E.metrics_from_confusion(E.confusion(y, pf))
        bs = E.metrics_from_confusion(boot["strict"][code])
        bf = E.metrics_from_confusion(boot["first_match"][code])
        mc = STATS09.mcnemar(ps == y, pf == y)
        changed = ps != pf
        row = {"code": code, "condition": P.condition(code),
               "changed_rows": int(changed.sum()),
               "changed_invalid_to_label": int((changed & (ps == E.INVALID)).sum()),
               "changed_label_to_label": int((changed & (ps != E.INVALID) & (pf != E.INVALID)).sum()),
               "changed_label_to_invalid": int((changed & (pf == E.INVALID)).sum()),
               "changed_now_correct": int((changed & (pf == y)).sum()),
               "changed_now_wrong_label": int((changed & (pf != y) & (pf != E.INVALID)).sum()),
               "invalid_strict": int((ps == E.INVALID).sum()),
               "invalid_first_match": int((pf == E.INVALID).sum())}
        for key in ("accuracy", "macro_f1", "weighted_f1", "invalid_rate"):
            lo, hi = E.ci95(bf[key] - bs[key])
            row.update({f"{key}_strict": float(ms[key]), f"{key}_first_match": float(mf[key]),
                        f"delta_{key}": float(mf[key] - ms[key]),
                        f"delta_{key}_ci_low": lo, f"delta_{key}_ci_high": hi})
        row.update({"n01": mc["n01"], "n10": mc["n10"], "mcnemar_p": mc["p_value"]})
        row["display_name"] = P.display(code)
        rows.append(row)
    return pd.DataFrame(rows)


def changed_rows(P: E.Profile, preds: dict, enc: dict, reasons_fm: dict) -> list[dict]:
    out = []
    for code, records in preds.items():
        _, ps = enc["strict"][code]
        _, pf = enc["first_match"][code]
        for i, r in enumerate(records):
            if ps[i] == pf[i]:
                continue
            out.append({
                "code": code, "condition": P.condition(code), "id": r["id"],
                "true_label": r["true_label"],
                "strict_pred": label_name(ps[i]), "strict_reason": r["parse_reason"],
                "first_match_pred": label_name(pf[i]), "first_match_reason": reasons_fm[code][i],
                "first_match_correct": bool(pf[i] == C.LABEL_TO_ID[r["true_label"]]),
                "raw_response": r["raw_response"],
                "display_name": P.display(code),
            })
    return out


def paper_vi_check(cond: pd.DataFrame) -> list[dict]:
    """哪一種規則重現論文 Table VI 的點估計（accuracy、macro-F1、invalid 數）。"""
    out = []
    for r in cond.itertuples():
        acc, _, f1, _, inv = S17.PAPER_TABLE_VI[r.code]
        out.append({
            "rule": r.rule, "code": r.code,
            "accuracy_match": bool(abs(r.accuracy - float(acc)) <= S17.tolerance(acc)),
            "macro_f1_match": bool(abs(r.macro_f1 - float(f1)) <= S17.tolerance(f1)),
            "invalid_count_match": bool(r.invalid_count == int(inv)),
        })
    return out


def save_csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, encoding="utf-8-sig")
    C.log(f"已寫入 {out / name}")


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in row.items()} for row in df.to_dict("records")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=E.N_BOOT)
    E.add_profile_args(ap)
    args = ap.parse_args()
    P, out_dir = E.resolve(args)
    out = out_dir / "label_parsing"
    out.mkdir(parents=True, exist_ok=True)

    preds = E.load_predictions(P)
    verify_strict(P, preds)

    enc = {"strict": {}, "first_match": {}}
    reasons_fm = {}
    for code, recs in preds.items():
        enc["strict"][code] = E.encode(recs)
        y, p, reasons = encode_first_match(recs)
        if not np.array_equal(y, enc["strict"][code][0]):
            C.die(f"{code}：y_true 不一致")
        enc["first_match"][code] = (y, p)
        reasons_fm[code] = reasons
    for rule in RULES:
        for y, p in enc[rule].values():
            E.check_against_sklearn(y, p)

    idx = E.bootstrap_indices(E.N_TEST, args.bootstrap, E.SEED)
    boot = {rule: {code: E.boot_confusions(y, p, idx) for code, (y, p) in enc[rule].items()}
            for rule in RULES}

    cond = pd.concat([S17.condition_table(P, enc[rule], boot[rule]).assign(rule=rule)
                      for rule in RULES], ignore_index=True)
    cond = cond[["rule"] + [c for c in cond.columns if c != "rule"]]
    within = within_condition(P, enc, boot)

    paired = {rule: S17.paired_table(P, enc[rule], boot[rule]) for rule in RULES}
    keep = ["delta_accuracy", "delta_accuracy_ci_low", "delta_accuracy_ci_high",
            "delta_macro_f1", "delta_macro_f1_ci_low", "delta_macro_f1_ci_high",
            "n01", "n10", "mcnemar_p", "mcnemar_p_holm", "significant_after_holm"]
    both = paired["strict"][["first", "second", "comparison"] + keep].merge(
        paired["first_match"][["first", "second"] + keep], on=["first", "second"],
        suffixes=("_strict", "_first_match"))
    both["holm_significance_flipped"] = (both["significant_after_holm_strict"]
                                         != both["significant_after_holm_first_match"])
    both["comparison_display"] = [S17.comparison_display(P, a, b)
                                  for a, b in zip(both["first"], both["second"])]

    changed = changed_rows(P, preds, enc, reasons_fm)
    vi_check = paper_vi_check(cond) if P.paper_check else None

    save_csv(cond, out, "parse_rule_conditions.csv")
    save_csv(within, out, "parse_rule_within_condition.csv")
    save_csv(both, out, "parse_rule_paired.csv")
    path = out / "parse_rule_changed_rows.jsonl"
    path.unlink(missing_ok=True)
    for rec in changed:
        C.append_jsonl(path, rec)
    C.log(f"已寫入 {path}（{len(changed)} 筆）")

    reason_counts = pd.DataFrame(changed).groupby(
        ["code", "strict_reason", "first_match_pred"]).size() if changed else pd.Series(dtype=int)
    summary = {
        "run": E.RUN,
        "experiment": P.experiment,
        "script": "src/23_label_parsing_sensitivity.py",
        "issue": 13,
        "codes": P.codes,
        "rules": RULE_DESCRIPTIONS,
        "conventions": {
            "correctness": "strict（無效回應計為答錯，分母一律 1998）",
            "f1": "四個有效類別上平均；INVALID 視為預測錯誤，不是第五類",
            "difference": "後者減前者（within_condition 為 first_match 減 strict）",
            "bootstrap": {"n_boot": args.bootstrap, "seed": E.SEED,
                          "scheme": "兩種規則與所有條件共用同一組重抽索引"},
            "mcnemar": "exact（09_stats.mcnemar）",
            "multiple_comparison": "Table VII 的 10 組配對各自在規則內做 Holm",
        },
        "inputs": {
            "test_split_sha256": E.test_split_sha256(),
            "predictions_sha256": E.predictions_sha256(P),
        },
        "strict_reparse_matches_15_records": True,
    }
    if vi_check is not None:
        summary["paper_table_vi_point_check"] = vi_check
    summary.update({
        "changed_rows_by_reason": [
            {"code": k[0], "strict_reason": k[1], "first_match_pred": k[2], "count": int(v)}
            for k, v in reason_counts.items()
        ],
        "conditions": records(cond),
        "within_condition": records(within),
        "paired": records(both),
        "profile": P.name,
        "display_names": P.display_names(),
    })
    C.save_json(out / "summary.json", summary)
    C.log(f"已寫入 {out / 'summary.json'}")

    C.log("")
    for r in within.itertuples():
        C.log(f"{r.code} 變動 {r.changed_rows} 筆（其中改判正確 {r.changed_now_correct}）"
              f" acc {r.accuracy_strict:.4f} → {r.accuracy_first_match:.4f}"
              f"（Δ{r.delta_accuracy:+.4f} [{r.delta_accuracy_ci_low:+.4f},{r.delta_accuracy_ci_high:+.4f}]）"
              f" macroF1 {r.macro_f1_strict:.4f} → {r.macro_f1_first_match:.4f}"
              f" invalid {r.invalid_strict} → {r.invalid_first_match}"
              f" McNemar {r.n01}/{r.n10} p={r.mcnemar_p:.3g}")
    for r in both.itertuples():
        flag = "  ← 顯著性翻轉" if r.holm_significance_flipped else ""
        C.log(f"{r.first}->{r.second} Δacc strict={r.delta_accuracy_strict:+.4f} "
              f"first_match={r.delta_accuracy_first_match:+.4f} "
              f"holm {r.mcnemar_p_holm_strict:.3g} / {r.mcnemar_p_holm_first_match:.3g}{flag}")
    for rule in RULES if vi_check is not None else ():
        ok = [c for c in vi_check if c["rule"] == rule
              and c["accuracy_match"] and c["macro_f1_match"] and c["invalid_count_match"]]
        C.log(f"論文 Table VI 點估計：{rule} 規則重現 {len(ok)}/5 個條件")


if __name__ == "__main__":
    main()
