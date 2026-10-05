"""v1（15_）與 v2（27_，aug_v2）事後分析結果的並列比對（issue #7、#8、#10、#13）。

只讀兩邊已落地的輸出，不重算統計，也不呼叫 LLM：
* v1：CUSTOM_PROMPT_EVAL/（凍結，唯讀）
* v2：MAIN_EVAL_AUGV2/（17_、18_、23_ 以 --profile v2 產生）

兩版的 C3／C5 條件 ID 不同（rag_aug_* vs rag_augv2_*），一律以代號 C1–C5 合併，
配對比較以 first、second 合併；每列附兩版的條件 ID 與顯示名稱（兩版顯示名稱相同）。
欄名慣例：<指標>_v1、<指標>_v2、delta_<指標>（v2 減 v1）。

輸出到 --out-dir（預設 MAIN_EVAL_AUGV2/v1_v2_posthoc/）：

統計（#7）
* conditions.csv        各條件 accuracy、macro-F1、weighted-F1、invalid rate（含 CI）
* paired.csv            10 組配對的 Δ（含 CI）、n01／n10、McNemar、Holm；significance_changed
* per_class.csv         各類 precision／recall／F1

無效回覆（#8）
* invalid.csv           invalid rate（含 CI）、strict 與 valid-only 指標
* invalid_paired.csv    「是否無效」的配對 McNemar ＋ Holm；significance_changed
* invalid_reasons.csv   invalid_reason 分布（某版沒有的原因記為 0）

錯誤分析（#10）
* error_helped_hurt.csv      C1→C2 答對翻轉（RAG 有幫助／有害）的筆數與拆分
* error_top_confusions.csv   C4／C5 的錯誤類型（真實 → 預測，含 INVALID）與兩版排名
* error_retrieval_copy.csv   照抄率與檢索標籤正確率（兩版共同欄位；v2 另附 k 篇多數標籤的欄位）
* v1_examples_in_v2.csv      v1 examples.md 的每個範例在 v2 是否仍屬同一組
* v1_groups_overlap.json     v1 各組全部樣本在 v2 仍屬同組的筆數

標籤解析（#13）
* label_parsing.csv          strict 與 first_match 兩種規則在各條件的差異
* label_parsing_paired.csv   兩種規則下 10 組配對的 Holm 結論與是否翻轉

全部彙整於 summary.json（含每個輸入檔的 sha256）。寫檔前逐格核對：輸出的每個 v1／v2 值
都必須等於來源檔的對應格。

用法：
    python 28_v1_v2_posthoc_compare.py [--out-dir 目錄]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

V1 = E.get_profile("v1")
V2 = E.get_profile("v2")
DEFAULT_OUT = E.V2_OUT / "v1_v2_posthoc"
VERSIONS = (("v1", V1), ("v2", V2))
INPUTS: dict[str, dict[str, str]] = {"v1": {}, "v2": {}}


# ---------------------------------------------------------------- 讀檔

def read_csv(version: str, rel: str) -> pd.DataFrame:
    P = V1 if version == "v1" else V2
    path = P.in_dir / rel
    if not path.exists():
        C.die(f"找不到 {version} 的 {path}")
    INPUTS[version][rel] = C.sha256_file(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(version: str, rel: str) -> dict:
    P = V1 if version == "v1" else V2
    path = P.in_dir / rel
    if not path.exists():
        C.die(f"找不到 {version} 的 {path}")
    INPUTS[version][rel] = C.sha256_file(path)
    return C.load_json(path)


def check_display_names() -> None:
    if list(V1.codes) != list(V2.codes):
        C.die("v1 與 v2 的代號不同")
    for code in V1.codes:
        if V1.display(code) != V2.display(code):
            C.die(f"{code} 的顯示名稱在 v1、v2 不同：{V1.display(code)} vs {V2.display(code)}")


def code_columns(code: str) -> dict:
    return {"display_name": V2.display(code),
            "condition_v1": V1.condition(code), "condition_v2": V2.condition(code)}


def comparison_display(a: str, b: str) -> str:
    return f"{V2.display(a)} → {V2.display(b)}"


# ---------------------------------------------------------------- 並列與核對

def side_by_side(rel: str, keys: list[str], cols: list[str], deltas: list[str] = (),
                 how: str = "inner", v2_only: list[str] = ()) -> pd.DataFrame:
    """以 keys 合併兩版的 rel，cols 加上 _v1／_v2 字尾，deltas 另算 v2 − v1。"""
    src = {v: read_csv(v, rel) for v, _ in VERSIONS}
    for v, df in src.items():
        missing = [c for c in keys + cols if c not in df.columns]
        if missing:
            C.die(f"{v} 的 {rel} 缺少欄位：{missing}")
        if df.duplicated(keys).any():
            C.die(f"{v} 的 {rel} 以 {keys} 合併時有重複列")
    if how == "inner":
        k1 = set(map(tuple, src["v1"][keys].astype(str).values))
        k2 = set(map(tuple, src["v2"][keys].astype(str).values))
        if k1 != k2:
            C.die(f"{rel} 的 {keys} 在 v1、v2 不同：只在 v1 {sorted(k1 - k2)}，只在 v2 {sorted(k2 - k1)}")
    left = src["v1"][keys + cols].rename(columns={c: f"{c}_v1" for c in cols})
    right = src["v2"][keys + cols + list(v2_only)].rename(
        columns={c: f"{c}_v2" for c in list(cols) + list(v2_only)})
    df = left.merge(right, on=keys, how=how, validate="one_to_one")
    for v, _ in VERSIONS:
        verify(df, src[v], keys, cols + (list(v2_only) if v == "v2" else []), v, rel)
    for c in deltas:
        df[f"delta_{c}"] = df[f"{c}_v2"] - df[f"{c}_v1"]
    return df


def verify(df: pd.DataFrame, src: pd.DataFrame, keys: list[str], cols: list[str],
           version: str, rel: str) -> None:
    """輸出中每個 <欄>_<版本> 必須等於來源檔以 keys 對齊後的同一格（含 NaN）。"""
    present = df.dropna(subset=[f"{cols[0]}_{version}"]) if cols else df
    ref = src.set_index(keys).loc[pd.MultiIndex.from_frame(present[keys])
                                  if len(keys) > 1 else present[keys[0]]]
    for c in cols:
        a = present[f"{c}_{version}"].to_numpy()
        b = ref[c].to_numpy()
        same = [(x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)]
        if not all(same):
            C.die(f"{rel} 的 {c}（{version}）與來源不一致")


def add_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame([{"code": code, **code_columns(code)} for code in df["code"]])
    return pd.concat([df[["code"]], meta.drop(columns="code"), df.drop(columns="code")], axis=1)


def add_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(2, "comparison_display", [comparison_display(a, b)
                                        for a, b in zip(df["first"], df["second"])])
    return df


def save_csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, encoding="utf-8-sig")
    C.log(f"已寫入 {out / name}")


# ---------------------------------------------------------------- #7 統計

METRICS = ("accuracy", "macro_f1", "weighted_f1", "invalid_rate")


def with_ci(names) -> list[str]:
    return [c for m in names for c in (m, f"{m}_ci_low", f"{m}_ci_high")]


def conditions_table() -> pd.DataFrame:
    cols = ["n", "correct_count", "invalid_count"] + with_ci(METRICS)
    df = side_by_side("stats_conditions.csv", ["code"], cols,
                      deltas=["correct_count", "invalid_count", *METRICS])
    return add_code_columns(df)


def paired_table() -> pd.DataFrame:
    cols = (["accuracy_first", "accuracy_second"]
            + with_ci(["delta_accuracy", "delta_macro_f1", "delta_weighted_f1"])
            + ["n01", "n10", "n_discordant", "mcnemar_p", "mcnemar_p_holm",
               "significant_after_holm", "relative_error_reduction_pct"])
    df = side_by_side("stats_paired.csv", ["first", "second"], cols)
    df["significance_changed"] = df["significant_after_holm_v1"] != df["significant_after_holm_v2"]
    for v, _ in VERSIONS:
        lo, hi = df[f"delta_macro_f1_ci_low_{v}"], df[f"delta_macro_f1_ci_high_{v}"]
        df[f"delta_macro_f1_ci_excludes_zero_{v}"] = (lo > 0) | (hi < 0)
    return add_pair_columns(df)


def per_class_table() -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "support"]
    df = side_by_side("stats_per_class.csv", ["code", "label"], cols,
                      deltas=["precision", "recall", "f1"])
    return add_code_columns(df)


# ---------------------------------------------------------------- #8 無效回覆

def invalid_table() -> pd.DataFrame:
    cols = (["n", "invalid_count", "valid_count"] + with_ci(["invalid_rate"])
            + ["strict_accuracy", "strict_macro_f1", "valid_only_accuracy", "valid_only_macro_f1"])
    df = side_by_side("stats_invalid.csv", ["code"], cols,
                      deltas=["invalid_count", "invalid_rate", "strict_accuracy", "strict_macro_f1",
                              "valid_only_accuracy", "valid_only_macro_f1"])
    return add_code_columns(df)


def invalid_paired_table() -> pd.DataFrame:
    cols = ["invalid_rate_first", "invalid_rate_second", "delta_invalid_rate",
            "only_first_invalid", "only_second_invalid", "mcnemar_p", "mcnemar_p_holm",
            "significant_after_holm"]
    df = side_by_side("stats_invalid_paired.csv", ["first", "second"], cols)
    df["significance_changed"] = df["significant_after_holm_v1"] != df["significant_after_holm_v2"]
    return add_pair_columns(df)


def invalid_reasons_table() -> pd.DataFrame:
    cols = ["count", "share_of_invalid", "share_of_all"]
    df = side_by_side("stats_invalid_reasons.csv", ["code", "invalid_reason"], cols, how="outer")
    for v, _ in VERSIONS:                       # 某版沒有這個原因＝0 筆
        for c in cols:
            df[f"{c}_{v}"] = df[f"{c}_{v}"].fillna(0)
        df[f"count_{v}"] = df[f"count_{v}"].astype(int)
    df["delta_count"] = df["count_v2"] - df["count_v1"]
    df = df.sort_values(["code", "count_v2", "count_v1"], ascending=[True, False, False],
                        kind="stable").reset_index(drop=True)
    return add_code_columns(df)


# ---------------------------------------------------------------- #10 錯誤分析

GROUP_KEYS = {"helped": "rag_helped_C1wrong_C2right", "hurt": "rag_hurt_C1right_C2wrong"}
GROUP_DISPLAY = {"helped": "RAG 有幫助（C1 錯 → C2 對）", "hurt": "RAG 有害（C1 對 → C2 錯）"}


def helped_hurt_table() -> pd.DataFrame:
    summary = {v: read_json(v, "error_analysis/error_analysis_summary.json") for v, _ in VERSIONS}
    rows = []
    for group, key in GROUP_KEYS.items():
        fields = list(summary["v1"][key])
        if fields != list(summary["v2"][key]):
            C.die(f"error_analysis_summary.json 的 {key} 欄位在 v1、v2 不同")
        row = {"group": group, "group_display": GROUP_DISPLAY[group]}
        for f in fields:
            a, b = summary["v1"][key][f], summary["v2"][key][f]
            row.update({f"{f}_v1": a, f"{f}_v2": b, f"delta_{f}": b - a})
        rows.append(row)
    return pd.DataFrame(rows)


def top_confusions_table() -> pd.DataFrame:
    frames = []
    for code in ("C4", "C5"):
        rel = f"error_analysis/confusions_{code}.csv"
        cols = ["count", "share_of_errors", "share_of_true_class"]
        df = side_by_side(rel, ["code", "true_label", "pred_label"], cols, how="outer")
        for v, _ in VERSIONS:
            for c in cols:
                df[f"{c}_{v}"] = df[f"{c}_{v}"].fillna(0)
            df[f"count_{v}"] = df[f"count_{v}"].astype(int)
            df[f"rank_{v}"] = (df[f"count_{v}"].rank(ascending=False, method="min")
                               .where(df[f"count_{v}"] > 0).astype("Int64"))
        df["delta_count"] = df["count_v2"] - df["count_v1"]
        frames.append(df.sort_values(["count_v2", "count_v1"], ascending=False, kind="stable"))
    return add_code_columns(pd.concat(frames, ignore_index=True))


def retrieval_copy_table() -> pd.DataFrame:
    v1_cols = list(read_csv("v1", "error_analysis/retrieval_copy_stats.csv").columns)
    v2_cols = list(read_csv("v2", "error_analysis/retrieval_copy_stats.csv").columns)
    common_cols = [c for c in v1_cols if c not in ("code", "condition")]
    extra = [c for c in v2_cols if c not in v1_cols and c != "display_name"]
    numeric = [c for c in common_cols if c != "n"]
    df = side_by_side("error_analysis/retrieval_copy_stats.csv", ["code"], common_cols,
                      deltas=numeric, v2_only=extra)
    return add_code_columns(df)


def pred_name(record: dict) -> str:
    return "INVALID" if record["pred_label_id"] is None else record["pred_label"]


def parse_v1_examples() -> list[dict]:
    """v1 examples.md 的範例 id 與所屬組別（依小節標題）。"""
    rel = "error_analysis/examples.md"
    path = V1.in_dir / rel
    INPUTS["v1"][rel] = C.sha256_file(path)
    group, out = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 範例：RAG 有幫助"):
            group = {"group": "helped", "code": "C2", "true": None, "pred": None}
        elif line.startswith("## 範例：RAG 有害"):
            group = {"group": "hurt", "code": "C2", "true": None, "pred": None}
        elif m := re.match(r"^### (C[45])：(\S+) → (\S+)（", line):
            code, true, pred = m.groups()
            group = {"group": f"{code} {true}→{pred}", "code": code, "true": true, "pred": pred}
        elif line.startswith("## "):
            group = None
        elif (m := re.match(r"^#### `(test_\d+)`", line)) and group:
            out.append({**group, "id": m.group(1)})
    if not out:
        C.die(f"無法從 {path} 讀出範例 id")
    return out


def in_group(group: dict, preds: dict, rid: str) -> bool:
    if group["group"] in GROUP_KEYS:
        c1 = preds["C1"][rid]["correct"]
        c2 = preds["C2"][rid]["correct"]
        return (not c1 and c2) if group["group"] == "helped" else (c1 and not c2)
    r = preds[group["code"]][rid]
    return r["true_label"] == group["true"] and pred_name(r) == group["pred"]


def examples_tables() -> tuple[pd.DataFrame, dict]:
    preds = {}
    for v, P in VERSIONS:
        loaded = E.load_predictions(P)
        preds[v] = {code: {str(r["id"]): r for r in recs} for code, recs in loaded.items()}
        INPUTS[v].update({f"predictions_{P.condition(code)}.jsonl": sha
                          for code, sha in E.predictions_sha256(P).items()})

    examples = parse_v1_examples()
    rows = []
    for ex in examples:
        rid = ex["id"]
        row = {"group": ex["group"], "id": rid, "true_label": preds["v1"]["C1"][rid]["true_label"]}
        for code in ("C1", "C2", "C4", "C5"):
            for v, _ in VERSIONS:
                row[f"pred_{code}_{v}"] = pred_name(preds[v][code][rid])
        row["in_group_v1"] = in_group(ex, preds["v1"], rid)
        row["still_holds_v2"] = in_group(ex, preds["v2"], rid)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df["in_group_v1"].all():
        C.die("v1 examples.md 的範例與 v1 預測不符：" + ", ".join(df.loc[~df["in_group_v1"], "id"]))

    # 各組全部樣本：v1 與 v2 的組員、交集
    overlap = {}
    groups = [{"group": g, "code": "C2", "true": None, "pred": None} for g in GROUP_KEYS]
    groups += [{k: ex[k] for k in ("group", "code", "true", "pred")}
               for ex in examples if ex["group"] not in GROUP_KEYS]
    seen = set()
    order = [str(i) for i in E.load_test()["id"]]
    for g in groups:
        if g["group"] in seen:
            continue
        seen.add(g["group"])
        members = {v: {rid for rid in order if in_group(g, preds[v], rid)} for v, _ in VERSIONS}
        both = members["v1"] & members["v2"]
        overlap[g["group"]] = {
            "n_v1": len(members["v1"]), "n_v2": len(members["v2"]), "n_both": len(both),
            "share_of_v1_still_in_v2": len(both) / len(members["v1"]) if members["v1"] else None,
        }
    summary = {v: read_json(v, "error_analysis/error_analysis_summary.json") for v, _ in VERSIONS}
    for group, key in GROUP_KEYS.items():
        for v, _ in VERSIONS:
            if overlap[group][f"n_{v}"] != summary[v][key]["n"]:
                C.die(f"{group} 的 {v} 筆數 {overlap[group][f'n_{v}']} 與 18_ 的 "
                      f"{summary[v][key]['n']} 不同")
    for v, _ in VERSIONS:                       # 與 18_ 的 jsonl 組員一致
        for group, key in GROUP_KEYS.items():
            rel = f"error_analysis/{key}.jsonl"
            P = V1 if v == "v1" else V2
            ids = {str(r["id"]) for r in C.read_jsonl(P.in_dir / rel)}
            INPUTS[v][rel] = C.sha256_file(P.in_dir / rel)
            if len(ids) != overlap[group][f"n_{v}"]:
                C.die(f"{v} 的 {rel} 筆數與重算的組員不同")
    return df, overlap


# ---------------------------------------------------------------- #13 標籤解析

def label_parsing_table() -> pd.DataFrame:
    cols = ["changed_rows", "changed_now_correct", "changed_now_wrong_label",
            "invalid_strict", "invalid_first_match", "accuracy_strict", "accuracy_first_match",
            "delta_accuracy", "delta_accuracy_ci_low", "delta_accuracy_ci_high",
            "macro_f1_strict", "macro_f1_first_match", "delta_macro_f1", "mcnemar_p"]
    df = side_by_side("label_parsing/parse_rule_within_condition.csv", ["code"], cols)
    return add_code_columns(df)


def label_parsing_paired_table() -> pd.DataFrame:
    cols = ["delta_accuracy_strict", "delta_accuracy_first_match",
            "mcnemar_p_holm_strict", "mcnemar_p_holm_first_match",
            "significant_after_holm_strict", "significant_after_holm_first_match",
            "holm_significance_flipped"]
    df = side_by_side("label_parsing/parse_rule_paired.csv", ["first", "second"], cols)
    return add_pair_columns(df)


# ---------------------------------------------------------------- main

def pair_list(df: pd.DataFrame, mask) -> list[str]:
    return [f"{a}→{b}" for a, b in zip(df.loc[mask, "first"], df.loc[mask, "second"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    frozen = E.V1_OUT.resolve()
    if out == frozen or frozen in out.parents:
        C.die(f"--out-dir {out} 位於凍結的 v1 目錄 {frozen} 之內")
    out.mkdir(parents=True, exist_ok=True)

    check_display_names()
    stats = {v: read_json(v, "stats.json") for v, _ in VERSIONS}
    split = {v: stats[v]["inputs"]["test_split_sha256"] for v, _ in VERSIONS}
    if split["v1"] != split["v2"] or split["v1"] != C.sha256_file(E.TEST_PATH):
        C.die(f"v1、v2 的 test split sha256 與現行 test.csv 不全相同：{split}")

    tables = {
        "conditions.csv": conditions_table(),
        "paired.csv": paired_table(),
        "per_class.csv": per_class_table(),
        "invalid.csv": invalid_table(),
        "invalid_paired.csv": invalid_paired_table(),
        "invalid_reasons.csv": invalid_reasons_table(),
        "error_helped_hurt.csv": helped_hurt_table(),
        "error_top_confusions.csv": top_confusions_table(),
        "error_retrieval_copy.csv": retrieval_copy_table(),
        "label_parsing.csv": label_parsing_table(),
        "label_parsing_paired.csv": label_parsing_paired_table(),
    }
    examples, overlap = examples_tables()
    tables["v1_examples_in_v2.csv"] = examples
    for name, df in tables.items():
        save_csv(df, out, name)
    C.save_json(out / "v1_groups_overlap.json", overlap)
    C.log(f"已寫入 {out / 'v1_groups_overlap.json'}")

    paired, inv_paired = tables["paired.csv"], tables["invalid_paired.csv"]
    lp, lpp = tables["label_parsing.csv"], tables["label_parsing_paired.csv"]
    conf = tables["error_top_confusions.csv"]
    summary = {
        "run": E.RUN,
        "script": "src/28_v1_v2_posthoc_compare.py",
        "issues": [7, 8, 10, 13],
        "procedure": ("只讀 v1（CUSTOM_PROMPT_EVAL/，15_）與 v2（MAIN_EVAL_AUGV2/，27_）既有的 17_、"
                      "18_、23_ 輸出，以代號 C1–C5（配對以 first、second）合併並列；不重算統計。"),
        "v1": {"experiment": V1.experiment, "top_k": V1.top_k, "corpora": list(V1.corpora)},
        "v2": {"experiment": V2.experiment, "top_k": V2.top_k, "corpora": list(V2.corpora)},
        "display_names": {code: {"display_name": V2.display(code),
                                 "condition_v1": V1.condition(code),
                                 "condition_v2": V2.condition(code)} for code in V2.codes},
        "test_split_sha256": split["v1"],
        "inputs_sha256": INPUTS,
        "findings": {
            "stats_paired_significance_changed": pair_list(paired, paired["significance_changed"]),
            "stats_paired_significant_v1": pair_list(paired, paired["significant_after_holm_v1"]),
            "stats_paired_significant_v2": pair_list(paired, paired["significant_after_holm_v2"]),
            "invalid_paired_significance_changed":
                pair_list(inv_paired, inv_paired["significance_changed"]),
            "invalid_paired_not_significant_v1":
                pair_list(inv_paired, ~inv_paired["significant_after_holm_v1"]),
            "invalid_paired_not_significant_v2":
                pair_list(inv_paired, ~inv_paired["significant_after_holm_v2"]),
            "label_parsing_max_abs_delta_accuracy": {
                v: float(lp[f"delta_accuracy_{v}"].abs().max()) for v, _ in VERSIONS},
            "label_parsing_holm_flipped": {
                v: pair_list(lpp, lpp[f"holm_significance_flipped_{v}"]) for v, _ in VERSIONS},
            "invalid_in_top5_confusions": {
                v: sorted(conf.loc[(conf["pred_label"] == "INVALID")
                                   & (conf[f"rank_{v}"] <= 5), "code"].tolist())
                for v, _ in VERSIONS},
            "v1_examples_still_hold_in_v2": {
                "n_examples": int(len(examples)),
                "n_still_hold": int(examples["still_holds_v2"].sum()),
                "by_group": {g: f"{int(d['still_holds_v2'].sum())}/{len(d)}"
                             for g, d in examples.groupby("group", sort=False)},
            },
            "v1_groups_overlap": overlap,
        },
    }
    C.save_json(out / "summary.json", summary)
    C.log(f"已寫入 {out / 'summary.json'}")


if __name__ == "__main__":
    main()
