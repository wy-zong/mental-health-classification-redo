"""主實驗（v1：15_；v2：27_）的質性範例與錯誤分析（issue #10）。

只讀已落地的逐筆預測。輸出到 <輸出目錄>/error_analysis/（v2 預設 MAIN_EVAL_AUGV2/；
v1 的 CUSTOM_PROMPT_EVAL/ 是凍結的存檔，必須以 --out-dir 另外指定）：

* rag_helped_C1wrong_C2right.jsonl   C1 錯、C2 對（RAG 有幫助）的全部樣本
* rag_hurt_C1right_C2wrong.jsonl     C1 對、C2 錯（RAG 有害）的全部樣本
* confusions_C4.csv / confusions_C5.csv   錯誤類型排名（真實 → 預測，含 INVALID）
* retrieval_copy_stats.csv           C2–C5：照抄率、檢索標籤正確率、依檢索標籤對錯分開的準確率
                                     （以 top-1 標籤計；另附 k 篇多數標籤的版本，同票記為 tie）
* examples.md                        給人讀的範例（固定 seed 抽樣，文本截短）

同一代號在 v1、v2 指的條件不同，輸出一律附條件 ID 與顯示名稱。

examples.md 的文本來自公開的心理健康貼文資料集，仍屬敏感內容；放進論文前必須再改寫或摘要，
不可逐字引用。

用法：
    python 18_custom_prompt_error_analysis.py --profile v2 [--per-group 5] [--per-confusion 3]
    python 18_custom_prompt_error_analysis.py --profile v1 --out-dir 目錄
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

TRUNCATE = 300


def pred_name(rec: dict) -> str:
    return rec["pred_label"] or "INVALID"


def example(P: E.Profile, code_a: str, rec_a: dict, code_b: str, rec_b: dict,
            statement: str) -> dict:
    """一筆樣本在兩個條件下的完整紀錄。"""
    out = {"id": rec_a["id"], "true_label": rec_a["true_label"], "statement": statement}
    for code, rec in ((code_a, rec_a), (code_b, rec_b)):
        out[code] = {
            "condition": P.condition(code),
            "pred_label": pred_name(rec),
            "correct": rec["correct"],
            "parse_reason": rec["parse_reason"],
            "raw_response": rec["raw_response"],
            "retrieved_ids": rec.get("retrieved_ids"),
            "retrieved_labels": rec.get("retrieved_labels"),
            "retrieved_similarities": rec.get("retrieved_similarities"),
            "retrieved_docs": rec.get("retrieved_docs"),
            "rendered_prompt": rec["rendered_prompt"],
            "display_name": P.display(code),
        }
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    C.log(f"已寫入 {path}（{len(rows)} 筆）")


def confusion_ranking(P: E.Profile, code: str, records: list[dict]) -> pd.DataFrame:
    errors = [(r["true_label"], pred_name(r)) for r in records if not r["correct"]]
    df = pd.DataFrame(errors, columns=["true_label", "pred_label"])
    table = df.value_counts().rename("count").reset_index()
    support = pd.Series([r["true_label"] for r in records]).value_counts()
    table["share_of_errors"] = table["count"] / len(errors)
    table["share_of_true_class"] = table["count"] / table["true_label"].map(support)
    table.insert(0, "condition", P.condition(code))
    table.insert(0, "code", code)
    table["display_name"] = P.display(code)
    return table


def majority_label(labels: list[str]) -> str | None:
    """k 篇檢索標籤的多數；最高票同票時回傳 None（tie）。k = 1 時就是 top-1 標籤。"""
    counts = collections.Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def retrieval_copy_stats(P: E.Profile, preds: dict) -> pd.DataFrame:
    rows = []
    for code in P.rag_codes:
        recs = preds[code]
        retrieved = np.array([r["retrieved_labels"][0] for r in recs])
        true = np.array([r["true_label"] for r in recs])
        pred = np.array([pred_name(r) for r in recs])
        correct = np.array([r["correct"] for r in recs])
        valid = pred != "INVALID"
        ret_ok = retrieved == true
        sims = np.array([r["retrieved_similarities"][0] for r in recs])
        majority = np.array([majority_label(r["retrieved_labels"]) for r in recs], dtype=object)
        tie = np.array([m is None for m in majority])
        maj_ok = majority == true
        rows.append({
            "code": code, "condition": P.condition(code), "n": len(recs),
            "retrieved_label_correct_rate": float(ret_ok.mean()),
            "copy_rate_all": float((pred == retrieved).mean()),
            "copy_rate_valid_only": float((pred[valid] == retrieved[valid]).mean()),
            "accuracy": float(correct.mean()),
            "accuracy_when_retrieved_correct": float(correct[ret_ok].mean()),
            "accuracy_when_retrieved_wrong": float(correct[~ret_ok].mean()),
            "n_retrieved_correct": int(ret_ok.sum()),
            "n_retrieved_wrong": int((~ret_ok).sum()),
            # 檢索錯時，模型照抄錯誤標籤的比例（被誤導）。
            "copied_wrong_label_rate_when_retrieved_wrong": float(
                (pred[~ret_ok] == retrieved[~ret_ok]).mean()),
            "invalid_rate_when_retrieved_correct": float((~valid[ret_ok]).mean()),
            "invalid_rate_when_retrieved_wrong": float((~valid[~ret_ok]).mean()),
            "median_top1_similarity": float(np.median(sims)),
            "median_top1_similarity_when_correct": float(np.median(sims[correct])),
            "median_top1_similarity_when_wrong": float(np.median(sims[~correct])),
            "top_k": int(len(recs[0]["retrieved_labels"])),
            "majority_tie_rate": float(tie.mean()),
            "majority_label_correct_rate": float(maj_ok.mean()),
            "copy_rate_majority_all": float((pred == majority).mean()),
            "copy_rate_majority_valid_only": float((pred[valid] == majority[valid]).mean()),
            "accuracy_when_majority_correct": float(correct[maj_ok].mean()),
            "accuracy_when_majority_wrong_or_tie": float(correct[~maj_ok].mean()),
            "display_name": P.display(code),
        })
    return pd.DataFrame(rows)


def short(text: str | None, limit: int = TRUNCATE) -> str:
    if text is None:
        return "(none)"
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " …"


def md_example(ex: dict, codes: tuple[str, str]) -> list[str]:
    lines = [f"#### `{ex['id']}`（true = {ex['true_label']}）", "",
             f"- 輸入：{short(ex['statement'])}"]
    for code in codes:
        d = ex[code]
        lines.append(f"- **{code}**（{d['display_name']}，`{d['condition']}`）→ 預測 **{d['pred_label']}**"
                     f"{'（正確）' if d['correct'] else '（錯誤）'}，"
                     f"raw response：`{short(d['raw_response'], 80)}`")
        if d["retrieved_docs"]:
            # 檢索文件尾端已附 "true_label is X"，截短後另行標示標籤。
            lines.append(f"  - 檢索 top-1：label = {d['retrieved_labels'][0]}，"
                         f"similarity = {d['retrieved_similarities'][0]:.4f}，"
                         f"id = `{d['retrieved_ids'][0]}`")
            k = len(d["retrieved_labels"])
            if k > 1:
                lines.append(f"  - 檢索 top-{k} 標籤：" + "、".join(
                    f"{lab}（{sim:.4f}）" for lab, sim in
                    zip(d["retrieved_labels"], d["retrieved_similarities"])))
            lines.append(f"  - 檢索文本{'（top-1）' if k > 1 else ''}：{short(d['retrieved_docs'][0])}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--per-confusion", type=int, default=3)
    E.add_profile_args(ap)
    args = ap.parse_args()
    P, out_dir = E.resolve(args)
    out = out_dir / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)

    preds = E.load_predictions(P)
    test = E.load_test()
    statement = dict(zip(test["id"].astype(str), test["statement"]))

    c1, c2 = preds["C1"], preds["C2"]
    helped = [example(P, "C1", a, "C2", b, statement[str(a["id"])])
              for a, b in zip(c1, c2) if not a["correct"] and b["correct"]]
    hurt = [example(P, "C1", a, "C2", b, statement[str(a["id"])])
            for a, b in zip(c1, c2) if a["correct"] and not b["correct"]]
    write_jsonl(out / "rag_helped_C1wrong_C2right.jsonl", helped)
    write_jsonl(out / "rag_hurt_C1right_C2wrong.jsonl", hurt)

    rankings = {}
    for code in ("C4", "C5"):
        rankings[code] = confusion_ranking(P, code, preds[code])
        rankings[code].to_csv(out / f"confusions_{code}.csv", index=False, encoding="utf-8-sig")
        C.log(f"已寫入 {out / f'confusions_{code}.csv'}")

    copy = retrieval_copy_stats(P, preds)
    copy.to_csv(out / "retrieval_copy_stats.csv", index=False, encoding="utf-8-sig")
    C.log(f"已寫入 {out / 'retrieval_copy_stats.csv'}")

    # helped / hurt 的細分：C1 的錯是 INVALID 還是錯類別；C2 是否照抄檢索標籤。
    def breakdown(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "C1_invalid": sum(r["C1"]["pred_label"] == "INVALID" for r in rows),
            "C2_invalid": sum(r["C2"]["pred_label"] == "INVALID" for r in rows),
            "C2_pred_equals_retrieved": sum(
                r["C2"]["pred_label"] == r["C2"]["retrieved_labels"][0] for r in rows),
            "retrieved_label_correct": sum(
                r["C2"]["retrieved_labels"][0] == r["true_label"] for r in rows),
        }

    rng = np.random.default_rng(E.SEED)

    def sample(rows: list, k: int) -> list:
        if len(rows) <= k:
            return list(rows)
        return [rows[i] for i in sorted(rng.choice(len(rows), k, replace=False))]

    md = [
        f"# {P.name}（{P.experiment}）質性範例與錯誤分析",
        "",
        "> ⚠️ 文本來自心理健康貼文資料集，屬敏感內容。此處已截短至 "
        f"{TRUNCATE} 字元，放進論文前必須再改寫或摘要，不可逐字引用。",
        "",
        f"產生方式：`src/18_custom_prompt_error_analysis.py`，固定 seed = {E.SEED} 抽樣。"
        "完整樣本見同目錄的 jsonl。",
        "",
        f"條件：" + "；".join(P.label(code) + f"（`{P.condition(code)}`）" for code in P.codes)
        + (f"。RAG 條件 top-k = {P.top_k}；下表「檢索標籤」指 top-1。" if P.top_k > 1 else "。"),
        "",
        f"## RAG 的效果（{P.label('C1')} → {P.label('C2')}）",
        "",
        "| 群組 | n | C1 為 INVALID | C2 為 INVALID | C2 預測＝檢索標籤 | 檢索標籤正確 |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in (("RAG 有幫助（C1 錯 → C2 對）", helped), ("RAG 有害（C1 對 → C2 錯）", hurt)):
        b = breakdown(rows)
        md.append(f"| {name} | {b['n']} | {b['C1_invalid']} | {b['C2_invalid']} | "
                  f"{b['C2_pred_equals_retrieved']} | {b['retrieved_label_correct']} |")
    md += ["", "## 檢索標籤與預測的關係（C2–C5）", "",
           "| 條件 | 檢索標籤正確率 | 照抄率（全部） | 照抄率（有效回覆） | 檢索對時準確率 | 檢索錯時準確率 | 檢索錯時照抄錯標籤 |",
           "|---|---|---|---|---|---|---|"]
    for r in copy.itertuples():
        md.append(f"| {P.label(r.code)} | {r.retrieved_label_correct_rate:.4f} | {r.copy_rate_all:.4f} | "
                  f"{r.copy_rate_valid_only:.4f} | {r.accuracy_when_retrieved_correct:.4f} | "
                  f"{r.accuracy_when_retrieved_wrong:.4f} | "
                  f"{r.copied_wrong_label_rate_when_retrieved_wrong:.4f} |")

    md += ["", f"## 範例：RAG 有幫助（{len(helped)} 筆中抽 {args.per_group} 筆）", ""]
    for ex in sample(helped, args.per_group):
        md += md_example(ex, ("C1", "C2"))
    md += [f"## 範例：RAG 有害（{len(hurt)} 筆中抽 {args.per_group} 筆）", ""]
    for ex in sample(hurt, args.per_group):
        md += md_example(ex, ("C1", "C2"))

    for code in ("C4", "C5"):
        table = rankings[code]
        md += [f"## {P.label(code)}（`{P.condition(code)}`）的主要錯誤類型", "",
               "| 真實 | 預測 | 筆數 | 佔錯誤 | 佔該真實類別 |", "|---|---|---|---|---|"]
        for r in table.head(8).itertuples():
            md.append(f"| {r.true_label} | {r.pred_label} | {r.count} | "
                      f"{r.share_of_errors:.3f} | {r.share_of_true_class:.3f} |")
        md.append("")
        recs = preds[code]
        for r in table.head(3).itertuples():
            rows = [x for x in recs if x["true_label"] == r.true_label and pred_name(x) == r.pred_label]
            md += [f"### {code}：{r.true_label} → {r.pred_label}（{r.count} 筆中抽 "
                   f"{min(args.per_confusion, len(rows))} 筆）", ""]
            for rec in sample(rows, args.per_confusion):
                ex = example(P, code, rec, code, rec, statement[str(rec["id"])])
                md += md_example({k: v for k, v in ex.items()}, (code,))

    (out / "examples.md").write_text("\n".join(md), encoding="utf-8")
    C.log(f"已寫入 {out / 'examples.md'}")

    C.save_json(out / "error_analysis_summary.json", {
        "run": E.RUN,
        "experiment": P.experiment,
        "script": "src/18_custom_prompt_error_analysis.py",
        "seed": E.SEED,
        "rag_helped_C1wrong_C2right": breakdown(helped),
        "rag_hurt_C1right_C2wrong": breakdown(hurt),
        "top_confusions": {code: rankings[code].head(5).to_dict("records") for code in rankings},
        "retrieval_copy_stats": copy.to_dict("records"),
        "profile": P.name,
        "top_k": P.top_k,
        "display_names": P.display_names(),
    })
    for name, rows in (("helped", helped), ("hurt", hurt)):
        C.log(name, breakdown(rows))
    C.log(copy[["code", "retrieved_label_correct_rate", "copy_rate_valid_only",
                "accuracy_when_retrieved_correct", "accuracy_when_retrieved_wrong"]].to_string())


if __name__ == "__main__":
    main()
