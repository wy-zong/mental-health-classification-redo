"""15_ 所用 aug 語料的改寫文本標籤保留驗證（issue #1，審稿意見 R1C15、R2C10、R2C11）。

對象是 15_ C3、C5 實際檢索的 ``corpus/aug_docs.json`` 中 11,101 筆改寫（``aug_`` 開頭），
開跑前確認該檔 sha256 與 15_ manifest 記錄的相同。原文取自同一份語料的原始 doc
（以 19_ 的 base_id 規則對應）。只量測，不改動語料。

檢查項目（對應論文 3.E 的宣稱）：

1. 規則：長度比（字元數，改寫／原文）超出 0.5–2.0、與原文相同（空白正規化＋小寫）、
   改寫集合內重複；
2. Sentence-BERT（all-MiniLM-L6-v2、CPU、normalized，同 04_ 與 15_ 檢索的 embedder）
   cosine ≥ 0.80 與 ≥ 0.50 的通過率；
3. LLM 重分類：15_ 的 base_norag prompt，temperature 0.1（其餘同 15_ 的 GEN_OPTIONS），
   以 15_ 的 parse_label 解析，與原標籤一致才算保留；INVALID 不算一致。

``runs/MAIN/augmented.csv`` 與 ``augmented_new.csv`` 只用來標記批次（main／nested）：
兩批的既有過濾不同，nested 批沒有類別詞與 SBERT 過濾。

輸出（runs/NESTED_70_10_20_INCREMENTAL/AUGMENT_LABEL_VALIDATION/）：
    rule_checks.csv       逐筆規則檢查與 SBERT 分數
    llm_reclassify.jsonl  逐筆 LLM 重分類（可中斷續跑）
    manifest.json         prompt、生成參數、model digest、輸入 sha256
    summary.json          整體／各類／各批結果、混淆矩陣、交叉表、假設過濾後剩餘筆數

用法：
    python 22_augment_label_validation.py [--skip-llm] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

OUT = E.BASE_OUT / "AUGMENT_LABEL_VALIDATION"
AUG_DOCS = E.BASE_OUT / "corpus" / "aug_docs.json"
MANIFEST_15 = E.OUT / "custom_prompt_eval_manifest.json"
BATCH_FILES = {"main": C.RUNS / "MAIN" / "augmented.csv",
               "nested": E.BASE_OUT / "augmented_new.csv"}
PROMPT_NAME = "base_norag"
LLM_OPTIONS = {"temperature": 0.1}
LENGTH_RATIO = (0.5, 2.0)
SBERT_THRESHOLDS = (0.80, 0.50)
SBERT_MAIN = 0.80
EXPECTED_N = 11101


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [centre - half, centre + half]


def rate(mask: pd.Series) -> dict:
    k, n = int(mask.sum()), int(len(mask))
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": wilson(k, n)}


def breakdown(df: pd.DataFrame, col: str) -> dict:
    """整體、各類、各批的比率。"""
    out = {"overall": rate(df[col])}
    out["by_label"] = {lab: rate(df.loc[df.label == lab, col]) for lab in C.LABELS}
    out["by_batch"] = {b: rate(df.loc[df.batch == b, col]) for b in BATCH_FILES}
    return out


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def render(template: str, text: str) -> str:
    """與 15_ 的 render 相同規則（LLM-only，無 reference_context）。"""
    prompt = template.replace("{text}", text)
    if "{" in prompt or "}" in prompt:
        C.die("rendered prompt contains an unreplaced placeholder")
    return prompt


# ---------------------------------------------------------------- 載入

def load_pairs(R) -> pd.DataFrame:
    """從 15_ 的 aug 語料取出改寫與對應原文。"""
    recorded = C.load_json(MANIFEST_15)["corpora"]["aug"]["docs_sha256_actual"]
    C.require_hash(AUG_DOCS, recorded, "15_ 的 aug 語料（aug_docs.json）")
    payload = C.load_json(AUG_DOCS)
    docs = dict(zip(payload["doc_ids"], payload["docs"]))
    if len(docs) != len(payload["doc_ids"]):
        C.die("aug_docs.json 的 doc_ids 有重複")
    base_id = C.load_module("19_retrieval_leakage_check").base_id

    def split(doc: str) -> tuple[str, str]:
        label = R.label_from_corpus_doc(doc)
        if label is None:
            C.die(f"語料 doc 沒有 true_label 結尾：{doc[-80:]!r}")
        return doc[: -len(f" true_label is {label}")], label

    batch_of = {}
    for batch, path in BATCH_FILES.items():
        for doc_id in pd.read_csv(path, usecols=["id"])["id"]:
            batch_of[str(doc_id)] = batch

    rows = []
    for doc_id, doc in docs.items():
        if not doc_id.startswith("aug_"):
            continue
        src = base_id(doc_id)
        if src not in docs or src.startswith("aug_"):
            C.die(f"{doc_id} 找不到原文 {src}")
        text, label = split(doc)
        original, src_label = split(docs[src])
        if label != src_label:
            C.die(f"{doc_id} 的標籤 {label} 與原文 {src} 的 {src_label} 不同")
        if doc_id not in batch_of:
            C.die(f"{doc_id} 不在 MAIN／nested 的改寫檔中")
        rows.append({"id": doc_id, "source_id": src, "batch": batch_of[doc_id], "label": label,
                     "text": text, "original": original})
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_N or set(df.id) != set(batch_of):
        C.die(f"改寫筆數 {len(df)} ≠ {EXPECTED_N}，或與 MAIN／nested 改寫檔的 id 不一致")
    C.log(f"15_ aug 語料改寫 {len(df):,} 筆：" + "，".join(
        f"{b} {n:,}" for b, n in df.batch.value_counts().items()))
    return df


# ---------------------------------------------------------------- 規則與 SBERT

def rule_checks(df: pd.DataFrame, R) -> pd.DataFrame:
    df = df.copy()
    df["len_original"] = df.original.str.len()
    df["len_paraphrase"] = df.text.str.len()
    df["length_ratio"] = df.len_paraphrase / df.len_original
    df["length_ratio_lt_0_5"] = df.length_ratio < LENGTH_RATIO[0]
    df["length_ratio_gt_2_0"] = df.length_ratio > LENGTH_RATIO[1]
    df["length_ratio_ok"] = ~(df.length_ratio_lt_0_5 | df.length_ratio_gt_2_0)

    key = df.text.map(norm)
    df["identical_to_original"] = key == df.original.map(norm)
    payload = C.load_json(AUG_DOCS)
    original_keys = {norm(doc.rsplit(" true_label is ", 1)[0])
                     for doc_id, doc in zip(payload["doc_ids"], payload["docs"])
                     if not doc_id.startswith("aug_")}
    df["equals_any_original"] = key.isin(original_keys)
    df["dup_group_size"] = key.map(key.value_counts())
    df["is_duplicate"] = df.dup_group_size > 1
    # 假設過濾時每組保留第一筆（語料順序），其餘算重複
    df["duplicate_extra"] = key.duplicated(keep="first")
    df["_key"] = key

    C.log("計算 SBERT 相似度（CPU）…")
    _, encode = R.embedder()
    t0 = time.perf_counter()
    sims = (encode(list(df.original)) * encode(list(df.text))).sum(axis=1)
    C.log(f"SBERT 完成，{time.perf_counter() - t0:.0f} 秒")
    df["sbert_cosine"] = sims.astype(float)
    for th in SBERT_THRESHOLDS:
        df[f"sbert_ge_{th:.2f}".replace(".", "_")] = df.sbert_cosine >= th
    return df


def duplicate_summary(df: pd.DataFrame) -> dict:
    dup = df[df.is_duplicate]
    groups = dup.groupby("_key")
    return {
        "rows_in_duplicate_groups": int(len(dup)),
        "groups": int(groups.ngroups),
        "extra_rows_beyond_first": int(df.duplicate_extra.sum()),
        "groups_spanning_multiple_sources": int((groups.source_id.nunique() > 1).sum()),
        "groups_with_conflicting_labels": int((groups.label.nunique() > 1).sum()),
        "examples": [{"ids": list(g.id), "labels": sorted(set(g.label)), "text": g.text.iloc[0][:200]}
                     for _, g in list(groups)[:10]],
    }


def sbert_check(df: pd.DataFrame) -> dict:
    """MAIN 批在 04_ 存過 similarity_to_source，重算值應與之相同。"""
    main = pd.read_csv(BATCH_FILES["main"], usecols=["id", "similarity_to_source"])
    merged = df.merge(main, on="id")
    diff = (merged.sbert_cosine - merged.similarity_to_source).abs()
    return {"n_compared": int(len(merged)), "max_abs_diff_vs_04_stored": float(diff.max()),
            "mean_abs_diff_vs_04_stored": float(diff.mean())}


# ---------------------------------------------------------------- LLM 重分類

def llm_manifest(R, template: str) -> dict:
    return {
        "run": E.RUN, "script": "src/22_augment_label_validation.py",
        "prompt_source": f"src/15_nested_custom_prompt_eval.py PROMPTS[{PROMPT_NAME!r}]",
        "prompt_template": template, "prompt_sha256": C.sha256_text(template),
        "llm_generation_options": {**C.GEN_OPTIONS, **LLM_OPTIONS},
        "model": R.probe_llm_identity(),
        "inputs": {"aug_docs_sha256": C.sha256_file(AUG_DOCS),
                   "custom_prompt_eval_manifest_sha256": C.sha256_file(MANIFEST_15),
                   **{f"{b}_augmented_csv_sha256": C.sha256_file(p) for b, p in BATCH_FILES.items()}},
    }


def run_llm(df: pd.DataFrame, R, limit: int | None) -> None:
    C.single_instance("augment_judge")
    prompts, _ = E.source_15()
    template = prompts[PROMPT_NAME]["template"]
    manifest = llm_manifest(R, template)
    expected_digest = C.load_json(MANIFEST_15)["model"]["digest"]
    if manifest["model"].get("digest") != expected_digest:
        C.die(f"model digest {manifest['model'].get('digest')} 與 15_ 的 {expected_digest} 不同")
    path = OUT / "manifest.json"
    old = C.load_json(path)
    if old:
        for key in ("prompt_sha256", "llm_generation_options", "inputs"):
            if old.get(key) != manifest[key]:
                C.die(f"{path.name} 的 {key} 與這次不同，不能接續既有的 llm_reclassify.jsonl")
        if old["model"].get("digest") != manifest["model"].get("digest"):
            C.die("model digest 與既有 manifest 不同，不能接續")
    else:
        manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
        C.save_json(path, manifest)
        C.log(f"已寫入 {path}")

    out = OUT / "llm_reclassify.jsonl"
    done = C.done_ids(out)
    todo = [r for r in df.itertuples(index=False) if r.id not in done]
    if limit is not None:
        todo = todo[:limit]
    t0 = time.perf_counter()
    for i, row in enumerate(todo, 1):
        response = C.chat(render(template, row.text), options=LLM_OPTIONS)
        pred, reason = C.parse_label(response["raw_response"])
        C.append_jsonl(out, {
            "id": row.id, "source_id": row.source_id, "batch": row.batch,
            "original_label": row.label, "original_label_id": C.LABEL_TO_ID[row.label],
            "raw_response": response["raw_response"], "pred_label_id": pred,
            "pred_label": None if pred is None else C.ID_TO_LABEL[pred],
            "parse_reason": reason, "invalid": pred is None,
            "consistent": pred == C.LABEL_TO_ID[row.label],
            "elapsed_s": response["elapsed_s"], "prompt_eval_count": response["prompt_eval_count"],
            "eval_count": response["eval_count"], "prompt_sha256": manifest["prompt_sha256"],
            "model_digest": manifest["model"].get("digest"),
        })
        if i % 100 == 0 or i == len(todo):
            eta = (time.perf_counter() - t0) / i * (len(todo) - i)
            C.log(f"LLM 重分類 {len(done) + i:,}/{len(df):,}（剩約 {eta / 60:.0f} 分）")


# ---------------------------------------------------------------- 彙總

def llm_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    recs = {r["id"]: r for r in C.read_jsonl(OUT / "llm_reclassify.jsonl")}
    if not recs:
        return df.assign(llm_done=False), {"status": "not_run"}
    df = df.copy()
    df["llm_done"] = df.id.isin(recs)
    df["llm_pred"] = df.id.map(lambda i: recs[i]["pred_label"] if i in recs else None)
    df["llm_invalid"] = df.id.map(lambda i: recs[i]["invalid"] if i in recs else None)
    df["llm_consistent"] = df.id.map(lambda i: recs[i]["consistent"] if i in recs else None)
    sub = df[df.llm_done].copy()
    sub["llm_invalid"] = sub.llm_invalid.astype(bool)
    sub["llm_consistent"] = sub.llm_consistent.astype(bool)
    cols = C.LABELS + ["INVALID"]
    cm = pd.crosstab(sub.label, sub.llm_pred.fillna("INVALID")).reindex(
        index=C.LABELS, columns=cols, fill_value=0)
    reasons = pd.Series([recs[i]["parse_reason"] for i in sub.id]).value_counts()
    return df, {
        "status": "complete" if len(sub) == len(df) else f"partial {len(sub)}/{len(df)}",
        "consistency": breakdown(sub, "llm_consistent"),
        "invalid": breakdown(sub, "llm_invalid"),
        "consistency_among_valid": rate(sub.loc[~sub.llm_invalid, "llm_consistent"]),
        "parse_reasons": {k: int(v) for k, v in reasons.items()},
        "confusion_rows_original_cols_llm": {lab: {c: int(cm.loc[lab, c]) for c in cols}
                                             for lab in C.LABELS},
    }


def crosstab_llm_sbert(df: pd.DataFrame) -> dict | None:
    sub = df[df.llm_done]
    if sub.empty:
        return None
    tab = pd.crosstab(sub.llm_consistent.astype(bool), sub.sbert_cosine >= SBERT_MAIN)
    return {f"llm_consistent={a}, sbert_ge_0_80={b}": int(tab.loc[a, b]) if a in tab.index
            and b in tab.columns else 0 for a in (True, False) for b in (True, False)}


def hypothetical_filter(df: pd.DataFrame, llm_complete: bool) -> dict:
    """假設把論文 3.E 的規則全部套用，各類還剩幾筆（不實際改動語料）。"""
    checks = {
        "length_ratio_ok": df.length_ratio_ok,
        "not_identical_to_original": ~df.identical_to_original,
        "not_duplicate_extra": ~df.duplicate_extra,
        "sbert_ge_0_80": df.sbert_cosine >= SBERT_MAIN,
    }
    if llm_complete:
        checks["llm_consistent"] = df.llm_consistent.astype(bool)
    keep = pd.concat(checks, axis=1).all(axis=1)

    def counts(mask: pd.Series) -> dict:
        return {"total": int(mask.sum()),
                **{lab: int(mask[df.label == lab].sum()) for lab in C.LABELS}}

    return {
        "note": "只是模擬；語料與 15_ 結果都沒有改動" + ("" if llm_complete else
                                                    "；LLM 重分類未完成，未納入"),
        "before": counts(pd.Series(True, index=df.index)),
        "each_rule_pass": {name: counts(m) for name, m in checks.items()},
        "all_rules_pass": counts(keep),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="只做規則檢查與 SBERT")
    ap.add_argument("--limit", type=int, default=None, help="這次最多跑幾筆 LLM（冒煙測試用）")
    args = ap.parse_args()

    R = C.load_module("12_rebuild_experiment")
    pairs = load_pairs(R)
    OUT.mkdir(parents=True, exist_ok=True)

    checks_path = OUT / "rule_checks.csv"
    if checks_path.exists():
        df = pd.read_csv(checks_path, encoding="utf-8-sig")
        if list(df.id) != list(pairs.id):
            C.die(f"{checks_path.name} 與語料不一致，請刪除後重算")
        df = pairs.merge(df.drop(columns=["source_id", "batch", "label"]), on="id")
        df["_key"] = df.text.map(norm)
        C.log(f"沿用既有 {checks_path.name}")
    else:
        df = rule_checks(pairs, R)
        cols = [c for c in df.columns if c not in {"text", "original", "_key"}]
        df[cols].to_csv(checks_path, index=False, encoding="utf-8-sig")
        C.log(f"已寫入 {checks_path}")

    if not args.skip_llm:
        run_llm(df, R, args.limit)

    df, llm = llm_summary(df)
    llm_complete = llm.get("status") == "complete"
    summary = {
        "run": E.RUN, "experiment": "AUGMENT_LABEL_VALIDATION",
        "script": "src/22_augment_label_validation.py", "issue": "#1",
        "scope": "15_ C3、C5 所用 aug 語料中的全部改寫；只量測，語料不變",
        "inputs": {"aug_docs_sha256": C.sha256_file(AUG_DOCS),
                   "custom_prompt_eval_manifest_sha256": C.sha256_file(MANIFEST_15)},
        "counts": {"total": int(len(df)),
                   "by_label": {lab: int((df.label == lab).sum()) for lab in C.LABELS},
                   "by_batch": {b: int((df.batch == b).sum()) for b in BATCH_FILES},
                   "sources_with_paraphrase": int(df.source_id.nunique())},
        "length_ratio": {
            "definition": "len(改寫) / len(原文)，字元數",
            "lt_0_5": breakdown(df, "length_ratio_lt_0_5"),
            "gt_2_0": breakdown(df, "length_ratio_gt_2_0"),
            "within_0_5_2_0": breakdown(df, "length_ratio_ok"),
            "quantiles": {q: float(df.length_ratio.quantile(q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)},
        },
        "identical_to_original": breakdown(df, "identical_to_original"),
        "equals_any_original": breakdown(df, "equals_any_original"),
        "duplicates_within_paraphrases": duplicate_summary(df),
        "sbert": {
            "model": "all-MiniLM-L6-v2（CPU、normalized、max_seq_length 512）",
            **{f"ge_{th:.2f}".replace(".", "_"): breakdown(df, f"sbert_ge_{th:.2f}".replace(".", "_"))
               for th in SBERT_THRESHOLDS},
            "mean": float(df.sbert_cosine.mean()), "median": float(df.sbert_cosine.median()),
            "p05": float(df.sbert_cosine.quantile(0.05)),
            "consistency_with_04_augment": sbert_check(df),
        },
        "llm_reclassification": {
            "prompt": f"15_ PROMPTS[{PROMPT_NAME!r}]",
            "options": {**C.GEN_OPTIONS, **LLM_OPTIONS},
            "rule": "parse_label 解析結果與原標籤相同才算一致；INVALID 算不一致",
            **llm,
        },
        "crosstab_llm_consistent_x_sbert_ge_0_80": crosstab_llm_sbert(df),
        "hypothetical_filter": hypothetical_filter(df, llm_complete),
        "reference_tfidf_lr_retention_main_batch": C.load_json(
            C.RUNS / "MAIN" / "augment_audit.json").get("label_preservation"),
    }
    C.save_json(OUT / "summary.json", summary)
    C.log(f"已寫入 {OUT / 'summary.json'}")
    lr = summary["length_ratio"]
    C.log(f"長度比 <0.5：{lr['lt_0_5']['overall']['k']}，>2.0：{lr['gt_2_0']['overall']['k']}；"
          f"與原文相同：{summary['identical_to_original']['overall']['k']}；"
          f"重複：{summary['duplicates_within_paraphrases']['rows_in_duplicate_groups']} 筆／"
          f"{summary['duplicates_within_paraphrases']['groups']} 群")
    C.log(f"SBERT ≥0.80：{summary['sbert']['ge_0_80']['overall']['rate']:.4f}"
          f"（main {summary['sbert']['ge_0_80']['by_batch']['main']['rate']:.4f}）")
    if "consistency" in llm:
        C.log(f"LLM 一致率（{llm['status']}）：{llm['consistency']['overall']['rate']:.4f}")


if __name__ == "__main__":
    main()
