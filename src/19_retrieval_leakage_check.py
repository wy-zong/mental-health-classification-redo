"""檢索洩漏檢查：評估切分（test、val）對 RAG 語料庫的最近鄰相似度（issue #12）。

問題：改寫文本（augmentation）或語料庫中的原文，是否和評估樣本近乎重複，
使 RAG 等於把答案直接放進 prompt？

* test：直接讀 15_ 的 retrieval audit（top-1），不重算。base 與 optimized prompt 的
  檢索與 prompt 無關，兩份 audit 會先檢查完全相同。
* val：以和實驗相同的 embedder（12_ 的 RebuildRagIndex，CPU）對 1,000 筆 val 做 top-1 搜尋。

IndexFlatIP 是精確最近鄰，top-1 相似度的最大值就是該切分對整個語料（含改寫文本）的
全域最高 cosine 相似度。

輸出：runs/NESTED_70_10_20_INCREMENTAL/retrieval_leakage_check.json
     ＋ retrieval_leakage_val_top1.jsonl（val 的逐筆 top-1，供稽核）

用法：
    python 19_retrieval_leakage_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

CORPORA = ("noaug", "aug")
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
OUT_JSON = E.BASE_OUT / "retrieval_leakage_check.json"
OUT_VAL = E.BASE_OUT / "retrieval_leakage_val_top1.jsonl"


def corpus_payload(name: str) -> tuple[dict, list[str], set[str]]:
    """回傳 (meta, doc_ids, nested 新增的 doc_id 集合)，並核對 meta 記錄的雜湊。"""
    meta = C.load_json(E.BASE_OUT / "corpus" / f"{name}_meta.json")
    docs_path = E.BASE_OUT / "corpus" / f"{name}_docs.json"
    C.require_hash(docs_path, meta["docs_sha256"], f"{name} docs")
    doc_ids = C.load_json(docs_path)["doc_ids"]
    added = doc_ids[meta["base_n_docs"]:]
    if (len(added) != meta["added_doc_count"]
            or C.sha256_text(json.dumps(added, ensure_ascii=False)) != meta["added_doc_ids_sha256"]):
        C.die(f"{name}：語料尾端 {meta['added_doc_count']} 筆與 meta 記錄的新增 doc id 不符")
    return meta, doc_ids, set(added)


def origin(doc_id: str) -> str:
    return "paraphrase" if doc_id.startswith("aug_") else "original"


def summarize(sims: np.ndarray, top_ids: list[str], query_ids: list[str], added: set[str]) -> dict:
    origins = pd.Series([origin(d) for d in top_ids])
    part = pd.Series(["nested_added" if d in added else "base" for d in top_ids])
    out = {
        "n": int(len(sims)),
        "top1_similarity": {
            "max": float(sims.max()), "mean": float(sims.mean()),
            "median": float(np.median(sims)), "p95": float(np.percentile(sims, 95)),
            "p99": float(np.percentile(sims, 99)), "min": float(sims.min()),
        },
        "count_at_or_above": {f"{t:.2f}": int((sims >= t).sum()) for t in THRESHOLDS},
        "top1_source": {
            "by_origin": origins.value_counts().to_dict(),
            "by_corpus_part": part.value_counts().to_dict(),
            "max_similarity_by_origin": {
                k: float(sims[(origins == k).to_numpy()].max()) for k in origins.unique()
            },
            "mean_similarity_by_origin": {
                k: float(sims[(origins == k).to_numpy()].mean()) for k in origins.unique()
            },
        },
        # 語料 doc id 沿用舊切分的 id；若 top-1 的 id（去掉 aug_ 前綴與改寫序號）
        # 等於查詢本身的 id，就代表同一則貼文同時在語料與評估切分裡。
        "top1_same_source_id_as_query": int(sum(
            base_id(d) == q for d, q in zip(top_ids, query_ids))),
    }
    order = np.argsort(-sims)[:5]
    out["highest_pairs"] = [
        {"query_id": query_ids[i], "doc_id": top_ids[i], "similarity": float(sims[i]),
         "origin": origin(top_ids[i])}
        for i in order
    ]
    return out


def base_id(doc_id: str) -> str:
    """aug_train_05610_1 → train_05610；train_00659 → train_00659。"""
    if doc_id.startswith("aug_"):
        return doc_id[len("aug_"):].rsplit("_", 1)[0]
    return doc_id


def test_top1(name: str) -> tuple[np.ndarray, list[str], list[str]]:
    base = {r["id"]: r for r in C.read_jsonl(E.OUT / f"retrieval_audit_rag_{name}_base.jsonl")}
    opt = {r["id"]: r for r in C.read_jsonl(E.OUT / f"retrieval_audit_rag_{name}_optimized.jsonl")}
    order = [str(x) for x in E.load_test()["id"]]
    if set(base) != set(order) or set(opt) != set(order):
        C.die(f"{name} retrieval audit 的 id 與 test 不符")
    for i in order:
        if base[i]["retrieved_ids"] != opt[i]["retrieved_ids"] \
                or base[i]["similarities"] != opt[i]["similarities"]:
            C.die(f"{name}：base 與 optimized 的檢索結果不同（id={i}）")
    sims = np.array([base[i]["similarities"][0] for i in order])
    return sims, [base[i]["retrieved_ids"][0] for i in order], order


def val_top1(name: str, R) -> tuple[np.ndarray, list[str], list[str]]:
    val = pd.read_csv(E.BASE_OUT / "splits" / "val.csv")
    if len(val) != 1000:
        C.die(f"val 筆數 {len(val)} ≠ 1000")
    rag = R.RebuildRagIndex(E.RUN, name)
    vecs = rag.encode([str(s) for s in val["statement"]])
    sims, idxs = rag.index.search(vecs, 1)
    ids = [rag.doc_ids[int(i)] for i in idxs[:, 0]]
    query_ids = [str(x) for x in val["id"]]
    with open(OUT_VAL, "a", encoding="utf-8") as fh:
        for q, d, s in zip(query_ids, ids, sims[:, 0]):
            fh.write(json.dumps({"corpus": name, "query_id": q, "top1_doc_id": d,
                                 "similarity": float(s)}, ensure_ascii=False) + "\n")
    return sims[:, 0].astype(float), ids, query_ids


def main() -> None:
    R = C.load_module("12_rebuild_experiment")
    if OUT_VAL.exists():
        OUT_VAL.unlink()

    result = {
        "run": E.RUN,
        "script": "src/19_retrieval_leakage_check.py",
        "metric": "cosine similarity (all-MiniLM-L6-v2, normalized, IndexFlatIP exact search)",
        "note": "top-1 為精確最近鄰，max 即該切分對整個語料的全域最高相似度",
        "sources": {
            "test": "CUSTOM_PROMPT_EVAL/retrieval_audit_rag_{noaug,aug}_base.jsonl（15_）",
            "val": "以 12_ RebuildRagIndex 重新嵌入與搜尋",
        },
        "corpora": {},
    }
    for name in CORPORA:
        meta, doc_ids, added = corpus_payload(name)
        result["corpora"][name] = {
            "n_docs": meta["n_docs"], "base_n_docs": meta["base_n_docs"],
            "added_doc_count": meta["added_doc_count"],
            "paraphrase_docs": sum(d.startswith("aug_") for d in doc_ids),
            "exact_overlap_with_eval_splits": meta["exact_overlap_with_eval_splits"],
            "index_sha256": meta["index_sha256"],
        }
        for split, fetch in (("test", lambda: test_top1(name)), ("val", lambda: val_top1(name, R))):
            sims, ids, query_ids = fetch()
            result.setdefault(split, {})[name] = summarize(sims, ids, query_ids, added)
            s = result[split][name]
            C.log(f"{split} vs {name}: max={s['top1_similarity']['max']:.4f} "
                  f"mean={s['top1_similarity']['mean']:.4f} "
                  f">=0.90: {s['count_at_or_above']['0.90']} "
                  f"origin={s['top1_source']['by_origin']} "
                  f"same_id={s['top1_same_source_id_as_query']}")

    C.save_json(OUT_JSON, result)
    C.log(f"已寫入 {OUT_JSON}")
    C.log(f"已寫入 {OUT_VAL}")


if __name__ == "__main__":
    main()
