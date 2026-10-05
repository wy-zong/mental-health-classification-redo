"""檢索洩漏檢查：評估切分（test、val）對 RAG 語料庫的最近鄰相似度（issue #12）。

問題：改寫文本（augmentation）或語料庫中的原文，是否和評估樣本近乎重複，
使 RAG 等於把答案直接放進 prompt？

* test：直接讀主實驗的 retrieval audit（v1：15_；v2：27_），不重算。base 與 optimized
  prompt 的檢索與 prompt 無關，兩份 audit 會先檢查完全相同。
* val：以和實驗相同的 embedder（12_ 的 RebuildRagIndex，CPU）對 1,000 筆 val 搜尋前 k 篇。
* 語料與 k 取自 profile：v1 為 noaug／aug、k = 1；v2 為 noaug／aug_v2、k = manifest 的 top_k。
  k > 1 時另外彙整全部 k 篇（all_k），v2 並與 #20 的 AUGMENT_V2/split_safety_check.json 核對。

IndexFlatIP 是精確最近鄰，top-1 相似度的最大值就是該切分對整個語料（含改寫文本）的
全域最高 cosine 相似度。

輸出：<輸出目錄>/retrieval_leakage/retrieval_leakage_check.json
     ＋ retrieval_leakage_val_top1.jsonl（val 的逐筆檢索結果，供稽核）
（v2 預設 MAIN_EVAL_AUGV2/；v1 的舊輸出在 NESTED_70_10_20_INCREMENTAL/ 底下，是凍結的存檔，
 --profile v1 必須以 --out-dir 另外指定。）

用法：
    python 19_retrieval_leakage_check.py --profile v2
    python 19_retrieval_leakage_check.py --profile v1 --out-dir 目錄
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
SPLIT_SAFETY = E.BASE_OUT / "AUGMENT_V2" / "split_safety_check.json"


def corpus_payload(name: str) -> tuple[dict, list[str], set[str], str, dict]:
    """回傳 (meta, doc_ids, 新增 doc_id 集合, 新增部分的名稱, docs sha 資訊)，並核對 meta。

    v1 的 noaug／aug 是巢狀遞增語料：尾端 added_doc_count 筆是新增文件（nested_added）。
    aug_v2 是 train 原文＋改寫：新增部分就是改寫（aug_ 開頭的 id）。
    """
    meta = C.load_json(E.BASE_OUT / "corpus" / f"{name}_meta.json")
    docs_info = E.check_corpus_docs(name, meta)
    doc_ids = [str(d) for d in C.load_json(E.BASE_OUT / "corpus" / f"{name}_docs.json")["doc_ids"]]
    if len(doc_ids) != meta["n_docs"]:
        C.die(f"{name}：doc_ids {len(doc_ids)} 筆 ≠ meta 的 n_docs {meta['n_docs']}")
    if "base_n_docs" in meta:
        added = doc_ids[meta["base_n_docs"]:]
        if (len(added) != meta["added_doc_count"]
                or C.sha256_text(json.dumps(added, ensure_ascii=False)) != meta["added_doc_ids_sha256"]):
            C.die(f"{name}：語料尾端 {meta['added_doc_count']} 筆與 meta 記錄的新增 doc id 不符")
        return meta, doc_ids, set(added), "nested_added", docs_info
    originals = doc_ids[:meta["n_original_docs"]]
    added = doc_ids[meta["n_original_docs"]:]
    if (any(d.startswith("aug_") for d in originals) or not all(d.startswith("aug_") for d in added)
            or len(added) != meta["n_paraphrase_docs"]):
        C.die(f"{name}：前 {meta['n_original_docs']} 筆原文、後 {meta['n_paraphrase_docs']} 筆改寫"
              "的結構與 meta 不符")
    return meta, doc_ids, set(added), "paraphrase", docs_info


def origin(doc_id: str) -> str:
    return "paraphrase" if doc_id.startswith("aug_") else "original"


def summarize(sims: np.ndarray, top_ids: list[str], query_ids: list[str], added: set[str],
              added_name: str) -> dict:
    origins = pd.Series([origin(d) for d in top_ids])
    part = pd.Series([added_name if d in added else "base" for d in top_ids])
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


def summarize_all_k(sims: np.ndarray, ids: list[list[str]], query_ids: list[str]) -> dict:
    """k > 1：全部 k 篇（查詢 × 文件配對）的相似度與來源。"""
    flat = sims.ravel()
    origins = np.array([origin(d) for row in ids for d in row])
    return {
        "k": int(sims.shape[1]),
        "n_pairs": int(flat.size),
        "similarity": {"max": float(flat.max()), "mean": float(flat.mean()),
                       "median": float(np.median(flat)), "p99": float(np.percentile(flat, 99)),
                       "min": float(flat.min())},
        "pairs_at_or_above": {f"{t:.2f}": int((flat >= t).sum()) for t in THRESHOLDS},
        "queries_with_any_at_or_above": {f"{t:.2f}": int((sims >= t).any(axis=1).sum())
                                         for t in THRESHOLDS},
        "by_origin": {k: int((origins == k).sum()) for k in ("original", "paraphrase")},
        "max_similarity_by_origin": {k: float(flat[origins == k].max())
                                     for k in ("original", "paraphrase") if (origins == k).any()},
        "mean_similarity_by_origin": {k: float(flat[origins == k].mean())
                                      for k in ("original", "paraphrase") if (origins == k).any()},
        "queries_with_same_source_id_in_top_k": int(sum(
            any(base_id(d) == q for d in row) for row, q in zip(ids, query_ids))),
    }


def base_id(doc_id: str) -> str:
    """aug_train_05610_1 → train_05610；train_00659 → train_00659。"""
    if doc_id.startswith("aug_"):
        return doc_id[len("aug_"):].rsplit("_", 1)[0]
    return doc_id


def audit_paths(P: E.Profile, name: str) -> tuple[Path, Path]:
    """同一語料的 base／optimized 兩份 retrieval audit。"""
    by_prompt = {prompt: cond for cond, (prompt, corpus) in P.conditions.items() if corpus == name}
    return (P.in_dir / f"retrieval_audit_{by_prompt['base_rag']}.jsonl",
            P.in_dir / f"retrieval_audit_{by_prompt['optimized_rag']}.jsonl")


def test_topk(P: E.Profile, name: str) -> tuple[np.ndarray, list[list[str]], list[str]]:
    base_path, opt_path = audit_paths(P, name)
    base = {r["id"]: r for r in C.read_jsonl(base_path)}
    opt = {r["id"]: r for r in C.read_jsonl(opt_path)}
    order = [str(x) for x in E.load_test()["id"]]
    if set(base) != set(order) or set(opt) != set(order):
        C.die(f"{name} retrieval audit 的 id 與 test 不符")
    for i in order:
        if base[i]["retrieved_ids"] != opt[i]["retrieved_ids"] \
                or base[i]["similarities"] != opt[i]["similarities"]:
            C.die(f"{name}：base 與 optimized 的檢索結果不同（id={i}）")
        if len(base[i]["retrieved_ids"]) != P.top_k:
            C.die(f"{name}：audit 的檢索篇數 {len(base[i]['retrieved_ids'])} ≠ k = {P.top_k}（id={i}）")
    sims = np.array([base[i]["similarities"] for i in order], dtype=float)
    return sims, [[str(d) for d in base[i]["retrieved_ids"]] for i in order], order


def val_topk(name: str, R, k: int, out_val: Path) -> tuple[np.ndarray, list[list[str]], list[str]]:
    val = pd.read_csv(E.BASE_OUT / "splits" / "val.csv")
    if len(val) != 1000:
        C.die(f"val 筆數 {len(val)} ≠ 1000")
    rag = R.RebuildRagIndex(E.RUN, name)
    vecs = rag.encode([str(s) for s in val["statement"]])
    sims, idxs = rag.index.search(vecs, k)
    ids = [[str(rag.doc_ids[int(i)]) for i in row] for row in idxs]
    query_ids = [str(x) for x in val["id"]]
    with open(out_val, "a", encoding="utf-8") as fh:
        for q, row_ids, row_sims in zip(query_ids, ids, sims):
            rec = {"corpus": name, "query_id": q, "top1_doc_id": row_ids[0],
                   "similarity": float(row_sims[0])}
            if k > 1:
                rec.update({"top_k_doc_ids": row_ids,
                            "top_k_similarities": [float(s) for s in row_sims]})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sims.astype(float), ids, query_ids


def split_safety_cross_check(result: dict, name: str) -> dict:
    """aug_v2 的 top-1 統計與 #20 split_safety_check.json（exp15 切分）互相核對。"""
    ref = C.load_json(SPLIT_SAFETY)["splits"]["exp15"]
    out = {"source": str(SPLIT_SAFETY.relative_to(E.BASE_OUT)).replace("\\", "/"),
           "corpus": name, "tolerance": 1e-6, "splits": {}}
    for split in ("test", "val"):
        ours = result[split][name]
        n = ours["n"]
        theirs = ref[f"retrieval_top1_{split}"]
        pairs = {
            "max": (ours["top1_similarity"]["max"], theirs["max"]),
            "mean": (ours["top1_similarity"]["mean"], theirs["mean"]),
            "p99": (ours["top1_similarity"]["p99"], theirs["p99"]),
            "n_ge_0_90": (ours["count_at_or_above"]["0.90"], theirs["n_ge_0_90"]),
            "share_paraphrase": (ours["top1_source"]["by_origin"].get("paraphrase", 0) / n,
                                 theirs["share_paraphrase"]),
        }
        out["splits"][split] = {
            key: {"ours": a, "split_safety_check": b, "match": bool(abs(a - b) <= 1e-6)}
            for key, (a, b) in pairs.items()
        }
    out["all_match"] = all(v["match"] for s in out["splits"].values() for v in s.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    E.add_profile_args(ap)
    args = ap.parse_args()
    P, out_dir = E.resolve(args)
    out = out_dir / "retrieval_leakage"
    out.mkdir(parents=True, exist_ok=True)
    out_json = out / "retrieval_leakage_check.json"
    out_val = out / "retrieval_leakage_val_top1.jsonl"

    R = C.load_module("12_rebuild_experiment")
    if out_val.exists():
        out_val.unlink()

    test_source = ("CUSTOM_PROMPT_EVAL/retrieval_audit_rag_{noaug,aug}_base.jsonl（15_）"
                   if P.name == "v1" else
                   "、".join(f"{P.in_dir.name}/{audit_paths(P, name)[0].name}" for name in P.corpora)
                   + f"（{Path(P.source_script).name}）")
    result = {
        "run": E.RUN,
        "script": "src/19_retrieval_leakage_check.py",
        "metric": "cosine similarity (all-MiniLM-L6-v2, normalized, IndexFlatIP exact search)",
        "note": "top-1 為精確最近鄰，max 即該切分對整個語料的全域最高相似度",
        "sources": {
            "test": test_source,
            "val": "以 12_ RebuildRagIndex 重新嵌入與搜尋",
        },
        "corpora": {},
    }
    for name in P.corpora:
        meta, doc_ids, added, added_name, docs_info = corpus_payload(name)
        result["corpora"][name] = {
            "n_docs": meta["n_docs"],
            "base_n_docs": meta.get("base_n_docs", meta.get("n_original_docs")),
            "added_doc_count": meta.get("added_doc_count", meta.get("n_paraphrase_docs")),
            "paraphrase_docs": sum(d.startswith("aug_") for d in doc_ids),
            "exact_overlap_with_eval_splits": meta["exact_overlap_with_eval_splits"],
            "index_sha256": meta["index_sha256"],
            "added_part_name": added_name,
            **docs_info,
        }
        for split, fetch in (("test", lambda: test_topk(P, name)),
                             ("val", lambda: val_topk(name, R, P.top_k, out_val))):
            sims, ids, query_ids = fetch()
            result.setdefault(split, {})[name] = summarize(
                sims[:, 0], [row[0] for row in ids], query_ids, added, added_name)
            if P.top_k > 1:
                result[split][name]["all_k"] = summarize_all_k(sims, ids, query_ids)
            s = result[split][name]
            C.log(f"{split} vs {name}: max={s['top1_similarity']['max']:.4f} "
                  f"mean={s['top1_similarity']['mean']:.4f} "
                  f">=0.90: {s['count_at_or_above']['0.90']} "
                  f"origin={s['top1_source']['by_origin']} "
                  f"same_id={s['top1_same_source_id_as_query']}")
            if P.top_k > 1:
                a = s["all_k"]
                C.log(f"  all {P.top_k}: max={a['similarity']['max']:.4f} "
                      f">=0.90 pairs={a['pairs_at_or_above']['0.90']} origin={a['by_origin']} "
                      f"same_id={a['queries_with_same_source_id_in_top_k']}")

    result.update({"profile": P.name, "experiment": P.experiment, "top_k": P.top_k})
    if "aug_v2" in P.corpora:
        result["split_safety_cross_check"] = split_safety_cross_check(result, "aug_v2")
        C.log(f"與 split_safety_check.json 核對：all_match = "
              f"{result['split_safety_cross_check']['all_match']}")

    C.save_json(out_json, result)
    C.log(f"已寫入 {out_json}")
    C.log(f"已寫入 {out_val}")


if __name__ == "__main__":
    main()
