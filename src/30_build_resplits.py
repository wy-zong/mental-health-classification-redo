"""Issue #4：建立 3 組重切（resplit_seed0–2）的切分檔與 RAG 語料，供 27_ --resplit 與 07_ 使用。

切分
  直接呼叫 26_ 的 resplits()：val 1,000 筆固定，train／test 依主切分（實驗 15）各類別筆數分層重切，
  每組 train 6,994、test 1,998。26_ verify 已用同一套指派檢查過洩漏（split_safety_check.json），
  本腳本重算改寫數與 top-1 檢索統計，與該檔逐項比對。

語料（與主切分相同的規則）
  noaug  = 該切分 train 的原文
  aug_v2 = noaug + accepted.csv 中 source_id 屬於該切分 train 的改寫（26_ select_for_split）
  文件格式 R.corpus_entry，all-MiniLM-L6-v2（CPU、max_seq 512、正規化）、IndexFlatIP。
  主切分的語料是增量建出來的（原文向量取自 noaug.index）；這裡全部重新編碼，浮點層級可能
  與主切分不同，--check-exp15 會用主切分的指派重建一次，回報 top-1 與 27_ 主結果不同的筆數。

換行：所有 csv／json 以 LF 寫出，manifest 同時記錄磁碟 sha 與 LF 正規化 sha；
  27_ --resplit 接受兩者其一相符（autocrlf=true 的 checkout 會把 LF 轉成 CRLF）。

輸出：runs/NESTED_70_10_20_INCREMENTAL/RESPLITS/<resplit>/{splits,corpus}/、RESPLITS/build_summary.json

用法
  python src/30_build_resplits.py                     # 建 resplit_seed0–2
  python src/30_build_resplits.py --check-exp15       # 另以主切分指派重建到暫存目錄並比對 top-1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common as C  # noqa: E402

A = C.load_module("26_augment_v2")
R = A.R

RUN = A.RUN
BASE = A.BASE
OUT_ROOT = BASE / "RESPLITS"
RESPLITS = ("resplit_seed0", "resplit_seed1", "resplit_seed2")
SAFETY = A.OUT_ROOT / "split_safety_check.json"
ACCEPTED = A.OUT_ROOT / "accepted.csv"
K1_AUDIT = BASE / "MAIN_EVAL_AUGV2_K1" / "retrieval_audit_{cond}.jsonl"
SPLIT_COLS = ["id", "source_id", "statement", "status", "label_id",
              "exp15_split", "exp15_id", "new_split"]
TOL = 1e-3                                # 與 26_ verify 的 top-1 統計比對容許差（不同編碼批次）


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_shas(path: Path) -> dict:
    raw = path.read_bytes()
    return {"sha256": sha_bytes(raw), "sha256_lf": sha_bytes(raw.replace(b"\r\n", b"\n"))}


def write_json_lf(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def load_assignments() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """呼叫 26_ 的 load_pool／resplits，不建完整 Ctx（避免寫 cache、檢查 Ollama）。"""
    pool = A.load_pool(A.load_records_15())
    shim = SimpleNamespace(
        pool=pool,
        split15=pool["exp15_split"].tolist(),
        label=pool["status"].tolist(),
        idx={sid: i for i, sid in enumerate(pool["source_id"])},
    )
    return pool, A.resplits(shim)


def exp15_ids() -> dict[str, str]:
    """source_id → 主切分的 id（train_added 的 id 與 source_id 不同）。"""
    out = {}
    for name in ("train", "val", "test"):
        df = A.read_csv_text(A.SPLITS / f"{name}.csv")
        out.update(zip(df["source_id"], df["id"]))
    return out


def split_frames(pool: pd.DataFrame, assign: dict[str, str], ids15: dict[str, str]) -> dict[str, pd.DataFrame]:
    """依指派產生 train／val／test。順序：train 依主切分的 train.csv 順序在前、其餘依 source_uid；
    val／test 依 source_uid。id：原本就在同一個切分的沿用主切分 id，否則用 source_id。"""
    df = pool.copy()
    df["new_split"] = df["source_id"].map(assign)
    if df["new_split"].isna().any():
        C.die("指派缺少部分 source_id")
    df["exp15_id"] = df["source_id"].map(ids15)
    same = df["new_split"] == df["exp15_split"]
    df["id"] = np.where(same, df["exp15_id"], df["source_id"])
    order15 = {}
    for name in ("train", "val", "test"):
        sids = A.read_csv_text(A.SPLITS / f"{name}.csv")["source_id"]
        order15.update({s: i for i, s in enumerate(sids)})
    out = {}
    for name in ("train", "val", "test"):
        part = df[df["new_split"] == name].copy()
        part["_kept"] = part["exp15_split"] == name
        part["_o15"] = part["source_id"].map(order15)
        part = part.sort_values(["_kept", "_o15", "source_uid"], ascending=[False, True, True],
                                kind="mergesort")
        out[name] = part[SPLIT_COLS].reset_index(drop=True)
    return out


class Encoder:
    """所有原文與改寫各編碼一次，依 key 取用（同一個 R.embedder，CPU）。"""

    def __init__(self, pool: pd.DataFrame, accepted: pd.DataFrame):
        _, encode = R.embedder()
        self.encode = encode
        t0 = time.time()
        docs = [R.corpus_entry(t, s) for t, s in zip(pool["statement"], pool["status"])]
        self.orig = dict(zip(pool["source_id"], encode(docs)))
        C.log(f"  原文文件 {len(docs):,} 筆編碼完成（{time.time() - t0:.0f}s）")
        t0 = time.time()
        pdocs = [R.corpus_entry(t, s) for t, s in zip(accepted["statement"], accepted["status"])]
        self.para = dict(zip(accepted["id"], encode(pdocs)))
        C.log(f"  改寫文件 {len(pdocs):,} 筆編碼完成（{time.time() - t0:.0f}s）")
        t0 = time.time()
        self.query = dict(zip(pool["source_id"], encode(pool["statement"].tolist())))
        C.log(f"  查詢文本 {len(pool):,} 筆編碼完成（{time.time() - t0:.0f}s）")


def write_corpus(out_dir: Path, name: str, rows: list[tuple[str, str]], vecs: np.ndarray,
                 extra: dict) -> dict:
    faiss = C.import_faiss()
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.ascontiguousarray(vecs, dtype="float32"))
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / f"{name}.index"
    faiss.write_index(index, str(index_path))
    docs_path = out_dir / f"{name}_docs.json"
    write_json_lf(docs_path, {"doc_ids": [r[0] for r in rows], "docs": [r[1] for r in rows]})
    meta = {
        "run": RUN, "name": name, "n_docs": len(rows),
        "embedding_model": R.EMBED_MODEL, "max_seq_length": R.EMBED_MAX_SEQ,
        "normalized": True, "metric": "cosine (IndexFlatIP on normalized vectors)",
        "top_k": 1, "doc_format": "{statement} true_label is {status}（R.corpus_entry）",
        "encoding": "全部以 R.embedder 重新編碼（30_build_resplits.py）",
        **extra,
        "index_sha256": C.sha256_file(index_path),
        "docs_sha256": C.sha256_file(docs_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_lf(out_dir / f"{name}_meta.json", meta)
    return meta


def top1(enc: Encoder, query_ids: list[str], vecs: np.ndarray, is_para: np.ndarray,
         doc_ids: list[str]) -> tuple[dict, list[str]]:
    q = np.vstack([enc.query[s] for s in query_ids])
    sims = q @ vecs.T
    k = sims.argmax(1)
    best = sims[np.arange(len(k)), k]
    src = [d[4:].rsplit("_", 1)[0] if d.startswith("aug_") else d for d in (doc_ids[i] for i in k)]
    return {
        "n": len(query_ids), "max": float(best.max()), "mean": float(best.mean()),
        "p99": float(np.quantile(best, 0.99)), "n_ge_0_90": int((best >= 0.90).sum()),
        "share_paraphrase": float(is_para[k].mean()),
        "same_source_as_query": int(sum(a == b for a, b in zip(src, query_ids))),
    }, [doc_ids[i] for i in k]


def build_one(name: str, frames: dict[str, pd.DataFrame], accepted: pd.DataFrame, enc: Encoder,
              out: Path, assign_sha: str) -> dict:
    sdir, cdir = out / "splits", out / "corpus"
    if out.exists():
        shutil.rmtree(out)
    files = {}
    for split, df in frames.items():
        path = sdir / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, lineterminator="\n")
        files[split] = {"rows": int(len(df)), **file_shas(path)}

    train, val, test = frames["train"], frames["val"], frames["test"]
    train_sids = train["source_id"].tolist()
    eval_sids = set(val["source_id"]) | set(test["source_id"])
    used = A.select_for_split(accepted, train_sids)
    leaked = int(used["source_id"].isin(eval_sids).sum())

    orig_rows = [(s, R.corpus_entry(t, l)) for s, t, l in zip(train_sids, train["statement"], train["status"])]
    para_rows = [(i, R.corpus_entry(t, l)) for i, t, l in zip(used["id"], used["statement"], used["status"])]
    orig_vec = np.vstack([enc.orig[s] for s in train_sids])
    para_vec = np.vstack([enc.para[i] for i in used["id"]])

    eval_norm = {R.normalize_text(t) for t in pd.concat([val["statement"], test["statement"]])}
    overlap_noaug = int(sum(R.normalize_text(t) in eval_norm for t in train["statement"]))
    overlap_para = int(sum(R.normalize_text(t) in eval_norm for t in used["statement"]))

    common_extra = {"resplit": name, "assignment_sha256": assign_sha}
    meta_noaug = write_corpus(cdir, "noaug", orig_rows, orig_vec, {
        **common_extra, "source": "train originals", "n_original_docs": len(orig_rows),
        "exact_overlap_with_eval_splits": overlap_noaug})
    meta_aug = write_corpus(cdir, "aug_v2", orig_rows + para_rows, np.vstack([orig_vec, para_vec]), {
        **common_extra, "source": "train originals + augment_v2 train-source paraphrases",
        "n_original_docs": len(orig_rows), "n_paraphrase_docs": len(para_rows),
        "paraphrases_by_label": used["status"].value_counts().to_dict(),
        "accepted_sha256_lf": file_shas(ACCEPTED)["sha256_lf"], "usage_rule": A.USAGE_RULE,
        "exact_overlap_with_eval_splits": overlap_noaug + overlap_para,
        "paraphrases_from_eval_sources": leaked})

    is_para_n = np.zeros(len(orig_rows), bool)
    is_para_a = np.r_[np.zeros(len(orig_rows), bool), np.ones(len(para_rows), bool)]
    ids_a = [r[0] for r in orig_rows + para_rows]
    retrieval = {}
    top1_ids = {}
    for split, df in (("val", val), ("test", test)):
        sids = df["source_id"].tolist()
        retrieval[f"noaug_{split}"], top1_ids[f"noaug_{split}"] = top1(enc, sids, orig_vec, is_para_n, train_sids)
        retrieval[f"aug_v2_{split}"], top1_ids[f"aug_v2_{split}"] = top1(
            enc, sids, np.vstack([orig_vec, para_vec]), is_para_a, ids_a)

    manifest = {
        "run": RUN, "resplit": name, "assignment_sha256": assign_sha,
        "rule": "26_ resplits()：val 固定；train／test 依主切分各類別筆數分層重切（np.random.default_rng(seed)）",
        "files": files,
        "counts_by_label": {s: df["status"].value_counts().to_dict() for s, df in frames.items()},
        "n_moved_from_exp15": {s: int((df["exp15_split"] != s).sum()) for s, df in frames.items()},
        "corpus": {"noaug": {k: meta_noaug[k] for k in ("n_docs", "index_sha256", "docs_sha256")},
                   "aug_v2": {k: meta_aug[k] for k in ("n_docs", "index_sha256", "docs_sha256")}},
        "leakage": {"paraphrases_from_eval_sources": leaked,
                    "exact_overlap_corpus_vs_eval": overlap_noaug + overlap_para},
        "retrieval_top1": retrieval,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": R.git_revision(),
    }
    write_json_lf(sdir / "split_manifest.json", manifest)
    return {"manifest": manifest, "top1_ids": top1_ids, "n_paraphrases": len(para_rows)}


def compare_safety(name: str, res: dict, safety: dict) -> list[str]:
    ref = safety["splits"].get(name)
    if ref is None:
        return [f"{name}：split_safety_check.json 沒有這組"]
    problems = []
    m = res["manifest"]
    if res["n_paraphrases"] != ref["n_paraphrases_in_corpus"]:
        problems.append(f"{name}：改寫數 {res['n_paraphrases']} ≠ {ref['n_paraphrases_in_corpus']}")
    if m["leakage"]["paraphrases_from_eval_sources"] or m["leakage"]["exact_overlap_corpus_vs_eval"]:
        problems.append(f"{name}：洩漏檢查未通過 {m['leakage']}")
    for split in ("val", "test"):
        got, want = m["retrieval_top1"][f"aug_v2_{split}"], ref[f"retrieval_top1_{split}"]
        for key in ("max", "mean", "p99", "share_paraphrase"):
            if abs(got[key] - want[key]) > TOL:
                problems.append(f"{name} {split} top-1 {key}：{got[key]:.4f} vs 26_ {want[key]:.4f}")
        if got["n_ge_0_90"] != want["n_ge_0_90"]:
            problems.append(f"{name} {split} top-1 ≥0.90：{got['n_ge_0_90']} vs {want['n_ge_0_90']}")
    return problems


def check_exp15(pool, assigns, accepted, enc, ids15) -> dict:
    """以主切分的指派重建：切分檔必須與現有 train／val／test 的 id 集合相同，
    top-1 文件與 27_ 主結果（MAIN_EVAL_AUGV2_K1 的 retrieval_audit）比對。"""
    tmp = OUT_ROOT / "_check_exp15"
    frames = split_frames(pool, assigns["exp15"], ids15)
    for split, df in frames.items():
        ref = A.read_csv_text(A.SPLITS / f"{split}.csv")
        if set(df["id"]) != set(ref["id"]) or len(df) != len(ref):
            C.die(f"exp15 重建的 {split} id 集合與主切分不同")
    res = build_one("exp15", frames, accepted, enc, tmp, "exp15")
    test_ids = frames["test"]["id"].tolist()
    out = {}
    for corpus, cond in (("noaug", "rag_noaug_base"), ("aug_v2", "rag_augv2_base")):
        audit = {r["id"]: r["retrieved_ids"][0] for r in C.read_jsonl(Path(str(K1_AUDIT).format(cond=cond)))}
        got = dict(zip(test_ids, res["top1_ids"][f"{corpus}_test"]))
        diff = [i for i in test_ids if audit[i] != got[i]]
        out[corpus] = {"n": len(test_ids), "n_top1_different_from_k1": len(diff), "examples": diff[:10]}
        C.log(f"  exp15 {corpus}：top-1 與 K1 主結果不同 {len(diff)}／{len(test_ids)}")
    shutil.rmtree(tmp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-exp15", action="store_true", help="另以主切分指派重建並比對 top-1")
    args = ap.parse_args()
    C.single_instance("30_build_resplits")

    pool, assigns = load_assignments()
    ids15 = exp15_ids()
    accepted = A.read_csv_text(ACCEPTED)
    safety = C.load_json(SAFETY)
    if safety["accepted_sha256"] not in file_shas(ACCEPTED).values():   # autocrlf 下磁碟是 CRLF
        C.die("accepted.csv 與 split_safety_check.json 記錄的 sha 不同")
    C.log("編碼原文、改寫與查詢 …")
    enc = Encoder(pool, accepted)

    summary = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "resplits": {}, "problems": []}
    for name in RESPLITS:
        assign = assigns[name]
        assign_sha = C.sha256_text(json.dumps(sorted(assign.items()), ensure_ascii=False))
        C.log(f"=== {name} ===")
        res = build_one(name, split_frames(pool, assign, ids15), accepted, enc, OUT_ROOT / name, assign_sha)
        problems = compare_safety(name, res, safety)
        summary["resplits"][name] = {k: res["manifest"][k] for k in
                                     ("assignment_sha256", "files", "n_moved_from_exp15", "corpus",
                                      "leakage", "retrieval_top1")}
        summary["problems"] += problems
        m = res["manifest"]
        C.log(f"  train／test 換邊 {m['n_moved_from_exp15']['train']}／{m['n_moved_from_exp15']['test']}，"
              f"改寫 {res['n_paraphrases']:,}，test top-1 max {m['retrieval_top1']['aug_v2_test']['max']:.4f}"
              + ("；與 26_ 一致" if not problems else f"；問題 {problems}"))
    if args.check_exp15:
        summary["exp15_rebuild_vs_k1"] = check_exp15(pool, assigns, accepted, enc, ids15)
    write_json_lf(OUT_ROOT / "build_summary.json", summary)
    if summary["problems"]:
        C.die("檢查未通過：" + "；".join(summary["problems"]))
    C.log(f"完成：{OUT_ROOT}")


if __name__ == "__main__":
    main()
