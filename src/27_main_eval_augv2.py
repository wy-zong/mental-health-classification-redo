"""Issue #21 第二階段：以 aug_v2 語料與可調的 top-k 重跑實驗 15 的五組條件（v2）。

背景
  實驗 15（v1，CUSTOM_PROMPT_EVAL/）的「原文＋擴寫檢索」語料沒有套用論文 3.E 的過濾。
  #20 重做擴寫並建立防洩漏語料 aug_v2。本腳本用 aug_v2 重跑主實驗；v1 不覆蓋。
  #3（val 選 k）與 #22（test 五組重跑）共用這支腳本，兩邊的 prompt 組法完全一致。

與 15_ 的關係
  * prompt 文字、render、生成參數、標籤解析全部沿用 15_（importlib 載入，不重抄）；
    開跑前檢查三個 template 的 sha256 與 15_ manifest 相同。
  * 檢索直接呼叫 12_ RebuildRagIndex 的 encode() 與 index.search(vec, k)。12_ 的
    TOP_K = 1 與 search() 都不改，v1 照常可重現。
  * {reference_context} = 依相似度由高到低、以換行分隔的 k 篇文件；k = 1 時與 15_ 逐字相同
    （--parity-check 以 15_ 存下的 rendered prompt 驗證）。
  * aug_v2 不依來源去重；每筆另記前 k 篇的不同來源數（改寫 aug_<source>_<slot> 歸到 <source>）。

條件（顯示名稱見 #21；val 只跑四組 RAG 條件）
  norag_base / rag_noaug_base / rag_augv2_base / rag_noaug_optimized / rag_augv2_optimized

用法
  --split val --top-k 3                   val 掃描（輸出 TOPK_VAL_AUGV2/k3/）
  --split test --top-k K                  test 五組；K 必須等於 selected_k.json 的 k
  --smoke N                               每組只跑前 N 筆，輸出到 <輸出目錄>/_smoke/
  --parity-check N                        不呼叫 LLM：k = 1 時與 15_ rendered prompt 逐字對照
  --retrieval-diagnostic                  不呼叫 LLM：val 上兩種語料 k = 1..8 的檢索結構
  --select-k                              彙總 val 掃描，依規則選 k，寫 selected_k.json
  --compare-v1                            test 的 v1→v2 對照表

輸出：runs/NESTED_70_10_20_INCREMENTAL/TOPK_VAL_AUGV2/、MAIN_EVAL_AUGV2/
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common as C  # noqa: E402


def load_v1():
    path = SRC / "15_nested_custom_prompt_eval.py"
    spec = importlib.util.spec_from_file_location("custom_prompt_eval_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_v1()                 # 15_：PROMPTS、render、check_complete
R = M.R                       # 12_：RebuildRagIndex、label_from_corpus_doc、probe_llm_identity

RUN = M.RUN
BASE_OUT = M.BASE_OUT
EXPERIMENT = "MAIN_EVAL_AUGV2"
TEST_OUT = BASE_OUT / "MAIN_EVAL_AUGV2"
VAL_OUT = BASE_OUT / "TOPK_VAL_AUGV2"
V1_OUT = M.OUT                # CUSTOM_PROMPT_EVAL/（v1，唯讀）
V1_MANIFEST = V1_OUT / "custom_prompt_eval_manifest.json"
SPLIT_MANIFEST = BASE_OUT / "splits" / "split_manifest.json"
SELECTED_K = VAL_OUT / "selected_k.json"

K_CANDIDATES = (1, 3, 5)
CORPORA = ("noaug", "aug_v2")
SPLITS = ("val", "test")

# (條件 ID, 15_ prompt, 語料, 顯示名稱)
CONDITIONS = (
    ("norag_base", "base_norag", None, "無檢索・基礎 prompt"),
    ("rag_noaug_base", "base_rag", "noaug", "原文檢索・基礎 prompt"),
    ("rag_augv2_base", "base_rag", "aug_v2", "原文＋擴寫檢索・基礎 prompt"),
    ("rag_noaug_optimized", "optimized_rag", "noaug", "原文檢索・最佳化 prompt"),
    ("rag_augv2_optimized", "optimized_rag", "aug_v2", "原文＋擴寫檢索・最佳化 prompt"),
)
RAG_CONDITIONS = tuple(c for c in CONDITIONS if c[2] is not None)
DISPLAY = {c[0]: c[3] for c in CONDITIONS}
# v2 條件 → v1（15_）條件：只換語料 aug → aug_v2
V1_CONDITION = {
    "norag_base": "norag_base",
    "rag_noaug_base": "rag_noaug_base",
    "rag_augv2_base": "rag_aug_base",
    "rag_noaug_optimized": "rag_noaug_optimized",
    "rag_augv2_optimized": "rag_aug_optimized",
}

# 截斷檢查（num_ctx 不改，與 15_ 相同）
NUM_CTX = C.GEN_OPTIONS["num_ctx"]
NEAR_CTX = int(NUM_CTX * 0.9)                          # 3,686
AT_CTX = NUM_CTX - C.GEN_OPTIONS["num_predict"]        # 4,080

LABEL_IDS = list(range(len(C.LABELS)))
INVALID = len(C.LABELS)


# ---------------------------------------------------------------- 環境與輸入核對

def ollama_version() -> str | None:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001
        return None


def v1_manifest() -> dict:
    manifest = C.load_json(V1_MANIFEST)
    if not manifest:
        C.die(f"找不到 15_ manifest：{V1_MANIFEST}")
    return manifest


def check_prompts() -> dict:
    """三個 template 的 sha256 必須與 15_ manifest 相同。"""
    recorded = v1_manifest()["prompts"]
    out = {}
    for name, spec in M.PROMPTS.items():
        sha = C.sha256_text(spec["template"])
        if recorded.get(name, {}).get("sha256") != sha:
            C.die(f"prompt {name} 的 sha256 與 15_ manifest 不符：{sha}")
        out[name] = {
            "prompt_id": spec["prompt_id"],
            "sha256": sha,
            "template": spec["template"],
            "uses_rag": spec["uses_rag"],
        }
    return out


def check_model() -> dict:
    identity = R.probe_llm_identity()
    expected = v1_manifest()["model"]["digest"]
    if identity.get("digest") != expected:
        C.die(f"model digest {identity.get('digest')} 與 15_ manifest 的 {expected} 不符"
              "（Ollama 未啟動或模型不同）")
    return {**identity, "ollama_version": ollama_version()}


def split_path(split: str) -> Path:
    return BASE_OUT / "splits" / f"{split}.csv"


def load_split(split: str) -> pd.DataFrame:
    """讀 val／test，筆數與 sha256 必須與 split_manifest.json 相同。"""
    expected = C.load_json(SPLIT_MANIFEST)["files"][split]
    path = split_path(split)
    sha = C.sha256_file(path)
    if sha != expected["sha256"]:
        C.die(f"{split}.csv 的 sha256 {sha} 與 split_manifest 不符")
    df = pd.read_csv(path)
    missing = {"id", "statement", "label_id", "status"} - set(df.columns)
    if missing:
        C.die(f"{split}.csv 缺欄位：{sorted(missing)}")
    if len(df) != expected["rows"]:
        C.die(f"{split}.csv 筆數 {len(df)} ≠ {expected['rows']}")
    return df


def split_info(split: str, df: pd.DataFrame) -> dict:
    return {
        "name": split,
        "path": f"splits/{split}.csv",
        "n": int(len(df)),
        "sha256": C.sha256_file(split_path(split)),
        "label_counts": df["status"].value_counts().to_dict(),
    }


def corpus_info(name: str) -> dict:
    """語料的 index 與 docs sha 必須與 meta 相同。

    在 core.autocrlf=true 下，aug_v2_docs.json 會 checkout 成 CRLF，磁碟 sha 與 meta 不同；
    因此 docs 接受「磁碟 sha」或「LF 正規化 sha」其一相符，兩個都記錄。CRLF 只出現在 JSON
    的縮排換行，不影響解析出來的文件字串。
    """
    d = BASE_OUT / "corpus"
    meta_path = d / f"{name}_meta.json"
    meta = C.load_json(meta_path)
    if not meta:
        C.die(f"找不到語料 meta：{meta_path}")
    index_sha = C.sha256_file(d / f"{name}.index")
    if index_sha != meta["index_sha256"]:
        C.die(f"{name}.index 的 sha256 與 meta 不符")
    raw = (d / f"{name}_docs.json").read_bytes()
    disk = _sha256_bytes(raw)
    lf = _sha256_bytes(raw.replace(b"\r\n", b"\n"))
    if meta["docs_sha256"] == disk:
        match = "disk"
    elif meta["docs_sha256"] == lf:
        match = "lf_normalized"
    else:
        C.die(f"{name}_docs.json 的 sha256 與 meta 不符（磁碟 {disk}，LF 正規化 {lf}）")
    return {
        "meta": meta,
        "meta_sha256": C.sha256_file(meta_path),
        "index_sha256_actual": index_sha,
        "docs_sha256_disk": disk,
        "docs_sha256_lf_normalized": lf,
        "docs_sha256_match": match,
    }


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- 檢索

def source_of(doc_id: str) -> str:
    """改寫 aug_<source>_<slot> → <source>；原文就是自己。"""
    doc_id = str(doc_id)
    if doc_id.startswith("aug_"):
        return doc_id[len("aug_"):].rsplit("_", 1)[0]
    return doc_id


def load_rag(name: str):
    rag = R.RebuildRagIndex(RUN, name)
    if name == "aug_v2":
        originals = {i for i in rag.doc_ids if not str(i).startswith("aug_")}
        orphans = {source_of(i) for i in rag.doc_ids if str(i).startswith("aug_")} - originals
        if orphans:
            C.die(f"aug_v2 有 {len(orphans)} 個改寫的來源不在語料原文中")
    return rag


def retrieve(rag, text: str, k: int) -> tuple[list[str], list[str], list[float]]:
    """與 RebuildRagIndex.search 相同的單筆編碼，但取前 k 篇。"""
    vec = rag.encode([text])
    sims, idxs = rag.index.search(vec, k)
    if len(idxs[0]) != k or int(idxs[0].min()) < 0:
        C.die(f"{rag.name} 檢索結果不足 {k} 篇")
    return (
        [rag.docs[int(i)] for i in idxs[0]],
        [str(rag.doc_ids[int(i)]) for i in idxs[0]],
        [float(v) for v in sims[0]],
    )


def build_prompt(spec: dict, text: str, rag, k: int):
    """回傳 (rendered prompt, docs, ids, sims)；評估與 parity check 共用。"""
    docs = ids = sims = None
    reference_context = None
    if rag is not None:
        docs, ids, sims = retrieve(rag, text, k)
        reference_context = "\n".join(docs)
    return M.render(spec["template"], text, reference_context), docs, ids, sims


# ---------------------------------------------------------------- 評估

def out_dir(split: str, k: int, smoke: int | None) -> Path:
    base = VAL_OUT / f"k{k}" if split == "val" else TEST_OUT
    if smoke:
        base = (VAL_OUT if split == "val" else TEST_OUT) / "_smoke" / f"{split}_k{k}"
    return base


def record_for(row, split: str, k: int, cond: tuple, prompt: str, response: dict,
               pred: int | None, reason: str, docs, ids, sims, model_digest: str) -> dict:
    condition, prompt_name, corpus, display = cond
    spec = M.PROMPTS[prompt_name]
    sources = [source_of(i) for i in ids] if ids else None
    return {
        "run": RUN,
        "experiment": EXPERIMENT,
        "split": split,
        "condition": condition,
        "condition_description": display,
        "id": row.id,
        "true_label_id": int(row.label_id),
        "true_label": row.status,
        "prompt_id": spec["prompt_id"],
        "prompt_sha256": C.sha256_text(spec["template"]),
        "prompt_template": spec["template"],
        "rendered_prompt": prompt,
        "corpus": corpus,
        "top_k": k if corpus else None,
        "retrieval_used": bool(corpus),
        "retrieved_ids": ids,
        "retrieved_docs": docs,
        "retrieved_labels": [R.label_from_corpus_doc(d) for d in docs] if docs else None,
        "retrieved_similarities": sims,
        "retrieved_source_ids": sources,
        "n_distinct_sources": len(set(sources)) if sources else None,
        "raw_response": response["raw_response"],
        "pred_label_id": pred,
        "pred_label": C.ID_TO_LABEL.get(pred) if pred is not None else None,
        "parse_reason": reason,
        "invalid_reason": reason if pred is None else None,
        "invalid": pred is None,
        "correct": bool(pred is not None and pred == int(row.label_id)),
        "elapsed_s": round(response["elapsed_s"], 3),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "token_count": (response.get("prompt_eval_count") or 0)
        + (response.get("eval_count") or 0),
        "model_digest": model_digest,
    }


def evaluate_condition(df: pd.DataFrame, split: str, k: int, cond: tuple, out: Path,
                       rag_cache: dict, model_digest: str) -> list[dict]:
    condition, prompt_name, corpus, _ = cond
    spec = M.PROMPTS[prompt_name]
    path = out / f"predictions_{condition}.jsonl"
    done = {str(r["id"]) for r in C.read_jsonl(path)}
    rag = rag_cache.setdefault(corpus, load_rag(corpus)) if corpus else None

    for row in df.itertuples(index=False):
        if str(row.id) in done:
            continue
        prompt, docs, ids, sims = build_prompt(spec, str(row.statement), rag, k)
        response = C.chat(prompt)
        pred, reason = C.parse_label(response["raw_response"])
        C.append_jsonl(path, record_for(row, split, k, cond, prompt, response, pred, reason,
                                        docs, ids, sims, model_digest))
        done.add(str(row.id))
        C.log(f"{split} k={k} {condition}: {len(done):,}/{len(df):,}, id={row.id}")

    records = M.check_complete(path, df)
    if corpus:
        audit_path = out / f"retrieval_audit_{condition}.jsonl"
        audit_path.unlink(missing_ok=True)
        for rec in records:
            C.append_jsonl(audit_path, {
                "run": RUN,
                "experiment": EXPERIMENT,
                "split": split,
                "condition": condition,
                "id": rec["id"],
                "retrieved_ids": rec["retrieved_ids"],
                "retrieved_docs": rec["retrieved_docs"],
                "retrieved_labels": rec["retrieved_labels"],
                "similarities": rec["retrieved_similarities"],
                "retrieved_source_ids": rec["retrieved_source_ids"],
                "n_distinct_sources": rec["n_distinct_sources"],
                "top_k": rec["top_k"],
            })
    save_metrics(out, split, k, cond, records)
    save_confusion(out, condition, records)
    return records


def encode(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([C.LABEL_TO_ID[r["true_label"]] for r in records])
    y_pred = np.array([INVALID if r["pred_label_id"] is None else int(r["pred_label_id"])
                       for r in records])
    return y_true, y_pred


def score(records: list[dict]) -> dict:
    """無效回應算答錯；F1 只在四個有效類別上平均（同 custom_eval_common 的慣例）。"""
    from sklearn.metrics import accuracy_score, f1_score

    y_true, y_pred = encode(records)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted",
                                      zero_division=0)),
        "invalid_rate": float(np.mean(y_pred == INVALID)),
    }


def save_metrics(out: Path, split: str, k: int, cond: tuple, records: list[dict]) -> dict:
    condition, prompt_name, corpus, display = cond
    spec = M.PROMPTS[prompt_name]
    n = len(records)
    invalid = sum(bool(r["invalid"]) for r in records)
    correct = sum(bool(r["correct"]) for r in records)
    valid = [r for r in records if not r["invalid"]]
    tokens = np.array([r["prompt_eval_count"] or 0 for r in records])
    s = score(records)
    distinct = [r["n_distinct_sources"] for r in records if r["n_distinct_sources"] is not None]
    result = {
        "run": RUN,
        "experiment": EXPERIMENT,
        "split": split,
        "condition": condition,
        "condition_description": display,
        "corpus": corpus,
        "n": n,
        "correct_count": correct,
        "invalid_count": invalid,
        "invalid_rate": invalid / n,
        "strict_accuracy": correct / n,
        "valid_only_accuracy": sum(bool(r["correct"]) for r in valid) / len(valid) if valid else None,
        "macro_f1": s["macro_f1"],
        "weighted_f1": s["weighted_f1"],
        "prompt_id": spec["prompt_id"],
        "prompt_sha256": C.sha256_text(spec["template"]),
        "prompt_template": spec["template"],
        "top_k": k if corpus else None,
        "split_sha256": C.sha256_file(split_path(split)),
        "model_digest": records[0]["model_digest"],
        "prompt_eval_count_mean": float(tokens.mean()),
        "prompt_eval_count_max": int(tokens.max()),
        "near_ctx_threshold": NEAR_CTX,
        "near_ctx_count": int((tokens >= NEAR_CTX).sum()),
        "near_ctx_rate": float((tokens >= NEAR_CTX).mean()),
        "at_ctx_threshold": AT_CTX,
        "at_ctx_count": int((tokens >= AT_CTX).sum()),
        "at_ctx_rate": float((tokens >= AT_CTX).mean()),
        "mean_distinct_sources": float(np.mean(distinct)) if distinct else None,
        "elapsed_s_total": round(float(sum(r["elapsed_s"] for r in records)), 3),
        "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in records])),
    }
    C.save_json(out / f"metrics_{condition}.json", result)
    return result


def save_confusion(out: Path, condition: str, records: list[dict]) -> None:
    labels = C.LABELS + ["INVALID"]
    table = pd.crosstab(
        pd.Series([r["true_label"] for r in records], name="true_label"),
        pd.Series([r["pred_label"] or "INVALID" for r in records], name="pred_label"),
    ).reindex(index=C.LABELS, columns=labels, fill_value=0)
    table.to_csv(out / f"confusion_{condition}.csv", encoding="utf-8-sig")


def save_paired(out: Path, records_by_condition: dict[str, list[dict]]) -> None:
    by_id = {}
    for condition, records in records_by_condition.items():
        for rec in records:
            row = by_id.setdefault(str(rec["id"]), {"id": rec["id"], "true_label": rec["true_label"]})
            row[f"{condition}_pred"] = rec["pred_label"] or "INVALID"
            row[f"{condition}_invalid"] = rec["invalid"]
            row[f"{condition}_correct"] = rec["correct"]
    pd.DataFrame(sorted(by_id.values(), key=lambda x: str(x["id"]))).to_csv(
        out / "paired_comparison.csv", index=False, encoding="utf-8-sig")


def k_source(split: str, k: int, smoke: int | None) -> dict:
    """test 的 k 必須來自 #3 的 selected_k.json；val 的 k 是掃描候選。"""
    if split == "val":
        if k not in K_CANDIDATES and not smoke:
            C.die(f"val 掃描的 k 必須是 {K_CANDIDATES} 之一")
        return {"role": "val_candidate", "candidates": list(K_CANDIDATES)}
    if smoke:
        return {"role": "smoke"}
    selected = C.load_json(SELECTED_K)
    if not selected:
        C.die(f"尚未在 val 上選定 k：{SELECTED_K}（先跑 --select-k）")
    if selected["selected_k"] != k:
        C.die(f"--top-k {k} 與 selected_k.json 的 k = {selected['selected_k']} 不同")
    return {
        "role": "selected_on_val",
        "selected_k_path": str(SELECTED_K.relative_to(BASE_OUT)).replace("\\", "/"),
        "selected_k_sha256": C.sha256_file(SELECTED_K),
    }


def write_manifest(out: Path, split: str, df: pd.DataFrame, k: int, conditions: tuple,
                   prompts: dict, model: dict, corpora: dict, k_src: dict,
                   smoke: int | None) -> None:
    C.save_json(out / "manifest.json", {
        "run": RUN,
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": R.git_revision(),
        "model": model,
        "ollama_version": model.get("ollama_version"),
        "llm_generation_options": C.GEN_OPTIONS,
        "split": split_info(split, df),
        "smoke_n": smoke,
        "top_k": k,
        "top_k_source": k_src,
        "reference_context_format": "top-k docs by descending similarity joined with '\\n'",
        "truncation_thresholds": {"num_ctx": NUM_CTX, "near_ctx": NEAR_CTX, "at_ctx": AT_CTX},
        "prompt_source": "15_nested_custom_prompt_eval.py PROMPTS (sha256 checked against 15_ manifest)",
        "prompts": prompts,
        "conditions": [
            {"condition": c, "display_name": disp, "prompt": p, "corpus": corpus,
             "v1_condition": V1_CONDITION[c]}
            for c, p, corpus, disp in conditions
        ],
        "corpora": corpora,
        "v1_manifest_sha256": C.sha256_file(V1_MANIFEST),
    })


def run_eval(split: str, k: int, condition: str, smoke: int | None) -> None:
    if k < 1:
        C.die("--top-k 必須 ≥ 1")
    pool = RAG_CONDITIONS if split == "val" else CONDITIONS
    selected = pool if condition == "all" else tuple(c for c in pool if c[0] == condition)
    if not selected:
        C.die(f"{split} 沒有條件 {condition}")

    df = load_split(split)
    if smoke:
        df = df.head(smoke).reset_index(drop=True)
    k_src = k_source(split, k, smoke)
    prompts = check_prompts()
    model = check_model()
    corpora = {name: corpus_info(name) for name in CORPORA}
    out = out_dir(split, k, smoke)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, split, df, k, pool, prompts, model, corpora, k_src, smoke)
    C.log(f"{split} k={k} → {out}（{len(df):,} 筆，Ollama {model.get('ollama_version')}）")

    rag_cache: dict = {}
    for cond in selected:
        evaluate_condition(df, split, k, cond, out, rag_cache, model["digest"])

    metrics = [C.load_json(out / f"metrics_{c[0]}.json") for c in pool
               if (out / f"metrics_{c[0]}.json").exists()]
    pd.DataFrame(metrics).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    available = {c[0]: M.check_complete(out / f"predictions_{c[0]}.jsonl", df)
                 for c in pool if (out / f"predictions_{c[0]}.jsonl").exists()}
    if len(available) == len(pool):
        save_paired(out, available)
        C.save_json(out / "completion.json", {
            "run": RUN,
            "experiment": EXPERIMENT,
            "split": split,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "conditions": [c[0] for c in pool],
            "top_k": k,
            "n": len(df),
            "smoke_n": smoke,
        })
        C.log(f"{split} k={k} 全部完成")
        for m in metrics:
            C.log(f"  {m['condition']}: acc={m['strict_accuracy']:.4f} "
                  f"macroF1={m['macro_f1']:.4f} invalid={m['invalid_count']} "
                  f"at_ctx={m['at_ctx_count']}")


# ---------------------------------------------------------------- 不呼叫 LLM 的檢查

def parity_check(n: int) -> None:
    """k = 1、noaug：27_ 組出的 prompt 必須與 15_ 存下的 rendered prompt 逐字相同。"""
    check_prompts()
    corpus_info("noaug")
    test = load_split("test")
    rng = np.random.default_rng(42)
    sample = [str(x) for x in rng.choice(test["id"].astype(str).to_numpy(), size=n, replace=False)]
    text_of = dict(zip(test["id"].astype(str), test["statement"].astype(str)))
    rag = load_rag("noaug")
    cache: dict[str, tuple] = {}
    result = {"run": RUN, "experiment": EXPERIMENT, "n_sampled": n, "seed": 42, "top_k": 1,
              "corpus": "noaug", "conditions": {}}
    ok = True
    for v1_cond, prompt_name in (("rag_noaug_base", "base_rag"),
                                 ("rag_noaug_optimized", "optimized_rag")):
        v1 = {str(r["id"]): r for r in C.read_jsonl(V1_OUT / f"predictions_{v1_cond}.jsonl")}
        spec = M.PROMPTS[prompt_name]
        mismatches = []
        max_sim_diff = 0.0
        for i in sample:
            if i not in cache:
                cache[i] = retrieve(rag, text_of[i], 1)
            docs, ids, sims = cache[i]
            prompt = M.render(spec["template"], text_of[i], "\n".join(docs))
            rec = v1[i]
            diff = abs(sims[0] - rec["retrieved_similarities"][0])
            max_sim_diff = max(max_sim_diff, diff)
            problems = []
            if prompt != rec["rendered_prompt"]:
                problems.append("rendered_prompt")
            if ids != [str(x) for x in rec["retrieved_ids"]]:
                problems.append("retrieved_ids")
            if diff > 1e-6:
                problems.append("similarity")
            if problems:
                mismatches.append({"id": i, "problems": problems})
        result["conditions"][v1_cond] = {
            "n_checked": len(sample),
            "n_identical": len(sample) - len(mismatches),
            "max_similarity_abs_diff": max_sim_diff,
            "mismatches": mismatches,
        }
        ok = ok and not mismatches
        C.log(f"parity {v1_cond}: {len(sample) - len(mismatches)}/{len(sample)} 相同，"
              f"相似度最大差 {max_sim_diff:.2e}")
    result["passed"] = ok
    C.save_json(TEST_OUT / "parity_check_k1.json", result)
    if not ok:
        C.die("k = 1 對照未通過，見 parity_check_k1.json")


def is_tie(labels: list) -> bool:
    """最多票的標籤不只一個。"""
    counts = list(collections.Counter(labels).values())
    return counts.count(max(counts)) > 1


def retrieval_diagnostic(max_k: int = 8) -> None:
    """val 上兩種語料的檢索結構，作為選擇候選 k 的依據（不用於選 k 本身）。"""
    val = load_split("val")
    texts = val["statement"].astype(str).tolist()
    rows = []
    for name in CORPORA:
        corpus_info(name)
        rag = load_rag(name)
        _, idxs = rag.index.search(rag.encode(texts), max_k)
        labels = [[R.label_from_corpus_doc(rag.docs[int(j)]) for j in row] for row in idxs]
        sources = [[source_of(rag.doc_ids[int(j)]) for j in row] for row in idxs]
        lengths = [[len(rag.docs[int(j)]) for j in row] for row in idxs]
        for k in range(1, max_k + 1):
            knn = []
            for labs, truth in zip(labels, val["status"]):
                counts = collections.Counter(labs[:k])
                top = max(counts.values())
                knn.append(next(x for x in labs[:k] if counts[x] == top) == truth)
            chars = np.array([sum(x[:k]) for x in lengths])
            rows.append({
                "corpus": name,
                "k": k,
                "mean_distinct_sources": float(np.mean([len(set(s[:k])) for s in sources])),
                "label_pure_rate": float(np.mean([len(set(x[:k])) == 1 for x in labels])),
                "label_tie_rate": float(np.mean([is_tie(x[:k]) for x in labels])),
                "knn_majority_accuracy": float(np.mean(knn)),
                "context_chars_mean": float(chars.mean()),
                "context_chars_p99": float(np.percentile(chars, 99)),
                "context_chars_max": int(chars.max()),
            })
    VAL_OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(VAL_OUT / "retrieval_diagnostic.csv", index=False, encoding="utf-8-sig")
    C.log("檢索診斷 → " + str(VAL_OUT / "retrieval_diagnostic.csv"))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def select_k() -> None:
    """四組 RAG 條件平均 val macro-F1 最高的 k；完全相等才算同分，同分取較小的 k。"""
    val = load_split("val")
    rows, files = [], {}
    for k in K_CANDIDATES:
        out = VAL_OUT / f"k{k}"
        for cond in RAG_CONDITIONS:
            condition = cond[0]
            records = M.check_complete(out / f"predictions_{condition}.jsonl", val)
            if any(r["top_k"] != k or r["split"] != "val" for r in records):
                C.die(f"k{k}/predictions_{condition}.jsonl 的 top_k 或 split 欄不符")
            metric_path = out / f"metrics_{condition}.json"
            m = C.load_json(metric_path)
            s = score(records)
            if abs(s["macro_f1"] - m["macro_f1"]) > 1e-12:
                C.die(f"k{k} {condition} 的 macro-F1 與 metrics 檔不一致")
            files[f"k{k}/metrics_{condition}.json"] = C.sha256_file(metric_path)
            rows.append({
                "k": k, "condition": condition, "display_name": DISPLAY[condition],
                "corpus": cond[2], "prompt": cond[1], "n": m["n"],
                "accuracy": m["strict_accuracy"], "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"], "invalid_rate": m["invalid_rate"],
                "near_ctx_rate": m["near_ctx_rate"], "at_ctx_rate": m["at_ctx_rate"],
                "prompt_eval_count_mean": m["prompt_eval_count_mean"],
                "prompt_eval_count_max": m["prompt_eval_count_max"],
                "mean_distinct_sources": m["mean_distinct_sources"],
                "elapsed_s_mean": m["elapsed_s_mean"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(VAL_OUT / "topk_summary.csv", index=False, encoding="utf-8-sig")

    by_k = []
    for k in K_CANDIDATES:
        sub = table[table.k == k]
        by_k.append({"k": k, "mean_macro_f1": float(sub.macro_f1.mean()),
                     "mean_accuracy": float(sub.accuracy.mean()),
                     **{f"macro_f1_{c}": float(sub[sub.condition == c].macro_f1.iloc[0])
                        for c in sub.condition}})
    pd.DataFrame(by_k).to_csv(VAL_OUT / "topk_by_k.csv", index=False, encoding="utf-8-sig")

    best = max(r["mean_macro_f1"] for r in by_k)
    tied = [r["k"] for r in by_k if r["mean_macro_f1"] == best]
    chosen = min(tied)
    C.save_json(SELECTED_K, {
        "run": RUN,
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "rule": ("All RAG conditions share one k. Choose the k in {1,3,5} with the highest "
                 "mean val macro-F1 over the four RAG conditions (unrounded floats); "
                 "only exact equality counts as a tie, and a tie takes the smaller k. "
                 "k is never re-selected on test."),
        "candidates": list(K_CANDIDATES),
        "conditions": [c[0] for c in RAG_CONDITIONS],
        "mean_macro_f1_by_k": {str(r["k"]): r["mean_macro_f1"] for r in by_k},
        "best_mean_macro_f1": best,
        "tied_k": tied,
        "tie": len(tied) > 1,
        "selected_k": chosen,
        "val_split_sha256": C.sha256_file(split_path("val")),
        "metrics_sha256": files,
    })
    C.log(f"選定 k = {chosen}（各 k 平均 macro-F1：" +
          "、".join(f"k={r['k']} {r['mean_macro_f1']:.4f}" for r in by_k) + "）")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def compare_v1() -> None:
    """test 上 v1（15_）與 v2 的對照；v1 的 macro-F1 以同一定義從逐筆預測重算。"""
    test = load_split("test")
    completion = C.load_json(TEST_OUT / "completion.json")
    if not completion or completion.get("smoke_n"):
        C.die("test 五組尚未完成")
    v1_metrics_note = []
    rows = []
    for condition, _, _, display in CONDITIONS:
        v1_cond = V1_CONDITION[condition]
        v1_records = M.check_complete(V1_OUT / f"predictions_{v1_cond}.jsonl", test)
        if any(r["experiment"] != "CUSTOM_PROMPT_EVAL" for r in v1_records):
            C.die(f"v1 {v1_cond} 的 experiment 欄不是 CUSTOM_PROMPT_EVAL")
        v1_saved = C.load_json(V1_OUT / f"metrics_{v1_cond}.json")
        v1 = score(v1_records)
        if (abs(v1["accuracy"] - v1_saved["strict_accuracy"]) > 1e-12
                or abs(v1["invalid_rate"] - v1_saved["invalid_rate"]) > 1e-12):
            C.die(f"v1 {v1_cond} 重算的 accuracy／invalid rate 與 15_ metrics 不同")
        v1_metrics_note.append(v1_cond)
        v2_records = M.check_complete(TEST_OUT / f"predictions_{condition}.jsonl", test)
        v2 = score(v2_records)
        v1_pred = {str(r["id"]): r["pred_label"] for r in v1_records}
        agree = np.mean([v1_pred[str(r["id"])] == r["pred_label"] for r in v2_records])
        rows.append({
            "display_name": display,
            "v1_condition": v1_cond,
            "v2_condition": condition,
            "v2_top_k": completion["top_k"] if v2_records[0]["corpus"] else None,
            "v1_accuracy": v1["accuracy"], "v2_accuracy": v2["accuracy"],
            "delta_accuracy": v2["accuracy"] - v1["accuracy"],
            "v1_macro_f1": v1["macro_f1"], "v2_macro_f1": v2["macro_f1"],
            "delta_macro_f1": v2["macro_f1"] - v1["macro_f1"],
            "v1_invalid_rate": v1["invalid_rate"], "v2_invalid_rate": v2["invalid_rate"],
            "delta_invalid_rate": v2["invalid_rate"] - v1["invalid_rate"],
            "pred_agreement_v1_v2": float(agree),
        })
    table = pd.DataFrame(rows)
    table.to_csv(TEST_OUT / "v1_v2_comparison.csv", index=False, encoding="utf-8-sig")
    C.log(f"v1 重算檢查通過：{', '.join(v1_metrics_note)}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------- 進入點

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--condition", choices=["all"] + [c[0] for c in CONDITIONS], default="all")
    parser.add_argument("--smoke", type=int, metavar="N")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parity-check", type=int, metavar="N")
    mode.add_argument("--retrieval-diagnostic", action="store_true")
    mode.add_argument("--select-k", action="store_true")
    mode.add_argument("--compare-v1", action="store_true")
    args = parser.parse_args()

    if args.parity_check:
        parity_check(args.parity_check)
    elif args.retrieval_diagnostic:
        retrieval_diagnostic()
    elif args.select_k:
        select_k()
    elif args.compare_v1:
        compare_v1()
    else:
        if args.split is None or args.top_k is None:
            parser.error("評估需要 --split 與 --top-k")
        C.single_instance("27_main_eval_augv2")
        run_eval(args.split, args.top_k, args.condition, args.smoke)


if __name__ == "__main__":
    main()
