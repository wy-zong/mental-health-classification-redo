"""主實驗（v1：15_；v2：27_）的推論成本與延遲（issue #9，離線與檢索部分）。

* cost_latency.csv：由逐筆預測的 elapsed_s、prompt_eval_count、eval_count 彙整
  （elapsed_s 是 Ollama chat 呼叫的牆鐘時間，不含檢索）。
* --measure-retrieval：以和實驗相同的設定（12_ RebuildRagIndex，CPU embedder）
  從 test 以 seed 42 抽樣，分別計時 query embedding 與 FAISS 搜尋（取前 k 篇，k 依 profile），
  輸出 retrieval_latency.json。語料依 profile：v1 為 noaug／aug，v2 為 noaug／aug_v2。

VRAM 峰值需要 Ollama 實際推論時以 nvidia-smi 量測，不在本腳本範圍（與確定性驗證一起跑）。
輸出到 --out-dir（v2 預設 MAIN_EVAL_AUGV2/；v1 的 CUSTOM_PROMPT_EVAL/ 是凍結的存檔）。

用法：
    python 20_custom_prompt_cost.py --profile v2 [--measure-retrieval] [--n 200]
    python 20_custom_prompt_cost.py --profile v1 --out-dir 目錄
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402


def describe(values: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(values.max()),
    }


def cost_table(P: E.Profile, preds: dict) -> pd.DataFrame:
    rows = []
    for code, recs in preds.items():
        elapsed = np.array([r["elapsed_s"] for r in recs], dtype=float)
        prompt_tok = np.array([r["prompt_eval_count"] or 0 for r in recs], dtype=float)
        out_tok = np.array([r["eval_count"] or 0 for r in recs], dtype=float)
        row = {"code": code, "condition": P.condition(code), "n": len(recs),
               "llm_total_hours": float(elapsed.sum() / 3600)}
        row.update(describe(elapsed, "llm_latency_s"))
        row.update(describe(prompt_tok, "prompt_tokens"))
        row.update(describe(out_tok, "output_tokens"))
        row["total_tokens"] = int(prompt_tok.sum() + out_tok.sum())
        row["missing_token_counts"] = int(sum(r["prompt_eval_count"] is None for r in recs))
        row["top_k"] = P.top_k if P.corpus(code) else None
        row["display_name"] = P.display(code)
        rows.append(row)
    return pd.DataFrame(rows)


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 —— 只是記錄環境
        info["torch"] = f"unavailable: {exc}"
    for pkg in ("faiss", "sentence_transformers", "numpy"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:  # noqa: BLE001
            info[pkg] = None
    return info


def measure_retrieval(P: E.Profile, n: int) -> dict:
    R = C.load_module("12_rebuild_experiment")
    test = E.load_test()
    rng = np.random.default_rng(E.SEED)
    sample = test.iloc[sorted(rng.choice(len(test), n, replace=False))]
    texts = [str(s) for s in sample["statement"]]
    k = P.top_k
    runner = Path(P.source_script).name[:3]
    out = {"n_queries": n, "seed": E.SEED, "embedder_device": f"cpu（與 {runner} 實驗相同）",
           "procedure": f"逐筆查詢（batch=1，與 {runner} 相同）；先暖機 5 筆不計時",
           "hardware": hardware(), "corpora": {}}
    if P.name != "v1":
        out["top_k"] = k
    for name in P.corpora:
        rag = R.RebuildRagIndex(E.RUN, name)
        for t in texts[:5]:
            rag.index.search(rag.encode([t]), k)
        embed_s, search_s = [], []
        for t in texts:
            t0 = time.perf_counter()
            vec = rag.encode([t])
            t1 = time.perf_counter()
            rag.index.search(vec, k)
            t2 = time.perf_counter()
            embed_s.append(t1 - t0)
            search_s.append(t2 - t1)
        embed, search = np.array(embed_s), np.array(search_s)
        out["corpora"][name] = {
            "n_docs": int(rag.index.ntotal),
            **describe(embed * 1000, "embed_ms"),
            **describe(search * 1000, "faiss_search_ms"),
            **describe((embed + search) * 1000, "retrieval_total_ms"),
        }
        s = out["corpora"][name]
        C.log(f"{name}: embed median={s['embed_ms_median']:.2f}ms "
              f"search median={s['faiss_search_ms_median']:.3f}ms "
              f"total p95={s['retrieval_total_ms_p95']:.2f}ms")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-retrieval", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    E.add_profile_args(ap)
    args = ap.parse_args()
    P, out = E.resolve(args)

    preds = E.load_predictions(P)
    table = cost_table(P, preds)
    table.to_csv(out / "cost_latency.csv", index=False, encoding="utf-8-sig")
    C.log(f"已寫入 {out / 'cost_latency.csv'}")
    C.log(table[["code", "llm_latency_s_median", "llm_latency_s_p95", "prompt_tokens_median",
                 "output_tokens_median", "llm_total_hours"]].to_string(index=False))

    if args.measure_retrieval:
        result = measure_retrieval(P, args.n)
        main_manifest = C.load_json(C.RUNS / "MAIN" / "run_manifest.json")
        if P.name == "v1":
            experiment_manifest = P.manifest
            result["experiment_environment"] = {
                "custom_prompt_eval_platform": experiment_manifest.get("platform"),
                "custom_prompt_eval_python": experiment_manifest.get("python"),
                "custom_prompt_eval_model": experiment_manifest.get("model"),
                "MAIN_hardware": main_manifest.get("hardware"),
                "note": "15_ 的 manifest 只記錄 platform 與 python，沒有 GPU 欄位；"
                        "platform 字串與 MAIN 相同，硬體欄位引用 MAIN（RTX 2070 8 GB、CUDA 12.6）。",
            }
        else:
            result["experiment_environment"] = {
                "experiment": P.experiment,
                "platform": P.manifest.get("platform"),
                "python": P.manifest.get("python"),
                "ollama_version": P.manifest.get("ollama_version"),
                "model": P.manifest.get("model"),
                "MAIN_hardware": main_manifest.get("hardware"),
                "note": "27_ 的 manifest 沒有 GPU 欄位；硬體欄位引用 MAIN（RTX 2070 8 GB、CUDA 12.6）。",
            }
        result["profile"] = P.name
        C.save_json(out / "retrieval_latency.json", result)
        C.log(f"已寫入 {out / 'retrieval_latency.json'}")


if __name__ == "__main__":
    main()
