"""指令段最佳化：兩階段選型，全程只碰 val。

**這支腳本絕對不讀 test。** 原始實驗把 222 筆測試子集拿去當 Optuna 的目標函數
（編輯部意見 EDC13 指控的正是這件事），因此所有已發表數字都不可用。

兩階段設計，回應前一輪實驗的實測失敗：val 只有 222 筆時，選出的指令在 test 上
顯著變差（-2.59pp, p=1.1e-4）—— 那不是「最佳化沒用」，是在小樣本上過擬合。

    階段一  val_search（400 筆）   評估全部候選，成本 = 候選數 × 400
    階段二  val_confirm（1,100 筆）取前 N 名重跑，最終贏家在這裡定案

候選集只有 36 個且維度結構化（見 02_prompts.py），直接窮舉比 TPE 更省也更可解釋：
窮舉成本 36×400 = 14,400 次，TPE 跑 60 trials 反而要 24,000 次，而且結果無法拆解成
「是哪個維度在起作用」。原始實驗的 122 個候選全是同一句話的同義詞替換，
連這種分析都做不了。

用法：
    python 05_optimize_prompt.py --mode norag   --run MAIN
    python 05_optimize_prompt.py --mode rag_aug --run MAIN
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODES = {
    "norag":   {"rag": None,  "out_suffix": "norag"},
    "rag_aug": {"rag": "aug", "out_suffix": "aug"},
}


def evaluate(stage_name: str, candidates: list[dict], df: pd.DataFrame,
             prompts, rag, top_k: int, out_path: Path,
             ref_cache: dict | None) -> dict[str, dict]:
    """在給定樣本集上評估每個候選，逐筆落地、可中斷續跑。

    ref_cache：同一個樣本在不同候選下的檢索結果完全相同（檢索只取決於查詢與語料庫，
    與指令無關），因此預先算一次給所有候選共用，省下數十倍的 embedding 與檢索成本。
    """
    already = {rec["eval_key"] for rec in C.read_jsonl(out_path) if "eval_key" in rec}
    total = len(candidates) * len(df)
    C.log(f"[{stage_name}] {len(candidates)} 候選 x {len(df):,} 樣本 = {total:,} 次，"
          f"已完成 {len(already):,}")

    t0 = time.perf_counter()
    done_now = 0
    for cand in candidates:
        for row in df.itertuples(index=False):
            key = f"{cand['id']}|{row.id}"
            if key in already:
                continue

            reference_block = None
            if rag is not None:
                docs = ref_cache[row.id] if ref_cache else rag.search(row.statement, top_k)[0]
                reference_block = prompts.build_reference_block(docs)

            prompt = prompts.assemble(cand["instruction"], row.statement, reference_block)
            resp = C.chat(prompt)
            pred, reason = C.parse_label(resp["raw_response"])

            C.append_jsonl(out_path, {
                "eval_key": key,
                "stage": stage_name,
                "candidate_id": cand["id"],
                "id": row.id,
                "true_label_id": int(row.label_id),
                "pred_label_id": pred,
                "correct": bool(pred is not None and pred == int(row.label_id)),
                "parse_reason": reason,
                "raw_response": resp["raw_response"],
                "elapsed_s": round(resp["elapsed_s"], 3),
            })
            done_now += 1
            if done_now % 100 == 0:
                rate = done_now / (time.perf_counter() - t0)
                remain = max(total - len(already) - done_now, 0)
                C.log(f"  {done_now:,} 次已跑（{rate:.2f}/秒，預估剩餘 {remain / rate / 60:.1f} 分）")

    scores: dict[str, dict] = {}
    for rec in C.read_jsonl(out_path):
        if rec.get("stage") != stage_name:
            continue
        s = scores.setdefault(rec["candidate_id"], {"n": 0, "correct": 0, "invalid": 0})
        s["n"] += 1
        s["correct"] += int(rec["correct"])
        s["invalid"] += int(rec["pred_label_id"] is None)
    for _cid, s in scores.items():
        s["accuracy"] = s["correct"] / s["n"] if s["n"] else 0.0
        s["invalid_rate"] = s["invalid"] / s["n"] if s["n"] else 0.0
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--run", default="MAIN")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--finalists", type=int, default=5,
                    help="進入階段二的候選數")
    args = ap.parse_args()

    cfg = MODES[args.mode]
    run_dir = C.RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = C.load_module("02_prompts")

    manifest = C.load_json(run_dir / "run_manifest.json")
    if not manifest:
        C.die(f"請先執行：python 00_probe_env.py --run {args.run}")

    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    files = audit.get("split_files", {})

    def load_split(name: str) -> pd.DataFrame:
        path = C.DATA_SPLITS / f"{name}.csv"
        if files.get(name, {}).get("sha256"):
            C.require_hash(path, files[name]["sha256"], f"{name} 切分檔")
        return pd.read_csv(path)

    search_df = load_split("val_search")
    confirm_df = load_split("val_confirm")

    # 明確宣告：這支腳本不開啟 test，驗收 gate 會檢查這件事
    C.log(f"最佳化模式={args.mode}（僅使用 val_search / val_confirm，不讀 test）")

    rag = None
    if cfg["rag"]:
        corpus = C.load_module("03_build_corpus")
        rag = corpus.RagIndex(cfg["rag"])
        C.log(f"  RAG 語料庫={cfg['rag']}（{rag.meta['n_docs']:,} 條目）top_k={args.top_k}")

    candidates = prompts.build_candidates()
    C.log(f"  候選指令 {len(candidates)} 個（四維度完整交叉）")

    def build_cache(df: pd.DataFrame) -> dict | None:
        """同一樣本的檢索鄰居與指令無關，算一次給所有候選共用。"""
        if rag is None:
            return None
        C.log(f"  預先檢索 {len(df):,} 筆（所有候選共用）…")
        t0 = time.perf_counter()
        cache = {row.id: rag.search(row.statement, args.top_k)[0]
                 for row in df.itertuples(index=False)}
        C.log(f"    完成，耗時 {time.perf_counter() - t0:.1f} 秒")
        return cache

    # --- 階段一 -----------------------------------------------------------
    s1_path = run_dir / f"optimize_{args.mode}_stage1.jsonl"
    s1 = evaluate("stage1", candidates, search_df, prompts, rag, args.top_k,
                  s1_path, build_cache(search_df))

    ranked = sorted(s1.items(), key=lambda kv: -kv[1]["accuracy"])
    C.log(f"[階段一] 前 {args.finalists} 名：")
    for cid, s in ranked[:args.finalists]:
        C.log(f"    {cid:40s} acc={s['accuracy']:.4f}  invalid={s['invalid_rate']:.4f}")

    finalist_ids = [cid for cid, _ in ranked[:args.finalists]]
    finalists = [c for c in candidates if c["id"] in finalist_ids]

    # --- 階段二 -----------------------------------------------------------
    s2_path = run_dir / f"optimize_{args.mode}_stage2.jsonl"
    s2 = evaluate("stage2", finalists, confirm_df, prompts, rag, args.top_k,
                  s2_path, build_cache(confirm_df))

    ranked2 = sorted(s2.items(), key=lambda kv: -kv[1]["accuracy"])
    C.log("[階段二] 結果：")
    for cid, s in ranked2:
        C.log(f"    {cid:40s} acc={s['accuracy']:.4f}  invalid={s['invalid_rate']:.4f}")

    best_id = ranked2[0][0]
    best = next(c for c in candidates if c["id"] == best_id)
    baseline_sha = prompts.SIMPLE_INSTRUCTION_SHA256

    out = {
        "mode": args.mode,
        "instruction": best["instruction"],
        "instruction_sha256": best["sha256"],
        "candidate_id": best_id,
        "dims": best["dims"],
        "identical_to_baseline": best["sha256"] == baseline_sha,
        "stage1": {"split": "val_search", "n": int(len(search_df)),
                   "n_candidates": len(candidates), "scores": s1},
        "stage2": {"split": "val_confirm", "n": int(len(confirm_df)),
                   "finalists": finalist_ids, "scores": s2},
        "selected_accuracy_val_confirm": ranked2[0][1]["accuracy"],
        "top_k": args.top_k if rag else None,
        "corpus": cfg["rag"],
        "model_digest": manifest.get("model", {}).get("digest"),
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    out_path = run_dir / f"best_instruction_{cfg['out_suffix']}.json"
    C.save_json(out_path, out)

    C.log(f"勝出：{best_id}  維度={best['dims']}")
    C.log(f"  val_confirm 準確率 {ranked2[0][1]['accuracy']:.4f}")
    C.log(f"  與基準指令相同？{out['identical_to_baseline']}")
    C.log(f"已寫入 {out_path}")


if __name__ == "__main__":
    main()
