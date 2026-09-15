"""執行單一實驗條件，把逐筆預測落地。

**中斷續跑是這支腳本的核心保證。** 一輪完整實驗要數萬次 LLM 呼叫、跑好幾個小時，
中途一定會遇到關機、當機或系統強制重啟（2026-09-15 就被 Windows Update 的計劃內
升級砍斷過一次）。因此：

* 每一筆結果**立刻** append 進 JSONL 並 flush，不做批次緩衝 —— 斷在任何一刻，
  已經跑完的都留得住。
* 啟動時讀取既有 JSONL，跳過已完成的 id，只補沒跑到的。
* 續跑前比對設定指紋（指令 SHA256、語料庫 hash、模型 digest、生成參數）。
  **指紋不符就拒絕續跑** —— 否則同一個檔案裡會混進兩種設定的結果，
  而這種汙染事後幾乎不可能發現。

用法：
    python 06_run_condition.py --condition S0 [--run MAIN] [--limit N] [--top-k 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# 條件定義：rag 指用哪個語料庫（None = 不接 RAG），instruction 指用哪一版指令段。
# 「有沒有 RAG」與「指令有沒有被最佳化」是兩個獨立開關 —— 這正是原始實驗做不到的，
# 它的候選模板每一個都內建 reference_context，兩個變因永遠綁在一起。
CONDITIONS = {
    "S0":  {"rag": None,    "instruction": "simple"},
    "S1":  {"rag": None,    "instruction": "optimized_norag"},
    "S2a": {"rag": "noaug", "instruction": "simple"},
    "S2b": {"rag": "noaug", "instruction": "optimized_norag"},
    "S3a": {"rag": "aug",   "instruction": "simple"},
    "S3b": {"rag": "aug",   "instruction": "optimized_norag"},
    "S4":  {"rag": "aug",   "instruction": "optimized_aug"},
}


def resolve_instruction(kind: str, run_dir: Path, prompts) -> tuple[str, str]:
    """取得指令段全文與其 SHA256。

    simple 直接來自 02_prompts（跑過 S0 之後就不得再改，其 hash 由驗收 gate 斷言）；
    optimized_* 來自 05_optimize_prompt 的輸出，尚未產生時直接中止，不做任何回退，
    以免安靜地用錯指令。
    """
    if kind == "simple":
        return prompts.SIMPLE_INSTRUCTION, prompts.SIMPLE_INSTRUCTION_SHA256

    path = run_dir / f"best_instruction_{kind.replace('optimized_', '')}.json"
    if not path.exists():
        C.die(f"需要 {kind} 指令，但 {path} 不存在。\n"
              f"請先執行：python 05_optimize_prompt.py --mode "
              f"{'norag' if kind == 'optimized_norag' else 'rag_aug'} --run {run_dir.name}")
    data = C.load_json(path)
    text = data["instruction"]
    return text, C.sha256_text(text)


def build_fingerprint(args, cond: dict, instr_sha: str, manifest: dict,
                      corpus_meta: dict | None, split_sha: str) -> dict:
    """這一輪跑的所有設定的指紋，續跑時逐項比對。"""
    return {
        "condition": args.condition,
        "instruction_kind": cond["instruction"],
        "instruction_sha256": instr_sha,
        "rag_corpus": cond["rag"],
        "corpus_sha256": (corpus_meta or {}).get("index_sha256"),
        "top_k": args.top_k if cond["rag"] else None,
        "test_split_sha256": split_sha,
        "model_digest": manifest.get("model", {}).get("digest"),
        "generation_options": manifest.get("generation_options"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--run", default="MAIN")
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 筆（試跑用；正式執行不得設定）")
    ap.add_argument("--top-k", type=int, default=3, help="RAG 檢索筆數")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    run_dir = C.RUNS / args.run
    prompts = C.load_module("02_prompts")

    # --- 環境指紋（00_probe_env.py 的輸出）-------------------------------
    manifest = C.load_json(run_dir / "run_manifest.json")
    if not manifest:
        C.die(f"找不到 {run_dir / 'run_manifest.json'}。請先執行：python 00_probe_env.py --run {args.run}")

    # --- 測試集（切分當時的 hash 必須相符）-------------------------------
    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    split_sha = audit.get("split_files", {}).get("test", {}).get("sha256")
    test_path = C.DATA_SPLITS / "test.csv"
    if split_sha:
        C.require_hash(test_path, split_sha, "test 切分檔")
    df = pd.read_csv(test_path)
    if args.limit:
        df = df.head(args.limit)

    # --- 指令段 ----------------------------------------------------------
    instruction, instr_sha = resolve_instruction(cond["instruction"], run_dir, prompts)

    # --- RAG（只有需要的條件才載入）--------------------------------------
    rag = None
    corpus_meta = None
    if cond["rag"]:
        corpus = C.load_module("03_build_corpus")
        rag = corpus.RagIndex(cond["rag"])
        corpus_meta = rag.meta

    # --- 續跑檢查 --------------------------------------------------------
    out_path = run_dir / "preds" / f"{args.condition}.jsonl"
    fp_path = run_dir / "preds" / f"{args.condition}.fingerprint.json"
    fingerprint = build_fingerprint(args, cond, instr_sha, manifest, corpus_meta, split_sha)

    if fp_path.exists():
        old = C.load_json(fp_path)
        diffs = [k for k in fingerprint if old.get(k) != fingerprint[k]]
        if diffs:
            C.die(
                f"設定與既有結果不符，拒絕續跑（避免同一檔案混入兩種設定）。\n"
                f"  不符欄位：{diffs}\n"
                f"  既有：{ {k: old.get(k) for k in diffs} }\n"
                f"  現在：{ {k: fingerprint[k] for k in diffs} }\n"
                f"若確定要重跑，請先刪除：\n  {out_path}\n  {fp_path}"
            )
    else:
        C.save_json(fp_path, fingerprint)

    already = C.done_ids(out_path)
    todo = df[~df["id"].isin(already)]
    C.log(f"條件 {args.condition}：共 {len(df):,} 筆，"
          f"已完成 {len(already):,}，待跑 {len(todo):,}")
    if cond["rag"]:
        C.log(f"  RAG 語料庫={cond['rag']}  top_k={args.top_k}")
    C.log(f"  指令={cond['instruction']}  sha256={instr_sha[:16]}…")
    if todo.empty:
        C.log("已全部完成，無須續跑。")
        return

    # --- 主迴圈 ----------------------------------------------------------
    t_start = time.perf_counter()
    done = 0
    for row in todo.itertuples(index=False):
        reference_block = None
        src_ids = None
        if rag is not None:
            docs, src_ids, sims = rag.search(row.statement, args.top_k)
            reference_block = prompts.build_reference_block(docs)

        prompt = prompts.assemble(instruction, row.statement, reference_block)
        resp = C.chat(prompt)
        pred, reason = C.parse_label(resp["raw_response"])

        C.append_jsonl(out_path, {
            "id": row.id,
            "condition": args.condition,
            "true_label_id": int(row.label_id),
            "true_label": row.status,
            "pred_label_id": pred,
            "pred_label": C.ID_TO_LABEL.get(pred) if pred is not None else None,
            "parse_reason": reason,
            "raw_response": resp["raw_response"],
            "retrieved_ids": src_ids,
            "elapsed_s": round(resp["elapsed_s"], 3),
            "prompt_eval_count": resp["prompt_eval_count"],
            "eval_count": resp["eval_count"],
        })

        done += 1
        if done % args.progress_every == 0 or done == len(todo):
            rate = done / (time.perf_counter() - t_start)
            eta = (len(todo) - done) / rate if rate else 0
            C.log(f"  {done:,}/{len(todo):,}  "
                  f"{rate:.2f} 筆/秒  預估剩餘 {eta / 60:.1f} 分")

    total = time.perf_counter() - t_start
    C.log(f"完成 {done:,} 筆，耗時 {total / 60:.1f} 分（{done / total:.2f} 筆/秒）")
    C.log(f"輸出：{out_path}")


if __name__ == "__main__":
    main()
