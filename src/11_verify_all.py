"""驗收 gate：沒有通過（0 FAIL）之前，不得引用任何數字。

每一項都是針對原始實驗真實發生過的某個問題設計的，不是形式檢查。

用法：
    python 11_verify_all.py --run MAIN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


class Checker:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def check(self, step: str, what: str, ok: bool, detail: str = "") -> None:
        self.rows.append((step, what, "PASS" if ok else "FAIL", detail))
        mark = "PASS" if ok else "FAIL"
        C.log(f"  [{mark}] {step} {what}" + (f" — {detail}" if detail else ""))

    def skip(self, step: str, what: str, why: str) -> None:
        self.rows.append((step, what, "SKIP", why))
        C.log(f"  [SKIP] {step} {what} — {why}")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r[2] == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.rows if r[2] == "SKIP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN")
    args = ap.parse_args()

    run_dir = C.RUNS / args.run
    k = Checker()
    prompts = C.load_module("02_prompts")
    run_cond = C.load_module("06_run_condition")

    # --- 01 資料切分 ------------------------------------------------------
    C.log("01 資料切分")
    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    k.check("01", "切分稽核檔存在", bool(audit))
    if audit:
        overlaps = audit.get("exact_overlaps", {})
        k.check("01", "四集合兩兩逐字互斥",
                bool(overlaps) and all(v == 0 for v in overlaps.values()),
                f"{len(overlaps)} 組比較")

        dedup = audit.get("dedup", {})
        k.check("01", "已執行逐字去重",
                dedup.get("exact_duplicate_rows_removed", 0) >= 0 and bool(dedup),
                f"移除 {dedup.get('exact_duplicate_rows_removed', 0):,} 筆重複")

        near = audit.get("near_dedup", {})
        k.check("01", "已執行近似去重（非 SKIPPED）",
                bool(near) and near.get("status") != "SKIPPED",
                f"門檻 {near.get('threshold')}，摺疊 {near.get('rows_collapsed_same_label', 0):,} 筆")

        scan = audit.get("near_duplicate_scan", {})
        if scan and scan.get("status") != "SKIPPED":
            worst = max((scan[s]["max_cosine_to_train"]
                         for s in ("val_search", "val_confirm", "test") if s in scan),
                        default=None)
            thr = near.get("threshold", 0.90)
            k.check("01", "切分後無跨集合近似重複",
                    worst is not None and worst < thr,
                    f"最大相似度 {worst:.4f} < 門檻 {thr}")
        else:
            k.skip("01", "切分後近似重複複驗", "未執行")

        for name, info in audit.get("split_files", {}).items():
            path = C.DATA_SPLITS / f"{name}.csv"
            ok = path.exists() and C.sha256_file(path) == info["sha256"]
            k.check("01", f"{name}.csv hash 未變", ok, f"{info['rows']:,} 筆")

    # --- 00 環境 ----------------------------------------------------------
    C.log("00 環境指紋")
    manifest = C.load_json(run_dir / "run_manifest.json")
    k.check("00", "run_manifest.json 存在", bool(manifest))
    if manifest:
        model = manifest.get("model", {})
        k.check("00", "量化等級已探測（非佔位值）", bool(model.get("quantization_level")),
                str(model.get("quantization_level")))
        k.check("00", "權重 blob digest 已探測", bool(model.get("weights_blob_digest")),
                str(model.get("weights_blob_digest"))[:20] + "…")
        gen = manifest.get("generation_options", {})
        k.check("00", "num_ctx 已明確指定", bool(gen.get("num_ctx")), str(gen.get("num_ctx")))
        hw = manifest.get("hardware", {})
        k.check("00", "torch 可用 CUDA", bool(hw.get("torch_cuda_available")),
                str(hw.get("gpu_name")))

    # --- 03 語料庫 --------------------------------------------------------
    C.log("03 語料庫")
    corpus_dir = C.ROOT / "data" / "corpus"
    for name in ["noaug", "aug"]:
        meta = C.load_json(corpus_dir / f"{name}_meta.json")
        if not meta:
            k.skip("03", f"語料庫 {name}", "尚未建立")
            continue
        k.check("03", f"語料庫 {name} 僅來自 train",
                meta.get("source") in ("train", "train+aug"), str(meta.get("source")))
        leak = meta.get("exact_overlap_with_eval_splits", {})
        k.check("03", f"語料庫 {name} 與評估集無重複",
                bool(leak) and all(v == 0 for v in leak.values()), str(leak))
        k.check("03", f"語料庫 {name} 索引 hash 未變",
                (corpus_dir / f"{name}.index").exists()
                and C.sha256_file(corpus_dir / f"{name}.index") == meta.get("index_sha256"))
        k.check("03", f"語料庫 {name} 已記錄截斷筆數",
                meta.get("truncated_docs") is not None,
                f"{meta.get('truncated_docs', 0):,} 筆（{meta.get('truncated_pct')}%）")

    # --- 04 擴增 ----------------------------------------------------------
    C.log("04 擴增資料")
    aug_audit = C.load_json(run_dir / "augment_audit.json")
    if not aug_audit:
        k.skip("04", "擴增稽核", "尚未執行")
    else:
        k.check("04", "改寫指令不含類別名",
                aug_audit.get("instruction_mentions_class_names") is False)
        k.check("04", "擴增只來自 train", aug_audit.get("source_split") == "train")
        k.check("04", "已記錄保留率與標籤保持率",
                aug_audit.get("retention_rate") is not None
                and aug_audit.get("label_preservation", {}).get("status") == "ok",
                f"保留 {aug_audit.get('retention_rate')}，"
                f"標籤保持 {aug_audit.get('label_preservation', {}).get('preserved_rate')}")

    # --- 05 最佳化：絕不可讀 test -----------------------------------------
    C.log("05 指令最佳化")
    found_opt = False
    for mode in ["norag", "rag_aug"]:
        files = [run_dir / f"optimize_{mode}_stage{s}.jsonl" for s in (1, 2)]
        if not any(f.exists() for f in files):
            k.skip("05", f"最佳化 {mode}", "尚未執行")
            continue
        found_opt = True
        bad = 0
        total = 0
        for f in files:
            for rec in C.read_jsonl(f):
                total += 1
                if str(rec.get("id", "")).startswith("test_"):
                    bad += 1
        # 這是 EDC13 指控的核心：原始實驗拿 test 子集當最佳化目標函數
        k.check("05", f"最佳化 {mode} 全程未觸及 test",
                bad == 0, f"{total:,} 筆評估紀錄，test 樣本 {bad} 筆")

        suffix = "norag" if mode == "norag" else "aug"
        best = C.load_json(run_dir / f"best_instruction_{suffix}.json")
        if best:
            k.check("05", f"最佳化 {mode} 結果已持久化",
                    bool(best.get("instruction")) and bool(best.get("instruction_sha256")),
                    f"{best.get('candidate_id')}")
    if not found_opt:
        k.skip("05", "最佳化結果", "兩種模式都尚未執行")

    # --- 06 條件 ----------------------------------------------------------
    C.log("06 實驗條件")
    test_rows = audit.get("split_files", {}).get("test", {}).get("rows")
    baseline_sha = prompts.SIMPLE_INSTRUCTION_SHA256
    for name, cond in run_cond.CONDITIONS.items():
        path = run_dir / "preds" / f"{name}.jsonl"
        if not path.exists():
            k.skip("06", f"條件 {name}", "尚未執行")
            continue
        records = list(C.read_jsonl(path))
        k.check("06", f"{name} 筆數完整",
                test_rows is None or len(records) == test_rows,
                f"{len(records):,}/{test_rows}")
        k.check("06", f"{name} 100% 記錄 raw_response",
                all("raw_response" in r for r in records))
        k.check("06", f"{name} 無重複 id",
                len({r["id"] for r in records}) == len(records))

        fp = C.load_json(run_dir / "preds" / f"{name}.fingerprint.json")
        if fp:
            k.check("06", f"{name} 模型 digest 與 manifest 一致",
                    fp.get("model_digest") == manifest.get("model", {}).get("digest"))
            if cond["instruction"] == "simple":
                # 基準指令一旦跑過 S0 就不得再改，否則等同用 test 結果做選擇
                k.check("06", f"{name} 基準指令未被竄改",
                        fp.get("instruction_sha256") == baseline_sha)

    # --- 報告 -------------------------------------------------------------
    df = pd.DataFrame(k.rows, columns=["step", "check", "status", "detail"])
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "acceptance_report.csv", index=False, encoding="utf-8")

    n_pass = int((df["status"] == "PASS").sum())
    C.log("")
    C.log(f"PASS {n_pass} / FAIL {k.failed} / SKIP {k.skipped}")
    if k.failed:
        C.log("!! 有項目未通過，不得引用任何數字")
        for row in k.rows:
            if row[2] == "FAIL":
                C.log(f"   FAIL: {row[0]} {row[1]} {row[3]}")
        raise SystemExit(1)
    if k.skipped:
        C.log("（尚有步驟未執行，全流程跑完後需重新驗收）")


if __name__ == "__main__":
    main()
