"""建立論文交接文件的上傳包（issue #29）：給 Claude chat／Claude Project 讀的扁平資料夾。

論文最終版本採用 v2、top-k = 1（#30）。交接文件 reports/論文交接_v2_k1_<日期>.md 已把要引用的
數字寫成表格；本腳本把文件本身、論文用圖與小型結果檔複製到同一個資料夾，方便整包上傳。

* 只收 claude.ai 讀得到的格式（md、csv、json、png），不收逐筆 jsonl、FAISS 索引與大檔
* 檔名加前綴避免撞名：fig_ 圖、k1_ 主結果、k5_ k = 5、v1_ v1、val_ val 選 k、augv2_ 擴寫、
  corpus_ 語料、suicidal_ Suicidal、splits_ 切分
* 位元組原樣複製，複製後逐檔核對 sha256；00_MANIFEST.csv 記錄來源路徑、大小與 sha256
* 每個上傳包檔名都必須出現在交接文件裡，確保文件的路徑索引沒有漏列
* 每次重建前清空輸出資料夾（只清本腳本產生過的資料夾，也就是含 00_MANIFEST.csv 或為空）

用法：
    python 29_paper_handoff_bundle.py [--doc 交接文件] [--out-dir 目錄]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DEFAULT_DOC = C.ROOT / "reports" / "論文交接_v2_k1_20261017.md"
DEFAULT_OUT = C.ROOT / "reports" / "論文交接_v2_k1_上傳包"
DOC_NAME = "00_論文交接_v2_k1.md"
MANIFEST_NAME = "00_MANIFEST.csv"

RUN = "runs/NESTED_70_10_20_INCREMENTAL"
K1 = f"{RUN}/MAIN_EVAL_AUGV2_K1"
CONDITIONS = (
    ("C1", "norag_base", "Fig. 5"),
    ("C2", "rag_noaug_base", "Fig. 6"),
    ("C3", "rag_augv2_base", "Fig. 7"),
    ("C4", "rag_noaug_optimized", "Fig. 9"),
    ("C5", "rag_augv2_optimized", "Fig. 10"),
)

# (上傳包檔名, 來源路徑（相對於 repo 根目錄）, 說明)
BUNDLE: list[tuple[str, str, str]] = [
    *[(f"fig_confusion_paper_{code}_{cond}.png", f"{K1}/figures/confusion_paper_{code}_{cond}.png",
       f"論文 {fig} 格式的 {code} 次數混淆矩陣（不畫 INVALID）") for code, cond, fig in CONDITIONS],
    ("fig_confusion_rownorm_C1-C5.png", f"{K1}/figures/confusion_rownorm_C1-C5.png",
     "五個條件的 row-normalized 混淆矩陣（含 INVALID 欄，審稿人要求）"),
    ("k1_stats_conditions.csv", f"{K1}/stats_conditions.csv", "各條件點估計與 95% CI（Table VI）"),
    ("k1_stats_paired.csv", f"{K1}/stats_paired.csv", "10 組配對比較：Δ、McNemar、Holm（Table VII）"),
    ("k1_stats_per_class.csv", f"{K1}/stats_per_class.csv", "各類別 precision／recall／F1"),
    ("k1_stats_invalid.csv", f"{K1}/stats_invalid.csv", "無效數、無效率、只計有效的指標"),
    ("k1_stats_invalid_paired.csv", f"{K1}/stats_invalid_paired.csv", "無效率的配對檢定"),
    ("k1_stats_invalid_reasons.csv", f"{K1}/stats_invalid_reasons.csv", "無效原因分布"),
    ("k1_summary.csv", f"{K1}/summary.csv", "各條件的 prompt 全文與 sha、token 數、截斷、執行時間"),
    *[(f"k1_confusion_counts_{code}_{cond}.csv", f"{K1}/figures/confusion_{code}_{cond}_counts.csv",
       f"{code} 混淆矩陣次數（含 INVALID）") for code, cond, _ in CONDITIONS],
    ("k1_cost_latency.csv", f"{K1}/cost_latency.csv", "LLM 延遲與 token 數"),
    ("k1_retrieval_latency.json", f"{K1}/retrieval_latency.json", "檢索延遲與硬體"),
    ("k1_determinism_summary.json", f"{K1}/determinism/summary.json", "解碼確定性、VRAM、執行環境"),
    ("k1_label_parsing_within_condition.csv", f"{K1}/label_parsing/parse_rule_within_condition.csv",
     "唯一命中 vs first-match：各條件內的差異"),
    ("k1_label_parsing_paired.csv", f"{K1}/label_parsing/parse_rule_paired.csv",
     "唯一命中 vs first-match：配對結論是否翻轉"),
    ("k1_error_analysis_summary.json", f"{K1}/error_analysis/error_analysis_summary.json", "錯誤分析摘要"),
    ("k1_error_examples.md", f"{K1}/error_analysis/examples.md", "質性範例（敏感內容，已截短）"),
    ("k1_error_confusions_C4.csv", f"{K1}/error_analysis/confusions_C4.csv", "C4 的錯誤類型"),
    ("k1_error_confusions_C5.csv", f"{K1}/error_analysis/confusions_C5.csv", "C5 的錯誤類型"),
    ("k1_error_retrieval_copy_stats.csv", f"{K1}/error_analysis/retrieval_copy_stats.csv", "檢索標籤與預測的關係"),
    ("k1_retrieval_leakage_check.json", f"{K1}/retrieval_leakage/retrieval_leakage_check.json", "檢索洩漏檢查"),
    ("k1_v1_v2_comparison.csv", f"{K1}/v1_v2_comparison.csv", "v1 與 v2 k = 1 的主指標與逐筆一致率"),
    ("k1_v1v2_conditions.csv", f"{K1}/v1_v2_posthoc/conditions.csv", "v1／v2 點估計並列"),
    ("k1_v1v2_paired.csv", f"{K1}/v1_v2_posthoc/paired.csv", "v1／v2 配對比較並列"),
    ("k1_v1v2_per_class.csv", f"{K1}/v1_v2_posthoc/per_class.csv", "v1／v2 各類別並列"),
    ("k1_v1v2_invalid_paired.csv", f"{K1}/v1_v2_posthoc/invalid_paired.csv", "v1／v2 無效率配對並列"),
    ("k1_v1v2_examples_in_v2.csv", f"{K1}/v1_v2_posthoc/v1_examples_in_v2.csv", "v1 範例在 v2 是否仍成立"),
    ("k1_v1v2_summary.json", f"{K1}/v1_v2_posthoc/summary.json", "v1／v2 比對結論"),
    ("val_topk_summary.csv", f"{RUN}/TOPK_VAL_AUGV2/topk_summary.csv", "val 上 k ∈ {1, 3, 5} 的結果"),
    ("k5_stats_conditions.csv", f"{RUN}/MAIN_EVAL_AUGV2/stats_conditions.csv", "k = 5 點估計（敏感度分析）"),
    ("k5_stats_paired.csv", f"{RUN}/MAIN_EVAL_AUGV2/stats_paired.csv", "k = 5 配對比較（敏感度分析）"),
    ("v1_stats_conditions.csv", f"{RUN}/CUSTOM_PROMPT_EVAL/stats_conditions.csv", "v1 點估計（R1 版）"),
    ("v1_stats_paired.csv", f"{RUN}/CUSTOM_PROMPT_EVAL/stats_paired.csv", "v1 配對比較（R1 版）"),
    ("augv2_run_config.json", f"{RUN}/AUGMENT_V2/run_config.json", "改寫 prompt、生成參數、過濾規則"),
    ("augv2_sbert_checkpoint.json", f"{RUN}/AUGMENT_V2/sbert_checkpoint.json", "SBERT 門檻檢查點"),
    ("corpus_noaug_meta.json", f"{RUN}/corpus/noaug_meta.json", "noaug 語料設定"),
    ("corpus_aug_v2_meta.json", f"{RUN}/corpus/aug_v2_meta.json", "aug_v2 語料設定與組成"),
    ("suicidal_summary.json", f"{RUN}/SUICIDAL_SAFETY/summary.json", "Suicidal 量測結果與檢定"),
    ("suicidal_rates.csv", f"{RUN}/SUICIDAL_SAFETY/rates.csv", "Suicidal 與對照組的比率"),
    ("splits_split_manifest.json", f"{RUN}/splits/split_manifest.json", "巢狀切分的筆數與 sha"),
    ("splits_near_duplicate_audit.json", f"{RUN}/splits/near_duplicate_audit.json", "切分之間的近似重複掃描"),
]


def rel(path: Path) -> str:
    """repo 內的檔案記相對路徑，repo 外的（例如 --doc 指向別處）記絕對路徑。"""
    path = path.resolve()
    try:
        return path.relative_to(C.ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries and not (out_dir / MANIFEST_NAME).exists():
            C.die(f"{out_dir} 不是本腳本產生的上傳包（沒有 {MANIFEST_NAME}），拒絕清空")
        if any(p.is_dir() for p in entries):
            C.die(f"{out_dir} 含有子資料夾，拒絕清空")
        for p in entries:
            p.unlink()
    else:
        out_dir.mkdir(parents=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="交接文件（Markdown）")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="上傳包輸出資料夾")
    args = ap.parse_args()

    doc = args.doc.resolve()
    if not doc.exists():
        C.die(f"找不到交接文件 {doc}")
    items = [(DOC_NAME, rel(doc), "交接文件（主文件）")] + BUNDLE

    names = [name for name, _, _ in items]
    if len(set(names)) != len(names):
        C.die("上傳包檔名重複")
    missing = [src for _, src, _ in items if not (C.ROOT / src).exists()]
    if missing:
        C.die("找不到來源檔：\n  " + "\n  ".join(missing))
    doc_text = doc.read_text(encoding="utf-8")
    unlisted = [name for name, _, _ in BUNDLE if name not in doc_text]
    if unlisted:
        C.die("交接文件沒有提到這些上傳包檔名：\n  " + "\n  ".join(unlisted))

    out_dir = args.out_dir.resolve()
    prepare_out_dir(out_dir)
    rows = []
    for name, src, desc in items:
        src_path = C.ROOT / src
        dest = out_dir / name
        shutil.copyfile(src_path, dest)
        sha = C.sha256_file(dest)
        if sha != C.sha256_file(src_path):
            C.die(f"複製後 sha256 不符：{src}")
        rows.append({"bundle_file": name, "source": src, "bytes": dest.stat().st_size,
                     "sha256": sha, "description": desc})

    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["bytes"] for r in rows)
    C.log(f"上傳包：{rel(out_dir)}，{len(rows)} 個檔案＋{MANIFEST_NAME}，共 {total / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
