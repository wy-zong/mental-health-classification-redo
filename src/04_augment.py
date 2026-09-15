"""用 LLM 改寫 train 的文本，產生擴增語料。

**只對 train 做。** val/test 一筆都不進來 —— 否則擴增出來的變體會把評估資料的內容
帶進檢索庫，形成新的洩漏管道。

改寫指令**完全不提類別名**，這是與原研究最關鍵的差異。原本的寫法是
`Please rewrite the following text with {emotion} emotion into 1 different version.`
其中 `{emotion}` 直接就是類別名，結果 57.9% 的改寫文本自帶自己的類別詞、
13.4% 含其他類別的詞（那是標籤雜訊）。這讓擴增資料變成「貼標籤」而不是「增加多樣性」，
很可能正是原研究擴增效果只有 +0.0009 的原因。

三道關卡：
  1. **改寫引入了原文沒有的類別詞** -> 丟棄。注意是「引入」而非「含有」：
     原文本身提到自己的診斷是真實的病患語言，把這種樣本剔掉會系統性排除
     資訊量最高的一批，反而製造選擇偏誤。
  2. 與原文語意相似度過低（改寫跑題）-> 丟棄
  3. 標籤保持驗證：用 train 上訓練的分類器檢查改寫後是否仍判為同一類
     -> **只記錄、不丟棄**。分類器本身並不完美，拿它當篩子會系統性地
     偏向「容易分類的樣本」，反而消滅掉擴增本來要帶進來的多樣性。
     這道量化取代了原研究中因時間不足而未執行的人工稽核。

用法：
    python 04_augment.py --run MAIN [--n-paraphrases 2] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REWRITE_INSTRUCTION = (
    "Rewrite the following text in different words while preserving its original "
    "meaning, tone, and emotional content. Output only the rewritten text."
)

# 改寫要輸出完整文本，不能沿用分類用的 num_predict=16
REWRITE_OPTIONS = {
    "temperature": 0.7,   # 需要一些變化才有擴增價值；seed 固定以維持可重現
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 1024,
}

# 常見的樣板開頭，原研究有 66% 的改寫以這類句子開頭
BOILERPLATE_RE = re.compile(
    r"^\s*(here('s| is)|sure[,!]|okay[,!]|i('ve| have)|below is|rewritten version)",
    re.IGNORECASE,
)
_LABEL_RE = {
    label: re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE) for label in C.LABELS
}
MIN_SIMILARITY = 0.50   # 與原文的 cosine 下限；低於此視為改寫跑題


def strip_boilerplate(text: str) -> tuple[str, bool]:
    """移除「Here is a rewritten version:」這類開場白，回傳 (文本, 是否命中樣板)。"""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "", False
    if BOILERPLATE_RE.match(lines[0]) and len(lines) > 1:
        return "\n".join(lines[1:]).strip(), True
    hit = bool(BOILERPLATE_RE.match(lines[0]))
    return "\n".join(lines).strip(), hit


def label_words_in(text: str) -> list[str]:
    return [label for label, rx in _LABEL_RE.items() if rx.search(text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN")
    ap.add_argument("--n-paraphrases", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 筆（試跑用）")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    C.single_instance("augment")

    run_dir = C.RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = C.load_json(run_dir / "run_manifest.json")
    if not manifest:
        C.die(f"請先執行：python 00_probe_env.py --run {args.run}")

    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    train_path = C.DATA_SPLITS / "train.csv"
    if audit.get("split_files", {}).get("train", {}).get("sha256"):
        C.require_hash(train_path, audit["split_files"]["train"]["sha256"], "train 切分檔")
    train = pd.read_csv(train_path)
    if args.limit:
        train = train.head(args.limit)

    raw_path = run_dir / "augment_raw.jsonl"
    already = {rec["gen_key"] for rec in C.read_jsonl(raw_path) if "gen_key" in rec}
    total = len(train) * args.n_paraphrases
    C.log(f"擴增來源：train {len(train):,} 筆 × {args.n_paraphrases} 版 = {total:,} 次，"
          f"已完成 {len(already):,}")
    C.log(f"  改寫參數：{REWRITE_OPTIONS}")
    C.log("  改寫指令不含任何類別名")

    # --- 生成（可中斷續跑）-------------------------------------------------
    t0 = time.perf_counter()
    done_now = 0
    for row in train.itertuples(index=False):
        for k in range(args.n_paraphrases):
            key = f"{row.id}#{k}"
            if key in already:
                continue
            prompt = f"{REWRITE_INSTRUCTION}\n\nText:\n{row.statement.strip()}"
            # 每個變體換一個 seed，否則固定 seed 會讓同一筆的多個版本完全相同
            resp = C.chat(prompt, options={**REWRITE_OPTIONS,
                                           "seed": REWRITE_OPTIONS["seed"] + k})
            C.append_jsonl(raw_path, {
                "gen_key": key,
                "source_id": row.id,
                "variant": k,
                "status": row.status,
                "label_id": int(row.label_id),
                "original": row.statement,
                "raw_response": resp["raw_response"],
                "elapsed_s": round(resp["elapsed_s"], 3),
                "eval_count": resp["eval_count"],
            })
            done_now += 1
            if done_now % args.progress_every == 0:
                rate = done_now / (time.perf_counter() - t0)
                remain = max(total - len(already) - done_now, 0)
                C.log(f"  {done_now:,} 已生成（{rate:.2f}/秒，剩餘約 {remain / rate / 60:.1f} 分）")

    if done_now:
        C.log(f"生成完成 {done_now:,} 筆，耗時 {(time.perf_counter() - t0) / 60:.1f} 分")

    # --- 過濾 ---------------------------------------------------------------
    C.log("過濾與稽核 …")
    records = list(C.read_jsonl(raw_path))
    stats = {
        "generated": len(records),
        "empty": 0,
        "boilerplate_prefix": 0,
        "dropped_introduced_label_word": 0,
        "dropped_low_similarity": 0,
        "kept": 0,
    }
    label_word_hits = {label: 0 for label in C.LABELS}
    preserved_label_words = {label: 0 for label in C.LABELS}
    other_label_hits = 0

    staged: list[dict] = []
    for rec in records:
        text, had_boiler = strip_boilerplate(rec["raw_response"])
        stats["boilerplate_prefix"] += int(had_boiler)
        if not text:
            stats["empty"] += 1
            continue

        # 只有「改寫引入了原文沒有的類別詞」才算污染。原文本身就提到自己的診斷
        # （"my depression got worse"）是真實的病患語言，把這種樣本丟掉會系統性
        # 排除掉資訊量最高的一批，反而製造選擇偏誤。
        orig_hits = set(label_words_in(rec["original"]))
        new_hits = set(label_words_in(text))
        for h in new_hits & orig_hits:
            preserved_label_words[h] += 1
        introduced = new_hits - orig_hits
        if introduced:
            for h in introduced:
                label_word_hits[h] += 1
            if any(h != rec["status"] for h in introduced):
                other_label_hits += 1
            stats["dropped_introduced_label_word"] += 1
            continue

        staged.append({**rec, "text": text})

    # 語意相似度（與原文），一次批次計算
    if staged:
        from sentence_transformers import SentenceTransformer

        # 刻意指定 CPU：在 GPU 上跑這一步會拋
        # `CUDA error: an illegal memory access was encountered`（torch 2.14.0+cu126
        # ＋ RTX 2070 / sm_75 ＋ sentence-transformers 6.0.1），分批與縮小 batch 都無效，
        # 而同一張卡上 03_build_corpus 的 encode、矩陣運算、DistilBERT 微調都正常，
        # 所以問題侷限在這個組合。這一步整輪只跑一次、又不是瓶頸，用 CPU 換穩定。
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        model.max_seq_length = 512
        def enc(xs: list[str]) -> np.ndarray:
            """分批編碼。一次送上萬筆會讓 GPU 記憶體壓力過大；2026-09-16 曾在
            這一行遇到 CUDA illegal memory access（當時另有一個程序同時佔用 GPU）。"""
            out = []
            for i in range(0, len(xs), 512):
                out.append(np.asarray(
                    model.encode(xs[i:i + 512], batch_size=32, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False),
                    dtype="float32"))
            return np.vstack(out)

        sims = (enc([s["original"] for s in staged]) * enc([s["text"] for s in staged])).sum(1)
    else:
        sims = np.array([])

    kept_rows = []
    for rec, sim in zip(staged, sims):
        if sim < MIN_SIMILARITY:
            stats["dropped_low_similarity"] += 1
            continue
        kept_rows.append({
            "id": f"aug_{rec['source_id']}_{rec['variant']}",
            "source_id": rec["source_id"],
            "statement": rec["text"],
            "status": rec["status"],
            "label_id": rec["label_id"],
            "similarity_to_source": round(float(sim), 4),
        })
    stats["kept"] = len(kept_rows)

    aug = pd.DataFrame(kept_rows)
    out_csv = run_dir / "augmented.csv"
    aug.to_csv(out_csv, index=False, encoding="utf-8")

    # --- 標籤保持驗證：只量化、不拿來篩選 ----------------------------------
    preservation = label_preservation_check(train, aug)

    report = {
        "instruction": REWRITE_INSTRUCTION,
        "instruction_mentions_class_names": False,
        "options": REWRITE_OPTIONS,
        "n_paraphrases": args.n_paraphrases,
        "source_split": "train",
        "source_rows": int(len(train)),
        "filters": stats,
        "retention_rate": round(stats["kept"] / stats["generated"], 4) if stats["generated"] else 0,
        "label_words_introduced_by_rewrite": label_word_hits,
        "label_words_already_in_source": preserved_label_words,
        "introduced_other_class_word": other_label_hits,
        "filter_rule": "只丟棄改寫引入的類別詞；原文本身含有的不視為污染",
        "min_similarity_threshold": MIN_SIMILARITY,
        "similarity": {
            "mean": round(float(np.mean(sims)), 4) if len(sims) else None,
            "p05": round(float(np.percentile(sims, 5)), 4) if len(sims) else None,
        },
        "label_preservation": preservation,
        "output_rows": int(len(aug)),
        "output_sha256": C.sha256_file(out_csv),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    C.save_json(run_dir / "augment_audit.json", report)

    C.log(f"  生成 {stats['generated']:,} → 保留 {stats['kept']:,}"
          f"（保留率 {report['retention_rate']:.1%}）")
    C.log(f"  改寫引入類別詞而丟棄：{stats['dropped_introduced_label_word']:,}"
          f"（其中引入他類詞 {other_label_hits:,}）")
    C.log(f"  原文本就含類別詞（保留，不視為污染）：{sum(preserved_label_words.values()):,}")
    C.log(f"  相似度過低而丟棄：{stats['dropped_low_similarity']:,}")
    C.log(f"  樣板開頭出現：{stats['boilerplate_prefix']:,}")
    if preservation.get("status") == "ok":
        C.log(f"  標籤保持率（分類器驗證，僅記錄）：{preservation['preserved_rate']:.1%}")
    C.log(f"已寫入 {out_csv}")


def label_preservation_check(train: pd.DataFrame, aug: pd.DataFrame) -> dict:
    """用 train 上訓練的 TF-IDF 分類器檢查改寫後是否仍被判為同一類。

    **只量化，不拿來篩選**（見模組說明）。這道檢查取代原研究中因時間不足而
    未執行的人工標籤保持稽核，且完全可重現。
    """
    if aug.empty:
        return {"status": "skipped", "reason": "no augmented rows"}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, n_jobs=-1),
    )
    clf.fit(train["statement"], train["label_id"])
    pred = clf.predict(aug["statement"])
    preserved = (pred == aug["label_id"].to_numpy())
    per_class = {
        label: round(float(preserved[aug["status"] == label].mean()), 4)
        for label in C.LABELS if (aug["status"] == label).any()
    }
    return {
        "status": "ok",
        "classifier": "TF-IDF(1-2gram) + LogisticRegression, fitted on train",
        "note": "僅供量化，未用於篩選擴增資料",
        "n": int(len(aug)),
        "preserved_rate": round(float(preserved.mean()), 4),
        "per_class": per_class,
        "train_self_accuracy": round(float(clf.score(train["statement"], train["label_id"])), 4),
    }


if __name__ == "__main__":
    main()
