"""從原始資料建立 train / val_search / val_confirm / test 四個互斥集合。

這是整條流程的第一步，也是唯一一次碰原始資料。設計重點：

* **先去重再切分，且去重涵蓋「逐字」與「近似」兩種。** 原始實驗的 `預處理.py` 直接
  train_test_split，沒有任何去重步驟，而資料集本身有大量重複文本 —— 同一份文字會各站
  train/test 一邊，從第一步就洩漏。實測顯示光靠逐字去重不夠：`'Restless and agitated.'`
  與 `'Restless and agitated'` 只差一個句點，正規化後仍是兩筆；另有大量同一則貼文的
  不同節錄／改寫版本（cosine 0.92–0.998）。近似去重必須在切分**之前**做 —— 切分後才剔除
  會破壞類別平衡，而 Bipolar 去重後僅 2,501 筆，沒有任何緩衝空間。
* **切分結果立刻鎖死。** 四個集合各自存檔並記錄 SHA256，之後每支腳本開場都要驗 hash。
* **互斥驗證涵蓋逐字與近似兩種。** 逐字重複必須為 0；近似重複列出跨集合最大相似度，
  作為論文可引用的證據。

用法：
    python 01_prepare_data.py [--per-class 2500] [--seed 42] [--skip-neighbor-scan]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# 四個集合的比例，加總必須為 1。val 拆成 search/confirm 兩層是為了讓 TPE 的迭代成本
# 可控（只吃 search），但最終選型在較大的 confirm 上定案 —— 直接回應前一輪實驗
# val=222 過小導致 prompt 最佳化過擬合、在 test 上顯著變差的教訓。
SPLIT_RATIOS = {
    "train": 0.60,
    "val_search": 0.04,
    "val_confirm": 0.11,
    "test": 0.25,
}

_WS = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """去重用的正規化鍵：統一空白、去頭尾、轉小寫。

    只用於判斷「是不是同一份文字」，落地的一律是原文。
    """
    return _WS.sub(" ", str(text)).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2500,
                    help="每類的目標樣本數；若去重後不足則自動下修到最小類別數")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-neighbor-scan", action="store_true",
                    help="跳過近似去重與掃描（需要 sentence-transformers，第一次會下載模型）")
    ap.add_argument("--near-dup-threshold", type=float, default=0.90,
                    help="近似重複的 cosine 門檻；實測 0.92–0.998 幾乎都是同一則貼文的"
                         "節錄或改寫，0.90 誤殺率低（test 集僅 0.4%%）")
    args = ap.parse_args()

    audit: dict = {"seed": args.seed, "split_ratios": SPLIT_RATIOS}

    # --- 0. 確認原始資料就是我們以為的那一份 -------------------------------
    C.log("驗證原始資料 hash …")
    C.require_hash(C.RAW_DATASET, C.RAW_DATASET_SHA256, "原始資料集")
    audit["raw_dataset"] = {
        "path": C.RAW_DATASET.name,
        "sha256": C.RAW_DATASET_SHA256,
    }

    df = pd.read_csv(C.RAW_DATASET)
    df = df[["statement", "status"]].copy()
    audit["raw_rows"] = int(len(df))
    C.log(f"原始資料 {len(df):,} 筆，{df['status'].nunique()} 類")

    # --- 1. 取四類 ---------------------------------------------------------
    before = len(df)
    df = df[df["status"].isin(C.LABELS)]
    audit["dropped_other_classes"] = {
        "rows": int(before - len(df)),
        "classes": sorted(set(pd.read_csv(C.RAW_DATASET)["status"].dropna().unique())
                          - set(C.LABELS)),
    }
    C.log(f"取四類後 {len(df):,} 筆"
          f"（捨棄 {before - len(df):,} 筆，類別：{audit['dropped_other_classes']['classes']}）")

    # --- 2. 清掉空值，然後去重 --------------------------------------------
    df = df.dropna(subset=["statement", "status"])
    df["statement"] = df["statement"].astype(str)
    df = df[df["statement"].str.strip().astype(bool)]
    n_before_dedup = len(df)

    df["_key"] = df["statement"].map(normalize_for_dedup)
    dup_mask = df["_key"].duplicated(keep="first")
    n_dups = int(dup_mask.sum())

    # 同一份文字掛在不同類別下 —— 這種樣本連「正確答案是什麼」都無法確定，整組剔除
    conflict_keys = (
        df.groupby("_key")["status"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    n_conflict_rows = int(df["_key"].isin(conflict_keys).sum())

    df = df[~df["_key"].isin(conflict_keys)]
    df = df[~df["_key"].duplicated(keep="first")]

    audit["dedup"] = {
        "rows_before": int(n_before_dedup),
        "exact_duplicate_rows_removed": n_dups,
        "label_conflicting_texts": len(conflict_keys),
        "label_conflicting_rows_removed": n_conflict_rows,
        "rows_after": int(len(df)),
    }
    C.log(f"去重：{n_before_dedup:,} → {len(df):,} 筆"
          f"（逐字重複 {n_dups:,}；標籤互斥的同文 {len(conflict_keys):,} 組／{n_conflict_rows:,} 筆已整組剔除）")

    counts = df["status"].value_counts()
    C.log("逐字去重後各類：" + "，".join(f"{k} {v:,}" for k, v in counts.items()))
    audit["per_class_after_exact_dedup"] = {k: int(v) for k, v in counts.items()}

    # --- 2b. 近似去重（必須在切分前做）------------------------------------
    if args.skip_neighbor_scan:
        C.log("!! 依參數跳過近似去重（正式執行不得跳過）")
        audit["near_dedup"] = {"status": "SKIPPED"}
    else:
        df, audit["near_dedup"] = collapse_near_duplicates(df, args.near_dup_threshold)
        counts = df["status"].value_counts()
        C.log("近似去重後各類：" + "，".join(f"{k} {v:,}" for k, v in counts.items()))
    audit["per_class_after_dedup"] = {k: int(v) for k, v in counts.items()}

    # --- 3. 類別平衡 -------------------------------------------------------
    smallest = int(counts.min())
    per_class = min(args.per_class, smallest)
    if per_class < args.per_class:
        C.log(f"!! 最小類別只有 {smallest:,} 筆，每類目標自動下修為 {per_class:,}")
    audit["per_class"] = per_class

    rng = np.random.RandomState(args.seed)
    balanced = pd.concat([
        df[df["status"] == label].sample(n=per_class, random_state=rng)
        for label in C.LABELS
    ]).reset_index(drop=True)
    C.log(f"類別平衡後共 {len(balanced):,} 筆（每類 {per_class:,}）")

    # --- 4. 四向切分（每類各自切，確保四個集合都是類別平衡的）--------------
    sizes = {name: int(round(per_class * r)) for name, r in SPLIT_RATIOS.items()}
    sizes["train"] = per_class - sum(v for k, v in sizes.items() if k != "train")  # 補足餘數
    audit["per_class_split_sizes"] = sizes
    C.log("每類切分：" + "，".join(f"{k} {v:,}" for k, v in sizes.items()))

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in C.SPLITS}
    for label in C.LABELS:
        pool = balanced[balanced["status"] == label].sample(frac=1.0, random_state=rng)
        start = 0
        for name in C.SPLITS:
            end = start + sizes[name]
            parts[name].append(pool.iloc[start:end])
            start = end

    splits = {
        name: pd.concat(frames)
        .sample(frac=1.0, random_state=rng)
        .reset_index(drop=True)
        for name, frames in parts.items()
    }

    # --- 5. 互斥驗證（逐字）------------------------------------------------
    C.log("逐字互斥驗證 …")
    keysets = {name: set(sdf["_key"]) for name, sdf in splits.items()}
    exact_overlaps = {}
    for i, a in enumerate(C.SPLITS):
        for b in C.SPLITS[i + 1:]:
            n = len(keysets[a] & keysets[b])
            exact_overlaps[f"{a}|{b}"] = n
            if n:
                C.die(f"{a} 與 {b} 有 {n} 筆逐字重複 —— 切分邏輯有誤，不得繼續")
    audit["exact_overlaps"] = exact_overlaps
    C.log("  四個集合兩兩逐字重複皆為 0 ✓")

    # --- 6. 落地 -----------------------------------------------------------
    C.DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, sdf in splits.items():
        out = sdf[["statement", "status"]].copy()
        out["label_id"] = out["status"].map(C.LABEL_TO_ID)
        out.insert(0, "id", [f"{name}_{i:05d}" for i in range(len(out))])
        path = C.DATA_SPLITS / f"{name}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        hashes[name] = C.sha256_file(path)
        C.log(f"  {path.name}: {len(out):,} 筆  sha256={hashes[name][:16]}…")
    audit["split_files"] = {
        name: {"rows": int(len(splits[name])), "sha256": hashes[name]}
        for name in C.SPLITS
    }

    # --- 7. 近似重複掃描 ---------------------------------------------------
    if args.skip_neighbor_scan:
        C.log("!! 依參數跳過近似重複複驗（正式執行不得跳過）")
        audit["near_duplicate_scan"] = {"status": "SKIPPED"}
    else:
        audit["near_duplicate_scan"] = near_duplicate_scan(splits, args.near_dup_threshold)

    C.save_json(C.DATA_SPLITS / "split_audit.json", audit)
    C.log(f"稽核報告：{C.DATA_SPLITS / 'split_audit.json'}")
    C.log("完成。")


EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MAX_SEQ = 512  # 明確指定，不吃 sentence-transformers 預設的 256


def _embedder():
    """回傳一個把文字列表轉成正規化向量的函式。

    向量一律正規化、用內積（＝cosine）。原始實驗是未正規化向量配 IndexFlatL2，
    等於在算「未正規化向量的平方 L2」，那是 bug 不是設計選擇。
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = EMBED_MAX_SEQ

    def embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    return embed


def collapse_near_duplicates(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    """把語意上相同的樣本摺疊成一筆，並剔除標籤互相矛盾的群組。

    逐字去重抓不到「只差一個標點」或「同一則貼文的不同節錄」。這裡對全池做
    range_search，把相似度 ≥ threshold 的樣本用 union-find 連成群組：

      * 群組內標籤一致 -> 只保留一筆（其餘視為重複）
      * 群組內標籤不一致 -> **整組剔除**（連正確答案是什麼都無法確定的樣本，
        留著只會同時污染訓練訊號與評估基準）

    這一步必須在切分之前執行，切分後才做會破壞類別平衡。
    """
    faiss = C.import_faiss()
    C.log(f"近似去重（門檻 cosine ≥ {threshold}，全池 {len(df):,} 筆，載入 embedding 模型 …）")

    texts = df["statement"].tolist()
    vecs = _embedder()(texts)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # range_search 的 radius 是內積下界；回傳含自己（相似度 1.0），稍後排除
    lims, _sims, idxs = index.range_search(vecs, float(threshold))

    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_pairs = 0
    for i in range(len(df)):
        for j in idxs[lims[i]:lims[i + 1]]:
            if int(j) != i:
                union(i, int(j))
                n_pairs += 1

    groups: dict[int, list[int]] = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)

    labels = df["status"].to_numpy()
    keep_positions: list[int] = []
    n_groups_multi = n_collapsed = n_conflict_groups = n_conflict_rows = 0
    conflict_examples: list[dict] = []

    for members in groups.values():
        if len(members) == 1:
            keep_positions.append(members[0])
            continue
        n_groups_multi += 1
        member_labels = {labels[m] for m in members}
        if len(member_labels) > 1:
            # 同樣的內容被標成不同類別 —— 整組剔除
            n_conflict_groups += 1
            n_conflict_rows += len(members)
            if len(conflict_examples) < 10:
                conflict_examples.append({
                    "labels": sorted(member_labels),
                    "n_members": len(members),
                    "sample_text": str(df["statement"].iloc[members[0]])[:160],
                })
            continue
        keep_positions.append(members[0])
        n_collapsed += len(members) - 1

    kept = df.iloc[sorted(keep_positions)].copy()
    info = {
        "threshold": threshold,
        "embedding_model": EMBED_MODEL,
        "max_seq_length": EMBED_MAX_SEQ,
        "metric": "cosine",
        "rows_before": int(len(df)),
        "similar_pairs_found": int(n_pairs // 2),
        "groups_with_multiple_members": int(n_groups_multi),
        "rows_collapsed_same_label": int(n_collapsed),
        "label_conflicting_groups": int(n_conflict_groups),
        "label_conflicting_rows_removed": int(n_conflict_rows),
        "rows_after": int(len(kept)),
        "conflict_examples": conflict_examples,
    }
    C.log(f"  近似群組 {n_groups_multi:,} 組："
          f"同標籤摺疊掉 {n_collapsed:,} 筆；"
          f"標籤矛盾 {n_conflict_groups:,} 組／{n_conflict_rows:,} 筆整組剔除")
    C.log(f"  {len(df):,} → {len(kept):,} 筆")
    return kept, info


def near_duplicate_scan(splits: dict[str, pd.DataFrame], threshold: float) -> dict:
    """對每個 val/test 樣本，在 train 裡找最近鄰，回報相似度分布。

    逐字重複已經在上一步保證為 0，這裡要抓的是「換了幾個字但實質相同」的樣本。
    用 cosine（向量正規化 + 內積），不是原始實驗那種未正規化向量配 L2 的組合。
    """
    faiss = C.import_faiss()

    C.log("切分後近似重複複驗 …")
    embed = _embedder()

    train_vecs = embed(splits["train"]["statement"].tolist())
    index = faiss.IndexFlatIP(train_vecs.shape[1])
    index.add(train_vecs)

    result: dict = {"reference_set": "train", "metric": "cosine",
                    "embedding_model": EMBED_MODEL, "max_seq_length": EMBED_MAX_SEQ}
    for name in ["val_search", "val_confirm", "test"]:
        qs = splits[name]["statement"].tolist()
        sims, idxs = index.search(embed(qs), 1)
        sims = sims[:, 0]
        n_over = int((sims >= threshold).sum())
        result[name] = {
            "n": int(len(sims)),
            "max_cosine_to_train": float(sims.max()),
            "mean_cosine_to_train": float(sims.mean()),
            "count_ge_threshold": n_over,
            "count_ge_0.95": int((sims >= 0.95).sum()),
            "count_ge_0.90": int((sims >= 0.90).sum()),
            "count_ge_0.80": int((sims >= 0.80).sum()),
        }
        C.log(f"  {name}: 最大相似度 {sims.max():.4f}，"
              f"≥0.95 有 {int((sims >= 0.95).sum())} 筆，"
              f"≥0.90 有 {int((sims >= 0.90).sum())} 筆")
        # 近似去重已在切分前執行過，這裡若還抓得到超過門檻的樣本，代表去重沒生效
        if n_over:
            C.die(f"{name} 仍有 {n_over} 筆與 train 的相似度 ≥ {threshold} —— "
                  f"近似去重未生效，不得繼續")
    C.log(f"  跨集合皆無 ≥{threshold} 的近似重複 ✓")
    return result


if __name__ == "__main__":
    main()
