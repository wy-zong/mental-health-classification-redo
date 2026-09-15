"""建立 RAG 檢索語料庫，並提供檢索介面給其他腳本使用。

設計重點：

* **只從 train 建。** 原始實驗的語料庫來源池從未與測試集互斥，測試樣本連同
  `true_label` 一起躺在檢索庫裡。這裡的來源固定是 01_prepare_data.py 切出的 train，
  建完還要對 val/test 做**全量**（非抽樣）掃描，逐字重複必須為 0。
* **明確指定 max_seq_length。** `all-MiniLM-L6-v2` 的預設是 256 tokens，實測本資料集
  有 18.9% 的文本超過該長度 —— 近兩成語料庫條目只有前段被編碼。截斷筆數一律量化記錄。
* **正規化 + 內積（cosine）。** 原始實驗是未正規化向量配 IndexFlatL2，等於在算
  「未正規化向量的平方 L2」，那是 bug 不是設計選擇。

條目格式沿用原研究的 `<文本> true_label is <類別>` —— 檢索到的範例要帶標籤才有參考
價值。真正的問題從來不是附標籤，而是語料庫混進了測試資料。

用法：
    python 03_build_corpus.py --source train          # 未擴增語料庫（S2a/S2b 用）
    python 03_build_corpus.py --source train+aug      # 擴增語料庫（S3a/S3b/S4 用）
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

CORPUS_DIR = C.ROOT / "data" / "corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MAX_SEQ = 512

SOURCE_TO_NAME = {"train": "noaug", "train+aug": "aug"}
_WS = re.compile(r"\s+")


def corpus_entry(statement: str, status: str) -> str:
    """語料庫條目格式（沿用原研究）。"""
    return f"{statement} true_label is {status}"


def _load_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = EMBED_MAX_SEQ
    return model


def _encode(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def count_truncated(model, texts: list[str]) -> int:
    """量化有多少條目在 embedding 階段被截斷。"""
    tok = model.tokenizer
    n = 0
    for t in texts:
        if len(tok.encode(t, add_special_tokens=True)) > EMBED_MAX_SEQ:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_TO_NAME))
    ap.add_argument("--run", default="MAIN", help="擴增資料的來源 run")
    args = ap.parse_args()

    faiss = C.import_faiss()
    name = SOURCE_TO_NAME[args.source]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    files = audit.get("split_files", {})

    # --- 來源只能是 train ---------------------------------------------------
    train_path = C.DATA_SPLITS / "train.csv"
    if files.get("train", {}).get("sha256"):
        C.require_hash(train_path, files["train"]["sha256"], "train 切分檔")
    train = pd.read_csv(train_path)

    rows = [(r.id, r.statement, r.status) for r in train.itertuples(index=False)]
    C.log(f"語料庫來源：train {len(rows):,} 筆")

    if args.source == "train+aug":
        aug_path = C.RUNS / args.run / "augmented.csv"
        if not aug_path.exists():
            C.die(f"找不到擴增資料 {aug_path}。請先執行："
                  f"python 04_augment.py --run {args.run}")
        aug = pd.read_csv(aug_path)
        rows += [(r.id, r.statement, r.status) for r in aug.itertuples(index=False)]
        C.log(f"  併入擴增資料 {len(aug):,} 筆，合計 {len(rows):,} 筆")

    doc_ids = [r[0] for r in rows]
    docs = [corpus_entry(r[1], r[2]) for r in rows]

    # --- 編碼 ---------------------------------------------------------------
    C.log(f"編碼中（{EMBED_MODEL}, max_seq_length={EMBED_MAX_SEQ}）…")
    model = _load_embedder()
    t0 = time.perf_counter()
    vecs = _encode(model, docs)
    C.log(f"  完成，耗時 {time.perf_counter() - t0:.1f} 秒，維度 {vecs.shape}")

    n_trunc = count_truncated(model, docs)
    C.log(f"  超過 {EMBED_MAX_SEQ} tokens 而被截斷：{n_trunc:,} 筆"
          f"（{100 * n_trunc / len(docs):.2f}%）")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    index_path = CORPUS_DIR / f"{name}.index"
    faiss.write_index(index, str(index_path))
    C.save_json(CORPUS_DIR / f"{name}_docs.json",
                {"doc_ids": doc_ids, "docs": docs})

    # --- 互斥驗證：語料庫 vs val/test，全量、非抽樣 ------------------------
    C.log("語料庫與 val/test 互斥驗證（全量）…")
    corpus_keys = {_WS.sub(" ", r[1]).strip().lower() for r in rows}
    leakage = {}
    for split in ["val_search", "val_confirm", "test"]:
        sdf = pd.read_csv(C.DATA_SPLITS / f"{split}.csv")
        keys = {_WS.sub(" ", s).strip().lower() for s in sdf["statement"]}
        n = len(corpus_keys & keys)
        leakage[split] = n
        if n:
            C.die(f"語料庫含有 {split} 的 {n} 筆文本 —— 檢索庫混入評估資料，不得繼續")
    C.log("  三個評估集皆無逐字重複 ✓")

    meta = {
        "name": name,
        "source": args.source,
        "n_docs": len(docs),
        "embedding_model": EMBED_MODEL,
        "max_seq_length": EMBED_MAX_SEQ,
        "normalized": True,
        "metric": "cosine (IndexFlatIP on normalized vectors)",
        "truncated_docs": n_trunc,
        "truncated_pct": round(100 * n_trunc / len(docs), 3),
        "entry_format": "<statement> true_label is <status>",
        "index_sha256": C.sha256_file(index_path),
        "exact_overlap_with_eval_splits": leakage,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    C.save_json(CORPUS_DIR / f"{name}_meta.json", meta)
    C.log(f"已寫入 {index_path}（{len(docs):,} 條目）")


class RagIndex:
    """載入已建好的語料庫並提供檢索。供 05/06 使用。"""

    def __init__(self, name: str):
        faiss = C.import_faiss()
        self.name = name
        meta_path = CORPUS_DIR / f"{name}_meta.json"
        if not meta_path.exists():
            source = "train" if name == "noaug" else "train+aug"
            C.die(f"語料庫 {name} 尚未建立。請先執行："
                  f"python 03_build_corpus.py --source {source}")
        self.meta = C.load_json(meta_path)

        index_path = CORPUS_DIR / f"{name}.index"
        C.require_hash(index_path, self.meta["index_sha256"], f"語料庫 {name} 索引")
        self.index = faiss.read_index(str(index_path))

        payload = C.load_json(CORPUS_DIR / f"{name}_docs.json")
        self.docs = payload["docs"]
        self.doc_ids = payload["doc_ids"]
        self.model = _load_embedder()

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str], list[float]]:
        vec = _encode(self.model, [query])
        sims, idxs = self.index.search(vec, top_k)
        idxs = idxs[0].tolist()
        sims = [float(s) for s in sims[0]]
        return ([self.docs[i] for i in idxs],
                [self.doc_ids[i] for i in idxs],
                sims)


if __name__ == "__main__":
    main()
