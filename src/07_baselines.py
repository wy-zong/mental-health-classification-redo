"""傳統與微調基準模型，作為 LLM 管線的對照。

全部使用 01_prepare_data.py 切出的同一份 train / test，與 LLM 條件完全可比。

每個基準都跑兩版：**不含擴增**與**含擴增**。LLM 那邊的 S3/S4 用了擴增語料庫，
若基準只跑未擴增版，比較就不對等 —— 原始實驗沒有處理這個對稱性。

DistilBERT 微調這次跑在 GPU 上。前一輪之所以在 CPU 上跑了 2.84 小時，根本原因是
環境裝的是 `torch+cpu`（本機的 RTX 2070 完全沒被用到），不是腳本沒偵測 CUDA。

用法：
    python 07_baselines.py --run MAIN [--seeds 3] [--skip-distilbert]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_splits(run_dir: Path, with_aug: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = C.load_json(C.DATA_SPLITS / "split_audit.json")
    files = audit.get("split_files", {})

    def load(name: str) -> pd.DataFrame:
        path = C.DATA_SPLITS / f"{name}.csv"
        if files.get(name, {}).get("sha256"):
            C.require_hash(path, files[name]["sha256"], f"{name} 切分檔")
        return pd.read_csv(path)

    train, test = load("train"), load("test")
    if with_aug:
        aug_path = run_dir / "augmented.csv"
        if not aug_path.exists():
            C.die(f"找不到擴增資料 {aug_path}，請先執行 04_augment.py")
        aug = pd.read_csv(aug_path)[["statement", "status", "label_id"]]
        train = pd.concat([train[["statement", "status", "label_id"]], aug],
                          ignore_index=True)
    return train, test


def eval_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def run_sklearn(kind: str, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
        Xtr = vec.fit_transform(train["statement"])
        fit_feat = time.perf_counter() - t0
        t1 = time.perf_counter()
        Xte = vec.transform(test["statement"])
        feat_infer = time.perf_counter() - t1
    else:  # minilm
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.max_seq_length = 512
        enc = lambda xs: model.encode(list(xs), batch_size=64, convert_to_numpy=True,  # noqa: E731
                                      normalize_embeddings=True, show_progress_bar=False)
        Xtr = enc(train["statement"])
        fit_feat = time.perf_counter() - t0
        t1 = time.perf_counter()
        Xte = enc(test["statement"])
        feat_infer = time.perf_counter() - t1

    t2 = time.perf_counter()
    clf = LogisticRegression(max_iter=2000, n_jobs=1, random_state=seed)
    clf.fit(Xtr, train["label_id"])
    fit_time = time.perf_counter() - t2

    t3 = time.perf_counter()
    pred = clf.predict(Xte)
    predict_time = time.perf_counter() - t3

    return {
        **eval_preds(test["label_id"].to_numpy(), pred),
        "train_seconds": round(fit_feat + fit_time, 2),
        "inference_seconds": round(feat_infer + predict_time, 2),
        "seed": seed,
    }


def run_distilbert(train: pd.DataFrame, test: pd.DataFrame, seed: int,
                   epochs: int, batch_size: int, max_len: int) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(C.LABELS)).to(device)

    def encode(df: pd.DataFrame) -> TensorDataset:
        enc = tok(df["statement"].tolist(), truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(df["label_id"].to_numpy()))

    train_loader = DataLoader(encode(train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(encode(test), batch_size=batch_size * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        for step, (ids, mask, y) in enumerate(train_loader):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            opt.step()
        C.log(f"      epoch {ep + 1}/{epochs} 完成")
    train_seconds = time.perf_counter() - t0

    model.eval()
    preds = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for ids, mask, _y in test_loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds.append(logits.argmax(-1).cpu().numpy())
    infer_seconds = time.perf_counter() - t1
    pred = np.concatenate(preds)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        **eval_preds(test["label_id"].to_numpy(), pred),
        "train_seconds": round(train_seconds, 1),
        "inference_seconds": round(infer_seconds, 1),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_len,
        "seed": seed,
    }


def summarize(runs: list[dict]) -> dict:
    accs = [r["accuracy"] for r in runs]
    f1s = [r["macro_f1"] for r in runs]
    return {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_sd": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_sd": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-distilbert", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--variants", default="noaug,aug",
                    help="要跑的資料版本，逗號分隔")
    args = ap.parse_args()

    C.single_instance("baselines")

    run_dir = C.RUNS / args.run
    out_path = run_dir / "baselines.json"
    results = C.load_json(out_path) or {"run": args.run, "variants": {}}

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        with_aug = variant == "aug"
        try:
            train, test = load_splits(run_dir, with_aug)
        except SystemExit:
            if with_aug:
                C.log(f"!! 跳過 {variant}（擴增資料尚未產生）")
                continue
            raise

        C.log(f"=== 資料版本 {variant}：train {len(train):,} / test {len(test):,} ===")
        bucket = results["variants"].setdefault(variant, {})
        bucket["train_rows"] = int(len(train))
        bucket["test_rows"] = int(len(test))

        for kind in ["tfidf", "minilm"]:
            if kind in bucket:
                C.log(f"  {kind}: 已有結果，跳過")
                continue
            runs = []
            for s in range(args.seeds):
                r = run_sklearn(kind, train, test, seed=42 + s)
                runs.append(r)
                C.log(f"  {kind} seed={42 + s}: acc={r['accuracy']:.4f} "
                      f"f1={r['macro_f1']:.4f} train={r['train_seconds']}s")
            bucket[kind] = summarize(runs)
            C.save_json(out_path, results)     # 每完成一項就落地，可中斷續跑

        if not args.skip_distilbert and "distilbert" not in bucket:
            runs = []
            for s in range(args.seeds):
                C.log(f"  distilbert seed={42 + s} 訓練中 …")
                r = run_distilbert(train, test, 42 + s, args.epochs,
                                   args.batch_size, args.max_length)
                runs.append(r)
                C.log(f"    acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} "
                      f"train={r['train_seconds']}s on {r['device']}")
            bucket["distilbert"] = summarize(runs)
            C.save_json(out_path, results)

    C.save_json(out_path, results)
    C.log(f"已寫入 {out_path}")


if __name__ == "__main__":
    main()
