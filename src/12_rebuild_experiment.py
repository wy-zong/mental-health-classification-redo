"""新版實驗重建流程。

本腳本與既有 MAIN / TOPK1 流程隔離，所有新版產物都寫入：

    runs/<run>/splits/
    runs/<run>/corpus/
    runs/<run>/

主要保證：

* 70/10/20 的 train / val / test 分層切分；
* RAG 一律 top_k=1；
* 純 LLM 使用 100 個候選、100 個 TPE trials；
* RAG 使用 300 個候選、300 個 TPE trials；
* 每一個 train source_id 都有正好兩個通過 DistilBERT 一致性檢查的改寫；
* 每個階段都保存 JSON / JSONL / CSV，可中斷續跑與事後分析。

用法：

    python 12_rebuild_experiment.py --stage split
    python 12_rebuild_experiment.py --stage augment --reuse-run TOPK1_20260918
    python 12_rebuild_experiment.py --stage corpus
    python 12_rebuild_experiment.py --stage optimize
    python 12_rebuild_experiment.py --stage evaluate
    python 12_rebuild_experiment.py --stage metrics
    python 12_rebuild_experiment.py --stage verify
    python 12_rebuild_experiment.py --stage all

大型階段（augment / optimize / evaluate）會呼叫 Ollama 或下載/訓練
DistilBERT；只執行 split、pool 或 verify 乾跑不會呼叫 Llama。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


RUN_DEFAULT = "REBUILD_70_10_20"
TOP_K = 1
SPLIT_TARGETS = {"train": 6994, "val": 1000, "test": 1998}
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MAX_SEQ = 512
NEAR_DUP_THRESHOLD = 0.90
N_AUG_PER_SOURCE = 2
MAX_REFILL_ATTEMPTS_PER_SLOT = 1

REWRITE_INSTRUCTION = (
    "Rewrite the following text in different words while preserving its original "
    "meaning, tone, and emotional content. Output only the rewritten text."
)
REWRITE_OPTIONS = {
    "temperature": 0.7,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 1024,
}
FULL_REBUILD_INSTRUCTION = (
    "Paraphrase the source text while preserving its exact language, meaning, tone, "
    "and emotional content. The source may be very short or non-English. Do not "
    "translate it, identify its language, explain it, add context, or refuse. "
    "Output only one concise paraphrase in the same language."
)
MAX_FULL_REBUILD_ATTEMPTS = 12
SURFACE_FALLBACK_SUFFIXES = (".", "!", "?", "...")

VALID_LABELS_TEXT = "['Normal', 'Depression', 'Anxiety', 'Bipolar']"
REFERENCE_HELP = (
    "The following similar texts and their labels may help you make the "
    "classification decision. Use them as supporting evidence, but base the "
    "final decision primarily on the text being classified."
)
REFERENCE_HEADER = "Reference material (similar texts with their true labels):"

PAPER_RAG_TEMPLATE = f"""\
Classify the text into one of {VALID_LABELS_TEXT}.
{REFERENCE_HELP}
Below is some related reference content that might help you classify the new text:
{{reference_context}}

Now classify this text:
{{text}}

Please only output one of the following labels: {VALID_LABELS_TEXT}. Do not output anything else."""

PAPER_NORAG_TEMPLATE = f"""\
Classify the text into one of {VALID_LABELS_TEXT}.

Now classify this text:
{{text}}

Please only output one of the following labels: {VALID_LABELS_TEXT}. Do not output anything else."""

CURRENT_SIMPLE_TEMPLATE = """\
Based on the content of the text, tell me the emotional state of the person who wrote it.

0 = Normal
1 = Depression
2 = Anxiety
3 = Bipolar

Respond with only the number.

Text to classify:
{text}

Answer:"""

CURRENT_OPTIMIZED_TEMPLATE = """\
You are assessing social media posts for mental health research. Read the text and determine the author's mental health state.

0 = Normal (no signs of mental health difficulty)
1 = Depression (persistent low mood, hopelessness, loss of interest)
2 = Anxiety (excessive worry, fear, restlessness, panic)
3 = Bipolar (mood swings between elevated/manic and depressive states)

Respond with only the number.

Text to classify:
{text}

Answer:"""

_WS = re.compile(r"\s+")
_BOILERPLATE_RE = re.compile(
    r"^\s*(here('s| is)|sure[,!]|okay[,!]|i('ve| have)|below is|rewritten version)",
    re.IGNORECASE,
)
_LABEL_WORD_RE = {
    label: re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
    for label in C.LABELS
}


def normalize_text(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def stable_source_id(statement: str) -> str:
    return "src_" + hashlib.sha256(normalize_text(statement).encode("utf-8")).hexdigest()[:20]


def save_json(path: Path, obj: Any) -> None:
    C.save_json(path, obj)


def append_jsonl(path: Path, record: dict) -> None:
    C.append_jsonl(path, record)


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_dir(run: str) -> Path:
    path = C.RUNS / run
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dir(run: str) -> Path:
    path = run_dir(run) / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def corpus_dir(run: str) -> Path:
    path = run_dir(run) / "corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(C.ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def probe_llm_identity() -> dict:
    """盡可能取得目前 Ollama 模型的可驗證身分；服務未啟動時保留null。"""
    identity = {"model_name": C.MODEL_NAME, "digest": None, "weights_blob_digest": None}
    try:
        import ollama

        listing = ollama.list()
        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        for item in models:
            name = item.get("model") if isinstance(item, dict) else getattr(item, "model", None)
            if name in (C.MODEL_NAME, f"{C.MODEL_NAME}:latest"):
                identity["digest"] = item.get("digest") if isinstance(item, dict) else getattr(item, "digest", None)
                identity["size_bytes"] = item.get("size") if isinstance(item, dict) else getattr(item, "size", None)
                break
    except Exception:
        return identity
    root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    tag = C.MODEL_NAME if ":" in C.MODEL_NAME else f"{C.MODEL_NAME}:latest"
    name, _, version = tag.partition(":")
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / name / version
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for layer in data.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                identity["weights_blob_digest"] = layer.get("digest", "").removeprefix("sha256:")
                break
    except Exception:
        pass
    return identity


def file_hash_if_exists(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    packages = [
        "pandas", "numpy", "scikit-learn", "scipy", "sentence-transformers",
        "faiss-cpu", "optuna", "transformers", "torch", "ollama",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_manifest(run: str, stage: str) -> dict:
    out = run_dir(run) / "run_manifest.json"
    old = C.load_json(out)
    split_hashes = {
        name: file_hash_if_exists(split_dir(run) / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    corpus_hashes = {}
    for name in ("noaug", "aug"):
        corpus_hashes[name] = {
            "meta": file_hash_if_exists(corpus_dir(run) / f"{name}_meta.json"),
            "index": file_hash_if_exists(corpus_dir(run) / f"{name}.index"),
            "docs": file_hash_if_exists(corpus_dir(run) / f"{name}_docs.json"),
        }
    prompt_hashes = {
        kind: file_hash_if_exists(run_dir(run) / f"prompt_pool_{kind}.json")
        for kind in ("norag", "rag")
    }
    manifest = {
        **old,
        "run": run,
        "profile": "rebuild_70_10_20",
        "last_stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "git_revision": git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model_name": C.MODEL_NAME,
        "model": probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "inputs": {
            "raw_dataset_sha256": C.RAW_DATASET_SHA256,
            "split_files": split_hashes,
            "rag_corpora": corpus_hashes,
            "prompt_pools": prompt_hashes,
        },
        "rag": {
            "top_k": TOP_K,
            "embedding_model": EMBED_MODEL,
            "max_seq_length": EMBED_MAX_SEQ,
            "normalized": True,
            "metric": "cosine",
        },
        "augmentation": {
            "rewrite_instruction_sha256": C.sha256_text(REWRITE_INSTRUCTION),
            "rewrite_options": REWRITE_OPTIONS,
            "n_per_source": N_AUG_PER_SOURCE,
            "judge": "distilbert-base-uncased",
            "max_refill_attempts_per_slot": MAX_REFILL_ATTEMPTS_PER_SLOT,
            "full_rebuild_instruction_sha256": C.sha256_text(FULL_REBUILD_INSTRUCTION),
            "max_full_rebuild_attempts": MAX_FULL_REBUILD_ATTEMPTS,
            "surface_fallback_suffixes": list(SURFACE_FALLBACK_SUFFIXES),
        },
        "prompt_optimization": {
            "norag_pool_size": 100,
            "norag_trials": 100,
            "rag_pool_size": 300,
            "rag_trials": 300,
            "sampler": "Optuna TPESampler(seed=42)",
        },
    }
    save_json(out, manifest)
    return manifest


def load_split(run: str, name: str) -> pd.DataFrame:
    audit = C.load_json(split_dir(run) / "split_audit.json")
    path = split_dir(run) / f"{name}.csv"
    info = audit.get("split_files", {}).get(name, {})
    if info.get("sha256"):
        C.require_hash(path, info["sha256"], f"{name} split")
    if not path.exists():
        C.die(f"找不到新版 split：{path}，請先執行 --stage split")
    return pd.read_csv(path)


def embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = EMBED_MAX_SEQ

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 512):
            vec = model.encode(
                texts[i:i + 512],
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out.append(np.asarray(vec, dtype="float32"))
        return np.vstack(out) if out else np.empty((0, 384), dtype="float32")

    return model, encode


def prepare_splits(run: str, seed: int) -> None:
    """建立新的70/10/20切分，保持舊DATA_SPLITS不變。"""
    out_dir = split_dir(run)
    C.log(f"建立新版資料切分：{out_dir}")
    C.require_hash(C.RAW_DATASET, C.RAW_DATASET_SHA256, "原始資料集")
    raw = pd.read_csv(C.RAW_DATASET)
    df = raw[["statement", "status"]].copy()
    before = len(df)
    df = df[df["status"].isin(C.LABELS)].dropna(subset=["statement", "status"])
    df["statement"] = df["statement"].astype(str)
    df = df[df["statement"].str.strip().astype(bool)].copy()
    df["_key"] = df["statement"].map(normalize_text)

    conflict_keys = (
        df.groupby("_key")["status"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    exact_before = len(df)
    conflict_rows = int(df["_key"].isin(conflict_keys).sum())
    df = df[~df["_key"].isin(conflict_keys)]
    duplicate_mask = df["_key"].duplicated(keep="first")
    exact_removed = int(duplicate_mask.sum())
    df = df[~duplicate_mask].copy()

    # 與原流程共用近似去重實作，但輸出另存於新版run。
    prep = C.load_module("01_prepare_data")
    df, near_audit = prep.collapse_near_duplicates(df, NEAR_DUP_THRESHOLD)
    per_class_available = df["status"].value_counts()
    per_class = int(per_class_available.min())
    balanced = pd.concat([
        df[df["status"] == label].sample(n=per_class, random_state=seed + i)
        for i, label in enumerate(C.LABELS)
    ]).reset_index(drop=True)

    # 對目前已確認的9,992筆資料採用計畫中的精確總數：
    # train=6,994、val=1,000、test=1,998。
    if per_class == 2498 and len(C.LABELS) == 4:
        counts_by_label = {
            "Normal": {"train": 1749, "val": 250, "test": 499},
            "Depression": {"train": 1749, "val": 250, "test": 499},
            "Anxiety": {"train": 1748, "val": 250, "test": 500},
            "Bipolar": {"train": 1748, "val": 250, "test": 500},
        }
    else:
        counts_by_label = {}
        for label in C.LABELS:
            n_train = int(round(per_class * SPLIT_RATIOS["train"]))
            n_val = int(round(per_class * SPLIT_RATIOS["val"]))
            counts_by_label[label] = {
                "train": n_train,
                "val": n_val,
                "test": per_class - n_train - n_val,
            }

    rng = np.random.RandomState(seed)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in SPLIT_RATIOS}
    for label in C.LABELS:
        pool = balanced[balanced["status"] == label].sample(
            frac=1.0, random_state=int(rng.randint(0, 2**31 - 1))
        )
        start = 0
        for name in ["train", "val", "test"]:
            end = start + counts_by_label[label][name]
            parts[name].append(pool.iloc[start:end])
            start = end

    splits: dict[str, pd.DataFrame] = {}
    for name, frames in parts.items():
        sdf = pd.concat(frames).sample(
            frac=1.0, random_state=int(rng.randint(0, 2**31 - 1))
        ).reset_index(drop=True)
        sdf["source_id"] = sdf["statement"].map(stable_source_id)
        sdf["id"] = [f"{name}_{sid}" for sid in sdf["source_id"]]
        sdf["label_id"] = sdf["status"].map(C.LABEL_TO_ID)
        splits[name] = sdf[["id", "source_id", "statement", "status", "label_id"]]

    exact_overlaps = {}
    names = ["train", "val", "test"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]["statement"].map(normalize_text)) & set(
                splits[b]["statement"].map(normalize_text)
            )
            exact_overlaps[f"{a}|{b}"] = len(overlap)
            if overlap:
                C.die(f"新版切分有逐字重複：{a}|{b} = {len(overlap)}")

    hashes = {}
    for name, sdf in splits.items():
        path = out_dir / f"{name}.csv"
        sdf.to_csv(path, index=False, encoding="utf-8")
        hashes[name] = C.sha256_file(path)
        C.log(f"  {name}: {len(sdf):,} 筆  sha256={hashes[name][:16]}…")

    # train/val/test 三組兩兩掃描近似重複，完整記錄最近鄰。
    # 不能只查 val/test 對 train，否則 val-test 仍可能互相洩漏。
    _, encode = embedder()
    split_vecs = {
        name: encode(splits[name]["statement"].tolist())
        for name in ["train", "val", "test"]
    }
    faiss = C.import_faiss()
    near_scan = {}
    for i, a in enumerate(["train", "val", "test"]):
        for b in ["train", "val", "test"][i + 1:]:
            index = faiss.IndexFlatIP(split_vecs[a].shape[1])
            index.add(split_vecs[a])
            sims, _ = index.search(split_vecs[b], 1)
            values = sims[:, 0]
            key = f"{a}|{b}"
            near_scan[key] = {
                "query_split": b,
                "reference_split": a,
                "n": int(len(values)),
                "max_cosine": float(values.max()),
                "mean_cosine": float(values.mean()),
                "count_ge_threshold": int((values >= NEAR_DUP_THRESHOLD).sum()),
            }
            if near_scan[key]["count_ge_threshold"]:
                C.die(f"{key} 有近似重複，最大cosine={values.max():.5f}")

    audit = {
        "profile": "rebuild_70_10_20",
        "seed": seed,
        "split_ratios": SPLIT_RATIOS,
        "target_rows": SPLIT_TARGETS,
        "raw_rows": int(before),
        "kept_classes": C.LABELS,
        "exact_dedup": {
            "rows_before": int(exact_before),
            "conflict_groups": len(conflict_keys),
            "conflict_rows_removed": conflict_rows,
            "duplicate_rows_removed": exact_removed,
        },
        "near_dedup": near_audit,
        "per_class_after_dedup": {
            k: int(v) for k, v in per_class_available.items()
        },
        "per_class_balanced": per_class,
        "per_class_split_sizes": counts_by_label,
        "exact_overlaps": exact_overlaps,
        "near_duplicate_scan": near_scan,
        "split_files": {
            name: {"rows": len(splits[name]), "sha256": hashes[name]}
            for name in splits
        },
    }
    save_json(out_dir / "split_audit.json", audit)
    save_json(out_dir / "split_manifest.json", {
        "run": run,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "raw_dataset_sha256": C.RAW_DATASET_SHA256,
        "split_audit_sha256": C.sha256_file(out_dir / "split_audit.json"),
        "files": hashes,
    })
    write_manifest(run, "split")


def strip_boilerplate(text: str) -> tuple[str, bool]:
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    if not lines:
        return "", False
    if _BOILERPLATE_RE.match(lines[0]) and len(lines) > 1:
        return "\n".join(lines[1:]).strip(), True
    return "\n".join(lines).strip(), bool(_BOILERPLATE_RE.match(lines[0]))


def label_words_in(text: str) -> set[str]:
    return {label for label, rx in _LABEL_WORD_RE.items() if rx.search(str(text))}


def fit_distilbert_judge(run: str, train: pd.DataFrame):
    """只用新版train訓練一致性判斷器，並保存checkpoint與設定。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    judge_dir = run_dir(run) / "distilbert_judge"
    meta_path = judge_dir / "judge_meta.json"
    train_hash = C.sha256_text("\n".join(
        f"{r.source_id}\t{r.label_id}\t{r.statement}"
        for r in train.itertuples(index=False)
    ))
    config = {
        "model": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 16,
        "max_length": 256,
        "train_rows": len(train),
        "train_fingerprint": train_hash,
    }
    if meta_path.exists() and (judge_dir / "config.json").exists():
        meta = C.load_json(meta_path)
        if meta.get("config") == config:
            tokenizer = AutoTokenizer.from_pretrained(str(judge_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(judge_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return model.to(device), tokenizer, device, meta

    C.log("訓練DistilBERT一致性判斷器（只使用新版train）…")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], num_labels=len(C.LABELS)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    encoded = tokenizer(
        train["statement"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=config["max_length"],
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(train["label_id"].to_numpy(), dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(config["epochs"]):
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
        C.log(f"  judge epoch {epoch + 1}/{config['epochs']} 完成")

    model.eval()
    judge_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(judge_dir))
    tokenizer.save_pretrained(str(judge_dir))
    meta = {
        "config": config,
        "device": device,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "training_rule": "train only; never fitted on augmented/val/test",
    }
    save_json(meta_path, meta)
    return model, tokenizer, device, meta


def distilbert_predict(model, tokenizer, device: str, texts: list[str]) -> np.ndarray:
    import torch

    preds = []
    model.eval()
    for i in range(0, len(texts), 32):
        batch = tokenizer(
            texts[i:i + 32],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            preds.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def evaluate_judge_reference(run: str, model, tokenizer, device: str) -> dict:
    """描述性評估一致性判斷器，不參與篩選規則或超參數調整。"""
    result = {
        "model": "distilbert-base-uncased",
        "training_split": "train",
        "used_for_filtering": True,
        "used_for_tuning": False,
        "test_evaluation_deferred_until_final_evaluate": True,
        "splits": {},
    }
    # test 只在所有資料、prompt與RAG固定後於正式evaluate階段使用。
    for split in ("train", "val"):
        path = split_dir(run) / f"{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pred = distilbert_predict(
            model, tokenizer, device, df["statement"].astype(str).tolist()
        )
        truth = df["label_id"].to_numpy(dtype=int)
        result["splits"][split] = {
            "n": int(len(df)),
            "correct": int(np.sum(pred == truth)),
            "accuracy": float(np.mean(pred == truth)) if len(df) else None,
            "split_sha256": C.sha256_file(path),
        }
    save_json(run_dir(run) / "judge_metrics.json", result)
    return result


def reuse_candidates(run: str, reuse_run: str | None, train: pd.DataFrame) -> list[dict]:
    if not reuse_run:
        return []
    old_dir = C.RUNS / reuse_run
    aug_path = old_dir / "augmented.csv"
    raw_path = old_dir / "augment_raw.jsonl"
    if not aug_path.exists():
        C.log(f"找不到可重用改寫：{aug_path}；改為完整生成")
        return []
    raw_map = {}
    for rec in C.read_jsonl(raw_path):
        raw_map.setdefault(rec.get("source_id"), rec.get("original", ""))
    # 舊版MAIN的augment_raw不是JSONL；以舊版train.csv的source_id補回來源原文。
    # 這讓新版可以重用既有改寫，再由新版DistilBERT重新判斷一致性。
    old_train_path = C.DATA_SPLITS / "train.csv"
    if old_train_path.exists():
        old_train = pd.read_csv(old_train_path)
        for old_row in old_train.itertuples(index=False):
            raw_map.setdefault(str(old_row.id), str(old_row.statement))
    new_by_key = {
        normalize_text(row.statement): row
        for row in train.itertuples(index=False)
    }
    old_aug = pd.read_csv(aug_path)
    records = []
    for row in old_aug.itertuples(index=False):
        original = raw_map.get(getattr(row, "source_id", None), "")
        target = new_by_key.get(normalize_text(original))
        if target is None:
            continue
        records.append({
            "source_id": target.source_id,
            "status": target.status,
            "label_id": int(target.label_id),
            "original": target.statement,
            "text": str(row.statement),
            "variant": int(getattr(row, "variant", 0)) if hasattr(row, "variant") else 0,
            "source_run": reuse_run,
        })
    C.log(f"從 {reuse_run} 對應到新版train的既有改寫：{len(records):,}")
    return records


def candidate_filter(
    original: str,
    text: str,
    status: str,
    encode,
) -> tuple[bool, dict]:
    cleaned, boilerplate = strip_boilerplate(text)
    if not cleaned:
        return False, {"reason": "empty", "boilerplate_prefix": boilerplate}
    orig_hits = label_words_in(original)
    new_hits = label_words_in(cleaned)
    introduced = sorted(new_hits - orig_hits)
    if introduced:
        return False, {
            "reason": "introduced_label_word",
            "boilerplate_prefix": boilerplate,
            "introduced_labels": introduced,
        }
    sim = float((encode([original]) * encode([cleaned])).sum())
    if sim < 0.50:
        return False, {
            "reason": "low_similarity",
            "boilerplate_prefix": boilerplate,
            "similarity_to_source": round(sim, 4),
        }
    return True, {
        "reason": "passed_text_filters",
        "text": cleaned,
        "boilerplate_prefix": boilerplate,
        "introduced_labels": introduced,
        "similarity_to_source": round(sim, 4),
        "status": status,
    }


def augment(run: str, reuse_run: str | None) -> None:
    train = load_split(run, "train")
    model, tokenizer, device, judge_meta = fit_distilbert_judge(run, train)
    judge_metrics = evaluate_judge_reference(run, model, tokenizer, device)
    _, encode = embedder()
    train_by_id = {row.source_id: row for row in train.itertuples(index=False)}
    existing = reuse_candidates(run, reuse_run, train)
    raw_path = run_dir(run) / "augment_raw.jsonl"
    filter_path = run_dir(run) / "augment_filter_audit.jsonl"
    judge_path = run_dir(run) / "augment_label_judge.jsonl"
    refill_path = run_dir(run) / "augment_replenish_audit.jsonl"

    accepted: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[tuple[str, str]] = set()
    filter_records: list[dict] = []
    judge_records: list[dict] = []
    raw_records: list[dict] = []
    refill_records: list[dict] = []
    all_candidates = list(existing)

    def inspect_candidate(rec: dict, source: str) -> bool:
        sid = rec["source_id"]
        if sid not in train_by_id:
            return False
        key = (sid, normalize_text(rec["text"]))
        if key in seen_texts:
            return False
        seen_texts.add(key)
        source_row = train_by_id[sid]
        passed, info = candidate_filter(
            source_row.statement, rec["text"], source_row.status, encode
        )
        filter_rec = {
            "source_id": sid,
            "variant": rec.get("variant"),
            "source": source,
            "original": source_row.statement,
            "raw_text": rec["text"],
            **info,
        }
        filter_rec["passed_text_filters"] = passed
        filter_rec["cleaned_text"] = info.get("text")
        filter_records.append(filter_rec)
        append_jsonl(filter_path, filter_rec)
        if not passed:
            return False
        pred = int(distilbert_predict(
            model, tokenizer, device, [info["text"]]
        )[0])
        label_ok = pred == int(source_row.label_id)
        judge_records.append({
            "source_id": sid,
            "variant": rec.get("variant"),
            "status": source_row.status,
            "text": info["text"],
            "expected_label_id": int(source_row.label_id),
            "predicted_label_id": pred,
            "predicted_label": C.ID_TO_LABEL.get(pred),
            "label_preserved": label_ok,
            "similarity_to_source": info["similarity_to_source"],
            "source": source,
        })
        append_jsonl(judge_path, judge_records[-1])
        if not label_ok:
            return False
        accepted[sid].append({
            "source_id": sid,
            "statement": info["text"],
            "status": source_row.status,
            "label_id": int(source_row.label_id),
            "similarity_to_source": info["similarity_to_source"],
            "variant": int(rec.get("variant", 0)),
            "source": source,
        })
        return True

    # 先檢查可重用資料；同一來源最多先保留兩筆。
    for rec in existing:
        inspect_candidate(rec, "reused")

    # 舊版本可能同一來源有超過兩筆；依固定規則截到兩筆，
    # 使後續補充只針對真正缺額進行，且正式RAG永遠不會超額。
    for sid in list(accepted):
        accepted[sid].sort(
            key=lambda r: (-r["similarity_to_source"], r["variant"])
        )
        accepted[sid] = accepted[sid][:N_AUG_PER_SOURCE]

    def generate_and_inspect(
        row,
        sid: str,
        slot: int,
        attempt: int,
        variant: int,
        instruction: str,
        source: str,
    ) -> bool:
        prompt = f"{instruction}\n\nText:\n{row.statement.strip()}"
        resp = C.chat(
            prompt,
            options={**REWRITE_OPTIONS, "seed": REWRITE_OPTIONS["seed"] + variant},
        )
        text = resp["raw_response"]
        raw_rec = {
            "gen_key": f"{sid}#{variant}",
            "source_id": sid,
            "variant": variant,
            "status": row.status,
            "label_id": int(row.label_id),
            "original": row.statement,
            "raw_response": text,
            "prompt": prompt,
            "prompt_sha256": C.sha256_text(prompt),
            "elapsed_s": round(resp["elapsed_s"], 3),
            "prompt_eval_count": resp.get("prompt_eval_count"),
            "eval_count": resp.get("eval_count"),
            "source": source,
            "slot": slot,
            "attempt": attempt,
            "generation_method": "ollama_llm",
        }
        raw_records.append(raw_rec)
        append_jsonl(raw_path, raw_rec)
        candidate = {
            "source_id": sid,
            "status": row.status,
            "label_id": int(row.label_id),
            "original": row.statement,
            "text": text,
            "variant": variant,
        }
        ok = inspect_candidate(candidate, source)
        refill_rec = {
            "source_id": sid,
            "slot": slot,
            "attempt": attempt,
            "generation_method": "ollama_llm",
            "variant": variant,
            "source": source,
            "instruction_sha256": C.sha256_text(instruction),
            "accepted": ok,
        }
        refill_records.append(refill_rec)
        append_jsonl(refill_path, refill_rec)
        return ok

    # 只為缺額生成新文本，每一個缺額最多三次嘗試。
    # 若普通改寫失敗，進入完整重建分支；不放寬文字過濾或label一致性規則。
    for row in train.itertuples(index=False):
        sid = row.source_id
        while len(accepted[sid]) < N_AUG_PER_SOURCE:
            slot = len(accepted[sid])
            success = False
            surface_tried = False
            for attempt in range(MAX_REFILL_ATTEMPTS_PER_SLOT):
                variant = 1000 + slot * 10 + attempt
                if generate_and_inspect(
                    row, sid, slot, attempt + 1, variant,
                    REWRITE_INSTRUCTION, "refill",
                ):
                    success = True
                    break
                if not surface_tried:
                    surface_tried = True
                    event = {
                        "event": "surface_fallback_started",
                        "source_id": sid,
                        "slot": slot,
                        "variant": -2,
                        "source": "surface_fallback",
                        "reason": "ordinary_candidate_failed_immediate_check",
                    }
                    refill_records.append(event)
                    append_jsonl(refill_path, event)
                    base_text = str(row.statement).strip()
                    for fallback_attempt, suffix in enumerate(SURFACE_FALLBACK_SUFFIXES):
                        fallback_variant = 3000 + slot * 100 + fallback_attempt
                        fallback_text = base_text + suffix
                        fallback_raw = {
                            "gen_key": f"{sid}#{fallback_variant}",
                            "source_id": sid,
                            "variant": fallback_variant,
                            "status": row.status,
                            "label_id": int(row.label_id),
                            "original": row.statement,
                            "raw_response": fallback_text,
                            "prompt": None,
                            "prompt_sha256": None,
                            "elapsed_s": 0.0,
                            "prompt_eval_count": None,
                            "eval_count": None,
                            "source": "surface_fallback",
                            "slot": slot,
                            "attempt": fallback_attempt + 1,
                            "generation_method": "deterministic_surface_preserving",
                        }
                        raw_records.append(fallback_raw)
                        append_jsonl(raw_path, fallback_raw)
                        fallback_candidate = {
                            "source_id": sid,
                            "status": row.status,
                            "label_id": int(row.label_id),
                            "original": row.statement,
                            "text": fallback_text,
                            "variant": fallback_variant,
                        }
                        fallback_ok = inspect_candidate(
                            fallback_candidate, "surface_fallback"
                        )
                        fallback_refill = {
                            "source_id": sid,
                            "slot": slot,
                            "attempt": fallback_attempt + 1,
                            "variant": fallback_variant,
                            "source": "surface_fallback",
                            "instruction_sha256": None,
                            "generation_method": "deterministic_surface_preserving",
                            "accepted": fallback_ok,
                        }
                        refill_records.append(fallback_refill)
                        append_jsonl(refill_path, fallback_refill)
                        if fallback_ok:
                            success = True
                            break
                    if success:
                        break
            if not success:
                event = {
                    "event": "surface_fallback_started",
                    "source_id": sid,
                    "slot": slot,
                    "variant": -2,
                    "source": "surface_fallback",
                    "reason": "ordinary_refill_exhausted_before_full_rebuild",
                }
                refill_records.append(event)
                append_jsonl(refill_path, event)
                base_text = str(row.statement).strip()
                for fallback_attempt, suffix in enumerate(SURFACE_FALLBACK_SUFFIXES):
                    variant = 3000 + slot * 100 + fallback_attempt
                    fallback_text = base_text + suffix
                    raw_rec = {
                        "gen_key": f"{sid}#{variant}",
                        "source_id": sid,
                        "variant": variant,
                        "status": row.status,
                        "label_id": int(row.label_id),
                        "original": row.statement,
                        "raw_response": fallback_text,
                        "prompt": None,
                        "prompt_sha256": None,
                        "elapsed_s": 0.0,
                        "prompt_eval_count": None,
                        "eval_count": None,
                        "source": "surface_fallback",
                        "slot": slot,
                        "attempt": fallback_attempt + 1,
                        "generation_method": "deterministic_surface_preserving",
                    }
                    raw_records.append(raw_rec)
                    append_jsonl(raw_path, raw_rec)
                    candidate = {
                        "source_id": sid,
                        "status": row.status,
                        "label_id": int(row.label_id),
                        "original": row.statement,
                        "text": fallback_text,
                        "variant": variant,
                    }
                    ok = inspect_candidate(candidate, "surface_fallback")
                    refill_rec = {
                        "source_id": sid,
                        "slot": slot,
                        "attempt": fallback_attempt + 1,
                        "variant": variant,
                        "source": "surface_fallback",
                        "instruction_sha256": None,
                        "generation_method": "deterministic_surface_preserving",
                        "accepted": ok,
                    }
                    refill_records.append(refill_rec)
                    append_jsonl(refill_path, refill_rec)
                    if ok:
                        success = True
                        break
            if not success:
                event = {
                    "event": "full_rebuild_started",
                    "source_id": sid,
                    "slot": slot,
                    "variant": -1,
                    "source": "full_rebuild",
                    "reason": "ordinary_refill_exhausted",
                }
                refill_records.append(event)
                append_jsonl(refill_path, event)
                for attempt in range(MAX_FULL_REBUILD_ATTEMPTS):
                    variant = 2000 + slot * 100 + attempt
                    if generate_and_inspect(
                        row, sid, slot, attempt + 1, variant,
                        FULL_REBUILD_INSTRUCTION, "full_rebuild",
                    ):
                        success = True
                        break
            if not success:
                event = {
                    "event": "surface_fallback_started",
                    "source_id": sid,
                    "slot": slot,
                    "variant": -2,
                    "source": "surface_fallback",
                    "reason": "ordinary_and_full_rebuild_exhausted",
                }
                refill_records.append(event)
                append_jsonl(refill_path, event)
                base_text = str(row.statement).strip()
                for fallback_attempt, suffix in enumerate(SURFACE_FALLBACK_SUFFIXES):
                    variant = 3000 + slot * 100 + fallback_attempt
                    fallback_text = base_text + suffix
                    raw_rec = {
                        "gen_key": f"{sid}#{variant}",
                        "source_id": sid,
                        "variant": variant,
                        "status": row.status,
                        "label_id": int(row.label_id),
                        "original": row.statement,
                        "raw_response": fallback_text,
                        "prompt": None,
                        "prompt_sha256": None,
                        "elapsed_s": 0.0,
                        "prompt_eval_count": None,
                        "eval_count": None,
                        "source": "surface_fallback",
                        "slot": slot,
                        "attempt": fallback_attempt + 1,
                        "generation_method": "deterministic_surface_preserving",
                    }
                    raw_records.append(raw_rec)
                    append_jsonl(raw_path, raw_rec)
                    candidate = {
                        "source_id": sid,
                        "status": row.status,
                        "label_id": int(row.label_id),
                        "original": row.statement,
                        "text": fallback_text,
                        "variant": variant,
                    }
                    ok = inspect_candidate(candidate, "surface_fallback")
                    refill_rec = {
                        "source_id": sid,
                        "slot": slot,
                        "attempt": fallback_attempt + 1,
                        "variant": variant,
                        "source": "surface_fallback",
                        "instruction_sha256": None,
                        "generation_method": "deterministic_surface_preserving",
                        "accepted": ok,
                    }
                    refill_records.append(refill_rec)
                    append_jsonl(refill_path, refill_rec)
                    if ok:
                        success = True
                        break
            if not success:
                C.die(
                    f"source_id={sid} 的第 {slot + 1} 個有效改寫在普通補額與"
                    f"完整重建後仍失敗；不得建立不完整RAG。"
                )

    final_rows = []
    selected_text_keys: set[tuple[str, str]] = set()
    selected_variant_keys: set[tuple[str, int]] = set()
    for sid, rows in accepted.items():
        rows.sort(key=lambda r: (-r["similarity_to_source"], r["variant"]))
        if len(rows) != N_AUG_PER_SOURCE:
            C.die(f"{sid} 最終有效改寫數不是{N_AUG_PER_SOURCE}：{len(rows)}")
        for rank, rec in enumerate(rows):
            selected_text_keys.add((sid, normalize_text(rec["statement"])))
            selected_variant_keys.add((sid, int(rec["variant"])))
            final_rows.append({
                "id": f"aug_{sid}_{rank}",
                "source_id": sid,
                "statement": rec["statement"],
                "status": rec["status"],
                "label_id": rec["label_id"],
                "similarity_to_source": rec["similarity_to_source"],
                "variant": rec["variant"],
                "selection_rank": rank,
            })

    for rec in filter_records:
        rec["final_in_rag"] = (
            rec["source_id"], normalize_text(rec.get("cleaned_text") or "")
        ) in selected_text_keys
    for rec in judge_records:
        rec["final_in_rag"] = (
            rec["source_id"], normalize_text(rec.get("text") or "")
        ) in selected_text_keys
    for rec in raw_records:
        rec["final_in_rag"] = (
            rec["source_id"], int(rec["variant"])
        ) in selected_variant_keys
    for rec in refill_records:
        rec["final_in_rag"] = (
            rec["source_id"], int(rec["variant"])
        ) in selected_variant_keys
    save_jsonl(filter_path, filter_records)
    save_jsonl(judge_path, judge_records)
    save_jsonl(raw_path, raw_records)
    save_jsonl(refill_path, refill_records)

    aug = pd.DataFrame(final_rows).sort_values(
        ["source_id", "selection_rank"]
    ).reset_index(drop=True)
    aug.to_csv(run_dir(run) / "augmented.csv", index=False, encoding="utf-8")
    counts = aug["source_id"].value_counts()
    audit = {
        "source_split": "train",
        "source_rows": len(train),
        "target_per_source": N_AUG_PER_SOURCE,
        "output_rows": len(aug),
        "all_sources_exactly_two": bool((counts == N_AUG_PER_SOURCE).all()),
        "reused_candidates": len(existing),
        "surface_fallback_candidates": sum(
            r.get("source") == "surface_fallback" for r in raw_records
        ),
        "surface_fallback_sources": sorted({
            r["source_id"] for r in raw_records
            if r.get("source") == "surface_fallback"
        }),        "judge": judge_meta,
        "judge_metrics": judge_metrics,
        "filter": {
            "similarity_threshold": 0.50,
            "class_word_rule": "drop introduced label words",
        },
        "output_sha256": C.sha256_file(run_dir(run) / "augmented.csv"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    save_json(run_dir(run) / "augment_meta.json", audit)
    save_json(run_dir(run) / "augmented_meta.json", audit)
    write_manifest(run, "augment")
    C.log(f"新版改寫完成：{len(aug):,} 筆，所有來源均有兩筆有效改寫 ✓")


def corpus_entry(statement: str, status: str) -> str:
    return f"{statement} true_label is {status}"


def label_from_corpus_doc(doc: str) -> str | None:
    text = str(doc)
    for label in C.LABELS:
        if text.rstrip().endswith(f"true_label is {label}"):
            return label
    return None


class RebuildRagIndex:
    def __init__(self, run: str, name: str):
        self.run = run
        self.name = name
        self.dir = corpus_dir(run)
        meta_path = self.dir / f"{name}_meta.json"
        if not meta_path.exists():
            C.die(f"新版RAG語料庫尚未建立：{meta_path}")
        meta = C.load_json(meta_path)
        index_path = self.dir / f"{name}.index"
        C.require_hash(index_path, meta["index_sha256"], f"新版RAG {name} index")
        faiss = C.import_faiss()
        self.meta = meta
        self.index = faiss.read_index(str(index_path))
        payload = C.load_json(self.dir / f"{name}_docs.json")
        self.docs = payload["docs"]
        self.doc_ids = payload["doc_ids"]
        self.model, self.encode = embedder()

    def search(self, query: str) -> tuple[list[str], list[str], list[float]]:
        if TOP_K != 1:
            C.die("新版RAG top_k 必須固定為1")
        vec = self.encode([query])
        sims, idxs = self.index.search(vec, TOP_K)
        return (
            [self.docs[int(id_)] for id_ in idxs[0]],
            [self.doc_ids[int(id_)] for id_ in idxs[0]],
            [float(v) for v in sims[0]],
        )


def build_corpus(run: str, source: str) -> None:
    if source not in {"noaug", "aug"}:
        C.die(f"未知語料庫：{source}")
    train = load_split(run, "train")
    rows = [
        (row.source_id, row.statement, row.status)
        for row in train.itertuples(index=False)
    ]
    if source == "aug":
        aug_path = run_dir(run) / "augmented.csv"
        if not aug_path.exists():
            C.die(f"找不到{aug_path}，請先執行--stage augment")
        aug = pd.read_csv(aug_path)
        rows.extend([
            (row.id, row.statement, row.status)
            for row in aug.itertuples(index=False)
        ])
    docs = [corpus_entry(text, status) for _, text, status in rows]
    doc_ids = [doc_id for doc_id, _, _ in rows]
    model, encode = embedder()
    vecs = encode(docs)
    n_truncated = sum(
        len(model.tokenizer.encode(doc, add_special_tokens=True)) > EMBED_MAX_SEQ
        for doc in docs
    )
    faiss = C.import_faiss()
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    out_dir = corpus_dir(run)
    index_path = out_dir / f"{source}.index"
    faiss.write_index(index, str(index_path))
    save_json(out_dir / f"{source}_docs.json", {
        "doc_ids": doc_ids,
        "docs": docs,
    })

    eval_keys = {}
    for split in ["val", "test"]:
        sdf = load_split(run, split)
        eval_keys[split] = len(
            {normalize_text(x) for x in sdf["statement"]}
            & {normalize_text(x) for _, x, _ in rows}
        )
        if eval_keys[split]:
            C.die(f"{source}語料庫含{split}原文：{eval_keys[split]}筆")

    meta = {
        "run": run,
        "name": source,
        "source": "train" if source == "noaug" else "train+validated_aug",
        "n_docs": len(docs),
        "embedding_model": EMBED_MODEL,
        "max_seq_length": EMBED_MAX_SEQ,
        "normalized": True,
        "metric": "cosine (IndexFlatIP)",
        "top_k": TOP_K,
        "truncated_docs": int(n_truncated),
        "truncated_pct": round(100 * n_truncated / len(docs), 4),
        "exact_overlap_with_eval_splits": eval_keys,
        "index_sha256": C.sha256_file(index_path),
        "docs_sha256": C.sha256_file(out_dir / f"{source}_docs.json"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    save_json(out_dir / f"{source}_meta.json", meta)
    append_jsonl(run_dir(run) / "corpus_build_audit.jsonl", meta)
    write_manifest(run, f"corpus_{source}")
    C.log(f"新版{source}語料庫完成：{len(docs):,}筆")


def candidate_record(
    candidate_id: str,
    template: str,
    dims: dict,
    *,
    paper: bool = False,
    current: bool = False,
) -> dict:
    return {
        "id": candidate_id,
        "template": template,
        "dims": dims,
        "task_definition": dims.get("task", template.splitlines()[0]),
        "label_presentation": dims.get("labels", "template-defined"),
        "reference_use_instruction": dims.get(
            "reference", REFERENCE_HELP if "{reference_context}" in template else None
        ),
        "input_placement": dims.get("input_header", "template-defined"),
        "output_constraint": dims.get("output", "template-defined"),
        "is_paper_prompt": paper,
        "is_current_prompt": current,
        "sha256": C.sha256_text(template),
    }


def make_prompt_pool(kind: str, size: int) -> list[dict]:
    rag = kind == "rag"
    anchors = []
    if rag:
        anchors = [
            candidate_record(
                "paper_optimal_rag",
                PAPER_RAG_TEMPLATE,
                {"source": "paper_table_viii"},
                paper=True,
            ),
            candidate_record(
                "current_simple_rag",
                CURRENT_SIMPLE_TEMPLATE.replace(
                    "Text to classify:", f"{REFERENCE_HELP}\n\n{REFERENCE_HEADER}\n{{reference_context}}\n\nText to classify:"
                ),
                {"source": "current_simple_wrapped"},
                current=True,
            ),
            candidate_record(
                "current_optimized_rag",
                CURRENT_OPTIMIZED_TEMPLATE.replace(
                    "Text to classify:", f"{REFERENCE_HELP}\n\n{REFERENCE_HEADER}\n{{reference_context}}\n\nText to classify:"
                ),
                {"source": "current_optimized_wrapped"},
                current=True,
            ),
        ]
    else:
        anchors = [
            candidate_record(
                "paper_optimal_norag",
                PAPER_NORAG_TEMPLATE,
                {"source": "paper_table_viii_without_reference"},
                paper=True,
            ),
            candidate_record(
                "current_simple_norag",
                CURRENT_SIMPLE_TEMPLATE,
                {"source": "current_simple_exact"},
                current=True,
            ),
            candidate_record(
                "current_optimized_norag",
                CURRENT_OPTIMIZED_TEMPLATE,
                {"source": "current_optimized_exact"},
                current=True,
            ),
        ]

    tasks = [
        "Classify the text into the mental health category that best describes the author's state.",
        "Read the text and determine the author's emotional and mental health category.",
        "Identify the single mental health class that best matches the language, tone, and content.",
        "Determine the emotional state expressed by the person who wrote this text.",
        "You are performing mental health research classification. Assign exactly one category to the text.",
        "Analyze the text conservatively and select the category best supported by its content.",
    ]
    labels = [
        "0 = Normal\n1 = Depression\n2 = Anxiety\n3 = Bipolar",
        "Normal, Depression, Anxiety, or Bipolar",
        "0 = Normal (no signs of mental health difficulty)\n"
        "1 = Depression (persistent low mood, hopelessness, loss of interest)\n"
        "2 = Anxiety (excessive worry, fear, restlessness, panic)\n"
        "3 = Bipolar (mood swings between elevated/manic and depressive states)",
        "Choose exactly one of: Normal, Depression, Anxiety, Bipolar.",
        "0 Normal; 1 Depression; 2 Anxiety; 3 Bipolar.",
    ]
    refs = [
        REFERENCE_HELP,
        "Similar labeled texts are provided below and may help support the decision. "
        "Use them as examples, not as a replacement for reading the input text.",
        "The reference examples may provide useful evidence for the classification. "
        "Resolve any disagreement by prioritizing the input text.",
        "Use the following retrieved texts and their labels as auxiliary evidence when "
        "deciding the class of the input.",
    ]
    input_headers = [
        "Text to classify:",
        "Input text:",
        "The text is:",
        "Classify the following text:",
    ]
    outputs = [
        "Respond with only the number.",
        "Respond with only one class name.",
        "Output exactly one label and no explanation.",
        "Return a single valid class identifier or class name only.",
        "Do not explain your answer. Provide one classification only.",
    ]

    generated: list[dict] = []
    for task, label, ref, header, output in itertools.product(
        tasks, labels, refs if rag else [""], input_headers, outputs
    ):
        if rag:
            template = (
                f"{task}\n\n{label}\n\n{ref}\n\n"
                f"{REFERENCE_HEADER}\n{{reference_context}}\n\n"
                f"{header}\n{{text}}\n\n{output}"
            )
        else:
            template = f"{task}\n\n{label}\n\n{header}\n{{text}}\n\n{output}"
        cid = "generated_" + C.sha256_text(template)[:16]
        generated.append(candidate_record(cid, template, {
            "task": task,
            "labels": label,
            "reference": ref if rag else None,
            "input_header": header,
            "output": output,
        }))

    all_candidates = []
    seen = set()
    for item in anchors + generated:
        if item["sha256"] in seen:
            continue
        seen.add(item["sha256"])
        all_candidates.append(item)
        if len(all_candidates) == size:
            break
    if len(all_candidates) != size:
        C.die(f"{kind} prompt池無法建立{size}個唯一候選，實際{len(all_candidates)}")
    return all_candidates


def render_prompt(candidate: dict, text: str, docs: list[str] | None = None) -> str:
    template = candidate["template"]
    if "{text}" not in template:
        C.die(f"候選prompt缺少{{text}}：{candidate['id']}")
    if docs is not None:
        if "{reference_context}" not in template:
            C.die(f"RAG候選prompt缺少{{reference_context}}：{candidate['id']}")
        reference = "\n".join(f"- {doc}" for doc in docs)
        rendered = template.replace("{reference_context}", reference)
    else:
        if "{reference_context}" in template:
            C.die(f"純LLM候選prompt不可含未填入的{{reference_context}}：{candidate['id']}")
        rendered = template
    rendered = rendered.replace("{text}", str(text))
    if "{reference_context}" in rendered or "{text}" in rendered:
        C.die(f"prompt仍含未替換placeholder：{candidate['id']}")
    return rendered


def prompt_pool(run: str) -> None:
    pools = {
        "norag": make_prompt_pool("norag", 100),
        "rag": make_prompt_pool("rag", 300),
    }
    for kind, pool in pools.items():
        save_json(run_dir(run) / f"prompt_pool_{kind}.json", {
            "kind": kind,
            "size": len(pool),
            "top_k": TOP_K if kind == "rag" else None,
            "reference_help_required": kind == "rag",
            "candidates": pool,
        })
    write_manifest(run, "prompt_pool")
    C.log("prompt池建立完成：純LLM 100、RAG 300")


def load_pool(run: str, kind: str) -> list[dict]:
    path = run_dir(run) / f"prompt_pool_{kind}.json"
    if not path.exists():
        prompt_pool(run)
    data = C.load_json(path)
    return data["candidates"]


def build_reference_cache(run: str, split: str) -> dict[str, dict]:
    rag = RebuildRagIndex(run, "noaug")
    cache = {}
    df = load_split(run, split)
    audit_path = run_dir(run) / f"retrieval_{split}.jsonl"
    for row in df.itertuples(index=False):
        docs, ids, sims = rag.search(row.statement)
        cache[row.id] = {"docs": docs, "ids": ids, "sims": sims}
        append_jsonl(audit_path, {
            "split": split,
            "id": row.id,
            "top_k": TOP_K,
            "retrieved_ids": ids,
            "retrieved_docs": docs,
            "retrieved_labels": [label_from_corpus_doc(doc) for doc in docs],
            "similarities": sims,
        })
    return cache


def optimize_prompt(run: str, kind: str) -> None:
    if kind not in {"norag", "rag"}:
        C.die(f"未知最佳化模式：{kind}")
    pool_size = 100 if kind == "norag" else 300
    n_trials = pool_size
    pool = load_pool(run, kind)
    if len(pool) != pool_size:
        C.die(f"{kind} prompt池數量錯誤：{len(pool)} != {pool_size}")
    val = load_split(run, "val")
    refs = build_reference_cache(run, "val") if kind == "rag" else {}
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db_path = run_dir(run) / f"tpe_{kind}.db"
    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name=f"{run}_{kind}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,
    )
    candidate_map = {item["id"]: item for item in pool}
    anchors = [item["id"] for item in pool if item["is_paper_prompt"] or item["is_current_prompt"]]
    if not study.trials:
        for cid in anchors:
            study.enqueue_trial({"candidate_id": cid})

    response_path = run_dir(run) / f"tpe_{kind}_responses.jsonl"
    trial_path = run_dir(run) / f"tpe_{kind}_trials.jsonl"
    model_digest = probe_llm_identity().get("digest")
    cache: dict[str, list[dict]] = defaultdict(list)
    for rec in C.read_jsonl(response_path):
        cache[rec["candidate_id"]].append(rec)

    def evaluate_candidate(candidate_id: str, trial_number: int | None = None) -> dict:
        candidate = candidate_map[candidate_id]
        existing = cache.get(candidate_id, [])
        done = {r["id"] for r in existing}
        for row in val.itertuples(index=False):
            if row.id in done:
                continue
            docs = refs[row.id]["docs"] if kind == "rag" else None
            prompt = render_prompt(candidate, row.statement, docs)
            resp = C.chat(prompt)
            pred, reason = C.parse_label(resp["raw_response"])
            rec = {
                "trial": trial_number,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["sha256"],
                "id": row.id,
                "true_label_id": int(row.label_id),
                "true_label": row.status,
                "pred_label_id": pred,
                "pred_label": C.ID_TO_LABEL.get(pred) if pred is not None else None,
                "parse_reason": reason,
                "invalid": pred is None,
                "invalid_reason": reason if pred is None else None,
                "correct": bool(pred is not None and pred == int(row.label_id)),
                "raw_response": resp["raw_response"],
                "rendered_prompt": prompt,
                "retrieved_ids": refs[row.id]["ids"] if kind == "rag" else None,
                "retrieved_docs": refs[row.id]["docs"] if kind == "rag" else None,
                "retrieved_labels": refs[row.id]["retrieved_labels"] if kind == "rag" else None,
                "retrieved_similarities": refs[row.id]["sims"] if kind == "rag" else None,
                "top_k": TOP_K if kind == "rag" else None,
                "model_digest": model_digest,
                "elapsed_s": round(resp["elapsed_s"], 3),
                "token_count": (resp.get("prompt_eval_count") or 0) + (resp.get("eval_count") or 0),
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "eval_count": resp.get("eval_count"),
            }
            append_jsonl(response_path, rec)
            existing.append(rec)
        cache[candidate_id] = existing
        correct = sum(int(r["correct"]) for r in existing)
        invalid = sum(int(r["pred_label_id"] is None) for r in existing)
        n = len(existing)
        return {
            "candidate_id": candidate_id,
            "n": n,
            "correct": correct,
            "invalid": invalid,
            "accuracy": correct / n if n else 0.0,
            "invalid_rate": invalid / n if n else 0.0,
        }

    def objective(trial):
        cid = trial.suggest_categorical("candidate_id", [item["id"] for item in pool])
        result = evaluate_candidate(cid, trial.number)
        trial.set_user_attr("candidate_sha256", candidate_map[cid]["sha256"])
        trial.set_user_attr("invalid", result["invalid"])
        trial.set_user_attr("n", result["n"])
        trial.set_user_attr("candidate_dims", candidate_map[cid]["dims"])
        append_jsonl(trial_path, {
            "study": kind,
            "trial": trial.number,
            "candidate_id": cid,
            "candidate_sha256": candidate_map[cid]["sha256"],
            "prompt": candidate_map[cid]["template"],
            "model_digest": model_digest,
            **result,
        })
        return result["accuracy"]

    remaining = max(n_trials - len(study.trials), 0)
    if remaining:
        C.log(f"TPE {kind}: pool={pool_size}, trials_remaining={remaining}, val={len(val)}")
        study.optimize(objective, n_trials=remaining)

    # Optuna 的 objective 以 strict accuracy 為主；正式選擇另做固定的
    # accuracy -> invalid較少 -> candidate ID 字典序 tie-break。
    complete_trials = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None
    ]
    if not complete_trials:
        C.die(f"TPE {kind} 沒有完成的trial")
    best_trial = sorted(
        complete_trials,
        key=lambda t: (
            -float(t.value),
            int(t.user_attrs.get("invalid", 10**9)),
            str(t.params["candidate_id"]),
        ),
    )[0]
    best = candidate_map[best_trial.params["candidate_id"]]
    save_json(run_dir(run) / f"best_prompt_{kind}.json", {
        "kind": kind,
        "candidate_id": best["id"],
        "template": best["template"],
        "sha256": best["sha256"],
        "dims": best["dims"],
        "is_paper_prompt": best["is_paper_prompt"],
        "is_current_prompt": best["is_current_prompt"],
        "pool_size": pool_size,
        "tpe_trials": len(study.trials),
        "selected_accuracy": best_trial.value,
        "selected_invalid": best_trial.user_attrs.get("invalid"),
        "selection_rule": "strict_accuracy desc, invalid asc, candidate_id asc",
        "top_k": TOP_K if kind == "rag" else None,
        "validation_split": "val",
        "test_used": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    })
    write_manifest(run, f"optimize_{kind}")
    C.log(f"TPE {kind}完成：{best['id']} accuracy={best_trial.value:.4f}")


def resolve_condition_prompt(run: str, kind: str, anchor: str) -> dict:
    if anchor == "tpe":
        return C.load_json(run_dir(run) / f"best_prompt_{kind}.json")
    pool = load_pool(run, kind)
    if anchor == "paper":
        return next(item for item in pool if item["is_paper_prompt"])
    if anchor == "current":
        return next(item for item in pool if item["is_current_prompt"])
    C.die(f"未知prompt anchor：{anchor}")


CONDITIONS = [
    ("norag_paper", None, "norag", "paper"),
    ("norag_current", None, "norag", "current"),
    ("norag_tpe", None, "norag", "tpe"),
    ("rag_noaug_paper", "noaug", "rag", "paper"),
    ("rag_noaug_current", "noaug", "rag", "current"),
    ("rag_noaug_tpe", "noaug", "rag", "tpe"),
    ("rag_aug_paper", "aug", "rag", "paper"),
    ("rag_aug_current", "aug", "rag", "current"),
    ("rag_aug_tpe", "aug", "rag", "tpe"),
]

CONDITION_DESCRIPTIONS = {
    "norag_paper": "純LLM；論文Table VIII原始最佳分類prompt",
    "norag_current": "純LLM；目前版本數字輸出分類prompt",
    "norag_tpe": "純LLM；100候選/100 trials選出的TPE最佳prompt",
    "rag_noaug_paper": "未擴增RAG；論文Table VIII原始最佳分類prompt；top_k=1",
    "rag_noaug_current": "未擴增RAG；目前版本含reference說明的分類prompt；top_k=1",
    "rag_noaug_tpe": "未擴增RAG；300候選/300 trials選出的TPE最佳prompt；top_k=1",
    "rag_aug_paper": "擴增RAG；論文Table VIII原始最佳分類prompt；top_k=1",
    "rag_aug_current": "擴增RAG；目前版本含reference說明的分類prompt；top_k=1",
    "rag_aug_tpe": "擴增RAG；沿用未擴增RAG的TPE最佳prompt；top_k=1",
}


def evaluate_conditions(run: str, limit: int | None = None) -> None:
    test = load_split(run, "test")
    if limit:
        test = test.head(limit)
    model_digest = probe_llm_identity().get("digest")
    for name, corpus_name, prompt_kind, anchor in CONDITIONS:
        out_path = run_dir(run) / "preds" / f"{name}.jsonl"
        done = {rec["id"] for rec in C.read_jsonl(out_path)}
        candidate = resolve_condition_prompt(run, prompt_kind, anchor)
        rag = RebuildRagIndex(run, corpus_name) if corpus_name else None
        for row in test.itertuples(index=False):
            if row.id in done:
                continue
            docs = ids = sims = None
            if rag:
                docs, ids, sims = rag.search(row.statement)
            prompt = render_prompt(candidate, row.statement, docs)
            resp = C.chat(prompt)
            pred, reason = C.parse_label(resp["raw_response"])
            append_jsonl(out_path, {
                "condition": name,
                "condition_description": CONDITION_DESCRIPTIONS[name],
                "id": row.id,
                "true_label_id": int(row.label_id),
                "true_label": row.status,
                "pred_label_id": pred,
                "pred_label": C.ID_TO_LABEL.get(pred) if pred is not None else None,
                "parse_reason": reason,
                "invalid_reason": reason if pred is None else None,
                "correct": bool(pred is not None and pred == int(row.label_id)),
                "raw_response": resp["raw_response"],
                "rendered_prompt": prompt,
                "prompt_id": candidate.get("candidate_id", candidate.get("id")),
                "prompt_sha256": candidate.get("sha256"),
                "retrieved_ids": ids,
                "retrieved_docs": docs,
                "retrieved_labels": [label_from_corpus_doc(doc) for doc in docs] if docs else None,
                "retrieved_similarities": sims,
                "top_k": TOP_K if rag else None,
                "model_digest": model_digest,
                "elapsed_s": round(resp["elapsed_s"], 3),
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "eval_count": resp.get("eval_count"),
                "token_count": (resp.get("prompt_eval_count") or 0) + (resp.get("eval_count") or 0),
            })
        records_for_export = list(C.read_jsonl(out_path))
        save_jsonl(run_dir(run) / f"predictions_{name}.jsonl", records_for_export)
        if corpus_name:
            save_jsonl(
                run_dir(run) / f"{corpus_name}_retrieval_audit.jsonl",
                [
                    {
                        "condition": name,
                        "id": rec["id"],
                        "retrieved_ids": rec.get("retrieved_ids"),
                        "retrieved_docs": rec.get("retrieved_docs"),
                        "retrieved_labels": rec.get("retrieved_labels"),
                        "similarities": rec.get("retrieved_similarities"),
                        "top_k": rec.get("top_k"),
                    }
                    for rec in records_for_export
                ],
            )
        save_json(run_dir(run) / "preds" / f"{name}.fingerprint.json", {
            "condition": name,
            "condition_description": CONDITION_DESCRIPTIONS[name],
            "n_test": len(test),
            "corpus": corpus_name,
            "top_k": TOP_K if rag else None,
            "prompt_id": candidate.get("candidate_id", candidate.get("id")),
            "prompt_sha256": candidate.get("sha256"),
            "test_split_sha256": C.sha256_file(split_dir(run) / "test.csv"),
        })
        C.log(f"完成條件：{name}")
    write_manifest(run, "evaluate")


def calculate_metrics(run: str) -> None:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    summary_rows = []
    all_metrics = {"run": run, "conditions": {}}
    for name, _, _, _ in CONDITIONS:
        path = run_dir(run) / "preds" / f"{name}.jsonl"
        records = list(C.read_jsonl(path))
        if not records:
            continue
        y_true = [int(r["true_label_id"]) for r in records]
        y_pred = [r["pred_label_id"] for r in records]
        invalid = sum(p is None for p in y_pred)
        y_pred_strict = [int(p) if p is not None else -1 for p in y_pred]
        strict_acc = float(np.mean([
            a == b for a, b in zip(y_true, y_pred_strict)
        ]))
        valid_pairs = [(a, b) for a, b in zip(y_true, y_pred) if b is not None]
        valid_acc = (
            float(np.mean([a == b for a, b in valid_pairs]))
            if valid_pairs else None
        )
        labels = list(range(len(C.LABELS)))
        p, r, f, sup = precision_recall_fscore_support(
            y_true, y_pred_strict, labels=labels, zero_division=0
        )
        matrix = confusion_matrix(
            y_true, y_pred_strict, labels=labels + [-1]
        )
        entry = {
            "n": len(records),
            "invalid_count": invalid,
            "invalid_rate": invalid / len(records),
            "strict_accuracy": strict_acc,
            "valid_only_accuracy": valid_acc,
            "macro_f1": float(f1_score(
                y_true, y_pred_strict, labels=labels,
                average="macro", zero_division=0
            )),
            "per_class": {
                label: {
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "f1": float(f[i]),
                    "support": int(sup[i]),
                }
                for i, label in enumerate(C.LABELS)
            },
            "confusion_labels": C.LABELS + ["INVALID"],
            "confusion_matrix": matrix.tolist(),
        }
        entry["condition_description"] = CONDITION_DESCRIPTIONS[name]
        all_metrics["conditions"][name] = entry
        save_json(run_dir(run) / f"metrics_{name}.json", {
            "run": run,
            "condition": name,
            "condition_description": CONDITION_DESCRIPTIONS[name],
            "metrics": entry,
        })
        pd.DataFrame(
            matrix,
            index=C.LABELS + ["INVALID"],
            columns=C.LABELS + ["INVALID"],
        ).to_csv(
            run_dir(run) / f"confusion_{name}.csv",
            encoding="utf-8",
        )
        summary_rows.append({
            "condition": name,
            "condition_description": CONDITION_DESCRIPTIONS[name],
            "n": len(records),
            "accuracy_strict": strict_acc,
            "accuracy_valid_only": valid_acc,
            "macro_f1": entry["macro_f1"],
            "invalid_count": invalid,
            "invalid_rate": invalid / len(records),
        })
    save_json(run_dir(run) / "metrics.json", all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        run_dir(run) / "summary.csv", index=False, encoding="utf-8"
    )
    write_manifest(run, "metrics")
    C.log(f"metrics完成：{len(summary_rows)}個條件")


def verify(run: str) -> None:
    checks = []
    manifest = C.load_json(run_dir(run) / "run_manifest.json")

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({
            "check": name,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        })
        C.log(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    check(
        "llm_model_digest_present",
        bool(manifest.get("model", {}).get("digest")),
        str(manifest.get("model", {}).get("digest")),
    )
    audit = C.load_json(split_dir(run) / "split_audit.json")
    check("split_audit_exists", bool(audit))
    if audit:
        sizes = {
            name: info["rows"]
            for name, info in audit.get("split_files", {}).items()
        }
        check("split_ratio_70_10_20", sizes == SPLIT_TARGETS, str(sizes))
        check(
            "split_exact_disjoint",
            all(v == 0 for v in audit.get("exact_overlaps", {}).values()),
            str(audit.get("exact_overlaps")),
        )
        check(
            "split_near_disjoint",
            all(
                x.get("count_ge_threshold", 1) == 0
                for x in audit.get("near_duplicate_scan", {}).values()
            ),
            str(audit.get("near_duplicate_scan")),
        )

    aug_meta = C.load_json(run_dir(run) / "augment_meta.json")
    if aug_meta:
        aug = pd.read_csv(run_dir(run) / "augmented.csv")
        counts = aug["source_id"].value_counts()
        check(
            "every_train_source_has_two_aug",
            bool(len(counts) == audit.get("split_files", {}).get("train", {}).get("rows")
                 and (counts == N_AUG_PER_SOURCE).all()),
            f"sources={len(counts)} rows={len(aug)}",
        )
        check("augmentation_output_expected", len(aug) == SPLIT_TARGETS["train"] * 2)
    else:
        check("augmentation_meta_exists", False)
    for audit_name in (
        "augment_raw.jsonl",
        "augment_filter_audit.jsonl",
        "augment_label_judge.jsonl",
        "augment_replenish_audit.jsonl",
        "judge_metrics.json",
    ):
        check(
            f"augmentation_audit_{audit_name}",
            (run_dir(run) / audit_name).exists(),
        )

    for name, expected in [("noaug", SPLIT_TARGETS["train"]),
                           ("aug", SPLIT_TARGETS["train"] * 3)]:
        meta = C.load_json(corpus_dir(run) / f"{name}_meta.json")
        check(
            f"corpus_{name}_exists",
            bool(meta),
            f"n_docs={meta.get('n_docs') if meta else None}",
        )
        if meta:
            check(f"corpus_{name}_topk", meta.get("top_k") == TOP_K)
            check(
                f"corpus_{name}_no_eval_overlap",
                all(v == 0 for v in meta.get("exact_overlap_with_eval_splits", {}).values()),
                str(meta.get("exact_overlap_with_eval_splits")),
            )
            check(f"corpus_{name}_row_count", meta.get("n_docs") == expected)

    for kind, expected in [("norag", 100), ("rag", 300)]:
        pool = C.load_json(run_dir(run) / f"prompt_pool_{kind}.json")
        candidates = pool.get("candidates", []) if pool else []
        check(
            f"prompt_pool_{kind}_size",
            len(candidates) == expected
            and len({x.get("sha256") for x in candidates}) == expected,
            f"n={len(candidates)}",
        )
        check(
            f"prompt_pool_{kind}_anchors",
            any(x.get("is_paper_prompt") for x in candidates)
            and any(x.get("is_current_prompt") for x in candidates),
        )
        if kind == "rag":
            check(
                "rag_prompt_reference_help",
                all(
                    "{reference_context}" in str(x.get("template", ""))
                    and any(
                        term in (
                            str(x.get("template", ""))
                            + str(x.get("reference_use_instruction", ""))
                        ).lower()
                        for term in ("help", "support", "evidence", "example")
                    )
                    for x in candidates
                ),
            )

    for kind, expected in [("norag", 100), ("rag", 300)]:
        best = C.load_json(run_dir(run) / f"best_prompt_{kind}.json")
        check(
            f"tpe_{kind}_completed",
            bool(best) and best.get("tpe_trials") == expected
            and best.get("test_used") is False,
            str(best.get("tpe_trials") if best else None),
        )

    metric_path = run_dir(run) / "metrics.json"
    check("metrics_exists", metric_path.exists())
    for name, _, _, _ in CONDITIONS:
        pred_path = run_dir(run) / "preds" / f"{name}.jsonl"
        records = list(C.read_jsonl(pred_path)) if pred_path.exists() else []
        check(
            f"condition_{name}_complete",
            len(records) == SPLIT_TARGETS["test"]
            and len({r["id"] for r in records}) == len(records),
            f"n={len(records)}",
        )
        check(
            f"prediction_export_{name}",
            (run_dir(run) / f"predictions_{name}.jsonl").exists(),
        )
        check(
            f"metrics_export_{name}",
            (run_dir(run) / f"metrics_{name}.json").exists()
            and (run_dir(run) / f"confusion_{name}.csv").exists(),
        )

    for corpus_name in ("noaug", "aug"):
        check(
            f"retrieval_audit_{corpus_name}",
            (run_dir(run) / f"{corpus_name}_retrieval_audit.jsonl").exists(),
        )

    df = pd.DataFrame(checks)
    df.to_csv(run_dir(run) / "acceptance_report.csv", index=False, encoding="utf-8")
    save_json(run_dir(run) / "acceptance_report.json", {
        "run": run,
        "pass": int((df["status"] == "PASS").sum()),
        "fail": int((df["status"] == "FAIL").sum()),
        "checks": checks,
    })
    if (df["status"] == "FAIL").any():
        C.die("新版實驗驗收失敗，不得引用結果")
    write_manifest(run, "verify")


def dependency_audit(run: str, stage: str) -> None:
    """在真正開始長時間階段前，留下可分析的依賴檢查。"""
    stage_modules = {
        "split": ["faiss", "sentence_transformers"],
        "augment": ["faiss", "sentence_transformers", "torch", "transformers", "ollama"],
        "pool": [],
        "corpus": ["faiss", "sentence_transformers"],
        "optimize": ["faiss", "sentence_transformers", "optuna", "ollama"],
        "evaluate": ["faiss", "sentence_transformers", "ollama"],
        "metrics": ["sklearn"],
        "verify": [],
        "all": ["faiss", "sentence_transformers", "torch", "transformers", "optuna", "ollama", "sklearn"],
    }
    modules = sorted(set(stage_modules.get(stage, [])))
    rows = []
    missing = []
    import importlib.util
    for module in modules:
        available = importlib.util.find_spec(module) is not None
        rows.append({"module": module, "available": available})
        if not available:
            missing.append(module)
    save_json(run_dir(run) / "dependency_audit.json", {
        "run": run,
        "stage": stage,
        "python": sys.version.split()[0],
        "modules": rows,
        "missing": missing,
        "status": "PASS" if not missing else "BLOCKED",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    })
    if missing:
        C.die(
            f"stage={stage} 缺少必要套件：{', '.join(missing)}。"
            "請依requirements.txt安裝後再執行；未產生正式實驗結果。"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=["split", "augment", "pool", "corpus", "optimize",
                 "evaluate", "metrics", "verify", "all"],
    )
    ap.add_argument("--run", default=RUN_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-run", default="TOPK1_20260918")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    dependency_audit(args.run, args.stage)

    if args.stage in {"split", "all"}:
        prepare_splits(args.run, args.seed)
    if args.stage in {"augment", "all"}:
        augment(args.run, args.reuse_run)
    if args.stage in {"pool", "all"}:
        prompt_pool(args.run)
    if args.stage in {"corpus", "all"}:
        build_corpus(args.run, "noaug")
        build_corpus(args.run, "aug")
    if args.stage in {"optimize", "all"}:
        prompt_pool(args.run)
        optimize_prompt(args.run, "norag")
        optimize_prompt(args.run, "rag")
    if args.stage in {"evaluate", "all"}:
        evaluate_conditions(args.run, args.limit)
    if args.stage in {"metrics", "all"}:
        calculate_metrics(args.run)
    if args.stage in {"verify", "all"}:
        verify(args.run)


if __name__ == "__main__":
    main()
