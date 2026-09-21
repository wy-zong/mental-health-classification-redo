"""Nested 70/10/20 split and incremental RAG construction.

This run is deliberately isolated from the independent REBUILD_70_10_20 run.
It preserves the existing 5,996-row train split, moves exactly 998 rows from
the old validation/test pools into train, reuses the existing RAG indexes, and
appends only the new documents.  Existing augmentation is reused as-is;
exactly two new paraphrases are generated for each of the 998 added sources.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common as C  # noqa: E402


def load_rebuild_module():
    path = SRC / "12_rebuild_experiment.py"
    spec = importlib.util.spec_from_file_location("rebuild_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_rebuild_module()

RUN = "NESTED_70_10_20_INCREMENTAL"
SEED = 42
TOP_K = 1
NEAR_DUP_THRESHOLD = 0.90
EMBED_MODEL = R.EMBED_MODEL
EMBED_MAX_SEQ = R.EMBED_MAX_SEQ
N_NEW_TRAIN = 998
N_AUG_PER_NEW_SOURCE = 2
MAX_ATTEMPTS_PER_SLOT = 5
TARGETS = {"train": 6994, "val": 1000, "test": 1998}
TARGETS_BY_LABEL = {
    "Normal": {"train": 1749, "val": 250, "test": 499},
    "Depression": {"train": 1749, "val": 250, "test": 499},
    "Anxiety": {"train": 1748, "val": 250, "test": 500},
    "Bipolar": {"train": 1748, "val": 250, "test": 500},
}
REPORT = {
    "train": {"Normal": 1499, "Depression": 1499, "Anxiety": 1499, "Bipolar": 1499},
    "old_val": {"Normal": 375, "Depression": 375, "Anxiety": 375, "Bipolar": 375},
    "old_test": {"Normal": 624, "Depression": 624, "Anxiety": 624, "Bipolar": 624},
}

BASE_CORPUS = C.ROOT / "data" / "corpus"
OLD_SPLITS = C.ROOT / "data" / "splits"
BOILERPLATE_RE = re.compile(
    r"^\s*(here('s| is)|sure[,!]|okay[,!]|below is|rewritten version)\s*:?",
    re.IGNORECASE,
)


def run_dir() -> Path:
    path = C.RUNS / RUN
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dir() -> Path:
    path = run_dir() / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def corpus_dir() -> Path:
    path = run_dir() / "corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: object) -> None:
    C.save_json(path, obj)


def old_split_hash(name: str) -> str:
    audit = C.load_json(OLD_SPLITS / "split_audit.json")
    expected = audit.get("split_files", {}).get(name, {}).get("sha256")
    path = OLD_SPLITS / f"{name}.csv"
    if expected:
        C.require_hash(path, expected, f"舊 {name} split")
    return C.sha256_file(path)


def load_old_splits() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "val_search", "val_confirm", "test"):
        path = OLD_SPLITS / f"{name}.csv"
        old_split_hash(name)
        frame = pd.read_csv(path)
        if "source_id" not in frame.columns:
            frame["source_id"] = frame["id"].astype(str)
        frame["old_split"] = name
        frames[name] = frame
    return frames


def row_columns(frame: pd.DataFrame, new_split: str, role: str) -> pd.DataFrame:
    out = frame.copy()
    out["new_split"] = new_split
    out["nested_role"] = role
    out["nested_source_id"] = out["source_id"]
    out["nested_id"] = out["id"]
    if role == "train_added":
        out["nested_id"] = "train_added_" + out["id"].astype(str)
    return out[[
        "nested_id", "source_id", "statement", "status", "label_id",
        "old_split", "new_split", "nested_role", "nested_source_id",
    ]].rename(columns={"nested_id": "id"})


def exact_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    left = {R.normalize_text(x) for x in a["statement"]}
    right = {R.normalize_text(x) for x in b["statement"]}
    return len(left & right)


def encode_splits_for_audit(splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    _, encode = R.embedder()
    return {
        name: encode(frame["statement"].astype(str).tolist())
        for name, frame in splits.items()
    }


def near_duplicate_audit(splits: dict[str, pd.DataFrame]) -> dict:
    faiss = C.import_faiss()
    vectors = encode_splits_for_audit(splits)
    result = {
        "threshold": NEAR_DUP_THRESHOLD,
        "embedding_model": EMBED_MODEL,
        "max_seq_length": EMBED_MAX_SEQ,
        "metric": "cosine on normalized vectors",
        "pairs": {},
    }
    names = ["train", "val", "test"]
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            index = faiss.IndexFlatIP(vectors[left_name].shape[1])
            index.add(vectors[left_name])
            sims, _ = index.search(vectors[right_name], 1)
            values = sims[:, 0]
            key = f"{left_name}|{right_name}"
            result["pairs"][key] = {
                "reference_split": left_name,
                "query_split": right_name,
                "n": int(len(values)),
                "max_cosine": float(values.max()),
                "mean_cosine": float(values.mean()),
                "count_ge_threshold": int((values >= NEAR_DUP_THRESHOLD).sum()),
            }
    return result


def prepare_nested_splits() -> None:
    out = split_dir()
    old = load_old_splits()
    labels = C.LABELS
    old_train = old["train"].copy()
    old_val = pd.concat([old["val_search"], old["val_confirm"]], ignore_index=True)
    old_test = old["test"].copy()

    val_selected = []
    val_remaining = []
    test_selected = []
    test_remaining = []
    for i, label in enumerate(labels):
        val_pool = old_val[old_val["status"] == label]
        val_take = val_pool.sample(n=TARGETS_BY_LABEL[label]["val"], random_state=SEED + 100 + i)
        val_selected.append(val_take)
        val_remaining.append(val_pool.drop(val_take.index))

        test_pool = old_test[old_test["status"] == label]
        test_take = test_pool.sample(n=TARGETS_BY_LABEL[label]["test"], random_state=SEED + 200 + i)
        test_selected.append(test_take)
        test_remaining.append(test_pool.drop(test_take.index))

    new_val = pd.concat(val_selected, ignore_index=True)
    new_test = pd.concat(test_selected, ignore_index=True)
    added = pd.concat(val_remaining + test_remaining, ignore_index=True)
    if len(added) != N_NEW_TRAIN:
        C.die(f"新增 train 數量錯誤：{len(added)} != {N_NEW_TRAIN}")

    new_train = pd.concat([
        row_columns(old_train, "train", "train_legacy"),
        row_columns(added, "train", "train_added"),
    ], ignore_index=True)
    new_val = row_columns(new_val, "val", "val_nested")
    new_test = row_columns(new_test, "test", "test_nested")
    splits = {"train": new_train, "val": new_val, "test": new_test}

    exact = {f"{a}|{b}": exact_overlap(splits[a], splits[b])
             for i, a in enumerate(("train", "val", "test"))
             for b in ("train", "val", "test")[i + 1:]}
    if any(exact.values()):
        C.die(f"nested split 有 exact duplicate：{exact}")

    near = near_duplicate_audit(splits)
    if any(info["count_ge_threshold"] for info in near["pairs"].values()):
        C.die("nested split 有 near duplicate，詳見 near_duplicate_audit.json")

    for name, frame in splits.items():
        frame = frame.sample(frac=1.0, random_state=SEED + 300 + len(name)).reset_index(drop=True)
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        splits[name] = frame
    added_frame = splits["train"][splits["train"]["nested_role"] == "train_added"].copy()
    added_frame.to_csv(out / "train_added.csv", index=False, encoding="utf-8")

    mapping = pd.concat([
        splits["train"], splits["val"], splits["test"]
    ], ignore_index=True)
    mapping[["id", "nested_source_id", "source_id", "old_split",
             "new_split", "nested_role"]].to_csv(
        out / "source_mapping.csv", index=False, encoding="utf-8"
    )

    counts = {
        name: {label: int((frame["status"] == label).sum()) for label in labels}
        for name, frame in splits.items()
    }
    manifest = {
        "run": RUN,
        "strategy": "nested_old_train_plus_998_added",
        "seed": SEED,
        "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "targets": TARGETS,
        "counts": {name: len(frame) for name, frame in splits.items()},
        "counts_by_label": counts,
        "old_train_preserved": True,
        "new_val_is_subset_of_old_val": True,
        "new_test_is_subset_of_old_test": True,
        "new_train_added_rows": len(added_frame),
        "old_split_hashes": {name: old_split_hash(name) for name in
                              ("train", "val_search", "val_confirm", "test")},
        "raw_dataset_sha256": C.RAW_DATASET_SHA256,
        "files": {
            name: {"rows": len(frame), "sha256": C.sha256_file(out / f"{name}.csv")}
            for name, frame in splits.items()
        },
        "source_mapping_sha256": C.sha256_file(out / "source_mapping.csv"),
    }
    save_json(out / "split_manifest.json", manifest)
    save_json(out / "near_duplicate_audit.json", near)
    save_json(out / "split_audit.json", {
        "run": RUN,
        "exact_overlaps": exact,
        "near_duplicate_scan": near,
        "nested_invariants": {
            "old_train_rows": len(old_train),
            "old_train_in_new_train": int(sum(
                R.normalize_text(x) in {R.normalize_text(y) for y in splits["train"]["statement"]}
                for x in old_train["statement"]
            )),
            "new_val_rows": len(new_val),
            "new_test_rows": len(new_test),
            "new_train_added_rows": len(added_frame),
        },
    })
    C.log(f"nested split 完成：train={len(new_train):,}, val={len(new_val):,}, test={len(new_test):,}")


def clean_rewrite(text: str) -> str:
    value = str(text or "").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines and BOILERPLATE_RE.match(lines[0]) and len(lines) > 1:
        lines = lines[1:]
    value = "\n".join(lines).strip()
    value = value.strip("`\"'").strip()
    return value


def rewrite_is_valid(original: str, candidate: str, previous: set[str]) -> tuple[bool, str]:
    if not candidate:
        return False, "empty"
    if R.normalize_text(candidate) == R.normalize_text(original):
        return False, "exact_source_duplicate"
    key = R.normalize_text(candidate)
    if key in previous:
        return False, "duplicate_with_source_variants"
    return True, "accepted"


def augment_new_sources() -> None:
    out = run_dir()
    train_added = pd.read_csv(split_dir() / "train_added.csv")
    raw_path = out / "augment_raw.jsonl"
    filter_path = out / "augment_filter_audit.jsonl"
    replenish_path = out / "augment_replenish_audit.jsonl"
    judge_path = out / "augment_label_judge.jsonl"

    raw = list(C.read_jsonl(raw_path))
    accepted: dict[tuple[str, int], dict] = {}
    used_by_source: dict[str, set[str]] = {}
    filter_records = []
    for rec in raw:
        source_id = str(rec["source_id"])
        slot = int(rec["slot"])
        key = (source_id, slot)
        candidate = clean_rewrite(rec.get("raw_response", ""))
        used = used_by_source.setdefault(source_id, set())
        ok, reason = rewrite_is_valid(rec["original"], candidate, used)
        filter_records.append({
            "source_id": source_id,
            "slot": slot,
            "attempt": rec.get("attempt"),
            "candidate": candidate,
            "accepted": ok and key not in accepted,
            "reason": reason if key not in accepted else "slot_already_filled",
        })
        if ok and key not in accepted:
            accepted[key] = {**rec, "cleaned_text": candidate}
            used.add(R.normalize_text(candidate))

    rows_by_source = {str(row.source_id): row for row in train_added.itertuples(index=False)}
    start = time.perf_counter()
    for source_id, row in rows_by_source.items():
        for slot in range(N_AUG_PER_NEW_SOURCE):
            key = (source_id, slot)
            attempts = [int(r.get("attempt", 0)) for r in raw
                        if str(r.get("source_id")) == source_id and int(r.get("slot", -1)) == slot]
            attempt = max(attempts, default=0)
            while key not in accepted:
                attempt += 1
                if attempt > MAX_ATTEMPTS_PER_SLOT:
                    C.die(f"{source_id} slot={slot} 超過改寫嘗試上限")
                prompt = f"{R.REWRITE_INSTRUCTION}\n\nText:\n{str(row.statement).strip()}"
                options = {**R.REWRITE_OPTIONS, "seed": int(R.REWRITE_OPTIONS["seed"] + slot + attempt * 1000)}
                response = C.chat(prompt, options=options)
                record = {
                    "gen_key": f"{source_id}#{slot}#{attempt}",
                    "source_id": source_id,
                    "slot": slot,
                    "attempt": attempt,
                    "status": row.status,
                    "label_id": int(row.label_id),
                    "original": row.statement,
                    "raw_response": response["raw_response"],
                    "prompt": prompt,
                    "prompt_sha256": C.sha256_text(prompt),
                    "elapsed_s": round(response["elapsed_s"], 3),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                    "generation_options": options,
                }
                C.append_jsonl(raw_path, record)
                raw.append(record)
                candidate = clean_rewrite(record["raw_response"])
                used = used_by_source.setdefault(source_id, set())
                ok, reason = rewrite_is_valid(record["original"], candidate, used)
                audit = {
                    "source_id": source_id,
                    "slot": slot,
                    "attempt": attempt,
                    "candidate": candidate,
                    "accepted": ok,
                    "reason": reason,
                }
                C.append_jsonl(filter_path, audit)
                if ok:
                    accepted[key] = {**record, "cleaned_text": candidate}
                    used.add(R.normalize_text(candidate))
                C.log(f"augment {len(accepted):,}/{len(train_added) * N_AUG_PER_NEW_SOURCE:,}")

    # Rewrite filter audit from all historical attempts so a resumed run is
    # complete, but keep exactly one audit row per generated attempt.
    final_filter_records = []
    for rec in raw:
        source_id = str(rec.get("source_id"))
        slot = int(rec.get("slot", -1))
        if source_id not in rows_by_source or slot < 0:
            continue
        selected = accepted.get((source_id, slot))
        selected_here = bool(selected and rec.get("gen_key") == selected.get("gen_key"))
        candidate = clean_rewrite(rec.get("raw_response", ""))
        ok, reason = rewrite_is_valid(rec.get("original", ""), candidate, set())
        final_filter_records.append({
            "source_id": source_id,
            "slot": slot,
            "attempt": rec.get("attempt"),
            "gen_key": rec.get("gen_key"),
            "candidate": candidate,
            "accepted": selected_here,
            "reason": "final_selected" if selected_here else reason,
        })
    R.save_jsonl(filter_path, final_filter_records)

    final_rows = []
    for source_id, row in rows_by_source.items():
        for slot in range(N_AUG_PER_NEW_SOURCE):
            rec = accepted[(source_id, slot)]
            final_rows.append({
                "id": f"aug_{source_id}_{slot}",
                "source_id": source_id,
                "statement": rec["cleaned_text"],
                "status": row.status,
                "label_id": int(row.label_id),
                "variant": slot,
                "generation_attempt": int(rec["attempt"]),
                "consistency_judge": "skipped_by_plan",
                "similarity_to_source": None,
            })
            C.append_jsonl(judge_path, {
                "source_id": source_id,
                "variant": slot,
                "original_label": row.status,
                "judge": "skipped_by_plan",
                "reason": "existing augmentation reused; expensive consistency classifier not run",
                "final_in_rag": True,
            })
    augmented = pd.DataFrame(final_rows).sort_values(["source_id", "variant"])
    augmented.to_csv(out / "augmented_new.csv", index=False, encoding="utf-8")
    save_json(out / "augment_replenish_audit.json", {
        "source_split": "train_added",
        "source_rows": len(train_added),
        "target_per_source": N_AUG_PER_NEW_SOURCE,
        "output_rows": len(augmented),
        "all_sources_exactly_two": bool(
            (augmented["source_id"].value_counts() == N_AUG_PER_NEW_SOURCE).all()
        ),
        "max_attempts_per_slot": MAX_ATTEMPTS_PER_SLOT,
        "elapsed_s": round(time.perf_counter() - start, 3),
    })
    save_json(out / "augmented_meta.json", {
        "run": RUN,
        "source_split": "train_added",
        "source_rows": len(train_added),
        "output_rows": len(augmented),
        "target_per_source": N_AUG_PER_NEW_SOURCE,
        "output_sha256": C.sha256_file(out / "augmented_new.csv"),
        "rewrite_instruction": R.REWRITE_INSTRUCTION,
        "rewrite_instruction_sha256": C.sha256_text(R.REWRITE_INSTRUCTION),
        "rewrite_options": R.REWRITE_OPTIONS,
        "consistency_judge": "skipped_by_plan",
        "existing_augmentation_reused_without_replenishing_old_sources": True,
    })
    C.log(f"新增改寫完成：{len(augmented):,} 筆")


def load_base_corpus(name: str) -> tuple[dict, dict, object]:
    meta_path = BASE_CORPUS / f"{name}_meta.json"
    index_path = BASE_CORPUS / f"{name}.index"
    docs_path = BASE_CORPUS / f"{name}_docs.json"
    meta = C.load_json(meta_path)
    if not meta:
        C.die(f"缺少既有 corpus metadata：{meta_path}")
    C.require_hash(index_path, meta["index_sha256"], f"既有 {name} index")
    payload = C.load_json(docs_path)
    faiss = C.import_faiss()
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(payload["docs"]) or index.ntotal != len(payload["doc_ids"]):
        C.die(f"既有 {name} corpus index/docs 數量不一致")
    return meta, payload, index


def build_incremental_corpora() -> None:
    out = corpus_dir()
    added = pd.read_csv(split_dir() / "train_added.csv")
    new_aug = pd.read_csv(run_dir() / "augmented_new.csv")
    base_meta = {}
    base_payload = {}
    base_indices = {}
    for name in ("noaug", "aug"):
        meta, payload, index = load_base_corpus(name)
        base_meta[name] = meta
        base_payload[name] = payload
        base_indices[name] = index

    noaug_rows = [R.corpus_entry(row.statement, row.status) for row in added.itertuples(index=False)]
    noaug_ids = [str(row.source_id) for row in added.itertuples(index=False)]
    aug_rows = noaug_rows + [R.corpus_entry(row.statement, row.status) for row in new_aug.itertuples(index=False)]
    aug_ids = noaug_ids + [str(row.id) for row in new_aug.itertuples(index=False)]
    all_new_docs = noaug_rows + aug_rows
    _, encode = R.embedder()
    vectors = encode(all_new_docs)
    noaug_vecs = vectors[:len(noaug_rows)]
    aug_vecs = vectors[len(noaug_rows):]
    faiss = C.import_faiss()

    for name, added_docs, added_ids, added_vecs in (
        ("noaug", noaug_rows, noaug_ids, noaug_vecs),
        ("aug", aug_rows, aug_ids, aug_vecs),
    ):
        index = base_indices[name]
        base_count = int(index.ntotal)
        index.add(added_vecs)
        docs = base_payload[name]["docs"] + added_docs
        doc_ids = base_payload[name]["doc_ids"] + added_ids
        index_path = out / f"{name}.index"
        docs_path = out / f"{name}_docs.json"
        faiss.write_index(index, str(index_path))
        save_json(docs_path, {"doc_ids": doc_ids, "docs": docs})
        eval_texts = {R.normalize_text(x) for x in pd.read_csv(split_dir() / "val.csv")["statement"]}
        eval_texts |= {R.normalize_text(x) for x in pd.read_csv(split_dir() / "test.csv")["statement"]}
        corpus_texts = {
            R.normalize_text(doc.rsplit(" true_label is ", 1)[0])
            for doc in docs
        }
        overlap = len(eval_texts & corpus_texts)
        if overlap:
            C.die(f"增量 {name} corpus 含 val/test 原文：{overlap} 筆")
        meta = {
            "run": RUN,
            "name": name,
            "source": "existing_corpus_plus_nested_train_added" if name == "noaug"
            else "existing_aug_corpus_plus_nested_train_added_augmented",
            "n_docs": len(docs),
            "base_n_docs": base_count,
            "added_doc_count": len(added_docs),
            "added_source_rows": len(added),
            "added_augmented_rows": len(new_aug) if name == "aug" else 0,
            "embedding_model": EMBED_MODEL,
            "max_seq_length": EMBED_MAX_SEQ,
            "normalized": True,
            "metric": "cosine (IndexFlatIP on normalized vectors)",
            "top_k": TOP_K,
            "exact_overlap_with_eval_splits": {"val": 0, "test": 0},
            "base_index_sha256": base_meta[name]["index_sha256"],
            "base_docs_sha256": C.sha256_file(BASE_CORPUS / f"{name}_docs.json"),
            "added_doc_ids_sha256": C.sha256_text(json.dumps(added_ids, ensure_ascii=False)),
            "index_sha256": C.sha256_file(index_path),
            "docs_sha256": C.sha256_file(docs_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        save_json(out / f"{name}_meta.json", meta)
        C.append_jsonl(run_dir() / "incremental_corpus_audit.jsonl", meta)
    save_json(run_dir() / "corpus_reuse_manifest.json", {
        "run": RUN,
        "base_corpus_dir": str(BASE_CORPUS),
        "base_corpora": {
            name: {
                "index_sha256": base_meta[name]["index_sha256"],
                "n_docs": base_meta[name]["n_docs"],
            }
            for name in ("noaug", "aug")
        },
        "nested_train_added_rows": len(added),
        "new_augmented_rows": len(new_aug),
        "top_k": TOP_K,
    })
    C.log("增量 noaug/aug corpus 完成")


def verify() -> None:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    sdir = split_dir()
    splits = {name: pd.read_csv(sdir / f"{name}.csv") for name in ("train", "val", "test")}
    check("split_counts", {k: len(v) for k, v in splits.items()} == TARGETS,
          str({k: len(v) for k, v in splits.items()}))
    train_texts = {R.normalize_text(y) for y in splits["train"]["statement"]}
    old_train_texts = {
        R.normalize_text(x) for x in pd.read_csv(OLD_SPLITS / "train.csv")["statement"]
    }
    check("old_train_preserved", len(old_train_texts) == 5996 and old_train_texts <= train_texts)
    old_val_texts = {R.normalize_text(x) for x in pd.concat([
        pd.read_csv(OLD_SPLITS / "val_search.csv"), pd.read_csv(OLD_SPLITS / "val_confirm.csv")
    ])["statement"]}
    old_test_texts = {R.normalize_text(x) for x in pd.read_csv(OLD_SPLITS / "test.csv")["statement"]}
    check("val_subset_old_val", set(splits["val"].statement.map(R.normalize_text)) <= old_val_texts)
    check("test_subset_old_test", set(splits["test"].statement.map(R.normalize_text)) <= old_test_texts)
    check("new_train_added_count", len(pd.read_csv(sdir / "train_added.csv")) == N_NEW_TRAIN)
    check("split_exact_disjoint", all(exact_overlap(splits[a], splits[b]) == 0
                                       for i, a in enumerate(("train", "val", "test"))
                                       for b in ("train", "val", "test")[i + 1:]))
    near = C.load_json(sdir / "near_duplicate_audit.json")
    check("split_near_disjoint", all(x["count_ge_threshold"] == 0 for x in near["pairs"].values()))

    aug_meta = C.load_json(run_dir() / "augmented_meta.json")
    check("new_augmented_count", int(aug_meta.get("output_rows", -1)) == N_NEW_TRAIN * N_AUG_PER_NEW_SOURCE,
          str(aug_meta.get("output_rows")))
    if (run_dir() / "augmented_new.csv").exists():
        new_aug = pd.read_csv(run_dir() / "augmented_new.csv")
        counts = new_aug["source_id"].value_counts()
        check("new_sources_exactly_two", len(counts) == N_NEW_TRAIN and bool((counts == 2).all()), str(counts.value_counts().to_dict()))

    expected_corpus = {"noaug": 6994, "aug": 18095}
    for name, expected in expected_corpus.items():
        meta = C.load_json(corpus_dir() / f"{name}_meta.json")
        check(f"{name}_count", int(meta.get("n_docs", -1)) == expected, str(meta.get("n_docs")))
        check(f"{name}_top_k", meta.get("top_k") == TOP_K, str(meta.get("top_k")))
        if (corpus_dir() / f"{name}.index").exists():
            faiss = C.import_faiss()
            index = faiss.read_index(str(corpus_dir() / f"{name}.index"))
            check(f"{name}_index_count", index.ntotal == expected, str(index.ntotal))
        check(f"{name}_no_eval_overlap", meta.get("exact_overlap_with_eval_splits") == {"val": 0, "test": 0}, str(meta.get("exact_overlap_with_eval_splits")))

    df = pd.DataFrame(checks)
    df.to_csv(run_dir() / "acceptance_report.csv", index=False, encoding="utf-8-sig")
    save_json(run_dir() / "acceptance_report.json", {
        "run": RUN,
        "pass": int((df["status"] == "PASS").sum()),
        "fail": int((df["status"] == "FAIL").sum()),
        "checks": checks,
    })
    if (df["status"] == "FAIL").any():
        C.die("nested incremental RAG 驗收失敗")
    C.log("nested incremental RAG 驗收通過")


def write_run_manifest(stage: str) -> None:
    save_json(run_dir() / "run_manifest.json", {
        "run": RUN,
        "profile": "nested_70_10_20_incremental",
        "last_stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "git_revision": R.git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": R.probe_llm_identity(),
        "llm_generation_options": C.GEN_OPTIONS,
        "seed": SEED,
        "targets": TARGETS,
        "top_k": TOP_K,
        "embedding_model": EMBED_MODEL,
        "max_seq_length": EMBED_MAX_SEQ,
        "augmentation": {
            "existing_corpus_reused": True,
            "new_sources": N_NEW_TRAIN,
            "new_variants_per_source": N_AUG_PER_NEW_SOURCE,
            "consistency_judge": "skipped_by_plan",
        },
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["split", "augment", "corpus", "verify", "all"])
    args = ap.parse_args()
    if args.stage in {"split", "all"}:
        prepare_nested_splits()
        write_run_manifest("split")
    if args.stage in {"augment", "all"}:
        if not (split_dir() / "train_added.csv").exists():
            prepare_nested_splits()
        augment_new_sources()
        write_run_manifest("augment")
    if args.stage in {"corpus", "all"}:
        if not (run_dir() / "augmented_new.csv").exists():
            C.die("缺少 augmented_new.csv，請先執行 --stage augment")
        build_incremental_corpora()
        write_run_manifest("corpus")
    if args.stage in {"verify", "all"}:
        verify()
        write_run_manifest("verify")


if __name__ == "__main__":
    main()
