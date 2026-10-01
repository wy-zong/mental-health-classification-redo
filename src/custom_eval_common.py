"""15_（CUSTOM_PROMPT_EVAL）事後分析的共用工具。

17_～20_ 與混淆矩陣繪圖都讀同一批逐筆預測。這裡集中處理三件事：

* C1–C5 代號與 15_ 條件名稱的對照（依論文 Table VI）；
* 載入預測檔時的來源檢查：experiment 欄、prompt sha256（對照現行 15_ 原始碼）、
  test split sha256、筆數與 id 集合，任何一項不符就停止；
* 向量化的 accuracy / macro-F1 / weighted-F1 與 paired bootstrap。

指標慣例與論文一致：無效回應計為答錯；F1 只在四個有效類別上平均，
INVALID 是一種「預測錯」而不是第五個類別（等同 sklearn
``f1_score(labels=四類, average=...)``）。
"""
from __future__ import annotations

import ast
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RUN = "NESTED_70_10_20_INCREMENTAL"
BASE_OUT = C.RUNS / RUN
OUT = BASE_OUT / "CUSTOM_PROMPT_EVAL"
TEST_PATH = BASE_OUT / "splits" / "test.csv"
N_TEST = 1998
SEED = 42
N_BOOT = 10000

# 論文 Table VI 的代號。
CODES = {
    "C1": "norag_base",
    "C2": "rag_noaug_base",
    "C3": "rag_aug_base",
    "C4": "rag_noaug_optimized",
    "C5": "rag_aug_optimized",
}
CODE_OF = {cond: code for code, cond in CODES.items()}
DESCRIPTIONS = {
    "C1": "LLM only",
    "C2": "Llama 3.1 & RAG",
    "C3": "Llama 3.1 & RAG & augmentation",
    "C4": "Llama 3.1 & RAG, prompt optimized, without augmentation",
    "C5": "Llama 3.1 & RAG, prompt optimized, augmentation",
}
# Table VII 的 10 組比較：C(5,2) 全部配對，差值為後者減前者。
PAIRS = list(combinations(CODES, 2))

LABELS = C.LABELS                      # 0..3
INVALID = len(LABELS)                  # 4
PLOT_LABELS = LABELS + ["INVALID"]


def source_15() -> tuple[dict, tuple]:
    """從 15_ 原始碼讀出 PROMPTS 與 CONDITIONS 字面值。

    直接 import 15_ 會在載入時探測 Ollama，因此改用 ast 讀字面值。
    """
    path = Path(__file__).resolve().parent / "15_nested_custom_prompt_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in {"PROMPTS", "CONDITIONS"}:
                found[name] = ast.literal_eval(node.value)
    if set(found) != {"PROMPTS", "CONDITIONS"}:
        C.die(f"無法從 {path.name} 讀出 PROMPTS / CONDITIONS")
    return found["PROMPTS"], found["CONDITIONS"]


def load_test() -> pd.DataFrame:
    test = pd.read_csv(TEST_PATH)
    if len(test) != N_TEST:
        C.die(f"test 筆數 {len(test)} ≠ {N_TEST}")
    return test


def load_predictions() -> dict[str, list[dict]]:
    """回傳 {代號: 依 test.csv 順序排列的逐筆預測}，並做來源檢查。"""
    prompts, conditions = source_15()
    prompt_of = {cond: prompts[p] for cond, p, _ in conditions}
    corpus_of = {cond: corpus for cond, _, corpus in conditions}
    if set(prompt_of) != set(CODES.values()):
        C.die(f"15_ 的條件與 C1–C5 對照不符：{sorted(prompt_of)}")

    test = load_test()
    test_sha = C.sha256_file(TEST_PATH)
    order = [str(x) for x in test["id"]]
    truth = dict(zip(order, test["status"]))

    out = {}
    for code, cond in CODES.items():
        path = OUT / f"predictions_{cond}.jsonl"
        records = list(C.read_jsonl(path))
        by_id = {str(r["id"]): r for r in records}
        expected_sha = C.sha256_text(prompt_of[cond]["template"])
        problems = []
        if len(records) != N_TEST or len(by_id) != N_TEST or set(by_id) != set(order):
            problems.append(f"筆數或 id 集合不符（{len(records)} 筆）")
        if any(r.get("experiment") != "CUSTOM_PROMPT_EVAL" for r in records):
            problems.append("experiment 欄不是 CUSTOM_PROMPT_EVAL")
        if any(r.get("condition") != cond for r in records):
            problems.append("condition 欄與檔名不符")
        if any(r.get("prompt_sha256") != expected_sha for r in records):
            problems.append("prompt_sha256 與現行 15_ 的 template 不符")
        if any(r.get("corpus") != corpus_of[cond] for r in records):
            problems.append("corpus 欄與 15_ 的設定不符")
        if len({r.get("model_digest") for r in records}) != 1:
            problems.append("model_digest 不只一種")
        if not problems and any(truth[i] != by_id[i]["true_label"] for i in order):
            problems.append("true_label 與 test.csv 不符")
        metrics = C.load_json(OUT / f"metrics_{cond}.json")
        if metrics.get("test_split_sha256") != test_sha:
            problems.append("metrics 記錄的 test split sha256 與現行 test.csv 不符")
        if problems:
            C.die(f"{path.name} 來源檢查失敗：" + "；".join(problems))
        out[code] = [by_id[i] for i in order]
    return out


def encode(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, y_pred) 整數陣列；y_pred 的 INVALID 編為 4。"""
    y_true = np.array([C.LABEL_TO_ID[r["true_label"]] for r in records], dtype=np.int64)
    y_pred = np.array(
        [INVALID if r["pred_label_id"] is None else int(r["pred_label_id"]) for r in records],
        dtype=np.int64,
    )
    return y_true, y_pred


# ---------------------------------------------------------------- 指標

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """4×5 計數矩陣（列＝真實、欄＝預測，含 INVALID 欄）。"""
    k = len(PLOT_LABELS)
    return np.bincount(y_true * k + y_pred, minlength=len(LABELS) * k).reshape(len(LABELS), k)


def _prf_from_counts(tp, pred_n, true_n):
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_n > 0, tp / pred_n, 0.0)
        recall = np.where(true_n > 0, tp / true_n, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """accuracy、macro-F1、weighted-F1、invalid rate。cm 可以帶前置批次維度。"""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum(axis=(-2, -1))
    tp = np.diagonal(cm[..., :, : len(LABELS)], axis1=-2, axis2=-1)
    true_n = cm.sum(axis=-1)
    pred_n = cm[..., :, : len(LABELS)].sum(axis=-2)
    _, _, f1 = _prf_from_counts(tp, pred_n, true_n)
    return {
        "accuracy": tp.sum(axis=-1) / n,
        "macro_f1": f1.mean(axis=-1),
        "weighted_f1": (f1 * true_n).sum(axis=-1) / true_n.sum(axis=-1),
        "invalid_rate": cm[..., INVALID].sum(axis=-1) / n,
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion(y_true, y_pred)
    tp = np.diag(cm[:, : len(LABELS)])
    true_n = cm.sum(axis=1)
    pred_n = cm[:, : len(LABELS)].sum(axis=0)
    p, r, f = _prf_from_counts(tp, pred_n, true_n)
    rows = [
        {"label": lab, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(true_n[i])}
        for i, lab in enumerate(LABELS)
    ]
    rows.append({"label": "macro avg", "precision": p.mean(), "recall": r.mean(),
                 "f1": f.mean(), "support": int(true_n.sum())})
    w = true_n / true_n.sum()
    rows.append({"label": "weighted avg", "precision": (p * w).sum(), "recall": (r * w).sum(),
                 "f1": (f * w).sum(), "support": int(true_n.sum())})
    return pd.DataFrame(rows)


def check_against_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """確認向量化實作與 sklearn 的定義一致。"""
    from sklearn.metrics import accuracy_score, f1_score

    m = metrics_from_confusion(confusion(y_true, y_pred))
    labels = list(range(len(LABELS)))
    ref = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    for key, val in ref.items():
        if abs(float(m[key]) - float(val)) > 1e-12:
            C.die(f"{key} 與 sklearn 不一致：{float(m[key])} vs {val}")


# ---------------------------------------------------------------- bootstrap

def bootstrap_indices(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """與 09_stats.paired_bootstrap 相同的抽法：每次 rng.integers(0, n, n)。"""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, n) for _ in range(n_boot)])


def boot_confusions(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray,
                    chunk: int = 1000) -> np.ndarray:
    """每組重抽索引的 4×5 混淆矩陣，形狀 (n_boot, 4, 5)。"""
    k = len(PLOT_LABELS)
    cells = len(LABELS) * k
    codes = y_true * k + y_pred
    out = np.empty((len(idx), len(LABELS), k), dtype=np.int64)
    for s in range(0, len(idx), chunk):
        block = codes[idx[s:s + chunk]]                       # (b, n)
        offset = np.arange(len(block))[:, None] * cells
        counts = np.bincount((block + offset).ravel(), minlength=len(block) * cells)
        out[s:s + chunk] = counts.reshape(len(block), len(LABELS), k)
    return out


def ci95(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni 校正。

    與 09_stats.holm 相同，但不做 round(…, 6)：Table VII 的 Holm p 小到 1e-64，
    四捨五入到小數第六位會變成 0。
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted
