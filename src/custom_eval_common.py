"""實驗 15 主實驗（v1：15_ CUSTOM_PROMPT_EVAL；v2：27_ MAIN_EVAL_AUGV2）事後分析的共用工具。

17_～20_、23_ 與混淆矩陣繪圖都讀同一批逐筆預測。這裡集中處理四件事：

* profile：v1 與 v2 各自的結果目錄、C1–C5 代號與條件 ID 的對照、語料、top-k
  （``--profile {v1,v2}``，預設 v1；v1 是凍結的存檔，只能輸出到 ``--out-dir``）；
* 載入預測檔時的來源檢查：experiment 欄、prompt sha256（對照現行 15_ 原始碼）、
  test split sha256、筆數與 id 集合，任何一項不符就停止；
* 向量化的 accuracy / macro-F1 / weighted-F1 與 paired bootstrap；
* 語料 docs 的 sha 檢查（容許 core.autocrlf 造成的 CRLF，同 27_ corpus_info）。

代號沿用論文 Table VI 的 C1–C5，但同一代號在 v1 與 v2 指的條件不同
（v2 的 C3／C5 用 aug_v2 語料，RAG 條件的 k 也不同），所以輸出一律同時寫出
代號、條件 ID 與顯示名稱。

指標慣例與論文一致：無效回應計為答錯；F1 只在四個有效類別上平均，
INVALID 是一種「預測錯」而不是第五個類別（等同 sklearn
``f1_score(labels=四類, average=...)``）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RUN = "NESTED_70_10_20_INCREMENTAL"
BASE_OUT = C.RUNS / RUN
V1_OUT = BASE_OUT / "CUSTOM_PROMPT_EVAL"
V2_OUT = BASE_OUT / "MAIN_EVAL_AUGV2"
OUT = V1_OUT                           # v1；21_、22_、24_ 直接使用
TEST_PATH = BASE_OUT / "splits" / "test.csv"
# test.csv 的 sha256。原檔以 CRLF 分隔記錄、statement 欄內另有裸 LF，換行混用：
# git blob 是 CRLF→LF 正規化後的內容，乾淨 checkout 的位元組因此隨 core.autocrlf 而不同
# （LF checkout 即 blob；autocrlf=true 的新 clone 會把裸 LF 也轉成 CRLF），但解析出的資料相同。
# 各輸出與 manifest 一律記錄原檔的 TEST_SHA256；核對時以 LF 正規化後的內容比對 TEST_SHA256_LF。
TEST_SHA256 = "f5ec6983dec361b01ba1ee2e1700ce7c2e59234deea349a58261fc03751bbd09"
TEST_SHA256_LF = "5c798730ea9d79d11030ae8447a214b2a87bbae710f6dcea27ba6585e2a7437d"
N_TEST = 1998
SEED = 42
N_BOOT = 10000

# 顯示名稱（#21 命名表）。v1 的 rag_aug_* 與 v2 的 rag_augv2_* 用同一個名稱。
DISPLAY_NAMES = {
    "norag_base": "無檢索・基礎 prompt",
    "rag_noaug_base": "原文檢索・基礎 prompt",
    "rag_aug_base": "原文＋擴寫檢索・基礎 prompt",
    "rag_augv2_base": "原文＋擴寫檢索・基礎 prompt",
    "rag_noaug_optimized": "原文檢索・最佳化 prompt",
    "rag_aug_optimized": "原文＋擴寫檢索・最佳化 prompt",
    "rag_augv2_optimized": "原文＋擴寫檢索・最佳化 prompt",
}

# v1 的代號（論文 Table VI）。模組層級的 CODES／DESCRIPTIONS／PAIRS 都是 v1 的值，
# 供尚未參數化的 21_、22_、24_ 使用；17_～20_、23_ 一律從 profile 取。
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


def source_27() -> tuple:
    """從 27_ 原始碼讀出 CONDITIONS 字面值（import 27_ 會連帶載入 15_ 並探測 Ollama）。"""
    path = Path(__file__).resolve().parent / "27_main_eval_augv2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "CONDITIONS"):
            return ast.literal_eval(node.value)
    C.die(f"無法從 {path.name} 讀出 CONDITIONS")


# ---------------------------------------------------------------- profile

@dataclass(frozen=True)
class Profile:
    name: str                                   # v1 / v2
    experiment: str                             # 逐筆預測的 experiment 欄
    in_dir: Path                                # 逐筆預測所在目錄
    codes: dict                                 # 代號 → 條件 ID
    conditions: dict                            # 條件 ID → (15_ prompt 名, 語料或 None)
    corpora: tuple                              # RAG 語料，依 base_rag 條件的順序
    top_k: int
    paper_check: bool                           # 只有 v1 與論文 R1 稿 Table VI／VII 比對
    source_script: str
    metrics_split_sha_key: str
    descriptions: dict                          # 代號 → 英文描述（17_ 的 description 欄）
    manifest: dict

    @property
    def code_of(self) -> dict:
        return {cond: code for code, cond in self.codes.items()}

    @property
    def pairs(self) -> list:
        return list(combinations(self.codes, 2))

    def condition(self, code: str) -> str:
        return self.codes[code]

    def description(self, code: str) -> str:
        return self.descriptions[code]

    def display(self, code: str) -> str:
        return DISPLAY_NAMES[self.codes[code]]

    def label(self, code: str) -> str:
        """「C3　原文＋擴寫檢索・基礎 prompt」：代號與顯示名稱並列。"""
        return f"{code}　{self.display(code)}"

    def corpus(self, code: str) -> str | None:
        return self.conditions[self.codes[code]][1]

    @property
    def rag_codes(self) -> list:
        return [code for code in self.codes if self.corpus(code)]

    def display_names(self) -> dict:
        return {code: {"condition": cond, "display_name": DISPLAY_NAMES[cond]}
                for code, cond in self.codes.items()}


def _v1_profile() -> Profile:
    _, conditions = source_15()
    cond_map = {cond: (prompt, corpus) for cond, prompt, corpus in conditions}
    if set(cond_map) != set(CODES.values()):
        C.die(f"15_ 的條件與 C1–C5 對照不符：{sorted(cond_map)}")
    return Profile(
        name="v1", experiment="CUSTOM_PROMPT_EVAL", in_dir=V1_OUT, codes=dict(CODES),
        conditions=cond_map, corpora=("noaug", "aug"), top_k=1, paper_check=True,
        source_script="src/15_nested_custom_prompt_eval.py",
        metrics_split_sha_key="test_split_sha256", descriptions=dict(DESCRIPTIONS),
        manifest=C.load_json(V1_OUT / "custom_prompt_eval_manifest.json"),
    )


V2_CODES = {
    "C1": "norag_base",
    "C2": "rag_noaug_base",
    "C3": "rag_augv2_base",
    "C4": "rag_noaug_optimized",
    "C5": "rag_augv2_optimized",
}


def v2_descriptions(k: int) -> dict:
    return {
        "C1": "LLM only",
        "C2": f"Llama 3.1 & RAG (top-{k})",
        "C3": f"Llama 3.1 & RAG & filtered augmentation aug_v2 (top-{k})",
        "C4": f"Llama 3.1 & RAG (top-{k}), prompt optimized, without augmentation",
        "C5": f"Llama 3.1 & RAG (top-{k}), prompt optimized, filtered augmentation aug_v2",
    }


def _v2_profile() -> Profile:
    conditions = source_27()
    cond_map = {cond: (prompt, corpus) for cond, prompt, corpus, _ in conditions}
    if [c[0] for c in conditions] != list(V2_CODES.values()):
        C.die(f"27_ 的條件與 v2 的 C1–C5 對照不符：{[c[0] for c in conditions]}")
    for cond, _, _, display in conditions:
        if DISPLAY_NAMES[cond] != display:
            C.die(f"{cond} 的顯示名稱與 27_ 不同：{DISPLAY_NAMES[cond]} vs {display}")
    manifest = C.load_json(V2_OUT / "manifest.json")
    if not manifest:
        C.die(f"找不到 v2 manifest：{V2_OUT / 'manifest.json'}")
    recorded = [(c["condition"], c["prompt"], c["corpus"], c["display_name"])
                for c in manifest["conditions"]]
    if recorded != [tuple(c) for c in conditions]:
        C.die("v2 manifest 的 conditions 與 27_ 的 CONDITIONS 不同")
    if manifest["split"]["name"] != "test" or manifest.get("smoke_n") is not None:
        C.die("v2 manifest 不是 test 的完整執行")
    return Profile(
        name="v2", experiment="MAIN_EVAL_AUGV2", in_dir=V2_OUT, codes=dict(V2_CODES),
        conditions=cond_map, corpora=("noaug", "aug_v2"), top_k=int(manifest["top_k"]),
        paper_check=False, source_script="src/27_main_eval_augv2.py",
        metrics_split_sha_key="split_sha256", descriptions=v2_descriptions(int(manifest["top_k"])),
        manifest=manifest,
    )


@lru_cache(maxsize=None)
def get_profile(name: str = "v1") -> Profile:
    if name == "v1":
        return _v1_profile()
    if name == "v2":
        return _v2_profile()
    C.die(f"未知的 profile：{name}")


def add_profile_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--profile", choices=("v1", "v2"), default="v1",
                    help="v1＝15_（CUSTOM_PROMPT_EVAL/，凍結）；v2＝27_（MAIN_EVAL_AUGV2/）")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="輸出目錄（v2 預設 MAIN_EVAL_AUGV2/；v1 必須指定，且不能在 "
                         "CUSTOM_PROMPT_EVAL/ 之內）")


def resolve(args: argparse.Namespace) -> tuple[Profile, Path]:
    """回傳 (profile, 輸出目錄)。v1 的既有輸出是凍結的存檔，不能寫回去。"""
    profile = get_profile(args.profile)
    if args.out_dir is None:
        if profile.name == "v1":
            C.die("v1（CUSTOM_PROMPT_EVAL/）是凍結的存檔：--profile v1 必須以 --out-dir "
                  "指定其他輸出目錄")
        out = profile.in_dir
    else:
        out = Path(args.out_dir).resolve()
    frozen = V1_OUT.resolve()
    if profile.name == "v1" and (out == frozen or frozen in out.parents):
        C.die(f"--out-dir {out} 位於凍結的 v1 目錄 {frozen} 之內")
    out.mkdir(parents=True, exist_ok=True)
    return profile, out


def test_split_sha256() -> str:
    """核對磁碟上的 test.csv（容許 checkout 造成的換行差異），回傳記錄用的原檔 sha256。"""
    raw = TEST_PATH.read_bytes()
    disk = hashlib.sha256(raw).hexdigest()
    lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if disk != TEST_SHA256 and lf != TEST_SHA256_LF:
        C.die(f"{TEST_PATH} 的 sha256 不符（磁碟 {disk}，LF 正規化 {lf}）")
    return TEST_SHA256


def load_test() -> pd.DataFrame:
    test = pd.read_csv(TEST_PATH)
    if len(test) != N_TEST:
        C.die(f"test 筆數 {len(test)} ≠ {N_TEST}")
    return test


def load_predictions(profile: Profile | None = None) -> dict[str, list[dict]]:
    """回傳 {代號: 依 test.csv 順序排列的逐筆預測}，並做來源檢查（預設 v1）。"""
    profile = profile or get_profile("v1")
    prompts, _ = source_15()

    test = load_test()
    test_sha = test_split_sha256()
    order = [str(x) for x in test["id"]]
    truth = dict(zip(order, test["status"]))

    if profile.name == "v2":
        completion = C.load_json(profile.in_dir / "completion.json")
        if (completion.get("n") != N_TEST or completion.get("smoke_n") is not None
                or completion.get("split") != "test"
                or completion.get("top_k") != profile.top_k
                or completion.get("conditions") != list(profile.codes.values())):
            C.die(f"{profile.in_dir / 'completion.json'} 不是 test 五組條件的完整執行")

    out = {}
    for code, cond in profile.codes.items():
        prompt_name, corpus = profile.conditions[cond]
        path = profile.in_dir / f"predictions_{cond}.jsonl"
        records = list(C.read_jsonl(path))
        by_id = {str(r["id"]): r for r in records}
        expected_sha = C.sha256_text(prompts[prompt_name]["template"])
        problems = []
        if len(records) != N_TEST or len(by_id) != N_TEST or set(by_id) != set(order):
            problems.append(f"筆數或 id 集合不符（{len(records)} 筆）")
        if any(r.get("experiment") != profile.experiment for r in records):
            problems.append(f"experiment 欄不是 {profile.experiment}")
        if any(r.get("condition") != cond for r in records):
            problems.append("condition 欄與檔名不符")
        if any(r.get("prompt_sha256") != expected_sha for r in records):
            problems.append("prompt_sha256 與現行 15_ 的 template 不符")
        if any(r.get("corpus") != corpus for r in records):
            problems.append(f"corpus 欄與 {Path(profile.source_script).name} 的設定不符")
        if len({r.get("model_digest") for r in records}) != 1:
            problems.append("model_digest 不只一種")
        if profile.name == "v2":
            if profile.manifest["prompts"][prompt_name]["sha256"] != expected_sha:
                problems.append("v2 manifest 記錄的 prompt sha256 與 15_ 的 template 不符")
            if any(r.get("split") != "test" for r in records):
                problems.append("split 欄不是 test")
            expected_k = profile.top_k if corpus else None
            if any(r.get("top_k") != expected_k for r in records):
                problems.append(f"top_k 欄不是 {expected_k}")
        if not problems and any(truth[i] != by_id[i]["true_label"] for i in order):
            problems.append("true_label 與 test.csv 不符")
        metrics = C.load_json(profile.in_dir / f"metrics_{cond}.json")
        if metrics.get(profile.metrics_split_sha_key) != test_sha:
            problems.append("metrics 記錄的 test split sha256 與現行 test.csv 不符")
        if problems:
            C.die(f"{path.name} 來源檢查失敗：" + "；".join(problems))
        out[code] = [by_id[i] for i in order]
    return out


def predictions_sha256(profile: Profile) -> dict[str, str]:
    return {code: C.sha256_file(profile.in_dir / f"predictions_{cond}.jsonl")
            for code, cond in profile.codes.items()}


def check_corpus_docs(name: str, meta: dict) -> dict:
    """docs 的 sha 必須與 meta 相同，接受「磁碟 sha」或「LF 正規化 sha」其一（同 27_）。

    在 core.autocrlf=true 下，aug_v2_docs.json 會 checkout 成 CRLF；CRLF 只出現在 JSON 的
    縮排換行，不影響解析出來的文件字串。
    """
    raw = (BASE_OUT / "corpus" / f"{name}_docs.json").read_bytes()
    disk = hashlib.sha256(raw).hexdigest()
    lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if meta["docs_sha256"] == disk:
        match = "disk"
    elif meta["docs_sha256"] == lf:
        match = "lf_normalized"
    else:
        C.die(f"{name}_docs.json 的 sha256 與 meta 不符（磁碟 {disk}，LF 正規化 {lf}）")
    return {"docs_sha256_disk": disk, "docs_sha256_lf_normalized": lf,
            "docs_sha256_match": match}


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
