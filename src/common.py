"""共用工具。

設計原則（每一條都是為了修掉原始實驗的某個具體缺陷）：

1. 標籤解析嚴格且可稽核 —— 原始碼用 `for label in LABELS: if label.lower() in result.lower()`
   取第一個命中，"This is not Normal; it is Depression" 會被判成 Normal。這裡改為
   「唯一命中才算數，多重/零命中一律 INVALID 並記錄原因」。
2. 每次呼叫的原始回應、token 數、耗時全部落地 —— 原始實驗沒存逐筆預測，導致配對統計
   在數學上無法補算。
3. LLM 參數全部明確指定 —— 原始碼從未設定 num_ctx，吃 Ollama 預設值。
4. 檔案 hash 驗證 —— 每支腳本開場先確認自己吃到的資料沒被換過。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------- 路徑

ROOT = Path(os.environ.get("MHC_ROOT", Path(__file__).resolve().parent.parent))
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
RUNS = Path(os.environ.get("MHC_RUNS", ROOT / "runs"))

RAW_DATASET = DATA_RAW / "Combined Data.csv"
# 來源：Kaggle "Sentiment Analysis for Mental Health"（Combined Data.csv）
# 本 repo 於 2026-09-15 自舊 repo C:\Users\wuc120\Code\csv\ 原封複製，未經任何前處理。
RAW_DATASET_SHA256 = "67274f52dec3b0155aec4d7d1ff31d2a92b0e9b383ab917614591cd168cc2474"

# ---------------------------------------------------------------- 標籤

# 本研究沿用原論文的四類設定。原始資料集另有 Suicidal / Stress /
# Personality disorder 三類（共 27% 的資料）未納入，論文需明確交代此一取捨。
LABELS = ["Normal", "Depression", "Anxiety", "Bipolar"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

SPLITS = ["train", "val_search", "val_confirm", "test"]


# ---------------------------------------------------------------- 輸出

def log(*parts: Any) -> None:
    """印一行訊息。Windows 主控台 cp950 印不出某些符號時退而求其次，不讓它中斷腳本。"""
    msg = " ".join(str(p) for p in parts)
    stamp = time.strftime("%H:%M:%S")
    try:
        print(f"[{stamp}] {msg}", flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(f"[{stamp}] {msg.encode(enc, 'replace').decode(enc)}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    log("!! " + msg)
    raise SystemExit(1)


# ---------------------------------------------------------------- 檔案 / hash

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_hash(path: Path, expected: str, what: str) -> None:
    """確認檔案就是我們以為的那一份。切分檔、原始資料都要過這關。"""
    if not path.exists():
        die(f"{what} 不存在：{path}")
    actual = sha256_file(path)
    if actual != expected:
        die(f"{what} 的 SHA256 不符\n  路徑：{path}\n  預期：{expected}\n  實際：{actual}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- JSONL 落地 / 中斷續跑

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_ids(path: Path, key: str = "id") -> set:
    """已經跑完的樣本 id，用來中斷續跑。"""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


# ---------------------------------------------------------------- 標籤解析

_LABEL_WORD_RE = {
    label: re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE) for label in LABELS
}
_DIGIT_RE = re.compile(r"\d+")


def parse_label(raw: str) -> tuple[int | None, str]:
    """把 LLM 的原始回應解析成類別 id。

    回傳 (label_id, reason)。label_id 為 None 代表 INVALID，reason 說明原因，
    兩者都會寫進逐筆預測檔供事後稽核。

    規則（依序）：
      1. 整段回應就是一個有效數字        -> 接受
      2. 回應中恰好出現一個有效數字      -> 接受（容許 "Answer: 2"）
      3. 出現多個不同的有效數字          -> INVALID(multi_number)
      4. 沒有數字時退而比對標籤文字，
         恰好唯一命中一個標籤            -> 接受
      5. 多重命中 / 零命中               -> INVALID

    第 4、5 條是原始實驗的致命傷所在：原本依清單順序取第一個命中，
    "not Normal; it is Depression" 會被判成 Normal。
    """
    if raw is None:
        return None, "empty"
    text = raw.strip()
    if not text:
        return None, "empty"

    # 1 + 2 + 3：數字模式（本研究的指令要求模型只回覆數字）
    numbers = {int(n) for n in _DIGIT_RE.findall(text)}
    valid_numbers = {n for n in numbers if n in ID_TO_LABEL}
    if len(valid_numbers) == 1 and len(numbers) == 1:
        return next(iter(valid_numbers)), "number"
    if len(valid_numbers) > 1:
        return None, "multi_number"
    if len(numbers) > len(valid_numbers) and not valid_numbers:
        return None, "out_of_range_number"

    # 4 + 5：文字標籤模式（模型不聽話、改用文字作答時的後備）
    hits = [label for label, rx in _LABEL_WORD_RE.items() if rx.search(text)]
    if len(hits) == 1:
        return LABEL_TO_ID[hits[0]], "unique_label_word"
    if len(hits) > 1:
        return None, "multi_label"
    if valid_numbers:  # 數字有效但混了其它數字（例如 "2 (out of 4)"）
        return None, "ambiguous_number"
    return None, "no_label"


# ---------------------------------------------------------------- LLM 呼叫

# 生成參數全部明確指定，不吃任何預設值；這組設定會被 00_probe_env.py 落地，
# 並由 11_verify_all.py 斷言所有步驟之間一致。
GEN_OPTIONS = {
    "temperature": 0.0,
    "seed": 42,
    "num_ctx": 4096,     # 實測資料 p99=884 tokens，最長 5,857；4096 可容納 99.4%
    "num_predict": 16,   # 只要一個數字，不需要長輸出
    "top_p": 1.0,
}

MODEL_NAME = os.environ.get("MHC_MODEL", "llama3.1")


def chat(prompt: str, model: str | None = None, options: dict | None = None) -> dict:
    """呼叫一次 LLM，回傳原始回應與計量資訊。

    不做任何解析或後處理 —— 解析交給 parse_label()，且原始回應一定落地。
    """
    import ollama

    opts = dict(GEN_OPTIONS)
    if options:
        opts.update(options)

    t0 = time.perf_counter()
    resp = ollama.chat(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=opts,
    )
    elapsed = time.perf_counter() - t0

    return {
        "raw_response": resp["message"]["content"],
        "elapsed_s": elapsed,
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
    }


# ---------------------------------------------------------------- 匯入防護

def import_faiss():
    """確認 import 到的是 faiss 套件本身。

    舊 repo 根目錄有一個叫 faiss/ 的資料夾，會把套件名稱遮蔽成 namespace package，
    導致 read_index 不存在。新 repo 沒有這個資料夾，但保留這道檢查以防日後誤建。
    """
    import faiss  # type: ignore

    if not hasattr(faiss, "IndexFlatIP"):
        die(
            "import 到的 faiss 不是套件本身（可能被同名資料夾遮蔽）：\n"
            f"  faiss.__file__ = {getattr(faiss, '__file__', '<namespace package>')}"
        )
    return faiss
