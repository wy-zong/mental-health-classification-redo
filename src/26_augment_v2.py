"""Issue #1：在限制條件下重做實驗 15 的擴寫，並建立防洩漏的 RAG 資料庫。

背景
  實驗 15 的 RAG 語料（corpus/aug_docs.json）有 11,101 筆改寫，分成兩批：
  * MAIN 批：04_augment.py 對舊 train 5,996 筆各生成 2 次，共 11,992 筆；
  * 巢狀批：13_ 對新加入 train 的 998 筆各生成 2 次，共 1,996 筆。
  PR #17 的量測發現，這份語料沒有套用論文 3.E 的過濾，還混進 Llama 的拒答。另外，04_／12_ 的
  開場白規則含有 `I've`／`I have`，因此有些改寫的真正第一段被刪掉了。val／test 原文完全沒有
  改寫，無法支援重複切分（#4）。

做法（決策記錄在 issue #1）
  * 生成 prompt 與參數完全沿用實驗 15：REWRITE_INSTRUCTION（不含標籤）＋原文，
    temperature 0.7、num_ctx 4096、num_predict 1024。改寫沿用原文的標籤。
  * 對象是實驗 15 的 train 與 test 原文，共 8,992 筆；val 不擴寫。
  * 實驗 15 的 13,988 筆原始生成都當作 train 的候選，從原始回應重新處理。
  * 清理時不刪除任何句子，只去掉頭尾空白，以及包住整段的引號或反引號。
  * 依序過濾：空字串、拒答、改寫說明句（開場白、結尾說明、對使用者說話）、與原文相同、
    長度比 0.5–2.0、SBERT ≥ 門檻、與已接受的改寫重複。不做 LLM 重分類。
  * SBERT 門檻先用 0.6，只在 pilot 檢查一次：任一類別「新生成 5 次後仍補不滿」的原文
    比例 > 5% 時，改用 0.5（底線）。門檻鎖定在 sbert_checkpoint.json。
  * 每筆原文依固定的候選順序，取前 N 筆通過的。不足就換 seed 新生成，條件不放寬。
    新生成上限第一輪為 5 次（pilot 檢查點也以 5 次判定），之後提高到 10 次，
    只對仍補不滿的原文續生成；調整紀錄寫在 run_config.json 的 max_new_attempts_history。

防洩漏
  每筆改寫只依賴自己的原文：prompt 只含該筆文本，seed 只依候選序號決定，規則只比較改寫與
  自己的原文，以及已接受的改寫集合。因此整份結果與切分無關，任何切分都只是挑出子集：
      RAG 語料 = 該切分 train 的原文 + source_id 屬於該切分 train 的改寫（select_for_split）
  verify 階段以實驗 15 的切分，以及 5 組固定 val、train／test 分層重切的切分實際檢查。

階段
  --check-only       核對輸入、prompt、模型 digest，不生成
  --stage reuse      評估實驗 15 的 13,988 筆候選（不需要 LLM）
  --smoke            每類 3 筆原文、新生成上限 3 次，跑完所有階段（輸出到 AUGMENT_V2/_smoke）
  --stage pilot      SBERT 門檻檢查點：每類 50 筆原文以 0.6 跑完，決定並鎖定門檻（只做一次）
  --stage topup      為不足的原文新生成；可續跑，到 --time-budget-h 就不再開始新的生成
  --stage finalize   依固定順序選定改寫，寫出結果檔
  --stage corpus     依實驗 15 切分建立 RAG 資料庫 corpus/aug_v2.*
  --stage verify     檢查不變條件、語料與切分洩漏

輸出：runs/NESTED_70_10_20_INCREMENTAL/AUGMENT_V2/、corpus/aug_v2.{index,_docs.json,_meta.json}
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common as C  # noqa: E402

R = C.load_module("12_rebuild_experiment")
N13 = C.load_module("13_nested_incremental_rag")

RUN = "NESTED_70_10_20_INCREMENTAL"
BASE = C.RUNS / RUN
SPLITS = BASE / "splits"
CORPUS_DIR = BASE / "corpus"
MANIFEST_15 = BASE / "CUSTOM_PROMPT_EVAL" / "custom_prompt_eval_manifest.json"
AUG_DOCS = CORPUS_DIR / "aug_docs.json"
MAIN_RAW = C.RUNS / "MAIN" / "augment_raw.jsonl"
MAIN_CSV = C.RUNS / "MAIN" / "augmented.csv"
MAIN_AUDIT = C.RUNS / "MAIN" / "augment_audit.json"
NESTED_RAW = BASE / "augment_raw.jsonl"
NESTED_CSV = BASE / "augmented_new.csv"
NESTED_META = BASE / "augmented_meta.json"
OUT_ROOT = BASE / "AUGMENT_V2"
CORPUS_NAME = "aug_v2"

N_TARGET = R.N_AUG_PER_SOURCE            # 2，與 12_／13_ 相同
MAX_NEW_ATTEMPTS = 10                    # 第一輪為 5（PILOT_MAX_NEW），之後提高
PILOT_MAX_NEW = 5
NEW_SEED_BASE = 20000                    # 舊 seed 為 42／43（MAIN）與 1042／1043（巢狀）
LENGTH_RATIO = (0.5, 2.0)                # 含端點，字元數，改寫／原文
SBERT_PRIMARY = 0.60
SBERT_FLOOR = 0.50
TARGET_SPLITS = ("train", "test")        # val 不擴寫
N_POOL = 9992
N_PER_LABEL = 2498
N_TARGETS = 8992
N_MAIN_RAW = 11992
N_NESTED_RAW = 1996
N_15_PARAPHRASES = 11101
PILOT_PER_LABEL = 50
PILOT_SEED = 2026
PILOT_MAX_SHORTFALL_RATE = 0.05
RESPLIT_SEEDS = (0, 1, 2, 3, 4)
SMOKE_PER_LABEL_ROLES = ("train_legacy", "train_added", "test_nested")
SMOKE_MAX_NEW = 3

# 依檢查順序，記錄第一個不通過的規則。sbert 的門檻在選取時才套用。
RULE_ORDER = ("empty", "refusal", "meta_commentary", "identical_to_source", "length_ratio", "sbert")
PRE_SBERT_RULES = RULE_ORDER[:-1]

# 拒答偵測：在實驗 15 的 13,988 筆原始回應上校準過。
# * 開頭句型：Llama 的拒答幾乎都以「I can't / cannot … <動詞>」開頭，原文不會這樣開頭。
#   刻意不收「I won't …」與單獨的「I can't help」：前者在原文裡常是真實內容（I won't drink that…），
#   後者會誤中「I can't help but feel…」。
# * 安全訊息：改寫加入了原文沒有的求助資源。刻意不收單獨的 lifeline／hotline（常是比喻或原文內容）。
_A = "['’]"
REFUSAL_OPENING_RE = re.compile(
    rf"^(?:(?:i{_A}?m sorry|i apologize|i{_A}?m afraid),?\s*(?:but\s*)?)?(?:i{_A}?m not sure\s+)?"
    rf"i\s*(?:can{_A}?t|cannot|can not|am unable to|{_A}m unable to|am not able to|{_A}m not able to|"
    rf"won{_A}?t be able to)\s+"
    r"(?:write|rewrite|fulfill|provide|create|generate|process|rephrase|paraphrase|carry out|support|"
    r"complete|engage|comply|accommodate|produce|reproduce|modify|edit|translate|respond|answer|"
    r"(?:help|assist)\s+(?:with|you))\b",
    re.IGNORECASE,
)
SAFETY_MESSAGE_RE = re.compile(
    r"(suicide prevention|suicide (?:and|&) crisis|crisis (?:text )?line|crisis hotline|crisis lifeline|"
    r"1-800-273|741741|\b988\b|text home to|samaritans|qualified mental health professional|"
    r"seek help from a|is there (?:anything|something) else i can help)",
    re.IGNORECASE,
)

# 改寫說明句：含有就整筆淘汰，不刪句子。只要原文本身有同類用語，該條就不適用。
# * 開場白：多行回覆的第一行談「改寫」本身，且以冒號結尾或很短（"Here's a rewritten version of the text:"）
# * 內嵌開場白：同一行先有開場白再接正文（"Here's a rewritten version: I feel…"）
# * 結尾說明：最後一行或 Note 行，以 Note／I've kept… 等開頭，且談到改寫或原文的語氣、意義
# * 對使用者說話：提到「the text you provided」「text to rewrite」「me to rewrite」等
# 刻意不以 `I've`／`I have`／`Here's my` 本身為依據：那常是改寫的真正內容。
META_WORD_RE = re.compile(
    r"(re-?writ|rephras|paraphras|re-?word|version of (?:the|this|your) (?:text|post|passage|poem|message|statement)"
    r"|another way to (?:say|put|phrase)|in different words)",
    re.IGNORECASE,
)
META_INLINE_RE = re.compile(
    rf"^(?:[^\n]{{0,100}}?[.!]\s+)?(?:here{_A}?s|here is|below is)\b[^:\n]{{0,120}}?"
    r"(?:re-?writ|rephras|paraphras|re-?word|version)[^:\n]{0,80}:\s*\S",
    re.IGNORECASE,
)
META_TRAILING_LEAD_RE = re.compile(
    rf"^[(\[*_\s]*(?:note\b|please note|let me know|i hope (?:this|that|you)|feel free|"
    rf"i{_A}?ve (?:tried|aimed|attempted|kept|maintained|preserved|rewritten|rephrased|changed|made|used|"
    r"removed|added|replaced|condensed)|i (?:tried|aimed|attempted|kept|maintained|preserved|rewrote|rephrased|"
    r"changed|made|used|removed|added|replaced|condensed)|in (?:this|the|my) (?:rewritten|revised|rephrased)|"
    r"th(?:is|e) (?:rewritten|revised|rephrased) (?:text|version))",
    re.IGNORECASE,
)
META_NOTE_LINE_RE = re.compile(r"^[(\[*_\s]*(?:note\b|please note)", re.IGNORECASE)
META_TRAILING_WORD_RE = re.compile(
    r"(re-?writ|rephras|paraphras|re-?word|original (?:text|tone|meaning|message|content|post|wording|intent)"
    r"|emotional (?:tone|content)|the (?:same )?(?:tone|meaning) of the)",
    re.IGNORECASE,
)
META_TASK_RE = re.compile(
    r"((?:re-?writ\w*|rephras\w*|paraphras\w*|re-?word\w*) (?:the|this|your|that) (?:original |provided |given )?"
    r"(?:text|statement|passage|message)\b"
    r"|\b(?:a|the|my) (?:rewritten|rephrased|revised|paraphrased) (?:text|version)\b"
    r"|\b(?:original|provided|given) text\b"
    rf"|\btext (?:you(?:{_A}ve| have)? provided|to re-?write|for me to)\b"
    r"|\bme to (?:re-?write|rephrase|paraphrase)\b"
    r"|^rewritten (?:text|version)\s*:)",
    re.IGNORECASE | re.MULTILINE,
)
OPENING_MAX_LEN = 160

USAGE_RULE = (
    "任何切分的 RAG 語料 = 該切分 train 的原文 + accepted.csv 中 source_id 屬於該切分 train 的改寫；"
    "val／test 原文的改寫在該切分中一律不用（select_for_split）。val 原文沒有改寫，重複切分時 val 固定，"
    "只重新切分 train＋test。改寫只依賴自己的原文，所以結果與切分無關。"
)


# ---------------------------------------------------------------- 小工具

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jdefault(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=jdefault)
    os.replace(tmp, path)


def str_keys(d: dict) -> dict:
    """value_counts().to_dict() 的鍵是 numpy 整數，json 不接受。"""
    return {str(k): int(v) for k, v in d.items()}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(C.ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_csv_text(path: Path) -> pd.DataFrame:
    """文本欄一律以字串讀入；pandas 預設會把 "NA"、"null" 之類的貼文變成 NaN。"""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean_candidate(raw: str) -> str:
    """不刪任何句子：只去頭尾空白，以及包住整段的成對引號／反引號。"""
    text = str(raw or "").strip()
    pairs = {'"': '"', "'": "'", "`": "`", "“": "”"}
    while len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        text = text[1:-1].strip()
    return text


def refusal_type(candidate: str, source: str) -> str | None:
    src = str(source).strip()
    if REFUSAL_OPENING_RE.match(candidate) and not REFUSAL_OPENING_RE.match(src):
        return "refusal_opening"
    if SAFETY_MESSAGE_RE.search(candidate) and not SAFETY_MESSAGE_RE.search(src):
        return "safety_message_added"
    return None


def meta_commentary_type(candidate: str, source: str) -> str | None:
    lines = [x.strip() for x in candidate.splitlines() if x.strip()]
    if not lines:
        return None
    src_norm = R.normalize_text(source)
    src_has_meta = bool(META_WORD_RE.search(source))

    def in_source(line: str) -> bool:
        return R.normalize_text(line) in src_norm

    first = lines[0]
    if (len(lines) > 1 and not src_has_meta and not in_source(first) and META_WORD_RE.search(first)
            and (first.endswith(":") or len(first) < OPENING_MAX_LEN)):
        return "opening"
    if META_INLINE_RE.match(first) and not src_has_meta and not in_source(first.split(":")[0]):
        return "inline_opening"
    if not META_TRAILING_WORD_RE.search(source):
        for k, line in enumerate(lines[1:], start=1):
            if ((k == len(lines) - 1 or META_NOTE_LINE_RE.match(line)) and META_TRAILING_LEAD_RE.match(line)
                    and META_TRAILING_WORD_RE.search(line) and not in_source(line)):
                return "trailing_note"
    if not src_has_meta and META_TASK_RE.search(candidate) and not META_TASK_RE.search(source):
        return "addresses_user"
    return None


def old_15_cleaning(origin: str, raw: str) -> str:
    """實驗 15 當時的清理（MAIN：04_／12_ strip_boilerplate；巢狀：13_ clean_rewrite），只用於來源核對。"""
    return R.strip_boilerplate(raw)[0] if origin == "main" else N13.clean_rewrite(raw)


def effective_options(options: dict) -> dict:
    """C.chat 會把 options 疊在 GEN_OPTIONS 上；這裡算出實際送給 Ollama 的參數。"""
    return {**C.GEN_OPTIONS, **options}


# Ollama 在模型陷入重複迴圈時中止生成。曾發生同一請求連續重試都失敗、之後重跑又正常的情況；
# 重試用完仍是這個錯誤時，記為一次失敗的嘗試，不中斷整個 topup
REPEAT_LIMIT_RE = re.compile(r"token repeat limit", re.IGNORECASE)


class GenerationAborted(Exception):
    pass


def chat_retry(prompt: str, options: dict, tries: int = 6) -> dict:
    delay = 5
    for k in range(tries):
        try:
            return C.chat(prompt, options=options)
        except Exception as exc:  # noqa: BLE001
            if k == tries - 1:
                if REPEAT_LIMIT_RE.search(str(exc)):
                    raise GenerationAborted(str(exc)) from exc
                raise
            C.log(f"  Ollama 呼叫失敗（{type(exc).__name__}: {exc}），{delay} 秒後重試")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def ollama_version() -> str | None:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001
        return None


def gpu_info() -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()] or None
    except Exception:  # noqa: BLE001
        return None


class Log:
    """以 candidate_id 為鍵的 append-only JSONL；續跑時載入，最後一行不完整就截掉。"""

    def __init__(self, path: Path, key: str = "candidate_id"):
        self.path = path
        self.key = key
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        self._fh = None
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = self.path.read_bytes()
        lines = raw.decode("utf-8").split("\n")
        good: list[str] = []
        dropped = 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if any(x.strip() for x in lines[i + 1:]):
                    C.die(f"{self.path} 第 {i + 1} 行不是合法 JSON（不是最後一行，無法自動修復）")
                dropped += 1
                continue
            if rec[self.key] in self.data:
                continue
            self.data[rec[self.key]] = rec
            good.append(line)
        if dropped or not raw.endswith(b"\n"):
            # 中斷時寫到一半的最後一行：重寫檔案，避免下一筆接在半行後面
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(x.rstrip("\r") + "\n" for x in good))
            os.replace(tmp, self.path)
            if dropped:
                C.log(f"  {self.path.name}：截掉 {dropped} 行不完整的紀錄")

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def append(self, rec: dict) -> None:
        with self.lock:
            if rec[self.key] in self.data:
                return
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=jdefault) + "\n")
            self._fh.flush()
            self.data[rec[self.key]] = rec

    def close(self) -> None:
        with self.lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ---------------------------------------------------------------- 實驗 15 的紀錄

def ast_literals(path: Path, names: set[str]) -> dict:
    """從原始碼讀出字面值。直接 import 15_ 會在載入時探測 Ollama。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in names:
                found[name] = ast.literal_eval(node.value)
    if set(found) != names:
        C.die(f"無法從 {path.name} 讀出 {sorted(names - set(found))}")
    return found


def load_records_15() -> dict:
    """實驗 15 的改寫 prompt、參數與模型身分；任何一項與紀錄不符就停止。"""
    man = C.load_json(MANIFEST_15)
    if not man:
        C.die(f"找不到 15_ manifest：{MANIFEST_15}")

    # 改寫 prompt 與參數：04_、12_／13_、MAIN audit、巢狀 meta 四處必須一致
    a04 = ast_literals(SRC / "04_augment.py", {"REWRITE_INSTRUCTION", "REWRITE_OPTIONS"})
    audit = C.load_json(MAIN_AUDIT)
    meta = C.load_json(NESTED_META)
    checks = {
        "04_ instruction": a04["REWRITE_INSTRUCTION"] == R.REWRITE_INSTRUCTION,
        "04_ options": a04["REWRITE_OPTIONS"] == R.REWRITE_OPTIONS,
        "MAIN audit instruction": audit.get("instruction") == R.REWRITE_INSTRUCTION,
        "MAIN audit options": audit.get("options") == R.REWRITE_OPTIONS,
        "MAIN audit no class names": audit.get("instruction_mentions_class_names") is False,
        "nested meta instruction": meta.get("rewrite_instruction") == R.REWRITE_INSTRUCTION,
        "nested meta instruction sha": meta.get("rewrite_instruction_sha256") == C.sha256_text(R.REWRITE_INSTRUCTION),
        "nested meta options": meta.get("rewrite_options") == R.REWRITE_OPTIONS,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        C.die(f"改寫 prompt／參數與實驗 15 的紀錄不一致：{bad}")
    if any(lab.lower() in R.REWRITE_INSTRUCTION.lower() for lab in C.LABELS):
        C.die("改寫指令含類別名稱，與實驗 15 不符")
    return {"manifest": man, "manifest_sha256": C.sha256_file(MANIFEST_15), "model": man["model"]}


def check_model(rec15: dict) -> dict:
    identity = R.probe_llm_identity()
    want = rec15["model"]
    if identity.get("digest") is None:
        C.die("無法取得 Ollama 模型 digest：Ollama 沒有啟動，或沒有 llama3.1")
    if identity["digest"] != want["digest"]:
        C.die(f"模型 digest {identity['digest']} 與 15_ manifest {want['digest']} 不符")
    if identity.get("weights_blob_digest") not in (None, want.get("weights_blob_digest")):
        C.die(f"權重 blob {identity['weights_blob_digest']} 與 15_ manifest 不符")
    identity["ollama_version"] = ollama_version()
    return identity


def load_pool(rec15: dict) -> pd.DataFrame:
    """實驗 15 的完整資料：nested train／val／test，共 9,992 筆（val 只用於相似度與 verify）。"""
    sm = C.load_json(SPLITS / "split_manifest.json")
    frames = []
    for name in ("train", "val", "test"):
        path = SPLITS / f"{name}.csv"
        C.require_hash(path, sm["files"][name]["sha256"], f"nested {name}.csv")
        df = read_csv_text(path)
        if len(df) != sm["files"][name]["rows"]:
            C.die(f"{name}.csv 筆數 {len(df)} ≠ {sm['files'][name]['rows']}")
        frames.append(df)
    C.require_hash(SPLITS / "test.csv", rec15["manifest"]["test"]["sha256"], "15_ test.csv")

    pool = pd.concat(frames, ignore_index=True)
    pool["label_id"] = pool["label_id"].astype(int)
    if len(pool) != N_POOL or pool["source_id"].duplicated().any():
        C.die(f"pool 筆數 {len(pool)} 或 source_id 不唯一")
    counts = pool["status"].value_counts().to_dict()
    if counts != {lab: N_PER_LABEL for lab in C.LABELS}:
        C.die(f"pool 類別數不符：{counts}")
    if (pool["label_id"] != pool["status"].map(C.LABEL_TO_ID)).any():
        C.die("label_id 與 status 不一致")
    if (pool["statement"].str.strip() == "").any():
        C.die("pool 有空白原文")
    norm = pool["statement"].map(R.normalize_text)
    if norm.duplicated().any():
        C.die("pool 有正規化後相同的原文")
    pool["source_uid"] = pool["statement"].map(R.stable_source_id)
    if pool["source_uid"].duplicated().any():
        C.die("source_uid 碰撞")
    pool = pool.rename(columns={"new_split": "exp15_split", "nested_role": "exp15_role"})
    pool = pool[["source_id", "source_uid", "statement", "status", "label_id",
                 "exp15_split", "exp15_role", "old_split"]]
    return pool.sort_values("source_uid", kind="mergesort").reset_index(drop=True)


def load_reused(pool: pd.DataFrame) -> tuple[dict[str, list[dict]], dict]:
    """實驗 15 的 13,988 筆原始生成，並確認它們就是產生 15_ 語料的那一份。"""
    by_sid = pool.set_index("source_id")
    role = by_sid["exp15_role"].to_dict()
    cands: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()

    def add(c: dict) -> None:
        if c["candidate_id"] in seen:
            C.die(f"候選重複：{c['candidate_id']}")
        sid = c["source_id"]
        if sid not in role:
            C.die(f"候選的原文不在 pool：{sid}")
        if c["label"] != by_sid.at[sid, "status"]:
            C.die(f"候選標籤與原文不符：{c['candidate_id']}")
        if R.normalize_text(c["original"]) != R.normalize_text(by_sid.at[sid, "statement"]):
            C.die(f"候選的原文與 pool 不符：{c['candidate_id']}")
        seen.add(c["candidate_id"])
        cands[sid].append(c)

    opts = R.REWRITE_OPTIONS
    main = list(C.read_jsonl(MAIN_RAW))
    if len(main) != N_MAIN_RAW:
        C.die(f"MAIN 原始生成 {len(main)} 筆 ≠ {N_MAIN_RAW}")
    for r in main:
        k = int(r["variant"])
        if role.get(r["source_id"]) != "train_legacy":
            C.die(f"MAIN 候選的原文不是 train_legacy：{r['source_id']}")
        add({
            "candidate_id": f"{r['source_id']}#m{k}", "source_id": r["source_id"], "origin": "main",
            "code": f"m{k}", "order": k, "seed": int(opts["seed"]) + k, "label": r["status"],
            "original": r["original"], "raw_response": r["raw_response"],
            "generation_options": effective_options({**opts, "seed": int(opts["seed"]) + k}),
            "elapsed_s": float(r["elapsed_s"]), "eval_count": int(r["eval_count"]),
            "source_file": "runs/MAIN/augment_raw.jsonl", "corpus_id_15": f"aug_{r['source_id']}_{k}",
        })

    nested = list(C.read_jsonl(NESTED_RAW))
    if len(nested) != N_NESTED_RAW:
        C.die(f"巢狀原始生成 {len(nested)} 筆 ≠ {N_NESTED_RAW}")
    for r in nested:
        slot, attempt = int(r["slot"]), int(r["attempt"])
        if attempt != 1:
            C.die(f"巢狀候選 attempt ≠ 1：{r['gen_key']}")
        if role.get(r["source_id"]) != "train_added":
            C.die(f"巢狀候選的原文不是 train_added：{r['source_id']}")
        if r["prompt"] != f"{R.REWRITE_INSTRUCTION}\n\nText:\n{str(r['original']).strip()}":
            C.die(f"巢狀候選的 prompt 與實驗 15 指令不符：{r['gen_key']}")
        add({
            "candidate_id": f"{r['source_id']}#n{slot}", "source_id": r["source_id"], "origin": "nested",
            "code": f"n{slot}", "order": 10 + slot, "seed": int(r["generation_options"]["seed"]),
            "label": r["status"], "original": r["original"], "raw_response": r["raw_response"],
            "generation_options": effective_options(r["generation_options"]),
            "elapsed_s": float(r["elapsed_s"]), "eval_count": r.get("eval_count"),
            "source_file": f"runs/{RUN}/augment_raw.jsonl", "corpus_id_15": f"aug_{r['source_id']}_{slot}",
        })
    for sid in cands:
        cands[sid].sort(key=lambda c: c["order"])

    # 來源核對：原始回應依當時的舊規則清理，必須逐字重現兩份改寫 CSV 與 aug_docs.json
    main_csv = read_csv_text(MAIN_CSV)
    nested_csv = read_csv_text(NESTED_CSV)
    C.require_hash(MAIN_CSV, C.load_json(MAIN_AUDIT)["output_sha256"], "MAIN augmented.csv")
    C.require_hash(NESTED_CSV, C.load_json(NESTED_META)["output_sha256"], "巢狀 augmented_new.csv")
    raw_of = {c["corpus_id_15"]: c for cs in cands.values() for c in cs}
    bad = [row.id for df in (main_csv, nested_csv) for row in df.itertuples()
           if old_15_cleaning(raw_of[row.id]["origin"], raw_of[row.id]["raw_response"]) != row.statement]
    if bad:
        C.die(f"原始生成無法以當時規則重現 15_ 的改寫：{len(bad)} 筆，例如 {bad[:3]}")

    C.require_hash(AUG_DOCS, C.load_json(MANIFEST_15)["corpora"]["aug"]["docs_sha256_actual"], "15_ aug_docs.json")
    payload = C.load_json(AUG_DOCS)
    corpus_aug = {i: d for i, d in zip(payload["doc_ids"], payload["docs"]) if i.startswith("aug_")}
    if len(corpus_aug) != N_15_PARAPHRASES:
        C.die(f"15_ 語料改寫 {len(corpus_aug)} 筆 ≠ {N_15_PARAPHRASES}")
    text_15 = {}
    for df in (main_csv, nested_csv):
        for row in df.itertuples():
            if corpus_aug.get(row.id) != R.corpus_entry(row.statement, row.status):
                C.die(f"15_ 語料與改寫 CSV 不符：{row.id}")
            text_15[row.id] = row.statement
    if len(text_15) != N_15_PARAPHRASES:
        C.die("15_ 語料與兩份改寫 CSV 的筆數不符")
    for cs in cands.values():
        for c in cs:
            c["in_15_corpus"] = c["corpus_id_15"] in text_15
            c["text_15"] = text_15.get(c["corpus_id_15"])

    stats = {
        "main_raw": len(main), "nested_raw": len(nested),
        "main_kept_15": len(main_csv), "nested_kept_15": len(nested_csv),
        "sources_with_reused": len(cands),
        "inputs_sha256": {rel(p): C.sha256_file(p)
                          for p in (MAIN_RAW, MAIN_CSV, MAIN_AUDIT, NESTED_RAW, NESTED_CSV, NESTED_META,
                                    AUG_DOCS, MANIFEST_15, SPLITS / "train.csv", SPLITS / "val.csv",
                                    SPLITS / "test.csv", SPLITS / "split_manifest.json")},
    }
    return dict(cands), stats


# ---------------------------------------------------------------- 執行環境

class Embedder:
    """12_ 的 embedder（all-MiniLM-L6-v2、CPU、max_seq 512、正規化），與 22_ 的 SBERT 一致。"""

    def __init__(self):
        self.model, self._encode = R.embedder()
        self.lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            return self._encode(list(texts))


class Ctx:
    def __init__(self, out: Path, n_target: int, max_new: int, targets: list[str] | None,
                 need_llm: bool = True, need_embed: bool = True, corpus_dir: Path = CORPUS_DIR):
        self.out = out
        self.n_target = n_target
        self.max_new = max_new
        self.corpus_dir = corpus_dir
        self.rec15 = load_records_15()
        self.pool = load_pool(self.rec15)
        self.cands_reused, self.reuse_stats = load_reused(self.pool)
        self.idx = {sid: i for i, sid in enumerate(self.pool["source_id"])}
        self.statement = self.pool["statement"].tolist()
        self.label = self.pool["status"].tolist()
        self.split15 = self.pool["exp15_split"].tolist()
        self.norm_orig = {R.normalize_text(t): sid for t, sid in zip(self.statement, self.pool["source_id"])}
        # 工作對象：實驗 15 的 train＋test，依 source_uid 排序（與切分無關）
        all_targets = [s for s, sp in zip(self.pool["source_id"], self.split15) if sp in TARGET_SPLITS]
        if len(all_targets) != N_TARGETS:
            C.die(f"train＋test 原文 {len(all_targets)} 筆 ≠ {N_TARGETS}")
        self.targets = all_targets if targets is None else targets
        if any(self.split15[self.idx[s]] not in TARGET_SPLITS for s in self.targets):
            C.die("工作對象含 val 原文")
        self.model = check_model(self.rec15) if need_llm else None
        self.emb = Embedder() if need_embed else None
        self.orig_emb = self._original_embeddings() if need_embed else None
        self.gens = Log(out / "generations.jsonl")
        self.checks = Log(out / "checks.jsonl")
        self.accept_lock = threading.Lock()
        self.accepted_texts: set[str] = set()

    def _original_embeddings(self) -> np.ndarray:
        key = C.sha256_text("\u0000".join(self.statement))[:16]
        path = OUT_ROOT / "cache" / f"orig_emb_{key}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape == (len(self.statement), 384):
                return arr
        C.log(f"編碼 {len(self.statement):,} 筆原文（CPU）…")
        arr = self.emb.encode(self.statement)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr

    # SBERT 門檻：pilot 決定後鎖定在 sbert_checkpoint.json
    def checkpoint(self) -> dict | None:
        return C.load_json(self.out / "sbert_checkpoint.json") or None

    def locked_sbert(self) -> float:
        cp = self.checkpoint()
        if not cp:
            C.die("SBERT 門檻尚未鎖定：請先跑 --stage pilot")
        return float(cp["locked_sbert_min"])

    def config(self) -> dict:
        return {
            "n_target": self.n_target,
            "max_new_attempts": self.max_new,
            "target_splits": list(TARGET_SPLITS),
            "new_seed_rule": f"seed = {NEW_SEED_BASE} + j（j = 0..max_new_attempts-1，與原文和切分無關）",
            "rewrite_instruction": R.REWRITE_INSTRUCTION,
            "rewrite_instruction_sha256": C.sha256_text(R.REWRITE_INSTRUCTION),
            "rewrite_prompt_format": "{instruction}\\n\\nText:\\n{statement.strip()}",
            "rewrite_options": R.REWRITE_OPTIONS,
            "rewrite_options_effective": effective_options(R.REWRITE_OPTIONS),
            "clean_rule": "只去頭尾空白與包住整段的成對引號／反引號；不刪任何句子",
            "rule_order": list(RULE_ORDER) + ["duplicate"],
            "refusal_opening_regex": REFUSAL_OPENING_RE.pattern,
            "safety_message_regex": SAFETY_MESSAGE_RE.pattern,
            "meta_word_regex": META_WORD_RE.pattern,
            "meta_inline_regex": META_INLINE_RE.pattern,
            "meta_trailing_lead_regex": META_TRAILING_LEAD_RE.pattern,
            "meta_trailing_word_regex": META_TRAILING_WORD_RE.pattern,
            "meta_task_regex": META_TASK_RE.pattern,
            "meta_opening_max_len": OPENING_MAX_LEN,
            "length_ratio": list(LENGTH_RATIO),
            "sbert_threshold_rule": {
                "primary": SBERT_PRIMARY, "floor": SBERT_FLOOR,
                "pilot_per_label": PILOT_PER_LABEL, "pilot_seed": PILOT_SEED,
                "switch_if_any_label_shortfall_rate_gt": PILOT_MAX_SHORTFALL_RATE,
            },
            "llm_reclassification": False,
            "embed_model": R.EMBED_MODEL, "embed_max_seq": R.EMBED_MAX_SEQ, "embed_device": "cpu",
            "model_digest": self.rec15["model"]["digest"],
            "targets_sha256": C.sha256_text("\n".join(self.targets)),
            "n_targets": len(self.targets),
        }

    def ensure_config(self) -> None:
        path = self.out / "run_config.json"
        cfg = self.config()
        old = C.load_json(path)
        if old:
            history = old.pop("max_new_attempts_history", [])
            if old.get("max_new_attempts") is not None and cfg["max_new_attempts"] > old["max_new_attempts"]:
                # 只允許提高新生成上限：已生成的候選序號與 seed 不變，只對補不滿的原文續生成
                history = history + [{"from": old["max_new_attempts"], "to": cfg["max_new_attempts"],
                                       "changed_at": now()}]
                C.log(f"新生成上限由 {old['max_new_attempts']} 提高到 {cfg['max_new_attempts']}，記入 run_config.json")
                old["max_new_attempts"] = cfg["max_new_attempts"]
                save_json(path, {**old, "max_new_attempts_history": history})
            diff = sorted(k for k in set(cfg) | set(old) if cfg.get(k) != old.get(k))
            if diff:
                C.die(f"設定與既有輸出不一致：{diff}\n  既有輸出：{self.out}\n"
                      "  請改用新的輸出目錄，或還原設定")
        else:
            save_json(path, cfg)

    def close(self) -> None:
        for log in (self.gens, self.checks):
            log.close()


# ---------------------------------------------------------------- 檢查、生成、選取

def check_candidates(ctx: Ctx, cands: list[dict]) -> list[dict]:
    texts = [clean_candidate(c["raw_response"]) for c in cands]
    nonempty = [i for i, t in enumerate(texts) if t]
    vecs = ctx.emb.encode([texts[i] for i in nonempty]) if nonempty else np.empty((0, 384), "float32")
    row_of = {i: k for k, i in enumerate(nonempty)}
    sims_all = vecs @ ctx.orig_emb.T if len(nonempty) else np.empty((0, len(ctx.statement)))
    out = []
    for i, (c, text) in enumerate(zip(cands, texts)):
        sid = c["source_id"]
        j = ctx.idx[sid]
        src = ctx.statement[j]
        rec = {
            "candidate_id": c["candidate_id"], "source_id": sid, "origin": c["origin"],
            "cleaned": text, "changed_by_cleaning": text != str(c["raw_response"]).strip(),
            "len_source": len(src), "len_candidate": len(text),
            "refusal": None, "meta_commentary": None, "identical_to_source": False,
            "equals_other_original": None, "length_ratio": None, "sbert": None,
            "max_other_cos": None, "max_other_source": None,
        }
        fails = {k: False for k in PRE_SBERT_RULES}
        fails["empty"] = not text
        if text:
            norm = R.normalize_text(text)
            rec["refusal"] = refusal_type(text, src)
            rec["meta_commentary"] = meta_commentary_type(text, src)
            rec["identical_to_source"] = norm == R.normalize_text(src)
            other = ctx.norm_orig.get(norm)
            rec["equals_other_original"] = other if other not in (None, sid) else None
            rec["length_ratio"] = len(text) / len(src)
            sims = sims_all[row_of[i]].copy()
            rec["sbert"] = float(sims[j])
            sims[j] = -np.inf
            k = int(np.argmax(sims))
            rec["max_other_cos"] = float(sims[k])
            rec["max_other_source"] = ctx.pool.at[k, "source_id"]
            fails["refusal"] = rec["refusal"] is not None
            fails["meta_commentary"] = rec["meta_commentary"] is not None
            fails["identical_to_source"] = rec["identical_to_source"]
            fails["length_ratio"] = not (LENGTH_RATIO[0] <= rec["length_ratio"] <= LENGTH_RATIO[1])
        rec["rule_fail"] = {k: bool(v) for k, v in fails.items()}
        rec["pre_sbert_pass"] = not any(fails.values())
        out.append(rec)
    return out


def first_fail(ch: dict, thr: float) -> str | None:
    for k in PRE_SBERT_RULES:
        if ch["rule_fail"][k]:
            return k
    return "sbert" if ch["sbert"] < thr else None


def passes(ch: dict | None, thr: float) -> bool:
    return ch is not None and ch["pre_sbert_pass"] and ch["sbert"] >= thr


def generate(ctx: Ctx, sid: str, j: int) -> dict:
    src = ctx.statement[ctx.idx[sid]]
    seed = NEW_SEED_BASE + j
    prompt = f"{R.REWRITE_INSTRUCTION}\n\nText:\n{str(src).strip()}"
    options = {**R.REWRITE_OPTIONS, "seed": seed}
    try:
        resp = chat_retry(prompt, options)
        error = None
    except GenerationAborted as exc:
        # 記為一次失敗的嘗試：回應留空（被 empty 規則淘汰），錯誤訊息存檔
        C.log(f"  {sid} 第 {j} 次生成被 Ollama 中止（{exc}），記為失敗的嘗試")
        resp = {"raw_response": "", "elapsed_s": None, "prompt_eval_count": None, "eval_count": None}
        error = str(exc)
    eff = effective_options(options)
    return {
        "candidate_id": f"{sid}#g{j:02d}", "source_id": sid, "origin": "new", "code": f"g{j:02d}",
        "order": 100 + j, "attempt": j, "seed": seed, "label": ctx.label[ctx.idx[sid]],
        "raw_response": resp["raw_response"], "prompt_sha256": C.sha256_text(prompt),
        "generation_options": eff, "elapsed_s": resp["elapsed_s"],
        "prompt_eval_count": resp["prompt_eval_count"], "eval_count": resp["eval_count"],
        "hit_num_predict": (resp["eval_count"] or 0) >= eff["num_predict"],
        "model_digest": ctx.model["digest"], "created_at": now(), "generation_error": error,
    }


def candidates_of(ctx: Ctx, sid: str) -> list[dict]:
    out = list(ctx.cands_reused.get(sid, []))
    for j in range(ctx.max_new):
        g = ctx.gens.get(f"{sid}#g{j:02d}")
        if g is not None:
            out.append(g)
    return out


def select(ctx: Ctx, thr: float) -> tuple[dict[str, list[str]], dict[str, str]]:
    """依固定順序（source_uid、候選順序）取前 N 筆通過的改寫。

    只看已評估完的候選。回傳 ({source_id: [candidate_id]}, {被判重複的 candidate_id: 先前的 candidate_id})。
    """
    accepted: dict[str, list[str]] = {}
    dup_of: dict[str, str] = {}
    seen: dict[str, str] = {}
    for sid in ctx.targets:
        acc: list[str] = []
        for c in candidates_of(ctx, sid):
            if len(acc) >= ctx.n_target:
                break
            ch = ctx.checks.get(c["candidate_id"])
            if not passes(ch, thr):
                continue
            key = R.normalize_text(ch["cleaned"])
            if key in seen:
                dup_of[c["candidate_id"]] = seen[key]
                continue
            seen[key] = c["candidate_id"]
            acc.append(c["candidate_id"])
        accepted[sid] = acc
    return accepted, dup_of


def select_for_split(accepted: pd.DataFrame, train_source_ids) -> pd.DataFrame:
    """某個切分可用的改寫：只取原文在該切分 train 的。"""
    train = set(train_source_ids)
    return accepted[accepted["source_id"].isin(train)]


# ---------------------------------------------------------------- 統計

def funnel(ctx: Ctx, thr: float, accepted: dict[str, list[str]], dup_of: dict[str, str],
           sids: list[str]) -> dict:
    acc_set = {cid for v in accepted.values() for cid in v}
    rows = []
    for sid in sids:
        lab = ctx.label[ctx.idx[sid]]
        for c in candidates_of(ctx, sid):
            cid = c["candidate_id"]
            ch = ctx.checks.get(cid)
            if ch is None:
                status = "not_evaluated"
            elif first_fail(ch, thr):
                status = first_fail(ch, thr)
            elif cid in dup_of:
                status = "duplicate"
            elif cid in acc_set:
                status = "accepted"
            else:
                status = "surplus"      # 通過，但該原文已補滿（排在後面的候選）
            row = {"label": lab, "origin": c["origin"], "status": status,
                   "in_15_corpus": bool(c.get("in_15_corpus", False))}
            if ch:
                row |= {f"fail_{k}": v for k, v in ch["rule_fail"].items()}
                row["fail_sbert"] = ch["sbert"] is not None and ch["sbert"] < thr
                row["meta_type"] = ch.get("meta_commentary")
                row["refusal_type"] = ch.get("refusal")
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    out = {"sbert_min": thr, "first_fail_by_origin_label": {}, "independent_rule_fail_rate": {},
           "exp15_corpus_paraphrases_first_fail": {}, "meta_commentary_types": {}, "refusal_types": {}}
    for (origin, lab), g in df.groupby(["origin", "label"]):
        out["first_fail_by_origin_label"][f"{origin}/{lab}"] = {"n": len(g), **g["status"].value_counts().to_dict()}
    for origin, g in df.groupby("origin"):
        out["first_fail_by_origin_label"][f"{origin}/ALL"] = {"n": len(g), **g["status"].value_counts().to_dict()}
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    ev_all = df.dropna(subset=["fail_empty"]) if "fail_empty" in df else df
    for lab, g in list(ev_all.groupby("label")) + [("ALL", ev_all)]:
        out["independent_rule_fail_rate"][lab] = {
            c[5:]: round(float(g[c].fillna(False).astype(bool).mean()), 4) for c in fail_cols
        } | {"n_evaluated": int(len(g))}
    in15 = df[df["in_15_corpus"]]
    for lab, g in list(in15.groupby("label")) + [("ALL", in15)]:
        out["exp15_corpus_paraphrases_first_fail"][lab] = {"n": len(g), **g["status"].value_counts().to_dict()}
    if "meta_type" in df:
        out["meta_commentary_types"] = df["meta_type"].dropna().value_counts().to_dict()
        out["refusal_types"] = df["refusal_type"].dropna().value_counts().to_dict()
    return out


def coverage(ctx: Ctx, accepted: dict[str, list[str]], sids: list[str]) -> dict:
    df = pd.DataFrame({
        "sid": sids,
        "label": [ctx.label[ctx.idx[s]] for s in sids],
        "split": [ctx.split15[ctx.idx[s]] for s in sids],
        "n_acc": [len(accepted.get(s, [])) for s in sids],
        "n_new": [sum(1 for c in candidates_of(ctx, s) if c["origin"] == "new") for s in sids],
    })
    df["filled"] = df["n_acc"] >= ctx.n_target
    out = {"sources": len(df), "filled": int(df["filled"].sum()),
           "accepted": int(df["n_acc"].sum()), "target_total": len(df) * ctx.n_target,
           "by_label": {}, "by_exp15_split": {}}
    for key, col in (("by_label", "label"), ("by_exp15_split", "split")):
        for v, g in df.groupby(col):
            out[key][v] = {"sources": len(g), "filled": int(g["filled"].sum()),
                           "accepted": int(g["n_acc"].sum()),
                           "accepted_count_dist": str_keys(g["n_acc"].value_counts().sort_index().to_dict())}
    out["exhausted_unfilled"] = int(((~df["filled"]) & (df["n_new"] >= ctx.max_new)).sum())
    out["unfilled_with_attempts_left"] = int(((~df["filled"]) & (df["n_new"] < ctx.max_new)).sum())
    return out


def projection(ctx: Ctx, accepted: dict[str, list[str]], fun: dict) -> dict:
    """以沿用候選的整體通過率粗估還要生成幾次（假設同類別通過率相同）。"""
    rates = {}
    for lab in C.LABELS:
        n = acc = 0
        for origin in ("main", "nested"):
            d = fun.get("first_fail_by_origin_label", {}).get(f"{origin}/{lab}", {})
            n += d.get("n", 0)
            acc += d.get("accepted", 0) + d.get("surplus", 0)
        rates[lab] = acc / n if n else None
    est = {}
    for lab in C.LABELS:
        p = rates[lab]
        gens = unfilled = 0.0
        for sid in ctx.targets:
            if ctx.label[ctx.idx[sid]] != lab:
                continue
            need = ctx.n_target - len(accepted.get(sid, []))
            if need <= 0 or p is None:
                continue
            gens += ctx.max_new if p == 0 else min(ctx.max_new, need / p)
            # 新生成上限內湊不到 need 筆的機率（二項分布）
            unfilled += sum(math.comb(ctx.max_new, k) * p ** k * (1 - p) ** (ctx.max_new - k) for k in range(need))
        est[lab] = {"all_filter_pass_rate_reused": round(p, 4) if p is not None else None,
                    "expected_new_generations": int(round(gens)),
                    "expected_unfilled_sources_if_homogeneous": round(unfilled, 1)}
    total = sum(v["expected_new_generations"] for v in est.values())
    return {"sbert_min": fun.get("sbert_min"), "by_label": est, "expected_new_generations_total": total,
            "note": "假設同類別原文的通過率相同；實際上難改寫的原文集中在少數，缺額通常會比這個估計多"}


def exp15_cleaning_damage(ctx: Ctx) -> dict:
    """15_ 語料中，舊規則把哪些改寫的第一行刪掉了；新偵測不認為那是開場白的，就是刪到內容。"""
    out = Counter()
    examples = []
    for cs in ctx.cands_reused.values():
        for c in cs:
            if not c["in_15_corpus"]:
                continue
            raw_lines = [x for x in str(c["raw_response"]).strip().splitlines() if x.strip()]
            if c["text_15"].strip() == "\n".join(raw_lines).strip() or len(raw_lines) < 2:
                continue
            dropped_first = R.normalize_text(c["text_15"]) == R.normalize_text("\n".join(raw_lines[1:]))
            if not dropped_first:
                continue
            out[f"{c['origin']}/first_line_dropped"] += 1
            if meta_commentary_type(clean_candidate(c["raw_response"]), c["original"]) is None:
                out[f"{c['origin']}/content_dropped"] += 1
                if len(examples) < 10:
                    examples.append({"candidate_id": c["candidate_id"], "dropped_line": raw_lines[0][:200]})
    return {"counts": dict(out), "content_dropped_total": sum(v for k, v in out.items() if k.endswith("content_dropped")),
            "examples": examples,
            "note": "content_dropped：15_ 的清理刪掉第一行，但那一行不是改寫說明句（多為 04_／12_ 規則中 I've／I have 誤判）"}


def write_session(ctx: Ctx, stage: str, started: str, extra: dict) -> None:
    rec = {"stage": stage, "started_at": started, "ended_at": now(), "host": platform.node(),
           "platform": platform.platform(), "gpu": gpu_info(),
           "ollama_version": ctx.model.get("ollama_version") if ctx.model else None, **extra}
    ctx.out.mkdir(parents=True, exist_ok=True)
    with open(ctx.out / "sessions.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=jdefault) + "\n")


# ---------------------------------------------------------------- 階段

def reuse_incomplete(ctx: Ctx) -> int:
    return sum(1 for sid in ctx.targets for c in ctx.cands_reused.get(sid, [])
               if c["candidate_id"] not in ctx.checks)


def rule_samples(ctx: Ctx, k: int = 15) -> dict:
    """給人工確認的抽樣：各種拒答／說明句命中，以及差一點命中的邊界案例。"""
    rng = np.random.default_rng(0)
    groups: dict[str, list[dict]] = defaultdict(list)
    for ch in ctx.checks.data.values():
        text = ch["cleaned"]
        if not text:
            continue
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if ch.get("refusal"):
            groups[f"hit/{ch['refusal']}"].append({"id": ch["candidate_id"], "text": text[:220]})
        elif ch.get("meta_commentary"):
            t = ch["meta_commentary"]
            line = lines[0] if t in ("opening", "inline_opening") else (
                lines[-1] if t == "trailing_note" else META_TASK_RE.search(text).group(0) + " ‖ " + text[:200])
            groups[f"hit/{t}"].append({"id": ch["candidate_id"], "text": line[:240]})
        else:
            if META_WORD_RE.search(text):
                groups["near_miss/meta_word_anywhere"].append({"id": ch["candidate_id"], "text": text[:220]})
            if len(lines) > 1 and lines[0].endswith(":"):
                groups["near_miss/first_line_colon"].append({"id": ch["candidate_id"], "text": lines[0][:200]})
            if re.match(r"^(i{_A}?ve|i have|here{_A}?s my)\b".format(_A=_A), lines[0], re.IGNORECASE):
                groups["kept/starts_with_ive_or_heres_my"].append({"id": ch["candidate_id"], "text": lines[0][:160]})
    out = {}
    for name, items in sorted(groups.items()):
        pick = sorted(rng.choice(len(items), size=min(k, len(items)), replace=False).tolist())
        out[name] = {"n": len(items), "sample": [items[i] for i in pick]}
    return out


def stage_reuse(ctx: Ctx) -> None:
    started = now()
    cands = [c for sid in ctx.targets for c in ctx.cands_reused.get(sid, [])]
    todo = [c for c in cands if c["candidate_id"] not in ctx.checks]
    C.log(f"沿用候選 {len(cands):,} 筆，尚未檢查 {len(todo):,} 筆")
    for i in range(0, len(todo), 2000):
        for rec in check_candidates(ctx, todo[i:i + 2000]):
            ctx.checks.append(rec)
        C.log(f"  規則檢查 {min(i + 2000, len(todo)):,}/{len(todo):,}")

    report = {"created_at": now(), "by_threshold": {}}
    for thr in (SBERT_PRIMARY, SBERT_FLOOR):
        accepted, dup_of = select(ctx, thr)
        fun = funnel(ctx, thr, accepted, dup_of, ctx.targets)
        cov = coverage(ctx, accepted, ctx.targets)
        proj = projection(ctx, accepted, fun)
        report["by_threshold"][f"{thr:.2f}"] = {"funnel": fun, "coverage_after_reuse": cov, "projection": proj}
        C.log(f"SBERT ≥ {thr:.2f}：沿用後 {cov['filled']:,}/{cov['sources']:,} 筆原文已補滿 {ctx.n_target} 筆，"
              f"共接受 {cov['accepted']:,}/{cov['target_total']:,} 筆；預估還需新生成約 "
              f"{proj['expected_new_generations_total']:,} 次")
        for lab, v in proj["by_label"].items():
            C.log(f"    {lab:<10} 沿用候選通過率 {v['all_filter_pass_rate_reused']}，還需生成約 "
                  f"{v['expected_new_generations']:,} 次，補不滿約 {v['expected_unfilled_sources_if_homogeneous']} 筆")
    report["exp15_cleaning_damage"] = exp15_cleaning_damage(ctx)
    report["rule_samples"] = rule_samples(ctx)
    save_json(ctx.out / "reuse_report.json", report)
    fun = report["by_threshold"][f"{SBERT_PRIMARY:.2f}"]["funnel"]
    C.log(f"說明句類型：{fun.get('meta_commentary_types')}；拒答類型：{fun.get('refusal_types')}")
    C.log(f"15_ 語料中被舊清理刪到內容的改寫：{report['exp15_cleaning_damage']['content_dropped_total']} 筆")
    write_session(ctx, "reuse", started, {})


def topup(ctx: Ctx, thr: float, sids: list[str], workers: int, budget_h: float, stage: str) -> dict:
    """為 sids 中不足 N 筆的原文依序新生成（seed = NEW_SEED_BASE + j），直到補滿或用完上限。"""
    started = now()
    deadline = time.time() + budget_h * 3600
    missing_reuse = reuse_incomplete(ctx)
    if missing_reuse:
        C.die(f"還有 {missing_reuse} 筆沿用候選沒有檢查，請先跑 --stage reuse")
    accepted, _ = select(ctx, thr)
    ctx.accepted_texts = {R.normalize_text(ctx.checks.get(cid)["cleaned"]) for v in accepted.values() for cid in v}
    work = []
    for sid in sids:
        cands = candidates_of(ctx, sid)
        n_new = sum(1 for c in cands if c["origin"] == "new")
        pending = any(c["candidate_id"] not in ctx.checks for c in cands)
        if (len(accepted[sid]) < ctx.n_target and n_new < ctx.max_new) or pending:
            work.append(sid)
    C.log(f"{stage}：SBERT ≥ {thr:.2f}，需要處理 {len(work):,}/{len(sids):,} 筆原文"
          f"（workers={workers}，時間上限 {budget_h:g} 小時）")

    stop = threading.Event()
    counter = Counter()
    clock = threading.Lock()

    def worker(sid: str) -> str:
        acc = len(accepted[sid])
        for j in range(ctx.max_new):
            if acc >= ctx.n_target:
                return "filled"
            cid = f"{sid}#g{j:02d}"
            gen, ch = ctx.gens.get(cid), ctx.checks.get(cid)
            if gen is not None and ch is not None:
                continue                                  # 先前已評估完，已計入 select
            if gen is None:
                if stop.is_set() or time.time() > deadline:
                    return "stopped"
                gen = generate(ctx, sid, j)
                ctx.gens.append(gen)
                with clock:
                    counter["generated"] += 1
            if ch is None:
                ch = check_candidates(ctx, [gen])[0]
                ctx.checks.append(ch)
            if not passes(ch, thr):
                continue
            key = R.normalize_text(ch["cleaned"])
            with ctx.accept_lock:
                if key in ctx.accepted_texts:
                    continue
                ctx.accepted_texts.add(key)
            acc += 1
            with clock:
                counter["accepted"] += 1
        return "filled" if acc >= ctx.n_target else "exhausted"

    t0 = time.perf_counter()

    def monitor() -> None:
        while not stop.wait(300):
            el = (time.perf_counter() - t0) / 60
            with clock:
                g, a, f = counter["generated"], counter["accepted"], counter["filled"]
                ex_, done = counter["exhausted"], counter["done"]
            rate = g / max(el, 1e-9)
            C.log(f"  進度：完成原文 {done:,}/{len(work):,}（補滿 {f:,}、用完 {ex_:,}），新生成 {g:,} 次、"
                  f"接受 {a:,} 筆，{rate:.1f} 次生成/分，已 {el:.0f} 分")

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    results = Counter()
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(worker, sid) for sid in work]
        for fut in as_completed(futs):
            r = fut.result()
            results[r] += 1
            with clock:
                counter["done"] += 1
                counter[r] += 1
    except BaseException as exc:
        # 任何一個 worker 出錯或手動中斷：不再開始新的生成，等進行中的請求寫完再停
        C.log(f"停止 {stage}（{type(exc).__name__}）：等待進行中的請求結束；已完成的紀錄都已寫入")
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        ex.shutdown(wait=True)
        stop.set()
        el = (time.perf_counter() - t0) / 60
        accepted, _ = select(ctx, thr)
        cov = coverage(ctx, accepted, sids)
        C.log(f"{stage} 結束（{el:.0f} 分）：新生成 {counter['generated']:,} 次；"
              f"補滿 {cov['filled']:,}/{cov['sources']:,}，用完上限仍不足 {cov['exhausted_unfilled']:,}，"
              f"還可再生成的 {cov['unfilled_with_attempts_left']:,}")
        write_session(ctx, stage, started, {"workers": workers, "budget_h": budget_h, "sbert_min": thr,
                                            "minutes": round(el, 1), **counter,
                                            "results": dict(results), "coverage": cov})
    if results.get("stopped"):
        C.log(f"時間到，尚有原文未處理：重新執行同一個階段即可續跑")
    return cov


def pilot_targets(ctx: Ctx) -> list[str]:
    """每類從工作對象（依 source_uid 排序）以固定 seed 抽 PILOT_PER_LABEL 筆；與切分無關。"""
    rng = np.random.default_rng(PILOT_SEED)
    picked = set()
    for lab in C.LABELS:
        sids = [s for s in ctx.targets if ctx.label[ctx.idx[s]] == lab]
        n = min(PILOT_PER_LABEL, len(sids))
        picked |= {sids[i] for i in rng.choice(len(sids), size=n, replace=False)}
    return [s for s in ctx.targets if s in picked]


def pilot_stats(ctx: Ctx, thr: float, sids: list[str]) -> dict:
    accepted, _ = select(ctx, thr)
    out = {}
    for lab in C.LABELS:
        g = [s for s in sids if ctx.label[ctx.idx[s]] == lab]
        unfilled = [s for s in g if len(accepted[s]) < ctx.n_target]
        n_new = [sum(1 for c in candidates_of(ctx, s) if c["origin"] == "new") for s in g]
        out[lab] = {
            "sources": len(g), "unfilled": len(unfilled),
            "shortfall_rate": round(len(unfilled) / len(g), 4) if g else None,
            "by_exp15_split": {sp: {"sources": sum(ctx.split15[ctx.idx[s]] == sp for s in g),
                                    "unfilled": sum(ctx.split15[ctx.idx[s]] == sp for s in unfilled)}
                               for sp in TARGET_SPLITS},
            "unfilled_sources": unfilled,
            "new_generations": int(sum(n_new)),
        }
    return out


def stage_pilot(ctx: Ctx, workers: int, budget_h: float, sids: list[str] | None = None) -> None:
    cp = ctx.checkpoint()
    if cp:
        C.log(f"SBERT 門檻已鎖定為 {cp['locked_sbert_min']:.2f}（{cp['created_at']}），檢查點只做一次，不重跑")
        return
    started = now()
    sids = pilot_targets(ctx) if sids is None else sids
    full_max_new = ctx.max_new
    ctx.max_new = min(full_max_new, PILOT_MAX_NEW)     # 檢查點固定以 5 次判定
    try:
        topup(ctx, SBERT_PRIMARY, sids, workers, budget_h, "pilot")
        accepted, _ = select(ctx, SBERT_PRIMARY)
        incomplete = [s for s in sids if len(accepted[s]) < ctx.n_target and
                      sum(1 for c in candidates_of(ctx, s) if c["origin"] == "new") < ctx.max_new]
        if incomplete:
            C.die(f"pilot 尚未跑完（{len(incomplete)} 筆原文還可再生成）：請重新執行 --stage pilot")
        stats = {f"{thr:.2f}": pilot_stats(ctx, thr, sids) for thr in (SBERT_PRIMARY, SBERT_FLOOR)}
        pilot_max_new = ctx.max_new
    finally:
        ctx.max_new = full_max_new
    primary = stats[f"{SBERT_PRIMARY:.2f}"]
    too_strict = {lab: v["shortfall_rate"] for lab, v in primary.items()
                  if v["shortfall_rate"] > PILOT_MAX_SHORTFALL_RATE}
    locked = SBERT_FLOOR if too_strict else SBERT_PRIMARY
    cp = {
        "created_at": now(), "started_at": started,
        "rule": (f"每類抽 {PILOT_PER_LABEL} 筆原文，以 SBERT ≥ {SBERT_PRIMARY} 跑完（新生成上限 {pilot_max_new} 次）；"
                 f"任一類別補不滿 {ctx.n_target} 筆的原文比例 > {PILOT_MAX_SHORTFALL_RATE:.0%} 就改用 {SBERT_FLOOR}"
                 f"（底線，不再往下調）。只檢查一次。"),
        "pilot_seed": PILOT_SEED, "pilot_sources": sids,
        "pilot_sources_sha256": C.sha256_text("\n".join(sids)),
        "stats_by_threshold": stats,
        "labels_over_limit_at_primary": too_strict,
        "decision": "switch_to_floor" if too_strict else "keep_primary",
        "locked_sbert_min": locked,
        "note": "pilot 的候選就是正式候選（candidate_id 與 seed 相同）；生成與門檻無關，改用 0.5 只需重新套用門檻",
    }
    save_json(ctx.out / "sbert_checkpoint.json", cp)
    for thr, st in stats.items():
        C.log(f"  SBERT ≥ {thr}：" + "，".join(f"{lab} 缺額 {v['unfilled']}/{v['sources']}"
                                             f"（{v['shortfall_rate']:.0%}）" for lab, v in st.items()))
    C.log(f"檢查點判定：{'0.6 太嚴 ' + str(too_strict) + '，改用 ' + str(SBERT_FLOOR) if too_strict else '維持 0.6'}；"
          f"門檻鎖定為 {locked:.2f}")


def stage_topup(ctx: Ctx, workers: int, budget_h: float) -> None:
    topup(ctx, ctx.locked_sbert(), ctx.targets, workers, budget_h, "topup")


def stage_finalize(ctx: Ctx) -> None:
    started = now()
    thr = ctx.locked_sbert()
    pend_chk = [g for g in ctx.gens.data.values() if g["source_id"] in ctx.idx and g["candidate_id"] not in ctx.checks]
    for rec in check_candidates(ctx, pend_chk) if pend_chk else []:
        ctx.checks.append(rec)
    missing = reuse_incomplete(ctx)
    if missing:
        C.die(f"還有 {missing} 筆沿用候選沒有檢查，請先跑 --stage reuse")

    accepted, dup_of = select(ctx, thr)
    cov = coverage(ctx, accepted, ctx.targets)
    fun = funnel(ctx, thr, accepted, dup_of, ctx.targets)
    pool = ctx.pool.set_index("source_id")

    rows = []
    cand_by_id = {c["candidate_id"]: c for sid in ctx.targets for c in candidates_of(ctx, sid)}
    for sid in ctx.targets:
        for slot, cid in enumerate(accepted[sid]):
            c, ch = cand_by_id[cid], ctx.checks.get(cid)
            rows.append({
                "id": f"aug_{sid}_{c['code']}", "source_id": sid, "statement": ch["cleaned"],
                "status": ctx.label[ctx.idx[sid]], "label_id": int(pool.at[sid, "label_id"]),
                "source_uid": pool.at[sid, "source_uid"], "slot": slot, "candidate_id": cid,
                "origin": c["origin"], "seed": c["seed"], "length_ratio": round(ch["length_ratio"], 6),
                "sbert": round(ch["sbert"], 6), "max_other_cos": round(ch["max_other_cos"], 6),
                "max_other_source": ch["max_other_source"],
                "equals_other_original": ch["equals_other_original"] or "",
                "in_15_corpus": bool(c.get("in_15_corpus", False)),
                "same_text_as_15_corpus": c.get("text_15") == ch["cleaned"],
                "exp15_split": pool.at[sid, "exp15_split"], "exp15_role": pool.at[sid, "exp15_role"],
            })
    acc_df = pd.DataFrame(rows)
    acc_path = ctx.out / "accepted.csv"
    acc_df.to_csv(acc_path, index=False, encoding="utf-8", lineterminator="\n")

    st_rows = []
    for sid in ctx.targets:
        cs = candidates_of(ctx, sid)
        fails = Counter()
        for c in cs:
            ch = ctx.checks.get(c["candidate_id"])
            if ch and first_fail(ch, thr):
                fails[first_fail(ch, thr)] += 1
            elif c["candidate_id"] in dup_of:
                fails["duplicate"] += 1
        n_new = sum(1 for c in cs if c["origin"] == "new")
        st_rows.append({
            "source_id": sid, "source_uid": pool.at[sid, "source_uid"], "status": ctx.label[ctx.idx[sid]],
            "label_id": int(pool.at[sid, "label_id"]), "exp15_split": pool.at[sid, "exp15_split"],
            "exp15_role": pool.at[sid, "exp15_role"], "len_source": len(ctx.statement[ctx.idx[sid]]),
            "n_reused": len(cs) - n_new, "n_new_generated": n_new, "n_accepted": len(accepted[sid]),
            "filled": len(accepted[sid]) >= ctx.n_target,
            "shortfall": max(0, ctx.n_target - len(accepted[sid])),
            "fail_counts": json.dumps(dict(fails), ensure_ascii=False, sort_keys=True),
        })
    st_df = pd.DataFrame(st_rows)
    st_path = ctx.out / "source_status.csv"
    st_df.to_csv(st_path, index=False, encoding="utf-8", lineterminator="\n")

    # candidates.jsonl.gz：每筆候選一行（生成、清理、各關結果），依固定順序
    acc_ids = {cid for v in accepted.values() for cid in v}
    cand_path = ctx.out / "candidates.jsonl.gz"
    with gzip.open(cand_path, "wt", encoding="utf-8", newline="\n", compresslevel=9) as fh:
        for sid in ctx.targets:
            for c in candidates_of(ctx, sid):
                cid = c["candidate_id"]
                ch = ctx.checks.get(cid) or {}
                rec = {k: v for k, v in c.items() if k not in ("original", "order")}
                rec["checks"] = {k: v for k, v in ch.items() if k not in ("candidate_id", "source_id", "origin")}
                rec["first_fail"] = first_fail(ch, thr) if ch else None
                rec["selected"] = cid in acc_ids
                rec["duplicate_of"] = dup_of.get(cid)
                fh.write(json.dumps(rec, ensure_ascii=False, default=jdefault) + "\n")

    new_gens = list(ctx.gens.data.values())
    shortfall = st_df[~st_df["filled"]]
    cp = ctx.checkpoint()
    summary = {
        "created_at": now(), "n_target": ctx.n_target, "max_new_attempts": ctx.max_new,
        "max_new_attempts_history": (C.load_json(ctx.out / "run_config.json") or {}).get("max_new_attempts_history", []),
        "sbert_min": thr, "sbert_checkpoint_decision": cp["decision"],
        "coverage": cov, "funnel": fun,
        "shortfall_sources": shortfall[["source_id", "status", "exp15_split", "len_source",
                                        "n_accepted", "fail_counts"]].to_dict("records"),
        "shortfall_by_label": str_keys(shortfall.groupby("status")["shortfall"].sum().to_dict()),
        "accepted_by_origin": acc_df["origin"].value_counts().to_dict() if len(acc_df) else {},
        "accepted_from_15_corpus_same_text": int(acc_df["same_text_as_15_corpus"].sum()) if len(acc_df) else 0,
        "accepted_equal_to_other_original": int((acc_df["equals_other_original"] != "").sum()) if len(acc_df) else 0,
        "duplicates_rejected": len(dup_of),
        "exp15_cleaning_damage": exp15_cleaning_damage(ctx),
        "new_generation": {
            "n": len(new_gens),
            "attempts_per_source": str_keys(st_df["n_new_generated"].value_counts().sort_index().to_dict()),
            "generation_errors": Counter(g["generation_error"] for g in new_gens if g.get("generation_error")),
            "mean_elapsed_s": float(np.mean([g["elapsed_s"] for g in new_gens if g["elapsed_s"] is not None]))
            if new_gens else None,
            "mean_eval_count": float(np.mean([g["eval_count"] or 0 for g in new_gens])) if new_gens else None,
            "hit_num_predict": sum(bool(g["hit_num_predict"]) for g in new_gens),
        },
        "accepted_scores": ({k: {"min": float(acc_df[k].min()), "mean": float(acc_df[k].mean()),
                                 "max": float(acc_df[k].max())}
                             for k in ("length_ratio", "sbert", "max_other_cos")} if len(acc_df) else {}),
    }
    save_json(ctx.out / "summary.json", summary)

    outputs = {p.name: C.sha256_file(p) for p in (acc_path, st_path, cand_path, ctx.out / "summary.json",
                                                  ctx.out / "sbert_checkpoint.json")}
    manifest = {
        "issue": 1, "created_at": now(), "run": RUN, "output_dir": rel(ctx.out),
        "git_revision": R.git_revision(),
        "python": sys.version, "platform": platform.platform(), "gpu": gpu_info(),
        "packages": R.package_versions(), "model": ctx.model,
        "model_15_manifest": ctx.rec15["model"], "config": ctx.config(), "sbert_min_locked": thr,
        "reused_inputs": ctx.reuse_stats, "outputs_sha256": outputs,
        "usage_rule": USAGE_RULE,
        "determinism_note": ("單一 worker 時同一 seed 可重現；所有原始回應都已存檔，結果以存檔為準"),
    }
    save_json(ctx.out / "manifest.json", manifest)
    C.log(f"已寫出 {acc_path.name}（{len(acc_df):,} 筆）、{st_path.name}、{cand_path.name}、summary.json、manifest.json")
    C.log(f"補滿 {cov['filled']:,}/{cov['sources']:,}；缺額 {int(st_df['shortfall'].sum()):,} 筆"
          f"（用完上限 {cov['exhausted_unfilled']:,} 筆原文，還可再生成 {cov['unfilled_with_attempts_left']:,}）")
    if cov["unfilled_with_attempts_left"]:
        C.log("注意：仍有原文未用完生成上限，可再跑 --stage topup 後重新 finalize")
    write_session(ctx, "finalize", started, {"accepted": len(acc_df), "sbert_min": thr})


def read_accepted(ctx: Ctx) -> pd.DataFrame:
    acc = read_csv_text(ctx.out / "accepted.csv")
    for col in ("label_id", "slot"):
        acc[col] = acc[col].astype(int)
    for col in ("length_ratio", "sbert", "max_other_cos"):
        acc[col] = acc[col].astype(float)
    return acc


def noaug_reference(ctx: Ctx) -> tuple[dict, list[str], list[str], np.ndarray]:
    """15_ 的 noaug 語料（C2／C4 用的），作為新語料的原文部分。"""
    meta = C.load_json(CORPUS_DIR / "noaug_meta.json")
    index_path = CORPUS_DIR / "noaug.index"
    C.require_hash(index_path, meta["index_sha256"], "15_ noaug index")
    C.require_hash(index_path, ctx.rec15["manifest"]["corpora"]["noaug"]["index_sha256_actual"], "15_ noaug index（manifest）")
    payload = C.load_json(CORPUS_DIR / "noaug_docs.json")
    faiss = C.import_faiss()
    index = faiss.read_index(str(index_path))
    ids, docs = payload["doc_ids"], payload["docs"]
    if index.ntotal != len(ids) or len(ids) != len(docs):
        C.die("noaug index 與 docs 數量不一致")
    train = [s for s, sp in zip(ctx.pool["source_id"], ctx.split15) if sp == "train"]
    if set(ids) != set(train) or len(ids) != len(train):
        C.die("noaug 語料的 doc_ids 與實驗 15 的 train 不一致")
    for sid, doc in zip(ids, docs):
        j = ctx.idx[sid]
        if doc != R.corpus_entry(ctx.statement[j], ctx.label[j]):
            C.die(f"noaug 語料內容與 train.csv 不符：{sid}")
    return meta, ids, docs, index.reconstruct_n(0, index.ntotal)


def stage_corpus(ctx: Ctx) -> None:
    """依實驗 15 切分建立 RAG 資料庫：15_ noaug 的原文（向量原樣取出）＋ train 來源的改寫。"""
    started = now()
    thr = ctx.locked_sbert()
    acc = read_accepted(ctx)
    noaug_meta, ids, docs, vecs = noaug_reference(ctx)
    train_ids = [s for s, sp in zip(ctx.pool["source_id"], ctx.split15) if sp == "train"]
    para = select_for_split(acc, train_ids)
    para_docs = [R.corpus_entry(t, l) for t, l in zip(para["statement"], para["status"])]
    C.log(f"編碼 {len(para_docs):,} 筆改寫（語料格式，CPU）…")
    para_vecs = ctx.emb.encode(para_docs) if para_docs else np.empty((0, vecs.shape[1]), "float32")
    all_ids = list(ids) + para["id"].tolist()
    all_docs = list(docs) + para_docs
    all_vecs = np.vstack([vecs, para_vecs]).astype("float32")

    faiss = C.import_faiss()
    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    ctx.corpus_dir.mkdir(parents=True, exist_ok=True)
    index_path = ctx.corpus_dir / f"{CORPUS_NAME}.index"
    docs_path = ctx.corpus_dir / f"{CORPUS_NAME}_docs.json"
    faiss.write_index(index, str(index_path))
    save_json(docs_path, {"doc_ids": all_ids, "docs": all_docs})

    eval_norm = {sp: {R.normalize_text(ctx.statement[i]) for i, s in enumerate(ctx.split15) if s == sp}
                 for sp in ("val", "test")}
    para_norm = [R.normalize_text(t) for t in para["statement"]]
    meta = {
        "run": RUN, "name": CORPUS_NAME, "source": "exp15_noaug_train_originals_plus_augment_v2_train_paraphrases",
        "n_docs": len(all_ids), "n_original_docs": len(ids), "n_paraphrase_docs": len(para_docs),
        "paraphrases_by_label": para["status"].value_counts().to_dict(),
        "embedding_model": R.EMBED_MODEL, "max_seq_length": R.EMBED_MAX_SEQ, "normalized": True,
        "metric": "cosine (IndexFlatIP on normalized vectors)", "top_k": 1,
        "doc_format": "{statement} true_label is {status}（R.corpus_entry）",
        "original_vectors_from": "corpus/noaug.index（reconstruct_n，與 15_ C2／C4 的語料逐位元相同）",
        "exact_overlap_with_eval_splits": {sp: int(sum(t in s for t in para_norm)) for sp, s in eval_norm.items()},
        "paraphrase_source_splits": para["exp15_split"].value_counts().to_dict(),
        "sbert_min": thr, "accepted_sha256": C.sha256_file(ctx.out / "accepted.csv"),
        "noaug_index_sha256": noaug_meta["index_sha256"], "noaug_docs_sha256": C.sha256_file(CORPUS_DIR / "noaug_docs.json"),
        "index_sha256": C.sha256_file(index_path), "docs_sha256": C.sha256_file(docs_path),
        "usage_rule": USAGE_RULE, "created_at": now(),
    }
    save_json(ctx.corpus_dir / f"{CORPUS_NAME}_meta.json", meta)
    C.log(f"已建立 {rel(index_path)}：原文 {len(ids):,} + 改寫 {len(para_docs):,} = {len(all_ids):,} 筆；"
          f"與 val／test 逐字重疊 {meta['exact_overlap_with_eval_splits']}")
    write_session(ctx, "corpus", started, {"n_docs": len(all_ids), "sbert_min": thr})


def resplits(ctx: Ctx) -> dict[str, dict[str, str]]:
    """實驗 15 的切分 + 5 組「val 固定、train／test 依 15_ 各類別筆數分層重切」。"""
    sm = C.load_json(SPLITS / "split_manifest.json")["counts_by_label"]
    base = dict(zip(ctx.pool["source_id"], ctx.split15))
    out = {"exp15": base}
    for seed in RESPLIT_SEEDS:
        rng = np.random.default_rng(seed)
        assign = {s: "val" for s, sp in base.items() if sp == "val"}
        for lab in C.LABELS:
            sids = [s for s, sp in base.items() if sp in TARGET_SPLITS and ctx.label[ctx.idx[s]] == lab]
            sids.sort(key=lambda s: ctx.idx[s])                       # 依 source_uid
            perm = [sids[i] for i in rng.permutation(len(sids))]
            n_train = sm["train"][lab]
            if n_train + sm["test"][lab] != len(perm):
                C.die(f"{lab} 的 train＋test 筆數與 split_manifest 不符")
            assign |= {s: "train" for s in perm[:n_train]} | {s: "test" for s in perm[n_train:]}
        out[f"resplit_seed{seed}"] = assign
    return out


def stage_verify(ctx: Ctx) -> None:
    started = now()
    thr = ctx.locked_sbert()
    acc = read_accepted(ctx)
    pool = ctx.pool.set_index("source_id")
    problems = []

    # 1. 每筆改寫的不變條件（用 accepted.csv 的文字重新計算，不採信紀錄）
    if acc["id"].duplicated().any():
        problems.append("id 重複")
    if (acc.groupby("source_id").size() > ctx.n_target).any():
        problems.append(f"有原文超過 {ctx.n_target} 筆改寫")
    if not acc["source_id"].isin(set(ctx.targets)).all():
        problems.append("改寫的原文不在工作對象中")
    if (acc["source_id"].map(pool["exp15_split"]) == "val").any():
        problems.append("有 val 原文的改寫")
    if (acc["status"] != acc["source_id"].map(pool["status"])).any():
        problems.append("改寫標籤與原文不同")
    norm = acc["statement"].map(R.normalize_text)
    if norm.duplicated().any():
        problems.append(f"改寫之間有重複：{int(norm.duplicated().sum())} 筆")
    src_text = acc["source_id"].map(pool["statement"])
    if (norm == src_text.map(R.normalize_text)).any():
        problems.append("改寫與自己的原文相同")
    if (acc["statement"].map(clean_candidate) != acc["statement"]).any() or (acc["statement"] == "").any():
        problems.append("改寫文字未清理或為空")
    ratio = acc["statement"].str.len() / src_text.str.len()
    if not ratio.between(*LENGTH_RATIO).all():
        problems.append("長度比超出範圍")
    ref = [refusal_type(t, s) for t, s in zip(acc["statement"], src_text)]
    if any(ref):
        problems.append(f"拒答偵測命中 {sum(map(bool, ref))} 筆")
    meta = [meta_commentary_type(t, s) for t, s in zip(acc["statement"], src_text)]
    if any(meta):
        problems.append(f"改寫說明句偵測命中 {sum(map(bool, meta))} 筆")
    vec = ctx.emb.encode(acc["statement"].tolist()) if len(acc) else np.empty((0, 384), "float32")
    own = np.array([ctx.idx[s] for s in acc["source_id"]], dtype=int)
    sbert = (vec * ctx.orig_emb[own]).sum(1) if len(acc) else np.array([])
    if len(acc) and sbert.min() < thr:
        problems.append(f"SBERT 低於門檻 {thr}：{int((sbert < thr).sum())} 筆")
    if len(acc) and float(np.abs(sbert - acc["sbert"].to_numpy()).max()) > 1e-4:
        problems.append("重算的 SBERT 與記錄不符")
    archived = {}
    with gzip.open(ctx.out / "candidates.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            archived[rec["candidate_id"]] = rec
    selected = {cid for cid, rec in archived.items() if rec["selected"]}
    if selected != set(acc["candidate_id"]):
        problems.append("accepted.csv 與 candidates.jsonl.gz 的選定狀態不一致")
    text_ok = [archived.get(cid, {}).get("checks", {}).get("cleaned") == t
               for cid, t in zip(acc["candidate_id"], acc["statement"])]
    if not all(text_ok):
        problems.append(f"accepted.csv 文字與候選紀錄不符：{text_ok.count(False)} 筆")

    # 2. 實驗 15 切分的 RAG 資料庫
    corpus_check = None
    meta_path = ctx.corpus_dir / f"{CORPUS_NAME}_meta.json"
    if meta_path.exists():
        cmeta = C.load_json(meta_path)
        faiss = C.import_faiss()
        index_path = ctx.corpus_dir / f"{CORPUS_NAME}.index"
        if C.sha256_file(index_path) != cmeta["index_sha256"]:
            problems.append("aug_v2.index 的 sha 與 meta 不符")
        if cmeta["accepted_sha256"] != C.sha256_file(ctx.out / "accepted.csv"):
            problems.append("aug_v2 語料不是由目前的 accepted.csv 建立")
        index = faiss.read_index(str(index_path))
        payload = C.load_json(ctx.corpus_dir / f"{CORPUS_NAME}_docs.json")
        _, n_ids, n_docs, n_vecs = noaug_reference(ctx)
        cv = index.reconstruct_n(0, index.ntotal)
        src_of = dict(zip(acc["id"], acc["source_id"]))
        id_src = [src_of[i] if i.startswith("aug_") else i for i in payload["doc_ids"]]
        bad_src = sum(ctx.split15[ctx.idx[s]] != "train" for s in id_src)
        n_orig_same = bool(np.array_equal(cv[:len(n_ids)], n_vecs)) and payload["doc_ids"][:len(n_ids)] == n_ids \
            and payload["docs"][:len(n_ids)] == n_docs
        eval_norm = {R.normalize_text(ctx.statement[i]) for i, s in enumerate(ctx.split15) if s != "train"}
        exact = sum(R.normalize_text(d.rsplit(" true_label is ", 1)[0]) in eval_norm for d in payload["docs"])
        expect = len(n_ids) + len(select_for_split(acc, [s for s, sp in zip(ctx.pool["source_id"], ctx.split15)
                                                         if sp == "train"]))
        corpus_check = {"n_docs": int(index.ntotal), "expected_n_docs": expect,
                        "docs_from_non_train_sources": int(bad_src), "original_part_identical_to_noaug": n_orig_same,
                        "exact_overlap_with_val_test": int(exact), "top_k_meta": cmeta.get("top_k")}
        if index.ntotal != expect or index.ntotal != len(payload["doc_ids"]):
            problems.append("aug_v2 語料筆數不符")
        if bad_src:
            problems.append(f"aug_v2 語料含非 train 來源的文件：{bad_src} 筆")
        if not n_orig_same:
            problems.append("aug_v2 語料的原文部分與 noaug 不同")
        if exact:
            problems.append(f"aug_v2 語料與 val／test 原文逐字重疊：{exact} 筆")
        if cmeta.get("top_k") != 1:
            problems.append("aug_v2 meta 的 top_k 不是 1")
    else:
        problems.append("尚未建立 aug_v2 語料（--stage corpus）")

    # 3. 切分：實驗 15 的切分 + 5 組 val 固定、train／test 重切
    corpus_vec = None
    if len(acc):
        key = C.sha256_file(ctx.out / "accepted.csv")[:16]
        cpath = OUT_ROOT / "cache" / f"corpus_emb_{key}_{C.sha256_text(chr(0).join(ctx.statement))[:8]}.npz"
        if cpath.exists():
            z = np.load(cpath)
            orig_c, acc_c = z["orig"], z["acc"]
        else:
            C.log("編碼語料格式（true_label is X）的文件，供檢索相似度參考 …")
            orig_c = ctx.emb.encode([R.corpus_entry(t, l) for t, l in zip(ctx.statement, ctx.label)])
            acc_c = ctx.emb.encode([R.corpus_entry(t, l) for t, l in zip(acc["statement"], acc["status"])])
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cpath, orig=orig_c, acc=acc_c)
        corpus_vec = (orig_c, acc_c)

    acc_pos = {cid: i for i, cid in enumerate(acc["candidate_id"])}
    split_results = {}
    for name, assign in resplits(ctx).items():
        train_ids = [s for s in ctx.pool["source_id"] if assign[s] == "train"]
        eval_ids = [s for s in ctx.pool["source_id"] if assign[s] != "train"]
        used = select_for_split(acc, train_ids)
        leaked = int(used["source_id"].isin(set(eval_ids)).sum())
        tr_idx = np.array([ctx.idx[s] for s in train_ids])
        ev_idx = np.array([ctx.idx[s] for s in eval_ids])
        used_idx = np.array([acc_pos[c] for c in used["candidate_id"]], dtype=int)
        ev_norm = {R.normalize_text(ctx.statement[i]) for i in ev_idx}
        exact = int(sum(R.normalize_text(t) in ev_norm for t in used["statement"]))
        exact += int(sum(R.normalize_text(ctx.statement[i]) in ev_norm for i in tr_idx))
        para_max = None
        for i in range(0, len(used_idx), 2000):
            m = float((vec[used_idx[i:i + 2000]] @ ctx.orig_emb[ev_idx].T).max())
            para_max = m if para_max is None else max(para_max, m)
        orig_max = float((ctx.orig_emb[tr_idx] @ ctx.orig_emb[ev_idx].T).max())
        res = {"n_train_sources": len(train_ids), "n_eval_sources": len(eval_ids),
               "n_paraphrases_in_corpus": int(len(used)), "paraphrases_from_eval_sources": leaked,
               "exact_overlap_corpus_vs_eval": exact,
               "max_cos_paraphrase_vs_eval_original": para_max,
               "max_cos_train_original_vs_eval_original": orig_max,
               "n_paraphrase_vs_eval_cos_ge_0_90": int(sum(
                   int(((vec[used_idx[i:i + 2000]] @ ctx.orig_emb[ev_idx].T).max(1) >= 0.90).sum())
                   for i in range(0, len(used_idx), 2000)))}
        if corpus_vec is not None:
            docs = np.vstack([corpus_vec[0][tr_idx], corpus_vec[1][used_idx]])
            is_para = np.r_[np.zeros(len(tr_idx), bool), np.ones(len(used_idx), bool)]
            for sp in ("val", "test"):
                q = np.array([ctx.idx[s] for s in eval_ids if assign[s] == sp])
                top1, top1_para = [], []
                for i in range(0, len(q), 1000):
                    sims = ctx.orig_emb[q[i:i + 1000]] @ docs.T
                    k = sims.argmax(1)
                    top1.append(sims[np.arange(len(k)), k])
                    top1_para.append(is_para[k])
                top1 = np.concatenate(top1)
                top1_para = np.concatenate(top1_para)
                res[f"retrieval_top1_{sp}"] = {
                    "max": float(top1.max()), "mean": float(top1.mean()), "p99": float(np.quantile(top1, 0.99)),
                    "n_ge_0_90": int((top1 >= 0.90).sum()), "share_paraphrase": float(top1_para.mean()),
                }
        ok = leaked == 0 and exact == 0
        res["pass"] = bool(ok)
        if not ok:
            problems.append(f"切分 {name} 未通過洩漏檢查")
        split_results[name] = res
        C.log(f"  {name}: 語料改寫 {len(used):,} 筆，eval 來源 {leaked}，逐字重疊 {exact}，"
              f"改寫對 eval 原文最大 cosine {para_max if para_max is None else round(para_max, 4)}"
              f"（train 原文對 eval 原文 {orig_max:.4f}）")

    # 對照：15_ 的 aug 語料（test 查詢 top-1）
    ref15 = None
    if not str(ctx.out).endswith("_smoke"):
        faiss = C.import_faiss()
        idx15 = faiss.read_index(str(CORPUS_DIR / "aug.index"))
        q = np.array([ctx.idx[s] for s, sp in zip(ctx.pool["source_id"], ctx.split15) if sp == "test"])
        sims, ids = idx15.search(ctx.orig_emb[q], 1)
        p15 = C.load_json(AUG_DOCS)["doc_ids"]
        ref15 = {"test_top1_max": float(sims.max()), "test_top1_mean": float(sims.mean()),
                 "test_top1_share_paraphrase": float(np.mean([p15[i].startswith("aug_") for i in ids[:, 0]]))}

    out = {"created_at": now(), "sbert_min": thr, "accepted_sha256": C.sha256_file(ctx.out / "accepted.csv"),
           "n_accepted": int(len(acc)), "usage_rule": USAGE_RULE,
           "invariants": {"problems": problems,
                          "sbert_min_observed": float(sbert.min()) if len(acc) else None,
                          "length_ratio_range": [float(ratio.min()), float(ratio.max())] if len(acc) else None},
           "corpus_exp15": corpus_check, "exp15_aug_corpus_reference": ref15,
           "splits": split_results, "pass": not problems}
    save_json(ctx.out / "split_safety_check.json", out)
    write_session(ctx, "verify", started, {"pass": not problems})
    if problems:
        C.die("驗證未通過：" + "；".join(problems))
    C.log(f"驗證通過：{len(acc):,} 筆改寫，{len(split_results)} 組切分皆無洩漏")


# ---------------------------------------------------------------- 主程式

def smoke_targets(pool: pd.DataFrame) -> list[str]:
    out = []
    for lab in C.LABELS:
        for role in SMOKE_PER_LABEL_ROLES:
            sub = pool[(pool["status"] == lab) & (pool["exp15_role"] == role)]
            out.append(sub["source_id"].iloc[0])
    order = {sid: i for i, sid in enumerate(pool["source_id"])}
    return sorted(out, key=order.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["reuse", "pilot", "topup", "finalize", "corpus", "verify"])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget-h", type=float, default=12.0)
    args = ap.parse_args()
    if not (args.check_only or args.smoke or args.stage):
        ap.error("請指定 --check-only、--smoke 或 --stage")

    if args.check_only:
        ctx = Ctx(OUT_ROOT, N_TARGET, MAX_NEW_ATTEMPTS, None, need_llm=True, need_embed=False)
        roles = Counter(ctx.pool.at[ctx.idx[s], "exp15_role"] for s in ctx.cands_reused)
        C.log(f"pool {len(ctx.pool):,} 筆（每類 {N_PER_LABEL}）；工作對象 train＋test {len(ctx.targets):,} 筆；"
              f"有沿用候選的原文 {len(ctx.cands_reused):,} 筆 {dict(roles)}")
        C.log(f"沿用候選 MAIN {ctx.reuse_stats['main_raw']:,} + 巢狀 {ctx.reuse_stats['nested_raw']:,}；"
              f"原始回應依當時規則清理後逐字重現 15_ 的 {N_15_PARAPHRASES:,} 筆改寫 ✓")
        C.log(f"模型 digest {ctx.model['digest'][:12]}… 與 15_ 相符 ✓；Ollama {ctx.model.get('ollama_version')}")
        C.log(f"改寫指令 sha {C.sha256_text(R.REWRITE_INSTRUCTION)[:12]}… ✓")
        C.log("檢查完成")
        return

    C.single_instance("augment_v2")
    if args.smoke:
        out = OUT_ROOT / "_smoke"
        ctx0 = Ctx(out, N_TARGET, SMOKE_MAX_NEW, None, need_llm=False, need_embed=False)
        targets = smoke_targets(ctx0.pool)
        ctx0.close()
        ctx = Ctx(out, N_TARGET, SMOKE_MAX_NEW, targets, corpus_dir=out / "corpus")
        ctx.ensure_config()
        C.log(f"smoke：{len(targets)} 筆原文、新生成上限 {SMOKE_MAX_NEW} 次，輸出到 {out}")
        try:
            stage_reuse(ctx)
            stage_pilot(ctx, args.workers, args.time_budget_h, sids=list(ctx.targets))  # smoke：全部原文當 pilot
            stage_topup(ctx, args.workers, args.time_budget_h)
            stage_finalize(ctx)
            stage_corpus(ctx)
            stage_verify(ctx)
        finally:
            ctx.close()
        return

    ctx = Ctx(OUT_ROOT, N_TARGET, MAX_NEW_ATTEMPTS, None, need_llm=args.stage in ("pilot", "topup"))
    ctx.ensure_config()
    try:
        {"reuse": lambda: stage_reuse(ctx),
         "pilot": lambda: stage_pilot(ctx, args.workers, args.time_budget_h),
         "topup": lambda: stage_topup(ctx, args.workers, args.time_budget_h),
         "finalize": lambda: stage_finalize(ctx),
         "corpus": lambda: stage_corpus(ctx),
         "verify": lambda: stage_verify(ctx)}[args.stage]()
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
