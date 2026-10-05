"""主實驗（v1：15_；v2：27_）的解碼確定性驗證（issue #5，審稿意見 R3C13、R3C14）。

從 test 依類別分層抽樣（預設每類 50 筆、共 200 筆，seed 42；與 profile 無關，v1、v2 同一批），
把主實驗存下的 C4、C5 ``rendered_prompt`` 原封不動再送 Ollama（同 15_ 的 GEN_OPTIONS，
預設兩次），比對：

* 每次重跑與存檔的 raw_response 逐字一致率、解析後預測一致率；
* 重跑之間的一致率（每次重跑前先卸載模型，等同另起一次執行）：若重跑彼此完全一致、
  卻與存檔不同，差異來自環境漂移（例如 Ollama 版本），而不是重跑之間的隨機性；
* 以 profile 的 FAISS 索引（12_ RebuildRagIndex）重新檢索 top-k，``retrieved_ids``（k 篇）
  是否一致，以及用重新檢索結果依 15_ 的 render 規則（k 篇依相似度遞減以 \\n 串接，同 27_）
  重組的 prompt 是否與存檔逐字相同。

開跑前確認目前 Ollama 的 model digest 與 profile manifest、逐筆預測記錄的 digest 相同，
不同就停止。推論期間背景以 nvidia-smi 每 0.5 秒取樣 GPU 記憶體用量（供 issue #9）。

輸出到 <輸出目錄>/determinism/（v2 預設 MAIN_EVAL_AUGV2/；v1 的 CUSTOM_PROMPT_EVAL/
是凍結的存檔，必須以 --out-dir 另外指定）：
    sample_ids.csv           抽樣的 id 與標籤
    rerun_C4.jsonl、rerun_C5.jsonl  第 1 次重跑的逐筆比對（可中斷續跑）
    rerun_C4_pass2.jsonl …   第 2 次起的重跑
    sessions.jsonl           每次執行的環境（Ollama 版本、GPU、VRAM 峰值、size_vram）
    summary.json             一致率（Wilson 95% CI）、不一致清單、環境；v2 另附 v1 的一致率
    cache_state_probe.jsonl  （--probe-cache-state）不一致樣本在不同 KV cache 狀態下的輸出

用法：
    python 21_decoding_determinism.py --profile v2 [--n 200] [--passes 2]
    python 21_decoding_determinism.py --profile v1 --out-dir 目錄
    python 21_decoding_determinism.py --profile v2 --probe-cache-state
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import custom_eval_common as E  # noqa: E402

CODES = ("C4", "C5")
MANIFEST_FILES = {"v1": E.V1_OUT / "custom_prompt_eval_manifest.json",
                  "v2": E.V2_OUT / "manifest.json"}
V1_SUMMARY = E.V1_OUT / "determinism" / "summary.json"
NVSMI_POLL_S = 0.5
TRUNC_CHARS_PER_TOKEN = 6.0            # 同 27_：超過 num_ctx 被截斷時 prompt_eval_count 偏小


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return centre - half, centre + half


def render(template: str, text: str, reference_context: str | None = None) -> str:
    """與 15_ 的 render 相同規則。"""
    prompt = template.replace("{text}", text)
    if reference_context is not None:
        prompt = prompt.replace("{reference_context}", reference_context)
    if "{" in prompt or "}" in prompt:
        C.die("rendered prompt contains an unreplaced placeholder")
    return prompt


# ---------------------------------------------------------------- 環境

def nvidia_smi(fields: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]


class VramMonitor:
    """背景執行緒：每 NVSMI_POLL_S 秒取 GPU 0 的 memory.used（MiB），記錄峰值。"""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(int(nvidia_smi("memory.used")[0]))
            except Exception:  # noqa: BLE001 —— 量測失敗不影響驗證本身
                pass
            self._stop.wait(NVSMI_POLL_S)

    def __enter__(self) -> "VramMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def ollama_version() -> str | None:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version")
    except Exception:  # noqa: BLE001
        return None


def ollama_loaded() -> list[dict]:
    import ollama

    rows = []
    for m in getattr(ollama.ps(), "models", []) or []:
        rows.append({"model": getattr(m, "model", None), "digest": getattr(m, "digest", None),
                     "size_bytes": getattr(m, "size", None),
                     "size_vram_bytes": getattr(m, "size_vram", None)})
    return rows


def unload_model() -> None:
    """讓每一次重跑都從重新載入模型開始，等同另起一次執行。"""
    import ollama

    ollama.generate(model=C.MODEL_NAME, prompt="", keep_alive=0)
    for _ in range(60):
        if not ollama_loaded():
            return
        time.sleep(0.5)
    C.die("Ollama 模型未能卸載")


def environment(R) -> dict:
    env = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "git_revision": R.git_revision(), "packages": R.package_versions(),
           "ollama_server_version": ollama_version(), "model": R.probe_llm_identity(),
           "llm_generation_options": C.GEN_OPTIONS}
    try:
        name, driver, total = nvidia_smi("name,driver_version,memory.total")
        env["gpu"] = {"name": name, "driver_version": driver, "memory_total_mib": int(total)}
    except Exception as exc:  # noqa: BLE001
        env["gpu"] = f"unavailable: {exc}"
    return env


# ---------------------------------------------------------------- 抽樣

def stratified_sample(n: int, out: Path) -> pd.DataFrame:
    if n % len(C.LABELS):
        C.die(f"--n 必須是 {len(C.LABELS)} 的倍數（每類等量）")
    test = E.load_test()
    per = n // len(C.LABELS)
    parts = [test[test.status == lab].sample(n=per, random_state=E.SEED + i)
             for i, lab in enumerate(C.LABELS)]
    sample = pd.concat(parts)[["id", "status", "label_id"]].reset_index(drop=True)
    sample["id"] = sample["id"].astype(str)
    path = out / "sample_ids.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"id": str})
        if list(old["id"]) != list(sample["id"]):
            C.die(f"{path} 與這次的抽樣不同（--n 改過？），請換 --out 或刪除舊結果")
    else:
        out.mkdir(parents=True, exist_ok=True)
        sample.to_csv(path, index=False, encoding="utf-8-sig")
        C.log(f"已寫入 {path}")
    return sample.merge(test[["id", "statement"]].astype({"id": str}), on="id")


# ---------------------------------------------------------------- 重跑

def check_digest(R, P: E.Profile, preds: dict) -> str:
    current = R.probe_llm_identity().get("digest")
    recorded = P.manifest.get("model", {}).get("digest")
    in_records = {preds[code][0]["model_digest"] for code in CODES}
    if not current or current != recorded or in_records != {recorded}:
        C.die(f"model digest 不一致：目前 {current}，{P.name} manifest {recorded}，"
              f"逐筆預測 {in_records}")
    return current


def load_rags(P: E.Profile, R) -> dict:
    """profile 的 RAG 語料；index sha 由 RebuildRagIndex 檢查，docs sha 另外核對（容許 CRLF）。"""
    rags = {}
    for name in P.corpora:
        rag = R.RebuildRagIndex(E.RUN, name)
        E.check_corpus_docs(name, rag.meta)
        rags[name] = rag
    return rags


def retrieve(rag, text: str, k: int) -> tuple[list[str], list[str], list[float]]:
    """與 27_ retrieve 相同：單筆編碼，取前 k 篇（RebuildRagIndex.search 只允許 k = 1）。"""
    vec = rag.encode([text])
    sims, idxs = rag.index.search(vec, k)
    if len(idxs[0]) != k or int(idxs[0].min()) < 0:
        C.die(f"{rag.name} 檢索結果不足 {k} 篇")
    return ([rag.docs[int(i)] for i in idxs[0]], [str(rag.doc_ids[int(i)]) for i in idxs[0]],
            [float(v) for v in sims[0]])


def truncated(prompt: str, prompt_eval_count: int | None) -> bool:
    return len(prompt) > TRUNC_CHARS_PER_TOKEN * max(prompt_eval_count or 0, 1)


def rerun_path(out: Path, code: str, pass_no: int) -> Path:
    return out / (f"rerun_{code}.jsonl" if pass_no == 1 else f"rerun_{code}_pass{pass_no}.jsonl")


def rerun(P: E.Profile, code: str, pass_no: int, sample: pd.DataFrame, by_id: dict,
          template: str, rag, out: Path) -> None:
    path = rerun_path(out, code, pass_no)
    done = C.done_ids(path)
    for row in sample.itertuples(index=False):
        if row.id in done:
            continue
        orig = by_id[row.id]
        docs, ids, sims = retrieve(rag, str(row.statement), P.top_k)
        rebuilt = render(template, str(row.statement), "\n".join(docs))
        response = C.chat(orig["rendered_prompt"])
        pred, reason = C.parse_label(response["raw_response"])
        C.append_jsonl(path, {
            "id": row.id, "code": code, "pass": pass_no, "condition": P.condition(code),
            "top_k": P.top_k, "true_label": orig["true_label"],
            "orig_raw_response": orig["raw_response"], "rerun_raw_response": response["raw_response"],
            "raw_equal": response["raw_response"] == orig["raw_response"],
            "orig_pred_label_id": orig["pred_label_id"], "rerun_pred_label_id": pred,
            "pred_equal": pred == orig["pred_label_id"],
            "orig_parse_reason": orig["parse_reason"], "rerun_parse_reason": reason,
            "orig_retrieved_ids": orig["retrieved_ids"], "rerun_retrieved_ids": ids,
            "retrieved_equal": ids == orig["retrieved_ids"],
            "retrieved_similarity_abs_diff": max(
                abs(a - b) for a, b in zip(sims, orig["retrieved_similarities"])),
            "prompt_rebuilt_equal": rebuilt == orig["rendered_prompt"],
            "elapsed_s": response["elapsed_s"], "prompt_eval_count": response["prompt_eval_count"],
            "eval_count": response["eval_count"],
            "orig_prompt_eval_count": orig["prompt_eval_count"],
            "orig_truncated": truncated(orig["rendered_prompt"], orig["prompt_eval_count"]),
            "rerun_truncated": truncated(orig["rendered_prompt"], response["prompt_eval_count"]),
        })
        done.add(row.id)
        C.log(f"{code} pass {pass_no}: {len(done)}/{len(sample)} id={row.id} raw_equal="
              f"{response['raw_response'] == orig['raw_response']}")


# ---------------------------------------------------------------- 彙總

def rate(records: list[dict], key: str) -> dict:
    k, n = sum(bool(r[key]) for r in records), len(records)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


def check_cached(P: E.Profile, preds: dict, sample: pd.DataFrame, out: Path, passes: int) -> None:
    """既有的重跑紀錄必須屬於這個 profile 與現行輸入，否則停止（不能拿來續跑或彙總）。

    --out-dir 可以指到別的 profile 用過的目錄，主實驗的預測也可能重產；只比對 id 會把
    不相干的紀錄當成已完成。逐筆核對代號、條件、pass、top_k（k > 1 之前的紀錄沒有此欄，
    視為 1），以及存檔的 raw 回覆、預測與檢索 id 是否與現行預測檔相同。
    """
    ids = set(sample["id"])
    for code in CODES:
        by_id = {str(r["id"]): r for r in preds[code]}
        for pass_no in range(1, passes + 1):
            path = rerun_path(out, code, pass_no)
            for r in C.read_jsonl(path):
                orig = by_id.get(r["id"])
                problems = [name for name, ok in (
                    ("id 不在抽樣中", r["id"] in ids and orig is not None),
                    ("code", r.get("code") == code),
                    ("pass", r.get("pass") == pass_no),
                    ("condition", r.get("condition") == P.condition(code)),
                    ("top_k", r.get("top_k", 1) == P.top_k),
                    ("orig_raw_response", orig is not None
                     and r.get("orig_raw_response") == orig["raw_response"]),
                    ("orig_pred_label_id", orig is not None
                     and r.get("orig_pred_label_id") == orig["pred_label_id"]),
                    ("orig_retrieved_ids", orig is not None
                     and r.get("orig_retrieved_ids") == orig["retrieved_ids"]),
                ) if not ok]
                if problems:
                    C.die(f"{path} 的 {r['id']} 與 profile {P.name} 的現行輸入不符（"
                          + "、".join(problems) + "）：請換 --out-dir 或刪除舊結果")


def load_pass(out: Path, code: str, pass_no: int, sample: pd.DataFrame) -> list[dict]:
    path = rerun_path(out, code, pass_no)
    recs = list(C.read_jsonl(path))
    if {r["id"] for r in recs} != set(sample["id"]) or len(recs) != len(sample):
        C.die(f"{path.name} 尚未完成：{len(recs)}/{len(sample)}")
    return recs


def vs_original(recs: list[dict]) -> dict:
    keys = ("raw_equal", "pred_equal", "retrieved_equal", "prompt_rebuilt_equal")
    extra = {}
    if all("orig_truncated" in r for r in recs):    # 加入 k > 1 之後的紀錄才有
        extra = {"truncated": {"orig": sum(r["orig_truncated"] for r in recs),
                               "rerun": sum(r["rerun_truncated"] for r in recs), "n": len(recs)}}
    return {
        **{k: rate(recs, k) for k in keys},
        **extra,
            "by_label": {lab: {k: rate([r for r in recs if r["true_label"] == lab], k)
                               for k in ("raw_equal", "pred_equal")} for lab in C.LABELS},
            "max_retrieved_similarity_abs_diff": max(r["retrieved_similarity_abs_diff"] for r in recs),
            "mismatches": [{k: r[k] for k in ("id", "true_label", "orig_raw_response",
                                              "rerun_raw_response", "orig_pred_label_id",
                                              "rerun_pred_label_id", "retrieved_equal")}
                           for r in recs if not (r["raw_equal"] and r["pred_equal"]
                                                 and r["retrieved_equal"])],
    }


def between_passes(first: list[dict], other: list[dict]) -> dict:
    """重跑之間互相比較：區分「同一環境內的隨機性」與「相對 15_ 當時環境的漂移」。"""
    b = {(r["code"], r["id"]): r for r in other}
    pairs = [{"id": f"{r['code']}:{r['id']}",
              "raw_equal": r["rerun_raw_response"] == b[r["code"], r["id"]]["rerun_raw_response"],
              "pred_equal": r["rerun_pred_label_id"] == b[r["code"], r["id"]]["rerun_pred_label_id"]}
             for r in first]
    return {"raw_equal": rate(pairs, "raw_equal"), "pred_equal": rate(pairs, "pred_equal"),
            "mismatch_ids": [p["id"] for p in pairs if not (p["raw_equal"] and p["pred_equal"])]}


def summarize(P: E.Profile, sample: pd.DataFrame, out: Path, passes: int) -> dict:
    result, pooled = {}, {p: [] for p in range(1, passes + 1)}
    for code in CODES:
        runs = {p: load_pass(out, code, p, sample) for p in range(1, passes + 1)}
        for p, recs in runs.items():
            pooled[p] += recs
        result[code] = {
            "condition": P.condition(code),
            "display_name": P.display(code),
            "vs_original": {f"pass{p}": vs_original(recs) for p, recs in runs.items()},
            "between_passes": {f"pass1_vs_pass{p}": between_passes(runs[1], runs[p])
                               for p in range(2, passes + 1)},
        }
    keys = ("raw_equal", "pred_equal", "retrieved_equal")
    result["pooled"] = {
        "vs_original": {f"pass{p}": {k: rate(recs, k) for k in keys} for p, recs in pooled.items()},
        "between_passes": {f"pass1_vs_pass{p}": between_passes(pooled[1], pooled[p])
                           for p in range(2, passes + 1)},
    }
    return result


def v1_reference() -> dict:
    """v1（15_，k = 1）的一致率，供 v2 並列。"""
    v1 = C.load_json(V1_SUMMARY)
    if not v1:
        C.die(f"找不到 v1 的確定性結果：{V1_SUMMARY}")
    keys = ("raw_equal", "pred_equal", "retrieved_equal")
    pick = lambda d: {k: d[k] for k in keys if k in d}  # noqa: E731
    res = v1["results"]
    return {
        "path": str(V1_SUMMARY.relative_to(C.ROOT)).replace("\\", "/"),
        "sha256": C.sha256_file(V1_SUMMARY),
        "top_k": 1,
        "ollama_server_version": [s.get("ollama_server_version") for s in v1["sessions"]],
        "vram": [s.get("vram") for s in v1["sessions"]],
        "results": {code: {"condition": res[code].get("condition"),
                           "vs_original": {p: pick(v) for p, v in res[code]["vs_original"].items()},
                           "between_passes": {p: pick(v)
                                              for p, v in res[code]["between_passes"].items()}}
                    for code in (*CODES, "pooled")},
    }


def probe_cache_state(P: E.Profile, preds: dict, out: Path) -> dict:
    """對與 15_ 不一致的樣本，檢查差異是否來自 Ollama 的 KV cache（前一個請求）狀態。

    每筆在三種狀態下各送一次：cold＝剛載入模型直接送；after_predecessor＝剛載入後
    先送 15_ 執行順序中緊接在前的那一筆，再送目標；repeat_same＝緊接著把目標再送一次。
    """
    summary = C.load_json(out / "summary.json")
    if not summary:
        C.die("請先完成重跑（summary.json 不存在）")
    if summary.get("profile", "v1") != P.name:     # 加入 profile 之前的 summary 都是 v1
        C.die(f"{out / 'summary.json'} 屬於 profile {summary.get('profile', 'v1')}，不是 {P.name}")
    path = out / "cache_state_probe.jsonl"
    path.unlink(missing_ok=True)
    rows = []
    for code in CODES:
        order = preds[code]                        # E.load_predictions 依 test.csv 順序
        raw_order = list(C.read_jsonl(P.in_dir / f"predictions_{P.condition(code)}.jsonl"))
        if [r["id"] for r in raw_order] != [r["id"] for r in order]:
            C.die(f"{code} 預測檔順序與 test.csv 不同，無法推得主實驗的執行順序")
        pos = {r["id"]: i for i, r in enumerate(order)}
        for m in summary["results"][code]["vs_original"]["pass1"]["mismatches"]:
            i = pos[m["id"]]
            target = order[i]
            got = {}
            unload_model()
            got["cold"] = C.chat(target["rendered_prompt"])["raw_response"]
            unload_model()
            if i > 0:
                C.chat(order[i - 1]["rendered_prompt"])
            got["after_predecessor"] = C.chat(target["rendered_prompt"])["raw_response"]
            got["repeat_same"] = C.chat(target["rendered_prompt"])["raw_response"]
            row = {"code": code, "id": m["id"], "position_in_15_run": i,
                   "predecessor_id": order[i - 1]["id"] if i else None,
                   "orig_raw_response": target["raw_response"],
                   **{f"{k}_raw_response": v for k, v in got.items()},
                   **{f"{k}_equals_orig": v == target["raw_response"] for k, v in got.items()},
                   "outputs_differ_across_states": len(set(got.values())) > 1}
            C.append_jsonl(path, row)
            rows.append(row)
            C.log(f"{code} {m['id']}: " + " ".join(
                f"{k}={row[f'{k}_equals_orig']}" for k in got))
    states = ("cold", "after_predecessor", "repeat_same")
    result = {
        "procedure": "".join(probe_cache_state.__doc__.split("\n\n", 1)[1].split()),
        "n": len(rows),
        "equals_orig": {s: sum(r[f"{s}_equals_orig"] for r in rows) for s in states},
        "recovered_by_any_state": sum(any(r[f"{s}_equals_orig"] for s in states) for r in rows),
        "outputs_differ_across_states": sum(r["outputs_differ_across_states"] for r in rows),
        "interpretation": (
            "同一 prompt 的輸出會隨 KV cache 狀態（前一個請求）而改變；"
            "15_ 是依 test.csv 順序連續送出，前一筆無法完全重現時，該筆輸出也就不一定能重現"),
    }
    summary["cache_state_probe"] = result
    C.save_json(out / "summary.json", summary)
    C.log(f"已寫入 {path} 與 summary.json 的 cache_state_probe")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="抽樣總數（每類 n/4）")
    ap.add_argument("--passes", type=int, default=2,
                    help="重跑次數（每次先卸載模型）；第 2 次起用來區分重跑間的隨機性與環境漂移")
    ap.add_argument("--probe-cache-state", action="store_true",
                    help="重跑完成後，對不一致樣本檢查 KV cache 狀態的影響")
    E.add_profile_args(ap)
    args = ap.parse_args()

    P, base = E.resolve(args)
    out = base / "determinism"
    C.single_instance(f"determinism_{P.name}")
    R = C.load_module("12_rebuild_experiment")
    prompts, _ = E.source_15()
    template_of = {cond: prompts[prompt]["template"] for cond, (prompt, _) in P.conditions.items()}

    preds = {code: recs for code, recs in E.load_predictions(P).items() if code in CODES}
    digest = check_digest(R, P, preds)
    if args.probe_cache_state:
        res = probe_cache_state(P, preds, out)
        C.log(f"cache 探測 {res['n']} 筆：與存檔相同 {res['equals_orig']}；"
              f"任一狀態可重現 {res['recovered_by_any_state']}；"
              f"不同狀態輸出不同 {res['outputs_differ_across_states']}")
        return
    sample = stratified_sample(args.n, out)
    check_cached(P, preds, sample, out, args.passes)

    # 全部重跑都已落地時只重算 summary：不卸載模型、不量 VRAM，也不新增 session 紀錄
    pending = any(len(C.done_ids(rerun_path(out, code, p))) < len(sample)
                  for code in CODES for p in range(1, args.passes + 1))
    if pending:
        env = environment(R)
        env["profile"] = P.name
        env["top_k"] = P.top_k
        unload_model()   # 基準用量不含模型
        try:
            before = int(nvidia_smi("memory.used")[0])
        except Exception:  # noqa: BLE001
            before = None
        rags = load_rags(P, R)
        with VramMonitor() as mon:
            for pass_no in range(1, args.passes + 1):
                unload_model()
                for code in CODES:
                    cond = P.condition(code)
                    by_id = {str(r["id"]): r for r in preds[code]}
                    rerun(P, code, pass_no, sample, by_id, template_of[cond],
                          rags[P.corpus(code)], out)
                env.setdefault("ollama_ps_after", {})[f"pass{pass_no}"] = ollama_loaded()
        env["vram"] = {"poll_interval_s": NVSMI_POLL_S, "n_samples": len(mon.samples),
                       "used_before_run_mib": before,
                       "peak_used_mib": max(mon.samples) if mon.samples else None,
                       "note": "nvidia-smi memory.used 為整張 GPU 的用量（含其他程序）；"
                               "模型本身佔用看 ollama_ps_after 的 size_vram_bytes"}
        C.append_jsonl(out / "sessions.jsonl", env)
    else:
        C.log("所有重跑都已完成，只重算 summary.json")

    source = Path(P.source_script).name
    summary = {
        "run": E.RUN, "experiment": P.experiment, "profile": P.name, "top_k": P.top_k,
        "script": "src/21_decoding_determinism.py", "issue": "#5", "procedure": (
            f"test 依類別分層抽 {args.n} 筆（每類 {args.n // 4}，random_state=42+類別序）；"
            f"C4、C5 各以 {source} 存下的 rendered_prompt 原樣重送 Ollama {args.passes} 次"
            "（15_ 的 GEN_OPTIONS，每次先卸載模型），以 15_ 的 parse_label 解析；另以 "
            f"{'／'.join(P.corpora)} 的 FAISS 索引重新檢索 top-{P.top_k}，依 15_ 的 render 規則"
            "（k 篇依相似度遞減以 \\n 串接）重組 prompt"),
        "inputs": {"manifest_sha256": C.sha256_file(MANIFEST_FILES[P.name]),
                   "test_sha256": C.sha256_file(E.TEST_PATH),
                   **{f"predictions_{code}_sha256":
                      C.sha256_file(P.in_dir / f"predictions_{P.condition(code)}.jsonl")
                      for code in CODES}},
        "model_digest": digest,
        "llm_generation_options": C.GEN_OPTIONS,
        "display_names": {code: P.display_names()[code] for code in CODES},
        "results": summarize(P, sample, out, args.passes),
        "sessions": list(C.read_jsonl(out / "sessions.jsonl")),
    }
    if P.name == "v2":
        summary["v1_reference"] = v1_reference()
    C.save_json(out / "summary.json", summary)
    C.log(f"已寫入 {out / 'summary.json'}")
    for code in (*CODES, "pooled"):
        res = summary["results"][code]
        for name, v in res["vs_original"].items():
            C.log(f"{code} {name} vs 存檔: raw {v['raw_equal']['k']}/{v['raw_equal']['n']}  "
                  f"pred {v['pred_equal']['k']}/{v['pred_equal']['n']}  "
                  f"retrieval {v['retrieved_equal']['k']}/{v['retrieved_equal']['n']}")
        for name, v in res["between_passes"].items():
            C.log(f"{code} {name}: raw {v['raw_equal']['k']}/{v['raw_equal']['n']}  "
                  f"pred {v['pred_equal']['k']}/{v['pred_equal']['n']}")


if __name__ == "__main__":
    main()
