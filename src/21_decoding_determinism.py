"""15_（CUSTOM_PROMPT_EVAL）解碼確定性驗證（issue #5，審稿意見 R3C13、R3C14）。

從 test 依類別分層抽樣（預設每類 50 筆、共 200 筆，seed 42），把 15_ 存下的
C4、C5 ``rendered_prompt`` 原封不動再送 Ollama（同 15_ 的 GEN_OPTIONS，預設兩次），比對：

* 每次重跑與 15_ 存檔的 raw_response 逐字一致率、解析後預測一致率；
* 重跑之間的一致率（每次重跑前先卸載模型，等同另起一次執行）：若重跑彼此完全一致、
  卻與 15_ 存檔不同，差異來自環境漂移（例如 Ollama 版本），而不是重跑之間的隨機性；
* 以 15_ 的 FAISS 索引（12_ RebuildRagIndex）重新檢索 top-1，``retrieved_ids`` 是否一致，
  以及用重新檢索結果依 15_ 的 render 規則重組的 prompt 是否與存檔逐字相同。

開跑前確認目前 Ollama 的 model digest 與 15_ manifest、逐筆預測記錄的 digest 相同，
不同就停止。推論期間背景以 nvidia-smi 每 0.5 秒取樣 GPU 記憶體用量（供 issue #9）。

輸出（runs/NESTED_70_10_20_INCREMENTAL/CUSTOM_PROMPT_EVAL/determinism/）：
    sample_ids.csv           抽樣的 id 與標籤
    rerun_C4.jsonl、rerun_C5.jsonl  第 1 次重跑的逐筆比對（可中斷續跑）
    rerun_C4_pass2.jsonl …   第 2 次起的重跑
    sessions.jsonl           每次執行的環境（Ollama 版本、GPU、VRAM 峰值、size_vram）
    summary.json             一致率（Wilson 95% CI）、不一致清單、環境
    cache_state_probe.jsonl  （--probe-cache-state）不一致樣本在不同 KV cache 狀態下的輸出

用法：
    python 21_decoding_determinism.py [--n 200] [--passes 2] [--out 目錄]
    python 21_decoding_determinism.py --probe-cache-state
"""
from __future__ import annotations

import argparse
import json
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
OUT = E.OUT / "determinism"
MANIFEST = E.OUT / "custom_prompt_eval_manifest.json"
NVSMI_POLL_S = 0.5


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
    host = "http://localhost:11434"
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

def check_digest(R, preds: dict) -> str:
    current = R.probe_llm_identity().get("digest")
    recorded = C.load_json(MANIFEST).get("model", {}).get("digest")
    in_records = {preds[code][0]["model_digest"] for code in CODES}
    if not current or current != recorded or in_records != {recorded}:
        C.die(f"model digest 不一致：目前 {current}，15_ manifest {recorded}，逐筆預測 {in_records}")
    return current


def rerun_path(out: Path, code: str, pass_no: int) -> Path:
    return out / (f"rerun_{code}.jsonl" if pass_no == 1 else f"rerun_{code}_pass{pass_no}.jsonl")


def rerun(code: str, pass_no: int, sample: pd.DataFrame, by_id: dict, template: str, rag,
          out: Path) -> None:
    path = rerun_path(out, code, pass_no)
    done = C.done_ids(path)
    for row in sample.itertuples(index=False):
        if row.id in done:
            continue
        orig = by_id[row.id]
        docs, ids, sims = rag.search(str(row.statement))
        rebuilt = render(template, str(row.statement), docs[0])
        response = C.chat(orig["rendered_prompt"])
        pred, reason = C.parse_label(response["raw_response"])
        C.append_jsonl(path, {
            "id": row.id, "code": code, "pass": pass_no, "condition": E.CODES[code], "true_label": orig["true_label"],
            "orig_raw_response": orig["raw_response"], "rerun_raw_response": response["raw_response"],
            "raw_equal": response["raw_response"] == orig["raw_response"],
            "orig_pred_label_id": orig["pred_label_id"], "rerun_pred_label_id": pred,
            "pred_equal": pred == orig["pred_label_id"],
            "orig_parse_reason": orig["parse_reason"], "rerun_parse_reason": reason,
            "orig_retrieved_ids": orig["retrieved_ids"], "rerun_retrieved_ids": ids,
            "retrieved_equal": ids == orig["retrieved_ids"],
            "retrieved_similarity_abs_diff": abs(sims[0] - orig["retrieved_similarities"][0]),
            "prompt_rebuilt_equal": rebuilt == orig["rendered_prompt"],
            "elapsed_s": response["elapsed_s"], "prompt_eval_count": response["prompt_eval_count"],
            "eval_count": response["eval_count"],
        })
        done.add(row.id)
        C.log(f"{code} pass {pass_no}: {len(done)}/{len(sample)} id={row.id} raw_equal="
              f"{response['raw_response'] == orig['raw_response']}")


# ---------------------------------------------------------------- 彙總

def rate(records: list[dict], key: str) -> dict:
    k, n = sum(bool(r[key]) for r in records), len(records)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


def load_pass(out: Path, code: str, pass_no: int, sample: pd.DataFrame) -> list[dict]:
    path = rerun_path(out, code, pass_no)
    recs = list(C.read_jsonl(path))
    if {r["id"] for r in recs} != set(sample["id"]) or len(recs) != len(sample):
        C.die(f"{path.name} 尚未完成：{len(recs)}/{len(sample)}")
    return recs


def vs_original(recs: list[dict]) -> dict:
    keys = ("raw_equal", "pred_equal", "retrieved_equal", "prompt_rebuilt_equal")
    return {
        **{k: rate(recs, k) for k in keys},
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


def summarize(sample: pd.DataFrame, out: Path, passes: int) -> dict:
    result, pooled = {}, {p: [] for p in range(1, passes + 1)}
    for code in CODES:
        runs = {p: load_pass(out, code, p, sample) for p in range(1, passes + 1)}
        for p, recs in runs.items():
            pooled[p] += recs
        result[code] = {
            "condition": E.CODES[code],
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


def probe_cache_state(preds: dict, out: Path) -> dict:
    """對與 15_ 不一致的樣本，檢查差異是否來自 Ollama 的 KV cache（前一個請求）狀態。

    每筆在三種狀態下各送一次：cold＝剛載入模型直接送；after_predecessor＝剛載入後
    先送 15_ 執行順序中緊接在前的那一筆，再送目標；repeat_same＝緊接著把目標再送一次。
    """
    summary = C.load_json(out / "summary.json")
    if not summary:
        C.die("請先完成重跑（summary.json 不存在）")
    path = out / "cache_state_probe.jsonl"
    path.unlink(missing_ok=True)
    rows = []
    for code in CODES:
        order = preds[code]                        # E.load_predictions 依 test.csv 順序
        raw_order = list(C.read_jsonl(E.OUT / f"predictions_{E.CODES[code]}.jsonl"))
        if [r["id"] for r in raw_order] != [r["id"] for r in order]:
            C.die(f"{code} 預測檔順序與 test.csv 不同，無法推得 15_ 的執行順序")
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
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--probe-cache-state", action="store_true",
                    help="重跑完成後，對不一致樣本檢查 KV cache 狀態的影響")
    args = ap.parse_args()

    C.single_instance("determinism")
    R = C.load_module("12_rebuild_experiment")
    prompts, conditions = E.source_15()
    template_of = {cond: prompts[p]["template"] for cond, p, _ in conditions}
    corpus_of = {cond: corpus for cond, _, corpus in conditions}

    preds = {code: recs for code, recs in E.load_predictions().items() if code in CODES}
    digest = check_digest(R, preds)
    if args.probe_cache_state:
        res = probe_cache_state(preds, args.out)
        C.log(f"cache 探測 {res['n']} 筆：與 15_ 相同 {res['equals_orig']}；"
              f"任一狀態可重現 {res['recovered_by_any_state']}；"
              f"不同狀態輸出不同 {res['outputs_differ_across_states']}")
        return
    sample = stratified_sample(args.n, args.out)

    env = environment(R)
    unload_model()   # 基準用量不含模型
    try:
        before = int(nvidia_smi("memory.used")[0])
    except Exception:  # noqa: BLE001
        before = None
    rags = {name: R.RebuildRagIndex(E.RUN, name) for name in ("noaug", "aug")}
    with VramMonitor() as mon:
        for pass_no in range(1, args.passes + 1):
            unload_model()
            for code in CODES:
                cond = E.CODES[code]
                by_id = {str(r["id"]): r for r in preds[code]}
                rerun(code, pass_no, sample, by_id, template_of[cond], rags[corpus_of[cond]],
                      args.out)
            env.setdefault("ollama_ps_after", {})[f"pass{pass_no}"] = ollama_loaded()
    env["vram"] = {"poll_interval_s": NVSMI_POLL_S, "n_samples": len(mon.samples),
                   "used_before_run_mib": before,
                   "peak_used_mib": max(mon.samples) if mon.samples else None,
                   "note": "nvidia-smi memory.used 為整張 GPU 的用量（含其他程序）；"
                           "模型本身佔用看 ollama_ps_after 的 size_vram_bytes"}
    C.append_jsonl(args.out / "sessions.jsonl", env)

    summary = {
        "run": E.RUN, "experiment": "CUSTOM_PROMPT_EVAL", "script": "src/21_decoding_determinism.py",
        "issue": "#5", "procedure": (
            f"test 依類別分層抽 {args.n} 筆（每類 {args.n // 4}，random_state=42+類別序）；"
            f"C4、C5 各以 15_ 存下的 rendered_prompt 原樣重送 Ollama {args.passes} 次（15_ 的 GEN_OPTIONS），"
            "以 15_ 的 parse_label 解析；另以 15_ 的 FAISS 索引重新檢索 top-1 並依 15_ 規則重組 prompt"),
        "inputs": {"manifest_sha256": C.sha256_file(MANIFEST),
                   "test_sha256": C.sha256_file(E.TEST_PATH),
                   **{f"predictions_{code}_sha256": C.sha256_file(E.OUT / f"predictions_{E.CODES[code]}.jsonl")
                      for code in CODES}},
        "model_digest": digest,
        "llm_generation_options": C.GEN_OPTIONS,
        "results": summarize(sample, args.out, args.passes),
        "sessions": list(C.read_jsonl(args.out / "sessions.jsonl")),
    }
    C.save_json(args.out / "summary.json", summary)
    C.log(f"已寫入 {args.out / 'summary.json'}")
    for code in (*CODES, "pooled"):
        res = summary["results"][code]
        for name, v in res["vs_original"].items():
            C.log(f"{code} {name} vs 15_: raw {v['raw_equal']['k']}/{v['raw_equal']['n']}  "
                  f"pred {v['pred_equal']['k']}/{v['pred_equal']['n']}  "
                  f"retrieval {v['retrieved_equal']['k']}/{v['retrieved_equal']['n']}")
        for name, v in res["between_passes"].items():
            C.log(f"{code} {name}: raw {v['raw_equal']['k']}/{v['raw_equal']['n']}  "
                  f"pred {v['pred_equal']['k']}/{v['pred_equal']['n']}")


if __name__ == "__main__":
    main()
