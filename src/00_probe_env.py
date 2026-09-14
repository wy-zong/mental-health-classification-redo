"""探測並釘死執行環境，產出 run_manifest.json。

之後每一支腳本都會把自己用到的環境指紋寫進輸出，由 11_verify_all.py 斷言全程一致。
沒有這一步，「重跑得到不同數字」就無從判斷是模型換了、量化換了，還是程式改了。

原始實驗完全沒有這層記錄：論文寫的模型設定無法對應到任何可驗證的執行痕跡，
審稿人問實作細節時只能事後回推。

用法：
    python 00_probe_env.py [--run RUN_NAME]
"""
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PACKAGES = [
    "pandas", "numpy", "sklearn", "scipy", "sentence_transformers",
    "faiss", "optuna", "transformers", "torch", "ollama", "matplotlib",
]


def probe_packages() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "<no __version__>")
        except ImportError:
            versions[name] = None
    return versions


def probe_hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        info["torch_probe_error"] = str(exc)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            info["nvidia_smi"] = out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def probe_model(model: str) -> dict:
    """探測 LLM 的實際身分：digest 與量化等級。

    這兩項是「同一個模型」的唯一可驗證證據 —— 模型名稱（llama3.1）會隨上游更新
    指向不同權重。探不到就中止，不接受任何預設值或佔位字串。
    """
    import ollama

    info: dict = {"model_name": model}

    try:
        listing = ollama.list()
    except Exception as exc:  # noqa: BLE001
        C.die(f"無法連上 Ollama：{exc}\n請確認 Ollama 服務已啟動（ollama serve）")

    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    for entry in models:
        name = entry.get("model") if isinstance(entry, dict) else getattr(entry, "model", None)
        if name in (model, f"{model}:latest"):
            digest = entry.get("digest") if isinstance(entry, dict) else getattr(entry, "digest", None)
            size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", None)
            info["digest"] = digest
            info["size_bytes"] = size
            break
    else:
        available = [
            (e.get("model") if isinstance(e, dict) else getattr(e, "model", "?")) for e in models
        ]
        C.die(f"Ollama 沒有模型 {model!r}；目前有：{available}\n"
              f"請先執行：ollama pull {model}")

    show = ollama.show(model)
    details = show.get("details", {}) if isinstance(show, dict) else getattr(show, "details", {}) or {}
    if isinstance(details, dict):
        info["quantization_level"] = details.get("quantization_level")
        info["parameter_size"] = details.get("parameter_size")
        info["family"] = details.get("family")
    else:
        info["quantization_level"] = getattr(details, "quantization_level", None)
        info["parameter_size"] = getattr(details, "parameter_size", None)
        info["family"] = getattr(details, "family", None)

    if not info.get("quantization_level"):
        C.die("探測不到量化等級 —— 不接受未知的模型設定，請檢查 Ollama 版本")
    if not info.get("digest"):
        C.die("探測不到 model digest —— 無法證明後續各步驟用的是同一組權重")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MAIN", help="執行代號，輸出到 runs/<RUN>/")
    ap.add_argument("--model", default=C.MODEL_NAME)
    args = ap.parse_args()

    out_dir = C.RUNS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    C.log(f"探測執行環境（run={args.run}）…")
    manifest = {
        "run": args.run,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "model": probe_model(args.model),
        "generation_options": C.GEN_OPTIONS,
        "hardware": probe_hardware(),
        "packages": probe_packages(),
        "labels": C.LABELS,
        "raw_dataset_sha256": C.RAW_DATASET_SHA256,
    }

    m = manifest["model"]
    C.log(f"  模型 {m['model_name']}：{m.get('parameter_size')} / "
          f"{m.get('quantization_level')}  digest={str(m.get('digest'))[:20]}…")
    hw = manifest["hardware"]
    C.log(f"  硬體：{hw.get('gpu_name', 'CPU only')}"
          + (f"（{hw['gpu_total_memory_gb']} GB）" if hw.get("gpu_total_memory_gb") else ""))
    C.log(f"  生成參數：{C.GEN_OPTIONS}")

    path = out_dir / "run_manifest.json"
    C.save_json(path, manifest)
    C.log(f"已寫入 {path}")


if __name__ == "__main__":
    main()
