#!/usr/bin/env bash
# 完整流程的續跑腳本。可在任何時間點中斷（Ctrl+C、關機、系統強制重啟都行），
# 之後重新執行這支腳本即可接續 —— 每一步都會跳過已完成的部分。
#
# 用法：
#   bash run_pipeline.sh
#   RUN=MAIN bash run_pipeline.sh
#
# 前置：Ollama 服務需在執行中。
#
# 注意：01（資料切分）與 03（建語料庫）若已完成就會刻意跳過，不重跑。
# 它們的輸出 hash 被所有下游步驟驗證，重建一旦產生任何差異，
# 已完成條件的指紋檢查就會全部擋下 —— 那不是保護失效，是保護正確生效，
# 但代價是整輪實驗得重來。要重建請先自行清空 data/splits 或 data/corpus。
set -e
set -o pipefail   # 沒有這行，任何 `python ... | grep` 都會讓 grep 的 exit 0 掩蓋 python 的崩潰
                  # 2026-09-16 就是這樣讓一次 CUDA 崩潰被誤判成「執行成功」
trap 'echo "########## 中止：上一個步驟失敗（exit $?）##########" >&2' ERR
cd "$(dirname "$0")/src"

RUN="${RUN:-MAIN}"
PY="${PY:-/c/Users/wuc120/Code/.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8

run(){ echo "########## $* ##########"; "$PY" "$@"; }
skip(){ echo "########## 跳過 $1（$2）##########"; }

run 00_probe_env.py --run "$RUN"

if [ -f ../data/splits/split_audit.json ]; then
  skip 01_prepare_data.py "切分已存在，重跑會使既有結果的 hash 驗證失效"
else
  run 01_prepare_data.py
fi

if [ -f ../data/corpus/noaug_meta.json ]; then
  skip "03_build_corpus.py --source train" "語料庫已存在"
else
  run 03_build_corpus.py --source train
fi

run 06_run_condition.py   --condition S0  --run "$RUN" --progress-every 500
run 05_optimize_prompt.py --mode norag    --run "$RUN"
run 06_run_condition.py   --condition S1  --run "$RUN" --progress-every 500
run 06_run_condition.py   --condition S2a --run "$RUN" --progress-every 500
run 06_run_condition.py   --condition S2b --run "$RUN" --progress-every 500

run 04_augment.py --run "$RUN" --n-paraphrases 2 --progress-every 500

if [ -f ../data/corpus/aug_meta.json ]; then
  skip "03_build_corpus.py --source train+aug" "擴增語料庫已存在"
else
  run 03_build_corpus.py --source train+aug
fi

run 06_run_condition.py   --condition S3a --run "$RUN" --progress-every 500
run 06_run_condition.py   --condition S3b --run "$RUN" --progress-every 500
run 05_optimize_prompt.py --mode rag_aug  --run "$RUN"
run 06_run_condition.py   --condition S4  --run "$RUN" --progress-every 500

# 釋放 Ollama 佔用的 4.9GB VRAM，讓 DistilBERT 微調有足夠空間
echo "########## ollama stop ##########"; ollama stop llama3.1 || true

run 07_baselines.py  --run "$RUN" --seeds 3
run 08_metrics.py    --run "$RUN"
run 09_stats.py      --run "$RUN"
run 11_verify_all.py --run "$RUN"
echo "########## ALL DONE ##########"
