#!/bin/bash
# EARLYSTOP-PLAN v2, the GPU parts (inside the V100 lock):
#   CUDA_VISIBLE_DEVICES=<V100 UUID> /mnt/nvme/humming-fast/tools/v100 ./earlystop.sh unchanged OUT_DIR
#     identity check, the token-mode preflight gate, then unchanged arm F on the fresh questions with --record-decisions
#   CUDA_VISIBLE_DEVICES=<V100 UUID> /mnt/nvme/humming-fast/tools/v100 ./earlystop.sh live OUT_DIR LIVE_QUESTIONS.jsonl
#     identity check, then arm F with the frozen rule on (--early-stop --record-decisions) on the live-check questions
# The scoring and the offline evaluation run between the two (llamacpp/earlystop_analysis.py, Dev-Console).
cd /mnt/nvme/flashnext-test/llamacpp || exit 1
PART=$1; OUT=$2
P=/mnt/nvme/qwenputer/.venv/bin/python
export PYTHONPATH=/mnt/nvme/qwenputer/qwenputer/experimental/riscv TRANSFORMERS_VERBOSITY=error HOME=/mnt/nvme/qwenputer
GGUF=qwen35-4b-q8_0.gguf
set -eo pipefail
mkdir "$OUT" || { echo "can't create $OUT (it may exist): never overwrite a run"; exit 1; }
echo "== $PART: identity check $(date +%r)"
$P -c "import sys; sys.path.insert(0, '.'); import run_split as RS; print(RS.verify_artifacts('earlystop-artifacts.json'), 'artifacts match')"
if [ "$PART" = unchanged ]; then
  echo "== preflight, token mode $(date +%r)"
  $P run_helper_fixtures.py --mode plain ./qwen_half_helper $GGUF fixtures-v1 "$OUT/preflight.json" 2> "$OUT/preflight-helper.log"
  CUDA_VISIBLE_DEVICES= $P stage0_gate.py fixtures-v1 "$OUT/preflight.json" plain-hf-v2.json plain-llama.json | tee "$OUT/preflight-gate.txt"
  echo "== unchanged F, 160 fresh questions $(date +%r)"
  $P run_split.py /mnt/nvme/qwenputer/model $GGUF ./qwen_half_helper fresh-questions.jsonl "$OUT/run" \
    --lm-input tokens --arm F --early-stop off --record-decisions --expect earlystop-artifacts.json
elif [ "$PART" = live ]; then
  LIVEQ=$3
  echo "== live check: F with the rule, $(wc -l < "$LIVEQ") questions $(date +%r)"
  $P run_split.py /mnt/nvme/qwenputer/model $GGUF ./qwen_half_helper "$LIVEQ" "$OUT/run" \
    --lm-input tokens --arm F --early-stop on --record-decisions --expect earlystop-artifacts.json
else
  echo "usage: earlystop.sh unchanged|live OUT_DIR [LIVE_QUESTIONS]"; exit 1
fi
echo "== done $(date +%r)"
