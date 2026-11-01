#!/bin/bash
# LLAMACPP-PLAN v3 stage 3: preflight then the registered run, in one V100 session. Run inside the V100 lock:
#   CUDA_VISIBLE_DEVICES=<V100 UUID> /mnt/nvme/humming-fast/tools/v100 ./stage3.sh stage3-artifacts.json OUT_DIR
set -euo pipefail
cd /mnt/nvme/flashnext-test/llamacpp
EXPECT=$1; OUT=$2
P=/mnt/nvme/qwenputer/.venv/bin/python
export PYTHONPATH=/mnt/nvme/qwenputer/qwenputer/experimental/riscv TRANSFORMERS_VERBOSITY=error HOME=/mnt/nvme/qwenputer
[ -e "$OUT" ] && { echo "$OUT exists: never overwrite a run"; exit 1; }
mkdir "$OUT"
echo "== identity check $(date +%r)"
$P -c "import sys; sys.path.insert(0, '.'); import run_split as RS; print(RS.verify_artifacts('$EXPECT'), 'artifacts match')"
echo "== preflight: stage-0 split comparison with this helper and GGUF $(date +%r)"
$P run_helper_fixtures.py --mode split ./qwen_half_helper qwen-half-f16-bf16head.gguf fixtures-v1 "$OUT/preflight-split.json" 2> "$OUT/preflight-helper.log"
CUDA_VISIBLE_DEVICES= $P stage0_gate.py fixtures-v1 "$OUT/preflight-split.json" plain-hf-v2.json plain-llama.json | tee "$OUT/preflight-gate.txt"
echo "== stage 3: 160 questions $(date +%r)"
$P run_split.py /mnt/nvme/qwenputer/model qwen-half-f16-bf16head.gguf ./qwen_half_helper final-primary.jsonl "$OUT/run" --expect "$EXPECT"
echo "== done $(date +%r)"
