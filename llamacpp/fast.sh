#!/bin/bash
# FAST-PLAN: arms F then G, in one V100 session, inside the V100 lock:
#   CUDA_VISIBLE_DEVICES=<V100 UUID> /mnt/nvme/humming-fast/tools/v100 ./fast.sh OUT_DIR
# Each arm runs on its own: its identity check, its preflight (the stage-0 gate in token mode with its own GGUF),
# then its 160 questions. A failure in F doesn't skip G; every attempt folder is kept; the exit code is nonzero if
# either arm didn't complete.
cd /mnt/nvme/flashnext-test/llamacpp || exit 1
OUT=$1
P=/mnt/nvme/qwenputer/.venv/bin/python
export PYTHONPATH=/mnt/nvme/qwenputer/qwenputer/experimental/riscv TRANSFORMERS_VERBOSITY=error HOME=/mnt/nvme/qwenputer
mkdir "$OUT" || { echo "can't create $OUT (it may exist): never overwrite a run"; exit 1; }   # atomic: the gate
declare -A GGUF=([F]=qwen35-4b-q8_0.gguf [G]=/mnt/nvme/qwen35-4b/gguf/qwen35-4b-f16.gguf)
rc=0
for ARM in F G; do
  A="$OUT/$ARM"
  mkdir "$A" || { echo "can't create $A: arm $ARM not attempted"; rc=1; continue; }
  (
    set -eo pipefail
    echo "== arm $ARM: identity check $(date +%r)"
    $P -c "import sys; sys.path.insert(0, '.'); import run_split as RS; print(RS.verify_artifacts('fast-artifacts-$ARM.json'), 'artifacts match')"
    echo "== arm $ARM: preflight, token mode $(date +%r)"
    $P run_helper_fixtures.py --mode plain ./qwen_half_helper "${GGUF[$ARM]}" fixtures-v1 "$A/preflight.json" 2> "$A/preflight-helper.log"
    CUDA_VISIBLE_DEVICES= $P stage0_gate.py fixtures-v1 "$A/preflight.json" plain-hf-v2.json plain-llama.json | tee "$A/preflight-gate.txt"
    echo "== arm $ARM: 160 questions $(date +%r)"
    $P run_split.py /mnt/nvme/qwenputer/model "${GGUF[$ARM]}" ./qwen_half_helper final-primary.jsonl "$A/run" \
      --lm-input tokens --arm $ARM --early-stop off --expect fast-artifacts-$ARM.json
  ) > "$A/arm.out" 2>&1
  st=$?
  echo "== arm $ARM: exit $st $(date +%r)" | tee -a "$A/arm.out"
  [ $st -ne 0 ] && rc=1
done
echo "== done $(date +%r)"
exit $rc
