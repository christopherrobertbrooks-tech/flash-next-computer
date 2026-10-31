#!/bin/bash
# RISC-V viewer, phase 3: watch Qwenputer's computer live (needs the V100; takes its lock). From Dev-Console:
#   ssh -t ember-gateway /mnt/nvme/flashnext-test/viewer/rvlive.sh                (the default question)
#   ssh -t ember-gateway '/mnt/nvme/flashnext-test/viewer/rvlive.sh "Compute 2**64 // 7 with the calc tool."'
# (ssh joins its arguments into one remote command line, so a question needs the outer single quotes.)
cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv || exit 1
export CUDA_VISIBLE_DEVICES=GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796 PYTHONPATH=. PYTHONWARNINGS=ignore HOME=/mnt/nvme/qwenputer
Q="${1:-What is 123456789 * 987654321? Use the calc tool, then give the exact answer.}"
[ $# -gt 0 ] && shift
LOG=/mnt/nvme/flashnext-test/viewer/rvlive-stderr.log
echo "Loading Qwenputer onto the V100 (about 30 s); messages go to $LOG"
export TRANSFORMERS_VERBOSITY=error
exec 2>>"$LOG"                                  # library warnings would scramble the full-screen view
exec /mnt/nvme/humming-fast/tools/v100 /mnt/nvme/qwenputer/.venv/bin/python /mnt/nvme/flashnext-test/viewer/rvlive.py \
  /mnt/nvme/qwenputer/model "$Q" "$@" --log /mnt/nvme/flashnext-test/viewer/rvlive-mismatches.jsonl
