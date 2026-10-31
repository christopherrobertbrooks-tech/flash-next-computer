#!/bin/bash
# RISC-V viewer, phase 3: watch Qwenputer's computer live (needs the V100; takes its lock). From Dev-Console:
#   ssh -t ember-gateway /mnt/nvme/flashnext-test/viewer/rvlive.sh "What is 123456789 * 987654321? Use the calc tool."
cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv || exit 1
export CUDA_VISIBLE_DEVICES=GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796 PYTHONPATH=. PYTHONWARNINGS=ignore HOME=/mnt/nvme/qwenputer
exec /mnt/nvme/humming-fast/tools/v100 /mnt/nvme/qwenputer/.venv/bin/python /mnt/nvme/flashnext-test/viewer/rvlive.py \
  /mnt/nvme/qwenputer/model "$@" --log /mnt/nvme/flashnext-test/viewer/rvlive-mismatches.jsonl
