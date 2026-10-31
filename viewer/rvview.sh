#!/bin/bash
# RISC-V viewer, phase 1 (CPU only; no model). From Dev-Console:  ssh -t ember-gateway /mnt/nvme/flashnext-test/viewer/rvview.sh --program call
cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv || exit 1
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. exec /mnt/nvme/qwenputer/.venv/bin/python /mnt/nvme/flashnext-test/viewer/rvview.py "$@"
