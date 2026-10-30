#!/bin/bash
# Frozen inputs for stages 3-4: a sha256 of every file the results depend on. Model weight files (GGUF shards, the
# expert pack, MTP) are many GB: they get size + modification time + a sha256 of their first and last 64 MiB, marked
# "partial". Run on the gateway; writes the manifest to stdout.
set -uo pipefail
S=/mnt/steam/Strata-0141; D=/mnt/steam/Strata-data-0140; Q=/mnt/nvme/qwenputer; R=$Q/qwenputer/experimental/riscv
C=/mnt/nvme/flashnext-test/computer
full() { for f in "$@"; do [ -f "$f" ] && echo "sha256 $(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $f"; done; }
part() { for f in "$@"; do [ -f "$f" ] || continue
  h=$( { head -c 67108864 "$f"; tail -c 67108864 "$f"; } | sha256sum | cut -c1-64)
  echo "partial $h $(stat -c '%s %Y' "$f") $f"; done; }
echo "# flash-next-computer frozen inputs, $(date -Is)"
echo "git qwenputer $(git -C $Q/qwenputer rev-parse HEAD) dirty=$(git -C $Q/qwenputer status --porcelain | wc -l)"
echo "git strata $(git -C $S rev-parse HEAD 2>/dev/null || echo none)"
echo "python $($Q/.venv/bin/python -c 'import torch,triton,transformers;print(torch.__version__,triton.__version__,transformers.__version__)' 2>/dev/null)"
echo "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
full $R/tests/data/rv/calc-bridge.elf $R/chipformer/rv/env/calc/calc.c $R/chipformer/rv/calc.py $R/chipformer/rv/bridge.py \
     $R/chipformer/rv/tritonexec.py $R/chipformer/rv/spec.py
full $Q/model/config.json $Q/model/chipformer.json $Q/model/tokenizer.json $Q/model/tokenizer_config.json
full $S/engine-cuda12/strata $S/engine-cuda12/strata-vision $S/serve/server.py $S/serve/mcp.py $S/serve/frontend.py \
     $S/strata-coder-iq1_m.json $S/strata-coder-iq1_m-novision.json $S/data/expert-profile-coder.bin
full $C/*.py /mnt/nvme/qwenputer/stage4_mcp_test.sh
part $D/models/coder-IQ1_M/*.gguf $D/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf
for f in $D/packs/coder-iq1_m/* $D/mtp/rt/*; do [ -f "$f" ] && part "$f"; done | head -200
