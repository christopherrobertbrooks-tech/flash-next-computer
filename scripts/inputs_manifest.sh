#!/bin/bash
# Input inventory for stages 3-4: a full sha256 of every small file the results depend on, including Flash-Next's own
# tokenizer and chat template (every regular file under the pack, recursively). Files over 256 MiB (GGUF shards,
# expert pack, MTP weights) get only a PARTIAL FINGERPRINT: size + modification time + a sha256 of their first and last
# 64 MiB, which cannot detect a change in the middle. So this is provenance, not a full-content freeze of the weights.
# Run on the gateway; writes to stdout. Per-run configs are in the repo (results/stage4-runs/*/config.json).
set -uo pipefail
S=/mnt/steam/Strata-0141; D=/mnt/steam/Strata-data-0140; Q=/mnt/nvme/qwenputer; R=$Q/qwenputer/experimental/riscv
C=/mnt/nvme/flashnext-test/computer
full() { for f in "$@"; do [ -f "$f" ] && echo "sha256 $(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $f"; done; }
part() { for f in "$@"; do [ -f "$f" ] || continue
  h=$( { head -c 67108864 "$f"; tail -c 67108864 "$f"; } | sha256sum | cut -c1-64)
  echo "partial $h $(stat -c '%s %Y' "$f") $f"; done; }
echo "# flash-next-computer input inventory (full sha256 = complete; partial = first/last 64 MiB only), $(date -Is)"
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
auto() { while IFS= read -r f; do if [ $(stat -c %s "$f") -gt 268435456 ]; then part "$f"; else full "$f"; fi; done; }
find $D/packs/coder-iq1_m $D/mtp/rt -type f | sort | auto
