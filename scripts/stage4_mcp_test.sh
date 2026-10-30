#!/bin/bash
# Stage 4: Strata 0.1.41 (vision off, its own launch on port 8091, live config untouched) with the calc MCP tool.
# usage: stage4_mcp_test.sh LABEL computer|python   (inside the V100 lock)
# Every backend runs the same layout: --resident-experts keeps all 12,288 experts in VRAM (it overrides
# --expert-cache, so a smaller slot count can't be set this way); the computer then fits in the ~660 MiB left.
set -uo pipefail
LABEL=$1; BACKEND=$2
T=/mnt/nvme/flashnext-test/stage4/$LABEL; mkdir -p $T
S=/mnt/steam/Strata-0141; V100=GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796
if curl -s -o /dev/null localhost:8091/health; then echo "ABORT: something already answers on port 8091"; exit 1; fi
python3 - <<PY
import json
c = json.load(open("$S/strata-coder-iq1_m-novision.json"))
c["log"] = "$T/strata.log"
c["mcp_servers"] = {"computer": {
    "command": "/mnt/nvme/qwenputer/.venv/bin/python",
    "args": ["/mnt/nvme/flashnext-test/computer/mcp_server.py", "--backend", "$BACKEND",
             "--model", "/mnt/nvme/qwenputer/model", "--log", "$T/calls.jsonl"],
    "cwd": "/mnt/nvme/qwenputer/qwenputer/experimental/riscv",
    "env": {"CUDA_VISIBLE_DEVICES": "$V100", "PYTHONPATH": ".:/mnt/nvme/flashnext-test/computer",
            "PYTHONWARNINGS": "ignore"}}}
c["mcp"] = {"timeout_s": 180, "start_timeout_s": 180, "max_rounds": 8}
json.dump(c, open("$T/config.json", "w"), indent=1)
PY
SRV=""
cleanup() { [ -n "$SRV" ] && { kill -TERM -- -$SRV 2>/dev/null; for i in $(seq 40); do kill -0 -- -$SRV 2>/dev/null || break; sleep 0.5; done; kill -KILL -- -$SRV 2>/dev/null; }; [ -n "${MON:-}" ] && kill $MON 2>/dev/null; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
FAIL=0
nvidia-smi --id=1 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $T/mem.csv &
MON=$!
cd $S
STRATA_TRACE=1 STRATA_ALLOWED_HOSTS="localhost:8091,127.0.0.1:8091" setsid .venv/bin/python serve/server.py --engine strata \
  --config $T/config.json --port 8091 > $T/server.log 2>&1 &
SRV=$!
t0=$(date +%s); ok=0
for i in $(seq 1 900); do curl -sf localhost:8091/health > /dev/null 2>&1 && { ok=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 1; done
echo "strata ready: $ok after $(( $(date +%s) - t0 )) s"
[ $ok = 1 ] || { tail -20 $T/server.log; exit 1; }
sleep 3
echo "== /mcp status"; curl -s localhost:8091/mcp | python3 -c "import json,sys;d=json.load(sys.stdin);print(json.dumps(d)[:600])"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v -E "sunshine|indextts"
python3 /mnt/nvme/flashnext-test/computer/stage4_requests.py http://localhost:8091 $T/replies.json || FAIL=1
echo "== tool calls (calls.jsonl)"; cat $T/calls.jsonl 2>/dev/null
echo "== strata: tool lines, reuse"; grep -E "tool computer__calc|reused|prompt [0-9]+ tokens" $T/server.log $T/strata.log 2>/dev/null | cut -c1-200 | tail -30
grep -o "expert cache [0-9]* slots" $T/strata.log | head -1; grep -o "[0-9]* MiB of VRAM free with everything loaded" $T/strata.log | head -1
echo "== peak V100 MiB: $(cut -d, -f2 $T/mem.csv | sort -n | tail -1)"
[ $FAIL = 0 ] && echo "RUN OK" || echo "RUN FAILED"
exit $FAIL
