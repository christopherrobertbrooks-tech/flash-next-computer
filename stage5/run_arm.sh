#!/bin/bash
# Stage 5: one arm = one fresh Strata 0.1.41 launch (vision off, all experts resident, port 8091, live config
# untouched), then ask.py over a question file. Run inside the V100 lock.
# usage: run_arm.sh LABEL A|B|C QUESTIONS.jsonl [--subset every4]
#   A: no tools.  B: calc from the persistent host-Python MCP server.  C: calc from the computer service.
#   B and C offer the same tool under the same name (math__calc) and description; settings are identical.
set -uo pipefail
LABEL=$1; ARM=$2; Q=$3; shift 3
T=/mnt/nvme/flashnext-test/stage5/$LABEL; mkdir -p $T
S=/mnt/steam/Strata-0141; V100=GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796
HERE=/mnt/nvme/flashnext-test/stage5-code
if curl -s -o /dev/null localhost:8091/health; then echo "ABORT: something already answers on port 8091"; exit 1; fi
if [ -e $T/replies.jsonl ]; then echo "ABORT: $T already has replies (never overwrite a run)"; exit 1; fi
case $ARM in A) BACKEND="";; B) BACKEND=python;; C) BACKEND=computer;; *) echo "ARM must be A, B or C"; exit 1;; esac
python3 - <<PY
import json
c = json.load(open("$S/strata-coder-iq1_m-novision.json"))
c["log"] = "$T/strata.log"
if "$BACKEND":
    c["mcp_servers"] = {"math": {
        "command": "/mnt/nvme/qwenputer/.venv/bin/python",
        "args": ["/mnt/nvme/flashnext-test/computer/mcp_server.py", "--backend", "$BACKEND",
                 "--model", "/mnt/nvme/qwenputer/model", "--log", "$T/calls.jsonl",
                 "--question-file", "$T/current_question"],
        "cwd": "/mnt/nvme/qwenputer/qwenputer/experimental/riscv",
        "env": {"CUDA_VISIBLE_DEVICES": "$V100", "PYTHONPATH": ".:/mnt/nvme/flashnext-test/computer",
                "PYTHONWARNINGS": "ignore"}}}
    c["mcp"] = {"timeout_s": 180, "start_timeout_s": 180, "max_rounds": 20}
json.dump(c, open("$T/config.json", "w"), indent=1)
PY
SRV=""
cleanup() { [ -n "$SRV" ] && { kill -TERM -- -$SRV 2>/dev/null; for i in $(seq 40); do kill -0 -- -$SRV 2>/dev/null || break; sleep 0.5; done; kill -KILL -- -$SRV 2>/dev/null; }; [ -n "${MON:-}" ] && kill $MON 2>/dev/null; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
nvidia-smi --id=1 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $T/mem.csv &
MON=$!
cd $S
STRATA_TRACE=1 STRATA_ALLOWED_HOSTS="localhost:8091,127.0.0.1:8091" setsid .venv/bin/python serve/server.py --engine strata \
  --config $T/config.json --port 8091 > $T/server.log 2>&1 &
SRV=$!
t0=$(date +%s); ok=0
for i in $(seq 1 900); do curl -sf localhost:8091/health > /dev/null 2>&1 && { ok=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 1; done
echo "strata ready: $ok after $(( $(date +%s) - t0 )) s" | tee $T/startup.txt
[ $ok = 1 ] || { tail -20 $T/server.log; exit 1; }
if [ -n "$BACKEND" ]; then                 # wait until the tool server has listed its tool
  for i in $(seq 1 180); do curl -s localhost:8091/mcp | grep -q '"calc"' && break; sleep 1; done
  curl -s localhost:8091/mcp | grep -q '"calc"' || { echo "ABORT: the calc tool never appeared"; exit 1; }
  echo "tool ready after $(( $(date +%s) - t0 )) s from launch" | tee -a $T/startup.txt
fi
echo "== /mcp: $(curl -s localhost:8091/mcp | head -c 300)"
sha256sum $Q | tee $T/questions.sha256
python3 $HERE/ask.py http://localhost:8091 $ARM $Q $T "$@"
RC=$?
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v -E "sunshine|indextts" > $T/gpu-apps.txt
ps -eo pid,rss,args | grep "[m]cp_server.py" | awk '{print "tool process pid", $1, "host RSS MiB now (snapshot):", int($2/1024)}' | tee $T/tool-rss.txt
grep -o "expert cache [0-9]* slots" $T/strata.log | head -1 | tee $T/slots.txt
echo "== peak V100 MiB: $(cut -d, -f2 $T/mem.csv | sort -n | tail -1)" | tee $T/peak.txt
echo "ask.py exit $RC"
exit $RC
