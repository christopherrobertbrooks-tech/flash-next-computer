#!/bin/bash
# Stage 3: Strata 0.1.41 with vision off, its own launch on port 8091 (not llama-swap), memory trace on.
# usage: [CFG=config.json] strata_mem_test.sh LABEL [computer | computer-idle]   (inside the V100 lock; default config: novision)
set -uo pipefail
LABEL=$1; WITH=${2:-}; CFG=${CFG:-strata-coder-iq1_m-novision.json}
T=/mnt/nvme/flashnext-test/$LABEL; mkdir -p $T; cd $T
S=/mnt/steam/Strata-0141; V100=GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796
python3 - <<PY
import json
c = json.load(open("$S/$CFG"))
c["log"] = "$T/strata.log"
json.dump(c, open("$T/config.json", "w"), indent=1)
PY
if curl -s -o /dev/null localhost:8091/health; then echo "ABORT: something already answers on port 8091"; exit 1; fi
SRV=""; HOLD=""
cleanup() {                         # only the processes this script started (their own process groups)
  [ -n "$SRV" ] && { kill -TERM -- -$SRV 2>/dev/null; for i in $(seq 40); do kill -0 -- -$SRV 2>/dev/null || break; sleep 0.5; done; kill -KILL -- -$SRV 2>/dev/null; }
  [ -n "$HOLD" ] && { kill -TERM -- -$HOLD 2>/dev/null; sleep 1; kill -KILL -- -$HOLD 2>/dev/null; }
  [ -n "${MON:-}" ] && kill $MON 2>/dev/null
}
trap cleanup EXIT                   # INT/TERM exit (running cleanup), they don't carry on
trap 'exit 130' INT; trap 'exit 143' TERM
FAIL=0
nvidia-smi --id=1 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $T/mem.csv &
MON=$!
snap() { echo "== $1 $(date +%T)"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v sunshine; nvidia-smi --id=1 --query-gpu=memory.used,memory.total --format=csv,noheader; free -m | sed -n 2p; }
snap start
if [ -n "$WITH" ]; then
  IDLE=""; [ "$WITH" = computer-idle ] && IDLE=idle
  cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv
  : > $T/computer.log                # the holder appends: an old "ready" line must not count
  CUDA_VISIBLE_DEVICES=$V100 PYTHONPATH=. PYTHONWARNINGS=ignore \
     setsid /mnt/nvme/qwenputer/.venv/bin/python /mnt/nvme/qwenputer/computer_holder.py /mnt/nvme/qwenputer/model $T/computer.log $IDLE \
     > $T/computer.out 2>&1 &
  HOLD=$!
  for i in $(seq 1 120); do grep -q ready $T/computer.log 2>/dev/null && break; sleep 1; done
  if ! grep -q ready $T/computer.log 2>/dev/null; then
    echo "ABORT: the computer process did not become ready"; tail -5 $T/computer.out; exit 1
  fi
  cat $T/computer.log; snap "computer running"
fi
cd $S
STRATA_TRACE=1 STRATA_ALLOWED_HOSTS="localhost:8091,127.0.0.1:8091" setsid .venv/bin/python serve/server.py --engine strata \
  --config $T/config.json --port 8091 > $T/server.log 2>&1 &
SRV=$!
t0=$(date +%s); ok=0
for i in $(seq 1 900); do
  curl -sf localhost:8091/health > /dev/null 2>&1 && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || break
  sleep 1
done
echo "strata ready: $ok after $(( $(date +%s) - t0 )) s"
[ $ok = 1 ] || { echo "FAIL: strata did not become ready"; tail -20 $T/server.log; FAIL=1; }
snap "strata loaded"
if [ $ok = 1 ]; then
  PROMPT=$(python3 - <<'PY'
import json
code = open("/mnt/steam/Strata-0141/serve/frontend.py").read()[:6000]
q = "Here is part of a Python file:\n\n```python\n" + code + "\n```\n\nExplain in a short paragraph what the OutputParser-related code at the start does, then write one unit test for it."
print(json.dumps({"model": "qwen3.8-flash-next-coder-iq1_m", "messages": [{"role": "user", "content": q}],
                  "max_tokens": 600, "temperature": 0, "stream": False}))
PY
)
  t1=$(date +%s.%N)
  code=$(curl -s -o $T/reply.json -w '%{http_code}' localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d "$PROMPT")
  echo "request took $(python3 -c "print(round($(date +%s.%N)-$t1,1))") s, HTTP $code"
  [ "$code" = 200 ] || { echo "FAIL: HTTP $code"; FAIL=1; }
  python3 - "$T/reply.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    c = d["choices"][0]
    print("usage", d.get("usage"), "| finish", c.get("finish_reason"))
    print("content:", repr((c["message"].get("content") or "")[:200]), "| reasoning chars:",
          len(c["message"].get("reasoning_content") or ""))
except Exception as e:  # noqa: BLE001
    print("BAD REPLY:", type(e).__name__, open(sys.argv[1]).read()[:300])
    sys.exit(1)
PY
  [ $? = 0 ] || { echo "FAIL: bad reply"; FAIL=1; }
  snap "after request"
fi
[ -n "$WITH" ] && { echo "== computer speed log"; cat $T/computer.log; }
cleanup; SRV=""; HOLD=""; MON=""
sleep 2
snap end
echo "== peak V100 MiB during the run: $(cut -d, -f2 $T/mem.csv | sort -n | tail -1)"
echo "== strata log: memory and slots"
grep -E -i "slots|free|reserved|trace|draft head|VRAM|pinned" $T/strata.log | head -40
[ $FAIL = 0 ] && echo "RUN OK" || echo "RUN FAILED"
exit $FAIL
