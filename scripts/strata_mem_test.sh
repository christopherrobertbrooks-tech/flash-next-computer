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
nvidia-smi --id=1 --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $T/mem.csv &
MON=$!
snap() { echo "== $1 $(date +%T)"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v sunshine; nvidia-smi --id=1 --query-gpu=memory.used,memory.total --format=csv,noheader; free -m | sed -n 2p; }
snap start
if [ -n "$WITH" ]; then
  IDLE=""; [ "$WITH" = computer-idle ] && IDLE=idle
  (cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv && CUDA_VISIBLE_DEVICES=$V100 PYTHONPATH=. PYTHONWARNINGS=ignore \
     setsid /mnt/nvme/qwenputer/.venv/bin/python /mnt/nvme/qwenputer/computer_holder.py /mnt/nvme/qwenputer/model $T/computer.log $IDLE \
     > $T/computer.out 2>&1 &)
  for i in $(seq 1 120); do grep -q ready $T/computer.log 2>/dev/null && break; sleep 1; done
  if ! grep -q ready $T/computer.log 2>/dev/null; then
    echo "ABORT: the computer process did not become ready"; tail -5 $T/computer.out; kill $MON; exit 1
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
  curl -s localhost:8091/v1/chat/completions -H 'Content-Type: application/json' -d "$PROMPT" > $T/reply.json
  echo "request took $(python3 -c "print(round($(date +%s.%N)-$t1,1))") s"
  python3 -c "import json;d=json.load(open('$T/reply.json'));print('usage', d.get('usage'));print(str(d)[:300] if 'choices' not in d else d['choices'][0]['message'].get('content','')[:300])"
  snap "after request"
fi
kill -TERM -- -$SRV 2>/dev/null; for i in $(seq 40); do kill -0 -- -$SRV 2>/dev/null || break; sleep 0.5; done; kill -KILL -- -$SRV 2>/dev/null
if [ -n "$WITH" ]; then
  pkill -f "[c]omputer_holder.py /mnt/nvme/qwenputer/model $T/computer.log"; sleep 1; echo "== computer speed log"; cat $T/computer.log
fi
sleep 2; kill $MON
snap end
echo "== peak V100 MiB during the run: $(cut -d, -f2 $T/mem.csv | sort -n | tail -1)"
echo "== strata log: memory and slots"
grep -E -i "slots|free|reserved|trace|draft head|VRAM|pinned" $T/strata.log | head -40
