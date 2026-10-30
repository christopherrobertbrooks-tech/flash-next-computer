# Pass 2 questions: flash-next-computer stages 2-3

Only after pass 1 is pushed. Now read `README.md`, `results/*.md` and `mapping/`.

1. **Where does the write-up disagree with your pass 1?** Who's right, and why?
2. **The Strata map (`mapping/strata-0141-map.md`):** spot-check its key claims against the source: no raw-token API;
   the server's internal hidden-token continuation (`server.py:3621-3654`); server-side MCP tools re-rendering
   through the template; GDN state only rolled back via checkpoints; the 243 free vocabulary slots; the memory lines.
3. **"Vision off + the computer costs nothing compared with today's vision-on setup" (10.4 vs 10.5 s):** fair, from
   one prompt, three runs per setup, at the lock's fixed clocks? And the ~11% idle cost attributed to 136 missing
   experts?
4. **The service's check:** `computer.py` compares every circuit token with upstream's reference machine. Is that
   independent enough of the executor? Are session resets and multi-call state handled correctly? Is "handed back
   early" detected properly? Anything the corpus misses (e.g. Flash-Next's tokenizer writing the call differently
   from Qwen3.5's, concurrency, a call while Strata decodes)?
5. **Stage 4 design:** a separate computer process, called server-side while Flash-Next waits (take turns). Which
   hook (MCP rounds, or the continuation path) and what must be proven first (prompt-cache reuse, GDN/KV state,
   MTP verification, latency)?
6. **Stage 5 prospects:** the computer takes ~1.6 s per call where Python takes microseconds. What could still make
   it worth it, what's the fairest host-Python baseline, and would you recommend continuing to stage 4?
