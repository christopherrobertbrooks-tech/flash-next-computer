# Stage 1 review, pass 2 (open only after pass 1 is pushed)

Now read `results/llamacpp-stage1.md` and compare it with your pass 1. Questions:
1. Is anything on the page overstated or missing? In particular "speed is the robust result" (text tokens/s 72.7 vs
   36.8 on similar total text; median 4.67 vs 8.63 s) and the accuracy reading (129 vs 130, inconclusive).
2. inside-4b's run_plain.py on main differs from the copy that ran R (diagnostic fields; the computer-budget edge `>=`
   vs `>`). Is using main's copy acceptable, given R never came near the cap?
3. Memory: the helper's nvidia-smi process figure vs R's PyTorch peak allocated. Is there a fairer like-for-like
   number we should record in stage 3 (both processes, same tool)?
4. Before stage 2 (Qwenputer's Qwen half in llama.cpp fed rows, the circuit on upstream's executor): anything in the
   stage 1 runner, helper or client that must change? The plan's stage 2 controls are in `LLAMACPP-PLAN.md`; Muse
   suggested also checking where every op runs (`GGML_SCHED_DEBUG=2`) and testing a bf16 output head (the source
   weights are bf16, so it loses nothing vs the current f32 head and halves its size). Worth adding?
Append "Pass 2" to `REVIEW-LLAMACPP-STAGE1-FINDINGS.md`, commit, push.
