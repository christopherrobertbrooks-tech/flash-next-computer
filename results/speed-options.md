# Faster language part: the match test (2026-10-31, 9:35-9:55 pm)

After the inside-check (the language half's choices depend only on the text), three faster stand-ins for the
current split's language part (S: Qwenputer's widened Qwen half, f16 layers, bf16 head), tested the stage-0 way:
teacher-forced on the 12 fixtures, each candidate's greedy choice vs Qwenputer's own HF choice at the 13,746
Qwen-chosen positions, with the fail-closed stage-0 gate (rule: within 1.0 point of the plain-Qwen precision
control, 99.796%; no more than 2 questions with a disagreement at an HF margin >= 1.0). Speed: llama-bench tg128 /
pp512 on the V100 (clocks locked), 3 repeats. Development data only; not yet a registered comparison. Not yet
reviewed by Codex.

| | language part | agreement with Qwenputer's picks | clear disagreements | gate | text tokens/s (tg128) | size |
|---|---|---|---|---|---|---|
| S (stage 3) | widened half, f16, bf16 head | 99.782% | 0 | PASS | 50.7 | 12.1 GiB |
| A | widened half, **Q8_0**, bf16 head | 99.694% | 0 | PASS | 69.2 (1.37x) | 7.3 GiB |
| B | **stock Qwen3.5-4B**, f16 | 99.796% | 0 | PASS | 74.0 (1.46x) | 8.1 GiB |
| B + Q8_0 | **stock Qwen3.5-4B, Q8_0** | 99.687% | 0 | PASS | **98.9 (1.95x)** | 4.3 GiB |

- **B uses token ids, not rows:** the stock model is 2,560 wide, so it gets the text tokens. The inside-check shows
  that's all the widened half uses. B's numbers are stage 0's plain-llama run (`results/llamacpp/stage0/`) scored
  against Qwenputer's picks instead of plain HF's.
- **Prompt reading (pp512)** is slower for the Q8_0 files (3,289 vs 3,582; 4,346 vs 4,974 tokens/s): prompts are a
  small share of a question.
- **Whole-question effect (an estimate, not measured):** in stage 3 the text steps were 4,177 of S's 5,148 s, the
  computer 897 s. If text ran 1.95x faster with everything else unchanged, a question would take about 0.6x as long
  as S: about 1.65x S's speed, about 2.4x the original C's. The computer then becomes about 30% of the time.
- These are teacher-forced agreements on 12 development questions. Free-running accuracy on the 160 needs a
  registered run, like stage 3.

Raw: `results/speed-options/`; GGUF hashes in `gguf-sha256.txt`.
