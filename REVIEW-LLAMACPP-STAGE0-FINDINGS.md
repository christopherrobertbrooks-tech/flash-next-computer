# llama.cpp stage-0 premise review

Source `llamacpp-port` **ecddf63ddbde945d4594dac0bb0dbabe781daef9**, on `review/codex-llamacpp-stage0`. The brief names an earlier source revision; used the user's requested latest revision. Read-only review of plan/code/raw artifacts and allowed gateway files. CPU checks only, CUDA hidden; **neither of the two allowed V100 jobs was used**. No model load, installs or gateway project edits. Negative controls wrote only to own temporary folders.

## Pass 1 (blind)

**The recorded implementation is a credible premise success, but its reported top-1 counts are not exact greedy-argmax counts, and the automated PASS is not protected by the required integrity/hard gates.** Independently corrected/bounded numbers still satisfy the fixed stop rule. Fix the scoring/capture instrument before presenting its current agreement percentages as fully reviewed or relying on it for later stages.

### Findings

1. **REDLINE — `topk(..., 5)[0]` is not necessarily the greedy argmax being measured.**
   Plan: capture HF top-1 and compare teacher-forced greedy choices. Both `capture_fixtures.top5` and `plain_hf_reference.top5` record `torch.topk` order, and score_premise treats its first ID as HF's top-1. Upstream generation uses `torch.argmax(logits[:v0])`; the helper scans IDs upward with strict `>`, also choosing the first/lowest ID at a maximum. Torch topk can order tied maxima differently.

   Fixture evidence: **24 of 13,746 eligible positions** have stored HF top5[0] different from the actual next generated token, despite the position being Qwen-owned. There are **53 eligible top-logit ties**. For final-4-001 position-array index 6, HF records IDs `[567,353,...]` with both logits **30.125**, actual generation chooses **353**, and llama chooses **353**; the scorer nevertheless calls this a disagreement. Conversely some topk-agreement positions disagree with actual argmax. This is not evidence that the capture changed generation; it is an incorrect representation of its greedy choice.

   Fix: record **argmax explicitly from the full masked logits**, separately from top-5 diagnostics/margins, in both HF paths; score that ID with a shared deterministic tie policy. Sorting the saved five is not a complete repair: **four plain-control positions have five tied top logits**, so an unrecorded lower-ID tied maximum may exist. Do not change the registered metric to tie-set acceptance after seeing data; that can be a separate diagnostic. Existing generated IDs recover the true split-reference argmax at all eligible positions because these fixtures all end at EOS. Regenerate/recover the missing plain-reference IDs with explicit provenance, preserving originals.

2. **REDLINE — failed hard checks or missing fixed questions can still print `PREMISE PASS`.**
   Plan separates row/state correctness and zero computer mismatches from agreement. Capture records lock_error/stop but does not abort on lock failure; score_premise checks list lengths but not fixture hashes, fixed twelve IDs, duplicate manifest entries, lock/termination status, positions or reset flags. Actual stage0d.sh uses `set -e`, and run_helper_fixtures exits nonzero for its own failed reset, which is good; that does not protect the fixture capture/standalone final verdict from unvalidated data.

   CPU negative controls against the actual scorer, with changes only in own scratch, all exited **0 / PREMISE PASS**: (a) a fixture with `stop='lock error'`, a nonempty lock_error and now-invalid manifest hash; (b) split `_reset_check.same=False`; (c) a manifest containing only **one** of the twelve fixed questions. These are comfortable false passes. Validate exact unique question IDs and optional-extra policy, content hashes, stop/lock validity, row/position/index consistency, finite/in-range outputs, and required reset results before issuing the whole-stage verdict. Alternatively make this tool explicitly report agreement only and have a fail-closed stage driver combine it with independently validated hard gates. Add these negative controls. All supplied real fixtures currently pass the manually checked conditions; this finding does **not** claim an actual lock failure or failed reset occurred.

3. **NIT — provenance hashes do not yet pin the model contents or helper library build.**
   Plan asks for model/code hashes. Manifest has config/chipformer/index and selected script hashes, but not checkpoint shard content hashes; an unchanged index does not imply unchanged tensor values. Record the used shard hashes and the actual conversion/quantizer/helper/libllama build identity plus artifact hashes as one manifest. The raw summary separately records helper/GGUF hashes; consolidate them and verify them in later consumers. Current deployed capture/helper/scorer source hashes match the reviewed files, and every fixture hash verifies; this checks present identity, not an immutable launch-time environment.

4. **NIT — describe the converted artifact as mixed F16/F32, with actual size and scope.**
   Actual GGUF has **426 tensors: 248 F16, 178 F32**, notably a full **F32 tied token/output embedding**, not a uniform F16 model. Embedding max absolute value is **1,099,511,627,776 (2^40)**, far beyond F16's finite range; preserving it in F32 is justified. Quantizer records **14,314.39 MiB** (~13.98 GiB), above the plan's ~10 GB estimate. All GGUF tensors scanned finite. This is a valid mixed-precision candidate, not an accuracy failure, but name it explicitly when comparing against the plain-model F16 control or planning two-process memory. No live memory/speed claim is established by this stage.

### Stage 0a: fixture capture and ownership

The specified first three IDs per group are present exactly once, twelve total. Several have multiple calls (4/9/8/9), so no extra question is needed. All end **EOS**, have null lock_error, and are below both budgets. Question SHA256 equals the registered primary file. The plan's denominator clarification was committed **08:46:41 −0500**, before the recorded ~08:52 capture; no stop-rule change was found in the inspected plan/code.

Hooks copy rows with `detach().to('cpu', copy=True)`, return no replacement tensors, and do not mutate activations/logits. The wrapped Qwen step calls the real function once; the circuit wrapper records its real result; on_token records IDs only. Prefill rows are captured before the first Qwen layer in the full checkpoint path, decode rows before that same layer after `_exec_hc`. Circuit ownership is recorded for **the next token**, not inferred from the input token. All step input IDs equal the generated ID at their recorded gen_index, text positions increment P+k, and chip tokens have no Qwen row. Prompt-last is included explicitly. Forced response/echo inputs still update Qwen history; selection excludes positions whose **next output** is circuit-owned. This resolves the otherwise ambiguous shorthand that a HALT newline is “excluded”: the forced next-token choice is excluded, whereas the row after reading that newline can legitimately predict a Qwen-owned token.

Candidate prompt generation is performed after the real answer; it changes executor state, but the next question's upstream start zeroes states/history before its own initialization. It does not supply the captured answer's logits. Its docstring says it leaves the state as ExecRow.start would, but it does not call load_init at the end; do not reuse that state directly in a future split runner. This is harmless for the current sequence, which resets on the next start.

Capture adds synchronizing copies/topk work, so its wall time is not an uninstrumented performance measurement. No code-level mutation of the trusted mathematical path was found; the topk/argmax representation defect is finding 1.

### Stage 0b and 0c: prompt candidate, cut and conversion

- Every prompt-row matrix equals its captured HF matrix elementwise in BF16 for all **12** fixtures. The checker derives first-token equivalence for the same HF Qwen function from identical row inputs; it does not run/store a separate first-logit candidate experiment. That is a valid conditional argument, not a measured exact equality between HF and llama.cpp logits.
- Independently compared **every element of all 426 extracted tensors** with the current source: source layers **11–42 -> 0–31**, final norm and **248320 × 3968** text embedding slice. **Zero mismatches**, using bounded 256-row CPU slices. Only the intended retained tensor keys exist in the cut.
- Config preserves **hidden 3968, 32 layers**, explicit head dimensions and recurrent layout, with every fourth Qwen layer full-attention. The metadata maps RoPE to **[11,11,10,0]**, base **10,000,000**, rotary dimension **64**. Tokenizer JSON/config/chat-template files are byte-identical to the base copies. The extraction's JSON-vocabulary assertion allows only missing base slots paired with Qwenputer reserved names; it is not a claim that every possible AutoTokenizer-added audio/TTS token is interchangeable. The mathematical prompt/token fixtures are explicit IDs.
- GGUF metadata confirms qwen35 architecture, the exact 32-element recurrent-layer pattern, width **3968**, vocabulary **248320**. Output is tied (no separate output.weight). Independently checked **all F32 GGUF embedding rows equal the cut embedding promoted from BF16**, and final output_norm equals **1 + stored HF norm**, as the converter's norm convention requires. Logs show the expected tensor transformations/types; scanned all final GGUF data for nonfinite values: **none**. The file is a mixed-precision artifact as noted above.

### Stage 0d: helper and independent arithmetic

The helper sends ROWS as contiguous little-endian F32, with explicit text positions **[p,p,p,0]** for the four-section llama.cpp interface; only the first three RoPE sections are active. TOKENS uses the token-input position expansion. The parent supplies fixture prefill rows, then one text row per cached step, with **P+k assertions**; chip tokens never advance Qwen. The helper requests logits only for the final row of each batch and scans exactly the cut's text vocabulary. RESET calls `llama_memory_clear(..., true)`. Both saved A/B/.../A checks are **same=True**; repeated first-fixture outputs match bitwise in these runs. This is useful state evidence, not coverage of all future helper failures. The generic BF16->F32->BF16 assertion only verifies representability; it is not an end-to-end hash acknowledgement by the helper. Wrong shape/framing/error/EOF and reset-negative fixtures still deserve explicit tests before adopting this as the production stage-2 runner. No GPU helper was executed by this review.

Raw JSON decompressed hashes match the gateway files for split/plain-HF/plain-llama. Recomputed the existing scorer's numbers independently from the fixture-selected positions:

| Question | Eligible positions | Recorded split disagreements | Recorded control disagreements |
|---|---:|---:|---:|
| final-1-001 | 84 | 0 | 0 |
| final-1-002 | 84 | 0 | 0 |
| final-1-003 | 129 | 0 | 0 |
| final-2-001 | 1460 | 1 | 2 |
| final-2-002 | 2394 | 3 | 5 |
| final-2-003 | 2801 | 2 | 3 |
| final-3-001 | 437 | 0 | 1 |
| final-3-002 | 624 | 1 | 2 |
| final-3-003 | 269 | 1 | 1 |
| final-4-001 | 2164 | 10 | 10 |
| final-4-002 | 1126 | 1 | 2 |
| final-4-003 | 2174 | 10 | 7 |
| **Total** | **13746** | **29** | **33** |

That gives stored-topk agreement **99.78903% / 99.75993%** and zero questions with split disagreements at margin >=1.0: the original scorer prints PASS. Those are **recorded-topk-order** counts, not exact HF greedy-argmax counts.

Recovering true split-reference choices from generated IDs yields disagreements by question **0,0,0,1,5,1,1,1,1,8,2,10**, total **30**, so true split agreement is **99.78175%**. For plain HF, deterministic argmax is recoverable from a complete tied maximum set when fewer than five logits tie at the top. That gives **25 known control disagreements**, plus **four ambiguous positions** with all five top logits tied; the exact plain argmax at those four is not recoverable from top5 alone. Control disagreements are therefore **25–29**, agreement **99.78903–99.81813%**. Split trails by **0.00727–0.03637 percentage points**, still far within the allowed **1.0 point**. High-margin split-failure question count remains **0**: the altered tie cases have margin zero.

**Own verdict:** the observed data support a **premise PASS under the fixed numeric threshold even after correcting/bounding tie handling**; no full-system speed/accuracy or free-running equivalence is established. The automated acceptance instrument needs the top-1 and fail-closed fixes in findings 1–2 before being relied on. No extra GPU run is needed to establish this bounded verdict; exact corrected control counts require proper argmax capture/recovery, not guessing from a truncated tied set.

Blind boundary: `results/llamacpp-stage0.md` and `REVIEW-LLAMACPP-STAGE0-PASS2.md` were not opened before this pass was committed/pushed.

## Pass 2 (guided)

Blind pass **056631f** was pushed before opening the guided brief or `results/llamacpp-stage0.md`. The narrative's main premise conclusion survives independent checks, but its exact top-1 percentages need correction and its tools need the fail-closed acceptance checks identified above. No GPU jobs were used in either pass; all additional inspection was bounded CPU fixture/tensor work and scratch-only controls.

### 1. Is PASS correct, and what is overstated?

**Yes as an observed numeric premise result**, including after correcting/bounding ties: split **99.78175%**, control **99.78903–99.81813%**, no split-disagreement question with margin >=1.0. The largest corrected disagreement margin is still **0.125 logit**, consistent with the narrative's near-tie observation. It is not a free-running equivalence/speed result. Mark the published **99.79/99.76** as the original topk-order metric, replace them with corrected canonical-argmax results once captured, and retain the originals for provenance. The apparent ordering “split slightly better than control” is not supported after correction: the bounded canonical result has split slightly below control, still well inside the fixed 1.0-point allowance. The distinction is tiny numerically but material to describing the instrument accurately.

The captured twelve really are complete EOS answers with null lock errors; prompt rows and extraction checks pass. However a single unqualified automated stage PASS overstates the current validator because lock/reset/integrity/missing-question failures can still pass that tool. The author's three-high-margin mutation is valid: independently repeated it on CPU and got **exit 1 / PREMISE STOP**. This confirms the numeric branch works, not that the whole acceptance gate is fail-closed. Findings 1–2 remain redlines on the review tooling/report, not evidence that this observed premise fails its registered threshold.

### 2. Prompt-row comparison and stage-2 use

The pre-hook sees the actual input tensor to **layer 11, the first Qwen layer**, not the raw embedding or a post-Qwen activation. CPU copies are detached with copy=True; the hook returns None. The candidate processes the full prompt through the executor with zero S/history and assembles `_exec_hc` rows in BF16. Comparing `(candidate.to(hf.dtype) == hf)` is appropriate for the actual BF16 input contract; it is not a comparison of arbitrary rounded FP32 intermediates. All **5753** captured prompt rows agree across twelve questions, including tool syntax. The candidate calculation runs after that answer and does not feed its captured logits. Subsequent questions reset before generation.

This supports the candidate prompt builder for the tested inputs. Stage 2 must still reproduce the **two phases of state initialization**: prompt from zero recurrent state/history; then load the parked firmware snapshot while retaining the correct prompt lag history. `executor_prompt_rows` currently does **not** call load_init after its loop, despite its docstring suggesting it leaves the same final state as ExecRow.start. That is harmless in current fixture capture because the next answer starts fresh; do not copy that omission into the live runner. Preserve exact input rows and first-token conditioning, and retain fixture/regression checks when broadening to all 160 questions. The row equality implies the same HF Qwen function receives equal inputs; it does not by itself guarantee a different engine's logits or all future prompts match.

### 3. F32 embedding, computer columns and a safe optimization investigation

Keeping the tied embedding/output head in **F32 is correct for the present unmodified tensor**: its max magnitude is **2^40**, and the final norm reaches **2^60**. F16 cannot represent those values; the final GGUF scan has no nonfinite tensors. Calling this a mixed F16/F32 candidate is essential. The embedding contains **248320 × 3968 × 4 bytes = 3758.75 MiB**; that is its nominal matrix size, not a measured DRAM-byte counter. No new speed claim follows from it.

Additional CPU evidence narrows the optimization question:

- At **all 13,746 eligible Qwen-choice positions**, the captured input's only nonzero coordinate beyond the original 2560 language dimensions is **tau at index 2777**, value **2^-40**. Thus “the computer coordinates are exactly zero” is **false** on these fixtures.
- Every extra output row of all **64** Qwen residual-producing projections (attention out plus MLP down across 32 layers) is exactly zero in the current cut. This supports the algebraic invariant that these layers leave the extra residual coordinates unchanged, subject to ordinary finite computation; the normalization still sees the full residual.
- Final norm's effective tau scale is **2^60**. Every one of the **248,320** output embedding weights in column 2777 is **2^-40**. Therefore tau's contribution is **nonzero but common to all text logits**, given the invariant above; all other computer-coordinate contributions are zero at these sampled eligible positions. It must not be described as zero/no contribution.

A common logit offset does not change mathematical argmax/softmax, so the draft question's proposed zero-only condition is stronger than necessary for those observables. But dropping/reordering this term or changing matrix precision can alter finite-precision ties, which are already relevant here. A safe next check is to capture/verify final normalized hidden states, compute the complete versus decomposed projection on identical states (base-column contribution plus explicit common offset), compare canonical argmax/margins and output values, and enforce the ownership/extra-coordinate invariant. Keep the **full 3968-dimensional RMS normalization**: removing coordinates before normalization is a different change. Circuit-owned positions can contain nonzero code columns; do not extrapolate this eligibility-specific simplification to them. The live runner may avoid an unnecessary language-head projection when the circuit owns the next token, but Qwen's text-input layers/cache still have to advance.

This is a promising **separate optimization experiment**, not permission to edit the current checkpoint or assume a universal invariant from twelve samples. A proof from the compiler/weights plus runtime invariant checks and fixture/gate results can justify an optimized path with fallback; current stage 0 rightly retains the original F32 weights. No final-hidden-state or alternate-head GPU experiment was run by this review.

### 4. Is the control matched, and do ties change the reading?

Both HF and llama sides use one prompt prefill followed by cached one-text-token steps, with exactly the same teacher-forced textual prefixes and eligible output positions within each pair. This is a sensible **precision/runtime sensitivity control**, and the split/model width differences are the intended subject of the premise, not a claim that the models are identical. Forced response text updates histories on both sides even where a next-token prediction is excluded from the score. The common four-section text positions have only three nonzero RoPE section widths and reproduce the HF text-position convention.

The tie problem changes the exact reported agreement and its small relative ordering, not the numeric PASS. `topk`'s selected tied ID is not the same contract as the actual greedy argmax. Split's generated stream provides the true choices; plain HF stores only top five, which is insufficient at four positions. Store explicit masked argmax IDs in both references and keep top-five/margins as diagnostics. The 1.0-logit hard-disagreement criterion remains untriggered, and its positive mutation works. Do not infer free-running identity or choose a new tie-tolerant acceptance rule post hoc.

### 5. What must change before proceeding?

Before calling the stage-0 instrumentation complete or using it to gate later work:

1. Fix explicit argmax capture/scoring and version the corrected control evidence; preserve original fixtures/results and the fixed denominator/threshold. Correct the narrative's counts and characterize the unresolved saved control ties honestly until resolved.
2. Wire hard failures and integrity into a fail-closed stage gate: exact twelve IDs, optional-extra exclusion, matching manifest/file/output identities, hashes, lock/termination state, row/position/count validity and reset success. Retain missing-question, bad-hash, lock-error and failed-reset negative controls alongside the existing high-margin positive STOP control.
3. Freeze the exact mixed-precision GGUF, helper/library/converter and checkpoint content identities and state the real file size. This makes a later system comparison reproducible rather than relying on the path or index file remaining unchanged.

For stage 1's actual emulator baseline, add the production wrapper's forced-frame, budget, error/EOF and cleanup controls before running all 160; stage 0's helper driver is a fixture probe, not that runner. For stage 2, ensure prompt reset/firmware/history ordering, bit-preserved transmitted rows, cache resets, ownership override and fail-closed reference checking survive the process boundary. The original teacher-forced A/B/.../A evidence is useful but not a replacement for those runtime tests. Do not let the new output-head optimization expand the premise port silently; it needs its own measured change and correctness contract.

### Guided review outcome

The **premise is supported under the registered numeric stop rule**; the extraction and embedding-input approach look viable on the tested data. **Two review redlines remain in the capture/scoring acceptance instrument and published exact metrics.** Their correction should not require changing the plan's criterion, and the bounded audit already shows they do not overturn the observed PASS. No free-running port, system speedup, final-head simplification or all-input correctness has been demonstrated by this stage. Both review passes are findings only, with no source-branch/main push/merge or implementation changes.
