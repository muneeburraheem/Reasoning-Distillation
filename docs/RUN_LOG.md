# Evaluation run log — September 2026

Running record of what was run, what it produced, and what was concluded.
All timings on the **local RTX 4090 (24 GB)** unless stated. The RTX 5000 Ada
referenced in `september-evaluation-analysis.pdf` is a paid cloud GPU and is
**not** in use; do not use its timings for planning.

Environment: `/home/cs/Desktop/KD/KnowledgeDistillation/venv` (Python 3.12.3,
torch 2.9.1+cu128, transformers 5.8.0, datasets 4.4.1, math-verify 0.9.0).
Note `requirements.txt` pins newer versions than are installed — do **not**
`pip install -r` it; the installed set is what produced every result here.

---

## 1. Harness defect found and fixed (2026-09-25)

**Symptom.** Distilled checkpoints finished reasoning, emitted a correct
`\boxed{}`, then fell into a repetition loop until the token cap. The strict
extractor takes the *last* box and refuses an incomplete one, so the clipped
final box scored as no answer.

**Measured cost.** On an 8-question smoke: 3 of 8 logitKD and 2 of 8
ReasonDistill MATH-500 answers were correct but scored 0.

**Fix.** `custom_eval/answer_stopping.py` — `FinalAnswerStopper` and
`final_answer_boundary` lifted from `scripts/eval_gsm8k_generation_length.py`
(commit `7f637e4`) so both harnesses share one definition of "finished".
Generation now stops per-row at the first complete answer outside `<think>`,
and each row's tokens are sliced at its own stop point before decoding.
Wired through `generation.py`, `config.py`, `runner.py` as
`generation.stop_at_final_answer`, overridable per benchmark.

Guarantees, unit-checked: a clipped box, an empty box, and a box inside a
thinking block can never become the stop point or the parsed answer.

**Effect (8-question smoke, before → after)**

| model | benchmark | before | after |
|---|---|---|---|
| logitKD | math500 | 0.375 | 0.750 |
| ReasonDistill | math500 | 0.625 | 0.875 |
| logitKD | aime25 | 0.375 | 0.500 |
| base | both | unchanged | unchanged |

Also eliminated the `>=10 <answer> tags` repetition indicator that the
September PDF measured at 44–53% of ReasonDistill responses: now **0.0%**.

Raw: `eval_outputs/smoke_8q/` (see its README; the pre-fix JSONs were
overwritten by the second smoke run, the log survives).

---

## 2. Full math run (2026-09-25 → 26) — 7.6 GPU-h

`configs/eval/qwen_math_basic.yaml`, 3 models, thinking mode, pass@1, temp 1.0, seed 42,
batch 8. math500 @ 4096, aime25 @ 32768.
Raw: `eval_outputs/test_math/20260925T181224Z-112470/`

| model | benchmark | n | score | truncated |
|---|---|---|---|---|
| base Qwen3.5-2B | math500 | 500 | 0.088 | 91.2% |
| base Qwen3.5-2B | aime25 | 30 | 0.033 | 96.7% |
| logitKD | math500 | 500 | 0.540 | 40.2% |
| logitKD | aime25 | 30 | 0.167 | 20.0% |
| ReasonDistill | math500 | 500 | 0.514 | 36.4% |
| ReasonDistill | aime25 | 30 | 0.133 | 20.0% |

**Conclusion: budget-limited, not a capability measurement.** Accuracy among
answered attempts was 1.000 / 0.944 / 0.921 — i.e. whenever a model finished it
was nearly always right, so the score was measuring termination. logitKD vs
ReasonDistill on math500: +2.6 pp, McNemar **p = 0.267** (no effect).

---

## 3. Does raising the AIME budget help? No. (controlled)

Same 8 AIME questions at 16384 (smoke) and 32768 (full run):

- answered **0/8 → 1/8**
- "Wait" markers per output roughly doubled (123→290, 138→426), one case
  255 → **5895** (one every 5.6 tokens)
- novel 8-gram rate stays 0.97–0.98 to the final decile, so it is **not**
  verbatim looping — it is semantic thrashing: the same interpretive question
  re-asked 24,000 tokens apart

Base "Wait"/1k tokens: **aime25 10.2** (thrashing) vs **math500 3.2** (lowest of
all six cells — slower than the distilled models, i.e. genuinely cut off).

**Therefore: stop raising the AIME budget; raise MATH-500's.**

---

## 4. MATH-500 @ 32768 (2026-09-26 → 27) — 25.6 GPU-h — HEADLINE

`configs/eval/qwen_math500_32k.yaml`. Base excluded (its 4096 number is a floor, and
it is 1/3 of the runtime). Only variable changed vs §2: the token budget.
Raw: `eval_outputs/test_math/math500_32k/`

| model | n | score | 95% Wilson | truncated | empty |
|---|---|---|---|---|---|
| **logitKD** | 500 | **0.780** | [0.742, 0.814] | 4.8% | 10.2% |
| ReasonDistill | 500 | 0.682 | [0.640, 0.721] | 6.8% | 14.8% |

```
delta = +9.80 pp   95% CI [+5.20, +14.40]   97 newly-correct / 48 lost   McNemar p = 0.0001
```

Truncation fell ~38% → ~5%. **First cell in the project measuring math ability
rather than termination.** The same two checkpoints on the same 500 questions go
from "no detectable difference" (p=0.267 @ 4096) to "+9.8 pp, p=0.0001" @ 32768,
purely by removing the cap.

Token-cap retrodiction (discarding correct answers needing more than the cap):

| cap | logitKD | ReasonDistill | gap |
|---|---|---|---|
| 4096 | 0.530 | 0.526 | 0.004 |
| 8192 | 0.628 | 0.608 | 0.020 |
| 16384 | 0.710 | 0.638 | 0.072 |
| 32768 | 0.780 | 0.682 | 0.098 |

The cap is **not a neutral knob**: correct answers have p90 15524 tokens for
logitKD vs 8527 for ReasonDistill, so a low cap preferentially deletes the
better model's wins. Median is only ~2700; the 7.6k mean is tail-driven.

**Scoring audit (both runs): clean.** Zero normalise-and-match mismatches, zero
credited unclosed answers (closure-gating changes nothing — the September PDF's
central caveat does not apply here), no premature stopping, math-verify handles
`\dfrac`/`\left(`/`\sqrt2` variants correctly.

---

## 5. What this means for the research claim

The two continuation configs are a near-perfect ablation — identical data
(12,032 examples, skip 40,000), seed, LR schedule, teacher, `alpha_kd 0.30`,
`beta_ce 0.75`. The only difference:

```
logit_kd        alpha_align: 0.0
reasondistill   alpha_align: 0.25 + SAE alignment (512 neurons, teacher L16 -> student L12)
```

**ReasonDistill is logitKD plus the reasoning-activation alignment term, and it
is 9.8 pp worse on MATH-500.** Decomposition of logitKD's 97 wins: 45 where
ReasonDistill produced no answer, 52 where it answered and was wrong — both
channels favour logitKD, so this is not only a termination effect.

Reasoning behaviour *is* induced (6x base's answer rate) — but logitKD induces
it too, so it is attributable to the shared reasoning-trace data, not to the
alignment term.

**Caveat that keeps the claim alive:** `alpha_align: 0.25` was never validated
against accuracy. Its own config says it was raised to break a ~1.0
*training-loss* plateau. Only the two endpoints (0.0, 0.25) have ever been
sampled. Hence §6.

Not a single-variable ablation: each continuation resumes from its own stage-1
checkpoint, so this compares two pipelines, not one loss term in isolation.

---

## 6. Alpha sweep — COMPLETE (2026-09-29)

`configs/qwen_reasondistill_sweep_a{000,005,010,050}.yaml`. Each differs from
`qwen_reasondistill_final_continue.yaml` by exactly 4 lines (name,
`alpha_align`, `checkpoint_dir`, `log_dir`). Same `student_checkpoint` across
all four, so `alpha_align` is the only variable — `a000` is the true control and
is NOT the same as the existing logit_kd_continue checkpoint.

`scripts/train.py` accepts only `--config` ("training flags are intentionally
disabled"), hence one YAML per point.

Read the **shape**, not the absolute scores:
- monotonic decline -> alignment always hurts; publish the negative result
- interior peak -> 0.25 overshot; the claim survives with a dose-response curve

Cost per point: ~1 h training (derived, see below) + eval.
Eval at 8192 ≈ 3 h/checkpoint; at 32768 ≈ 12.9 h/checkpoint (measured).
Plan: sweep at 8192 for shape, re-run the winner at 32768 for the reported number.

### Measured: a000 completed in 1 h 24 m (12,032 examples, 2.39 it/s)

Final avg loss 1.432 (from 1.549 at the first logging window). Checkpoint at
`checkpoints/qwen_reasondistill_sweep_a000/epoch_1`. Remaining three points
~4.2 h total. The earlier ~1 h estimate derived from July mtimes was roughly
right, but only because max_length was cut to 512 (below).

### The 4090 cannot run this config as written — three deviations

Training had never been run in this venv or on this card; it needed `zstandard`
installed before the dataset would even load. Then it OOM'd repeatedly at batch
1-2. **Measured** breakdown at max_length 2048 (probe, not theory):

| stage | GiB allocated |
|---|---|
| teacher 4B int8 loaded | 4.51 |
| + student 2B bf16 | 8.02 |
| after teacher forward (no_grad) | 8.98 |
| after student forward (with grad) | **23.22 — OOM** |

The student forward alone costs **14.2 GiB**. Root cause is NOT the optimizer or
the KD loss (two wrong hypotheses chased first). It is the pure-PyTorch
linear-attention fallback, `modeling_qwen3_5.py:250`:

```python
query, key, value, beta, g = [
    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
]
```

Five fp32 `.contiguous()` copies per layer across 18 linear-attention layers,
all retained for backward. transformers warns about this at startup on every
run: *"The fast path is not available because one of the required library is not
installed"* — **`flash-linear-attention` and `causal-conv1d` are not installed.**
Eval was never affected because eval is no_grad.

Deviations applied to the SWEEP CONFIGS ONLY (originals unchanged):

| change | reason | risk |
|---|---|---|
| `max_length: 2048 -> 512` | 2048 and 1024 both OOM | **HIGH — changes what the model sees** |
| `optimizer_8bit: true` (PagedAdamW8bit) | fp32 Adam moments ~12 GiB at first optimizer.step() | numerics differ from fp32 AdamW |
| `kd_chunk_size: 256` | KL transients ~4.7 GiB over a 248,320 vocab | none — verified identical, gradient diff exactly 0.0 |

**The max_length cut matters.** AM-DeepSeek traces have a ~900-token median
(September PDF, Figure 7), so 512-token windows truncate most traces
mid-reasoning. These sweep checkpoints are therefore **NOT comparable to
`qwen_reasondistill_final_continue`** (trained at 2048, fp32 AdamW, other
hardware), and absolute scores will be depressed across the board. Valid for
reading the SHAPE of the alpha curve; not valid for any reported number.

**Before reporting anything from the sweep: install `flash-linear-attention`,
re-probe, and if the student forward fits, rerun at max_length 2048.** The 4090
likely can do full sequence length with the fused kernels — it is the fallback
path, not the card, that fails.

Opt-in plumbing added (defaults preserve original behaviour exactly):
`training/base_trainer.py::build_optimizer`, `losses/kd_loss.py` `chunk_size`
argument, and `optimizer_8bit` / `kd_chunk_size` config keys in
`scripts/train.py`.

---

## 6b. Alpha sweep result: monotonic decline, no interior optimum

`configs/sweep_alpha_eval.yaml`, MATH-500 @ 8192, n=500 per point, 9 h 21 m.
Raw: `eval_outputs/sweep_alpha_8k/`

| alpha_align | score | 95% Wilson | truncated |
|---|---|---|---|
| **0.00** | **0.626** | [0.583, 0.667] | 25.6% |
| 0.05 | 0.600 | [0.556, 0.642] | 29.4% |
| 0.10 | 0.576 | [0.532, 0.619] | 28.2% |
| 0.50 | 0.570 | [0.526, 0.613] | 31.4% |

Paired against the alpha=0 control: 0.05 p=0.275 (n.s.), **0.10 p=0.033**,
**0.50 p=0.015**; 0.10 vs 0.50 p=0.861 (flat). Damage saturates by ~0.10.

**No dose of reasoning-activation alignment beats switching the term off.** This
closes the "0.25 was just mistuned" escape hatch: the whole curve slopes down.

Mechanism is termination, not correctness: answered drops 350 -> 325-331 while
accuracy-given-answered holds flat at 0.874-0.906. Per-level, alpha=0 leads or
ties at four of five difficulty levels.

Caveat stands: these checkpoints were trained at `max_length: 512` (see §6), so
the SHAPE is valid and the absolute values are not reportable.

---

## 7. pass@k sweep — COMPLETE (2026-09-29 -> 10-11), 276 GPU-h

`configs/eval/qwen_passk.yaml`, all budgets 8192, `num_samples: 8`,
`pass_at_k: [2,4,8]`, 64 concurrent sequences, 18 cells.
Raw: `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/`

Added `save_partial_every: 5` to `custom_eval/runner.py` — writes a
`<model>__<benchmark>.partial.json` every 5 batches via tmp+atomic-replace, so a
multi-day run can be read while it runs. Deleted when the cell completes.
Defaults to 0 (off) for every other config.

### pass@8

| benchmark | base | Logit KD | ReasonDistill |
|---|---|---|---|
| math500 | 0.356 | 0.858 | **0.862** |
| gsm8k | 0.734 | **0.966** | 0.965 |
| aime25 | 0.000 | 0.233 | **0.267** |
| mmlu | 0.909 | **0.936** | 0.923 |
| arc-c | 0.970 | **0.979** | 0.974 |
| hellaswag | 0.825 | **0.858** | 0.821 |

### Both distilled models clear base on generative benchmarks only

```
                 Logit KD              ReasonDistill
math500        +50.2 pp  p=6e-76        +50.6 pp  p=1e-76
gsm8k          +23.2 pp  p=2e-68        +23.1 pp  p=2e-65
aime25         +23.3 pp  p=0.016        +26.7 pp  p=0.008
mmlu            +2.7 pp  p=0.005         +1.4 pp  p=0.12
arc-c           +0.9 pp  p=0.078         +0.4 pp  p=0.56
hellaswag       +3.3 pp  p=0.007         -0.4 pp  p=0.81
```

On MATH-500 Logit KD wins 251 questions and loses **zero** against base. The
multiple-choice gains are small because 8 samples on 4 options approach a
guessing ceiling — those three benchmarks cost ~62 of each model's 93 hours and
discriminate nothing.

### Logit KD vs ReasonDistill: tied on five of six

| benchmark | Logit KD | ReasonDistill | delta | p |
|---|---|---|---|---|
| math500 | 0.858 | 0.862 | -0.4 pp | 0.856 |
| gsm8k | 0.966 | 0.965 | +0.1 pp | 1.000 |
| aime25 | 0.233 | 0.267 | -3.3 pp | 1.000 |
| mmlu | 0.936 | 0.923 | +1.3 pp | 0.117 |
| arc-c | 0.979 | 0.974 | +0.5 pp | 0.302 |
| hellaswag | 0.858 | 0.821 | +3.7 pp | **0.002** |

### The two-regime result — report BOTH

```
32,768 tokens, 1 sample   logitKD 0.780   reasonDistill 0.682   +9.8 pp   p=0.0001
 8,192 tokens, pass@8     logitKD 0.858   reasonDistill 0.862   -0.4 pp   p=0.86
```

Per-sample accuracy inside the pass@8 run resolves the apparent contradiction:

```
logitKD        0.628 per-sample  ->  0.858 pass@8
reasonDistill  0.582 per-sample  ->  0.862 pass@8
```

**Logit KD is better per attempt (+4.6 pp); ReasonDistill catches up only when
given 8 tries.** pass@8 asks "can it ever get there", a more forgiving question
that compresses the gap. The budgets also differ (8192 vs 32768), and §4 showed
a low cap compresses this specific gap. Neither number is wrong; the writeup
must state which question it is answering.

---

## 8. Reasoning-trace comparison (2026-10-11)

Source: the 32,768-token single-sample MATH-500 run (lowest truncation, 4.8% /
6.8%), thinking block only, 500 matched questions. Rates are per 1,000
characters, paired per question, 10,000-resample bootstrap CI on the difference.
4 degenerate traces dropped (repeated tail block).

| marker | Logit KD | ReasonDistill | diff | 95% CI |
|---|---|---|---|---|
| Therefore / Thus / Hence | 0.928 | 1.263 | **+0.336** | [+0.253, +0.418] |
| But (contrastive) | 1.855 | 2.050 | **+0.195** | [+0.091, +0.299] |
| let me check / verify | 0.422 | 0.478 | **+0.056** | [+0.018, +0.094] |
| Hmm (hesitation) | 0.268 | 0.303 | **+0.035** | [+0.006, +0.064] |
| I think / I believe | 0.108 | 0.140 | **+0.031** | [+0.007, +0.056] |
| Alternatively | 0.406 | 0.354 | **-0.052** | [-0.095, -0.009] |
| equation-bearing lines | 4.347 | 3.899 | **-0.447** | [-0.664, -0.231] |
| Wait (backtrack) | 1.647 | 1.625 | -0.023 | [-0.116, +0.070] n.s. |
| \boxed in-thinking | 0.138 | 0.135 | -0.004 | [-0.024, +0.017] n.s. |

**Summary: ReasonDistill reasons in prose; Logit KD reasons in symbols.**
ReasonDistill marks conclusions, hedges, verifies and self-corrects in natural
language; Logit KD writes more equation lines and branches to alternative
methods more often. Backtracking frequency ("Wait") is identical — the two
differ in HOW they reason, not how much they second-guess.

**A trap avoided:** a naive pooled count made `<answer>` tags look 218x more
frequent in ReasonDistill. 1,112 of those 1,126 occurrences came from ONE
degenerate trace (`test/intermediate_algebra/1544.json`, hit the cap in a
repetition loop). Excluding it: 14 occurrences across 7 traces vs 5 across 3 —
negligible. Always check whether a pooled ratio is carried by one document.

Characteristic examples are listed in §8 of the artifact and in the chat
transcript; the clearest are `test/number_theory/342.json` (conclusion-marking:
0.00 -> 1.84 /1k) and `test/prealgebra/977.json` (verification: 0.00 -> 1.44 /1k).

### Length and truncation: NO reliable difference between the two models

The intuition that ReasonDistill is "more concise" and truncates less does not
survive a paired test. Recorded in full because two planned experiments were
premised on it.

**What suggested it.** Raw medians at the 4096 budget: logitKD 2702 vs
ReasonDistill 2190, i.e. ~19% shorter.

**Why that number is an artifact.** Those medians are computed over all 500
samples including truncated ones, which are pinned at exactly the cap. logitKD
truncated 201/500 and ReasonDistill 182/500, so the 250th-smallest value lands
at ~p84 of logitKD's real distribution but ~p79 of ReasonDistill's. The two
medians are not measuring the same quantity. Drop the truncated samples and the
sign flips:

| 4096 run, untruncated only | logitKD | ReasonDistill |
|---|---|---|
| median generated tokens | 1266 | **1303** (longer) |

**The paired test.** 32k run (4.8% / 6.8% truncation), keeping only the 457 of
500 questions where BOTH models terminated naturally — no censoring either side:

```
logitKD        mean 5761   median 2144
reasonDistill  mean 5584   median 1981
paired diff (RD - KD)   mean -177 tokens   95% CI [-751, +396]
RD shorter on           231/457 = 50.5% of questions
sign test               p = 0.85
length ratio (RD/KD)    geometric mean 0.994
```

Conditioning further on both-correct (n=293): medians 1475 vs 1473, RD shorter
on 51.5%. **A coin flip.**

**The tell.** The same model's median moves more between runs than the models
differ from each other — ReasonDistill is longer in one run, shorter in the
other. At temperature 1.0 with one sample per question, run-to-run variance
swamps the effect:

| untruncated median | logitKD | ReasonDistill |
|---|---|---|
| 4096 run | 1266 | **1303** |
| 32768 run | **2400** | 2090 |

**Truncation follows the same pattern** — narrows with budget, then reverses:

| budget | base | logitKD | ReasonDistill |
|---|---|---|---|
| 4096, 1 sample | 91.2% | 40.2% | **36.4%** |
| 8192, pass@8 | 82.7% | 28.9% | **27.7%** |
| 32768, 1 sample | n/a | **4.8%** | 6.8% |

Simulating caps from the uncensored 32k lengths, the number of traces exceeding
each threshold is within a handful and crosses over at 16k:
4096 -> 205 vs 197; 8192 -> 145 vs 134; 16384 -> 91 vs 91; 24576 -> 54 vs 57.

**Consequence for the planned work.** Experiment 2 (Qwen-4B @ 4096, "does
ReasonDistill's lower truncation match the teacher?") tests a difference that is
not established — there is no stable truncation gap to attribute to the teacher.
Experiment 3 (forced small budgets) is still worth running because it measures
accuracy-under-budget, which can differ even when lengths do not, but go in
expecting close curves rather than a sharper decline for logitKD.

**What DOES differ is register, not length** (see the marker table above): same
token budget, different spending — ReasonDistill on prose conclusion-marking and
verification, logitKD on equation lines and alternative branches.

---

## 8b. A finding that FAVOURS ReasonDistill: sampling gain (2026-10-13)

Analysis only, no GPU. From the completed pass@k sweep.

**ReasonDistill converts extra samples into accuracy better than Logit KD on
every generative benchmark.** "Sampling gain" = pass@8 minus mean single-sample
accuracy:

| benchmark | Base | Logit KD | ReasonDistill |
|---|---|---|---|
| math500 | +0.183 | +0.230 | **+0.280** |
| gsm8k | +0.292 | +0.127 | **+0.149** |
| aime25 | 0.000 | +0.129 | **+0.167** |
| **mean (generative)** | +0.159 | +0.162 | **+0.199** |

On multiple-choice it is mixed (mmlu KD +0.231 vs RD +0.226; hellaswag KD +0.398
vs RD +0.362), so the effect is specific to benchmarks that require producing an
answer.

**Mechanism: ReasonDistill explores more.** Distinct non-empty answers across the
8 samples, per question:

| level | n | Logit KD | ReasonDistill | diff |
|---|---|---|---|---|
| 1 | 43 | 1.21 | 1.28 | +0.07 |
| 2 | 90 | 1.33 | 1.52 | +0.19 |
| 3 | 105 | 1.29 | 1.42 | +0.13 |
| 4 | 128 | 1.26 | 1.40 | +0.14 |
| 5 (hardest) | 134 | 1.09 | **1.34** | **+0.25** |
| all | 500 | 1.23 | **1.40** | +0.17 |

Modal-answer share is correspondingly lower (0.576 vs 0.624): Logit KD converges
on one answer, ReasonDistill spreads across several. The diversity gap is widest
on the hardest problems, which is where exploration is worth most.

Consistent with this, at level 5 the k-progression separates:
1 sample KD 0.377 / RD 0.375 (tied) -> pass@2 +0.022 -> pass@4 +0.049 ->
pass@8 +0.060 to RD (McNemar p=0.0386, RD-only 10 / KD-only 2).

### Three caveats that must ship with this finding

1. **Diversity does NOT help under majority voting.** Self-consistency @8:
   Logit KD **0.840**, ReasonDistill 0.826. ReasonDistill's extra answers are
   spread, not concentrated on the right one. The gain only materialises with an
   oracle (pass@k) or a verifier good enough to pick the correct candidate out of
   a wider pool.
2. **The level-5 advantage does not replicate single-sample.** In the 32,768
   1-sample run ReasonDistill is WORSE at level 5 (0.485 vs 0.634, p=0.004) and
   at levels 3 and 4 (p=0.005, p=0.008). The level-5 pass@8 edge is specific to
   the multi-sample 8,192 setting.
3. One seed, one training run per checkpoint.

**Defensible claim:** ReasonDistill produces more diverse solution attempts, and
therefore benefits more from test-time sampling on generative tasks — making it
the better base for best-of-n *with a verifier*, while Logit KD remains better
single-shot and under self-consistency. This is a positioning claim about where
each method pays off, not a claim that ReasonDistill is stronger.

---

## 8c. Teacher experiment — DONE (2026-10-16, 4 h 43 min)

`configs/eval/teacher_math500.yaml` — Qwen3.5-4B, MATH-500, 8192 tokens,
1 sample, settings matched to the students' runs. ~3 h.

Question (reframed; the original truncation framing was dropped, see §8):
does the teacher show the **wrong-vs-truncate split** that ReasonDistill has?
ReasonDistill roughly doubles the wrong-but-finished rate at every budget
(12.2% vs 5.8% at 4096; 25.0% vs 17.2% at 32768) while truncating slightly less.
Teacher commits early -> ReasonDistill transferred a real teacher behaviour.
Teacher keeps reasoning -> the early-commit is an artifact of the objective.

**BLOCKED:** the NVIDIA driver was updated without reloading the kernel module.
`nvidia-smi` reports `Failed to initialize NVML: Driver/library version mismatch`
and torch gives `CUDA error 804: forward compatibility was attempted on non
supported HW`. Loaded module 580.173.02 vs NVML 580.178. Needs a reboot (or a
root-level module reload with no GPU processes running). No GPU work can run
until then.

**UNBLOCKED 2026-10-16:** after the reboot, driver and NVML both report
580.178.04 and torch CUDA works. Launched 11:09; log at
`logs/teacher_qwen4b_math500.log`, output in `eval_outputs/teacher_qwen4b/`.

---

## 8d. "Wrong-but-finished" splits into two failures (2026-10-16)

The "RD doubles wrong-but-finished" figure merges two different things:
**EOS inside `<think>`** (the model stops without ever closing the think block,
so there is no answer to grade, and it is scored wrong) and **committed wrong**
(closes `</think>`, gives an answer, answer is wrong). Per sample, MATH-500:

| run | LKD EOS-in-think | RD EOS-in-think | LKD committed-wrong | RD committed-wrong |
|---|---|---|---|---|
| 4096 x1 | 2.0% | 7.4% | 3.8% | 4.8% |
| 8192 x8 | 3.1% | 7.2% | 5.2% | 7.0% |
| 32768 x1 | 4.8% | 7.4% | 12.4% | 17.6% |

Paired sign tests (discordant questions, RD-only vs LKD-only):
- 4096: EOS-in-think 35 vs 8, p<0.0001; committed-wrong 19 vs 14, p=0.49.
- 32768: EOS-in-think 35 vs 22, p=0.11; committed-wrong 61 vs 35, p=0.010.

So at 4096 the whole excess is RD stopping inside its thinking block, not
committing to a wrong answer. Only at 32768 is committed-wrong excess
significant. The EOS-in-think traces often state an answer in prose
("so the answer should be 27"); whether those answers are right is not yet
checked. The teacher run must be scored with this split.

Answer-spread (distinct finished answers / question, 8192 x8), grouped with
`math_equivalent`, finished-but-empty answers excluded: RD - LKD = +0.134
(95% bootstrap 0.066-0.204); per level +0.07 / +0.13 / +0.04 / +0.13 / +0.23,
CI excludes 0 only at level 5. Restricted to questions where both models
finished the same number of samples, levels 3-4 go negative. The spread gap
is small and largely tracks finished-sample count (i.e. truncation).

**Follow-up: the EOS-in-think traces are mostly correct.** These are natural
EOS before `</think>`, not a harness cut: `FinalAnswerStopper` ignores boxes
inside thinking, and `extract_final_response` deliberately scores any output
without `</think>` as empty. But 75% (RD 216/287 at 8192x8) end with
`**Final Answer** \boxed{...}` inside the think block, and graded with
`math_equivalent` 80% of RD's and 73% of LKD's are correct. RD is penalised
more because it does this ~2x as often.

Strict (harness) vs lenient (grade last `\boxed` when `</think>` is missing),
LKD - RD, paired bootstrap 20,000:

| run | strict | lenient |
|---|---|---|
| 4096 x1 | +0.026 [-0.016, +0.068] | -0.016 [-0.056, +0.022] (sign flips) |
| 8192 x8 per-sample | +0.047 [+0.030, +0.064] | +0.015 [-0.001, +0.031], p=0.06 |
| 32768 x1 | +0.098 [+0.052, +0.144] | +0.078 [+0.036, +0.120], p=0.0006 |

alpha_align sweep, strict -> lenient: 0.626->0.640, 0.600->0.624,
0.576->0.602, 0.570->0.592. Still monotone; EOS-in-think rate does not track
alpha (2.4 / 3.0 / 4.4 / 2.4%).

Safe to report: the 32k Logit KD lead (shrinks ~2 pp, stays significant) and
the alpha_align dose-response. Not safe: the 8192 per-sample gap and any
4096 ordering - both depend on the scoring convention. Report both
conventions; "fails to close </think>" is itself a real RD behaviour.

---

## 8e. Teacher results — Qwen3.5-4B, MATH-500, 8192, 1 sample (2026-10-16)

`eval_outputs/teacher_qwen4b/`. Compared with the students' 8192 x 8 run
(`sweep_pass_k`), per sample, all 500 questions, paired per question:

| | Teacher | LKD | RD |
|---|---|---|---|
| accuracy strict / lenient | 0.432 / 0.432 | 0.628 / 0.646 | 0.582 / 0.631 |
| truncated | **0.566** | 0.289 | 0.277 |
| EOS inside `<think>` | **0 / 500** | 0.031 | 0.072 |
| committed wrong | **1 / 500** | 0.052 | 0.070 |
| median finished length | 4,106 tok | 1,306 | 1,364 |

Truncation by level 1-5: teacher 0.26 / 0.39 / 0.47 / 0.60 / 0.83.
No repetition loops (0% of truncated tails have >20% repeated 8-grams); the
traces are structured markdown/LaTeX working, just long.

Selection control: on the 217 questions the teacher finishes, it is wrong on 1.
On those same questions LKD commits wrong on 4.1% of answered samples, RD 6.5%;
EOS-in-think LKD 4.4%, RD 9.3%.

Style, finished traces, 217 questions with >=1 finished trace from each model,
medians per 1k tokens (teacher / LKD / RD): Therefore-Thus-Hence 0.26 / 1.99 / 2.25;
verification phrases 0.70 / 1.73 / 1.92; "Wait" 0.74 / 3.50 / 3.41;
"alternatively" 0.00 / 0.70 / 0.74; equation lines 12.7 / 3.0 / 2.4.
Lexical markers under-count the teacher's markdown-header style ("**Verify:**").

Reading:
- Neither RD failure (EOS-in-think, committed-wrong) is a teacher behaviour
  at 8192. EOS-in-think 0/500 is uncensored: truncated teacher traces ran to
  8192 without ending.
- The students' discursive style (Wait / Therefore / verification) is not
  the teacher's free-generation style either; it matches the AM-DeepSeek
  training text. KD trains on teacher logits over that text, not on teacher
  samples.
- Where the students differ, the teacher sits on LKD's side every time
  (fewer EOS-in-think, fewer committed-wrong, fewer Therefore, fewer
  verification phrases, more equation lines). Small next to the teacher-vs-both
  gap.
- Open: committed-wrong on HARD questions is censored (83% of level 5 is
  truncated). Needs the 32k teacher run (configs/eval/teacher_math500_32k.yaml).

---

## 8f. Teacher-forced scoring on held-out AM-DeepSeek traces (2026-10-16)

`scripts/eval_teacher_forced_kl.py`, `scripts/analyze_teacher_forced.py`; output
`eval_outputs/teacher_forced/{results.json,summary.txt}`. 1,000 held-out math
traces (rows 531k-700k; training used 0-52,031) + 200 natural_reasoning
(160k-225k), formatted exactly as in training (`question\nanswer`, no chat
template, 2048 tokens). Teacher in 8-bit as in training. Paired per trace,
bootstrap over traces. Math, think region (1.46M positions):

| | teacher | LKD | RD | base 2B |
|---|---|---|---|---|
| KL(teacher‖model), T=1 | - | 0.195 | 0.201 | 0.164 |
| KL(teacher‖model), T=4 (training temp) | - | 0.0714 | 0.0718 | 0.151 |
| entropy | 0.420 | 0.542 | 0.551 | 0.532 |
| NLL of DeepSeek's token | 0.545 | 0.537 | 0.545 | 0.612 |

- RD is slightly FURTHER from the teacher than LKD: KL1 +0.0052 [0.0048,
  0.0055], 868/1000 traces; natural_reasoning 155/200; answers 170/188. ~3%.
  The alignment term did not pull RD toward the teacher.
- KD worked where it was applied: at T=4 both students are 2x closer to the
  teacher than base. At T=1 they are FURTHER than base - CE (beta 0.75) pulled
  the distribution head toward DeepSeek's text; both students predict DeepSeek
  better than the teacher does.
- Answer tokens inside `\boxed{}` (442 traces): entropy RD - LKD +0.0072
  [0.0048, 0.0102], 320/442; RD - teacher +0.0127, p=0.009; LKD - teacher
  n.s. (p=0.20). Teacher-forced counterpart of RD's wider answer spread, and
  not inherited from the teacher (the teacher is the most certain). Small.
- KILLED: "the teacher's signal says never close </think>". On math the
  teacher gives </think> p~1e-5 at the close, but that is formatting: math
  traces write `.</think>`, natural_reasoning `.\n</think>`, and on the latter
  the teacher gives 0.99. It expects a newline first.
- No EOS-in-think mechanism here: RD vs LKD P(EOS) at the close position is
  +30% on math but -26% on natural_reasoning. Direction flips -> do not report.

Training-format facts (1,500 random training rows): 100% use
`<answer>...</answer>` after `</think>` (eval wants `\boxed` after
`</think>` under the chat template); 41% of math rows are cut at 2048 before
`</think>`; no row ever contains `<|im_end|>` or the chat template. Neither
student was trained on where to stop; stopping comes from base-model priors.

---

## 8g. The SAE alignment targets are ~zero (2026-10-16) - CRITICAL

Checked the HF SAE `qwen-3.5-4B-L16_16x-SAE/sae.pt` and `reasonscore.pt`
(the ones the RD configs load) by running the trainer's exact target path
(`ReasoningAlignmentLoss.encode_teacher_features` on teacher
`hidden_states[16]`, norm-64 rescale, x decoder norms, top-512 ReasonScore
features) over 12 training-range traces, 11,778 tokens, CPU fp32:

- **99 of 6,200,320 target entries are > 0 (1.6e-5); max value 0.053.**
- 20,061 of 40,960 dictionary latents fire somewhere, but only 30 of the 512
  ReasonScore-selected ones ever do.
- The SAE reconstructs nothing at hidden_states[15..18], raw or norm-64:
  fraction of variance explained -0.53 to -0.77.
- Training stopped early and stalled: 8,000 steps, 117M of the planned 500M
  tokens; step_4000 and step_8000 weights are identical; encoder norm 64.4 vs
  69.4 at Xavier init; 100% of encoder biases negative.

Also, independently: `sae_trainer.py` / `reason_score.py` read
`hidden_states[layer + 1]` but `reason_distill_trainer.py` reads
`hidden_states[teacher_align_layer]` - an off-by-one present since both files
were written. The SAE trainer also never rescales to norm 64, while
ReasonScore and the alignment loss do. Neither matters while the targets are
zero (also zero at hidden_states[17]).

Consequence: the alignment term carries essentially no teacher information.
Every part of it - value MSE to 0, presence BCE to 0, rank loss to a flat
vector - can be satisfied by the trainable aligner head driving its outputs
negative. Until it does, the term sends gradients into student layer 12
toward an arbitrary target, and shares the 1.2 grad-norm clip with KD and CE.
That fits alpha_align monotonically reducing accuracy, RD being slightly
further from the teacher (§8f), and the behaviour not tracking alpha. The
clip-sharing mechanism is a hypothesis, not tested.

Caveats: 12 traces; fp32 bf16 teacher vs the 8-bit teacher in training.
Neither plausibly turns 1.6e-5 into a usable signal.
**Implication: "ReasonDistill" as trained is Logit KD + a near-null
regulariser. No Qwen-arm result tests SAE reasoning-feature alignment.**

**Refinement (2026-10-17).** "Carries nothing" overstated it; what is true:
- The term was never inactive. Alpha sweep (shared stage-1 start), (total(a) -
  total(0)) / a ~= alignment loss: 0.27 / 0.25 / 0.20 at the start of training,
  0.19 / 0.15 / 0.08 at the end, for a = 0.05 / 0.10 / 0.50. Gradients flowed
  into student layer 12 throughout, so the representation WAS changed.
- What it changed it toward: the trained RD head
  (`qwen_reasondistill_final_continue/epoch_1/aligner.pt`) applied to RD's own
  layer-12 states (4 traces, 2,048 tokens, CPU) outputs logits mean -4.31,
  P(active) mean 0.018; only 1e-5 of slots exceed 0.5. The student learned to
  make all 512 readouts say "off", which is what a near-all-zero target asks.
  A constant target has no variation across tokens, so it carries no
  information about which tokens the teacher treats as reasoning.
- The remaining signal is ~17 nonzero target slots per 2,048-token example
  (of ~1M), each <= 0.05.
- Scope: Qwen arm only. The Llama arm (`KD_TRAIN_Deepseek_Dataset`) used a
  different SAE (`l19_sae_l1_0.0009.pt`), which was demonstrably live:
  steering its top-15/50 features added +1,227 / +2,235 reasoning tokens in
  the teacher (FINAL_VERDICT §4.1). Nothing here applies to those results.

---

## 8h. Teacher at 32k - 120 questions (2026-10-17)

`eval_outputs/teacher_qwen4b_32k/`, first 120 MATH-500 questions, 1 sample,
batch 6 (batch 8 OOMed at ~26k tokens). 10 h 7 min. Students: the 32k 1-sample
runs on the same ids.

| | Teacher | LKD | RD |
|---|---|---|---|
| accuracy strict / lenient | 0.725 / 0.725 | 0.783 / 0.792 | 0.692 / 0.733 |
| truncated at 32,768 | **26.7%** | 4.2% | 6.7% |
| EOS inside think | **0** | 6 | 5 |
| committed wrong | **1** | 15 | 24 |
| median finished length | 5,864 | 1,904 | 1,688 |

Paired sign tests: committed-wrong teacher-only 0 vs LKD-only 14 (p=0.0001),
0 vs RD-only 23 (p<0.0001). Accuracy n.s. (p=0.30, p=0.60). On the 88
questions the teacher finishes it is wrong on 1; LKD commits wrong on 8 of
those, RD 13. Truncation still 43% at level 5.

Reading: committing to a wrong answer is a student behaviour (both; RD ~1.6x
LKD), absent in the teacher even with 32k tokens. Given §8g it cannot be
attributed to SAE alignment either. 1 sample, small per-level cells.

---

## 8i. Budget forcing - DONE (2026-10-17, 02:55-10:22)

`scripts/eval_budget_forcing.py` (standalone; reuses the harness's chat template,
sampling settings, FinalAnswerStopper and math_equivalent; harness untouched).
LKD vs RD, MATH-500 all 500, 4 samples, budgets 512 / 1024 / 2048 / 4096,
seed 42 per (model, budget). Output `eval_outputs/budget_forcing/`, log
`logs/budget_forcing.log`. Analyse with
`python scripts/analyze_budget_forcing.py eval_outputs/budget_forcing`.

Phase 1 samples up to B tokens (temp 1.0, top_p 0.95, top_k 20). Any sample
not finished naturally (still thinking at B, EOS inside think, or closed think
but cut mid-answer) gets `\n</think>\n\nThe final answer is \boxed{`
(or just the answer prefix if think was closed) and up to 64 greedy tokens.
Three scores per sample: hard_strict (harness convention), hard_lenient
(+ in-think boxed), forced. Forced answer = content of the PREFILLED box: the
smoke test showed models close it and open a second box, so the last-box
extractor read a clipped box and returned "" - fixed before launch.

B=2048 OOMed in phase 2 (64 full-length sequences prefilled at once) after
288 LKD questions; relaunched with phase-2 chunks of 16 and
`PYTORCH_ALLOC_CONF=expandable_segments:True`. B=512/1024 were done with
chunk 64; greedy phase 2 under different left-padding can in principle flip a
rare token, not expected to matter.

Results (paired per question, 4 samples, bootstrap over questions;
`eval_outputs/budget_forcing/summary.txt`):

| B | LKD strict / lenient / forced | RD strict / lenient / forced | RD-LKD strict | RD-LKD lenient | RD-LKD forced |
|---|---|---|---|---|---|
| 512 | .058 / .059 / .398 | .036 / .037 / .379 | -2.2 pp, p=0.001 | -2.2, p=0.001 | -2.0 [-4.0, +0.1], p=0.06 |
| 1024 | .217 / .221 / .558 | .204 / .218 / .536 | -1.4, p=0.17 | -0.3, p=0.78 | -2.3 [-4.3, -0.2], p=0.03 |
| 2048 | .395 / .403 / .663 | .391 / .416 / .617 | -0.4, p=0.72 | +1.4, p=0.21 | **-4.7 [-6.7, -2.5], p=0.0001** |
| 4096 | .529 / .544 / .733 | .502 / .544 / .688 | -2.7, p=0.017 | 0.0, p=1.0 | **-4.6 [-6.6, -2.5], p=0.0001** |

- Validates against the harness: 4096 strict here 0.529 / 0.502 (4 samples)
  vs harness 1-sample 0.540 / 0.514.
- RD never leads. Under hard caps it is level once in-think answers are
  credited (lenient); its strict deficit is its EOS-in-think habit
  (6.1% vs 2.5% of samples at 4096).
- Interrupted mid-reasoning, RD's answer is worse, and more so with budget
  and difficulty: forced-only accuracy RD .394 / .405 vs LKD .460 / .460 at
  2048 / 4096; forced gap by level at 4096 -2.9 / -0.6 / -5.0 / -5.5 / -6.5 pp.
- Not formatting: empty/unclosed forced box <= 1.0% RD vs <= 0.6% LKD.
- Not selection: on questions where both models had forced samples,
  RD - LKD forced-only = -7.2 pp [-10.7, -3.7] (2048, n=354),
  -7.1 pp [-11.3, -2.8] (4096, n=284).
- Reading: at the same point in its reasoning RD is less likely to already
  hold the right answer - consistent with its committed-wrong excess in full
  runs. Per §8g this is not attributable to SAE alignment.

---

## 8j. Llama SAE vs Qwen SAE, measured identically (2026-10-17)

Same 12 AM-DeepSeek training-range traces (~12,000 tokens, <=1,024 each), each
teacher in bf16 on GPU, input rescaled to norm 64 (as the alignment loss
does), targets = ReLU(enc(x)) x decoder-norm, restricted to the ReasonScore
set the RD run used. Script: scratchpad `sae_compare.py`.

| | Llama `l19_sae_l1_0.0009` | Qwen L16, hidden_states[16] (what RD read) | Qwen L16, hidden_states[17] (trained site) |
|---|---|---|---|
| teacher / site | R1-Distill-Llama-8B, block 19 out | Qwen3.5-4B, block 15 out | Qwen3.5-4B, block 16 out |
| d_in x latents | 4096 x 65,536 | 2560 x 40,960 | 2560 x 40,960 |
| fraction of variance explained | **+0.524** | -0.824 | -0.667 |
| L0 mean / median | **126.6 / 124** | 9.9 / 2 | 0.1 / 0 |
| latents nonzero on >=1 token | 88.6% | 49.0% | 1.7% |
| target density (selected set) | **3.7e-3** (196 features) | 1.6e-5 (512) | 1.6e-7 (512) |
| selected features nonzero on >=1 token | 191 / 196 | 31 / 512 | 1 / 512 |
| nonzero target median / max | 1.04 / 29.1 | 0.007 / 0.053 | 0.011 / 0.011 |

Training recipes (from code/config on disk):

| | Llama (`KnowledgeDistillation/base/step2_train.py`) | Qwen (`training/sae_trainer.py`, HF config.json) |
|---|---|---|
| training text | lmsys-openthoughts, teacher's own generations | AM-DeepSeek (DeepSeek-R1 text) |
| input normalisation | rescaled to norm 64; padding (norm<1) dropped | none (raw norm ~10-11) |
| loss | mean-MSE + L1 x decoder norm | same form |
| L1 coefficient | **0.0009** | **5.0** (5% warm-up) |
| init | Kaiming decoder, encoder = decoder^T | Xavier, independent |
| lr / tokens | 5e-5 / ~102M (25k x 4,096) | 2e-5 / 117M (of 500M planned) |
| layer index in RD trainer | hidden_states[L+1] (correct) | hidden_states[L] (off by one) |

Training length is comparable (102M vs 117M tokens), so it is not the
difference. The likely cause is sparsity weight relative to reconstruction:
the L1 coefficient is 5,556x larger, and without normalisation the
per-element input variance is ~(64/11)^2 = 34x smaller, so the penalty is
~1.9e5x heavier relative to the MSE scale. That regime drives latents to
zero (the trained site has L0 0.1). Coefficient 5 matches the reference
paper's config, but that config also normalises activations
(`normalize_activations: expected_average_only_in`). Cause is inferred from
the recipe, not tested by retraining.

---

## 8k. Llama-arm pass@k - DONE (2026-10-17 to 10-19)

Llama-3.2-1B students of DeepSeek-R1-Distill-Llama-8B, epoch 2 of LKD and RD
(read-only from `KD_TRAIN_Deepseek_Dataset/checkpoints/`; the old-convention
RD is excluded on request). Everything else lives in this directory:
`scripts/eval_llama_passk.py`, harness code copied unchanged into
`llama_eval/batched_evals/` (native prompt `{q}\n<think>`, stop at
`</answer>`, commitment-gated extraction, numeric_or_exact_match), output
`eval_outputs/llama_passk/`, log `logs/llama_passk.log`, analysis
`scripts/analyze_llama_passk.py`.

Settings: 8 samples, T=0.7, top_p 0.95 (the arm's earlier pass@4 settings),
seed 42; caps AIME25 / MATH-500 8,192, GSM8K 4,096 (old runs used 3,072).

Why the generation loop differs from the arm's harness: old pass@4 outputs
show 60-78% of truncations are repetition loops (last 4,000 chars >50%
repeated 8-grams), and a static batch runs to its longest row, so the full
job was ~45 GPU-h. Two changes, validated before launch:
- LoopStopper: every 64 steps, a row whose last 400 tokens are >= 85%
  repeated 8-grams stops and is recorded as `loop` (counted as truncated).
  Calibrated on 4,800 earlier LKD/RD samples: fired on 0 / 2,115 that
  finished, 42.5% of the 2,685 truncated (0.75: 3 false stops; 0.6: 25).
- Slot refilling: rows advance 512 tokens per call; finished rows leave and
  queued samples take their slots. A row continues from its own token ids, so
  sampling is unchanged. Smoke (16 GSM8K q): 307 s vs 524 s static.
- A first version flagged every finished row as a loop (its padding repeats);
  fixed by skipping rows whose last token is an end token.
- Guard: the harness falls back to a one-question placeholder if a dataset
  fails to load; the script raises instead.
Estimated ~20-26 h total.

Relaunch note: first launch OOMed in prefill (48 rows x up to 8k tokens);
long rows now run in groups capped at 160k prefill tokens.

**AIME-25** (30 q x 8, 8,192): both 0/240 correct; truncated 97.9% / 99.6%,
loops 87% / 88%. Floor; the old RD pass@4 6.67% does not reproduce.

**MATH-500** (500 q x 8, 8,192), LKD / RD, paired:
- per-sample acc 0.0150 / 0.0180, diff +0.30 pp [-0.22, +0.85], p=0.29;
  pass@8 0.084 / 0.094, RD-only 27 vs LKD-only 22 questions, p=0.57.
- reliability: no question solved on all 8 by either; never solved 458 / 453.
- truncation 71.5% / 74.1% (+2.6 pp, p=0.003), mostly loops (61.4% / 63.6%).
- answered-only tokens: mean 1,007 / 966, median 838 / 804; paired per-question
  median -38 [-104, +28], p=0.25 (RD shorter on 174/311, sign p=0.04 - weak).
- distinct answers/question 2.06 / 1.86 (-0.20, p=0.001) is an artifact of
  RD committing fewer answers (25.9% vs 28.5%): per committed answer the gap is
  -0.002 (p=0.86); equal-count subsampling -0.04 (p=0.36). KILLED.
- Scoring check: old pass@4 per-sample 2.85% vs 1.5% here comes from the
  pre-gate extractor (last number of a loop); answered share (28.3% vs 28.5%)
  and accuracy when answered (5.5% vs 5.3%) match. Raising the cap from 3,072
  to 8,192 did not raise the answered share.

**GSM8K** (1,319 q x 8, 4,096), LKD / RD, paired:
- per-sample acc 0.1141 / 0.1163, +0.22 pp [-0.51, +0.96], p=0.57;
  pass@8 0.389 / 0.396, RD-only 152 vs LKD-only 143, p=0.64.
- reliability: solved on all 8: 3 / 7 (RD-only 6 vs LKD-only 2, p=0.29);
  >= 6 of 8: 35 / 41; never solved 806 / 797.
- truncation 38.4% / 39.9% (+1.5 pp, p=0.013); loops equal (25.6% / 25.7%),
  the excess is long reasoning (12.8% / 14.2%).
- answered-only tokens mean 744 / 725, median 636 / 622; paired per-question
  median -16 [-34, +1], p=0.07 (RD shorter on 646/1260, sign p=0.32).
- distinct answers 3.96 / 3.84 (-0.12, p=0.015) is again the committed-count
  artifact: per committed answer -0.001 (p=0.81), equal-count -0.004 (p=0.88).

Overall (Llama arm, live SAE): no benchmark shows RD ahead on accuracy,
pass@k, reliability, spread or length. The only significant differences are
small truncation excesses for RD (+2.6 pp MATH-500, +1.5 pp GSM8K).
`eval_outputs/llama_passk/summary.txt`.

---

## 8l. October ReasonDistill - basic evals DONE (2026-10-19 to 10-20)

New checkpoint: `Khaquan/qwen-khaquanS-distillations/qwen-reasondistill-october/
qwen_reasondistill_final_continue/epoch_1` (both RD stages retrained with the
layer-indexing fix; trained elsewhere - this checkout's trainer still reads
hidden_states[layer]). Loads correctly (prefix strip, lm_head tied).

Its aligner head differs from the old one: on its own student's layer-12
states (4 traces, 2,048 tokens, CPU) mean P(active) 0.143 (old 0.018), 1.7e-2
of slots > 0.5 (old 1e-5), max predicted value 2.99 (old 1.04). Consistent
with non-trivial alignment targets. The SAE used was not checked (user: not
needed). Alignment loss 0.166 -> 0.072 over the continuation.

Basic evals, new RD only (base / LKD / old RD outputs reused; seed reset per
model and benchmark, so the RNG stream matches):
- `configs/eval/rd_october_math_basic.yaml` = qwen_math_basic.yaml settings (MATH-500 @4096,
  AIME-25 @32768, 1 sample) -> `eval_outputs/test_math_rd_october/`
- `configs/eval/rd_october_math500_32k.yaml` = qwen_math500_32k.yaml settings
  -> `eval_outputs/test_math_rd_october/math500_32k/`
- `configs/september_rd_october_{think,no_think}.yaml` = September set
  (GSM8K, MMLU, ARC-C, HellaSwag; 1,000 q, 4096, batch 32, no answer-stopper,
  as in september-eval-scripts/qwen_kd_comparison-think.yaml) ->
  `eval_outputs/september_rd_october/`. Compare with base / LKD in
  `eval_outputs/september/20260922T180514Z-1112953/`; no local old-RD outputs
  exist for this set.
Queue: `scripts/run_rd_october_queue.sh` (basic math -> September think ->
no_think -> MATH-500 @32k; replaced run_rd_october_basic.sh after its first
stage started). Analysis:
`scripts/analyze_basic_qwen.py` (validated: reproduces old RD vs LKD at 4096).

**MATH-500 @4096 (1 sample), new RD / LKD / old RD / base:** strict
0.498 / 0.540 / 0.514 / 0.088; lenient 0.542 / 0.556 / 0.572. New RD - LKD
strict -4.2 pp [-8.6, +0.4] p=0.08, lenient -1.4 pp p=0.55; stopped inside
think 6.4% vs 2.0% (31 vs 9 questions, p=0.0007); committed wrong 4.2% vs
3.8% (n.s.). New RD vs old RD: no difference (strict p=0.52, lenient p=0.19).
The EOS-in-think habit survives the retrain.

**AIME-25 @32768 (30 q, 1 sample):** strict 0.133 / 0.167 / 0.133 / 0.033
(new RD / LKD / old RD / base); new RD - LKD -3.3 pp [-16.7, +10.0], p=0.82.
Stopped in think 4 / 0 / 4 questions. n=30 single-sample: anecdotal.

**September set** (no answer-stopper: models repeat `<answer>` blocks until
the cap, so truncation / outcome split are undefined; harness `correct` is the
measure; `scripts/analyze_september_qwen.py`). analyze_basic_qwen.py now
refuses these files (a first pass with re-tokenised lengths misread them).
- GSM8K think: new RD 0.819 / LKD 0.809 / base 0.379; RD - LKD +1.0 pp
  [-1.6, +3.7], 98 vs 88 discordant, p=0.51.
- think, all four (n=1000 each):
    think    gsm8k     n=1000 RD_new=0.819 LKD=0.809 base=0.379 | RD-LKD +1.0 pp [-1.6, +3.7] (98 vs 88, p=0.509) | RD-base +44.0 pp [+40.7, +47.3] (459 vs 19, p=0.000)
    think    mmlu      n=1000 RD_new=0.676 LKD=0.683 base=0.619 | RD-LKD -0.7 pp [-3.8, +2.3] (117 vs 124, p=0.699) | RD-base +5.7 pp [+2.6, +8.8] (160 vs 103, p=0.001)
    think    arc-c     n=1000 RD_new=0.820 LKD=0.840 base=0.815 | RD-LKD -2.0 pp [-4.8, +0.9] (95 vs 115, p=0.190) | RD-base +0.5 pp [-2.4, +3.4] (111 vs 106, p=0.786)
    think    hellaswag n=1000 RD_new=0.449 LKD=0.462 base=0.456 | RD-LKD -1.3 pp [-4.8, +2.2] (154 vs 167, p=0.503) | RD-base -0.7 pp [-4.3, +3.0] (171 vs 178, p=0.748)
- no_think, all four (n=1000 each):
    no_think gsm8k     n=1000 RD_new=0.735 LKD=0.768 base=0.794 | RD-LKD -3.3 pp [-6.1, -0.4] (90 vs 123, p=0.028) | RD-base -5.9 pp [-8.8, -2.9] (85 vs 144, p=0.000)
    no_think mmlu      n=1000 RD_new=0.562 LKD=0.539 base=0.559 | RD-LKD +2.3 pp [-1.3, +6.0] (191 vs 168, p=0.246) | RD-base +0.3 pp [-3.2, +3.9] (165 vs 162, p=0.912)
    no_think arc-c     n=1000 RD_new=0.663 LKD=0.664 base=0.724 | RD-LKD -0.1 pp [-3.5, +3.4] (162 vs 163, p=1.000) | RD-base -6.1 pp [-9.5, -2.7] (120 vs 181, p=0.001)
    no_think hellaswag n=1000 RD_new=0.352 LKD=0.370 base=0.425 | RD-LKD -1.8 pp [-5.6, +2.0] (179 vs 197, p=0.381) | RD-base -7.3 pp [-11.0, -3.5] (147 vs 220, p=0.000)

**MATH-500 @32768 (1 sample), new RD / LKD / old RD:** strict 0.668 / 0.780 /
0.682; lenient 0.734 / 0.808 / 0.730. New RD - LKD strict -11.2 pp
[-15.6, -6.8], lenient -7.4 pp [-11.2, -3.6], both p=0.0001. Committed wrong
20.8% vs 12.4% (64 vs 22 questions, p<0.0001); stopped in think 8.6% vs 4.8%
(41 vs 22, p=0.02); truncated 3.8% vs 4.8%. Gap concentrated at levels 4-5
(0.62 vs 0.77; 0.41 vs 0.63). New RD vs old RD: no difference (strict
p=0.57, lenient p=0.89).

Basic-eval verdict: the October RD is statistically indistinguishable from the
old RD on every benchmark, and keeps both RD signatures (EOS-in-think,
committed-wrong excess at large budgets). It does not beat LKD anywhere; LKD
leads significantly at MATH-500 32k and GSM8K no-think.

---

## 8m. October RD pass@8 - DONE (2026-10-20 to 10-21)

`configs/eval/rd_october_passk.yaml`: AIME-25 + MATH-500, 8 samples at 8,192,
settings copied from the 2026-09-29 sweep's run_config.json (T=1.0, top_p .95,
top_k 20, presence_penalty 1.5, answer-stopper on, batch 8 / AIME 2), same
library versions. Only new RD runs; LKD / old RD / base outputs in
`eval_outputs/sweep_pass_k/20260929T073130Z-1261224/` pair per question.
Output `eval_outputs/sweep_pass_k_rd_october/`, log `logs/rd_october_passk.log`.
Expected ~11.3 h (AIME 0.8 h, MATH-500 10.5 h).

**AIME-25 pass@8 (30 q x 8, 8,192)** new RD / LKD / old RD: pass@1 0.075 /
0.104 / 0.100, pass@8 0.167 / 0.233 / 0.267. New RD - LKD pass@1 -2.9 pp
(p=0.14); committed wrong +4.2 pp (p=0.006); stopped in think +2.1 pp
(p=0.03); truncated 83% / 87%. Distinct answers +0.23 (p=0.01) is the
committed-count artifact again (per answer +0.04, p=0.60, n=7). 83-87%
truncated at 8k: AIME is budget-bound here. `summary_aime25.txt`.

MATH-500 OOMed at batch 11/63 of the first launch (8.2 GiB reserved but
unallocated: fragmentation; PYTORCH_ALLOC_CONF alone may not have been read).
Restarted MATH-500 only with identical settings
(`configs/eval/rd_october_passk_math500.yaml`) and both
PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF = expandable_segments:True;
log `logs/rd_october_passk_math500.log`.

**MATH-500 pass@8 (500 q x 8, 8,192)** new RD / LKD / old RD:
pass@1 0.569 / 0.628 / 0.581; lenient 0.625 / 0.645 / 0.630; pass@8 0.852 /
0.858 / 0.862. Solved 8/8: 83 / 128 / 72; never: 74 / 71 / 69.
New RD - LKD: pass@1 strict -5.9 pp [-7.7, -4.2], lenient -2.1 pp [-3.7, -0.5]
(p=0.013); pass@8 tie (12 vs 15 q, p=0.70); all-8-correct 42 vs 87 q
(p=0.0001); committed wrong +2.5 pp, stopped in think +4.5 pp (both
p=0.0001); truncation 27.9% vs 28.9% (n.s.); median finished tokens -80
(p=0.12). New RD vs old RD: no difference on anything.

Answer spread: distinct per committed answer +0.044 [0.029, 0.060], p=0.0001
(n=400) - survives the committed-count control this time. But it mostly
tracks reliability: at equal #correct per question +0.015 [0.001, 0.031],
p=0.04 (n=98); among wrong answers only +0.028, p=0.47 (n=24). Old RD shows
the same pattern (+0.034 overall, +0.009 at equal #correct, n.s.), so it is
not specific to the October run. Reading: RD's answers are more scattered
mainly because it is right less consistently; a small residual at equal
accuracy is borderline. `summary_math500.txt`.

---

## 8n. CoT structure and answer entropy - PLAN WRITTEN BEFORE RESULTS (2026-10-21)

User hypotheses: (1) RD's CoT has distinctive structure in the terms of
Chen et al. 2026, "The Molecular Structure of Thought" (arXiv 2601.06002);
(2) RD has higher-entropy logits at the answer token. The user asked for a
thorough search for RD-favouring patterns. To keep that search honest, every
measure below is fixed now, all are reported, and p-values are Holm-adjusted
within each family. Comparison: RD_new vs LKD (primary), RD_old vs LKD
(replication), paired per question. Data: pass@8 MATH-500 outputs, finished
samples only.

**Method (follows the paper, Appendix 13.2-13.3, 14.3):**
- Steps: split the thinking text on "\n\n", "\n", ". ".
- Labels: each (previous step, current step) pair labelled with the paper's
  published prompt into normal operation / deep reasoning (covalent) /
  self-reflection (hydrogen) / exploration (van der Waals). Annotator:
  Qwen3.5-4B, thinking off (the paper used Qwen2.5-32B-Instruct; not
  available locally). Blind: the annotator never sees which model wrote the
  step. Label = argmax of the four label continuations' log-probabilities.
  Validated by hand-checking a random sample and by agreement with a keyword
  heuristic; both reported.
- Embedding geometry: steps embedded with base Qwen3.5-2B (mean of
  penultimate-layer states), a model neither student was trained from in
  this comparison's sense of being a scorer.
- Entropy dynamics: per-step mean token entropy from each model's own
  teacher-forced logits (`scripts/eval_answer_entropy.py`).

**Family A - bond distribution:** (A1-A4) share of each bond type; (A5)
transition-graph correlation with the other model; (A6) reflection->deep and
(A7) exploration->deep transition rates (repair / productive exploration);
(A8) oscillation rate: switches between {reflection, exploration} and
{deep, normal} per 100 steps.
**Family B - geometry:** (B1) reflection reconnection rate (min distance to an
earlier step below the trace's median step distance); (B2) exploration novelty
(step distance on exploration edges); (B3) trajectory compactness (mean
distance to the trace centroid).
**Family C - entropy dynamics:** (C1) convergence: mean step entropy in the
last third minus the first third; (C2) share of high-entropy-exploration steps
and (C3) low-entropy-validation steps (paper's 14.3 thresholds on step-entropy
change); (C4) entropy-convergence slope of the final 20% of steps.
**Family D - answer entropy (task 2):** (D1) first-answer-token entropy,
(D2) mean answer-token entropy, each self-scored and cross-scored (same text
read by both models); split by correct / wrong.
**Family E - links to correctness:** within each model, which of A-C predict a
correct sample (logistic regression, question fixed effects); then whether RD
has more of the predictive ones.

**Family D results - answer-token entropy (`eval_outputs/answer_entropy/`,
`scripts/analyze_answer_entropy.py`, `summary.txt`).** Up to 4 finished
samples per question per model with a final box (LKD 1,628, RD_new 1,672,
RD_old 1,656 texts), each read by all three models. 25 min per scorer.
- Self-scored, per question: first-answer-token entropy RD_new - LKD +0.0106
  nats [0.0055, 0.0161] (0.048 vs 0.038, ~+28%); mean answer-token entropy
  +0.0068 [0.0039, 0.0098]. RD_old replicates (+0.0087, +0.0051).
- Cross-scored, identical text: RD_new reading either model's text is
  higher at the first answer token by +0.0034 (RD texts) / +0.0039 (LKD texts),
  answer tokens +0.0015 / +0.0017; all p=0.0001. All 12 primary tests Holm
  p <= 0.0012. HYPOTHESIS SUPPORTED, in both RD versions.
- Not only the answer: whole-trace entropy is also higher (self +0.030,
  cross +0.007 / +0.011). But the answer-token excess is larger than the
  trace-wide excess: log-ratio difference +0.029 / +0.034 (RD_new, p=0.0001),
  +0.018 / +0.052 (RD_old).
- Exploratory (not pre-registered): first-answer-token entropy does not flag
  wrong answers for any model - AUROC LKD 0.52 [0.47, 0.58], RD_old 0.49,
  RD_new 0.50. RD's extra answer uncertainty is not better calibrated; it is
  just higher.
- Scale: all models are near-certain at the answer (~0.04-0.05 nats); the
  difference is ~0.01 nats.

**CORRECTION (same day) - answer location.** Explaining the method on one
example exposed two errors in `eval_answer_entropy.py`: (1) it took the LAST
\boxed{} after </think>; 4.5% (LKD) / 6.9% (RD_old) / 6.0% (RD_new) of texts
repeat the box, so a near-certain copy was measured instead of the committed
answer (bias against RD, which repeats more); (2) the first answer token was a
token straddling the opening brace ("{\\"). Fixed: first box after </think>
(last box in thinking if never closed); answer tokens = tokens starting inside
the box. Original outputs moved to `eval_outputs/answer_entropy_v1_lastbox/`
(its per-token trace entropies are unaffected). Self-scored recomputed from the
stored per-token entropies (`scripts/recompute_self_entropy.py`):
  first-answer-token RD_new - LKD +0.0074 [0.0008, 0.0140] p=0.03 (was +0.0106);
  answer tokens +0.0061 [0.0033, 0.0089] p=0.0001 (was +0.0068);
  RD_old +0.0097 (p=0.002) / +0.0049 (p=0.0002).
Direction holds; first-token effect is weaker. Cross-scored numbers above are
v1 (buggy) - corrected rerun queued (`logs/answer_entropy_v2.log`).

**Families A-C, E results (`eval_outputs/bonds/summary.txt`, `scripts/analyze_bonds.py`).**
431 paired questions, 175,521 steps labelled (2.9 h). Annotator check vs a
keyword heuristic: reflection keywords -> labelled reflection 84%;
exploration keywords -> exploration 62% (reflection 20%); no keyword ->
normal 88%. LIMITATION: the 4B annotator almost never assigns "deep
reasoning" (0.4% of steps), so D-based measures (A_D, A_R->D, A_E->D) carry
little information. Hand-check of a random sample still to do.

RD_new - LKD (Holm within family): self-reflection 19.0% vs 17.0% of steps
(+2.0 pp, Holm 0.0007); normal -2.6 pp (Holm 0.0007); oscillation between
{reflection, exploration} and {normal, deep} +1.6 switches / 100 steps (Holm
0.001); trajectory compactness -0.005 (more compact, Holm 0.001); fewer
high-entropy-exploration steps -1.1 pp (Holm 0.002); exploration +0.5 pp
(Holm 0.065; soft-label sensitivity Holm 0.03). Transition graph nearly
identical (r = 0.997). Not significant: reconnection, novelty, R->D,
convergence, tail slope, low-entropy validation.
RD_old replicates reflection (+1.2 pp, Holm 0.03) and normal (-1.3 pp);
oscillation and compactness point the same way but miss Holm.

Control - within every model, more reflection goes with WRONG traces (family
E), and RD is wrong more often. Restricting to questions where both traces
are correct (n=293): RD_new reflection +2.0 pp (p=0.0001), exploration
+0.7 pp (p=0.004), oscillation +2.1 (p=0.0006), compactness -0.004
(p=0.011), same number of steps (-2, p=0.70). RD_old on both-correct (n=301):
reflection +1.2 pp (p=0.015), oscillation +1.2 (p=0.03), compactness -0.004
(p=0.017). So the signature is not a by-product of RD's extra errors: at
equal correctness and equal length, RD reflects more, explores more and
alternates more, in a more compact semantic region.
Interpretation limit: in this data more reflection is associated with wrong
answers within each model, and RD is not more accurate. The paper links these
bonds to learnable long CoT; here they do not translate into accuracy.

**Annotator hand-check (`eval_outputs/bonds/handcheck_labels.json`).** 100
steps, 25 per annotator label, drawn at random with model identity hidden,
labelled by Claude against the paper's definitions (not an independent
human annotation). Agreement 66%, macro-F1 0.65 (paper: >0.85 for its 32B
annotator vs humans). Precision by annotator label: reflection 0.84, normal
0.76, deep 0.56, exploration 0.48 (7/25 exploration steps read as
reflection). The headline rests on reflection, the most reliable label;
exploration results are weaker evidence. Label noise is blind to model, so it
dilutes rather than creates a gap. Deep reasoning appears genuinely rare at
sentence granularity (0/25 "normal" steps were deep on inspection).

**Paper-format tables (`eval_outputs/bonds/paper_format_tables.txt`).**
Marginal pi(b), steps pooled, LKD / RD_old / RD_new: deep 0.6 / 0.7 / 0.6%;
self-reflection 18.4 / 20.0 / 20.7%; exploration 5.6 / 5.4 / 5.3%; normal
75.4 / 73.9 / 73.3%. (Pooled-by-step shares differ slightly from the
per-trace means in the paired tests; exploration's per-trace RD excess does
not appear pooled - weak evidence, as above.) Transfer graphs P(b'|b) are
near-identical (Pearson r 0.997-0.998, the paper's "stable structure"
criterion > 0.9). Semantic-isomer divergence from LKD: KL(pi) RD_new 0.0017,
RD_old 0.0009 vs a 0.0003 LKD split-half noise floor - RD is a semantic
isomer of LKD (same graph, shifted bond weights), not a different structure.
Folding geometry: reflection reconnection 96.5-96.8% for all (paper:
81.7%); covering-ball radius 0.710 / 0.705 / 0.702 (RD ~1% more compact;
paper reports a 22% reduction for its effect).
Exploratory: reflection chaining P(R->R) RD_new +0.030 (p=0.005) but n.s. on
both-correct questions (p=0.17) and in RD_old - tracks errors; not claimed.

**Characteristic example** (test/algebra/2193.json, both correct, 57 vs 53
steps): LKD computes straight through and reflects in one block at the end
(re-derives everything, steps 44-55). RD plants an anticipatory reflection at
step 9 ("Squaring both sides might introduce extraneous solutions, so I'll
have to check the solutions later"), reflects on a second premise (domain of
the square root, steps 38-42), and its reflection steps are spread through
the trace (5, 9, 26, 31, 38, 42, 44, 50) - the higher oscillation count.

**Corrected entropy rerun (v2, `eval_outputs/answer_entropy/summary.txt`).**
Same texts, first box after </think>, tokens starting inside the box.
RD_new - LKD: self first-token +0.0074 (p=0.03, Holm 0.066); self answer
tokens +0.0061 (Holm 0.001); cross first-token on RD texts +0.0053, on LKD
texts +0.0065 (both Holm 0.001); cross answer tokens +0.0009 (Holm 0.009) /
+0.0011 (Holm 0.006). RD_old: 5 of 6 Holm-significant (cross answer tokens on
its own texts n.s.). 10 / 12 primary tests Holm-significant. Final reading:
RD assigns higher entropy to the start of its answer than LKD, most clearly
when both read identical text (~+0.006 nats, ~+13% relative); uninformative
about correctness (AUROC ~0.5, v1).

## 8o. October RD budget forcing - DONE (2026-10-21)

`eval_outputs/budget_forcing/rd_oct__B*.json`, `summary_rd_oct.txt`; analysis
`python scripts/analyze_budget_forcing.py eval_outputs/budget_forcing lkd rd_oct`
(the "rd" row label = rd_oct there). Same settings as 8i (4 samples, budgets
512-4096; phase-2 chunk 16 throughout).

| B | rd_oct - LKD hard strict | hard lenient | forced |
|---|---|---|---|
| 512 | -2.4 pp, p=0.0002 | -2.4, p=0.0002 | -2.9 [-4.8, -1.0], p=0.002 |
| 1024 | -2.2, p=0.03 | -1.4, p=0.17 | -2.9 [-4.9, -0.9], p=0.004 |
| 2048 | -1.4, p=0.19 | +0.1, p=0.95 | -4.9 [-7.0, -2.9], p=0.0001 |
| 4096 | -3.2, p=0.004 | -0.8, p=0.47 | -5.5 [-7.6, -3.5], p=0.0001 |

rd_oct vs old RD: no difference at any budget on any score (all p >= 0.24).
Same picture as old RD: never ahead; forced gap widens with budget; level
with LKD only under lenient hard caps (EOS-in-think 6.0% vs 2.5% at 4096).

---

## 8p. Harder-benchmark candidates (2026-10-22) - for user decision

All on HF and downloaded to ./cache. Contamination vs the student training
prompts (AM-DeepSeek rows 0-52,031), word 13-gram overlap. Any-overlap counts
are dominated by competition boilerplate ("can be written as m/n, where m and
n are relatively prime ... Find m+n") - inspected, false positives. Share of a
problem's 13-grams found in training, >=30% (duplicate) / >=10% (sensitive):
AIME 2024 0/0 of 30; AIME 2025 0/1 of 30; HMMT Feb 2025 0/0 of 30; HMMT Feb
2024 0/0; BRUMO 2025 0/0 of 30; CMIMC 2025 0/0 of 40; AMC 2023 0/0 of 40;
AMC 22-23 (AIMO val) 1/1 of 83; OlympiadBench math EN 5/16 of 674; Minerva
0/0 of 272; BeyondAIME 0/2 of 100 (one is a 1997 China MO problem, 24%
shared - BeyondAIME rewrites old olympiad problems); Omni-MATH 116/145 of
4,428. Rule if used: drop items >= 10%.
Rows 52k-300k (SAE training text, not student training): <= 0.5% everywhere.
Base-model pretraining exposure is unknown but shared by both students.

---

## 8q. Harder benchmarks A/B/C - RUNNING (launched 2026-10-22)

User choice: A (OlympiadBench math EN), B (AIME 2024 + HMMT Feb 2025 + BRUMO
2025 + CMIMC 2025, 4 samples), C (Omni-MATH difficulty >= 6, random 500), all
at 16,384 tokens, LKD vs October RD. Loaders `custom_eval/benchmarks/
hard_math.py` (registered in benchmarks/__init__, config, templates,
generation BOXED_MATH); contamination exclusions in
`custom_eval/benchmarks/contamination_excluded.json` (>= 10% rule:
OlympiadBench 16 -> 658 items; Omni-MATH 145 + 4 empty-gold items before the
seeded 500 draw; competitions 0). Every gold answer self-matches its scorer.
Configs `configs/hard_{A,B,C}.yaml`, chain `scripts/run_hard_math.sh`, logs
`logs/hard_{A,B,C}.log`, outputs `eval_outputs/hard_math/{A,B,C}/`.

Runs in `venv_fla/` (inside this directory): a venv whose .pth reuses the
eval venv's site-packages read-only, plus flash-linear-attention 0.5.2,
fla-core 0.5.2 and einops. The original eval venv is unchanged. causal-conv1d
is not installed (no nvcc), so only the gated-delta-rule kernels are fast.
Validation vs fallback (LKD, one 3,000-token trace): top-1 next-token
agreement 100%, mean |log-prob diff| 0.0096, max 0.21 (bf16 noise).
Speed: 64 rows x 1,024 tokens 1.21x (1,782 vs 1,468 tok/s); 128 rows 1.45x
(2,197 vs 1,517); memory -14 to -18%. Earlier 2-5x estimate was wrong.

**Set A - OlympiadBench math EN (658, 16k, 1 sample), final** (`eval_outputs/hard_math/A/summary.txt`):
    ######## olympiadbench: 658 questions common to all runs, 1 sample each
      RD_new     strict 0.312  lenient 0.356 | committed wrong 0.226  stopped in think 0.070  truncated 0.392 | median finished tokens 4412
      LKD        strict 0.363  lenient 0.378 | committed wrong 0.179  stopped in think 0.021  truncated 0.436 | median finished tokens 4291
      RD_new - LKD:
        accuracy strict   -5.2 pp [-8.8, -1.5] p=0.0053
        accuracy lenient  -2.3 pp [-5.8, +1.4] p=0.2219
        correct (strict)   RD_new-only 56  LKD-only 90  sign p=0.0061
        committed wrong    RD_new-only 102  LKD-only 71  sign p=0.0223
        stopped in think   RD_new-only 45  LKD-only 13  sign p=0.0000
By subfield (exploratory, at 640/658 from the partial): algebra +0.8 pp (tie),
combinatorics -5.4 (n.s.), number theory -9.0 [-17.2, -0.8], geometry -11.3
[-20.2, -3.2]. LKD 4.9 h, RD ~5 h.

**Set B - AIME 2024 + HMMT Feb 2025 + BRUMO 2025 + CMIMC 2025 (130 x 4 samples, 16k), final**
(`scripts/analyze_hard_B.py`, `eval_outputs/hard_math/B/summary.txt`; columns LKD/RD):
    aime24   n=30 | acc 0.067/0.083 | len 0.067/0.083 | trunc 0.817/0.825 | cw 0.117/0.067 | eos 0.000/0.025 | pass@4 LKD 0.200 RD 0.200 (LKD-only 2, RD-only 2, p=1.00)
    hmmt25   n=30 | acc 0.067/0.033 | len 0.067/0.050 | trunc 0.833/0.750 | cw 0.083/0.167 | eos 0.017/0.050 | pass@4 LKD 0.100 RD 0.067 (LKD-only 2, RD-only 1, p=1.00)
    brumo25  n=30 | acc 0.192/0.150 | len 0.200/0.167 | trunc 0.717/0.658 | cw 0.083/0.142 | eos 0.008/0.050 | pass@4 LKD 0.400 RD 0.300 (LKD-only 6, RD-only 3, p=0.51)
    cmimc25  n=40 | acc 0.025/0.031 | len 0.031/0.044 | trunc 0.850/0.775 | cw 0.113/0.150 | eos 0.013/0.044 | pass@4 LKD 0.100 RD 0.125 (LKD-only 2, RD-only 3, p=1.00)
    
    Pooled, 130 problems x 4 samples (LKD / RD):
      per-sample accuracy strict   0.083 / 0.071  RD-LKD -1.2 pp [-4.2, +1.5] p=0.484
      per-sample accuracy lenient  0.087 / 0.083  RD-LKD -0.4 pp [-3.5, +2.5] p=0.856
      truncated                    0.808 / 0.754  RD-LKD -5.4 pp [-10.0, -1.0] p=0.021
      committed wrong              0.100 / 0.133  RD-LKD +3.3 pp [-0.8, +7.3] p=0.120
      stopped in think             0.010 / 0.042  RD-LKD +3.3 pp [+1.3, +5.4] p=0.001
      pass@4 (any of 4 correct)    0.192 / 0.169  LKD-only 12, RD-only 9, sign p=0.664

**Set C - Omni-MATH difficulty >= 6 (500, 16k, 1 sample), final** (`eval_outputs/hard_math/C/summary.txt`).
Pre-registered primary test (written before RD's results): trend of RD - LKD across difficulty 6-9.
Scorer caveat: some Omni-MATH golds are prose ("There are no integral solutions...") that
math_equivalent cannot match - affects both models equally (paired, same scorer).
    ######## omnimath_hard: 500 questions common to all runs, 1 sample each
      RD_new     strict 0.064  lenient 0.074 | committed wrong 0.302  stopped in think 0.062  truncated 0.572 | median finished tokens 9941
      LKD        strict 0.074  lenient 0.076 | committed wrong 0.226  stopped in think 0.010  truncated 0.690 | median finished tokens 10614
      RD_new - LKD:
        accuracy strict   -1.0 pp [-3.6, +1.6] p=0.5060
        accuracy lenient  -0.2 pp [-2.8, +2.4] p=0.9275
        correct (strict)   RD_new-only 21  LKD-only 26  sign p=0.5601
        committed wrong    RD_new-only 92  LKD-only 54  sign p=0.0021
        stopped in think   RD_new-only 30  LKD-only 4  sign p=0.0000
    
    Pre-registered primary test: does RD - LKD grow with difficulty? (paired, bootstrap over problems)
      [strict]
        difficulty 6: n=139  LKD 0.072  RD 0.058  RD-LKD -1.4 pp
        difficulty 7: n=186  LKD 0.059  RD 0.075  RD-LKD +1.6 pp
        difficulty 8: n=109  LKD 0.083  RD 0.055  RD-LKD -2.8 pp
        difficulty 9: n= 66  LKD 0.106  RD 0.061  RD-LKD -4.5 pp
        trend of RD-LKD per difficulty step: -1.22 pp [-4.02, +1.60] p=0.389
      [lenient]
        difficulty 6: n=139  LKD 0.079  RD 0.058  RD-LKD -2.2 pp
        difficulty 7: n=186  LKD 0.059  RD 0.091  RD-LKD +3.2 pp
        difficulty 8: n=109  LKD 0.083  RD 0.064  RD-LKD -1.8 pp
        difficulty 9: n= 66  LKD 0.106  RD 0.076  RD-LKD -3.0 pp
        trend of RD-LKD per difficulty step: -0.57 pp [-3.42, +2.34] p=0.691

---

## 8r. Artifacts and entropy write-up (2026-10-22)

**Answer-token entropy - final reading (logged for the paper).** Method:
teacher-forced forward pass over each finished MATH-500 trace (prompt + full
output); at every generated position the model's next-token distribution over
the 248,320-token vocabulary; entropy H = -sum p log p in nats. The answer is
the first \boxed{} after </think> (the committed answer); "first answer
token" is the distribution that produced the first token starting inside it.
Each model reads its own traces and the other model's (cross-scoring isolates
the model from the text). Stored: `eval_outputs/answer_entropy/rows.json`
(one record per scorer x text), `tokens_<model>.npz` (per-token entropy +
char offsets, own texts), `summary.txt`; v1 (last-box bug) kept in
`answer_entropy_v1_lastbox/`.
Result (v2, corrected): RD_new - LKD first answer token, own answers +0.0074
nats (p=0.03, Holm 0.066); same text +0.0053 (RD texts) / +0.0065 (LKD texts),
Holm 0.001; all answer tokens +0.0061 own (Holm 0.001), +0.0009 / +0.0011
same text. RD_old replicates 5/6. Means ~0.042 (LKD) vs ~0.048 (RD) nats -
both near-certain. Uninformative about correctness (AUROC ~0.5).
Safe claim: "ReasonDistill assigns higher entropy to its final answer than
Logit KD, even reading identical text." Not supported: that this uncertainty
is useful.

**Artifacts.**
- NEW `ReasonDistill's Molecular Signature`
  https://claude.ai/artifact/LN14Yx19fuhBt4M2rqugLt - CoT analysis in the
  paper's format (bond distribution, transfer graphs, semantic-isomer
  divergence, folding geometry, paired tests with both-correct control,
  answer-token entropy, case study, method and limits). Data from
  `scripts/export_cot_artifact_data.py`.
- UPDATED to the October RD (old RD numbers replaced; cells not re-run are
  labelled "earlier ReasonDistill"):
  Budget Forcing https://claude.ai/artifact/4D9j8oViS68BwbeMrtqosk (all four
  budgets re-run); Pass@k Sweep https://claude.ai/artifact/VRwKA1irRJ9KL5esvGQnx1
  and Logit KD vs ReasonDistill https://claude.ai/artifact/CyoakbM8sJK6fu5hDbUitg
  (MATH-500 / AIME new; GSM8K/MMLU/ARC-C/HellaSwag pass@8 and the alpha sweep
  remain earlier RD); Is ReasonDistill More Concise?
  https://claude.ai/artifact/7upRHHAudFxmPvVn872nkt (all RD numbers new).
  Recomputation validated: LKD cap curve and survival curve recomputed by the
  same code match the old pages exactly. Changed conclusions on the concise
  page: truncation gap no longer flips sign (RD 0.6-1.1 pp lower at every
  budget); still no conciseness difference (464 paired, RD shorter on 51.7%,
  sign p=0.49). 32k LKD lead now 11.2 pp (0.780 vs 0.668).

---

## 8s. ProcessBench - DONE 2026-10-23 (tie); option 4 (GSM-Plus) on hold

Question: is ReasonDistill's extra self-reflection (8n) a usable skill? ProcessBench
(Qwen/ProcessBench): given a problem and a step-by-step solution, return the index of
the earliest wrong step (-1 if none). 1,600 items: all 400 GSM8K + seeded 400 each of
MATH / OlympiadBench / Omni-MATH; 611 correct solutions. Official critique prompt,
thinking on, 8,192 tokens, 1 sample, LKD vs October RD. Loader
`custom_eval/benchmarks/processbench.py`, config `configs/eval/qwen_processbench.yaml`, log
`logs/processbench.log`, output `eval_outputs/processbench/`.
Smoke (32 items): truncation 6%, median ~1.4-1.8k tokens; only 13/32 (LKD) and 14/32
(RD) answers are valid step indices - both models often solve the problem instead of
critiquing it (trained only to solve). Official prompt kept; report official F1
(harmonic mean of accuracy on erroneous and on correct items, invalid = wrong) AND F1
restricted to valid indices, plus compliance rate per model.

**Results (2026-10-23; scripts/analyze_processbench.py -> eval_outputs/processbench/analysis*.txt).**
PARSER ARTEFACT FIRST: ~35% of answers wrote the index as `<answer>-1</answer>` with no
\boxed{}, which the harness scores as empty (557 LKD / 501 RD). The smoke-test "low
compliance / solves instead of critiquing" reading was mostly this. Primary scoring now
also reads the answer tag (bare integer, or exactly one "paragraph N"); boxed-only kept as
analysis_boxed_only.txt. (Math evals unaffected: closed-think-no-box < 1%, see side check.)
Tag-aware, official F1 (invalid = wrong), LKD / RD, RD-LKD 95% CI:
  GSM8K 0.513 / 0.466 [-0.127,+0.033] p=0.25 | MATH 0.476 / 0.531 [-0.005,+0.117] p=0.08
  OlympiadBench 0.428 / 0.411 p=0.62 | Omni-MATH 0.417 / 0.353 [-0.132,+0.002] p=0.06
  Pooled 0.460 / 0.443 [-0.051,+0.018] p=0.34. Valid-index-only F1 pooled 0.608 / 0.577 p=0.13
  (Omni-MATH 0.575 / 0.482 p=0.03, favours LKD; not Holm-significant over 4 splits).
Compliance (valid index) 0.758 / 0.767. Flags an error on erroneous solutions 0.637 / 0.641;
answers -1 on correct solutions 0.681 / 0.628 - RD is slightly more prone to false alarms.
Boxed-only scoring had suggested an RD win on MATH (p=0.026); it was compliance, and it
shrinks to p=0.08 once tags are read. Verdict: tie; RD's extra self-reflection (8n) does
not show up as better step-level error finding.

Side check (2026-10-23): "committed wrong" splits into a wrong boxed answer vs closed
</think> with no boxed answer. The latter is < 1% for both models in every run, so the
RD committed-wrong excess is genuinely wrong answers (32k: 63 vs 20 questions;
OlympiadBench 102 vs 71; Omni-MATH 92 vs 55).

---


**Artifacts, 2026-10-23 (all October RD only).** New: "October ReasonDistill Evaluation"
(basic + harder evals; data from scripts/export_eval_artifact_data.py) GncAWQt897ZT21ivZYrFjf;
"Teacher vs Students" 398AF77HbVFj8zKSwtRHJX. Updated: Pass@k (old-RD cells removed),
Molecular Signature (examples section added). Budget Forcing already October RD.

## 8u. Llama arm: KL, CoT bonds, ProcessBench (2026-10-25 to 10-26)

Question: does the second family (Llama-3.2-1B students of R1-Distill-Llama-8B, epoch 2, live
SAE - 8j) agree with the Qwen findings? Llama arm's own format throughout (native prompt
`{q}\n<think>`, LoopStopper, T=0.7, top_p .95, commitment-gated extraction).

**Teacher-forced KL** (`scripts/eval_llama_teacher_forced_kl.py`, `analyze_llama_teacher_forced.py`,
`eval_outputs/llama_teacher_forced/summary.txt`). Llama trained on ALL of am_0.9M, so held-out =
1,000 am_0.5M math rows (openR1Math_extended, streamed; 3,805 of 35,702 scanned math rows dropped
because the question is also in am_0.9M), + 300 in-sample am_0.9M rows (offset 800k). Teacher
8-bit as in training; split by position < / >= 512 (training max_length). Held-out think:
KL1 LKD 0.4750 / RD 0.4735 / base 0.982 (<512), RD - LKD -0.0015 (-0.3%), RD closer on 641/999;
>=512 -0.0021; KL4 -0.0001 (-0.1%). In-sample same direction. Opposite sign to Qwen (+3%), but
negligible. Boxed answer tokens: RD entropy +0.0067 (59/79 traces, p=0.0001; teacher ~0.01).

**CoT bonds** (`scripts/eval_llama_bonds.py`; `analyze_bonds.py --bonds eval_outputs/llama_bonds
--ent eval_outputs/llama_bonds/entropy` -> `summary.txt`, `answer_entropy_summary.txt`). GSM8K
pass@8 outputs (MATH-500 mostly truncates), one finished sample per question, 1,260 questions,
132,638 steps, same annotator / embedder as 8n. Annotator vs keywords: R 82%, E 60%, none->N 90%.
RD - LKD: NOTHING significant in families A, B, C, S (all Holm >= 0.45). Reflection 15.3 vs
15.6% (-0.3 pp, p=0.15 - opposite sign to Qwen's +2.0 pp); oscillation -0.36 (Qwen +1.6);
transition graph r=0.998. Answer-token entropy: no difference self or cross-scored (all p >= 0.09).
Qwen's behavioural signature does NOT replicate in Llama.

**ProcessBench** smoke only (48 shortest items, `eval_outputs/llama_smoke/processbench`): F1 0.000
both (never answer -1 on correct solutions), 31-40% valid indices. Not run in full.

## 8v. Llama arm: outcome split, MATH L1-2, budget forcing (2026-10-26)

**Outcome split of the 8k pass@8 outputs** (`scripts/analyze_llama_outcomes.py`,
`eval_outputs/llama_passk/outcomes_summary.txt`; per-sample rates, paired per question, LKD / RD):

| | strict | committed wrong | truncated |
|---|---|---|---|
| GSM8K | 0.113 / 0.115 ns | 0.502 / 0.485 (-1.7 pp, p=0.007) | 0.384 / 0.399 (+1.5 pp, p=0.02) |
| MATH-500 | 0.015 / 0.018 ns | 0.271 / 0.241 (-2.9 pp, p=0.002) | 0.715 / 0.741 (+2.6 pp, p=0.003) |
| MATH L1+2 (133 q) | 0.037 / 0.050 (p=0.14) | 0.398 / 0.371 ns | 0.566 / 0.579 ns |
| MATH L4+5 | 0.001 / 0.003 | -3.4 pp (p=0.0002) | +3.2 pp (p=0.0004) |

Stopped-in-think and closed-no-answer are 0 for both (Llama training text had no EOS). MATH L1+2
pass@8 0.211 / 0.226 (sign p=0.86); all-8-correct 0 / 0. Neither Qwen RD failure mode (EOS in
think, committed-wrong excess) appears; RD's only trait is a small truncation excess, accuracy tied.
Strict = lenient to 3 dp throughout.

**Budget forcing** (`scripts/eval_llama_budget_forcing.py`, `scripts/analyze_llama_budget_forcing.py`,
`eval_outputs/llama_budget_forcing/summary.txt`; 4 samples, forced = greedy 32 tokens after a
prefilled box; LKD / RD, paired per question):

| bench, B | hard strict | forced | forced RD-LKD |
|---|---|---|---|
| GSM8K 256 | 0.002 / 0.004 | 0.091 / 0.099 | +0.8 pp, p=0.08 |
| GSM8K 512 | 0.052 / 0.056 | 0.136 / 0.136 | 0.0, p=1.0 |
| GSM8K 1024 | 0.107 / 0.101 | 0.140 / 0.136 | -0.4, p=0.49 |
| GSM8K 2048 | 0.110 / 0.119 | 0.132 / 0.148 | **+1.6 pp [+0.5, +2.7], p=0.005** |
| MATH L1-2 512 | 0.023 / 0.015 | 0.092 / 0.070 | -2.3, p=0.13 |
| MATH L1-2 1024 | 0.030 / 0.039 | 0.079 / 0.094 | +1.5, p=0.33 |
| MATH L1-2 2048 | 0.043 / 0.051 | 0.094 / 0.096 | +0.2, p=0.95 |
| MATH L1-2 4096 | 0.051 / 0.038 | 0.075 / 0.085 | +0.9, p=0.57 |

State mixes match within ~1 pp at every GSM8K budget; empty forced answers 0-3 per run. The one hit
(GSM8K B=2048 forced) is Holm-borderline over the 8 forced tests (0.042), its forced-only part is ns
(+1.2 pp, p=0.19), and it does not carry to the full 4096 cap (pass@8 strict tied). Treat as noise.
Unlike Qwen (8i: RD never ahead, forced gap -2 to -5.5 pp), Llama RD is never significantly behind: no
"knows it but doesn't say it" deficit either way.

**Qwen budget forcing by level, October RD** (2026-10-26, re-analysis of the 8i files, no new
generation; `eval_outputs/budget_forcing/summary_rd_oct_by_level.txt`). The 8i run already covered
all of MATH-500, so MATH L1-2 for Qwen needed no GPU. Forced RD-LKD, paired per question:

| B | L1+2 (n=133) | L3 (n=105) | L4+5 (n=262) |
|---|---|---|---|
| 512 | -3.4 pp, p=0.10 | -1.7, p=0.54 | -3.1, p=0.011 |
| 1024 | -2.6, p=0.19 | **-6.4, p=0.006** | -1.6, p=0.27 |
| 2048 | -0.9, p=0.53 | **-8.1, p<0.001** | **-5.6, p<0.001** |
| 4096 | -3.8, p=0.063 | **-7.9, p<0.001** | **-5.4, p=0.001** |

Every cell is negative. On L1+2 the gap is ns at each budget (forced accuracy is 0.70-0.91, a
ceiling), but on the samples both models had to force, RD trails by -4 to -18 pp (4096: -17.9 pp,
p=0.02, n=41). At B=512, L1+2 hard strict is -5.8 pp (p=0.003): RD finishes fewer easy problems
inside 512 tokens. The forced deficit is concentrated in L3 and above.

**Llama verdict (8k, 8u, 8v):** RD = LKD on every measure, apart from a small truncation excess.
The second family neither supports RD nor reproduces Qwen's LKD advantage.

## 8w. Audit of the Llama teacher steering (`unified_sweep`) and patching (2026-10-28)

Sources: `KnowledgeDistillation/base/reasoning_neurons_validation/unified_sweep/` (run 2026-05-28/29),
`base/patching*.py`, `base/pathcing_layer.py`, FINAL_VERDICT §4.1/§7. GPU check
`scripts/mi_llama_sweep_audit.py` (20 MATH-500 prompts x 512 greedy tokens).

1. **Prompts were corrupted.** In the KnowledgeDistillation venv (transformers 5.8.0) the hub
   tokenizer of DeepSeek-R1-Distill-Llama-8B loads as `LlamaTokenizer`, encodes without spaces and
   without BOS, and decodes byte pieces (the Ġ in raw_results). Greedy regeneration with that
   tokenizer reproduces the stored baselines exactly; the correct tokenizer does not. Example:
   "4x − y = 5" was read as "4xy = 5". (The student checkpoints' tokenizer is fine: our 8k-8v
   evals are unaffected.)
2. **"top-1/15/50" is not a feature count.** fmax = 0.98 quantile, but the selected features fire on
   0.02-5.7% of tokens, so most get fmax = 0: top-1 = 0 features, top-15 = 4-5, top-50 = 7-9.
3. **The steering vector is ~3.5x too large.** fmax is measured in the SAE's norm-64 space, the
   vector is added to the raw residual (median norm 18.1 at L19). |delta| = 0.26-0.35x the residual
   norm at top-15 and 1.0-1.2x at top-50. The "dose-response" is a perturbation-size sweep with
   no random direction of matched norm, so it does not show feature specificity.
4. **Top-50 output is degenerate**, not "longer reasoning" (FINAL_VERDICT §4.1/§7.4 say otherwise):
   e.g. "Hmm, okay, so I think I should probably just write down the problem. Alright, so the
   problem is, right, ..." repeated to the cap. Top-15 is coherent and longer (re-verifies).
5. ARC accuracy unreadable: 327/500 baseline outputs unparsed (letters glued to Ġ fail `\b`), no
   chat template for ARC and qualitative prompts. AIME n=30 maj@4, 62% of baseline samples hit
   8,192. MATH/GPQA never ran. gamma only 0 and 2; no suppression arm.
6. **Patching is uninformative by construction**: the metric is "<think> token in the output", but
   the chat template already ends with "<think>\n", so baseline and patched are 0/10 always.
   Forward SAE patch (top-46, last prompt token only), full-residual patch and reverse patch: all
   0/0/0. Ten pairs; the "corrupt" prompts are one-word questions the model also reasons about.
7. Older steering files (gamma_sweep, arc_gamma_sweep, full_gamma_sweep, qualitative_diff, March)
   used `sae_l1_0.0015.pt`, now deleted, with a ReasonScore file overwritten on 2026-03-09:
   not reproducible. full_gamma_sweep is n=2. suppression_test.py has no output anywhere.
   Matryoshka steering is a different SAE never used for RD.

8. **ReasonScore (the RD feature set) is clean.** Recounting its token masks on the same 1,000
   OpenThoughts texts (CPU): saved file 511,000 tokens, 28,334 positive; correct tokenizer 511,000 /
   27,442; broken tokenizer 14,988 positive ("suppose" 80 saved vs 72 correct vs 0 broken). Same
   token total, counts within 3% of the correct tokenizer: it saw correctly tokenised text.
   `reason_result_basesaev{1,2}.txt` list other top features (older SAE runs); the .pt is the one
   md5-matched to RD training. The KD_TRAIN mechanistic suite (FINAL_VERDICT §3.3/§4.2) loads the
   student tokenizer for teacher and students alike (`mechanistic_analysis/common.py`), so is not
   affected.

Verdict: feature selection and the student-side analyses stand; the teacher steering (§4.1) and
all patching are unusable and must be rerun before any causal claim about the feature set.

---

## 8x. Llama teacher feature steering, redone with controls (2026-10-28 to 10-30)

`scripts/mi_steering.py --subject llama_teacher`, analysis `scripts/analyze_mi_steering.py`
(outputs `eval_outputs/mi_steering/llama_teacher/`). Fixes everything in 8w: correct tokenizer
(`tokenizer.json`), fmax re-calibrated on 100 held-out MATH-500 prompts (177/196 ranked features active),
push set as a fraction of the residual norm (h' = h + r|h|u, u = unit Σ fmax_k W_dec_k) so k and sign
are comparable, and every feature push has two controls at the same r: density-matched random SAE features
(randfeat) and a random Gaussian direction (randdir).

**Grid** (100 prompts, greedy, 1024 tokens; k = 1/5/15/50, r = ±0.05…±0.8; 161 conditions).
|r| ≥ 0.4 loops for feature and random pushes alike. ReasonScore vocabulary (wait/hmm/maybe/…, what
the features were selected on) barely moves and never clearly beyond the controls (k15 r+0.2: +3.3 per
1k words vs base, +1.5 vs randfeat p=0.08). The one feature-specific effect is the word "so" (k15 r+0.2
~2,770 vs ~1,565 with randdir; r+0.4 = "so, so, so" loops). Other reasoning words (actually/double-check/
alternatively, "so" excluded) fall 2-7 per 1k at r ≤ -0.1, beyond all controls.

**Full** (pre-registered rule picked k15 r = ±0.2; 200 prompts × 2, T 0.6, 4096 tokens), strict / lenient:

| condition | strict | lenient | tokens | cap |
|---|---|---|---|---|
| base | 0.682 | 0.728 | 2674 | 0.31 |
| feat r+0.2 | 0.565 | 0.615 | 3168 | 0.43 |
| randfeat r+0.2 | 0.600 | 0.635 | 3032 | 0.39 |
| randdir r+0.2 | 0.660 | 0.688 | 2740 | 0.33 |
| feat r-0.2 | 0.677 | 0.708 | 2673 | 0.30 |
| randfeat r-0.2 | 0.723 | 0.740 | 2485 | 0.26 |
| randdir r-0.2 | 0.662 | 0.693 | 2733 | 0.34 |

Paired per question (pp, 95% CI): feat r+0.2 vs base -11.8 [-16.8,-6.8] strict / -11.2 lenient; vs randdir
-9.5 / -7.2 (p≤0.0005); **vs randfeat -3.5 [-8.2,+1.0] p=0.15 / -2.0 p=0.41**. randfeat r+0.2 vs base
-8.2 / -9.2 (p<0.001). feat r-0.2 vs base -0.5 / -2.0 (ns); vs randfeat -4.5 (p=0.04) / -3.2 (ns).

**Verdict.** Pushing along SAE decoder directions makes the teacher think longer and hit the cap more,
costing ~10 pp, but density-matched *random* features do nearly the same (difference ns, strict and
lenient). An isotropic random direction of the same size barely hurts, so the effect belongs to "SAE
feature directions" in general, not to the ReasonScore set. Pushing the set negatively does nothing.
No evidence that the RD feature set causally controls reasoning in the teacher; the old FINAL_VERDICT
§4.1 claim stays withdrawn. Still to run: suppression (clamp-to-0) and patching with a working metric.

## 8y. Llama teacher patching and suppression with controls (2026-10-30)

`scripts/mi_ablation.py` (analysis `scripts/analyze_mi_ablation.py`, outputs `eval_outputs/mi_ablation/llama_teacher/`).
Teacher-forced on the 400 unsteered 4096-token traces of 8x; single-position interventions at L19 via the
clean KV cache (cached clean row reproduces the full forward, KL <= 0.007). Marker = next token is a
ReasonScore word (wait/hmm/maybe/perhaps/but/another/therefore/however; 35 token ids); 2,357 marker and
2,400 control positions. Sets: rd (first 15 / 50 / all 177 active ranked), 5 density-matched random sets,
a random direction of exactly rd's size, and a positive control attr = top latents by grad x activation of
log P(marker) on split A (even ids), tested on split B. Bootstrap over traces.

**Patching (necessity, ablate at marker positions; clean P(marker) ~0.6):** rd lowers P(marker) beyond both
controls at every k, but by little: k15 -0.5 pp, k50 -0.7 pp, all -3.6 pp (rd - rand -2.7 pp, rd - randdir
-3.4 pp, p<0.001; log p of the actual marker -0.14 nats). Same-size random direction ~0. The positive control
on held-out traces: attr k15 -8.1 pp, k50 -16.0, k177 -26.3 pp (attr - rd -22.6 pp at k177).
**Sufficiency (patch rd activations from a marker position into control positions):** P(marker) +0.17 pp at
k177 (vs rand/randdir p<=0.001), ~0 at k15/k50; attr +0.9 pp. Single-position insertion is weak for any set.
**Attribution:** rd median rank 43,642 / 65,536 (no enrichment on average), but 20 of 177 rd features in the
top-200 vs 3-5 for random sets; 11 of the attr top-50 are rd features.

Reading: the ReasonScore set contains a minority of features that causally feed reasoning-word prediction
(effect real, beyond size- and density-matched controls), but most of the set does not, and its total effect
is ~1/7 of what attribution-selected features of the same count achieve.

**Suppression (ablate the set at every position while generating).** Greedy grid (100 prompts, 1024 tokens),
per 1k words vs base / vs mean of 2 random sets: rd k15 ReasonScore words -0.3 / -0.7 (ns); rd k50 -2.3 /
-1.9 (p=0.04); rd all -1.4 / -0.3 (ns), but </think> reached 0.26 vs 0.16 base (+0.10 vs random, p=0.015).
rd k15/k50 raise "so" (+2.8 / +3.9 vs random). attr (positive control): k50 -8.0 / -7.7, k177 -11.5 / -10.3,
</think> 0.37. Full (200 q x 2, T 0.6, 4096; base reused from 8x, one random seed):

| condition | strict | lenient | tokens | cap |
|---|---|---|---|---|
| base | 0.682 | 0.728 | 2674 | 0.31 |
| rd all (177) | 0.735 | 0.755 | 2448 | 0.25 |
| rand 177 | 0.693 | 0.733 | 2637 | 0.30 |
| rd 50 | 0.705 | 0.735 | 2565 | 0.29 |
| rand 50 | 0.705 | 0.735 | 2633 | 0.29 |
| attr 50 | 0.735 | 0.762 | 2405 | 0.24 |

Paired: rd-all vs base strict +5.2 [+1.0,+9.5] p=0.016, lenient +2.8 ns, tokens -227 (p<0.001); vs rand-177
strict +4.2 (p=0.03), lenient +2.2 ns, tokens -189 (p<0.001). rd-50 vs rand-50: 0.0 / 0.0, tokens -68 ns.
attr-50 vs base: +5.2 strict, +3.5 lenient (p=0.08), tokens -269.

Reading: removing the whole set makes the teacher think ~8% shorter and finish more often, beyond a random set
of the same size; accuracy does not drop. The strict gain is the cap mechanism (lenient ns), so it is not a
reasoning-quality effect. The set is causally upstream of *how long* the teacher deliberates (and some of its
reasoning words), not of whether it gets the answer; the top-50 alone does nothing beyond random.
Attribution-selected features do the same, more strongly: the set is a diluted version of them.

## 8z. Qwen teacher steering, patching and suppression with controls (2026-11-02 to 11-05)

Same design and controls as 8x/8y (`scripts/mi_steering.py`, `scripts/mi_ablation.py`, subject `qwen_teacher`,
runner `scripts/run_mi_qwen_teacher.sh`). Analyses: `eval_outputs/mi_steering/qwen_teacher/analysis.txt`,
`eval_outputs/mi_ablation/qwen_teacher/analysis.txt`. Site hidden_states[16] (output of layers[15]), RD's SAE and
ReasonScore top-512. Qwen-specific: incremental prefix cache for the hybrid linear attention (KL vs full forward
0.0000); attribution swaps fla's fused recurrent gated-delta kernel (no backward) for transformers'
`torch_recurrent_gated_delta_rule` (KL vs fused 0.000000). First ablation launch crashed on the missing backward
(2026-11-03 14:08, GPU idle ~22 h until relaunch).

- Calibration (teacher's own MATH generations, 51k tokens): 272/512 ranked features active (>=20 nonzero), 37/50.
- Budget caveat: base hits the 4096 cap on 76% of samples (strict 24.2, lenient 34.3, closed </think> 27%).
  Accuracy at this budget mostly measures finishing in time.
- Steering (full, k15, 200 q x 2): r+0.4 vs base strict -10.8 / lenient -12.0; vs randfeat strict +7.5 (p<0.001),
  lenient +3.2 ns; vs randdir strict -6.0, lenient -6.8. r-0.4 vs base -17.8/-17.2, +271 tokens; vs randfeat
  -7.8/-4.8 (p=0.024), +211 tokens; vs randdir strict -4.0 (p=0.02), lenient -2.0 ns, +85 tokens.
  Grid (greedy 1024): r+0.4 raises markers less than random features (-1.6, p=0.003); r-0.8 raises markers +7.6 vs
  randfeat but = randdir. No sign-consistent reasoning-specific effect; steering = generic disruption.
- Attribution: RD set mean attr ~0 (= random); 0 RD features in top-200 (random 0-1); 0 in attr top-50
  (Llama: 20 and 11).
- Patching: all null. Decisive reason: intervention size. Ablating all 272 RD features changes h by 0.07% of |h|
  (Llama 17%); even the 272 most-attributed features only 0.65% (Llama 31%). Positive control fails too
  (attr all: dP(marker) -0.14 pp vs Llama -26 pp). The SAE's latents carry almost none of the residual, so
  ablation/patching through this SAE has no power on Qwen; the null is about the SAE, not just the RD set.
- Suppression: grid all = random; full rd all vs base strict -1.0 / lenient -1.8, vs rand strict -0.5 / lenient 0.0;
  rd50 vs rand50 -2.0/-1.2; attr50 vs base -3.2/-2.8; all ns. Tokens unchanged.
- Verdict: no mechanistic support for the Qwen RD feature set. Consistent with the dead-SAE finding (8j):
  the Qwen SAE's features are almost inert in the teacher. Llama (8x/8y) remains the only arm with a live,
  partly causal feature set.

## 8aa. Llama students: do they carry the teacher's RD features? (2026-11-05)

`scripts/mi_llama_students.py` (phases probe, pipe), `scripts/analyze_mi_llama_students.py`,
output `eval_outputs/mi_students/llama/analysis.txt`. 800 held-out AM math rows (am_0.5M minus am_0.9M
questions), training format, first 512 tokens, BOS dropped; even rows train, odd rows test (204k tokens
each). Targets: 8-bit teacher hidden_states[20], the trainer's exact encode, the 196 RD features (195 nonzero)
and 196 density-matched random features. Students base / LKD ep2 / RD ep2. R^2 pooled over live features
(>=20 nonzero on test), bootstrap over test sequences, paired.

- Ridge probe, every layer: RD - LKD on the RD set peaks at the guided layer h[11] (block 10 output):
  h[6] -0.0001, h[9] +0.0072, h[11] +0.0208 (0.493 -> 0.514), h[13] +0.0113, h[16] +0.0054. On the random set
  RD - LKD is +0.0003 to +0.0010 everywhere. Difference in differences at h[11] +0.0198 [+0.0193, +0.0203].
  LKD - base at h[11]: +0.0439 (RD set), +0.0223 (random set): KD alone moves the student toward the
  teacher about twice as much as the alignment term adds, and more on the RD features than on random ones.
- RD's own readout (LayerNorm + Linear -> frozen SAE), refitted per student: from a Xavier start every fit
  collapses to "all features off" (train MSE = mean y^2, flat). From a least-squares start (student h[11]
  -> teacher h*64/|h|) then RD's loss: RD 0.829, LKD 0.823, base 0.759 (RD set); random set RD = LKD
  (0.579). RD - LKD +0.0060 [+0.0057, +0.0062], diff-in-diff +0.0061.
- RD's trained aligner: weights barely moved from init (proj std 0.01809 vs Xavier 0.01804; LayerNorm
  weight 1.000 +- 0.002). Applied to each student: RD 0.373, LKD 0.343, base 0.189 (median per-feature R^2
  ~0 for all: it fits a few high-variance features). A random projection gives R^2 -10. The map RD's
  training produced works almost as well on LKD, and a properly fitted readout gets 0.82 from LKD:
  the alignment loss was far from minimised (aligner trained at the student lr 1e-5).
- Verdict: the alignment term changed the student in the intended place (peak at the guided layer) and
  in the intended direction (RD features, not random ones), but the change is small (+2 pp of R^2) next to
  what KD alone does (+4.4 pp). Decodability only; whether the student uses these directions is the next
  (causal) phase.

## 8ab. Llama students: causal tests of the RD-feature directions (2026-11-05 to 11-06)

`scripts/mi_llama_students.py` phases tf, gen; analysis in `eval_outputs/mi_students/llama/analysis.txt`.
Per student, subspaces at h[11] fixed on train rows: rd = top-k singular directions of the ridge readout of
the 196 RD features; rand = same for the density-matched random features (same |dh|/|h|); randdir =
random k-dim subspace (about 2x the |dh|/|h|); aligner = RD's trained readout W_enc[rd].proj.diag(ln_w).
Mean-ablation h <- h - UU^T(h - mu) at every position.

- Teacher-forced, 400 held-out rows, 1,784 reasoning-word targets inside <think>; clean P(marker class)
  0.369 LKD, 0.369 RD, 0.069 base. dP(marker):
  k32  LKD rd -4.7 pp, rand -0.2, randdir -2.2 | RD rd -5.2, rand -0.3, randdir -1.9
  k128 LKD rd -12.2 pp, rand +2.0, randdir -7.0 | RD rd -12.6, rand +1.8, randdir -6.3
  rd - rand and rd - randdir p<0.001 in both students. RD relies slightly more than LKD:
  (RD - LKD) of (rd - randdir) -0.8 pp (k32), -1.1 pp (k128), p<0.001. Aligner directions: LKD hurt
  slightly more than RD at k8/k32. Base barely affected.
- GSM8K generation (500 q x 2, T 0.6, cap 4096), k32. Removing rd directions shortens generation:
  LKD -315 tokens vs none (vs rand -281, vs randdir -173), RD -237 (vs rand -321, vs randdir -175), all
  p<=0.002; fewer loops (0.29 -> 0.22) and caps. Accuracy unchanged: LKD strict +1.6 / lenient +1.8 ns,
  RD -0.6 / -0.4 ns; rd vs rand LKD +1.3/+1.4 ns, RD +2.1/+2.3 (p=0.09/0.06). Aligner directions: RD
  -3.4 / -3.3 vs none (p=0.01), LKD -0.9 / -0.8 ns; RD rand also -2.7 (p=0.048): noisy.
- Verdict: the teacher's RD-feature directions are causally used by both distilled students for reasoning
  words and deliberation length (same signature as the teacher, 8y: suppression shortens, accuracy flat).
  KD installs this; RD adds a small increment in reliance (~1 pp of P(marker)), none visible in generation.

## 9. Outstanding

- **Nothing is committed.** `custom_eval/answer_stopping.py`, the
  `generation.py` / `config.py` / `runner.py` changes, all configs, all
  `eval_outputs/`, `scripts/analyze_like_september.py` and this file are
  working-tree only.
- `configs/eval/qwen_passk.yaml` still has `max_new_tokens: 4096` for math500 and
  gsm8k — the setting proven to report a tie between models differing by 9.8 pp.
  Must be fixed before that sweep runs.
- **pass@k is currently infeasible**: MATH-500 alone at 32768 x 8 samples x 3
  models ~= 310 h. Needs vLLM (10-30x on this workload) before it is worth
  starting.
- `origin/eval-fixes` (`e5ab3bd`, 2026-09-17) never merged — overlapping fixes
  plus a stricter parser. Decide: cherry-pick or close.
- **`september-evaluation-analysis.pdf` does not describe the runs in this
  repo.** All 16 cells differ from `eval_outputs/september/` (up to 3.6 pp); it
  cites RTX 5000 Ada and labels the distilled model ReasonDistill
  (`qwen_reasondistill_final_continue/epoch_1`) "from the supplied config/run
  names", whereas the local run used the uncommitted logitKD redirect. Its
  headline "ReasonDistill leads when thinking is enabled" rests on which
  checkpoint it actually evaluated. **Confirm with its author.**
- One seed, one sample per item, temperature 1.0 throughout. Nothing here
  replicates across seeds — the PDF's own priority-3 item, still open.
- GSM8K / MMLU / ARC-C have never been run at an adequate token budget.

---

## Reproducing the analysis

```bash
python3 scripts/analyze_like_september.py <cap> <result JSONs...>
```

Produces the September PDF's method set: Wilson intervals, paired bootstrap
(20,000 resamples), discordant counts, completion audit, closure-gated accuracy,
length statistics, minutes per 1,000 items.

## 8ac. Llama students: Exp 2-5 follow-ups (2026-11-05 to 11-06)
Outputs under eval_outputs/mi_students/llama/exp{2..5}/analysis.txt; scripts scripts/exp{2..5}_*.py.
Runner scripts/run_mi_llama_exp3_5.sh, no crashes.

### Exp 2 · Feature breakdown (which RD features RD's student gained on)
191 live RD features: 109 with gain CI > 0, 5 < 0; top 25 carry 64% of positive gain. Gainers are dense
discourse/opening features ("Okay, let's tackle this problem step by step.", "let's see", "So,").
Spearman(gain, density) +0.52, (gain, R2 LKD) +0.62, (gain, marker share) -0.16. Marker-firing features
(share > 0.9) gain -0.2 to -0.6 pp. Random set mean gain +0.04 pp vs RD set +0.92 pp. Reading: MSE
alignment improved dense framing features, not reasoning-word features.

### Exp 3 · Feature function transfer (push along feature direction, compare vocab effect vectors)
30 densest RD features + 30 density-matched random. Specific transfer (matched - mismatched corr):
base 0.058 / LKD 0.126 / RD 0.129 on RD set; random set 0.086 / 0.128 / 0.127.
RD - LKD: rd set +0.003 [-0.001,+0.006] p=0.09; rand set -0.001 ns. LKD - base +0.068 (p<0.001).
RD set vs random set: no difference in any student. Verdict: function transfer comes from LKD; RD adds nothing.

### Exp 4 · Student attribution overlap (grad of log P(reasoning word) onto feature directions)
373 texts. Spearman(student attribution, teacher attribution) over RD set: base +0.035, LKD +0.403,
RD +0.395; RD - LKD -0.007 [-0.016,+0.005] p=0.30. Top-20 overlap 11 for both. Teacher's top-10 RD
features have nearly identical ranks in LKD and RD. Caveat: features' share of student total > 1
(write directions not orthogonal), so the share is not interpretable. Verdict: null for RD; LKD alone
gives the teacher-like routing.

### Exp 5 · Layer similarity (linear CKA, 25,600 tokens, all layer pairs)
RD - LKD vs teacher h[20]: +0.0015 at guided h[11] (CI +0.0013,+0.0016), +0.0002 to +0.0023 over layers,
+0.0075 at h[16]. LKD - base at h[11] +0.039 (26x larger). Layer correspondence unchanged (best
teacher layer identical for LKD and RD at every student layer). Verdict: significant but negligible;
RD did not make the student globally more teacher-like.

## 8ad. What did October RD-Qwen actually align to? (2026-11-10)
Prompted by a request to run the Llama student analysis (8aa-8ac) on Qwen at a higher token limit.
Scripts: scripts/sae_old_signal.py, scripts/sae_old_aligner_check.py.

1. Feature signal at 2048 tokens (59k tokens of held-out AM math, Qwen3.5-4B bf16, RD's SAE + top-512):
   teacher hs[16]: density 2.2e-5, 11/512 features nonzero on >=20 tokens; hs[17] (SAE's trained site):
   density 1.3e-7, 0/512 (4 ever nonzero). Llama: 3.7e-3, 191/196. Token length does not change this.
2. HF repo holds one Qwen SAE (qwen-3.5-4B-L16_16x-SAE + step checkpoints); no other SAE was uploaded.
   October aligner vs old aligner: proj.weight corr 0.995, rel diff 0.096 (same seed-42 Xavier init, small drift).
3. Recomputed October RD's alignment loss with its own aligner on 20 rows of its training range
   (40,000-52,031), 2048 tokens, 20,659 tokens:
   | student site | P(active) pred | loss, real targets | loss, all-zero targets |
   | hs[12] | 0.166 | 0.2732 | 0.2732 |
   | hs[13] | 0.020 | 0.0719 | 0.0719 |
   Logged final alignment loss: 0.0717. So October RD read student hs[13] (index fix) and its targets were
   effectively all zero (real = zero to 4 decimals; teacher active rate 1.9e-7 at hs[17]).
   The 8l figure "P(active) 0.143, alignment probably live" was read at hs[12], the wrong site. At the
   trained site the head predicts 0.020 active, same as the old head (0.018).
Verdict: October RD = Logit KD + an "all 512 heads off" regulariser on student hs[13], exactly like the old
RD. The Llama-style feature analysis has no target on Qwen; any result would be null by construction.

Note 2026-11-11: the 4 alpha-sweep checkpoints (checkpoints/qwen_reasondistill_sweep_a0{00,05,10,50}, 88 GB,
not on HF) were deleted at the user's request to free disk. Results remain in eval_outputs/sweep_alpha_8k/
and section 6b; the configs can retrain them.

## 8ae. Side experiment: Qwen SAE retrained with norm-64 input (2026-11-11), docs/rd_newsae.md (was side_experiments/qwen_sae_l16_norm64/)
Isolated; nothing in the main repo changed. Details and tables in that folder's README.md.
Old SAE causes: (a) LR schedule sized by token_batch_size (131k) not real tokens/step, so LR hit 0 at step
3,817 (verified: HF step_4000 == step_8000 bit-for-bit); (b) L1 5.0 kills it even with norm-64 input
(pilot FVE -0.66, L0 0.1); small L1 on raw input keeps latents alive but reconstructs nothing (FVE -0.63).
New SAE (norm 64, L1 0.003, 150M tokens): FVE +0.54, L0 77, 84% never-firing latents (Llama 11%).
Gate on RD's exact target path: top-512 density 1.5e-2, 512/512 live (old: 9e-8, 0/512; Llama 3.7e-3, 191/196);
target mass 7x enriched on reasoning words; enrichment concentrated in ranks 0-195, tail 196-511 ~generic.

### 8af. Side SAE: teacher checks before RD retrain (2026-11-12)
See docs/rd_newsae.md "Teacher checks". Attribution and ablation: the new SAE RD set is causal in the teacher, Llama-sized (rd-rand dlogp -0.081 vs Llama -0.086; old SAE 0). Headroom: base 0.416 / LKD 0.422 / October RD 0.434 / ceiling 0.469 R^2 on the rd set at student hs[13]; October RD +0.011 over LKD on rd set only, unexplained. Generation check running.
Generation (check 2b): RD-196 off -> 28% "Wait." loops and shorter thinking; RD-50 off -> shorter thinking; random sets = base. Accuracy gains (+8.5 / +5.2 vs random) are a 4096-cap artifact (all finished answers correct). Reflection drops only for the attr positive control (-16%).

### 8ag. RD warm-start with the new SAE: Day 1 (2026-11-13)
Nov 5 two-phase scheme (LKD 40k -> RD 12k, alpha 0.05/0.15/0.40) set up with the new SAE (configs/train/rd_newsae_*.yaml, training code = commit a25867d in a separate worktree). Head pre-trained on phase-1 rows (held-out R^2 0.55, AUC 0.90, loss 0.069 vs constant 0.213). Smoke passed (8-bit Adam, 256 tok). Pass bar fixed in docs/rd_newsae.md: control LKD-continue 0.4225 R^2 at hs[13]; must beat it with CI and beat October RD 0.434.

### 8ah. Qwen students causal test, new-SAE RD (2026-11-19/20)
Pre-registered port of 8ab (docs/rd_newsae.md "Student causal test"; mi_qwen_student_causal.py).
Mean-ablating the RD-feature readout subspace at hs[13]: LKD uses it (k128 rd-rand -10.3 pp of P(marker) 0.525),
but RD_mod / RD_high rely LESS, not more: I = +0.27 / +0.35 pp at k32 (p=0.0001), +0.30 / +0.81 at k128,
dose-ordered (low +0.12 ns, Oct RD +0.08 ns, base +0.69); holds after normalising. Pre-registered verdict FAIL. The R^2 gain
from the alignment term is decodability without use. "RD gains teacher-like reasoning" has no support on Qwen.

### 8ai. Reflection style vs success, more traces and benchmarks (2026-11-20)
Pre-registered (docs/rd_newsae.md "Reflection follow-up 2"; eval_reflection_benchmarks.py; ~825k steps
labelled with the Chen et al. 2026 prompt). LKD vs RD_mod on the benchmarks where RD_mod scored above LKD.
Primary FAIL: MATH-500 within-question reflection coupling (correct - wrong traces, 4 traces/q) RD - LKD -0.08 pp
[-1.75, +1.56] p=0.90; neither model's reflection tracks success. No benchmark shows a usable coupling difference
(OlympiadBench +0.79 p=0.42, MMLU +0.85 p=0.42, HellaSwag +0.19 p=0.81; AIME/HMMT/CMIMC have 1-5 usable questions).
Style shift replicates: on questions both models solve, RD reflects more on all 4 benchmarks with enough data:
MATH-500 +2.05 pp (n=342), OlympiadBench +2.23 (n=178), MMLU-think +1.54 (n=528), HellaSwag-think +1.36 (n=309);
Holm-adjusted p 0.0004 / 0.0004 / 0.005 / 0.043. Normal steps drop 1.3-2.7 pp everywhere.

### 8aj. Repo layout cleanup (2026-11-22)
side_experiments/qwen_sae_l16_norm64/ folded into the repo layout (awesomeReasonDistill style): code -> scripts/, configs ->
configs/rd_newsae/, README -> docs/rd_newsae.md, results -> eval_outputs/rd_newsae/, weights -> checkpoints/ (qwen_sae_norm64,
rd_newsae_*), logs -> logs/rd_newsae*. The repo_nov5 worktree (= a25867d, RD phase-2 training code) was removed; scripts that
need it look for a checkout at ../rd_train_a25867d. Root clutter: logs -> logs/, notes and PDFs -> docs/notes/, root scripts ->
scripts/. Full old -> new list: docs/repo_history/restructure_2026-11-22_moves.json. No results were changed.

### 8ak. Paper-repo cleanup (2026-11-22)
Scope = what the paper reports: new-SAE RD vs LKD (all benchmarks), the SAE fix, the Qwen causal test, reasoning style,
the Llama arm, teacher MI, teacher accuracy; October RD as text. Archived to ../archive_paper_cleanup_2026-11-22/ (moved,
nothing deleted): the first RD run's outputs and analyses (RUN_LOG 2-8i), the alpha sweep (6, 6b), prompt / zero-shot
notes, old helpers, examples/, final-run-evals/, smoke runs. Scripts renamed to one scheme (train_ / eval_ / sae_ / mi_ /
analyze_ / run_), configs split into configs/train, eval, mech_interp; this log now uses the current names of files that
still exist (old -> new: docs/REPO_MAP.md). Two analyses that had only been run from edited temporary copies
(hard_math/B and ProcessBench for RD_mod) now take arguments; two inline analyses were saved as scripts
(analyze_difficulty_trend.py, analyze_llama_answer_entropy.py). scripts/run_all_analyses.sh regenerates every reported
summary: the 36 existing ones came out byte-identical; teacher-accuracy summaries (8e, 8h) are new and match those
sections. Summaries are now tracked in git (results/README.md); machine paths became settings; requirements-lock.txt
records the environment that produced the results; tests fixed to the configs as run (45/45 pass). No result changed.
