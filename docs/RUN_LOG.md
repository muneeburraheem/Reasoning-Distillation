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
`final_answer_boundary` lifted from `september-eval-scripts/gsm8k_generation_length.py`
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

`configs/test_math.yaml`, 3 models, thinking mode, pass@1, temp 1.0, seed 42,
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

`configs/test_math500_32k.yaml`. Base excluded (its 4096 number is a floor, and
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

`configs/sweep_pass_k.yaml`, all budgets 8192, `num_samples: 8`,
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

## 8c. Teacher experiment — CONFIGURED, BLOCKED ON GPU (2026-10-13)

`configs/teacher_qwen4b_math500.yaml` — Qwen3.5-4B, MATH-500, 8192 tokens,
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

---

## 9. Outstanding

- **Nothing is committed.** `custom_eval/answer_stopping.py`, the
  `generation.py` / `config.py` / `runner.py` changes, all configs, all
  `eval_outputs/`, `scripts/analyze_like_september.py` and this file are
  working-tree only.
- `configs/sweep_pass_k.yaml` still has `max_new_tokens: 4096` for math500 and
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
