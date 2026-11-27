# Qwen student mechanistic interpretability: what the RD alignment term does inside the student

Started 2026-11-23. Scripts `scripts/mi_qwen_*` (shared code in `mi_qwen_students_common.py`), launcher
`scripts/run_mi_qwen_students.sh`, outputs `eval_outputs/mi_qwen_students/{diffs,patching,crosscoder}/`,
dictionaries in `checkpoints/mi_qwen_crosscoder/` (not in git).

Behaviour this explains: on questions both models solve, RD_mod's thinking has +2.0 pp more self-reflection steps
than LKD-continue (RUN_LOG 8ai; replicated on OlympiadBench, MMLU-think, HellaSwag-think). Already established, not
repeated here: RD encodes the teacher's RD features slightly better at hs[13] (probe R^2 0.422 -> 0.441), but ablating
that subspace hurts RD *less* than LKD (RUN_LOG 8ah, pre-registered, failed).

## Models and the dose contrast
anc = LKD epoch 1 (start of every run below), lkd = LKD-continue, rd_low / rd_mod / rd_high = new-SAE RD with
alpha 0.05 / 0.15 / 0.40. All see the same 12,032 rows.

First result (Exp 4 weights, before any other phase ran): the three RD runs' weight updates agree with each other
(cosine 0.95-0.995 per layer) and partly with LKD-continue's (0.64-0.75); the part that grows with alpha is
concentrated in layers 9-12, peaking at layer 12, whose output is what RD's head reads. Contrasts reported for every
test below: **rd_mod vs lkd** = what the paper compares; **rd_high vs rd_low** = the alpha dose (same run settings);
from 2026-11-24 also every RD run vs **rd_zero** (alpha 0, same pipeline; section "alpha = 0 ablation").

## Metric
P_R = probability the model gives to starting a reflection step, at step boundaries of fixed text
(`mi_qwen_students_common.py`): the step-labelled MATH-500 traces of RUN_LOG 8n (one finished trace per question from
LKD and from RD_mod, 426 questions), every model run on both trace sets. Reflection-onset tokens fixed on the fit half
(even question index): first tokens of steps that are >= 50% self-reflection with >= 100 occurrences
(`reflection_tokens.json`). Test = odd questions. Statistics: per-question means, bootstrap over questions.

## Pre-registered expectations (written before any result except Exp 4 weights)
**Exp 4 - weight diffs** (`mi_qwen_diffs.py weights`). Descriptive.

**Exp 3 - P_R by model** (`mi_qwen_patching.py gap`).
- Prerequisite: P_R(rd_mod) - P_R(lkd) > 0, CI above 0. If not, the teacher-forced metric does not see the behaviour
  and Exp 3 is reported as such.
- Attribution: rd_high - rd_low > 0 (CI above 0) = the RD term raises P_R; otherwise the gap belongs to the run.

**Exp 1 - shared dictionaries** (`mi_qwen_crosscoder.py`; crosscoder + pooled SAE, hs[13], 32,768 latents, BatchTopK k=64).
- Few or no model-exclusive latents (relative decoder norm < 0.1 or > 0.9) between lkd and rd_mod; more skew for
  lkd -> rd_low (run) than for rd_low -> rd_high (term).
- Reflection-enriched latents (pooled SAE) are more active on reflection tokens in rd_mod than in lkd on the same
  text ("RD amplifies existing reflection features"); prediction for the dose pair: rd_high > rd_low.

**Exp 5 - teacher-feature transfer** (same script, `transfer`). Best correlation of each teacher RD feature with a
student latent: rd_mod > lkd and rd_high > rd_low on the rd set, larger than the same differences on the
density-matched random set.

## Results
(filled in as phases finish; analysis files under `eval_outputs/mi_qwen_students/*/analysis.txt`)

**Exp 3 - P_R by model** (213 test questions x 2 trace sets). anc 0.0963, lkd 0.0969, rd_low 0.1080, rd_mod 0.1081,
rd_high 0.1082. rd_mod - lkd +0.0112 [+0.0105, +0.0119]: the teacher-forced metric sees the behaviour (prerequisite met).
Dose rd_high - rd_low +0.0002 [+0.0001, +0.0003] p=0.003: CI above 0, so by the pre-registered attribution rule the
RD term raises P_R. Dose response from alpha 0 in the section "alpha = 0 ablation".

**Exp 1 - shared dictionaries** (held-out 1.54M tokens per model; FVE crosscoder 0.755-0.765, pooled SAE 0.754-0.762,
L0 ~68, dead 30 / 0 of 32,768). No model-exclusive latents in any pair: all 32,768 crosscoder latents have relative
decoder norm in 0.3-0.7 (sd 0.002-0.005); decoder cosine lkd vs rd_mod median 0.997. Pooled SAE firing rates on identical
text: 29 of 29,594 live latents change by more than 2x between lkd and rd_mod. RD does not add features; it reweights
existing ones (prediction met).
Reflection-enriched latents (top 512 by reflection/other activation, 17.5x enriched), mean activation on reflection
tokens, paired over traces: rd_mod vs lkd +4.4% [+4.3, +4.6]; rd_low vs lkd +1.3%; rd_high vs rd_low +9.8%.
Control added after the pre-registration (label_stats.pt, aggregate ratios): random live latents on reflection tokens
rd_mod vs lkd -1.5%, rd_high vs rd_low +4.2%; all latents +0.9% / +6.4%; the reflection latents on other tokens +0.6% /
+6.3%. So rd_mod vs lkd is reflection-specific (~+6% over random), while about half of rd_high vs rd_low is a general
alpha-driven rise in activation on reflection-step tokens. Enrichment of the reflection latents rises monotonically
with alpha: lkd 17.5, rd_low 17.9, rd_mod 18.2, rd_high 18.5 (rd_zero 17.6, alpha = 0 section). P_R rises with alpha
as well (Exp 3), by a smaller relative amount.

**Exp 5 - teacher-feature transfer** (291,912 tokens). Best correlation with a student latent, teacher RD features:
lkd 0.591, rd_low 0.592, rd_mod 0.594, rd_high 0.598; rd_mod - lkd +0.0035 [+0.0026, +0.0045], rd_high - rd_low
+0.0058 [+0.0046, +0.0071]. Random teacher features: all differences ~0 (p 0.38, 0.69). Prediction met: small, specific
to the RD features, increasing with alpha.

**Verdict.** (1) RD students have a higher propensity to start a reflection step than LKD-continue (P_R +0.011), made
of existing reflection features reweighted (no new features). (2) The RD term has an alpha-dependent footprint at the
aligned layer hs[13]: reflection latents become more selective and teacher RD features are tracked better (random
controls flat), and P_R rises with alpha (CI above 0).

## alpha = 0 ablation (2026-11-24)
Two new runs in the RD pipeline on one Vast RTX A6000 (same image, packages, code a25867d, student init + pre-trained
head, rows 40,000-52,031, seed 42 as the Nov 13 runs): rd_zero = alpha_align 0.0 (ablation), rd_low_rr = alpha 0.05
repeated (replication). Configs configs/train/rd_newsae_{zero,lowrerun}_vast.yaml; checkpoints on OneDrive
RD_newSAE_checkpoints/rd_newsae_{zero,lowrerun}. At alpha 0 the trainer computes 0 x alignment loss, so the term adds
nothing to the update; the logged (unweighted) alignment loss stayed at 0.069-0.073 (alpha 0.05 run: 0.069 -> 0.055).
P_R and weight statistics were computed on the instance (scripts/mi_qwen_alpha0.py; files in
eval_outputs/mi_qwen_students/alpha0/, analysis_vast.txt); the five shared models score as locally (max per-question
difference 0.002).

P_R: anc 0.0963, lkd 0.0969, rd_zero 0.1080, rd_low 0.1080, rd_low_rr 0.1081, rd_mod 0.1081, rd_high 0.1082.
  Dose vs rd_zero: rd_low_rr (same machine) +0.0001 [+0.0000, +0.0002] p=0.02; rd_mod +0.0001 [+0.0000, +0.0002]
  p=0.005; rd_high +0.0003 [+0.0001, +0.0004] p<0.001. P_R increases monotonically with alpha.
  rd_low_rr - rd_low +0.0000 (p 0.75): the pipeline replicates across machines.
Weight updates (vs anc), cosine by layer band 0-8 / 9-12 / 13-23: rd_zero ~ lkd 0.65 / 0.67 / 0.71 (same as every RD run);
rd_zero ~ rd_low 0.995 / 0.995 / 0.996; replication rd_low_rr ~ rd_low 0.995 / 0.996 / 0.997 (machine noise floor).
The term's own change is at the noise floor at alpha 0.05 and visible at 0.40, largest in layers 9-12
(|dW_high - dW_zero| / |dW_lkd| 0.27 in 9-12 vs 0.14-0.17 elsewhere).

Properties at hs[13] (scripts/mi_qwen_alpha0_props.py, alpha0/properties.txt; same procedures as Exp 1 / 5 and the
probe of RUN_LOG 8ah), rd_zero / rd_low / rd_mod / rd_high:
  probe R^2 of the teacher's RD features 0.4228 / 0.4293 / 0.4415 / 0.4640 (vs rd_zero +0.0065 / +0.019 / +0.041, all
  p=0.001); random features 0.2805 / 0.2810 / 0.2814 / 0.2825.
  teacher RD-feature transfer 0.5915 / 0.5924 / 0.5944 / 0.5982 (vs rd_zero +0.0010 / +0.0030 / +0.0068, p<0.001);
  random features 0.4339 / 0.4343 / 0.4342 / 0.4345.
  reflection-latent enrichment 17.59 / 17.91 / 18.23 / 18.46; random latents 0.967 / 0.969 / 0.977 / 0.994. Raw
  activation of the reflection latents on reflection tokens +0.9% / +4.0% / +10.8%; at alpha 0.40 about half of that
  is a general rise in activation scale (+5.5%), so the enrichment ratio is the measure to cite.

**Verdict.** Against a matched alpha = 0 run in the same pipeline, the alignment term produces a dose-dependent
effect: the student's aligned layer encodes the teacher's reasoning features better (probe R^2 +0.041 at alpha 0.40,
transfer +0.007, random-feature controls flat), its reflection latents become more selective, and the probability of
starting a reflection step rises monotonically with alpha (CI above 0 at every alpha). The term's weight change is
concentrated in layers 9-12. The pipeline replicates across machines (alpha 0.05 rerun: P_R difference 0.0000,
weight cosine 0.995-0.997).

## Causal tests of the alignment term vs alpha = 0 (2026-11-25)
`scripts/mi_qwen_alpha_causal.py` (launcher `run_mi_qwen_alpha_causal.sh`), outputs `eval_outputs/mi_qwen_students/alpha_causal/`
(analysis.txt). rd_high (alpha 0.40) vs rd_zero (alpha 0), Exp 3 test traces (213 questions x 2 sets): P_R gap
+0.00025 [+0.00011, +0.00040], log-odds +0.0069 [+0.0043, +0.0095].

**Weight graft** (rd_zero with rd_high's weights in a band). P_R: layers 0-8 +0.0001 (n.s.), layers 9-12 -0.0010,
layers 13-23 +0.0009, all layers +0.0002 (82% of the gap); reverse (rd_high with rd_zero's 9-12) +0.0009. Reflection-latent
enrichment at hs[13] (rd_zero 17.59, rd_high 18.46): + rd_high 9-12 18.27 (79% of the gap), rd_high with rd_zero's 9-12
17.84 (71% removed), + rd_high 0-8 17.84. Probe R^2 of the teacher's RD features at hs[13] (0.4228 -> 0.4640): + rd_high
0-8 0.4346, + rd_high 9-12 0.4355, rd_high with rd_zero's 9-12 0.4347 (needs layers 0-12 together; random set flat).
**Activation patching** rd_high <-> rd_zero, every layer, both directions agree. Log-odds share of the gap recovered
(all positions): ~0 through layer 10, negative at 11 (-2.0), 0.5-0.7 at 13-18, 1.3-1.6 at 19-22, 0.87 at 23.
Probability shares are larger and noisier (-3.2 at L12, +2.6 at L16, +3.5 at L19) because the gap is small.
**Other benchmarks** (finished LKD / RD_mod traces, every model on the same text; 1,190 questions). Pooled vs rd_zero:
rd_low +0.00006 [+0.00003, +0.00010], rd_mod +0.00010 [+0.00006, +0.00014], rd_high +0.00015 [+0.00008, +0.00021], all
p<0.001. Per benchmark (rd_low / rd_mod / rd_high): HellaSwag-think +0.00026 / +0.00037 / +0.00057 (all p<0.001);
MATH-500 second samples +0.00003 / +0.00015 / +0.00014; OlympiadBench +0.00003 / +0.00002 / +0.00000 (n.s.);
MMLU-think -0.00006 / -0.00014 / -0.00013 (rd_mod p=0.004).

**Verdict.** The alignment term acts in two stages. (1) It reshapes the layers that feed the aligned layer (mainly 9-12,
with 0-8 for the probe gain): reflection latents become more selective and the teacher's reasoning features are encoded
better. (2) The downstream layers (13-22) turn this into a higher propensity to start a reflection step; the aligned
layers' own change pushes the other way (graft 9-12 and patching at L11-12 lower P_R). The net increase is dose-dependent
and replicates pooled over four further test sets, carried mostly by HellaSwag-think and MATH-500 (OlympiadBench flat,
MMLU-think slightly negative).
Replication (rd_low_rr = alpha 0.05 repeated on the rd_zero machine) matches rd_low on every property: probe R^2 0.4294 vs
0.4293, teacher-feature transfer 0.5924 vs 0.5924, reflection-latent enrichment 17.90 vs 17.91, pooled benchmark P_R vs
rd_zero +0.00007 vs +0.00006.

## Depth profile of the alignment term (2026-11-26)
`scripts/mi_qwen_alpha_mech.py` (launcher `run_mi_qwen_alpha_mech.sh`), outputs `eval_outputs/mi_qwen_students/alpha_mech/`
(analysis.txt). All contrasts vs rd_zero, paired, CI = 95% bootstrap.
Expectation written after the per-layer probe numbers had been looked at, so M1 is descriptive, not a pre-registered test:
- **M1 probe R^2 by layer** (existing probe files): the gain over rd_zero on the teacher's RD features is positive at
  every layer from hs[13] to the output for rd_high, ordered by alpha at hs[13]-hs[16]; random-feature set flat.

**M1 probe R^2 by layer** (as expected). The encoding gain on the teacher's RD features is positive at every student
layer hs[1]-hs[24] for every alpha (CI above 0), ordered by alpha at every layer, and peaks at the aligned layer:
rd_low / rd_mod / rd_high +0.0065 / +0.019 / +0.041 at hs[13]; downstream +0.028 (hs[14]), +0.012 (hs[16]), +0.006
(hs[20]), +0.0025 (hs[24], rd_high); upstream +0.013 (hs[7]), +0.023 (hs[11]). rd_low_rr equals rd_low at every layer.
Random teacher features: the same shape, about 20x smaller (rd_high +0.0021 at hs[13]). lkd - rd_zero within +-0.0016.
The term's alignment is built up over layers 1-13 and carried, attenuating, to the output.

**Verdict.** The term's representational effect is dose-dependent and network-wide: the teacher's reasoning features
are encoded better at every layer, most at the aligned layer, and the gain reaches the output (M1).

## Teacher agreement and structure of the alignment term (2026-11-27)
`scripts/mi_qwen_alpha_ext.py` (launcher `run_mi_qwen_alpha_ext.sh`), outputs `eval_outputs/mi_qwen_students/alpha_ext/`
(analysis.txt). Teacher = Qwen3.5-4B 8-bit, as in RD training. Contrasts vs rd_zero, paired, CI = 95% bootstrap.
Expectations written before any of these ran:
- **X1 reflection placement vs the teacher.** Teacher and every student teacher-forced on the 426 test traces; reflection
  log-odds at each step boundary. Primary: per-trace Pearson r between student and teacher log-odds over boundaries
  (traces with >= 10 boundaries), averaged per question. Secondary: mean |student - teacher| log-odds at boundaries.
  Expected: r higher for rd_high than rd_zero (CI above 0), ordered rd_low < rd_mod < rd_high; rd_low_rr = rd_low;
  |difference| smaller for rd_high.
- **X2 weight arithmetic.** W(k) = W_zero + k (W_high - W_zero), k = 0.5, 1, 2, 3; control W_zero + k s (W_low_rr - W_low),
  s = same Frobenius norm as the term's change, k = 1, 2, 3. Measured: M1 probe R^2 (all layers), P_R / log-odds on the
  test traces, next-token loss on the test traces (health). Sanity: k = 1 reproduces rd_high. Expected: probe gain at
  hs[13] rises with k up to 2 at least, roughly linearly; control gain within +-0.005; log-odds rises with k; loss rises
  only at k = 3.
- **X4 shape of the weight change** (layers 9-12, every matrix; all layers for reference). Term = rd_high - rd_zero,
  noise = rd_low_rr - rd_low. (a) share of |change|^2 in the top 1 / 8 / 32 singular directions; (b) top-8 subspace
  overlap of (rd_mod - rd_low) with (rd_high - rd_zero) - no checkpoint in common - against the overlap of the noise with
  (rd_high - rd_zero). Expected: the term change is more concentrated than the noise (higher top-8 share) and the two
  alpha steps share a subspace (overlap above the noise control).
- **X5 M1 off math.** The M1 probe (same teacher features, same fit/test procedure) on (a) 400 general-science
  "natural_reasoning" rows of the held-out AM data (rows 531,000+), (b) 400 MMLU-think + HellaSwag-think traces (C4).
  Expected: the hs[13] gain over rd_zero is positive and ordered by alpha on both, smaller than on math; random set smaller.

Added after the first results, before the final analysis: (i) X4 control "train" = rd_low final - rd_low halfway: the
noise (rd_low_rr - rd_low) turned out to be mostly 1-ulp bf16 flips (70-75% of weights identical), so it is not a
structured update; (ii) X5 refit with lambda fixed at 0.1 and at 0.01 for every model: the bench probe chose lambda 0.01
for rd_high but 0.1 for the others at hs[13].

**X1 reflection placement vs the teacher** (prediction met in sign and order, small in size). The teacher's reflection
log-odds over the boundaries of a trace correlate r = 0.913 with every student's. vs rd_zero: rd_low / rd_low_rr / rd_mod
/ rd_high +0.0002 / +0.0002 / +0.0003 / +0.0004 (all CI above 0); mean |student - teacher| log-odds -0.0019 / -0.0014 /
-0.0025 / -0.0026 (CI below 0) on 1.602; lkd - rd_zero n.s. on both. The term moves the student's reflection read-out
toward the teacher's, dose-dependently, by about 0.1-0.2% of the distance.

**X2 weight arithmetic** (prediction met for the probe; partly for loss). Sanity: k = 0 and k = 1 reproduce rd_zero and
rd_high exactly (probe hs[13] 0.4228 / 0.4640, P_R 0.10797 / 0.10822). The probe gain at hs[13] is linear in k:
+0.020 / +0.041 / +0.081 / +0.113 for k = 0.5 / 1 / 2 / 3, and grows with k at every layer hs[1]-hs[24] (output +0.0013 /
+0.0025 / +0.0046 / +0.0071). The norm-matched control (a run-to-run direction scaled 1.85x, 3.7x, 5.6x) stays within
+-0.0005 at every layer. Reflection log-odds: -0.0019 / +0.0069 / +0.051 / +0.130 (P_R 0.1080 -> 0.1090 at k = 3); the
control also raises it, +0.004 / +0.009 / +0.015 at k = 1 / 2 / 3, so the term direction does 1.7x the control at k = 1
and 6-9x at k = 2-3. Next-token loss: flat to k = 1, +0.0027 at k = 2 (+0.6%), +0.0083 at k = 3 (+1.8%); control flat.

**X4 shape of the weight change** (prediction met, and it holds against the stronger training-update control). In
layers 9-12 (25 matrices) the top 8 singular directions carry 12.5% of the term's change, against 1.4% for the noise
and 3.0% for the training update (term larger in 25/25 and 24/25 matrices). The two alpha steps point the same way:
the top-8 output subspace of rd_mod - rd_low overlaps that of rd_high - rd_zero 0.85 (input side 0.86), against 0.008
for the noise and 0.017 for the training update (25/25 matrices). The overlap peaks at layer 12 (0.91) and falls off
on both sides (0.29 at layer 0, 0.05-0.11 at layers 17-23). In layers 13-23 the term is not more concentrated than the
training update (share_8 2.6% vs 3.2%). The term writes a compact, alpha-consistent edit centred on the aligned layer.

**X5 M1 off math** (sign and order met; "smaller than on math" failed: the gain is the same size). hs[13] rd-set gain
vs rd_zero, rd_low / rd_mod / rd_high: general-science text +0.0067 / +0.0195 / +0.0437; MMLU + HellaSwag traces +0.0058
/ +0.0164 / +0.0428 (lambda fixed 0.1: +0.0357; 0.01: +0.0388); math (M1) +0.0065 / +0.019 / +0.041. rd_low_rr =
rd_low; lkd - rd_zero within +-0.002; random set +0.0018 or less (bench rd_high -0.0023 with the chosen lambda, -0.0004
and +0.0016 with lambda fixed). Gains at hs[24]: +0.0021 / +0.0027.

**Verdict.** X2: the term's weight change acts as a dial: scaling it scales the encoding of the teacher's reasoning
features linearly at every layer, while a run-to-run change of the same size does nothing; along it reflection log-odds
also rise, far more than along the control at k >= 2, with a small loss cost. X4: the change is compact and the same
edit at every alpha, centred on layers 9-13. X5: the encoding gain is as large on general-science, MMLU and HellaSwag
text as on math. X1: the reflection read-out moves toward the teacher's, dose-dependently and by a small amount.
