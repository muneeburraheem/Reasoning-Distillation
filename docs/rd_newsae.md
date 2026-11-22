# New-SAE experiment: Qwen SAE retrained with norm-64 input, and RD warm-start with it

Formerly `side_experiments/qwen_sae_l16_norm64/`; folded into the repo layout on 2026-11-22 (old -> new names:
docs/REPO_MAP.md). Where things live now:
- Code (scripts/): SAE train_sae_norm64.py, sae_norm64_health.py, sae_norm64_reason_score.py, sae_norm64_gate.py
  (+ _perfeature); RD head train_rd_newsae_head.py, analyze_rd_newsae_head.py, train_rd_newsae_smoke.py; mechanism
  mi_qwen_teacher_newsae.py + analyze_mi_qwen_teacher_newsae.py, mi_qwen_student_probe.py, mi_qwen_student_causal.py;
  reasoning style analyze_reflection_coupling.py, eval_reflection_benchmarks.py; launchers run_sae_norm64_*.sh,
  run_rd_newsae_*.sh; every summary: run_all_analyses.sh.
- Configs: configs/train/ (sae_norm64_full.yaml, sae_norm64_pilot_*.yaml, rd_newsae_{low,moderate,high,smoke}.yaml),
  configs/eval/ (rd_{low,mod,high}_math500_32k.yaml, rd_mod_*.yaml, lkd_seed43_math500_32k.yaml), configs/mech_interp/.
- Results: eval_outputs/rd_newsae/ (sae/, eval/, eval_mod/, student_probe/, student_causal/, teacher_mi/, reflection_more/)
- Weights (gitignored): checkpoints/qwen_sae_norm64/ (SAE runs + eval_hidden.pt), checkpoints/rd_newsae_sae/
  (sae.pt + reasonscore.pt used by RD), checkpoints/rd_newsae_student_init/, checkpoints/rd_newsae_{low,moderate,high}/,
  checkpoints/rd_newsae_probe/ (probe targets and fits)
- Logs: logs/rd_newsae/, logs/rd_newsae_{low,moderate,high,smoke}/
- RD training code: commit a25867d (origin/awesomeReasonDistill), not this branch. train_rd_newsae_head.py,
  analyze_rd_newsae_head.py and train_rd_newsae_smoke.py import it from a checkout: `git worktree add ../rd_train_a25867d a25867d`
  (or set RD_TRAIN_REPO). Phase-2 training = that checkout's scripts/train.py with configs/train/rd_newsae_*.yaml,
  run from this repo root (paths in the configs are relative to it).
Path names below were updated to the new layout; the text is otherwise as written at the time.

Why: the HF SAE RD used is dead (RUN_LOG 8j, 8ad). Causes found:
1. Trained on raw activations (norm ~11), but RD reads it on norm-64 inputs; with L1 5.0 the penalty
   swamps reconstruction. (Inferred; tested by pilots `pilot_norm_l1_5` and `pilot_raw_l1_0015`.)
2. LR schedule length estimated from token_batch_size (131k) instead of real tokens per step (~15k),
   so LR hit 0 at step 3,817 and training stopped (verified: step_4000 == step_8000 bit-for-bit).

What `scripts/train_sae_norm64.py` changes (runtime patches over the repo's own SAE code): norm-64 rescale; schedule from
measured tokens/step; reads cached arrow shards directly (repo loader fails offline here). Configs differ
from configs/train/qwen_sae.yaml in L1 coefficient, sequence_batch_size 4 (4090 memory), token count.

Files: `scripts/train_sae_norm64.py`, `scripts/sae_norm64_health.py` (FVE, L0, dead fraction, reasoning latents), `configs/train/sae_norm64_*.yaml`,
`scripts/run_sae_norm64_pilots.sh`; logs in logs/rd_newsae/, eval.txt in eval_outputs/rd_newsae/sae/, metrics.csv in each
checkpoints/qwen_sae_norm64/<run>/. Baseline (old HF SAE): FVE -0.61, L0 0.0, 95% dead, 0 reasoning latents.

## Pilots (10M tokens each, 2026-11-11), health check on held-out text at hidden_states[17]
| pilot | input | L1 | FVE | L0 | dead | reasoning latents |
|---|---|---|---|---|---|---|
| old HF SAE | raw | 5.0 | -0.61 | 0.0 | 0.95 | 0 |
| norm_l1_5 | norm 64 | 5.0 | -0.66 | 0.1 | 0.92 | 0 |
| raw_l1_0015 | raw | 0.0015 | -0.63 | 39.9 | 0.01 | 0 |
| norm_l1_0005 | norm 64 | 0.0005 | +0.67 | 4104 | 0.00 | 34 |
| norm_l1_0015 | norm 64 | 0.0015 | +0.37 | 788 | 0.00 | 41 |
| norm_l1_005 | norm 64 | 0.005 | -0.44 | 29 | 0.03 | 0 |
Diagnosis: L1 5.0 kills the SAE even with normalised input; a small L1 on raw input keeps latents alive but
reconstructs nothing. Both fixes are needed. Full run: norm 64, L1 0.003, 150M tokens (configs/train/sae_norm64_full.yaml).

## Full run result (150M tokens, L1 0.003, 6 h on 4090): checkpoints/qwen_sae_norm64/full_l1_003/final/sae.pt
FVE +0.544, L0 77, 84% latents never fire on held-out text (Llama 11%), 43 reasoning latents.
ReasonScore (repo script, 10,000 rows, bf16): checkpoints/qwen_sae_norm64/full_l1_003/reasonscore/reasonscore.pt.

## Gate (scripts/sae_norm64_gate.py; exactly RD's target path: 8-bit teacher, hidden_states[17], norm 64, x dec norm, top-512)
| SAE | density | live >= 20 tokens | mean nonzero | mass on reasoning-vocab tokens |
| new | 1.53e-2 | 512/512 | 1.07 | 0.12 (7x the 0.017 base rate) |
| old HF | 9.3e-8 | 0/512 | 0.01 | 0.00 |
| Llama (ref) | 3.7e-3 | 191/196 | ~1.04 | |
PASS. Per-feature (scripts/sae_norm64_gate_perfeature.py): enrichment on/just-before reasoning words falls with rank:
ranks 0-49 median 4.7x / 4.1x, 50-195 2.6x / 3.3x, 196-511 1.2x / 2.1x. The 20 highest-mass features carry
14% of target mass and are generic (median 1.0x); MSE alignment will be weighted toward them (cf. Llama Exp 2).

## Teacher checks before any RD retrain (2026-11-12): scripts/mi_qwen_teacher_newsae.py, scripts/mi_qwen_student_probe.py, scripts/run_sae_norm64_teacher_checks.sh
Repo mi_steering/mi_ablation run unchanged on subject qwen_teacher_l16n64 (layers[16], new SAE, top-196); outputs eval_outputs/rd_newsae/teacher_mi/.
- Check 3 attribution (grad x act of log P(reasoning-word class)): rd-50 mean 5.2e-4 vs density-matched random max 2e-5;
  rd-196 has 60 features in the top-200 of 40,960 (random 2-4); 23 of the top-50 attribution features are RD features. Old SAE: 0 / 0.
- Check 2a necessity (ablate at marker positions): rd-196 dlogp(marker) -0.082, rd-rand -0.081 [-0.095,-0.070]; same-size random
  direction -0.002; positive control (attr set) -0.57. Llama teacher rd-rand -0.086; old Qwen SAE +0.0006 (null). Sufficiency ~0.
- Check 1 headroom (ridge R^2, 400 held-out texts, student hs[13]; ceiling = ridge from teacher hs[17]):
  rd set: ceiling 0.469, base 0.416, LKD 0.422, October RD 0.434. rand set: ceiling 0.340, base 0.275, LKD 0.281, Oct RD 0.280.
  LKD-base +0.006; remaining gap ceiling-LKD 0.047. Llama RD-LKD at its aligned site was +0.021 (rand +0.001).
  UNEXPLAINED: October RD (dead targets) is +0.011 [+0.010,+0.014] over LKD on the rd set only (rand -0.0005), peaked at hs[13-15].
  Not old-SAE overlap: new RD decoder columns have median max|cos| 0.10 with any old latent (new dead latents sit at the shared init, 0.95).
  So a retrained RD must beat October RD (+0.011), not LKD.
- Check 2b (generation, sfull) running.
- Check 2b generation (200 MATH-500 x 2, T 0.6, 4096 cap; eval_outputs/rd_newsae/teacher_mi/analysis.txt). Strict acc. = share finished under the cap
  (every finished answer was correct in all 6 conditions), so accuracy differences are cap effects, not reasoning quality.
  rd-196 off: 28% "Wait." loops, tokens -483 vs rand, finished 127 vs 93 -> strict +8.5 [+4.5,+12.8] vs rand.
  rd-50 off: no loops, finished 108 vs 87 -> strict +5.2 [+1.2,+9.8] vs rand; reflection (non-loop) -0.33/1k (p=0.04, tiny).
  attr-50 off (positive control): reflection -1.11/1k [-1.42,-0.80] vs rand (6.8 -> 5.8), accuracy n.s.
  rand-196 / rand-50 = base. Reading: the RD set sustains long thinking (removal shortens it, at 196 destabilises into loops);
  it is not what produces reflection words in free generation (the attr set is, modestly).

## RD warm-start with the new SAE (Day 1, 2026-11-13)
Scheme = Khaquan's Nov 5 configs (origin/awesomeReasonDistill a25867d): phase 1 = LKD 40k (existing
qwen-logitKD-final/qwen_logit_kd/epoch_1), phase 2 = RD on rows 40,000-52,031, alpha_align low 0.05 / moderate 0.15 /
high 0.40. Code: that commit, checked out as a separate git worktree at the time (removed 2026-11-22; see the top of this file).
- configs/train/rd_newsae_{low,moderate,high}.yaml: Nov 5 configs with only sae_checkpoint, reason_score_path (new SAE),
  reasoning_neuron_count 512 -> 196, student_checkpoint (local copy of LKD epoch_1 + pre-trained head), output dirs.
- Head pre-trained (scripts/train_rd_newsae_head.py): 1,000 rows of 30,000-39,000 (phase-1 range), head only, Nov 5 loss exactly.
  scripts/analyze_rd_newsae_head.py, 40 held-out rows: fitted head loss 0.069 (constant "mean" head 0.213, fresh 1.99); R^2 0.55, AUC(on) 0.90.
  Phase 2 is ~94 optimizer steps; a fresh head starts at loss ~1.8 and would absorb them.
- Smoke (scripts/train_rd_newsae_smoke.py, 8-bit AdamW and 256 tokens for the 4090 only): loads student + head, align loss starts 0.06,
  logged, checkpoint written. Full AdamW at 2048 OOMs on the 4090 (known) -> RunPod.
- October RD vs LKD-continue configs: identical except the RD term (and October used RD in both phases).
### Pass bar (fixed before any phase-2 result)
Control = LKD-continue (same start, same 12k rows, alpha_align 0). Check-1 ridge R^2 at student hs[13], rd set:
  LKD phase-1 0.4228, LKD-continue 0.4225 (12k more LKD changes nothing), October RD 0.4337, teacher ceiling 0.469.
1. Representation: RD - LKD-continue > 0 with bootstrap CI above 0, larger than the same difference on the rand set,
   and RD > 0.434 (October RD).
2. Behaviour: MATH-500 at >= 16k cap (4k is a finish-rate metric), strict and lenient, paired per question vs
   LKD-continue; loops / reflection reported. A behavioural claim needs p < 0.05 AND the same sign on a second seed.
3. Function (secondary): Qwen ports of Llama Exp 3/4.
### Phase-2 results (Vast A6000, 2026-11-13; checkpoints now in checkpoints/rd_newsae_{low,moderate,high}, backup OneDrive RD_newSAE_checkpoints)
Check-1 ridge R^2 at hs[13] (eval_outputs/rd_newsae/vast_probe_analysis.txt), RD - LKD-continue, bootstrap 95% CI:
  low  0.429  rd +0.0069 [+0.0067,+0.0070]  rand +0.0003  -> fails (below 0.434)
  mod  0.441  rd +0.0190 [+0.0187,+0.0192]  rand +0.0007  -> passes
  high 0.464  rd +0.0415 [+0.0410,+0.0418]  rand +0.0018  -> passes (teacher ceiling 0.469)
Dose-monotone in alpha; halfway (step 47) < final for every level. Caveat: the alignment loss trains these
same SAE targets, so this check confirms the objective took hold on held-out text; it is not evidence of better reasoning.
MATH-500 @32k (high, then moderate) queued: scripts/run_rd_newsae_math500.sh, configs/eval/rd_{high,mod,low}_math500_32k.yaml,
compared with LKD-continue in eval_outputs/test_math/math500_32k/ (same seed and settings).
MATH-500 @32k, high (eval_outputs/rd_newsae/eval/analysis_high.txt; scripts/analyze_basic_qwen.py): strict 0.742 vs LKD-continue
0.780 vs October RD 0.668. RD_high - LKD strict -3.8 pp [-8.0, +0.4] p=0.085, lenient -3.2 pp p=0.11 (50 vs 69 discordant);
committed wrong 14.6% vs 12.4% (n.s.); stopped in think 5.0% vs 4.8%. RD_high - October RD +7.4 pp p=0.002.
Gap vs LKD sits at levels 4-5 (0.73/0.55 vs 0.77/0.63). Behaviour pass bar not met: no gain over LKD.
MATH-500 @32k, moderate (eval_outputs/rd_newsae/eval/analysis_moderate.txt): strict 0.746, lenient 0.780. RD_mod - LKD strict
-3.4 pp [-7.8, +1.0] p=0.14 (54 vs 71 discordant); RD_mod - RD_high +0.4 pp p=0.88 (indistinguishable);
RD_mod - October RD +7.8 pp p=0.0009. Representation dose (R^2 0.441 vs 0.464) does not show up in behaviour.
Low queued (scripts/run_rd_newsae_math500_low.sh).
MATH-500 @32k, low (eval_outputs/rd_newsae/eval/analysis_low.txt): strict 0.756, lenient 0.778. RD_low - LKD strict -2.4 pp
[-6.6, +1.6] p=0.27; low vs moderate/high n.s.; low - October RD +8.8 pp p=0.0001.
Summary (strict): LKD 0.780 | low 0.756 | mod 0.746 | high 0.742 | Oct RD 0.668. Accuracy is flat (if anything falling)
in alpha while R^2 rises 0.429 -> 0.464: the representation gain does not transfer. All three sit 2-4 pp under LKD,
none significant; since alpha 0.05 already shows it, the gap may come from the run setup (Vast, pre-trained head,
software versions) rather than the RD term. Only an alpha-0 run on the same setup can separate those.
LKD-continue rerun, seed 43 (sampling-noise check; eval_outputs/rd_newsae/eval/analysis_lkd_seed43.txt): strict 0.770, lenient 0.790
(seed 42: 0.780 / 0.808). LKD43 - LKD42 -1.0 pp [-5.2, +3.2], 113 discordant questions; every RD-vs-LKD pair has 108-125.
LKD43 - RD low/mod/high +1.4 / +2.4 / +2.8 pp, all n.s.; LKD43 - October RD +10.2 pp p=0.0001 (replicates).
Pooled, 3 RD arms vs 2 LKD seeds (per-question means): -2.7 pp [-5.4, +0.1] p=0.054. Reading: new-SAE RD vs LKD is within
seed-to-seed noise per arm; a ~2-3 pp deficit is possible but not shown. October RD's deficit is real.

## Student causal test (Qwen port of Llama 8ab), pre-registered 2026-11-19 before any result
Question: does new-SAE RD's student *use* the teacher's RD-feature directions more than LKD-continue's does?
Script scripts/mi_qwen_student_causal.py; outputs eval_outputs/rd_newsae/student_causal/. Texts/targets reused from eval_outputs/rd_newsae/student_probe (400 held-out
AM rows, 2048 tok; even = fit, odd = test). Per student, at hs[13] (output of layers[12], RD's trained site):
rd = top-k left singular vectors of the ridge map h -> standardised RD-196 targets (fit rows); rand = same for the
196 density-matched random features; randdir = random k-dim subspace; aligner = top-k right singular vectors of
RD_moderate's trained head (proj . diag(ln_w)), same map for every student. Mean-ablation h <- h - UU^T(h - mu).
Teacher-forced on the 200 test rows: at marker targets inside <think> (wait/hmm/maybe/perhaps/but/another/
therefore/however, mi_ablation.marker_ids) dP(marker class); KL and NLL at all positions. k in 8/32/128.
Students: base, lkd (LKD-continue), rd_oct, rd_low, rd_moderate, rd_high (final_step_94).
**Primary (fixed now):** k = 32, effect E(s) = dP(marker | rd) - dP(marker | rand), row means, bootstrap over the
200 test rows (paired, same rows). Contrast I = E(RD) - E(LKD) for RD = rd_moderate and rd_high.
**Pass ("RD routes reasoning-word prediction through the teacher's features more than LKD")** requires all of:
 1. I < 0 with 95% CI below 0 for BOTH rd_moderate and rd_high;
 2. |I| >= 25% of |E(LKD)| for both (Llama 8ab was ~18%: -0.8 of -4.5 pp, judged small; this bar rejects that size);
 3. E(LKD) itself < 0 with CI below 0 (the features are used at all; otherwise the test has no power).
Anything else = fail; report as is. Secondary (descriptive, no claim on their own): k 8/128, randdir and aligner
controls, KL/NLL, rd_low dose ordering, base/rd_oct references.
Generation (secondary): MATH-500, 8192 cap, 1 sample, seed 42, mod_passk settings; lkd and rd_moderate under
rd-k32 and rand-k32 ablation; baseline = the existing 8-sample pass@8 runs (per-question mean). Report strict/lenient,
finished tokens, truncation, loops; RD-vs-LKD difference in the rd-minus-rand change. No accuracy claim from this
unless p < 0.05 and consistent with the teacher-forced result.
### Result (2026-11-20, eval_outputs/rd_newsae/student_causal/analysis.txt): FAIL, opposite sign
Clean P(marker) LKD 0.525, RD_mod 0.550, RD_high 0.551 (4,323 marker targets, 199 rows). E = rd-rand, k32:
LKD -1.33 pp [-1.75,-0.91] (condition 3 met), RD_mod -1.06, RD_high -0.98.
I(RD_mod) +0.27 pp [+0.13,+0.40], I(RD_high) +0.35 pp [+0.19,+0.50]: RD relies LESS on the RD-feature directions,
dose-ordered (high > mod). Same sign at k128 (+0.30 / +0.81 pp) and vs randdir; RD's rd-ablation KL also smaller.
Only k8 leans the other way (-0.1 to -0.2 pp, tiny). Normalised by clean P(marker) or by ablation size |dh|/|h|,
RD is still <= LKD (k32 -11.5/-11.7 vs -15.5 pp per unit; k128 -63/-62 vs -65). Aligner directions: RD = LKD.
LKD itself uses the features (k128 rd-rand -10.3 pp), so the test had power.
Reading: the alignment term made the features more linearly decodable at hs[13] (R^2 +0.019/+0.042) without making
next-token reasoning-word prediction depend on them more; if anything it moved the encoding into directions the
model leans on slightly less. Decodability != use. Same conclusion as Llama 8ab/8ac (KD installs; RD adds ~nothing).
Generation phase (secondary) not run: it cannot rescue a failed primary.
All six students (k32 I vs LKD): base +0.69 (p=0.0001), rd_oct +0.08 (ns), rd_low +0.12 (p=0.06), rd_mod +0.27, rd_high +0.35.
Monotone in alpha; October RD (dead targets) = LKD. KD builds reliance on the teacher's features (base -> LKD); the live
alignment term erodes part of it, in proportion to alpha.

## Reflection follow-up (2026-11-20, scripts/analyze_reflection_coupling.py -> eval_outputs/rd_newsae/eval_mod/bonds/reflection_coupling.txt)
Both-correct control (traces where RD and LKD are both right): RD_mod n=341 reflection 18.8% vs 16.7% (+2.0 pp, p=0.0001),
exploration +0.6 pp (p=0.006), normal -2.8 pp, oscillation +1.9/100 steps (p=0.0001), steps +12 (p=0.02). October RD
(n=290): +1.9 pp / +0.6 / -2.6 / +1.9, same length. RD's reflection excess is not a by-product of errors; replicates in
both RD runs. Reflection-success coupling (correct - wrong, level-stratified): LKD -1.7 pp, RD_mod +1.1, October RD -1.9;
RD_mod - LKD +2.8 pp [-1.7, +6.0] p=0.31 -> sign flips but n.s. (only 42 wrong RD_mod traces). Not claimable.

## Reflection follow-up 2: more traces, more benchmarks (pre-registered 2026-11-20, before any label)
Script scripts/eval_reflection_benchmarks.py -> eval_outputs/rd_newsae/reflection_more/. Same step labelling as bonds (Chen et al. prompt, Qwen3.5-4B
annotator, argmax), LKD vs RD_mod, existing eval outputs only (no new generation). Benchmarks = those where RD_mod
scored numerically above LKD and that have thinking traces: MATH-500 pass@8 (4 finished traces/question), AIME-25
pass@8 (all finished), AIME-25 32k, OlympiadBench, AIME-24, HMMT-25, CMIMC-25 (all finished of 4), MMLU and
HellaSwag with thinking (1 trace). Selection rule stated here so it can be reported: "benchmarks where RD was ahead".
Finished = not cut by the token cap (September-format sets: thinking block closed). Labels for MATH-500 traces already
labelled in eval_mod/bonds are reused.
**Primary (fixed now):** MATH-500, within-question reflection coupling. For each model and each question with at least
one correct and one wrong trace: mean reflection share of correct traces minus wrong traces; model coupling = mean over
those questions. I = coupling(RD) - coupling(LKD), bootstrap over questions (10,000). Supported if I > 0 with CI above 0.
Secondary (descriptive): each model's coupling with CI; same per other benchmark (within-question where several traces,
else plain correct - wrong); reflection / exploration / oscillation shares RD - LKD on both-correct questions per
benchmark (replication of the style shift); pooled over the non-MATH benchmarks.
**Primary result (2026-11-20 07:25):** NOT SUPPORTED. MATH-500, 4 traces/question (LKD 1631 traces / 199 wrong; RD
1608 / 196 wrong). Within-question reflection coupling LKD -0.09 pp [-1.13, +0.97] (136 q), RD -0.18 [-1.57, +1.22]
(125 q); I = -0.08 [-1.75, +1.56] p=0.90. Neither model's reflection share tracks success within a question; the earlier
one-trace +2.8 pp was noise. Exploration and oscillation coupling also n.s. (I +0.50 p=0.28; +1.14 p=0.24).
Both-correct style shift replicates with 4x the traces (n=342 q, first finished trace): reflection 18.80% vs 16.75%
(+2.05 pp [+1.15, +2.96] p=0.0001), exploration +0.59 (p=0.006), normal -2.74 (p=0.0001), oscillation +1.84/100 steps
(p=0.0004). Other benchmarks: still labelling (see eval_outputs/rd_newsae/reflection_more/analysis.txt when done).
**All benchmarks (2026-11-20 16:35, eval_outputs/rd_newsae/reflection_more/analysis.txt):** coupling RD - LKD: OlympiadBench +0.79 pp
p=0.42, MMLU-think +0.85 p=0.42, HellaSwag-think +0.19 p=0.81, AIME-25 p8 -0.13, AIME-25 32k -1.20 (all n.s.); AIME-24 /
HMMT-25 / CMIMC-25 have 1-5 questions with both outcomes, so their bootstrap CIs are degenerate (the p=0.0002 values for
HMMT +1.65 and CMIMC -4.88 are artefacts; ignore). RD above LKD on 5 of 8, mean -0.24. No coupling effect anywhere.
Both-correct reflection shift: MATH-500 +2.05 (n=342), OlympiadBench +2.23 [+0.98, +3.45] (n=178), MMLU-think +1.54
[+0.57, +2.53] (n=528), HellaSwag-think +1.36 [+0.06, +2.61] (n=309); Holm p 0.0004/0.0004/0.005/0.043. Exploration up on
MATH and MMLU only; oscillation up on MATH, Olympiad, MMLU, not HellaSwag. AIME sets: too few both-correct questions.
