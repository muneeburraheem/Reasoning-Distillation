# Qwen student MI suite: run tracker

Everything run for docs/mi_qwen_students.md (2026-11-23 onwards): what ran, where its code, outputs and logs are, and
its status. Use it to decide what goes into the final repo. Results and verdicts are in docs/mi_qwen_students.md.

## Experiments

| # | Experiment | Script (phases) | Launcher | Outputs | Log | Status |
|---|---|---|---|---|---|---|
| 1 | Shared dictionaries: crosscoder + pooled SAE at hs[13], reflection latents | `mi_qwen_crosscoder.py` (train, eval, label, transfer) | `run_mi_qwen_students.sh`, `logs/mi_qwen_students/queue_rest.sh` | `eval_outputs/mi_qwen_students/crosscoder/`; weights `checkpoints/mi_qwen_crosscoder/` (3.2 GB) | `logs/mi_qwen_students/crosscoder.log` | done |
| 3 | P_R by model | `mi_qwen_patching.py` (gap, analyze) | `run_mi_qwen_students.sh` | `eval_outputs/mi_qwen_students/patching/` | `logs/mi_qwen_students/patching.log` | done |
| 4 | Weight updates vs LKD epoch 1 | `mi_qwen_diffs.py` (weights, analyze) | `run_mi_qwen_students.sh` | `eval_outputs/mi_qwen_students/diffs/` | `logs/mi_qwen_students/diffs.log` | done |
| 5 | Teacher-feature transfer | `mi_qwen_crosscoder.py transfer` | as Exp 1 | `eval_outputs/mi_qwen_students/crosscoder/transfer.json` | as Exp 1 | done |
| A0 | alpha = 0 ablation run + alpha 0.05 rerun (Vast RTX A6000, $3.64) | trainer at a25867d; configs `configs/train/rd_newsae_{zero,lowrerun}_vast.yaml` | Vast bundle (not in repo) | checkpoints `checkpoints/rd_newsae_{zero,lowrerun}/final_step_94` (OneDrive `RD_newSAE_checkpoints/`) | `logs/rd_newsae_zero`, `logs/rd_newsae_lowrerun`, `logs/rd_newsae/vast_runlogs_zero/` | done (both checkpoints on this machine, sha256-verified) |
| A1 | P_R + weight cosines with rd_zero / rd_low_rr (computed on the Vast instance) | `mi_qwen_alpha0.py` (pr, weights, analyze, analyze_vast) | on instance | `eval_outputs/mi_qwen_students/alpha0/{gap_vast.jsonl,weights_vast.json,analysis_vast.txt}` | `logs/rd_newsae/vast_runlogs_zero/` | done |
| A2 | Properties of every student (Exp 1 enrichment, Exp 5 transfer, probe R^2) vs rd_zero / lkd | `mi_qwen_alpha0_props.py` (model NAME, scale_check, compare); probe `mi_qwen_student_probe.py` | `run_mi_qwen_alpha0_props.sh` | `eval_outputs/mi_qwen_students/alpha0/{props_*.pt,properties.txt}`; probes `checkpoints/rd_newsae_probe/probe_rd_zero.pt` | `logs/mi_qwen_students/alpha0_props.log` | done (rd_zero, rd_low_rr) |
| C1 | Activation patching rd_high <-> rd_zero, all layers, all / boundary positions | `mi_qwen_alpha_causal.py patch` | `run_mi_qwen_alpha_causal.sh` | `eval_outputs/mi_qwen_students/alpha_causal/patch.jsonl` | `logs/mi_qwen_students/alpha_causal.log` | done 2026-11-25 |
| C2 | Weight graft rd_high layers -> rd_zero (bands, single layers 9-12, reverse); P_R, enrichment, probe R^2 | `mi_qwen_alpha_causal.py graft` | as C1 | `alpha_causal/graft.json`; probes `checkpoints/rd_newsae_probe/probe_graft_*.pt` | as C1 | done |
| C4 | P_R on OlympiadBench, MMLU-think, HellaSwag-think, MATH-500 2nd samples, every model | `mi_qwen_alpha_causal.py bench` | as C1 | `alpha_causal/{bench_traces.json,bench.jsonl}` | as C1 | done (incl. rd_low_rr) |
| | Summary of C1, C2, C4 | `mi_qwen_alpha_causal.py analyze` | as C1 | `alpha_causal/analysis.txt` | | done |
| M1 | Probe R^2 of the teacher's RD features at every student layer, every model vs rd_zero (existing probe files) | `mi_qwen_alpha_mech.py analyze` | `run_mi_qwen_alpha_mech.sh` | `eval_outputs/mi_qwen_students/alpha_mech/analysis.txt` | `logs/mi_qwen_students/alpha_mech.log` | done 2026-11-26 |
| X1 | Reflection placement vs the teacher (per-boundary log-odds correlation) | `mi_qwen_alpha_ext.py teacher, students` | `run_mi_qwen_alpha_ext.sh` | `eval_outputs/mi_qwen_students/alpha_ext/{x1_lo.jsonl,analysis.txt}` | `logs/mi_qwen_students/alpha_ext.log` | done 2026-11-27 (kept) |
| X2 | Weight arithmetic rd_zero + k (rd_high - rd_zero), norm-matched run-to-run control; P_R, loss, M1 probe | `mi_qwen_alpha_ext.py arith` | as X1 | `alpha_ext/x2_arith.json`; probes `checkpoints/rd_newsae_probe/probe_x2_*.pt` | as X1 | done 2026-11-27 (kept) |
| X4 | SVD shape and subspace overlap of the weight change | `mi_qwen_alpha_ext.py svd` | as X1 | `alpha_ext/x4_svd.json` | as X1 | done 2026-11-27 (kept) |
| X5 | M1 probe off math (natural_reasoning rows, MMLU/HellaSwag traces), auto and fixed lambda | `mi_qwen_alpha_ext.py teacher, students, x5fix` | as X1 | `alpha_ext/x5_*`; targets `checkpoints/rd_newsae_probe/ext/x5*_targets.pt`; probes `probe_x5*.pt` | as X1 | done 2026-11-27 (kept) |

Shared code: `scripts/mi_qwen_students_common.py` (models incl. rd_zero / rd_low_rr, traces, step boundaries, P_R, hooks).
Inputs reused: step-labelled traces `eval_outputs/rd_newsae/eval_mod/bonds/steps.json`, benchmark traces listed in
`scripts/eval_reflection_benchmarks.py`, probe texts / targets `eval_outputs/rd_newsae/student_probe/`, `checkpoints/rd_newsae_probe/targets.pt`.

## Files created or changed in this work (the directory is not under git here)
- New scripts: `mi_qwen_crosscoder.py`, `mi_qwen_patching.py`, `mi_qwen_diffs.py`,
  `mi_qwen_students_common.py`, `mi_qwen_alpha0.py`, `mi_qwen_alpha0_props.py`, `mi_qwen_alpha_causal.py`,
  `mi_qwen_alpha_mech.py`, `mi_qwen_alpha_ext.py`, `run_mi_qwen_students.sh`, `run_mi_qwen_alpha0_props.sh`, `run_mi_qwen_alpha_causal.sh`,
  `run_mi_qwen_alpha_mech.sh`, `run_mi_qwen_alpha_ext.sh`.
- Changed: `scripts/mi_qwen_student_probe.py` (adds rd_zero, rd_lowrerun to the probed models).
- New configs: `configs/train/rd_newsae_zero_vast.yaml`, `configs/train/rd_newsae_lowrerun_vast.yaml`.
- Docs: `docs/mi_qwen_students.md` (pre-registration, results, alpha = 0 section), this tracker.
- Large, not for git: `checkpoints/mi_qwen_crosscoder/` (3.2 GB), `checkpoints/rd_newsae_{zero,lowrerun}/` (3.8 GB each),
  `checkpoints/rd_newsae_probe/probe_{rd_zero,graft_*,x2_*,x5*}.pt` (75 MB each), `checkpoints/rd_newsae_probe/ext/` (X5 teacher
  targets, ~1 GB).

## Outside the repo
- /home/cs/Desktop/KD/archive_mi_final_2026-11-28/: removed from the repo 2026-11-28 at the user's request (final MI
  selection): Exp 2 (reflection direction d_refl, teacher-forced steering, steered generation; script
  mi_qwen_reflection_steering.py, steering/ outputs, configs/mech_interp/steer_gen_*.yaml), Exp 3 activation patching
  rd_mod <-> lkd (patch.jsonl), Exp 4 activations and single-layer weight swap (acts.json, swap.json), C3 steering along
  d_term (steer.json), M3 logit lens (lens.jsonl); full scripts, launchers, analyses (*_full.txt), properties.txt,
  logs, doc and tracker before removal.
- /home/cs/Desktop/KD/archive_mi_ext_2026-11-27/: X3 (CKA with the teacher) and X6 (probe by token class), removed
  from the repo 2026-11-27 at the user's request: x3_cka.jsonl, X6 teacher targets and stats (ext/x6_*.pt), the full
  script and launcher, full analysis (analysis_full.txt), run log, doc and tracker before removal.
- /home/cs/Desktop/KD/archive_mi_mechanism_2026-11-26/: M2 (mediation), M4 (latent-restricted patching + attribution),
  M5 (feature-level steering), removed from the repo 2026-11-26 at the user's request: full script and launcher, full
  analysis (analysis_full.txt), latpatch.jsonl, attr_latent.pt, featsteer.json, run log, doc before removal.
- Vast instance 62314075: destroyed 2026-11-24 (no instances left on the account). Run bundle and helper scripts in the
  Claude session scratchpad (temporary).
- OneDrive `RD_newSAE_checkpoints/rd_newsae_{zero,lowrerun}/`: step_47 and step_94 of both runs, sha256.txt.
