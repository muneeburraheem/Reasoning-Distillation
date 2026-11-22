# configs/

Every config here was used for a reported run, exactly as run (only file names, folders and repo-relative paths
changed on 2026-11-22; old names in `docs/REPO_MAP.md`). Training: `python scripts/train.py --config configs/train/<yaml>`.
Evaluation: `python -m custom_eval.run_eval --config configs/eval/<yaml>` from the repo root.

## train/
| Config | Run |
|---|---|
| `qwen_logit_kd.yaml`, `qwen_logit_kd_continue.yaml` | LKD: 40k rows, then 12,032 more (= LKD-continue, the baseline) |
| `qwen_reasondistill_final.yaml`, `qwen_reasondistill_final_continue.yaml` | RD two-stage scheme, as used by October RD (trained outside this repo, RUN_LOG 8l) |
| `qwen_sae.yaml` | the original Qwen SAE (the one that turned out dead) |
| `sae_norm64_full.yaml`, `sae_norm64_pilot_*.yaml` | the retrained SAE and its five 10M-token pilots (`scripts/train_sae_norm64.py`) |
| `rd_newsae_{low,moderate,high}.yaml` | new-SAE RD phase 2, alpha 0.05 / 0.15 / 0.40 (code: commit a25867d, see docs/rd_newsae.md); `_smoke` = local smoke test |

## eval/
| Prefix | Models |
|---|---|
| `qwen_*` | base Qwen3.5-2B and LKD-continue (`qwen_math_basic`, `qwen_math500_32k`, `qwen_passk` also list the first RD run, which is not reported) |
| `rd_mod_*`, `rd_{low,mod,high}_math500_32k` | new-SAE RD (local checkpoints in `checkpoints/rd_newsae_*`) |
| `lkd_seed43_math500_32k` | LKD-continue resampled with seed 43 (sampling-noise check) |
| `rd_october_*` | October RD |
| `teacher_*` | the teacher, Qwen3.5-4B |
Suffixes: `math_basic` = MATH-500 @4096 + AIME-25 @32768; `math500_32k`; `passk` = 8 samples @8192; `hard_A` = OlympiadBench,
`hard_B` = AIME-24 / HMMT-25 / BRUMO-25 / CMIMC-25, `hard_C` = Omni-MATH hard; `september_{think,no_think}` = GSM8K, MMLU,
ARC-C, HellaSwag; `processbench`. The long-context `hard_*` runs (and the new-SAE `processbench` run) used `venv_fla` (see the
root README).

## mech_interp/
`causal_gen_{lkd,rd_mod}.yaml`: generation settings for the student causal test (`scripts/mi_qwen_student_causal.py gen`).
