# scripts/

Flat folder; the prefix gives the role. Run everything from the repo root. `run_all_analyses.sh` regenerates
every reported summary (see `results/README.md`). Old names: `docs/REPO_MAP.md`.

## Training

| Script | What it does |
|---|---|
| `reason_score.py` | ReasonScore ranking of SAE features by how specific they are to reasoning words (upstream). |
| `train.py` | Main trainer (upstream): logit KD, ReasonDistill, FitNets, hard-label; `--config configs/train/<yaml>`. |
| `train_logit_kd.sh` | LKD phase 1 (40k rows) and phase 2 continuation (configs/train/qwen_logit_kd*.yaml). |
| `train_rd_newsae_head.py` | Pre-train RD's feature head on the frozen phase-1 LKD student, before phase 2 (RunPod). |
| `train_rd_newsae_smoke.py` | Local smoke test of the Nov 5 RD warm-start training with the new SAE (4090, before RunPod). |
| `train_sae.py` | SAE training launcher (upstream); the original Qwen SAE (configs/train/qwen_sae.yaml). |
| `train_sae_norm64.py` | New-SAE experiment (docs/rd_newsae.md): retrain the Qwen3.5-4B layer-16 SAE with input normalisation. |

## Generation and measurement (write raw outputs)

| Script | What it does |
|---|---|
| `eval_answer_entropy.py` | Teacher-forced entropy on the pass@8 MATH-500 traces: answer tokens and full trace. |
| `eval_bond_annotate.py` | Label reasoning steps with the bond types of Chen et al. 2026 (arXiv 2601.06002), and embed them. |
| `eval_budget_forcing.py` | Budget forcing on MATH-500: what does each model believe after B thinking tokens? |
| `eval_gsm8k_generation_length.py` | Compare fresh generation lengths on the same first 500 GSM8K test items. |
| `eval_llama_bonds.py` | CoT bond structure (Chen et al. 2026; RUN_LOG 8n families A-E) for the Llama arm. |
| `eval_llama_budget_forcing.py` | Budget forcing (RUN_LOG 8i) for the Llama arm: what does each model believe after B tokens? |
| `eval_llama_passk.py` | pass@k for the Llama arm (Llama-3.2-1B students of DeepSeek-R1-Distill-Llama-8B). |
| `eval_llama_processbench.py` | ProcessBench (RUN_LOG 8s) for the Llama arm. |
| `eval_llama_teacher_forced_kl.py` | Teacher-forced KL for the Llama arm (the RUN_LOG 8f measurement, Llama students). |
| `eval_reflection_benchmarks.py` | Reflection follow-up 2: more traces and more benchmarks (README "Reflection follow-up 2", pre-registered 2026-11-20). |
| `eval_teacher_forced_kl.py` | Teacher-forced comparison of teacher and students on held-out AM-DeepSeek traces. |

## SAE checks

| Script | What it does |
|---|---|
| `sae_norm64_gate.py` | Go/no-go before any RD retraining: are the new SAE's top-512 ReasonScore features live on RD's own training text, read exactly as the RD trainer reads them? |
| `sae_norm64_gate_perfeature.py` | Per-feature version of the gate: enrichment of each target feature on (and just before) reasoning words, by ReasonScore rank. |
| `sae_norm64_health.py` | Health check for SAE checkpoints of the new-SAE experiment (and the old HF SAE as a baseline). |
| `sae_norm64_reason_score.py` | Runs the repo's scripts/reason_score.py unchanged, with the cached-arrow dataset reader from train_sae_norm64.py (the repo loader's offline cache lookup fails here). Same rows, order and text. |
| `sae_old_aligner_check.py` | Does October RD's logged alignment loss match all-zero targets? Rows from its own training range (40,000-52,031), 2048 tokens. Loss = the trainer's ReasoningAlignmentLoss. |
| `sae_old_signal.py` | Old (HF) SAE on RD's training text: how often the top-512 ReasonScore targets are on at RD's target site (RUN_LOG 8ad). |

## Mechanistic interpretability

| Script | What it does |
|---|---|
| `mi_ablation.py` | SAE feature patching and suppression with matched controls (RUN_LOG 8w steps 3-4). |
| `mi_llama_exp2_feature_breakdown.py` | Exp 2 · Feature breakdown: which of the 196 RD features did RD's student gain on, and what are they? |
| `mi_llama_exp3_feature_function.py` | Exp 3 · Feature function transfer: does a feature do the same job in the student as in the teacher? |
| `mi_llama_exp4_student_attribution.py` | Exp 4 · Student attribution overlap: do the students route reasoning-word prediction through the same teacher features as the teacher does? |
| `mi_llama_exp5_cka.py` | Exp 5 · Layer similarity (CKA): is RD's student more teacher-like overall, or only on the RD features? |
| `mi_llama_students.py` | Llama students: do they carry the teacher's ReasonScore SAE features? (representation phase) |
| `mi_llama_sweep_audit.py` | Audit unified_sweep: (1) were prompts encoded space-stripped? (2) fmax per feature, delta norm vs residual norm. |
| `mi_qwen_student_causal.py` | Qwen students: do they USE the teacher's RD-feature directions? (port of Llama 8ab, scripts/mi_llama_students.py) |
| `mi_qwen_student_probe.py` | Check 1 · Headroom: can the existing Qwen students already read the new SAE's RD features? |
| `mi_qwen_teacher_newsae.py` | Checks 2 + 3: does the new SAE's ReasonScore set matter to the Qwen teacher? |
| `mi_steering.py` | SAE feature steering, redone with the defects of unified_sweep fixed (RUN_LOG 8w). |

## Analyses (print the summaries behind reported numbers)

| Script | What it does |
|---|---|
| `analyze_answer_entropy.py` | Family D (RUN_LOG 8n): answer-token entropy, RD vs LKD, self- and cross-scored. |
| `analyze_basic_qwen.py` | Basic Qwen evals (1 sample per question): new RD vs LKD, old RD and base, paired. |
| `analyze_bonds.py` | Families A-C and E of RUN_LOG 8n: CoT bond structure, geometry and entropy dynamics. |
| `analyze_budget_forcing.py` | Summarise scripts/eval_budget_forcing.py: RD vs LKD at each budget, paired per question. |
| `analyze_difficulty_trend.py` | Accuracy difference by Omni-MATH difficulty level, and its linear trend (RUN_LOG 8q; new-SAE suite). |
| `analyze_hard_B.py` | Set B (RUN_LOG 8q): AIME 2024, HMMT Feb 2025, BRUMO 2025, CMIMC 2025; 4 samples, 16k. New RD vs LKD, paired per problem: per-sample accuracy (strict and lenient), pass@4, outcome split; per competition and pooled (bootstrap ... |
| `analyze_llama_answer_entropy.py` | Llama arm: answer-token entropy on GSM8K finished traces, RD - LKD paired per question (RUN_LOG 8u). |
| `analyze_llama_budget_forcing.py` | Analysis of scripts/eval_llama_budget_forcing.py (Llama arm version of RUN_LOG 8i). |
| `analyze_llama_outcomes.py` | Llama pass@8 outputs (RUN_LOG 8k) re-read with the Qwen outcome split (8d) and by MATH level. |
| `analyze_llama_passk.py` | Summarise scripts/eval_llama_passk.py: LKD vs RD on the Llama arm, paired per question. |
| `analyze_llama_processbench.py` | Analysis of the Llama-arm ProcessBench run (scripts/eval_llama_processbench.py). |
| `analyze_llama_teacher_forced.py` | Analysis of scripts/eval_llama_teacher_forced_kl.py (Llama arm version of RUN_LOG 8f). |
| `analyze_mi_ablation.py` | Analysis of scripts/mi_ablation.py. |
| `analyze_mi_llama_students.py` | Analysis of scripts/mi_llama_students.py (probe phase). |
| `analyze_mi_qwen_teacher_newsae.py` | Analysis of mi_qwen_teacher_newsae.py: the repo's analyze_mi_ablation (patch + sfull) pointed at this folder, plus reflection in the sfull generations (reasoning-marker words per 1k words of thinking, paired per question vs ... |
| `analyze_mi_steering.py` | Analysis of scripts/mi_steering.py (feature steering with matched random controls). |
| `analyze_processbench.py` | ProcessBench (RUN_LOG 8s): LKD vs October RD, earliest-error identification. |
| `analyze_qwen_passk.py` | Qwen pass@8 (MATH-500, AIME-25): new RD vs LKD and old RD, paired per question. |
| `analyze_rd_newsae_head.py` | Does the pre-trained RD head carry signal, or did it settle on "everything off" (October's failure)? |
| `analyze_reflection_coupling.py` | Is RD's extra reflection linked to success? Follow-up to bonds Family E (RUN_LOG 8n), RD moderate suite. |
| `analyze_september_qwen.py` | September benchmark set (GSM8K, MMLU, ARC-C, HellaSwag): new RD vs LKD and base. |
| `analyze_teacher_forced.py` | Summarise scripts/eval_teacher_forced_kl.py output: per-region means and paired tests. |

## Launchers

| Script | What it does |
|---|---|
| `run_all_analyses.sh` | Regenerates every summary behind a number reported in the paper, from the raw outputs in eval_outputs/. CPU only, about 6 minutes. Each summary is written to its usual place (results/README.md lists them by |
| `run_hard_math.sh` | Harder benchmarks A -> B -> C (RUN_LOG 8q), in venv_fla. |
| `run_mi_llama_exp3_5.sh` | Chains Exp 5 -> 3 -> 4 (Llama students, RUN_LOG 8ac). Each script resumes from its own outputs. |
| `run_mi_qwen_teacher.sh` | Qwen teacher mech-interp suite: mi_steering (calib grid full) then mi_ablation (patch sgrid sfull). Both scripts resume from their outputs; a phase that crashes is retried up to 3 times. |
| `run_rd_newsae_after_suite.sh` | After run_rd_newsae_suite.sh finishes: LKD-continue MATH-500 @32k with seed 43 (sampling-noise check). |
| `run_rd_newsae_characteristics.sh` | Characteristics for RD moderate (after its pass@8): step labels + embeddings, then answer-token entropy, for LKD / October RD / RD moderate together (same scripts and settings as 8n and the v2 entropy rerun). |
| `run_rd_newsae_math500.sh` | MATH-500 @32k for new-SAE RD high then moderate (run from repo root). One log per level. |
| `run_rd_newsae_math500_low.sh` | Queued after run_rd_newsae_math500.sh: waits for the moderate eval to end, then MATH-500 @32k for low. |
| `run_rd_newsae_suite.sh` | Full October-RD eval suite for new-SAE RD moderate (final_step_94), same settings as each original run. LKD / October RD outputs are reused for pairing. One log per stage in eval_outputs/rd_newsae/eval_mod/logs; |
| `run_sae_norm64_full.sh` | Full run (weights-only saves) + health check of each checkpoint as it appears. Log: logs/rd_newsae/full.log |
| `run_sae_norm64_gate.sh` | Waits for the new SAE's ReasonScore ranking, then runs sae_norm64_gate.py. |
| `run_sae_norm64_pilots.sh` | 5 pilots x 10M tokens: train (weights only) -> health check -> delete weights (disk). Log: logs/rd_newsae/pilots.log |
| `run_sae_norm64_teacher_checks.sh` | Checks 1-3 in GPU order: calib -> patch/attribution (checks 2a, 3) -> student probe (check 1) -> sfull (check 2b). Each step resumes from its outputs; retried up to 3 times. |
| `run_september_eval.sh` | GSM8K / MMLU / ARC-C / HellaSwag for base and LKD, thinking and non-thinking (docs/september_eval.md). |
