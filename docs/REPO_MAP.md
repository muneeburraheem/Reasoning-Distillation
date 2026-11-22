# Repo map: old names -> current names, and what was archived

The repo was reorganised twice on 2026-11-22: the new-SAE side experiment (`side_experiments/qwen_sae_l16_norm64/`)
was folded into the repo layout, then scripts and configs were renamed to one scheme and everything not reported in
the paper was archived next to the repo (`../archive_paper_cleanup_2026-11-22/`, same relative paths; nothing was
deleted). `docs/RUN_LOG.md` is the chronological lab log: it uses current names where a file still exists and keeps
the old names of archived files. Machine-readable lists: `docs/repo_history/*.json`.

## Naming scheme

| Prefix | Role |
|---|---|
| `train_*` | training (plus the upstream `train.py`, `train_sae.py`, `reason_score.py`, `train_logit_kd.sh`) |
| `eval_*` | generation or measurement that writes raw outputs to `eval_outputs/` |
| `sae_*` | SAE health / gate / ReasonScore checks (`sae_norm64_*` new SAE, `sae_old_*` the original SAE) |
| `mi_*` | mechanistic interpretability (teacher steering/patching, Llama and Qwen students) |
| `analyze_*` | reads raw outputs, prints the summary a reported number comes from |
| `run_*.sh` | launchers; `run_all_analyses.sh` regenerates every reported summary |

Configs: `configs/train/` (training), `configs/eval/` (evaluation; `qwen_*` = base/LKD runs, `rd_october_*`,
`rd_mod_*` / `rd_{low,mod,high}_*` = new-SAE RD, `teacher_*`), `configs/mech_interp/`.

## Renamed or moved code, configs and docs (old -> current)

| Old | Current |
|---|---|
| `100samples_rd.md` | archived (`docs/notes/100samples_rd.md`) |
| `100samples_rd_nontruncated.md` | archived (`docs/notes/100samples_rd_nontruncated.md`) |
| `base_zero_shot_summary.md` | archived (`docs/notes/base_zero_shot_summary.md`) |
| `base_zero_shot_traces.md` | archived (`docs/notes/base_zero_shot_traces.md`) |
| `configs/hard_A.yaml` | `configs/eval/qwen_hard_A.yaml` |
| `configs/hard_B.yaml` | `configs/eval/qwen_hard_B.yaml` |
| `configs/hard_C.yaml` | `configs/eval/qwen_hard_C.yaml` |
| `configs/processbench.yaml` | `configs/eval/qwen_processbench.yaml` |
| `configs/qwen_logit_kd.yaml` | `configs/train/qwen_logit_kd.yaml` |
| `configs/qwen_logit_kd_continue.yaml` | `configs/train/qwen_logit_kd_continue.yaml` |
| `configs/qwen_reasondistill_final.yaml` | `configs/train/qwen_reasondistill_final.yaml` |
| `configs/qwen_reasondistill_final_continue.yaml` | `configs/train/qwen_reasondistill_final_continue.yaml` |
| `configs/qwen_sae.yaml` | `configs/train/qwen_sae.yaml` |
| `configs/rd_newsae/causal_gen_lkd.yaml` | `configs/mech_interp/causal_gen_lkd.yaml` |
| `configs/rd_newsae/causal_gen_rd_moderate.yaml` | `configs/mech_interp/causal_gen_rd_mod.yaml` |
| `configs/rd_newsae/full_l1_003.yaml` | `configs/train/sae_norm64_full.yaml` |
| `configs/rd_newsae/lkd_seed43_math500_32k.yaml` | `configs/eval/lkd_seed43_math500_32k.yaml` |
| `configs/rd_newsae/math500_32k_rd_newsae_high.yaml` | `configs/eval/rd_high_math500_32k.yaml` |
| `configs/rd_newsae/math500_32k_rd_newsae_low.yaml` | `configs/eval/rd_low_math500_32k.yaml` |
| `configs/rd_newsae/math500_32k_rd_newsae_moderate.yaml` | `configs/eval/rd_mod_math500_32k.yaml` |
| `configs/rd_newsae/mod_basic_math.yaml` | `configs/eval/rd_mod_math_basic.yaml` |
| `configs/rd_newsae/mod_hard_A.yaml` | `configs/eval/rd_mod_hard_A.yaml` |
| `configs/rd_newsae/mod_hard_B.yaml` | `configs/eval/rd_mod_hard_B.yaml` |
| `configs/rd_newsae/mod_hard_C.yaml` | `configs/eval/rd_mod_hard_C.yaml` |
| `configs/rd_newsae/mod_passk.yaml` | `configs/eval/rd_mod_passk.yaml` |
| `configs/rd_newsae/mod_processbench.yaml` | `configs/eval/rd_mod_processbench.yaml` |
| `configs/rd_newsae/mod_september_no_think.yaml` | `configs/eval/rd_mod_september_no_think.yaml` |
| `configs/rd_newsae/mod_september_think.yaml` | `configs/eval/rd_mod_september_think.yaml` |
| `configs/rd_newsae/pilot_norm_l1_0005.yaml` | `configs/train/sae_norm64_pilot_norm_l1_0005.yaml` |
| `configs/rd_newsae/pilot_norm_l1_0015.yaml` | `configs/train/sae_norm64_pilot_norm_l1_0015.yaml` |
| `configs/rd_newsae/pilot_norm_l1_005.yaml` | `configs/train/sae_norm64_pilot_norm_l1_005.yaml` |
| `configs/rd_newsae/pilot_norm_l1_5.yaml` | `configs/train/sae_norm64_pilot_norm_l1_5.yaml` |
| `configs/rd_newsae/pilot_raw_l1_0015.yaml` | `configs/train/sae_norm64_pilot_raw_l1_0015.yaml` |
| `configs/rd_newsae/rd_newsae_high.yaml` | `configs/train/rd_newsae_high.yaml` |
| `configs/rd_newsae/rd_newsae_low.yaml` | `configs/train/rd_newsae_low.yaml` |
| `configs/rd_newsae/rd_newsae_moderate.yaml` | `configs/train/rd_newsae_moderate.yaml` |
| `configs/rd_newsae/rd_newsae_smoke.yaml` | `configs/train/rd_newsae_smoke.yaml` |
| `configs/september_rd_october_no_think.yaml` | `configs/eval/rd_october_september_no_think.yaml` |
| `configs/september_rd_october_think.yaml` | `configs/eval/rd_october_september_think.yaml` |
| `configs/sweep_pass_k.yaml` | `configs/eval/qwen_passk.yaml` |
| `configs/sweep_pass_k_rd_october.yaml` | `configs/eval/rd_october_passk.yaml` |
| `configs/sweep_pass_k_rd_october_math500.yaml` | `configs/eval/rd_october_passk_math500.yaml` |
| `configs/teacher_qwen4b_math500.yaml` | `configs/eval/teacher_math500.yaml` |
| `configs/teacher_qwen4b_math500_32k.yaml` | `configs/eval/teacher_math500_32k.yaml` |
| `configs/test_math.yaml` | `configs/eval/qwen_math_basic.yaml` |
| `configs/test_math500_32k.yaml` | `configs/eval/qwen_math500_32k.yaml` |
| `configs/test_math500_32k_rd_october.yaml` | `configs/eval/rd_october_math500_32k.yaml` |
| `configs/test_math_rd_october.yaml` | `configs/eval/rd_october_math_basic.yaml` |
| `docs/notes/restructure_2026-11-22_moves.json` | `docs/repo_history/restructure_2026-11-22_moves.json` |
| `prompt_experiment_results.md` | archived (`docs/notes/prompt_experiment_results.md`) |
| `prompt_statistics_nontruncated.md` | archived (`docs/notes/prompt_statistics_nontruncated.md`) |
| `prompt_statistics_summary.md` | archived (`docs/notes/prompt_statistics_summary.md`) |
| `run_base_model_raw_behavior.py` | archived (`scripts/run_base_model_raw_behavior.py`) |
| `run_prefill_experiments.py` | archived (`scripts/run_prefill_experiments.py`) |
| `scripts/analyze_llama_reasoning.py` | `scripts/analyze_llama_processbench.py` |
| `scripts/analyze_mi_students_llama.py` | `scripts/analyze_mi_llama_students.py` |
| `scripts/answer_entropy.py` | `scripts/eval_answer_entropy.py` |
| `scripts/bond_annotate.py` | `scripts/eval_bond_annotate.py` |
| `scripts/budget_forcing.py` | `scripts/eval_budget_forcing.py` |
| `scripts/exp2_feature_breakdown.py` | `scripts/mi_llama_exp2_feature_breakdown.py` |
| `scripts/exp3_feature_function.py` | `scripts/mi_llama_exp3_feature_function.py` |
| `scripts/exp4_student_attribution.py` | `scripts/mi_llama_exp4_student_attribution.py` |
| `scripts/exp5_cka.py` | `scripts/mi_llama_exp5_cka.py` |
| `scripts/llama_bonds.py` | `scripts/eval_llama_bonds.py` |
| `scripts/llama_budget_forcing.py` | `scripts/eval_llama_budget_forcing.py` |
| `scripts/llama_passk.py` | `scripts/eval_llama_passk.py` |
| `scripts/llama_reasoning_gen.py` | `scripts/eval_llama_processbench.py` |
| `scripts/llama_sweep_audit.py` | `scripts/mi_llama_sweep_audit.py` |
| `scripts/llama_teacher_forced_kl.py` | `scripts/eval_llama_teacher_forced_kl.py` |
| `scripts/mi_students_llama.py` | `scripts/mi_llama_students.py` |
| `scripts/qwen_aligner_check.py` | `scripts/sae_old_aligner_check.py` |
| `scripts/qwen_sae_signal.py` | `scripts/sae_old_signal.py` |
| `scripts/run_exp3_6.sh` | `scripts/run_mi_llama_exp3_5.sh` |
| `scripts/teacher_forced_kl.py` | `scripts/eval_teacher_forced_kl.py` |
| `september-eval-scripts/README.md` | `docs/september_eval.md` |
| `september-eval-scripts/gsm8k_generation_length.py` | `scripts/eval_gsm8k_generation_length.py` |
| `september-eval-scripts/qwen_2B-no_think.yaml` | `configs/eval/qwen_september_no_think.yaml` |
| `september-eval-scripts/qwen_2B-think.yaml` | `configs/eval/qwen_september_think.yaml` |
| `september-eval-scripts/run.sh` | `scripts/run_september_eval.sh` |
| `side_experiments/qwen_sae_l16_norm64/README.md` | `docs/rd_newsae.md` |
| `side_experiments/qwen_sae_l16_norm64/analyze_teacher_mi.py` | `scripts/analyze_mi_qwen_teacher_newsae.py` |
| `side_experiments/qwen_sae_l16_norm64/configs` | split into `configs/train/`, `configs/eval/`, `configs/mech_interp/` (rows below) |
| `side_experiments/qwen_sae_l16_norm64/eval_sae.py` | `scripts/sae_norm64_health.py` |
| `side_experiments/qwen_sae_l16_norm64/gate.py` | `scripts/sae_norm64_gate.py` |
| `side_experiments/qwen_sae_l16_norm64/gate_perfeature.py` | `scripts/sae_norm64_gate_perfeature.py` |
| `side_experiments/qwen_sae_l16_norm64/head_check.py` | `scripts/analyze_rd_newsae_head.py` |
| `side_experiments/qwen_sae_l16_norm64/pretrain_head.py` | `scripts/train_rd_newsae_head.py` |
| `side_experiments/qwen_sae_l16_norm64/rd_smoke.py` | `scripts/train_rd_newsae_smoke.py` |
| `side_experiments/qwen_sae_l16_norm64/reflection_coupling.py` | `scripts/analyze_reflection_coupling.py` |
| `side_experiments/qwen_sae_l16_norm64/reflection_more.py` | `scripts/eval_reflection_benchmarks.py` |
| `side_experiments/qwen_sae_l16_norm64/run_after_suite.sh` | `scripts/run_rd_newsae_after_suite.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_full.sh` | `scripts/run_sae_norm64_full.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_gate_after_rs.sh` | `scripts/run_sae_norm64_gate.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_math500_newsae.sh` | `scripts/run_rd_newsae_math500.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_math500_newsae_low.sh` | `scripts/run_rd_newsae_math500_low.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_mod_characteristics.sh` | `scripts/run_rd_newsae_characteristics.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_mod_suite.sh` | `scripts/run_rd_newsae_suite.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_pilots.sh` | `scripts/run_sae_norm64_pilots.sh` |
| `side_experiments/qwen_sae_l16_norm64/run_reason_score.py` | `scripts/sae_norm64_reason_score.py` |
| `side_experiments/qwen_sae_l16_norm64/run_teacher_checks.sh` | `scripts/run_sae_norm64_teacher_checks.sh` |
| `side_experiments/qwen_sae_l16_norm64/student_causal.py` | `scripts/mi_qwen_student_causal.py` |
| `side_experiments/qwen_sae_l16_norm64/student_probe.py` | `scripts/mi_qwen_student_probe.py` |
| `side_experiments/qwen_sae_l16_norm64/teacher_mi.py` | `scripts/mi_qwen_teacher_newsae.py` |
| `side_experiments/qwen_sae_l16_norm64/train.py` | `scripts/train_sae_norm64.py` |
| `summarize_nontruncated.py` | archived (`scripts/summarize_nontruncated.py`) |
| `test_base.py` | archived (`scripts/test_base.py`) |

## Side-experiment results and weights moved into the repo layout (first pass)

| Old | Current |
|---|---|
| `Logit KD vs ReasonDistill temp.pdf` | archived (`docs/notes/Logit KD vs ReasonDistill temp.pdf`) |
| `base_model_raw_behavior.log` | archived (`logs/base_model_raw_behavior.log`) |
| `logitkd_eval.log` | `logs/logitkd_eval.log` |
| `math500_32k.log` | `logs/math500_32k.log` |
| `pass@k.pdf` | archived (`docs/notes/pass@k.pdf`) |
| `prompt_experiment_results_distilled_only.jsonl` | archived (`docs/notes/prompt_experiment_results_distilled_only.jsonl`) |
| `september-evaluation-analysis.pdf` | archived (`docs/notes/september-evaluation-analysis.pdf`) |
| `side_experiments/qwen_sae_l16_norm64/repo_nov5/logs/rd_newsae_smoke` | `logs/rd_newsae_smoke` |
| `side_experiments/qwen_sae_l16_norm64/runs/checkpoint_evals_manual.txt` | `eval_outputs/rd_newsae/sae/checkpoint_evals_manual.txt` |
| `side_experiments/qwen_sae_l16_norm64/runs/eval.txt` | `eval_outputs/rd_newsae/sae/eval.txt` |
| `side_experiments/qwen_sae_l16_norm64/runs/eval_hidden.pt` | `checkpoints/qwen_sae_norm64/eval_hidden.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/full.log` | `logs/rd_newsae/full.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/full_l1_003` | `checkpoints/qwen_sae_norm64/full_l1_003` |
| `side_experiments/qwen_sae_l16_norm64/runs/full_l1_003.log` | `logs/rd_newsae/full_l1_003.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/gate.log` | `logs/rd_newsae/gate.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/gate.txt` | `eval_outputs/rd_newsae/sae/gate.txt` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_0005` | `checkpoints/qwen_sae_norm64/pilot_norm_l1_0005` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_0005.log` | `logs/rd_newsae/pilot_norm_l1_0005.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_0015` | `checkpoints/qwen_sae_norm64/pilot_norm_l1_0015` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_0015.log` | `logs/rd_newsae/pilot_norm_l1_0015.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_005` | `checkpoints/qwen_sae_norm64/pilot_norm_l1_005` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_005.log` | `logs/rd_newsae/pilot_norm_l1_005.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_5` | `checkpoints/qwen_sae_norm64/pilot_norm_l1_5` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_norm_l1_5.log` | `logs/rd_newsae/pilot_norm_l1_5.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_raw_l1_0015` | `checkpoints/qwen_sae_norm64/pilot_raw_l1_0015` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilot_raw_l1_0015.log` | `logs/rd_newsae/pilot_raw_l1_0015.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/pilots.log` | `logs/rd_newsae/pilots.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/eval` | `eval_outputs/rd_newsae/eval` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/eval_mod` | `eval_outputs/rd_newsae/eval_mod` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/head_check.txt` | `eval_outputs/rd_newsae/head_check.txt` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/pretrain_head.log` | `logs/rd_newsae/pretrain_head.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/pretrain_head.out` | `eval_outputs/rd_newsae/pretrain_head.out` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/reasonscore.pt` | `checkpoints/rd_newsae_sae/reasonscore.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/sae.pt` | `checkpoints/rd_newsae_sae/sae.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/smoke1.log` | `logs/rd_newsae/smoke1.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/smoke2.log` | `logs/rd_newsae/smoke2.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/student_init` | `checkpoints/rd_newsae_student_init` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/checkpoints/rd_newsae_high` | `checkpoints/rd_newsae_high` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/checkpoints/rd_newsae_low` | `checkpoints/rd_newsae_low` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/checkpoints/rd_newsae_moderate` | `checkpoints/rd_newsae_moderate` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/logs/rd_newsae_high` | `logs/rd_newsae_high` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/logs/rd_newsae_low` | `logs/rd_newsae_low` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/logs/rd_newsae_moderate` | `logs/rd_newsae_moderate` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/probe.log` | `logs/rd_newsae/vast_probe.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/probe_analysis.txt` | `eval_outputs/rd_newsae/vast_probe_analysis.txt` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/pull.log` | `logs/rd_newsae/vast_pull.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/rd_warmstart/vast/runlogs` | `logs/rd_newsae/vast_runlogs` |
| `side_experiments/qwen_sae_l16_norm64/runs/reason_score.log` | `logs/rd_newsae/reason_score.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/reflection_more` | `eval_outputs/rd_newsae/reflection_more` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_causal` | `eval_outputs/rd_newsae/student_causal` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe` | `eval_outputs/rd_newsae/student_probe` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_base.pt` | `checkpoints/rd_newsae_probe/probe_base.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_lkd.pt` | `checkpoints/rd_newsae_probe/probe_lkd.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_lkd_phase1.pt` | `checkpoints/rd_newsae_probe/probe_lkd_phase1.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_high.pt` | `checkpoints/rd_newsae_probe/probe_rd_high.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_high_h.pt` | `checkpoints/rd_newsae_probe/probe_rd_high_h.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_low.pt` | `checkpoints/rd_newsae_probe/probe_rd_low.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_low_h.pt` | `checkpoints/rd_newsae_probe/probe_rd_low_h.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_moderate.pt` | `checkpoints/rd_newsae_probe/probe_rd_moderate.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_moderate_h.pt` | `checkpoints/rd_newsae_probe/probe_rd_moderate_h.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_rd_oct.pt` | `checkpoints/rd_newsae_probe/probe_rd_oct.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/probe_teacher_ceiling.pt` | `checkpoints/rd_newsae_probe/probe_teacher_ceiling.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/student_probe/targets.pt` | `checkpoints/rd_newsae_probe/targets.pt` |
| `side_experiments/qwen_sae_l16_norm64/runs/teacher_checks.log` | `logs/rd_newsae/teacher_checks.log` |
| `side_experiments/qwen_sae_l16_norm64/runs/teacher_mi` | `eval_outputs/rd_newsae/teacher_mi` |
| `sweep_alpha_8k.log` | archived (`logs/sweep_alpha_8k.log`) |
| `sweep_pass_k.log` | `logs/sweep_pass_k.log` |
| `test_math_full.log` | `logs/test_math_full.log` |
| `train_a000.log` | archived (`logs/train_a000.log`) |
| `train_a005.log` | archived (`logs/train_a005.log`) |
| `train_a010.log` | archived (`logs/train_a010.log`) |
| `train_a050.log` | archived (`logs/train_a050.log`) |

## Archived (moved to `../archive_paper_cleanup_2026-11-22/`, not in the paper)

### September RD: the first RD run (`reasondistill_final`, RUN_LOG 2-8i) and its analyses. The October RD and new-SAE RD analyses do not read these files.

- `configs/qwen_reasondistill.yaml`
- `configs/qwen_reasondistill_new.yaml`
- `configs/qwen_reasondistill_resume.yaml`
- `eval_outputs/answer_entropy`
- `eval_outputs/answer_entropy_v1_lastbox`
- `eval_outputs/bonds`
- `eval_outputs/budget_forcing/rd__B1024.json`
- `eval_outputs/budget_forcing/rd__B2048.json`
- `eval_outputs/budget_forcing/rd__B4096.json`
- `eval_outputs/budget_forcing/rd__B512.json`
- `eval_outputs/budget_forcing/summary.txt`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__aime25.json`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__arc-c.json`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__gsm8k.json`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__hellaswag.json`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__math500.json`
- `eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-reasondistill-final-continue-epoch_1-think__mmlu.json`
- `eval_outputs/teacher_forced`
- `eval_outputs/test_math/20260925T181224Z-112470/qwen-reasondistill-final-continue-epoch_1-think__aime25.json`
- `eval_outputs/test_math/20260925T181224Z-112470/qwen-reasondistill-final-continue-epoch_1-think__math500.json`
- `eval_outputs/test_math/math500_32k/qwen-reasondistill-final-continue-epoch_1-think__math500.json`

### Prompt / prefill experiments, base zero-shot traces, draft PDFs.

- `docs/notes/100samples_rd.md`
- `docs/notes/100samples_rd_nontruncated.md`
- `docs/notes/Logit KD vs ReasonDistill temp.pdf`
- `docs/notes/base_zero_shot_summary.md`
- `docs/notes/base_zero_shot_traces.md`
- `docs/notes/pass@k.pdf`
- `docs/notes/prompt_experiment_results.md`
- `docs/notes/prompt_experiment_results_distilled_only.jsonl`
- `docs/notes/prompt_statistics_nontruncated.md`
- `docs/notes/prompt_statistics_summary.md`
- `docs/notes/september-evaluation-analysis.pdf`
- `logs/base_model_raw_behavior.log`
- `scripts/run_base_model_raw_behavior.py`
- `scripts/run_prefill_experiments.py`
- `scripts/summarize_nontruncated.py`
- `scripts/test_base.py`

### Old artifact exporters, superseded queue scripts, examples, smoke runs.

- `eval_outputs/llama_smoke/budget_forcing`
- `eval_outputs/qwen_sae_check`
- `eval_outputs/smoke_8q`
- `examples`
- `final-run-evals`
- `scripts/analyze_like_september.py`
- `scripts/export_cot_artifact_data.py`
- `scripts/export_eval_artifact_data.py`
- `scripts/export_teacher_artifact_data.py`
- `scripts/recompute_self_entropy.py`
- `scripts/run_bonds_then_bf.sh`
- `scripts/run_rd_october_basic.sh`
- `scripts/run_rd_october_queue.sh`
- `scripts/run_tf_then_teacher32k.sh`
- `september-eval-scripts/qwen_kd_comparison-think.yaml`
- `september-eval-scripts/run_kd_comparison.sh`

### Alpha sweep (RUN_LOG 6, 6b), not in the paper.

- `configs/qwen_reasondistill_sweep_a000.yaml`
- `configs/qwen_reasondistill_sweep_a005.yaml`
- `configs/qwen_reasondistill_sweep_a010.yaml`
- `configs/qwen_reasondistill_sweep_a050.yaml`
- `configs/sweep_alpha_eval.yaml`
- `eval_outputs/sweep_alpha_8k`
- `logs/qwen_reasondistill_sweep_a000`
- `logs/qwen_reasondistill_sweep_a005`
- `logs/qwen_reasondistill_sweep_a010`
- `logs/qwen_reasondistill_sweep_a050`
- `logs/sweep_alpha_8k.log`
- `logs/train_a000.log`
- `logs/train_a005.log`
- `logs/train_a010.log`
- `logs/train_a050.log`

### September launcher smoke run (2 questions).

- `eval_outputs/september/20260922T172432Z-1086805`

Also in the archive: `pre_cleanup_code/` = scripts/, configs/, docs/, tests/, custom_eval/ exactly as they were
before the renames (for reading old versions of a script).
