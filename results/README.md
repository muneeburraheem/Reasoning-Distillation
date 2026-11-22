# Results index

Where every reported number lives, what produced it, and how to regenerate it.

- **Tables as reported:** `paper_tables.md` (readable) and `paper_tables.json` (the same, machine-readable).
- **Summary files:** the `analysis*.txt` / `summary*.txt` files under `eval_outputs/`. They are tracked in git; the raw
  outputs they read (per-question JSON, about 6 GB) are not, and stay on the lab machine / shared drive.
- **Regenerate every summary** from the raw outputs (CPU, ~6 min): `PY=/path/to/python bash scripts/run_all_analyses.sh`.
  It writes 38 files. On 2026-11-22 the 36 that already existed came out byte-identical to the stored ones (library
  warning lines aside); the two teacher-accuracy summaries are new. Files it does not write are listed at the end.
- **Context:** `docs/RUN_LOG.md` (chronological lab log, section numbers below), `docs/rd_newsae.md` (new SAE and
  new-SAE RD, with the pre-registrations), `docs/REPO_MAP.md` (old -> new names; what was archived).

Models. Teacher Qwen3.5-4B; students Qwen3.5-2B. **LKD** = logit KD (40k rows, then 12,032 more = "LKD-continue").
**RD** = LKD + SAE-feature alignment. **New-SAE RD** (the RD of the paper) = warm-started from LKD epoch 1, 12,032
rows with the retrained SAE, alpha 0.05 / 0.15 / 0.40 = `rd_low` / `rd_mod` (main) / `rd_high`. **October RD**
(`RD_oct`, `RD_new` in older scripts) = earlier RD run whose SAE targets were empty; text only in the paper.
Checkpoints: LKD and October RD on the Hub (`Khaquan/qwen-khaquanS-distillations`); new-SAE RD and the new SAE
locally in `checkpoints/` (see the root README).

## 1. Old SAE problems and the fix (RUN_LOG 8g, 8j, 8ad-8af; docs/rd_newsae.md)
| What | Summary file | Produced by |
|---|---|---|
| Old vs new SAE health (FVE, L0, dead, reasoning latents), pilots, full-run checkpoints | `eval_outputs/rd_newsae/sae/eval.txt` | `scripts/sae_norm64_health.py` (GPU; appended per run) |
| Target liveness on RD's exact target path, old vs new | `eval_outputs/rd_newsae/sae/gate.txt` | `scripts/sae_norm64_gate.py` (GPU) |
| Teacher causal check of the new RD feature set (attribution, ablation, generation) | `eval_outputs/rd_newsae/teacher_mi/analysis.txt` | `scripts/analyze_mi_qwen_teacher_newsae.py` |
| Pre-trained RD head carries signal | `eval_outputs/rd_newsae/head_check.txt`, `pretrain_head.out` | `scripts/analyze_rd_newsae_head.py`, `train_rd_newsae_head.py` (GPU) |
Configs: `configs/train/qwen_sae.yaml` (original SAE), `configs/train/sae_norm64_full.yaml`, `sae_norm64_pilot_*.yaml`.

## 2. Student read-out of the teacher's reasoning features (docs/rd_newsae.md "Phase-2 results")
`eval_outputs/rd_newsae/student_probe/analysis.txt` (= `vast_probe_analysis.txt`, the same run's copy) -
`scripts/mi_qwen_student_probe.py --analyze`.

## 3. Accuracy, new-SAE RD vs LKD
| Benchmark | Summary file | Configs (RD / LKD) |
|---|---|---|
| MATH-500 @32k, 1 sample, all three alphas, LKD seeds 42 and 43 | `eval_outputs/rd_newsae/eval/analysis_{low,moderate,high,lkd_seed43}.txt` | `configs/eval/rd_{low,mod,high}_math500_32k.yaml`, `lkd_seed43_math500_32k.yaml` / `qwen_math500_32k.yaml` |
| MATH-500 @4k, AIME-25 @32k | `eval_outputs/rd_newsae/eval_mod/basic_math/analysis.txt` | `rd_mod_math_basic.yaml` / `qwen_math_basic.yaml` |
| pass@8 (MATH-500, AIME-25) and "solved on all 8" | `eval_outputs/rd_newsae/eval_mod/passk/analysis.txt` | `rd_mod_passk.yaml` / `qwen_passk.yaml` |
| OlympiadBench | `eval_outputs/rd_newsae/eval_mod/hard_math/A/analysis.txt` | `rd_mod_hard_A.yaml` / `qwen_hard_A.yaml` |
| AIME-24, HMMT-25, BRUMO-25, CMIMC-25 (4 samples) | `eval_outputs/rd_newsae/eval_mod/hard_math/B/analysis.txt` | `rd_mod_hard_B.yaml` / `qwen_hard_B.yaml` |
| Omni-MATH hard (+ difficulty trend) | `eval_outputs/rd_newsae/eval_mod/hard_math/C/analysis.txt` | `rd_mod_hard_C.yaml` / `qwen_hard_C.yaml` |
| GSM8K, MMLU, ARC-C, HellaSwag (thinking / non-thinking) | `eval_outputs/rd_newsae/eval_mod/september/analysis.txt` | `rd_mod_september_*.yaml` / `qwen_september_*.yaml` |
| ProcessBench (official F1) | `eval_outputs/rd_newsae/eval_mod/processbench/analysis.txt` | `rd_mod_processbench.yaml` / `qwen_processbench.yaml` |
| Budget forcing (512-4096 tokens) | `eval_outputs/rd_newsae/eval_mod/budget_forcing/analysis_vs_lkd.txt` | `scripts/eval_budget_forcing.py` |
All configs are in `configs/eval/`. Each analysis file also has an October RD column or block.

## 4. Reasoning style (RUN_LOG 8n, 8ai; Chen et al. 2026 step labels)
| What | Summary file |
|---|---|
| Step shares (reflection, exploration, ...), switching, step-graph similarity; both-correct control | `eval_outputs/rd_newsae/eval_mod/bonds/analysis.txt` |
| Reflection vs success, one trace per question | `eval_outputs/rd_newsae/eval_mod/bonds/reflection_coupling.txt` |
| Reflection on both-correct questions across 9 benchmarks; reflection vs success with 4 traces (pre-registered) | `eval_outputs/rd_newsae/reflection_more/analysis.txt` |
Produced by `scripts/eval_bond_annotate.py` (labels, GPU) -> `analyze_bonds.py`, `analyze_reflection_coupling.py`,
`eval_reflection_benchmarks.py label|analyze`.

## 5. Closeness to the teacher (RUN_LOG 8f, 8n)
`eval_outputs/rd_newsae/eval_mod/teacher_forced/summary.txt` (KL, entropy, NLL on the teacher's traces;
`scripts/eval_teacher_forced_kl.py` -> `analyze_teacher_forced.py`) and `eval_outputs/rd_newsae/eval_mod/answer_entropy/analysis.txt`
(`scripts/eval_answer_entropy.py` -> `analyze_answer_entropy.py`).

## 6. Mechanism: do the students use the teacher-feature directions? (RUN_LOG 8ah, pre-registered)
`eval_outputs/rd_newsae/student_causal/analysis.txt` - `scripts/mi_qwen_student_causal.py tf` (GPU) then `analyze`.

## 7. Teacher MI: steering, patching, suppression (RUN_LOG 8w-8z)
`eval_outputs/mi_steering/{llama_teacher,qwen_teacher}/analysis.txt` (`scripts/mi_steering.py` -> `analyze_mi_steering.py [subject]`),
`eval_outputs/mi_ablation/{llama_teacher,qwen_teacher}/analysis.txt` (`scripts/mi_ablation.py` -> `analyze_mi_ablation.py [subject]`).
Earlier partial outputs kept for the record: `mi_steering/llama_teacher/grid_analysis.txt`, `mi_ablation/llama_teacher/patch_analysis.txt`.

## 8. Llama arm (Llama-3.2-1B students of R1-Distill-Llama-8B; RUN_LOG 8k, 8u, 8v, 8aa-8ac)
| What | Summary file | Produced by |
|---|---|---|
| pass@k on GSM8K / MATH-500; outcome split | `eval_outputs/llama_passk/summary.txt`, `outcomes_summary.txt` | `eval_llama_passk.py` -> `analyze_llama_passk.py`, `analyze_llama_outcomes.py` |
| KL to the teacher | `eval_outputs/llama_teacher_forced/summary.txt` | `eval_llama_teacher_forced_kl.py` -> `analyze_llama_teacher_forced.py` |
| CoT bonds; answer entropy | `eval_outputs/llama_bonds/summary.txt`, `answer_entropy_summary.txt` | `eval_llama_bonds.py` -> `analyze_bonds.py`, `analyze_llama_answer_entropy.py` |
| Budget forcing | `eval_outputs/llama_budget_forcing/summary.txt` | `eval_llama_budget_forcing.py` -> `analyze_llama_budget_forcing.py` |
| Feature read-out and causal tests | `eval_outputs/mi_students/llama/analysis.txt` | `mi_llama_students.py` -> `analyze_mi_llama_students.py` |
| Exp 2-5 follow-ups (feature breakdown, function, attribution, CKA) | `eval_outputs/mi_students/llama/exp{2,3,4,5}/analysis.txt` | `mi_llama_exp*.py` (GPU; write their own analysis) |
| ProcessBench smoke (48 items) | `eval_outputs/llama_smoke/processbench/` | `eval_llama_processbench.py` |
The Llama students and teacher SAE are local (set `LLAMA_KD_DIR`, `KD_BASE_DIR`; see the root README).

## 9. Teacher accuracy (Qwen3.5-4B, MATH-500; RUN_LOG 8c, 8e, 8h)
`eval_outputs/teacher_qwen4b/analysis.txt` (8,192 tokens, 500 questions) and `eval_outputs/teacher_qwen4b_32k/analysis.txt`
(32,768 tokens, first 120 questions, with LKD and RD_mod on the same questions). Configs `configs/eval/teacher_math500{,_32k}.yaml`.
Students at 8,192 tokens: section 3 pass@8.

## 10. October RD (text only)
Its rows are in the section 3-5 analyses (`RD_oct`). LKD vs October RD summaries from October, kept for context
(recorded, not regenerated by the script): `eval_outputs/test_math_rd_october/math500_32k/summary.txt`,
`eval_outputs/september_rd_october/summary.txt`, `eval_outputs/sweep_pass_k_rd_october/summary_*.txt`,
`eval_outputs/budget_forcing/summary_rd_oct_by_level.txt`, `eval_outputs/hard_math/{A,C}/summary.txt`; regenerated:
`eval_outputs/budget_forcing/summary_rd_oct.txt`, `eval_outputs/processbench/analysis*.txt`, `eval_outputs/hard_math/B/summary.txt`.
October RD's training ran outside this repo (RUN_LOG 8l); its scheme is `configs/train/qwen_reasondistill_final*.yaml`.

## Files written by GPU runs (recorded, not regenerated by the script)
`eval_outputs/rd_newsae/sae/{eval,gate,checkpoint_evals_manual}.txt`, `eval_outputs/rd_newsae/head_check.txt`,
`eval_outputs/rd_newsae/pretrain_head.out`, `eval_outputs/mi_students/llama/exp*/analysis.txt`,
`eval_outputs/rd_newsae/eval_mod/september/analysis_think.txt` (an earlier, think-only cut of `analysis.txt`).
`docs/results/all_eval_summaries.csv` is an older (September-October) score table that also lists archived runs.
