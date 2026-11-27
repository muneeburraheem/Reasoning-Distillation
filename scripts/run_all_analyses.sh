#!/bin/bash
# Regenerates every summary behind a number reported in the paper, from the raw outputs in eval_outputs/.
# CPU only, about 6 minutes. Each summary is written to its usual place (results/README.md lists them by
# paper section). It writes 46 files; on 2026-11-22 the 36 that already existed came out identical to the stored
# ones (warning lines aside) and the two teacher-accuracy summaries were new. The 8 student-MI summaries (added
# 2026-11-28) came out identical on 2026-11-28.
#   bash scripts/run_all_analyses.sh            # PY=/path/to/venv/bin/python to pick the interpreter
cd "$(dirname "$0")/.."
PY=${PY:-python}
nw() { grep -v -i -E "warn|urllib3|chardet"; }      # drop library warnings from the summaries
E=eval_outputs/rd_newsae; M=$E/eval_mod; V=$E/eval
LKD=qwen-logit-kd-continue-epoch_1-think; OCT=qwen-reasondistill-october-continue-epoch_1-think; MOD=qwen-rd-newsae-moderate-final-think
T42=eval_outputs/test_math/math500_32k/${LKD}__math500.json
OCT32=eval_outputs/test_math_rd_october/math500_32k/${OCT}__math500.json
R() { echo "$V/math500_32k/qwen-rd-newsae-$1-final-think__math500.json"; }

## New-SAE RD vs LKD: accuracy
echo "== New-SAE RD vs LKD: accuracy"
# MATH-500 @32k, 1 sample: each alpha vs LKD-continue (seed 42) and October RD; seed-43 LKD rerun
$PY scripts/analyze_basic_qwen.py math500 RD_high=$(R high) LKD=$T42 RD_oct=$OCT32 2>&1 | nw > $V/analysis_high.txt
$PY scripts/analyze_basic_qwen.py math500 RD_mod=$(R moderate) LKD=$T42 RD_high=$(R high) RD_oct=$OCT32 2>&1 | nw > $V/analysis_moderate.txt
$PY scripts/analyze_basic_qwen.py math500 RD_low=$(R low) LKD=$T42 RD_mod=$(R moderate) RD_high=$(R high) RD_oct=$OCT32 2>&1 | nw > $V/analysis_low.txt
$PY scripts/analyze_basic_qwen.py math500 LKD43=$V/lkd_seed43_math500_32k/${LKD}__math500.json LKD42=$T42 RD_low=$(R low) \
    RD_mod=$(R moderate) RD_high=$(R high) RD_oct=$OCT32 2>&1 | nw > $V/analysis_lkd_seed43.txt
# MATH-500 @4096 and AIME-25 @32768, 1 sample
T=eval_outputs/test_math/20260925T181224Z-112470
for b in math500 aime25; do
  $PY scripts/analyze_basic_qwen.py $b RD_mod=$M/basic_math/${MOD}__$b.json LKD=$T/${LKD}__$b.json \
      RD_oct=eval_outputs/test_math_rd_october/${OCT}__$b.json 2>&1 | nw
done > $M/basic_math/analysis.txt
# pass@8 (MATH-500, AIME-25)
P=eval_outputs/sweep_pass_k/20260929T073130Z-1261224
for b in math500 aime25; do
  $PY scripts/analyze_qwen_passk.py $b RD_mod=$M/passk/${MOD}__$b.json LKD=$P/${LKD}__$b.json \
      RD_oct=eval_outputs/sweep_pass_k_rd_october/${OCT}__$b.json 2>&1 | nw
done > $M/passk/analysis.txt
# Harder sets: A = OlympiadBench, B = AIME-24 / HMMT-25 / BRUMO-25 / CMIMC-25 (4 samples), C = Omni-MATH hard
H=eval_outputs/hard_math
{ $PY scripts/analyze_basic_qwen.py olympiadbench LKD=$H/A/${LKD}__olympiadbench.json RD_oct=$H/A/${OCT}__olympiadbench.json \
      RD_mod=$M/hard_math/A/${MOD}__olympiadbench.json 2>&1 | nw
  $PY scripts/analyze_basic_qwen.py olympiadbench RD_mod=$M/hard_math/A/${MOD}__olympiadbench.json RD_oct=$H/A/${OCT}__olympiadbench.json 2>/dev/null
} > $M/hard_math/A/analysis.txt
{ echo "## L=LKD R=RD_mod"; $PY scripts/analyze_hard_B.py RD=$M/hard_math/B/$MOD
  echo; echo "## L=RD_oct R=RD_mod"; $PY scripts/analyze_hard_B.py LKD=$H/B/$OCT RD=$M/hard_math/B/$MOD; } 2>&1 | nw > $M/hard_math/B/analysis.txt
{ $PY scripts/analyze_basic_qwen.py omnimath_hard RD_mod=$M/hard_math/C/${MOD}__omnimath_hard.json LKD=$H/C/${LKD}__omnimath_hard.json \
      RD_oct=$H/C/${OCT}__omnimath_hard.json
  $PY scripts/analyze_difficulty_trend.py RD_mod=$M/hard_math/C/${MOD}__omnimath_hard.json LKD=$H/C/${LKD}__omnimath_hard.json; } 2>&1 | nw > $M/hard_math/C/analysis.txt
# GSM8K, MMLU, ARC-C, HellaSwag (thinking and non-thinking)
RD_MOD_SET=1 $PY scripts/analyze_september_qwen.py 2>&1 | nw > $M/september/analysis.txt
# ProcessBench (official F1)
PB=eval_outputs/processbench
{ echo "## LKD=LKD RD=RD_mod"; $PY scripts/analyze_processbench.py RD=$M/processbench/${MOD}__processbench.json
  echo; echo "## LKD=RD_oct RD=RD_mod"; $PY scripts/analyze_processbench.py LKD=$PB/${OCT}__processbench.json RD=$M/processbench/${MOD}__processbench.json
} 2>&1 | nw > $M/processbench/analysis.txt
# Budget forcing (MATH-500, 500 x 4 samples per budget)
$PY scripts/analyze_budget_forcing.py $M/budget_forcing lkd rd_mod > $M/budget_forcing/analysis_vs_lkd.txt
$PY scripts/analyze_budget_forcing.py $M/budget_forcing rd_oct rd_mod > $M/budget_forcing/analysis_vs_oct.txt

## Reasoning style and closeness to the teacher
echo "== Reasoning style and closeness to the teacher"
$PY scripts/analyze_teacher_forced.py $M/teacher_forced/results.json 2>&1 | nw > $M/teacher_forced/summary.txt
$PY scripts/analyze_answer_entropy.py $M/answer_entropy > $M/answer_entropy/analysis.txt 2>&1
$PY scripts/analyze_bonds.py --bonds $M/bonds --ent $M/answer_entropy > $M/bonds/analysis.txt 2>&1
$PY scripts/analyze_reflection_coupling.py > /dev/null 2>&1           # -> $M/bonds/reflection_coupling.txt
$PY scripts/eval_reflection_benchmarks.py analyze > /dev/null 2>&1      # -> $E/reflection_more/analysis.txt

## Mechanism (Qwen, new SAE)
echo "== Mechanism (Qwen, new SAE)"
$PY scripts/mi_qwen_student_probe.py --analyze > /dev/null 2>&1         # -> $E/student_probe/analysis.txt
$PY scripts/mi_qwen_student_causal.py analyze > /dev/null 2>&1          # -> $E/student_causal/analysis.txt
$PY scripts/analyze_mi_qwen_teacher_newsae.py 2>/dev/null > $E/teacher_mi/analysis.txt

## What the alignment term does inside the student (docs/mi_qwen_students.md)
echo "== What the alignment term does inside the student"
Q=eval_outputs/mi_qwen_students
for c in "mi_qwen_patching.py analyze" "mi_qwen_diffs.py analyze" "mi_qwen_crosscoder.py analyze" "mi_qwen_alpha0.py analyze_vast" \
         "mi_qwen_alpha0_props.py compare" "mi_qwen_alpha_causal.py analyze" "mi_qwen_alpha_mech.py analyze" "mi_qwen_alpha_ext.py analyze"; do
  $PY scripts/$c > /dev/null 2>&1                                      # -> $Q/*/analysis*.txt, $Q/alpha0/properties.txt
done

## Teacher MI: steering, patching and suppression (Llama and Qwen teachers)
echo "== Teacher MI: steering, patching and suppression (Llama and Qwen teachers)"
$PY scripts/analyze_mi_steering.py 2>/dev/null > eval_outputs/mi_steering/llama_teacher/analysis.txt
$PY scripts/analyze_mi_steering.py qwen_teacher 2>/dev/null > eval_outputs/mi_steering/qwen_teacher/analysis.txt
$PY scripts/analyze_mi_ablation.py 2>/dev/null > eval_outputs/mi_ablation/llama_teacher/analysis.txt
$PY scripts/analyze_mi_ablation.py qwen_teacher 2>/dev/null > eval_outputs/mi_ablation/qwen_teacher/analysis.txt

## Llama arm
echo "== Llama arm"
$PY scripts/analyze_mi_llama_students.py 2>/dev/null > eval_outputs/mi_students/llama/analysis.txt
$PY scripts/analyze_llama_passk.py eval_outputs/llama_passk 2>&1 | nw > eval_outputs/llama_passk/summary.txt
$PY scripts/analyze_llama_outcomes.py 2>&1 | nw > eval_outputs/llama_passk/outcomes_summary.txt
$PY scripts/analyze_llama_teacher_forced.py 2>&1 | nw > eval_outputs/llama_teacher_forced/summary.txt
$PY scripts/analyze_llama_budget_forcing.py eval_outputs/llama_budget_forcing 2>&1 | nw > eval_outputs/llama_budget_forcing/summary.txt
$PY scripts/analyze_bonds.py --bonds eval_outputs/llama_bonds --ent eval_outputs/llama_bonds/entropy 2>&1 | nw > eval_outputs/llama_bonds/summary.txt
$PY scripts/analyze_llama_answer_entropy.py 2>&1 | nw > eval_outputs/llama_bonds/answer_entropy_summary.txt

## Teacher accuracy (Qwen3.5-4B on MATH-500; students at 8k are in $M/passk/analysis.txt)
echo "== Teacher accuracy (Qwen3.5-4B on MATH-500; students at 8k are in $M/passk/analysis.txt)"
$PY scripts/analyze_basic_qwen.py math500 Teacher=eval_outputs/teacher_qwen4b/qwen-3.5-4b-teacher-think__math500.json 2>&1 | nw \
    > eval_outputs/teacher_qwen4b/analysis.txt
$PY scripts/analyze_basic_qwen.py math500 Teacher=eval_outputs/teacher_qwen4b_32k/qwen-3.5-4b-teacher-think__math500.json LKD=$T42 \
    RD_mod=$(R moderate) 2>&1 | nw > eval_outputs/teacher_qwen4b_32k/analysis.txt

## October RD context (text only in the paper): LKD vs October RD
echo "== October RD context (text only in the paper): LKD vs October RD"
$PY scripts/analyze_budget_forcing.py eval_outputs/budget_forcing lkd rd_oct > eval_outputs/budget_forcing/summary_rd_oct.txt 2>&1
$PY scripts/analyze_processbench.py > $PB/analysis.txt 2>&1
$PY scripts/analyze_processbench.py --boxed-only > $PB/analysis_boxed_only.txt 2>&1
$PY scripts/analyze_hard_B.py 2>&1 | nw > $H/B/summary.txt
echo "ALL ANALYSES DONE"
