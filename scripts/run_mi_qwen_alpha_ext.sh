#!/bin/bash
# Teacher agreement and structure of the alignment term vs alpha = 0 (docs/mi_qwen_students.md): X4 weight shape, teacher
# pass (X1 / X5 / X6 targets), per-student pass (X1, X3, X5, X6), X2 weight arithmetic, then analyze.
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
LOG=logs/mi_qwen_students/alpha_ext.log
mkdir -p logs/mi_qwen_students
for ph in svd teacher students arith x5fix analyze; do
  venv_fla/bin/python scripts/mi_qwen_alpha_ext.py $ph >> $LOG 2>&1 || { echo "FAILED $ph" >> $LOG; exit 1; }
done
echo ALPHA_EXT_DONE >> $LOG
