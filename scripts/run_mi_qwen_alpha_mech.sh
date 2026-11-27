#!/bin/bash
# Depth profile of the alignment term vs alpha = 0 (docs/mi_qwen_students.md): M1 probe-by-layer,
# computed in analyze from the existing probe files.
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
LOG=logs/mi_qwen_students/alpha_mech.log
mkdir -p logs/mi_qwen_students
venv_fla/bin/python scripts/mi_qwen_alpha_mech.py analyze >> $LOG 2>&1 || { echo "FAILED analyze" >> $LOG; exit 1; }
echo ALPHA_MECH_DONE >> $LOG
