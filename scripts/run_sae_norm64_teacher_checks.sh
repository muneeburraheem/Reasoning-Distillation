#!/bin/bash
# Checks 1-3 in GPU order: calib -> patch/attribution (checks 2a, 3) -> student probe (check 1) -> sfull (check 2b).
# Each step resumes from its outputs; retried up to 3 times.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYF:-venv_fla/bin/python}; R=eval_outputs/rd_newsae/teacher_mi; L=logs/rd_newsae/teacher_checks.log
mkdir -p $R logs/rd_newsae
step() { for try in 1 2 3; do echo "$(date) $1 try $try" >> $L; "${@:2}" >> $L 2>&1 && return 0; done; echo "$(date) $1 FAILED" >> $L; }
[ -f $R/steering/calib.json ] || step calib $PY scripts/mi_qwen_teacher_newsae.py steer --phases calib
[ -f $R/ablation/sets.json ] || step patch $PY scripts/mi_qwen_teacher_newsae.py ablate --phases patch
step probe $PY scripts/mi_qwen_student_probe.py
$PY scripts/mi_qwen_student_probe.py --analyze >> $L 2>&1
step sfull $PY scripts/mi_qwen_teacher_newsae.py ablate --phases sfull
echo "$(date) finished" >> $L
