#!/bin/bash
# Qwen student MI suite (docs/mi_qwen_students.md). Stages run in order; every phase resumes or skips finished work.
#   bash scripts/run_mi_qwen_students.sh [tf|crosscoder|all]      (default all)
# tf         Exp 4 (weight updates), Exp 3 (P_R by model)
# crosscoder Exp 1 (crosscoder + pooled SAE, ~100M tokens) and Exp 5
cd "$(dirname "$0")/.."
PY=${PYF:-venv_fla/bin/python}; L=logs/mi_qwen_students; mkdir -p $L
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
stage=${1:-all}
if [[ $stage == tf || $stage == all ]]; then
  [[ -f eval_outputs/mi_qwen_students/diffs/weights.json ]] || $PY scripts/mi_qwen_diffs.py weights >> $L/diffs.log 2>&1
  $PY scripts/mi_qwen_diffs.py analyze >> $L/diffs.log 2>&1
  for p in gap analyze; do $PY scripts/mi_qwen_patching.py $p >> $L/patching.log 2>&1; done
fi
if [[ $stage == crosscoder || $stage == all ]]; then
  [[ -f checkpoints/mi_qwen_crosscoder/final.pt ]] || $PY scripts/mi_qwen_crosscoder.py train >> $L/crosscoder.log 2>&1
  for p in eval label transfer analyze; do $PY scripts/mi_qwen_crosscoder.py $p >> $L/crosscoder.log 2>&1; done
fi
