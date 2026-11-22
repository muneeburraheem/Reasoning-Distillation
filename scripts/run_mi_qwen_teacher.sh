#!/bin/bash
# Qwen teacher mech-interp suite: mi_steering (calib grid full) then mi_ablation (patch sgrid sfull).
# Both scripts resume from their outputs; a phase that crashes is retried up to 3 times.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYF:-venv_fla/bin/python}
S=eval_outputs/mi_steering/qwen_teacher; A=eval_outputs/mi_ablation/qwen_teacher
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
for try in 1 2 3; do
  ph=""; [ -f $S/calib.json ] || ph="calib"
  [ "$(lines $S/grid.jsonl)" -ge 161 ] || ph="$ph grid"
  [ "$(lines $S/full.jsonl)" -ge 7 ] || ph="$ph full"
  [ -z "$ph" ] && break
  echo "$(date) steering try $try: $ph"
  $PY scripts/mi_steering.py --subject qwen_teacher --phases $ph >> logs/mi_steering_qwen_teacher.log 2>&1
done
for try in 1 2 3; do
  ph=""; [ -f $A/sets.json ] || ph="patch"
  [ "$(lines $A/sgrid.jsonl)" -ge 12 ] || ph="$ph sgrid"
  [ "$(lines $A/sfull.jsonl)" -ge 6 ] || ph="$ph sfull"
  [ -z "$ph" ] && break
  echo "$(date) ablation try $try: $ph"
  $PY scripts/mi_ablation.py --subject qwen_teacher --phases $ph >> logs/mi_ablation_qwen_teacher.log 2>&1
done
echo "$(date) finished"
