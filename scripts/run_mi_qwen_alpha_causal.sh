#!/bin/bash
# Causal tests of the alignment term vs the alpha = 0 run (scripts/mi_qwen_alpha_causal.py, docs/mi_qwen_students.md).
# $1 (optional): download log of the alpha 0.05 rerun; once it reports LOCAL_OK lowrerun, the rerun joins the bench phase.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=venv_fla/bin/python; L=logs/mi_qwen_students/alpha_causal.log
for ph in graft patch bench; do
  $PY scripts/mi_qwen_alpha_causal.py $ph >> $L 2>&1 || echo "PHASE_FAILED $ph" >> $L
done
$PY scripts/mi_qwen_alpha_causal.py analyze >> $L 2>&1
echo "ALPHA_CAUSAL_MAIN_DONE $(date)" >> $L
if [ -n "$1" ]; then
  until grep -qE "LOCAL_(OK|MISMATCH) lowrerun" "$1"; do sleep 120; done
  if grep -q "LOCAL_OK lowrerun" "$1"; then
    touch checkpoints/rd_newsae_lowrerun/VERIFIED
    $PY scripts/mi_qwen_alpha_causal.py bench >> $L 2>&1 || echo "PHASE_FAILED bench rerun" >> $L
    $PY scripts/mi_qwen_alpha_causal.py analyze >> $L 2>&1
  else
    echo "rerun checksum mismatch, bench without it" >> $L
  fi
fi
echo "ALPHA_CAUSAL_DONE $(date)" >> $L
