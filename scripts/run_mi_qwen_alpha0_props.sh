#!/bin/bash
# alpha = 0 comparison (docs/mi_qwen_students.md): once a checkpoint is downloaded and sha256-verified (LOCAL_OK in the
# download log given as $1), compute its properties and probe; finally compare all models.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DL=$1; PY=venv_fla/bin/python; L=logs/mi_qwen_students/alpha0_props.log
for pair in "zero rd_zero" "lowrerun rd_low_rr"; do
  set -- $pair
  until grep -qE "LOCAL_(OK|MISMATCH) $1" $DL; do sleep 120; done
  grep -q "LOCAL_OK $1" $DL || { echo "checksum mismatch for $1, skipped" >> $L; continue; }
  $PY scripts/mi_qwen_alpha0_props.py model $2 >> $L 2>&1
  $PY scripts/mi_qwen_student_probe.py >> $L 2>&1
done
$PY scripts/mi_qwen_alpha0_props.py compare >> $L 2>&1
echo "ALPHA0_PROPS_DONE $(date)" >> $L
