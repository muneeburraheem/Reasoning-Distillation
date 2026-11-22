#!/bin/bash
cd "$(dirname "$0")/.."; L=logs/rd_newsae; mkdir -p $L
while pgrep -f sae_norm64_reason_score.py >/dev/null; do sleep 30; done
[ -f checkpoints/qwen_sae_norm64/full_l1_003/reasonscore/reasonscore.pt ] || { echo "REASONSCORE FAILED" >> $L/gate.log; exit 1; }
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYF:-venv_fla/bin/python} scripts/sae_norm64_gate.py >> $L/gate.log 2>&1
echo "GATE DONE rc=$?" >> $L/gate.log
