#!/bin/bash
# 5 pilots x 10M tokens: train (weights only) -> health check -> delete weights (disk). Log: logs/rd_newsae/pilots.log
cd "$(dirname "$0")/.."
C=configs/train; R=checkpoints/qwen_sae_norm64; L=logs/rd_newsae; mkdir -p $L
PY=${PYF:-venv_fla/bin/python}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in pilot_norm_l1_0015 pilot_norm_l1_5 pilot_raw_l1_0015 pilot_norm_l1_0005 pilot_norm_l1_005; do
  flag=""; sfx=""; [[ $p == pilot_raw* ]] && flag="--raw" && sfx=":raw"
  echo "$(date) $p start" >> $L/pilots.log
  $PY scripts/train_sae_norm64.py --config $C/sae_norm64_$p.yaml --weights-only $flag > $L/$p.log 2>&1 || { echo "$(date) $p TRAIN FAILED" >> $L/pilots.log; continue; }
  $PY scripts/sae_norm64_health.py "$p=$R/$p/final/sae.pt$sfx" >> $L/pilots.log 2>&1 || echo "$(date) $p EVAL FAILED" >> $L/pilots.log
  rm -f $R/$p/final/sae.pt
done
echo "$(date) PILOTS DONE" >> $L/pilots.log
