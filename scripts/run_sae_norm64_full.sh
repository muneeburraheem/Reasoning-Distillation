#!/bin/bash
# Full run (weights-only saves) + health check of each checkpoint as it appears. Log: logs/rd_newsae/full.log
cd "$(dirname "$0")/.."
C=configs/train; R=checkpoints/qwen_sae_norm64/full_l1_003; L=logs/rd_newsae; PY=${PYF:-venv_fla/bin/python}; mkdir -p $L
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date) full_l1_003 start" >> $L/full.log
$PY scripts/train_sae_norm64.py --config $C/sae_norm64_full.yaml --weights-only > $L/full_l1_003.log 2>&1 &
TP=$!
done_list=""
while true; do
  for c in $R/step_*/sae.pt $R/final/sae.pt; do
    [ -f "$c" ] || continue; case " $done_list " in *" $c "*) continue;; esac
    sleep 30; n=$(basename $(dirname $c))
    EVAL_DEVICE=cpu $PY scripts/sae_norm64_health.py "full_l1_003/$n=$c" 2>/dev/null | grep FVE >> $L/full.log
    done_list="$done_list $c"
  done
  kill -0 $TP 2>/dev/null || break
  sleep 60
done
wait $TP; rc=$?
[ -f $R/final/sae.pt ] && ! grep -q "full_l1_003/final" $L/full.log && EVAL_DEVICE=cpu $PY scripts/sae_norm64_health.py "full_l1_003/final=$R/final/sae.pt" 2>/dev/null | grep FVE >> $L/full.log
echo "$(date) FULL DONE rc=$rc" >> $L/full.log
