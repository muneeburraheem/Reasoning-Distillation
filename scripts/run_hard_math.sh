#!/usr/bin/env bash
# Harder benchmarks A -> B -> C (RUN_LOG 8q), in venv_fla.
cd "$(dirname "$0")/.."
PY=${PYF:-venv_fla/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
for S in A B C; do
  $PY -m custom_eval.run_eval --config configs/eval/qwen_hard_$S.yaml > logs/hard_$S.log 2>&1
  echo "exit $?" >> logs/hard_$S.log
done
