#!/bin/bash
# MATH-500 @32k for new-SAE RD high then moderate (run from repo root). One log per level.
cd "$(dirname "$0")/.."
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
C=configs/eval; L=eval_outputs/rd_newsae/eval; mkdir -p $L
for l in high moderate; do
  c=${l/moderate/mod}
  $PY -m custom_eval.run_eval --config $C/rd_${c}_math500_32k.yaml > $L/math500_32k_$l.log 2>&1
  echo "exit $?" >> $L/math500_32k_$l.log
done
