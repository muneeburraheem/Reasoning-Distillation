#!/bin/bash
# Queued after run_rd_newsae_math500.sh: waits for the moderate eval to end, then MATH-500 @32k for low.
cd "$(dirname "$0")/.."
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
C=configs/eval; L=eval_outputs/rd_newsae/eval
until grep -q "^exit" $L/math500_32k_moderate.log 2>/dev/null; do sleep 60; done
$PY -m custom_eval.run_eval --config $C/rd_low_math500_32k.yaml > $L/math500_32k_low.log 2>&1
echo "exit $?" >> $L/math500_32k_low.log
