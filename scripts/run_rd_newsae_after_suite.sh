#!/bin/bash
# After run_rd_newsae_suite.sh finishes: LKD-continue MATH-500 @32k with seed 43 (sampling-noise check).
cd "$(dirname "$0")/.."
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
C=configs/eval; E=eval_outputs/rd_newsae; L=$E/eval
until grep -q ALL_DONE $E/eval_mod/logs/STATUS 2>/dev/null; do sleep 120; done
$PY -m custom_eval.run_eval --config $C/lkd_seed43_math500_32k.yaml > $L/lkd_seed43_math500_32k.log 2>&1
echo "exit $?" >> $L/lkd_seed43_math500_32k.log
