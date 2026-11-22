#!/bin/bash
# Full October-RD eval suite for new-SAE RD moderate (final_step_94), same settings as each original run.
# LKD / October RD outputs are reused for pairing. One log per stage in eval_outputs/rd_newsae/eval_mod/logs;
# STATUS holds the current stage. Characteristics (bonds, answer entropy) hook in after pass@8.
cd "$(dirname "$0")/.."
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
PYF=${PYF:-venv_fla/bin/python}   # flash-linear-attention environment, used for the long-context sets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
C=configs/eval; E=eval_outputs/rd_newsae/eval_mod; L=$E/logs; mkdir -p $L
step() {  # step <name> <command...>
  local n=$1; shift
  if grep -qx "exit 0" $L/$n.log 2>/dev/null; then echo "skip $n (done)" >> $L/queue.log; return; fi
  echo "RUNNING $n $(date +%FT%T)" > $L/STATUS; echo "start $n $(date +%FT%T)" >> $L/queue.log
  "$@" > $L/$n.log 2>&1; local c=$?
  echo "exit $c" >> $L/$n.log; echo "end $n exit $c $(date +%FT%T)" >> $L/queue.log
}
step teacher_forced $PY scripts/eval_teacher_forced_kl.py --no_base --students lkd rd_oct rd_mod --out $E/teacher_forced/results.json
step basic_math   $PY -m custom_eval.run_eval --config $C/rd_mod_math_basic.yaml
step sept_think   $PY -m custom_eval.run_eval --config $C/rd_mod_september_think.yaml
step sept_nothink $PY -m custom_eval.run_eval --config $C/rd_mod_september_no_think.yaml
step passk        $PY -m custom_eval.run_eval --config $C/rd_mod_passk.yaml
[ -x scripts/run_rd_newsae_characteristics.sh ] && step characteristics scripts/run_rd_newsae_characteristics.sh
step budget_forcing $PY scripts/eval_budget_forcing.py --models rd_mod --budgets 512 1024 2048 4096 --samples 4 --out_dir $E/budget_forcing
step hard_A       $PYF -m custom_eval.run_eval --config $C/rd_mod_hard_A.yaml
step hard_B       $PYF -m custom_eval.run_eval --config $C/rd_mod_hard_B.yaml
step hard_C       $PYF -m custom_eval.run_eval --config $C/rd_mod_hard_C.yaml
step processbench $PYF -m custom_eval.run_eval --config $C/rd_mod_processbench.yaml
echo "ALL_DONE $(date +%FT%T)" > $L/STATUS
