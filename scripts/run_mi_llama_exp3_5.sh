#!/bin/bash
# Chains Exp 5 -> 3 -> 4 (Llama students, RUN_LOG 8ac). Each script resumes from its own outputs.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
while pgrep -f "scripts/mi_llama_exp5_cka.py" > /dev/null; do sleep 30; done
for e in mi_llama_exp3_feature_function mi_llama_exp4_student_attribution; do
  for try in 1 2; do
    echo "$(date) $e try $try"
    $PY scripts/$e.py >> logs/$e.log 2>&1 && break
  done
done
echo "$(date) all done"
