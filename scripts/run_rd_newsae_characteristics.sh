#!/bin/bash
# Characteristics for RD moderate (after its pass@8): step labels + embeddings, then answer-token entropy,
# for LKD / October RD / RD moderate together (same scripts and settings as 8n and the v2 entropy rerun).
cd "$(dirname "$0")/.."
PY=${PY:-python}          # main environment (requirements.txt); set PY=/path/to/venv/bin/python
E=eval_outputs/rd_newsae/eval_mod
$PY scripts/eval_bond_annotate.py --out_dir $E/bonds || exit 1
$PY scripts/eval_answer_entropy.py --per_q 4 --out_dir $E/answer_entropy
