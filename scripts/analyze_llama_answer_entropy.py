"""Llama arm: answer-token entropy on GSM8K finished traces, RD - LKD paired per question (RUN_LOG 8u).

Reads eval_outputs/llama_bonds/entropy/rows.json (written by scripts/eval_llama_bonds.py). Three pairings:
each model scoring its own text, both scoring RD's text, both scoring LKD's text. Bootstrap 5,000.

Run: python scripts/analyze_llama_answer_entropy.py > eval_outputs/llama_bonds/answer_entropy_summary.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_llama_teacher_forced import boot  # noqa: E402

rows = json.load(open("eval_outputs/llama_bonds/entropy/rows.json"))
R = {(r["scorer"], r["author"], r["qid"]): r for r in rows}
qs = sorted({r["qid"] for r in rows})
print("Answer-token entropy, GSM8K finished traces (RD - LKD, paired per question)")
for k in ("first_ent", "ans_ent", "trace_ent"):
    for label, pairs in (("self (own text)", [(("RD", "RD", q), ("LKD", "LKD", q)) for q in qs]),
                         ("same text: RD's", [(("RD", "RD", q), ("LKD", "RD", q)) for q in qs]),
                         ("same text: LKD's", [(("RD", "LKD", q), ("LKD", "LKD", q)) for q in qs])):
        d = [R[a][k] - R[b][k] for a, b in pairs if a in R and b in R and k in R[a] and k in R[b]]
        m, lo, hi, p = boot(d, 5000)
        mean_l = sum(R[b][k] for a, b in pairs if a in R and b in R and k in R[a] and k in R[b]) / len(d)
        print(f"  {k:10s} {label:17s} n={len(d)} LKD {mean_l:.4f}  RD-LKD {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
