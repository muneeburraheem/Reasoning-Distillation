"""Analysis of the Llama-arm ProcessBench run (scripts/eval_llama_processbench.py).

Same measures as analyze_processbench.py (RUN_LOG 8s), with
the Llama arm's scoring: the commitment-gated extractor (strict = lower bound, lenient = the arm's
headline), answer matched by math_equivalent or numeric_or_exact_match.

Run:
  python scripts/analyze_llama_processbench.py processbench eval_outputs/llama_processbench
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "llama_eval"))
from batched_evals.generation import extract_answer  # noqa: E402
from batched_evals.scoring import numeric_or_exact_match  # noqa: E402
from custom_eval.math_scoring import math_equivalent  # noqa: E402

M = ("logit_kd_epoch2", "reasondistill_epoch2")


def same(a, b):
    return bool(a and b) and (math_equivalent(a, b) or numeric_or_exact_match(a, b))


def answers(text):
    out = {}
    for mode in ("strict", "lenient"):
        e = extract_answer(text, mode)
        out[mode] = e.answer if e.committed else ""
    return out


def load(d, model, cond):
    p = d / f"{model}__{cond}.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def pb_parse(r):
    """Index or None. Accepts the gated answer as a bare integer or exactly one 'paragraph N'."""
    a = answers(r["text"])["lenient"]
    a = re.sub(r"[*$\\]|boxed|\{|\}", "", a).strip()
    if re.fullmatch(r"-?\d+", a):
        return int(a)
    m = re.findall(r"paragraph[_ ]?(-?\d+)", a, re.I)
    return int(m[0]) if len(set(m)) == 1 else None


def f1(rows):
    err = [r for r in rows if r["label"] != -1]; ok = [r for r in rows if r["label"] == -1]
    ae = sum(r["exact"] for r in err) / len(err) if err else float("nan")
    ao = sum(r["exact"] for r in ok) / len(ok) if ok else float("nan")
    return ae, ao, (2 * ae * ao / (ae + ao) if ae + ao > 0 else 0.0)


def processbench(d):
    R = {}
    for m in M:
        R[m] = {}
        for i, r in load(d, m, "critique").items():
            p = pb_parse(r)
            R[m][i] = {"split": r["split"], "label": r["label"], "pred": p, "exact": p == r["label"],
                       "valid": p is not None and p >= -1, "trunc": r["hit_limit"], "loop": r["loop"],
                       "state": r["state"]}
    ids = sorted(set(R[M[0]]) & set(R[M[1]]))
    print(f"ProcessBench, {len(ids)} items both models; official F1 (invalid = wrong), LKD / RD")
    rng = random.Random(0)
    for sp in ("gsm8k", "math", "olympiadbench", "omnimath", "all"):
        pool = [i for i in ids if sp == "all" or R[M[0]][i]["split"] == sp]
        if not pool:
            continue
        L = f1([R[M[0]][i] for i in pool]); Rr = f1([R[M[1]][i] for i in pool])
        bs = []
        for _ in range(2000):
            s = rng.choices(pool, k=len(pool))
            bs.append(f1([R[M[1]][i] for i in s])[2] - f1([R[M[0]][i] for i in s])[2])
        bs.sort()
        p = min(1.0, 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / len(bs))
        comp = [sum(R[m][i]["valid"] for i in pool) / len(pool) for m in M]
        flag = [sum((R[m][i]["pred"] or -1) >= 0 for i in pool if R[m][i]["label"] != -1) /
                max(1, sum(R[m][i]["label"] != -1 for i in pool)) for m in M]
        trunc = [sum(R[m][i]["trunc"] for i in pool) / len(pool) for m in M]
        print(f"  {sp:13s} n={len(pool):4d} F1 {L[2]:.3f} / {Rr[2]:.3f} (RD-LKD {Rr[2]-L[2]:+.3f} "
              f"[{bs[50]:+.3f}, {bs[1949]:+.3f}] p={p:.3f}) | acc err {L[0]:.3f} / {Rr[0]:.3f} | "
              f"acc correct {L[1]:.3f} / {Rr[1]:.3f} | valid index {comp[0]:.3f} / {comp[1]:.3f} | "
              f"flags a step on erroneous {flag[0]:.3f} / {flag[1]:.3f} | truncated {trunc[0]:.3f} / {trunc[1]:.3f}")
    # Chance reference: an answer drawn uniformly from -1..n-1 would score about 1/(n+1).
    from collections import Counter
    for m in M:
        c = Counter(R[m][i]["pred"] for i in ids)
        print(f"  {m} most common predictions: {c.most_common(6)}")


if __name__ == "__main__":
    {"processbench": processbench}[sys.argv[1]](Path(sys.argv[2]))
