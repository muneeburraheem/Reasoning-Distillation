"""Basic Qwen evals (1 sample per question): new RD vs LKD, old RD and base, paired.

Per model: accuracy under the harness's strict rule (answer must follow
</think>) and the lenient rule (also credit the last \\boxed{} of a trace that
stopped inside thinking), and how each sample ended: correct, committed wrong,
stopped inside thinking (EOS before </think>), truncated at the cap.

Paired contrasts against the first model listed: accuracy difference with a
bootstrap CI over questions, and exact sign tests on discordant questions for
correct / committed-wrong / stopped-inside-thinking. MATH-500 also by level.

Run:
  python scripts/analyze_basic_qwen.py <benchmark> <label=path.json> [<label=path.json> ...]
"""

import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_eval.math_scoring import extract_boxed_answer, math_equivalent  # noqa: E402


def outcome(rec, cand):
    t = cand["raw_output"]
    if cand["hit_token_limit"]:
        return "truncated"
    if "</think>" not in t:
        b = extract_boxed_answer(t)
        return "eos_ok" if b and math_equivalent(b, rec["answer"]) else "eos_bad"
    return "correct" if cand["correct"] else "committed_wrong"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(d, b=20000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def main():
    bench = sys.argv[1]
    runs = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        d = json.load(open(path))
        recs = {r["id"]: r for r in d["records"]}
        if "hit_token_limit" not in next(iter(recs.values()))["candidates"][0]:
            # September-era runs had no answer-stopper: models keep writing after
            # their answer until the cap, so the outcome split is undefined there.
            # Use scripts/analyze_september_qwen.py for those.
            raise SystemExit(f"{path}: no token-limit field (pre answer-stopper run); not classifiable")
        runs.append((label, recs))
    ids = sorted(set.intersection(*(set(r) for _, r in runs)))
    print(f"######## {bench}: {len(ids)} questions common to all runs, 1 sample each")
    oc = {label: {i: outcome(r[i], r[i]["candidates"][0]) for i in ids} for label, r in runs}
    for label, r in runs:
        o = oc[label]
        n = len(ids)
        strict = sum(v == "correct" for v in o.values()) / n
        lenient = sum(v in ("correct", "eos_ok") for v in o.values()) / n
        fin = sorted(int(r[i]["candidates"][0]["generated_tokens"]) for i in ids if o[i] != "truncated")
        print(f"  {label:10s} strict {strict:.3f}  lenient {lenient:.3f} | committed wrong "
              f"{sum(v == 'committed_wrong' for v in o.values()) / n:.3f}  stopped in think "
              f"{sum(v.startswith('eos') for v in o.values()) / n:.3f}  truncated "
              f"{sum(v == 'truncated' for v in o.values()) / n:.3f} | median finished tokens "
              f"{fin[len(fin) // 2] if fin else '-'}")
    ref, _ = runs[0]
    for label, _ in runs[1:]:
        print(f"  {ref} - {label}:")
        for name, ok in (("strict", lambda v: v == "correct"),
                         ("lenient", lambda v: v in ("correct", "eos_ok"))):
            d = [int(ok(oc[ref][i])) - int(ok(oc[label][i])) for i in ids]
            mean, lo, hi, p = boot(d)
            print(f"    accuracy {name:8s} {100 * mean:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.4f}")
        for name, fn in (("correct (strict)", lambda v: v == "correct"),
                         ("committed wrong", lambda v: v == "committed_wrong"),
                         ("stopped in think", lambda v: v.startswith("eos"))):
            a = sum(fn(oc[ref][i]) and not fn(oc[label][i]) for i in ids)
            b = sum(fn(oc[label][i]) and not fn(oc[ref][i]) for i in ids)
            print(f"    {name:18s} {ref}-only {a}  {label}-only {b}  sign p={sign_p(a, b):.4f}")
    first = runs[0][1]
    if bench == "math500" and "level" in first[ids[0]].get("metadata", {}):
        print("  strict accuracy by level:")
        for lv in range(1, 6):
            q = [i for i in ids if first[i]["metadata"]["level"] == lv]
            print(f"    L{lv} n={len(q):3d} " + "  ".join(
                f"{label}={sum(oc[label][i] == 'correct' for i in q) / len(q):.2f}" for label, _ in runs))


if __name__ == "__main__":
    main()
