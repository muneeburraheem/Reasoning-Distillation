"""Qwen pass@8 (MATH-500, AIME-25): new RD vs LKD and old RD, paired per question.

Per model: pass@1/2/4/8 (strict = harness; lenient also credits the last
\\boxed{} of a trace that stopped inside thinking), reliability (#correct of 8),
answer spread (distinct finished answers per question, clustered with
math_equivalent; plus the committed-count control that killed the earlier
"wider distribution" claim), generated tokens (finished samples only, since
truncated ones sit at the cap), and how samples ended.

Contrasts against the first model listed: per-question differences
bootstrapped over questions; binary per-question outcomes get exact sign tests.

Run:
  python scripts/analyze_qwen_passk.py <benchmark> <label=path.json> ...
"""

import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_eval.math_scoring import aime_exact_match, extract_boxed_answer, math_equivalent  # noqa: E402


def outcome(cand, gold, eq):
    t = cand["raw_output"]
    if cand["hit_token_limit"]:
        return "truncated"
    if "</think>" not in t:
        b = extract_boxed_answer(t)
        return "eos_ok" if b and eq(b, gold) else "eos_bad"
    return "correct" if cand["correct"] else "committed_wrong"


def clusters(answers, eq):
    reps = []
    for a in answers:
        if not any(a == r or eq(a, r) or eq(r, a) for r in reps):
            reps.append(a)
    return len(reps)


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def boot(d, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_question(rec, eq):
    cs = rec["candidates"]
    oc = [outcome(c, rec["answer"], eq) for c in cs]
    finished = [(c.get("final_response") or "").strip() for c, o in zip(cs, oc) if o in ("correct", "committed_wrong")]
    finished = [a for a in finished if a]
    tok = [int(c["generated_tokens"]) for c, o in zip(cs, oc) if o != "truncated"]
    return {
        "n": len(cs),
        "strict": sum(o == "correct" for o in oc),
        "lenient": sum(o in ("correct", "eos_ok") for o in oc),
        "committed_wrong": sum(o == "committed_wrong" for o in oc) / len(cs),
        "eos": sum(o.startswith("eos") for o in oc) / len(cs),
        "trunc": sum(o == "truncated" for o in oc) / len(cs),
        "answers": finished,
        "distinct": clusters(finished, eq),
        "tok_median": st.median(tok) if tok else None,
        "tok_all": tok,
    }


def main():
    bench = sys.argv[1]
    eq = aime_exact_match if bench == "aime25" else math_equivalent
    runs = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        runs.append((label, {r["id"]: r for r in json.load(open(path))["records"]}))
    ids = sorted(set.intersection(*(set(r) for _, r in runs)))
    Q = {label: {i: per_question(r[i], eq) for i in ids} for label, r in runs}
    k = Q[runs[0][0]][ids[0]]["n"]
    print(f"######## {bench}: {len(ids)} questions x {k} samples")
    for label, _ in runs:
        q = Q[label]
        ps = " ".join(f"pass@{kk}={st.mean(pass_at(k, q[i]['strict'], kk) for i in ids):.3f}" for kk in (1, 2, 4, 8))
        pl = st.mean(pass_at(k, q[i]["lenient"], 1) for i in ids)
        hist = [sum(q[i]["strict"] == c for i in ids) for c in range(k + 1)]
        toks = [t for i in ids for t in q[i]["tok_all"]]
        print(f"  {label:8s} {ps} | lenient pass@1={pl:.3f}")
        print(f"           solved {k}/{k}: {hist[k]}  never: {hist[0]}  #correct 0..{k}: {hist}")
        print(f"           committed wrong {st.mean(q[i]['committed_wrong'] for i in ids):.3f}  stopped in think "
              f"{st.mean(q[i]['eos'] for i in ids):.3f}  truncated {st.mean(q[i]['trunc'] for i in ids):.3f}")
        if toks:
            print(f"           finished tokens: mean {st.mean(toks):.0f} median {st.median(toks):.0f} (n={len(toks)})"
                  f" | distinct answers/question {st.mean(q[i]['distinct'] for i in ids):.3f}")
    ref = runs[0][0]
    a = Q[ref]
    for label, _ in runs[1:]:
        b = Q[label]
        print(f"  {ref} - {label} (paired over questions):")
        for key, name, scale in (("strict", "pass@1 strict", 1 / k), ("lenient", "pass@1 lenient", 1 / k),
                                 ("committed_wrong", "committed wrong", 1), ("eos", "stopped in think", 1),
                                 ("trunc", "truncated", 1), ("distinct", "distinct answers", 1)):
            mean, lo, hi, p = boot([(a[i][key] - b[i][key]) * scale for i in ids])
            print(f"    {name:20s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
        d = []
        for i in ids:
            if len(a[i]["answers"]) >= 2 and len(b[i]["answers"]) >= 2:
                d.append(a[i]["distinct"] / len(a[i]["answers"]) - b[i]["distinct"] / len(b[i]["answers"]))
        if d:
            mean, lo, hi, p = boot(d)
            print(f"    {'distinct per answer':20s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} (n={len(d)}, both >=2 finished)")
        both = [i for i in ids if a[i]["tok_median"] is not None and b[i]["tok_median"] is not None]
        if both:
            dd = [a[i]["tok_median"] - b[i]["tok_median"] for i in both]
            mean, lo, hi, p = boot(dd)
            sh = sum(x < 0 for x in dd)
            print(f"    {'median fin. tokens':20s} {mean:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f} "
                  f"(n={len(both)}; {ref} shorter on {sh}, sign p={sign_p(sh, sum(x > 0 for x in dd)):.4f})")
        for name, fn in (("pass@8 (any)", lambda q: q["strict"] > 0), (f"all {k} correct", lambda q: q["strict"] == k)):
            x = sum(fn(a[i]) and not fn(b[i]) for i in ids)
            y = sum(fn(b[i]) and not fn(a[i]) for i in ids)
            print(f"    {name:20s} {ref}-only {x}  {label}-only {y}  sign p={sign_p(x, y):.4f}")


if __name__ == "__main__":
    main()
