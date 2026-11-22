"""Summarise scripts/eval_llama_passk.py: LKD vs RD on the Llama arm, paired per question.

For each benchmark:
  accuracy      per-sample accuracy (lenient and strict), pass@1/2/4/8
  reliability   questions solved on all 8 samples; distribution of #correct
  spread        distinct committed answers per question, clustered with the
                arm's own matcher (numeric_or_exact_match); all samples and
                answered-only
  length        mean / median generated tokens: all samples (censored by the
                cap), answered samples only, and paired per question
  truncation    hit the cap or stopped as a loop, split into loop vs long

RD - LKD differences are bootstrapped over questions (paired); binary
per-question outcomes also get an exact sign test on discordant questions.

Run: python scripts/analyze_llama_passk.py eval_outputs/llama_passk
"""

import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama_eval"))
from batched_evals.scoring import numeric_or_exact_match  # noqa: E402

MODELS = ("logit_kd_epoch2", "reasondistill_epoch2")
SHORT = {"logit_kd_epoch2": "LKD", "reasondistill_epoch2": "RD"}


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def boot(d, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def sign_test(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def clusters(answers):
    reps = []
    for a in answers:
        if not any(a == r or numeric_or_exact_match(a, r) or numeric_or_exact_match(r, a) for r in reps):
            reps.append(a)
    return len(reps)


def per_question(rec):
    cs = rec["candidates"]
    n = len(cs)
    ans = [c["answer_lenient"] for c in cs if c["answer_lenient"]]
    answered = [c for c in cs if c["state"] == "answered"]
    return {
        "acc": sum(c["correct_lenient"] for c in cs) / n,
        "acc_strict": sum(c["correct_strict"] for c in cs) / n,
        "ncorrect": sum(c["correct_lenient"] for c in cs),
        "n": n,
        "distinct": clusters(ans),
        "committed": len(ans),
        "trunc": sum(c["hit_limit"] for c in cs) / n,
        "loop": sum(c["loop"] for c in cs) / n,
        "tok_all": st.median(c["tokens"] for c in cs),
        "tok_answered": st.median(c["tokens"] for c in answered) if answered else None,
    }


def main(out_dir):
    out_dir = Path(out_dir)
    for bench in ("aime25", "math500", "gsm8k"):
        paths = {m: out_dir / f"{m}__{bench}.json" for m in MODELS}
        if not all(p.exists() for p in paths.values()):
            print(f"\n######## {bench}: incomplete, skipping")
            continue
        data = {m: {r["id"]: r for r in json.load(open(p))["records"]} for m, p in paths.items()}
        ids = list(data[MODELS[0]])
        cap = json.load(open(paths[MODELS[0]]))["max_new_tokens"]
        Q = {m: {i: per_question(data[m][i]) for i in ids} for m in MODELS}
        k = Q[MODELS[0]][ids[0]]["n"]
        print(f"\n######## {bench}: {len(ids)} questions x {k} samples, cap {cap} tokens")
        for m in MODELS:
            cs = [c for i in ids for c in data[m][i]["candidates"]]
            ans = [c["tokens"] for c in cs if c["state"] == "answered"]
            pk = {kk: st.mean(pass_at(k, Q[m][i]["ncorrect"], kk) for i in ids) for kk in (1, 2, 4, 8) if kk <= k}
            dist = [sum(Q[m][i]["ncorrect"] == c for i in ids) for c in range(k + 1)]
            states = {}
            for c in cs:
                states[c["state"]] = states.get(c["state"], 0) + 1
            print(f"  {SHORT[m]:3s} acc lenient {st.mean(c['correct_lenient'] for c in cs):.4f} "
                  f"strict {st.mean(c['correct_strict'] for c in cs):.4f} | "
                  + " ".join(f"pass@{kk}={v:.4f}" for kk, v in pk.items()))
            print(f"      solved on all {k}: {dist[k]}  never: {dist[0]}  #correct histogram 0..{k}: {dist}")
            print(f"      truncated {st.mean(c['hit_limit'] for c in cs):.3f} (loop {st.mean(c['loop'] for c in cs):.3f}, "
                  f"long {st.mean(c['hit_limit'] and not c['loop'] for c in cs):.3f}) | states "
                  + " ".join(f"{s}={v / len(cs):.3f}" for s, v in sorted(states.items())))
            print(f"      tokens all: mean {st.mean(c['tokens'] for c in cs):.0f} median {st.median(c['tokens'] for c in cs):.0f}"
                  f" (censored at cap) | answered only: mean {st.mean(ans):.0f} median {st.median(ans):.0f} (n={len(ans)})"
                  if ans else "      no answered samples")
            print(f"      distinct answers/question: mean {st.mean(Q[m][i]['distinct'] for i in ids):.3f}")
        a, b = (Q[m] for m in MODELS)
        print("  RD - LKD, paired over questions (95% bootstrap CI):")
        for key, label in (("acc", "per-sample accuracy"), ("acc_strict", "per-sample acc strict"),
                           ("trunc", "truncation rate"), ("loop", "loop rate"),
                           ("distinct", "distinct answers")):
            mean, lo, hi, p = boot([b[i][key] - a[i][key] for i in ids])
            print(f"    {label:24s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
        both = [i for i in ids if a[i]["committed"] == b[i]["committed"] == k]
        if both:
            mean, lo, hi, p = boot([b[i]["distinct"] - a[i]["distinct"] for i in both])
            print(f"    {'distinct, all 8 committed':24s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} (n={len(both)})")
        both = [i for i in ids if a[i]["tok_answered"] is not None and b[i]["tok_answered"] is not None]
        if both:
            d = [b[i]["tok_answered"] - a[i]["tok_answered"] for i in both]
            mean, lo, hi, p = boot(d)
            shorter = sum(x < 0 for x in d)
            print(f"    {'median answered tokens':24s} {mean:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f} (n={len(both)}; RD shorter on {shorter}, "
                  f"sign p={sign_test(shorter, sum(x > 0 for x in d)):.4f})")
        for label, fn in (("pass@8 (any correct)", lambda q: q["ncorrect"] > 0),
                          (f"solved on all {k}", lambda q: q["ncorrect"] == k)):
            rd_only = sum(fn(b[i]) and not fn(a[i]) for i in ids)
            lkd_only = sum(fn(a[i]) and not fn(b[i]) for i in ids)
            print(f"    {label:24s} RD-only {rd_only}  LKD-only {lkd_only}  sign p={sign_test(rd_only, lkd_only):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
