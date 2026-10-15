#!/usr/bin/env python3
"""Apply the september-evaluation-analysis.pdf method set to any saved runs.

Reproduces, per benchmark: Wilson 95% intervals, paired bootstrap deltas
(20,000 resamples), discordant item counts, the completion audit (unclosed
</think>, >=10 <answer> tags), closure-gated accuracy, generation-length
statistics and minutes per 1,000 items.
"""
import json, math, random, re, sys
from pathlib import Path

random.seed(42)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def paired_bootstrap(a, b, n=20000):
    """95% interval on mean(a) - mean(b) over matched items."""
    idx = range(len(a))
    diffs = [a[i] - b[i] for i in idx]
    out = []
    for _ in range(n):
        s = sum(diffs[random.randrange(len(diffs))] for _ in idx)
        out.append(s / len(diffs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def load(path):
    d = json.load(open(path))
    recs = {}
    for r in d["records"]:
        c = r["candidates"][0]
        raw = c.get("raw_output") or ""
        recs[r["id"]] = {
            "correct": bool(c["correct"]),
            "closed": "</think>" in raw,
            "answer_tags": len(re.findall(r"<answer\s*>", raw, re.I)),
            "tokens": c.get("generated_tokens") or 0,
            "at_cap": bool(c.get("hit_token_limit")),
            "empty": not (c.get("final_response") or "").strip(),
            "level": (r.get("metadata") or {}).get("level"),
        }
    return d["summary"], recs


def report(files, cap):
    runs = {}
    for f in files:
        s, r = load(f)
        runs[s["model"]] = (s, r)

    print("=" * 96)
    print(f"ACCURACY, 95% Wilson intervals   (cap {cap} tokens)")
    print(f"{'model':48s} {'n':>5} {'score':>7} {'95% CI':>16}")
    for name, (s, r) in runs.items():
        n = len(r); k = sum(v["correct"] for v in r.values())
        lo, hi = wilson(k, n)
        print(f"{name:48s} {n:5d} {k/n:7.3f} [{lo:.3f}, {hi:.3f}]")

    names = list(runs)
    print()
    print("PAIRED COMPARISONS (matched item IDs, 20,000 bootstrap resamples)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            ids = sorted(set(runs[A][1]) & set(runs[B][1]))
            a = [int(runs[A][1][x]["correct"]) for x in ids]
            b = [int(runs[B][1][x]["correct"]) for x in ids]
            lo, hi = paired_bootstrap(a, b)
            gained = sum(1 for x in ids if runs[A][1][x]["correct"] and not runs[B][1][x]["correct"])
            lost = sum(1 for x in ids if runs[B][1][x]["correct"] and not runs[A][1][x]["correct"])
            n_d = gained + lost
            kk = min(gained, lost)
            p = min(1.0, sum(math.comb(n_d, t) for t in range(kk + 1)) * 2 / 2 ** n_d) if n_d else 1.0
            print(f"  {A[:34]:34s} - {B[:34]:34s}")
            print(f"    delta = {(sum(a)-sum(b))/len(ids)*100:+.2f} pp   95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp"
                  f"   newly-correct {gained} / lost {lost}   McNemar p = {p:.4f}")

    print()
    print("COMPLETION AUDIT")
    print(f"{'model':48s} {'noClose%':>9} {'>=10 tags%':>11} {'empty%':>8} {'atCap%':>8}")
    for name, (s, r) in runs.items():
        n = len(r)
        print(f"{name:48s} {100*sum(1 for v in r.values() if not v['closed'])/n:9.1f}"
              f" {100*sum(1 for v in r.values() if v['answer_tags']>=10)/n:11.1f}"
              f" {100*sum(1 for v in r.values() if v['empty'])/n:8.1f}"
              f" {100*sum(1 for v in r.values() if v['at_cap'])/n:8.1f}")

    print()
    print("CLOSURE-GATED ACCURACY (correct with a closing tag vs total)")
    print(f"{'model':48s} {'score':>7} {'gated':>7} {'credited unclosed':>18}")
    for name, (s, r) in runs.items():
        n = len(r)
        tot = sum(v["correct"] for v in r.values())
        gated = sum(1 for v in r.values() if v["correct"] and v["closed"])
        print(f"{name:48s} {tot/n:7.3f} {gated/n:7.3f} {tot-gated:18d}")

    print()
    print("GENERATION LENGTH AND COST")
    print(f"{'model':48s} {'mean':>7} {'median':>7} {'p90':>7} {'atCap%':>7} {'min/1k':>8}")
    for name, (s, r) in runs.items():
        t = sorted(v["tokens"] for v in r.values()); n = len(t)
        print(f"{name:48s} {sum(t)/n:7.0f} {t[n//2]:7d} {t[int(.9*n)]:7d}"
              f" {100*sum(1 for v in r.values() if v['at_cap'])/n:7.1f}"
              f" {s['elapsed_seconds']/60/n*1000:8.1f}")


if __name__ == "__main__":
    cap = sys.argv[1]
    report(sys.argv[2:], cap)
