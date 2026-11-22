"""Summarise scripts/eval_teacher_forced_kl.py output: per-region means and paired tests.

Every comparison is paired per trace (same text, same positions). Per-trace
means are token-weighted within the trace; CIs are bootstrap over traces.

Run: python scripts/analyze_teacher_forced.py eval_outputs/rd_newsae/eval_mod/teacher_forced/results.json
"""

import json
import random
import sys

STATS = ("kl1", "kl4", "ent", "nll", "p_eos", "p_close")
REGIONS = ("think", "close", "answer", "boxed")


def trace_mean(rec, model, region, stat):
    r = rec["stats"][model][region]
    return r[stat] / r["n"] if r["n"] else None


def boot(xs, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(b))
    mean = sum(xs) / n
    p = 2 * min(sum(m <= 0 for m in means), sum(m >= 0 for m in means)) / b
    return mean, means[int(0.025 * b)], means[int(0.975 * b)], max(p, 1 / b)


def main(path):
    results = json.load(open(path))["results"]
    models = list(results[0]["stats"])
    for domain in ("math", "natural_reasoning"):
        recs = [r for r in results if r["domain"] == domain]
        print(f"\n######## {domain}: {len(recs)} traces, "
              f"{sum(r['has_close'] for r in recs)} contain </think> within 2048 tokens")
        for region in REGIONS:
            rs = [r for r in recs if r["stats"]["teacher"][region]["n"]]
            if not rs:
                continue
            ntok = sum(r["stats"]["teacher"][region]["n"] for r in rs)
            print(f"\n== {region}: {len(rs)} traces, {ntok} positions (token-weighted means)")
            for stat in STATS:
                cells = []
                for m in models:
                    num = sum(r["stats"][m][region][stat] for r in rs)
                    den = sum(r["stats"][m][region]["n"] for r in rs)
                    vals = sorted(trace_mean(r, m, region, stat) for r in rs)
                    cells.append(f"{m}={num / den:.3e} (med {vals[len(vals) // 2]:.3e})")
                print(f"  {stat:8s} " + "  ".join(cells))
            # Paired contrasts, per-trace means.
            # RD students = every model other than teacher / lkd / base (rd, or rd_oct and rd_mod).
            rds = [m for m in models if m not in ("teacher", "lkd", "base")]
            contrasts = ([(m, "lkd") for m in rds] + [(a, b) for i, a in enumerate(rds) for b in rds[:i]]
                         + [(m, "teacher") for m in ["lkd"] + rds if m in models])
            for stat in ("kl1", "kl4", "ent", "nll", "p_eos", "p_close"):
                for a, b in contrasts:
                    if stat.startswith("kl") and b == "teacher":
                        continue  # KL to the teacher is already a distance from it
                    d = [trace_mean(r, a, region, stat) - trace_mean(r, b, region, stat) for r in rs]
                    mean, lo, hi, p = boot(d)
                    print(f"    {stat:8s} {a}-{b:8s} {mean:+.3e} [{lo:+.3e}, {hi:+.3e}] p={p:.4f}"
                          f"  ({sum(x > 0 for x in d)}/{len(d)} traces >0)")
            # Concentration: is any pooled mean carried by a few traces?
            for stat in ("kl1", "p_eos"):
              for m in rds:
                tot = [r["stats"][m][region][stat] for r in rs]
                s = sum(tot)
                if s:
                    print(f"    max single-trace share of {m} {stat} sum: {max(tot) / s:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
