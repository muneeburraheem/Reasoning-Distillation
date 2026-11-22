"""Accuracy difference by Omni-MATH difficulty level, and its linear trend (RUN_LOG 8q; new-SAE suite).

Paired per question, 1 sample each, strict correctness. Bootstrap over questions (5,000) for the slope
of the per-difficulty mean difference. Appended to eval_outputs/rd_newsae/eval_mod/hard_math/C/analysis.txt.

Run: python scripts/analyze_difficulty_trend.py <label=rd_results.json> LKD=<lkd_results.json>
"""
import json
import sys

import numpy as np


def main():
    (lab, mp), (_, lp) = (a.split("=", 1) for a in sys.argv[1:3])
    ld = lambda p: {r['id']: r for r in json.load(open(p))['records']}
    M = ld(mp); L = ld(lp)
    ids = sorted(M); d = np.array([int(M[i]['metadata']['difficulty']) for i in ids])
    diff = np.array([M[i]['candidates'][0]['correct'] - L[i]['candidates'][0]['correct'] for i in ids], float)
    print(f"\nDifficulty trend, {lab} - LKD (strict):")
    for k in sorted(set(d)):
        s = d == k
        print(f"  difficulty {k}: n={s.sum():3d} LKD {np.mean([L[i]['candidates'][0]['correct'] for i, x in zip(ids, s) if x]):.3f} "
              f"{lab} {np.mean([M[i]['candidates'][0]['correct'] for i, x in zip(ids, s) if x]):.3f} diff {100 * diff[s].mean():+.1f} pp")

    def slope(idx):
        ks = sorted(set(d[idx])); m = [diff[idx][d[idx] == k].mean() for k in ks]
        return np.polyfit(ks, m, 1)[0]
    rng = np.random.default_rng(0); n = len(ids)
    bs = sorted(slope(rng.integers(0, n, n)) for _ in range(5000)); s0 = slope(np.arange(n))
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    print(f"  trend per difficulty step: {100 * s0:+.2f} pp [{100 * bs[125]:+.2f}, {100 * bs[4875]:+.2f}] p={max(p, 2e-4):.3f}")


if __name__ == "__main__":
    main()
