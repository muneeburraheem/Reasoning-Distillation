"""Analysis of scripts/mi_steering.py (feature steering with matched random controls).

Grid (greedy, 1024 tokens, 100 MATH-500 prompts): per condition and prompt
  in-vocab   reasoning-marker rate per 1k words (the ReasonScore vocabulary: hmm/maybe/wait/perhaps/
             let me/therefore/however/but/another - what the features were *selected* on)
  out-vocab  markers not in that vocabulary (alternatively/actually/hold on/double-check/verify/
             let's check/I think) - a lexical-only feature set should not move these
  so         rate of "so" alone (the steered direction's degenerate mode), reported apart
  tokens, </think> rate, loop (8-gram rep >= .85), NLL under the unsteered model.
Each feat (k, r) is compared, paired per prompt with bootstrap over prompts, to
  base, randfeat (mean of seeds 0/1, density-matched features) and randdir at the same (k, r).
Marker rates are also given on prompts where neither condition loops (a loop inflates or zeroes counts).

Full (T 0.6, 4096 tokens, 200 prompts x 2): strict / lenient accuracy, tokens, loop, cap, paired per
question vs base and vs the random controls at the same r.

Run: python scripts/analyze_mi_steering.py [subject]
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "llama_teacher"
D = Path("eval_outputs/mi_steering") / SUBJ
IN_VOCAB = r"\b(hmm|maybe|wait|perhaps|let me|therefore|however|but|another)\b"
OUT_VOCAB = r"\b(alternatively|actually|hold on|double[- ]check|verify|let's check|i think)\b"
SO = r"\bso\b"  # separate: the feature direction's degenerate mode is "so, so, so" (k=15, r>=+0.2)


def boot(d, B=4000):
    d = [x for x in d if x is not None and not np.isnan(x)]
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(d)
    rng = random.Random(0)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B
    return sum(d) / len(d), m[int(.025 * B)], m[int(.975 * B)], min(1.0, max(p, 1 / B)), len(d)


def rate(pat, text):
    w = len(text.split())
    return 1000 * len(re.findall(pat, text, flags=re.I)) / w if w else 0.0


def feats(x):
    loop = x["rep"] >= .85
    return {"in": rate(IN_VOCAB, x["text"]), "out": rate(OUT_VOCAB, x["text"]), "so": rate(SO, x["text"]), "tokens": x["tokens"],
            "closed": float(x["closed_think"]), "loop": float(loop), "nll": x["nll"], "_loop": loop}


def load(name):
    rows = {}
    if (D / name).exists():
        for l in (D / name).open():
            r = json.loads(l)
            rows[r["cond"]] = r
    return rows


def fmt(b):
    m, lo, hi, p, n = b
    return f"{m:+7.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}" if n >= 5 else "      n/a"


def grid():
    G = load("grid.jsonl")
    if not G:
        return
    P = {c: {x["id"]: feats(x) for x in r["records"]} for c, r in G.items()}
    base = P["base|k0|s0|r+0.00"]
    ks = sorted({r["k"] for r in G.values() if r["kind"] == "feat"})
    rs = sorted({r["r"] for r in G.values() if r["kind"] == "feat"})
    print(f"# {SUBJ} grid: {len(G)} conditions")
    b = {m: np.mean([v[m] for v in base.values() if v[m] is not None]) for m in ("in", "out", "so", "tokens", "closed", "loop", "nll")}
    print("base  " + "  ".join(f"{m} {v:.3f}" for m, v in b.items()))
    for k in ks:
        print(f"\n## k={k}   (feat level | feat-base | feat-randfeat | feat-randdir), paired per prompt")
        for r in rs:
            f = P.get(f"feat|k{k}|s0|r{r:+.2f}")
            if f is None:
                continue
            rf = [P.get(f"randfeat|k{k}|s{s}|r{r:+.2f}") for s in (0, 1)]
            rf = [x for x in rf if x]
            rd = P.get(f"randdir|k{k}|s0|r{r:+.2f}")
            ge = G[f"feat|k{k}|s0|r{r:+.2f}"]["gamma_equiv"]
            print(f"  r={r:+.2f} (paper gamma~{ge:.2f})" if ge else f"  r={r:+.2f}")
            for m in ("in", "out", "so", "tokens", "closed", "loop", "nll"):
                ids = list(f)
                lvl = np.mean([f[i][m] for i in ids if f[i][m] is not None])
                out = [f"    {m:6s} {lvl:8.3f}", fmt(boot([f[i][m] - base[i][m] for i in ids if f[i][m] is not None]))]
                out.append(fmt(boot([f[i][m] - np.mean([c[i][m] for c in rf]) for i in ids
                                     if f[i][m] is not None and all(c[i][m] is not None for c in rf)])) if rf else "      pending")
                out.append(fmt(boot([f[i][m] - rd[i][m] for i in ids
                                     if f[i][m] is not None and rd[i][m] is not None])) if rd else "      pending")
                print(" | ".join(out))
            # marker rates on prompts where nothing compared loops
            for m in ("in", "out", "so"):
                ok = [i for i in f if not f[i]["_loop"] and not base[i]["_loop"]
                      and all(not c[i]["_loop"] for c in rf) and (not rd or not rd[i]["_loop"])]
                out = [f"    {m}|noloop n={len(ok):3d}",
                       fmt(boot([f[i][m] - base[i][m] for i in ok]))]
                out.append(fmt(boot([f[i][m] - np.mean([c[i][m] for c in rf]) for i in ok])) if rf else "      pending")
                out.append(fmt(boot([f[i][m] - rd[i][m] for i in ok])) if rd else "      pending")
                print(" | ".join(out))


def full():
    F = load("full.jsonl")
    if not F:
        return
    print(f"\n# {SUBJ} full phase: {len(F)} conditions")
    per = {}
    for c, r in F.items():
        q = {}
        for x in r["records"]:
            q.setdefault(x["id"], []).append(x)
        per[c] = {i: {m: np.mean([float(x[m]) for x in v]) for m in ("strict", "lenient", "tokens", "loop", "hit_limit", "closed_think")}
                  for i, v in q.items()}
    base = per.get("base|k0|s0|r+0.00")
    for c in sorted(per):
        a = per[c]
        line = f"  {c:28s} " + " ".join(f"{m} {np.mean([v[m] for v in a.values()]):.3f}"
                                        for m in ("strict", "lenient", "tokens", "loop", "hit_limit", "closed_think"))
        print(line)
    for c in sorted(per):
        if not c.startswith("feat"):
            continue
        r = c.split("|r")[1]
        for ref in ("base|k0|s0|r+0.00", f"randfeat|k15|s0|r{r}", f"randdir|k15|s0|r{r}"):
            if ref not in per:
                continue
            for m in ("strict", "lenient", "tokens", "loop"):
                ids = sorted(set(per[c]) & set(per[ref]))
                mm, lo, hi, p, n = boot([per[c][i][m] - per[ref][i][m] for i in ids])
                sc = 100 if m in ("strict", "lenient", "loop") else 1
                print(f"  {c} - {ref:24s} {m:8s} {sc*mm:+7.1f} [{sc*lo:+.1f}, {sc*hi:+.1f}] p={p:.4f} n={n}")


if __name__ == "__main__":
    grid()
    full()
