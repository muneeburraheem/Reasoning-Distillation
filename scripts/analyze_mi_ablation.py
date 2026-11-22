"""Analysis of scripts/mi_ablation.py.

patch   per position, change vs the cached clean row of the same position; averaged per trace,
        bootstrap over traces (positions within a trace are not independent).
          necessity (marker positions, ablate): d log p(actual marker), d P(marker class), KL
          sufficiency (control positions, patch in from a marker position): d P(marker class), KL
        rd vs the mean of its 5 density-matched random sets and vs the same-size random direction;
        attr (positive control) only on split B (it was selected on split A). |delta|/|h| shown so
        the size of each intervention is visible.
sgrid   same tables as analyze_mi_steering (markers, length, </think>, loops, NLL) vs the unsteered
        greedy run of mi_steering (same prompts and settings).
sfull   strict / lenient accuracy paired per question vs base and vs the matched random set.

Run: python scripts/analyze_mi_ablation.py [subject]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_mi_steering as am  # noqa: E402

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "llama_teacher"
D = Path("eval_outputs/mi_ablation") / SUBJ
KS = (15, 50, 0)
SEEDS = range(5)


def fmt(b, sc=1.0):
    m, lo, hi, p, n = b
    return f"{sc*m:+8.4f} [{sc*lo:+.4f},{sc*hi:+.4f}] p={p:.3f}" if n >= 5 else "n/a"


def patch():
    if not (D / "patch.jsonl").exists():
        return
    T = [json.loads(l) for l in (D / "patch.jsonl").open()]
    print(f"# {SUBJ} patching: {len(T)} traces")
    s = json.loads((D / "attribution_summary.json").read_text())
    for k in KS:
        r = s[f"rd|k{k}"]
        print(f"  attribution rank, rd k{k}: mean attr {r['mean_attr']:+.2e} (random sets {np.mean(r['rand_mean_attr']):+.2e}, "
              f"max {max(r['rand_mean_attr']):+.2e}); median rank {r['median_rank']:.0f}/65536; in top-200 {r['in_top200']} "
              f"(random {r['rand_in_top200']})")
    print(f"  attr top-50 overlap with the ReasonScore set: {s['attr_top50_rd_overlap']}")

    def per_trace(kind, cond, metric, split=None):
        out = {}
        for tr in T:
            if split and tr["split"] != split:
                continue
            P = [p for p in tr["positions"] if p["kind"] == kind]
            if not P:
                continue
            vals = []
            for p in P:
                if isinstance(cond, list):                  # mean over random seeds
                    v = np.mean([val(p, x, metric) for x in cond])
                else:
                    v = val(p, cond, metric)
                vals.append(v)
            out[(tr["id"], tr["sample"])] = np.mean(vals)
        return out

    def val(p, c, metric):
        x, cl = p["cond"][c], p["cond"]["clean"]
        if metric == "kl":
            return x["kl"]
        if metric == "size":
            return x["dnorm"] / p["hnorm"]
        return x[metric] - cl[metric]

    for kind, label, metrics in (("mk", "NECESSITY: ablate at marker positions", ("lp", "pm", "kl", "size")),
                                 ("ct", "SUFFICIENCY: patch in at control positions", ("pm", "kl", "size"))):
        print(f"\n## {label}  (level per condition; rd - rand(mean of 5); rd - randdir), bootstrap over traces")
        for k in KS:
            rd, rdir, rand = f"rd|k{k}", f"randdir|k{k}", [f"rand|k{k}|s{s}" for s in SEEDS]
            print(f"  k={k or 'all'}")
            for m in metrics:
                a, b, c = per_trace(kind, rd, m), per_trace(kind, rand, m), per_trace(kind, rdir, m)
                ids = sorted(a)
                print(f"    {m:4s} rd {np.mean([a[i] for i in ids]):+.4f} rand {np.mean([b[i] for i in ids]):+.4f} "
                      f"randdir {np.mean([c[i] for i in ids]):+.4f} | rd-rand {fmt(am.boot([a[i] - b[i] for i in ids]))} "
                      f"| rd-randdir {fmt(am.boot([a[i] - c[i] for i in ids]))}")
            # positive control on held-out split B
            at = f"attr|k{k}"
            for m in metrics:
                a, b, c = per_trace(kind, at, m, "B"), per_trace(kind, rand, m, "B"), per_trace(kind, rd, m, "B")
                ids = sorted(a)
                print(f"    {m:4s} [split B] attr {np.mean([a[i] for i in ids]):+.4f} | attr-rand "
                      f"{fmt(am.boot([a[i] - b[i] for i in ids]))} | attr-rd {fmt(am.boot([a[i] - c[i] for i in ids]))}")


def sgrid():
    if not (D / "sgrid.jsonl").exists():
        return
    am.D = D
    G = am.load("sgrid.jsonl")
    am.D = Path("eval_outputs/mi_steering") / SUBJ
    B = am.load("grid.jsonl")["base|k0|s0|r+0.00"]
    base = {x["id"]: am.feats(x) for x in B["records"]}
    P = {c: {x["id"]: am.feats(x) for x in r["records"]} for c, r in G.items()}
    print(f"\n# {SUBJ} suppression grid (greedy 1024): level | cond - base | rd - rand(mean s0,s1)")
    for k in KS:
        for w in ("rd", "rand|s0", "rand|s1", "attr"):
            c = f"{w.split('|')[0]}|k{k}" + (f"|{w.split('|')[1]}" if "|" in w else "")
            if c not in P:
                continue
            print(f"  {c}")
            for m in ("in", "out", "so", "tokens", "closed", "loop", "nll"):
                f = P[c]
                ids = [i for i in f if f[i][m] is not None and base[i][m] is not None]
                line = f"    {m:6s} {np.mean([f[i][m] for i in ids]):8.3f} | {am.fmt(am.boot([f[i][m] - base[i][m] for i in ids]))}"
                if w in ("rd", "attr"):
                    rr = [P.get(f"rand|k{k}|s{s}") for s in (0, 1)]
                    rr = [x for x in rr if x]
                    if rr:
                        line += " | " + am.fmt(am.boot([f[i][m] - np.mean([x[i][m] for x in rr]) for i in ids
                                                        if all(x[i][m] is not None for x in rr)]))
                print(line)


def sfull():
    if not (D / "sfull.jsonl").exists():
        return
    F = {json.loads(l)["cond"]: json.loads(l) for l in (D / "sfull.jsonl").open()}
    per = {}
    for c, r in F.items():
        q = {}
        for x in r["records"]:
            q.setdefault(x["id"], []).append(x)
        per[c] = {i: {m: np.mean([float(x[m]) for x in v]) for m in ("strict", "lenient", "tokens", "loop", "hit_limit")}
                  for i, v in q.items()}
    print(f"\n# {SUBJ} suppression full (200 q x 2, 4096)")
    for c in per:
        print(f"  {c:24s} " + " ".join(f"{m} {np.mean([v[m] for v in per[c].values()]):.3f}"
                                       for m in ("strict", "lenient", "tokens", "loop", "hit_limit")))
    base = next(c for c in per if c.startswith("base|"))
    pairs = [(c, base) for c in per if c != base] + [("rd|k0", "rand|k0|s0"), ("rd|k50", "rand|k50|s0"),
                                                      ("attr|k50", "rand|k50|s0"), ("attr|k50", "rd|k50")]
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        ids = sorted(set(per[a]) & set(per[b]))
        for m in ("strict", "lenient", "tokens"):
            sc = 100 if m != "tokens" else 1
            mm, lo, hi, p, n = am.boot([per[a][i][m] - per[b][i][m] for i in ids])
            print(f"  {a:12s} - {b:18s} {m:8s} {sc*mm:+7.1f} [{sc*lo:+.1f}, {sc*hi:+.1f}] p={p:.4f}")


if __name__ == "__main__":
    patch()
    sgrid()
    sfull()
