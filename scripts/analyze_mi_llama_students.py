"""Analysis of scripts/mi_llama_students.py (probe phase).

R^2 per feature from summed per-sequence stats (SSE, sum y, sum y^2, n, nonzero) over test
sequences; features with < 20 nonzero test tokens are dropped. Reported: pooled R^2 (sum SSE /
sum SST over features) and median per-feature R^2; bootstrap over test sequences, paired
across students (same sequences). The rand set is the density-matched control.

Run: python scripts/analyze_mi_llama_students.py
"""
from pathlib import Path

import numpy as np
import torch

D = Path("eval_outputs/mi_students/llama")
N_SET, B = 196, 1000
P = {s: torch.load(D / f"probe_{s}.pt", weights_only=False) for s in ("base", "lkd", "rd") if (D / f"probe_{s}.pt").exists()}
SETS = {"rd": slice(0, N_SET), "rand": slice(N_SET, 2 * N_SET)}


def r2(st, live=None):
    """st [nseq, 5, k] -> pooled R^2, median per-feature R^2, live mask."""
    s = st.sum(0)
    sse, sy, sy2, n, nz = s
    sst = sy2 - sy ** 2 / n
    if live is None:
        live = nz >= 20
    per = 1 - sse[live] / sst[live]
    return float(1 - sse[live].sum() / sst[live].sum()), float(per.median()), live


def boot(fa, fb, sa, sb, seed=0):
    rng = np.random.default_rng(seed)
    n = sa.shape[0]
    pa, _, live = fa(sa)
    pb, _, _ = fb(sb, live)
    d = pa - pb
    bs = []
    for _ in range(B):
        i = torch.from_numpy(rng.integers(0, n, n))
        bs.append(fa(sa[i], live)[0] - fb(sb[i], live)[0])
    bs = np.array(bs)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return d, np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / B)


def ridge(s, l, sl):
    return P[s]["ridge"][:, l, :, sl]


print("# ridge probe, held-out R^2 (pooled | median per feature), per layer; hidden_states index")
print("  layer   " + "  ".join(f"{s:>5s}-{t:<4s}      " for t in SETS for s in P))
for l in range(16):
    row = []
    for t, sl in SETS.items():
        for s in P:
            pr, md, _ = r2(ridge(s, l, sl))
            row.append(f"{pr:+.3f} {md:+.3f}    ")
    print(f"  h[{l + 1:2d}]{'*' if l + 1 == 11 else ' '}  " + "  ".join(row))
print("  * = guided layer (student block 10 output)")

print("\n# paired differences in pooled R^2 (bootstrap over test sequences)")
f = lambda st, live=None: r2(st, live)  # noqa: E731
for l in (5, 8, 10, 12, 15):
    for t, sl in SETS.items():
        for a, b in (("rd", "lkd"), ("lkd", "base"), ("rd", "base")):
            if a in P and b in P:
                d, lo, hi, p = boot(f, f, ridge(a, l, sl), ridge(b, l, sl))
                print(f"  h[{l + 1:2d}] {t:4s} {a}-{b:4s} {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
# difference in differences at the guided layer: (rd-lkd on rd set) - (rd-lkd on rand set)
if "rd" in P and "lkd" in P:
    l = 10
    rng = np.random.default_rng(1)
    n = P["rd"]["ridge"].shape[0]
    def dd(i):
        out = []
        for t, sl in SETS.items():
            _, _, live = r2(ridge("lkd", l, sl))
            out.append(r2(ridge("rd", l, sl)[i], live)[0] - r2(ridge("lkd", l, sl)[i], live)[0])
        return out[0] - out[1]
    d0 = dd(torch.arange(n))
    bs = np.array([dd(torch.from_numpy(rng.integers(0, n, n))) for _ in range(B)])
    print(f"  h[11] (rd-lkd on rd set) - (rd-lkd on rand set) {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},"
          f"{np.percentile(bs, 97.5):+.4f}] p={max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B):.3f}")

print("\n# pipeline readout at h[11] (aligner + frozen SAE), held-out R^2 (pooled | median)")
for s in P:
    for t in SETS:
        pr, md, _ = r2(P[s]["pipe"][t])
        st = P[s]["pipe"][t].sum(0)
        print(f"  {s:4s} fresh aligner {t:4s} {pr:+.4f} {md:+.4f}")
    if P[s]["own"] is not None:
        pr, md, _ = r2(P[s]["own"])
        print(f"  {s:4s} RD's trained aligner rd   {pr:+.4f} {md:+.4f}")

if (D / "pipe.pt").exists():
    Q = torch.load(D / "pipe.pt", weights_only=False)
    print("\n# pipeline readout, least-squares start (init) then RD's loss (trained); held-out R^2 (pooled | median)")
    for stage in ("init", "trained"):
        for t in SETS:
            print(f"  {stage:7s} {t:4s} " + "   ".join(f"{s} {r2(Q[f'{s}|{t}|{stage}'])[0]:+.4f} {r2(Q[f'{s}|{t}|{stage}'])[1]:+.4f}"
                                                  for s in ("base", "lkd", "rd")))
    if "rd" in P:
        live = r2(Q["lkd|rd|trained"])[2]
        print(f"  RD's own trained aligner on RD, same live features: {r2(P['rd']['own'], live)[0]:+.4f}")
    for t in SETS:
        for a, b in (("rd", "lkd"), ("lkd", "base")):
            d, lo, hi, p = boot(f, f, Q[f"{a}|{t}|trained"], Q[f"{b}|{t}|trained"])
            print(f"  trained {t:4s} {a}-{b:4s} {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
    rng = np.random.default_rng(2)
    n = Q["rd|rd|trained"].shape[0]
    lv = {t: r2(Q[f"lkd|{t}|trained"])[2] for t in SETS}
    def dd(i):
        return sum(sg * (r2(Q[f"rd|{t}|trained"][i], lv[t])[0] - r2(Q[f"lkd|{t}|trained"][i], lv[t])[0])
                   for t, sg in (("rd", 1), ("rand", -1)))
    d0 = dd(torch.arange(n))
    bs = np.array([dd(torch.from_numpy(rng.integers(0, n, n))) for _ in range(B)])
    print(f"  trained (rd-lkd on rd set) - (rd-lkd on rand set) {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},"
          f"{np.percentile(bs, 97.5):+.4f}] p={max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B):.3f}")

if (D / "rd_aligner_cross.pt").exists():
    X = torch.load(D / "rd_aligner_cross.pt", weights_only=False)
    print("\n# RD's trained aligner (barely moved from its Xavier init) applied to every student, rd set, h[11]")
    live = r2(X["lkd"]["rd_aligner"])[2] | r2(X["rd"]["rd_aligner"])[2]
    for s in X:
        for k in ("rd_aligner", "random_proj"):
            st = X[s][k]
            pred_on = 1 - float(st[:, 0].sum() / st[:, 2].sum())
            print(f"  {s:4s} {k:11s} pooled R^2 {r2(st, live)[0]:+.4f}  median {r2(st, live)[1]:+.4f}")
    for a, b in (("rd", "lkd"), ("rd", "base")):
        d, lo, hi, p = boot(f, f, X[a]["rd_aligner"], X[b]["rd_aligner"])
        print(f"  rd_aligner {a}-{b:4s} {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")


def boot1(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)


if (D / "tf.jsonl").exists():
    import json
    TF = {json.loads(l)["student"]: json.loads(l)["records"] for l in (D / "tf.jsonl").open()}
    print("\n# teacher-forced mean-ablation at h[11], 400 held-out rows; per-row change vs clean, bootstrap over rows")
    print("  metrics: pm_mk = d P(marker class) at marker targets; lp_mk = d log p(actual marker); kl at all positions;"
          " nll_think = d NLL inside <think>; size = |dh|/|h|")
    for s, R in TF.items():
        print(f"  ## {s}")
        for k in (8, 32, 128):
            for t in ("rd", "rand", "randdir", "aligner"):
                c = f"{t}|k{k}"
                ok = [r for r in R if r["n_mk"] > 0]
                d = {m: [r["cond"][c][m] - r["cond"]["clean"][m] for r in ok] for m in ("pm_mk", "lp_mk", "nll_think")}
                kl = [r["cond"][c]["kl"] for r in R]
                sz = np.mean([r["cond"][c]["size"] for r in R])
                print(f"    {c:12s} pm_mk {np.mean(d['pm_mk']):+.4f} lp_mk {np.mean(d['lp_mk']):+.4f} "
                      f"nll_think {np.mean(d['nll_think']):+.4f} kl {np.mean(kl):.4f} size {sz:.3f}")
            for a, b in (("rd", "rand"), ("rd", "randdir"), ("aligner", "randdir")):
                ok = [r for r in R if r["n_mk"] > 0]
                for m in ("pm_mk", "kl"):
                    x = [r["cond"][f"{a}|k{k}"][m] - r["cond"][f"{b}|k{k}"][m] for r in ok]
                    mm, lo, hi, p = boot1(x)
                    print(f"      k{k} {a}-{b:7s} {m:5s} {mm:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
    if "rd" in TF and "lkd" in TF:
        print("  ## does ablating its own RD-readout subspace hurt RD more than LKD? (paired rows; effect = cond - clean)")
        A = {r["row"]: r for r in TF["rd"]}
        Bb = {r["row"]: r for r in TF["lkd"]}
        rows_ = [i for i in A if A[i]["n_mk"] > 0]
        for k in (8, 32, 128):
            for t in ("rd", "aligner"):
                for m in ("pm_mk", "kl"):
                    if m == "kl":
                        x = [(A[i]["cond"][f"{t}|k{k}"][m] - A[i]["cond"][f"randdir|k{k}"][m])
                             - (Bb[i]["cond"][f"{t}|k{k}"][m] - Bb[i]["cond"][f"randdir|k{k}"][m]) for i in rows_]
                    else:
                        x = [((A[i]["cond"][f"{t}|k{k}"][m] - A[i]["cond"]["clean"][m]) - (A[i]["cond"][f"randdir|k{k}"][m] - A[i]["cond"]["clean"][m]))
                             - ((Bb[i]["cond"][f"{t}|k{k}"][m] - Bb[i]["cond"]["clean"][m]) - (Bb[i]["cond"][f"randdir|k{k}"][m] - Bb[i]["cond"]["clean"][m]))
                             for i in rows_]
                    mm, lo, hi, p = boot1(x)
                    print(f"    k{k} {t:7s} {m:5s} (RD - LKD) of (cond - randdir) {mm:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")

if (D / "gen.jsonl").exists():
    import json
    G = {json.loads(l)["cond"]: json.loads(l)["records"] for l in (D / "gen.jsonl").open()}
    print("\n# GSM8K generation under mean-ablation at h[11] (500 q x 2, cap 4096); per-question means, paired")
    per = {c: {} for c in G}
    for c, R in G.items():
        q = {}
        for r in R:
            q.setdefault(r["id"], []).append(r)
        per[c] = {i: {m: np.mean([float(x[m]) for x in v]) for m in ("strict", "lenient", "tokens", "loop", "hit_limit")} for i, v in q.items()}
        print(f"  {c:20s} " + " ".join(f"{m} {np.mean([v[m] for v in per[c].values()]):.3f}" for m in ("strict", "lenient", "tokens", "loop", "hit_limit")))
    for s in ("lkd", "rd"):
        for a, b in (("rd|k32", "none"), ("rand|k32", "none"), ("randdir|k32", "none"), ("aligner|k32", "none"),
                     ("rd|k32", "rand|k32"), ("rd|k32", "randdir|k32"), ("aligner|k32", "randdir|k32")):
            ca, cb = f"{s}|{a}", f"{s}|{b}"
            if ca in per and cb in per:
                ids = sorted(set(per[ca]) & set(per[cb]))
                for m in ("strict", "lenient", "tokens"):
                    sc = 1 if m == "tokens" else 100
                    mm, lo, hi, p = boot1([per[ca][i][m] - per[cb][i][m] for i in ids])
                    print(f"  {s} {a:11s} - {b:11s} {m:8s} {sc*mm:+7.1f} [{sc*lo:+.1f}, {sc*hi:+.1f}] p={p:.3f}")
