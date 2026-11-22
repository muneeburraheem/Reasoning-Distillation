"""Exp 2 · Feature breakdown: which of the 196 RD features did RD's student gain on, and what are they?

Uses the probe phase of mi_llama_students.py (RUN_LOG 8aa): per-sequence ridge stats at h[11]
for LKD and RD on the 400 held-out test texts, and the teacher targets.

Per feature (live = >= 20 nonzero test tokens):
  r2_lkd, r2_rd, gain = r2_rd - r2_lkd with a bootstrap CI over test texts
  density           fraction of test tokens where the teacher feature is nonzero
  marker share      share of its activation mass on reasoning-word tokens (ReasonScore vocabulary),
                    on the token itself and on the token just before one
  rank              position in the ReasonScore ranking (0 = top)
  top contexts      the 6 strongest test activations, with 10 tokens of context
Summary: how concentrated the gain is (share carried by the top 10 / 25 features), Spearman of
gain with marker share, density, rank and r2_lkd (bootstrap over features), and the same gain
for the random-feature control set.

Run: python scripts/mi_llama_exp2_feature_breakdown.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mi_llama_students as M  # noqa: E402

OUT = M.OUT / "exp2"
MARKERS = {"wait", "hmm", "maybe", "perhaps", "but", "another", "therefore", "however"}
L = 10                                             # ridge index 10 = hidden_states[11]


def r2_per_feature(st):
    s = st.sum(0)
    sse, sy, sy2, n, nz = s
    return 1 - sse / (sy2 - sy ** 2 / n), nz


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M.STUDENTS["lkd"])
    rows = [json.loads(l) for l in (M.OUT / "texts.jsonl").open()]
    T = torch.load(M.OUT / "targets.pt")
    P = {s: torch.load(M.OUT / f"probe_{s}.pt", weights_only=False) for s in ("lkd", "rd")}
    te = P["rd"]["test_rows"]
    feats = T["rd"] + T["rand"]
    rank = {f: i for i, f in enumerate(torch.load(str(M.RANK_PT))["sorted_indices"][:2000].tolist())}

    stl, str_ = P["lkd"]["ridge"][:, L], P["rd"]["ridge"][:, L]       # [nseq, 5, 392]
    r2l, nz = r2_per_feature(stl)
    r2r, _ = r2_per_feature(str_)
    live = (nz >= 20).numpy()
    gain = (r2r - r2l).numpy()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        i = torch.from_numpy(rng.integers(0, len(te), len(te)))
        boots.append((r2_per_feature(str_[i])[0] - r2_per_feature(stl[i])[0]).numpy())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)

    # token-level descriptors on the test texts
    mk_ids = {i for i in range(len(tok)) if tok.decode([i]).strip().lower() in MARKERS}
    mass, mass_on, mass_pre, ntok, nonz = (np.zeros(len(feats)) for _ in range(5))
    tops = [[] for _ in feats]
    for i in te:
        ids = tok(rows[i]["text"], truncation=True, max_length=M.MAXLEN)["input_ids"][1:]
        y = T["Y"][i].numpy()                                         # [tokens, 392], BOS dropped
        on = np.array([t in mk_ids for t in ids])
        pre = np.array([ids[t + 1] in mk_ids if t + 1 < len(ids) else False for t in range(len(ids))])
        mass += y.sum(0); mass_on += y[on].sum(0); mass_pre += y[pre].sum(0)
        ntok += len(ids); nonz += (y > 0).sum(0)
        for k in range(len(feats)):
            t = int(y[:, k].argmax())
            if y[t, k] > 0:
                tops[k].append((float(y[t, k]), i, t))
    for k in range(len(feats)):
        tops[k] = sorted(tops[k], reverse=True)[:6]

    def context(i, t):
        ids = tok(rows[i]["text"], truncation=True, max_length=M.MAXLEN)["input_ids"][1:]
        a, b = max(0, t - 10), min(len(ids), t + 4)
        return (tok.decode(ids[a:t]) + "[[" + tok.decode(ids[t:t + 1]) + "]]" + tok.decode(ids[t + 1:b])).replace("\n", "\\n")

    recs = []
    for k, f in enumerate(feats):
        recs.append({"feature": f, "set": "rd" if k < M.N_SET else "rand", "rank": rank.get(f),
                     "live": bool(live[k]), "r2_lkd": float(r2l[k]), "r2_rd": float(r2r[k]),
                     "gain": float(gain[k]), "gain_lo": float(lo[k]), "gain_hi": float(hi[k]),
                     "density": float(nonz[k] / ntok[k]),
                     "marker_share_on": float(mass_on[k] / max(mass[k], 1e-9)),
                     "marker_share_before": float(mass_pre[k] / max(mass[k], 1e-9)),
                     "top_contexts": [{"act": a, "row": i, "context": context(i, t)} for a, i, t in tops[k]]})
    (OUT / "features.json").write_text(json.dumps(recs, indent=1))

    lines = ["# Exp 2 · Feature breakdown (h[11], ridge probe, 400 held-out test texts)"]
    base_on = np.mean([t in mk_ids for i in te for t in tok(rows[i]["text"], truncation=True,
                                                               max_length=M.MAXLEN)["input_ids"][1:]])
    lines.append(f"reasoning-word tokens are {100 * base_on:.2f}% of test tokens (marker share of a feature that ignores them)")
    for s in ("rd", "rand"):
        R = [r for r in recs if r["set"] == s and r["live"]]
        g = np.array([r["gain"] for r in R])
        pos = np.sort(g[g > 0])[::-1]
        sig = sum(r["gain_lo"] > 0 for r in R)
        sneg = sum(r["gain_hi"] < 0 for r in R)
        lines.append(f"\n## {s} set: {len(R)} live features; mean gain {g.mean():+.4f}, median {np.median(g):+.4f}; "
                     f"CI above 0: {sig}, below 0: {sneg}; top-10 carry {pos[:10].sum() / max(pos.sum(), 1e-9):.0%} "
                     f"and top-25 {pos[:25].sum() / max(pos.sum(), 1e-9):.0%} of the positive gain")
        for key in ("marker_share_on", "marker_share_before", "density", "rank", "r2_lkd"):
            x = np.array([r[key] if r[key] is not None else np.nan for r in R], float)
            ok = ~np.isnan(x)
            rho = spearmanr(x[ok], g[ok]).statistic
            bs = []
            for _ in range(1000):
                j = rng.integers(0, ok.sum(), ok.sum())
                bs.append(spearmanr(x[ok][j], g[ok][j]).statistic)
            lines.append(f"  Spearman(gain, {key:20s}) {rho:+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]")
    lines.append("\n## RD features by gain (live only): feature, rank, R2 LKD -> RD, gain [CI], density, marker share on / before")
    R = sorted([r for r in recs if r["set"] == "rd" and r["live"]], key=lambda r: -r["gain"])
    for r in R[:25] + [None] + R[-10:]:
        if r is None:
            lines.append("  ...")
            continue
        lines.append(f"  f{r['feature']:<6d} #{r['rank']:<4d} {r['r2_lkd']:+.3f} -> {r['r2_rd']:+.3f}  {r['gain']:+.4f} "
                     f"[{r['gain_lo']:+.4f},{r['gain_hi']:+.4f}]  dens {r['density']:.4f}  mk {r['marker_share_on']:.2f}/{r['marker_share_before']:.2f}")
    lines.append("\n## top contexts of the 12 highest-gain RD features ([[token]] = strongest activation)")
    for r in R[:12]:
        lines.append(f"  f{r['feature']} (gain {r['gain']:+.4f}, rank #{r['rank']})")
        for c in r["top_contexts"][:4]:
            lines.append(f"     {c['act']:7.2f}  {c['context']}")
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
