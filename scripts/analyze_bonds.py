"""Families A-C and E of RUN_LOG 8n: CoT bond structure, geometry and entropy dynamics.

Inputs: eval_outputs/rd_newsae/eval_mod/bonds/{steps.json, embeddings.npy} (scripts/eval_bond_annotate.py)
and eval_outputs/rd_newsae/eval_mod/answer_entropy/tokens_<model>.npz (scripts/eval_answer_entropy.py).
Models: LKD, RD_new (= October RD) and RD_mod (new-SAE RD, alpha 0.15). The first-RD ("RD_old") version of
this analysis is archived with its outputs (docs/REPO_MAP.md).
One trace per question per model; RD vs LKD paired per question, bootstrap over
questions, Holm-adjusted within each family. All pre-registered measures are
printed whether or not they favour RD.

Run: python scripts/analyze_bonds.py [--bonds DIR --ent DIR]  (Llama arm: scripts/eval_llama_bonds.py)
"""

import argparse
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BONDS = Path("eval_outputs/rd_newsae/eval_mod/bonds")
ENT = Path("eval_outputs/rd_newsae/eval_mod/answer_entropy")
L = ["normal operation", "deep reasoning", "self-reflection", "exploration"]
SHORT = {"normal operation": "N", "deep reasoning": "D", "self-reflection": "R", "exploration": "E"}
REFLECT_KW = re.compile(r"\b(wait|hmm|let me (check|verify|double|re)|double[- ]check|mistake|actually|hold on|re-?check|verify|recheck)\b", re.I)
EXPLORE_KW = re.compile(r"\b(alternatively|maybe|perhaps|what if|another (way|approach|method)|suppose|could it be|or maybe)\b", re.I)


def boot(d, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def step_spans(thinking):
    """Character spans matching bond_annotate.segment (same split rule)."""
    spans, pos = [], 0
    for m in re.finditer(r"\n\n|\n|(?<=\.) ", thinking + "\n"):
        piece = thinking[pos:m.start()]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            spans.append((pos + lead, pos + lead + len(piece.strip())))
        pos = m.end()
    return spans


def main():
    global BONDS, ENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonds", default=str(BONDS))
    ap.add_argument("--ent", default=str(ENT))
    a = ap.parse_args()
    BONDS, ENT = Path(a.bonds), Path(a.ent)
    steps = json.load(open(BONDS / "steps.json"))
    emb = np.load(BONDS / "embeddings.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    traces = defaultdict(list)
    for i, r in enumerate(steps):
        traces[(r["model"], r["qid"])].append((r, i))
    models = sorted({r["model"] for r in steps})

    # Annotator validation: agreement with keyword heuristic
    kw = Counter()
    for r in steps:
        k = "R" if REFLECT_KW.search(r["text"]) else "E" if EXPLORE_KW.search(r["text"]) else "-"
        kw[(k, SHORT[r["label"]])] += 1
    print("Annotator vs keyword heuristic (rows: keyword R / E / none; cols: N D R E):")
    for k in ("R", "E", "-"):
        tot = sum(kw[(k, c)] for c in "NDRE")
        print(f"  {k}: " + "  ".join(f"{c}={kw[(k, c)] / max(tot, 1):.2f}" for c in "NDRE") + f"  (n={tot})")

    # Per-trace features
    tok = {m: np.load(ENT / f"tokens_{m}.npz") for m in models if (ENT / f"tokens_{m}.npz").exists()}
    raw = {}
    if (BONDS / "raw.json").exists():  # Llama arm (scripts/eval_llama_bonds.py): texts stored alongside
        for k, v in json.load(open(BONDS / "raw.json")).items():
            m, q, i = k.split("|")
            raw[(m, q, int(i))] = v
    for m, path in ({} if raw else {"LKD": "eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-logit-kd-continue-epoch_1-think__math500.json",
                    "RD_new": "eval_outputs/sweep_pass_k_rd_october/qwen-reasondistill-october-continue-epoch_1-think__math500.json",
                    "RD_mod": "eval_outputs/rd_newsae/eval_mod/passk/qwen-rd-newsae-moderate-final-think__math500.json"}).items():
        for rec in json.load(open(path))["records"]:
            for c in rec["candidates"]:
                raw[(m, rec["id"], c["index"])] = c["raw_output"]
    F = {}
    for (m, q), items in traces.items():
        items.sort(key=lambda x: x[0]["step"])
        labs = [SHORT[r["label"]] for r, _ in items]
        n = len(labs)
        f = {f"A_{c}": labs.count(c) / n for c in "NDRE"}
        # Sensitivity (not pre-registered): soft label shares from the annotator's probabilities.
        P = np.array([r["p"] for r, _ in items])
        f.update({f"S_{c}": float(P[:, j].mean()) for j, c in enumerate("NDRE")})
        trans = Counter(zip(labs, labs[1:]))
        fromR, fromE = sum(trans[("R", c)] for c in "NDRE"), sum(trans[("E", c)] for c in "NDRE")
        f["A_R->D"] = trans[("R", "D")] / fromR if fromR else np.nan
        f["A_E->D"] = trans[("E", "D")] / fromE if fromE else np.nan
        meta = lambda c: c in "RE"
        f["A_oscillation"] = 100 * sum(meta(a) != meta(b) for a, b in zip(labs, labs[1:])) / max(n - 1, 1)
        f["_trans"] = trans
        H = emb[[i for _, i in items]]
        d = np.linalg.norm(H[1:] - H[:-1], axis=1) if n > 1 else np.array([0.0])
        thr = np.median(d)
        rec_hits, rec_n, nov = 0, 0, []
        for t in range(2, n):
            if labs[t] == "R":
                rec_n += 1
                rec_hits += np.linalg.norm(H[:t - 1] - H[t], axis=1).min() < thr
            if labs[t] == "E":
                nov.append(d[t - 1] / thr if thr > 0 else np.nan)
        f["B_reconnect"] = rec_hits / rec_n if rec_n else np.nan
        f["B_novelty"] = float(np.mean(nov)) if nov else np.nan
        f["B_compact"] = float(np.linalg.norm(H - H.mean(0), axis=1).mean())
        r0 = items[0][0]
        key = f"{q}|{r0['idx']}"
        if m in tok and key + "|ent" in tok[m]:
            ent, offs = tok[m][key + "|ent"].astype(np.float32), tok[m][key + "|offs"]
            think = raw[(m, q, r0["idx"])].split("</think>")[0]
            spans = step_spans(think)
            e = []
            for s0, s1 in spans:
                sel = ent[(offs >= s0) & (offs < s1)]
                e.append(float(sel.mean()) if len(sel) else np.nan)
            e = np.array([x for x in e if not np.isnan(x)])
            if len(e) >= 6:
                k3 = len(e) // 3
                f["C_convergence"] = float(e[-k3:].mean() - e[:k3].mean())
                de = np.diff(e)
                slope = de / np.where(e[1:] > 0, e[1:], np.nan)
                f["C_high_explore"] = float(np.nanmean((slope > 0.6) & (de > 0.05)))
                f["C_low_validate"] = float(np.nanmean((np.abs(slope) < 0.1) & (np.abs(de) < 0.05)))
                tail = e[-max(3, len(e) // 5):]
                f["C_tail_slope"] = float(np.polyfit(np.linspace(0, 1, len(tail)), tail, 1)[0])
        f["_correct"] = r0["correct"]
        f["_level"] = r0["level"]
        F[(m, q)] = f

    feats = [k for k in next(iter(F.values())) if not k.startswith("_")]
    feats = sorted({k for f in F.values() for k in f if not k.startswith("_")}, key=lambda k: (k[0], k))
    qs = sorted({q for (_, q) in F})
    print("\nMeans per model:")
    for k in feats:
        print(f"  {k:16s} " + "  ".join(f"{m}={np.nanmean([F[(m, q)].get(k, np.nan) for q in qs]):.4f}" for m in models))
    # A5: transition-graph correlation
    def graph(m):
        t = Counter()
        for q in qs:
            t.update(F[(m, q)]["_trans"])
        M = np.array([[t[(a, b)] for b in "NDRE"] for a in "NDRE"], dtype=float)
        return (M / M.sum(1, keepdims=True)).ravel()
    print("\nA5 transition-graph Pearson r with LKD: " + "  ".join(
        f"{m}={np.corrcoef(graph(m), graph('LKD'))[0, 1]:.4f}" for m in models if m != "LKD"))

    for rd in [m for m in models if m != "LKD"]:
        print(f"\n== {rd} - LKD, paired per question (Holm within family)")
        for fam in "ABCS":
            res = []
            for k in [k for k in feats if k.startswith(fam)]:
                d = [F[(rd, q)].get(k, np.nan) - F[("LKD", q)].get(k, np.nan) for q in qs]
                d = [x for x in d if not np.isnan(x)]
                if len(d) >= 20:
                    res.append((k, *boot(d), len(d)))
            for (k, m_, lo, hi, p, n), pa in zip(res, holm([r[4] for r in res])):
                print(f"  {k:16s} {m_:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} Holm {pa:.4f} (n={n})")

    print("\n== Family E: feature difference, correct minus wrong traces, within model (level-stratified)")
    for k in feats:
        line = f"  {k:16s}"
        for m in models:
            diffs = []
            for lv in range(1, 6):
                c = [F[(m, q)].get(k, np.nan) for q in qs if F[(m, q)]["_level"] == lv and F[(m, q)]["_correct"]]
                w = [F[(m, q)].get(k, np.nan) for q in qs if F[(m, q)]["_level"] == lv and not F[(m, q)]["_correct"]]
                c, w = [x for x in c if not np.isnan(x)], [x for x in w if not np.isnan(x)]
                if len(c) >= 5 and len(w) >= 5:
                    diffs.append(np.mean(c) - np.mean(w))
            line += f"  {m}={np.mean(diffs):+.4f}" if diffs else f"  {m}=  n/a "
        print(line)


if __name__ == "__main__":
    main()
