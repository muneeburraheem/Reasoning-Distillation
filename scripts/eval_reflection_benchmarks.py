"""Reflection follow-up 2: more traces and more benchmarks (README "Reflection follow-up 2", pre-registered 2026-11-20).

Labels reasoning steps (scripts/eval_bond_annotate.py: Chen et al. 2026 prompt, Qwen3.5-4B annotator, argmax over the four
labels) in existing LKD and RD_mod eval outputs, then tests whether reflection goes with correct answers more in RD.

label    one output file per (benchmark, model) in eval_outputs/rd_newsae/reflection_more/; resumable. MATH-500 traces already labelled in
         eval_outputs/rd_newsae/eval_mod/bonds/steps.json are reused, not relabelled.
analyze  primary: MATH-500 within-question coupling (mean reflection of correct minus wrong traces, per question with
         both), RD - LKD, bootstrap over questions. Secondary: per-benchmark coupling, both-correct style shares, pooled.

Run:
  python scripts/eval_reflection_benchmarks.py label
  python scripts/eval_reflection_benchmarks.py analyze
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT = ROOT / "eval_outputs/rd_newsae/reflection_more"
M = ROOT / "eval_outputs/rd_newsae/eval_mod"
SEP = ROOT / "eval_outputs/september/20260922T180514Z-1112953/think"
PK = ROOT / "eval_outputs/sweep_pass_k/20260929T073130Z-1261224"
HB = ROOT / "eval_outputs/hard_math/B"
# benchmark: (LKD file, RD_mod file, max finished traces per question, September format)
BENCH = {
    "math500_p8": (PK / "qwen-logit-kd-continue-epoch_1-think__math500.json", M / "passk/qwen-rd-newsae-moderate-final-think__math500.json", 4, False),
    "aime25_p8": (PK / "qwen-logit-kd-continue-epoch_1-think__aime25.json", M / "passk/qwen-rd-newsae-moderate-final-think__aime25.json", 8, False),
    "aime25_32k": (ROOT / "eval_outputs/test_math/20260925T181224Z-112470/qwen-logit-kd-continue-epoch_1-think__aime25.json", M / "basic_math/qwen-rd-newsae-moderate-final-think__aime25.json", 1, False),
    "olympiad": (ROOT / "eval_outputs/hard_math/A/qwen-logit-kd-continue-epoch_1-think__olympiadbench.json", M / "hard_math/A/qwen-rd-newsae-moderate-final-think__olympiadbench.json", 1, False),
    "aime24": (HB / "qwen-logit-kd-continue-epoch_1-think__aime24.json", M / "hard_math/B/qwen-rd-newsae-moderate-final-think__aime24.json", 4, False),
    "hmmt25": (HB / "qwen-logit-kd-continue-epoch_1-think__hmmt25.json", M / "hard_math/B/qwen-rd-newsae-moderate-final-think__hmmt25.json", 4, False),
    "cmimc25": (HB / "qwen-logit-kd-continue-epoch_1-think__cmimc25.json", M / "hard_math/B/qwen-rd-newsae-moderate-final-think__cmimc25.json", 4, False),
    "mmlu_think": (SEP / "qwen-logit-kd-continue-epoch_1-think__mmlu.json", M / "september/think/qwen-rd-newsae-moderate-final-think__mmlu.json", 1, True),
    "hellaswag_think": (SEP / "qwen-logit-kd-continue-epoch_1-think__hellaswag.json", M / "september/think/qwen-rd-newsae-moderate-final-think__hellaswag.json", 1, True),
}
MODELS = ("LKD", "RD_mod")
SHORT = {"normal operation": "N", "deep reasoning": "D", "self-reflection": "R", "exploration": "E"}
B = 10000


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def finished(c, sep):
    return "</think>" in c["raw_output"] if sep else not c["hit_token_limit"]


def traces(path, k, sep):
    """[(qid, idx, correct, level, steps)] for up to k finished traces per question, lowest index first."""
    from eval_bond_annotate import segment
    out = []
    for rec in json.load(open(path))["records"]:
        fin = [c for c in sorted(rec["candidates"], key=lambda c: c["index"]) if finished(c, sep)][:k]
        for c in fin:
            out.append((rec["id"], c["index"], bool(c["correct"]), (rec.get("metadata") or {}).get("level"),
                        segment(c["raw_output"].split("</think>")[0])))
    return out


def label():
    from eval_bond_annotate import LABELS, annotate
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for r in json.load(open(M / "bonds/steps.json")):
        if r["model"] in MODELS:
            cache[(r["model"], r["qid"], r["idx"], r["step"])] = (r["label"], r["p"])
    log(f"cache: {len(cache)} labelled MATH-500 steps from eval_mod/bonds")
    for bench, (lkd, rd, k, sep) in BENCH.items():
        for model, path in zip(MODELS, (lkd, rd)):
            dst = OUT / f"{bench}__{model}.json"
            if dst.exists():
                continue
            rows, todo = [], []
            for qid, idx, ok, level, steps in traces(path, k, sep):
                for s, cur in enumerate(steps):
                    row = {"model": model, "qid": qid, "idx": idx, "step": s, "n_steps": len(steps), "correct": ok,
                           "level": level, "text": cur}
                    hit = cache.get((model, qid, idx, s)) if bench == "math500_p8" else None
                    if hit:
                        row["label"], row["p"] = hit
                    else:
                        todo.append((len(rows), (steps[s - 1] if s else "", cur)))
                    rows.append(row)
            n_tr = len({(r["qid"], r["idx"]) for r in rows})
            log(f"{bench} {model}: {n_tr} traces, {len(rows)} steps, {len(todo)} to label")
            t0 = time.time()
            if todo:
                probs = annotate([p for _, p in todo], str(ROOT / "cache"))
                for (i, _), p in zip(todo, probs):
                    rows[i]["label"] = LABELS[int(p.argmax())]
                    rows[i]["p"] = [round(float(x), 4) for x in p]
            dst.write_text(json.dumps(rows))
            log(f"{bench} {model}: saved ({time.time() - t0:.0f}s)")
    log("ALL_DONE")


def trace_feats(rows):
    tr = defaultdict(list)
    for r in rows:
        tr[(r["qid"], r["idx"])].append(r)
    F = {}
    for key, items in tr.items():
        items.sort(key=lambda r: r["step"])
        labs = [SHORT[r["label"]] for r in items]
        n = len(labs)
        meta = [c in "RE" for c in labs]
        F[key] = {"R": 100 * labs.count("R") / n, "E": 100 * labs.count("E") / n, "N": 100 * labs.count("N") / n,
                  "osc": 100 * sum(a != b for a, b in zip(meta, meta[1:])) / max(n - 1, 1),
                  "correct": items[0]["correct"], "level": items[0]["level"]}
    return F


def ci(x):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if not len(x):
        return np.nan, np.nan, np.nan
    return np.percentile(x, 2.5), np.percentile(x, 97.5), max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1 / len(x))


def per_question(F, qs, k):
    """Per question: sum and count of feature k over correct and over wrong traces -> array [len(qs), 4]."""
    pos = {q: i for i, q in enumerate(qs)}
    A = np.zeros((len(qs), 4))
    for (q, _), f in F.items():
        j = 0 if f["correct"] else 2
        A[pos[q], j] += f[k]
        A[pos[q], j + 1] += 1
    return A


def coupling_w(A, w, within):
    """Coupling under question weights w (bootstrap counts). within: mean over questions having both kinds."""
    if within:
        ok = (A[:, 1] > 0) & (A[:, 3] > 0) & (w > 0)
        if not ok.any():
            return np.nan, 0
        d = A[ok, 0] / A[ok, 1] - A[ok, 2] / A[ok, 3]
        return float(np.average(d, weights=w[ok])), int(ok.sum())
    sc, nc, sw, nw = (w[:, None] * A).sum(0)
    return (sc / nc - sw / nw) if nc and nw else np.nan, int((A[:, 3] * (w > 0)).sum())


def analyze():
    L = ["# Reflection follow-up 2: LKD vs RD_mod (RD α 0.15); shares in % of steps",
         "  coupling = share in correct minus wrong traces; within question (n = questions with both) where several traces,\n"
         "  else pooled over traces (n = wrong traces)"]
    rng = np.random.default_rng(0)
    pooled_rows = []
    for bench, (_, _, k, _) in BENCH.items():
        if not all((OUT / f"{bench}__{m}.json").exists() for m in MODELS):
            L.append(f"\n## {bench}: not labelled yet")
            continue
        F = {m: trace_feats(json.load(open(OUT / f"{bench}__{m}.json"))) for m in MODELS}
        qs = sorted({q for m in MODELS for (q, _) in F[m]})
        idx = [rng.integers(0, len(qs), len(qs)) for _ in range(B)]
        n_tr = {m: len(F[m]) for m in MODELS}
        n_wr = {m: sum(not f["correct"] for f in F[m].values()) for m in MODELS}
        L.append(f"\n## {bench}  traces LKD {n_tr['LKD']} (wrong {n_wr['LKD']}) | RD {n_tr['RD_mod']} (wrong {n_wr['RD_mod']})"
                 + ("  [PRIMARY]" if bench == "math500_p8" else ""))
        W = np.stack([np.bincount(ix, minlength=len(qs)) for ix in idx]).astype(float)
        ones = np.ones(len(qs))
        for feat in ("R", "E", "osc"):
            A = {m: per_question(F[m], qs, feat) for m in MODELS}
            pt = {m: coupling_w(A[m], ones, k > 1) for m in MODELS}
            bs = {m: [coupling_w(A[m], w, k > 1)[0] for w in W] for m in MODELS}
            lo_l, hi_l, _ = ci(bs["LKD"]); lo_r, hi_r, _ = ci(bs["RD_mod"])
            lo, hi, p = ci(np.array(bs["RD_mod"]) - np.array(bs["LKD"]))
            L.append(f"  coupling {feat:3s} LKD {pt['LKD'][0]:+.2f} [{lo_l:+.2f},{hi_l:+.2f}] (n={pt['LKD'][1]}) | "
                     f"RD {pt['RD_mod'][0]:+.2f} [{lo_r:+.2f},{hi_r:+.2f}] (n={pt['RD_mod'][1]}) | "
                     f"RD-LKD {pt['RD_mod'][0] - pt['LKD'][0]:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.4f}")
            if feat == "R":
                pooled_rows.append((bench, pt["LKD"][0], pt["RD_mod"][0]))
        # style shift on questions where both models' (first) traces are correct
        first = {m: {} for m in MODELS}
        for m in MODELS:
            for (q, i), f in sorted(F[m].items(), key=lambda kv: kv[0][1]):
                first[m].setdefault(q, f)
        bq = [q for q in qs if q in first["LKD"] and q in first["RD_mod"] and first["LKD"][q]["correct"] and first["RD_mod"][q]["correct"]]
        if len(bq) >= 10:
            for feat in ("R", "E", "N", "osc"):
                d = np.array([first["RD_mod"][q][feat] - first["LKD"][q][feat] for q in bq])
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
                lo, hi, p = ci(bs)
                L.append(f"  both-correct {feat:3s} LKD {np.mean([first['LKD'][q][feat] for q in bq]):6.2f} "
                         f"RD {np.mean([first['RD_mod'][q][feat] for q in bq]):6.2f} diff {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.4f} (n={len(bq)})")
        else:
            L.append(f"  both-correct: only {len(bq)} questions, skipped")
    rest = [r for r in pooled_rows if r[0] != "math500_p8" and not np.isnan(r[1]) and not np.isnan(r[2])]
    if rest:
        L.append("\n## Non-MATH benchmarks, reflection coupling RD - LKD (unweighted mean over benchmarks; descriptive)")
        L.append("  " + "  ".join(f"{b} {r - l:+.2f}" for b, l, r in rest))
        L.append(f"  mean {np.mean([r - l for _, l, r in rest]):+.2f}; RD above LKD on {sum(r > l for _, l, r in rest)} of {len(rest)}")
    out = "\n".join(L)
    print(out)
    (OUT / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"label": label, "analyze": analyze}[sys.argv[1]]()
