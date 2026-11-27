"""alpha = 0 ablation of the RD term (docs/mi_qwen_students.md): rd_zero = RD pipeline with alpha_align 0, rd_low_rr = the
alpha 0.05 run repeated on the same machine (replication). Compared with lkd (LKD-continue) and the original RD runs.
Phases
  pr       P_R (probability of starting a reflection step at step boundaries) on the Exp 3 test traces, appended to
           patching/gap.jsonl so every model is scored on identical text
  weights  per layer: cosine and distance between training updates (all vs anc), every pair of
           lkd, rd_zero, rd_low_rr, rd_low, rd_mod, rd_high
  analyze
Run: python scripts/mi_qwen_alpha0.py pr|weights|analyze
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mi_qwen_diffs import group, state
from mi_qwen_patching import D3, _setup, per_q, run
from mi_qwen_students_common import MODELS, N_LAYERS, OUT, boot, load_model, log, split

D0 = OUT / "alpha0"
NEW = ("rd_zero", "rd_low_rr")
ALL = ("lkd", "rd_zero", "rd_low_rr", "rd_low", "rd_mod", "rd_high")


def pr():
    trs_all, r_ids = _setup()
    trs = split(trs_all, "test")
    path = D3 / "gap.jsonl"
    done = {json.loads(l)["model"] for l in path.open()}
    for name in NEW:
        if name in done or not (Path(MODELS[name]) / "model.safetensors").exists():
            continue
        m = load_model(name)
        recs = [run(m, t, r_ids) for t in trs]
        with path.open("a") as f:
            f.write(json.dumps({"model": name, "records": recs}) + "\n")
        log(f"alpha0 pr {name}: {len(recs)} traces")
        del m
        torch.cuda.empty_cache()


def weights():
    anc = state("anc")
    names = [n for n in ALL if (Path(MODELS[n]) / "model.safetensors").exists()]
    upd = {}
    for n in names:
        sd = state(n)
        assert set(sd) == set(anc), n
        upd[n] = sd
    rows = {}
    for k, w0 in anc.items():
        g = str(group(k))
        w0 = w0.float()
        d = {n: upd[n][k].float() - w0 for n in names}
        r = rows.setdefault(g, {})
        for i, a in enumerate(names):
            r[f"n_{a}"] = r.get(f"n_{a}", 0.0) + float((d[a] ** 2).sum())
            for b in names[i + 1:]:
                r[f"dot_{a}|{b}"] = r.get(f"dot_{a}|{b}", 0.0) + float((d[a] * d[b]).sum())
    D0.mkdir(parents=True, exist_ok=True)
    (D0 / "weights.json").write_text(json.dumps({"names": names, "rows": rows}))
    log("alpha0 weights: done")


def analyze(vast=False):
    """vast: read the files computed on the Vast instance (alpha0/gap_vast.jsonl, alpha0/weights_vast.json; every model
    scored there in one environment) and cross-check the shared models against the local gap.jsonl."""
    L = ["# alpha = 0 ablation of the RD term (rd_zero) and same-machine replication of alpha 0.05 (rd_low_rr)"
         + ("  [P_R and weights computed on the Vast instance]" if vast else "")]
    G = {json.loads(l)["model"]: json.loads(l)["records"] for l in ((D0 / "gap_vast.jsonl") if vast else (D3 / "gap.jsonl")).open()}
    if vast:
        Lc = {json.loads(l)["model"]: json.loads(l)["records"] for l in (D3 / "gap.jsonl").open()}
        L.append("\n## Environment cross-check: P_R Vast vs local, same models and text")
        for n in ("anc", "lkd", "rd_low", "rd_mod", "rd_high"):
            if n in G and n in Lc:
                a, b = per_q(G[n]), per_q(Lc[n])
                L.append(f"  {n:8s} vast {np.mean(list(a.values())):.4f}  local {np.mean(list(b.values())):.4f}"
                         f"  max |per-question diff| {max(abs(a[q] - b[q]) for q in a):.5f}")
    base = per_q(G["lkd"])
    qs = sorted(base)
    L.append(f"\n## P_R at step boundaries, identical text ({len(qs)} test questions x 2 trace sets)")
    for n in ("anc",) + ALL:
        if n not in G:
            continue
        pq = per_q(G[n])
        L.append(f"  {n:10s} {np.mean([pq[q] for q in qs]):.4f}")
    pairs = [("rd_zero", "lkd", ""), ("rd_low", "rd_zero", "the RD term at alpha 0.05 (cross-machine)"),
             ("rd_low_rr", "rd_zero", "the RD term at alpha 0.05 (SAME machine: the ablation)"),
             ("rd_low_rr", "rd_low", "replication: same config, other machine"), ("rd_high", "rd_low", "dose 0.05 -> 0.40")]
    for a, b, lab in pairs:
        if a in G and b in G:
            pa, pb = per_q(G[a]), per_q(G[b])
            m, lo, hi, p = boot([pa[q] - pb[q] for q in qs])
            L.append(f"  {a} - {b}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}   {lab}".rstrip())
    wf = D0 / ("weights_vast.json" if vast else "weights.json")
    if wf.exists():
        W = json.loads(wf.read_text())
        names, R = W["names"], W["rows"]

        def cos(r, a, b):
            k = f"dot_{a}|{b}" if f"dot_{a}|{b}" in r else f"dot_{b}|{a}"
            return r[k] / (r[f"n_{a}"] * r[f"n_{b}"]) ** 0.5

        def dist(r, a, b):          # |dW_a - dW_b| / |dW_lkd|
            k = f"dot_{a}|{b}" if f"dot_{a}|{b}" in r else f"dot_{b}|{a}"
            return ((r[f"n_{a}"] + r[f"n_{b}"] - 2 * r[k]) / r["n_lkd"]) ** 0.5
        prs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        L.append("\n## Weight updates (vs LKD epoch 1): cosine per pair, mean over layer bands")
        bands = {"0-8": range(0, 9), "9-12": range(9, 13), "13-23": range(13, 24)}
        L.append("  pair                      " + "".join(f"{b:>8s}" for b in bands) + "   | dist/|dW_lkd| " + "".join(f"{b:>8s}" for b in bands))
        for a, b in prs:
            c = [np.mean([cos(R[str(i)], a, b) for i in rg]) for rg in bands.values()]
            d = [np.mean([dist(R[str(i)], a, b) for i in rg]) for rg in bands.values()]
            L.append(f"  {a + ' ~ ' + b:26s}" + "".join(f"{x:8.3f}" for x in c) + "   |              " + "".join(f"{x:8.3f}" for x in d))
    out = "\n".join(L)
    print(out)
    D0.mkdir(parents=True, exist_ok=True)
    (D0 / ("analysis_vast.txt" if vast else "analysis.txt")).write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pr", "weights", "analyze", "analyze_vast"])
    ph = ap.parse_args().phase
    analyze(vast=True) if ph == "analyze_vast" else {"pr": pr, "weights": weights, "analyze": analyze}[ph]()
