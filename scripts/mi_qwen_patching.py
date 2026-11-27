"""Exp 3 (docs/mi_qwen_students.md): P_R by model, teacher-forced. Also the shared P_R helpers (_setup, run, per_q).

Metric  P_R at step boundaries (mi_qwen_students_common): probability of starting a reflection step.
Phases
  gap    P_R of anc, lkd, rd_low, rd_mod, rd_high on the same held-out text (dose check: does P_R follow alpha?);
         mi_qwen_alpha0.py pr appends rd_zero and rd_low_rr to the same file
  analyze
Run: python scripts/mi_qwen_patching.py gap|analyze
"""
import argparse
import json
import time

import numpy as np
import torch

from mi_qwen_students_common import OUT, boot, load_model, log, p_r, reflection_ids, split, tokenizer, traces

D3 = OUT / "patching"
GAP_MODELS = ("anc", "lkd", "rd_low", "rd_mod", "rd_high")


def _setup():
    tok = tokenizer()
    trs_all = traces(tok)
    r_ids = torch.tensor(reflection_ids(tok, trs_all)).cuda()
    return trs_all, r_ids


@torch.no_grad()
def run(m, t, r_ids):
    h = m.model(torch.tensor([t["ids"]]).cuda()).last_hidden_state[0]
    b = torch.tensor(t["bpos"]).cuda()
    pr, lo = p_r(m, h, b, r_ids)
    refl = torch.tensor([l == "self-reflection" for l in t["blab"]]).cuda()
    return {"qi": t["qi"], "src": t["src"], "pr": float(pr.mean()), "lo": float(lo.mean()),
            "pr_refl": float(pr[refl].mean()) if refl.any() else None,
            "pr_other": float(pr[~refl].mean()) if (~refl).any() else None}


def gap():
    trs_all, r_ids = _setup()
    trs = split(trs_all, "test")
    path = D3 / "gap.jsonl"
    D3.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["model"] for l in path.open()} if path.exists() else set()
    for name in GAP_MODELS:
        if name in done:
            continue
        t0 = time.time()
        m = load_model(name)
        recs = [run(m, t, r_ids) for t in trs]
        with path.open("a") as f:
            f.write(json.dumps({"model": name, "records": recs}) + "\n")
        log(f"patching gap {name}: {len(recs)} traces {time.time() - t0:.0f}s")
        del m
        torch.cuda.empty_cache()


def per_q(recs, key="pr"):
    d = {}
    for r in recs:
        if r[key] is not None:
            d.setdefault(r["qi"], []).append(r[key])
    return {q: float(np.mean(v)) for q, v in d.items()}


def analyze():
    L = ["# Exp 3 · P_R by model (teacher-forced, held-out MATH-500 traces of both models)",
         "  P_R = probability of starting a reflection step, at step boundaries; per-question means, bootstrap over questions"]
    if (D3 / "gap.jsonl").exists():
        G = {json.loads(l)["model"]: json.loads(l)["records"] for l in (D3 / "gap.jsonl").open()}
        L.append(f"\n## P_R by model ({len(per_q(G['lkd']))} questions x 2 trace sets)")
        base = per_q(G["lkd"])
        qs = sorted(base)
        for name in GAP_MODELS:
            if name not in G:
                continue
            pq = per_q(G[name])
            line = f"  {name:8s} P_R {np.mean([pq[q] for q in qs]):.4f}"
            for k in ("pr_refl", "pr_other"):
                line += f"  {k[3:]} {np.mean(list(per_q(G[name], k).values())):.4f}"
            if name != "lkd":
                m, lo, hi, p = boot([pq[q] - base[q] for q in qs])
                line += f"  | - lkd {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}"
            L.append(line)
        if all(k in G for k in ("rd_low", "rd_high")):
            lo_, hi_ = per_q(G["rd_low"]), per_q(G["rd_high"])
            m, lo, hi, p = boot([hi_[q] - lo_[q] for q in qs])
            L.append(f"  dose: rd_high - rd_low {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}  (same RD pipeline; the RD term's part)")
    out = "\n".join(L)
    print(out)
    (D3 / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gap", "analyze"])
    a = ap.parse_args()
    {"gap": gap, "analyze": analyze}[a.phase]()
