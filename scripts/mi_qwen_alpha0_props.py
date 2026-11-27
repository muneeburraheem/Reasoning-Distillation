"""Properties of every student for the alpha = 0 comparison (docs/mi_qwen_students.md, "alpha = 0 ablation").

Models  lkd (LKD-continue), rd_zero (RD pipeline, alpha 0), rd_low_rr (alpha 0.05 rerun, same machine as rd_zero),
        rd_low / rd_mod / rd_high (alpha 0.05 / 0.15 / 0.40). One model at a time, the same procedure for each; the
        dictionaries and latent sets are the fixed ones from Exp 1, never refitted.
Per model (phase `model NAME`)
  refl      pooled SAE (Exp 1, checkpoints/mi_qwen_crosscoder/final.pt) at hidden_states[13] on the step-labelled
            traces: per-latent mean activation on reflection / other tokens; per trace, summed mean activation on
            reflection and on other tokens of the fixed top-512 reflection-enriched latents (label_stats.pt) and of 512
            fixed random live latents (control)
  transfer  Exp 5: best correlation of each teacher feature (196 rd + 196 rand) with any pooled-SAE latent, probe test rows
  Activation scale for the pooled SAE: the stored one for the four models it was trained on; for new models the same
  rule (sqrt(D) / mean token norm on the first 50 training batches); `scale_check` recomputes it for rd_low.
Phase `compare`: table over models with paired contrasts against rd_zero (the matched baseline) and against lkd; also
  reads P_R / weights (alpha0/gap_vast.jsonl, weights_vast.json) and the probe R^2 (mi_qwen_student_probe.py files).
Run: python scripts/mi_qwen_alpha0_props.py model NAME | scale_check | compare
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import mi_qwen_crosscoder as CC
from mi_qwen_students_common import D, MODELS, N_LAYERS, OUT, REFL, boot, load_model, log, tokenizer, traces

D0 = OUT / "alpha0"
ORDER = ("lkd", "rd_zero", "rd_low_rr", "rd_low", "rd_mod", "rd_high")
PROBE_NAME = {"lkd": "lkd", "rd_zero": "rd_zero", "rd_low_rr": "rd_lowrerun", "rd_low": "rd_low", "rd_mod": "rd_moderate",
              "rd_high": "rd_high"}


def fresh_scale(name):
    """sqrt(D) / mean token norm at hs[13] on the first 50 training batches, massive-norm tokens dropped (as CC.train)."""
    tok = tokenizer()
    m = {name: load_model(name, n_layers=CC.HS)}
    it = CC.batches(tok, CC.ROW0)
    s, n = 0.0, 0
    for _ in range(50):
        nr = CC.acts(m, next(it))[:, 0].norm(dim=-1)
        nr = nr[nr < 10 * nr.mean()]
        s += float(nr.sum()); n += nr.shape[0]
    del m
    torch.cuda.empty_cache()
    return math.sqrt(D) / (s / n)


def sae_and_scale(name):
    S = torch.load(CC.CK / "final.pt")
    sae = CC.BatchTopK(1).cuda()
    sae.load_state_dict(S["sae"])
    sae.eval()
    if name in CC.MODELS:
        return sae, float(S["scale"][CC.MODELS.index(name)])
    return sae, fresh_scale(name)


def latent_sets():
    LS = torch.load(CC.CK / "label_stats.pt")
    sup = torch.where(LS["S"]["refl_n"].sum(0) >= 200)[0]
    rnd = sup[torch.randperm(len(sup), generator=torch.Generator().manual_seed(0))[:512]]
    return LS["top"], rnd


@torch.no_grad()
def model_props(name):
    D0.mkdir(parents=True, exist_ok=True)
    out_p = D0 / f"props_{name}.pt"
    sae, scale = sae_and_scale(name)
    top, rnd = latent_sets()
    tok = tokenizer()
    trs = traces(tok)
    m13 = {name: load_model(name, n_layers=CC.HS)}
    # refl
    S = {k: torch.zeros(CC.F_LAT, dtype=torch.float64, device="cuda") for k in ("refl", "oth", "refl_n", "oth_n")}
    nr = no = 0
    per_trace = []
    for t in trs:
        if not t["bpos"]:
            continue
        lab = torch.zeros(len(t["ids"]), dtype=torch.long)
        b = t["bpos"] + [len(t["ids"]) - 1]
        for k, l in enumerate(t["blab"]):
            lab[b[k] + 1:b[k + 1] + 1] = 1 if l == REFL else 2
        lab = lab[1:].cuda()
        r, o = lab == 1, lab == 2
        if not r.any() or not o.any():
            continue
        a = CC.acts(m13, torch.tensor([t["ids"]]).cuda()) * scale
        fs, _ = sae.encode(a)
        S["refl"] += fs[r].sum(0).double(); S["oth"] += fs[o].sum(0).double()
        S["refl_n"] += (fs[r] > 0).sum(0).double(); S["oth_n"] += (fs[o] > 0).sum(0).double()
        nr += int(r.sum()); no += int(o.sum())
        mr, mo = fs[r].mean(0), fs[o].mean(0)
        per_trace.append({"qi": t["qi"], "src": t["src"], "top_r": float(mr[top].sum()), "top_o": float(mo[top].sum()),
                          "rnd_r": float(mr[rnd].sum()), "rnd_o": float(mo[rnd].sum())})
    log(f"alpha0 props {name}: refl done, scale {scale:.4f}, {len(per_trace)} traces")
    # transfer
    from mi_qwen_student_probe import MAXLEN, OUT as PROBE, TENS as PROBE_TENS
    texts = [json.loads(l) for l in (PROBE / "texts.jsonl").open()]
    T = torch.load(PROBE_TENS / "targets.pt", mmap=True)
    nT = T["Y"][0].shape[1]
    sx = torch.zeros(CC.F_LAT, dtype=torch.float64, device="cuda"); sxx = torch.zeros_like(sx)
    sxy = torch.zeros(CC.F_LAT, nT, dtype=torch.float64, device="cuda")
    sy = torch.zeros(nT, dtype=torch.float64, device="cuda"); syy = torch.zeros_like(sy)
    n = 0
    for i in range(1, len(texts), 2):
        ids = tok(texts[i]["text"], truncation=True, max_length=MAXLEN)["input_ids"]
        y = T["Y"][i].cuda().double()[1:]
        fs, _ = sae.encode(CC.acts(m13, torch.tensor([ids]).cuda()) * scale)
        fs = fs.double()
        sy += y.sum(0); syy += y.pow(2).sum(0); n += y.shape[0]
        sx += fs.sum(0); sxx += fs.pow(2).sum(0); sxy += fs.T @ y
    cov = sxy / n - torch.outer(sx / n, sy / n)
    sdx = (sxx / n - (sx / n) ** 2).clamp(min=1e-12).sqrt()
    sdy = (syy / n - (sy / n) ** 2).clamp(min=1e-12).sqrt()
    c = cov / torch.outer(sdx, sdy)
    c[sdx < 1e-6] = 0
    best = c.max(0).values.cpu()
    del m13
    torch.cuda.empty_cache()
    log(f"alpha0 props {name}: transfer done, {n} tokens")
    torch.save({"scale": scale, "S": {k: v.cpu() for k, v in S.items()}, "n_refl": nr, "n_oth": no, "per_trace": per_trace,
                "transfer_best": best, "n_rd": len(T["rd"]), "transfer_tokens": n}, out_p)
    log(f"alpha0 props {name}: saved {out_p.name}")


def scale_check():
    S = torch.load(CC.CK / "final.pt")
    log(f"alpha0 scale check rd_low: stored {float(S['scale'][CC.MODELS.index('rd_low')]):.5f}, recomputed {fresh_scale('rd_low'):.5f}")


def per_q(rows, key):
    d = {}
    for r in rows:
        if r[key] is not None:
            d.setdefault(r["qi"], []).append(r[key])
    return {q: float(np.mean(v)) for q, v in d.items()}


def compare():
    P = {n: torch.load(D0 / f"props_{n}.pt", weights_only=False) for n in ORDER if (D0 / f"props_{n}.pt").exists()}
    names = [n for n in ORDER if n in P]
    L = ["# Properties of every student: LKD-continue, the alpha = 0 RD run, the alpha 0.05 rerun and alpha 0.05 / 0.15 / 0.40",
         "  contrasts vs rd_zero = the alignment term (same pipeline); vs lkd = the paper's original comparison",
         "  models present: " + ", ".join(names)]
    base = [b for b in ("rd_zero", "lkd") if b in P]

    def contrasts(vals, unit_fmt, rel=False):
        """vals: {model: {key: value}} paired over keys"""
        out = []
        for b in base:
            for n in names:
                if n == b:
                    continue
                ks = sorted(set(vals[n]) & set(vals[b]))
                d = [(vals[n][k] - vals[b][k]) / (vals[b][k] if rel else 1) for k in ks]
                m, lo, hi, p = boot(d)
                s = 100 if rel else 1
                out.append(f"    {n:9s} - {b:7s} {unit_fmt(m * s)} [{unit_fmt(lo * s)},{unit_fmt(hi * s)}] p={p:.4f}")
        return out

    # P_R
    gp = D0 / "gap_vast.jsonl"
    if gp.exists():
        G = {json.loads(l)["model"]: json.loads(l)["records"] for l in gp.open()}
        L.append("\n## P_R: probability of starting a reflection step, step boundaries of fixed text (Vast, 213 questions x 2 sets)")
        L.append("  " + "  ".join(f"{n} {np.mean(list(per_q(G[n], 'pr').values())):.4f}" for n in names if n in G))
        L += contrasts({n: per_q(G[n], "pr") for n in names if n in G}, lambda x: f"{x:+.4f}")
    # reflection latents
    L.append("\n## Exp 1 · reflection-enriched latents of the pooled SAE (fixed top-512) at hs[13], step-labelled traces")
    L.append("  enrichment = mean activation on reflection tokens / on other tokens; random = 512 fixed random live latents")
    for n in names:
        S = P[n]["S"]
        top, rnd = latent_sets()
        e_top = float((S["refl"][top].sum() / P[n]["n_refl"]) / (S["oth"][top].sum() / P[n]["n_oth"]))
        e_rnd = float((S["refl"][rnd].sum() / P[n]["n_refl"]) / (S["oth"][rnd].sum() / P[n]["n_oth"]))
        L.append(f"  {n:9s} enrichment top-512 {e_top:6.2f}   random {e_rnd:5.3f}   (scale {P[n]['scale']:.4f})")
    for key, what in (("top_r", "top-512 on reflection tokens"), ("top_o", "top-512 on other tokens"),
                      ("rnd_r", "random-512 on reflection tokens")):
        L.append(f"  activation, {what}, % change (paired over questions):")
        L += contrasts({n: per_q(P[n]["per_trace"], key) for n in names}, lambda x: f"{x:+.2f}%", rel=True)
    # transfer
    L.append("\n## Exp 5 · teacher-feature transfer: best correlation of each teacher feature with a pooled-SAE latent")
    nrd = P[names[0]]["n_rd"]
    for part, sl in (("rd set", slice(0, nrd)), ("rand set", slice(nrd, None))):
        L.append(f"  {part:8s} " + "  ".join(f"{n} {float(P[n]['transfer_best'][sl].mean()):.4f}" for n in names))
        L += contrasts({n: dict(enumerate(P[n]["transfer_best"][sl].tolist())) for n in names}, lambda x: f"{x:+.4f}")
    # probe R^2
    try:
        from mi_qwen_student_probe import N_SET, S_HS, TENS, r2
        PR = {n: torch.load(TENS / f"probe_{PROBE_NAME[n]}.pt") for n in names if (TENS / f"probe_{PROBE_NAME[n]}.pt").exists()}
        if PR:
            _, nz = r2(PR[next(iter(PR))]["ridge"][:, 0])
            live = nz >= 20
            L.append(f"\n## Check 1 · ridge R^2 of the teacher's SAE features from the student's hs[{S_HS}] (400 texts; bootstrap over texts)")
            nt = PR[next(iter(PR))]["ridge"].shape[0]
            rng = np.random.default_rng(0)
            boots = [torch.from_numpy(rng.integers(0, nt, nt)) for _ in range(1000)]
            for part, sl in (("rd", slice(0, N_SET)), ("rand", slice(N_SET, 2 * N_SET))):
                def mr(st):
                    r, _ = r2(st)
                    return float(r[sl][live[sl]].mean())
                L.append(f"  {part:4s} " + "  ".join(f"{n} {mr(PR[n]['ridge'][:, S_HS - 1]):.4f}" for n in names if n in PR))
                for b in [x for x in base if x in PR]:
                    for n in [x for x in names if x in PR and x != b]:
                        A, B_ = PR[n]["ridge"][:, S_HS - 1], PR[b]["ridge"][:, S_HS - 1]
                        d0 = mr(A) - mr(B_)
                        bs = np.array([mr(A[j]) - mr(B_[j]) for j in boots])
                        pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 1000)
                        L.append(f"    {n:9s} - {b:7s} {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={pv:.3f}")
    except FileNotFoundError:
        pass
    # weights
    wp = D0 / "weights_vast.json"
    if wp.exists():
        W = json.loads(wp.read_text())
        R = W["rows"]
        L.append("\n## Weight updates vs LKD epoch 1: |dW_model - dW_rd_zero| / |dW_lkd|, layer bands 0-8 / 9-12 / 13-23")
        for n in [x for x in W["names"] if x != "rd_zero"]:
            ds = []
            for rg in (range(0, 9), range(9, 13), range(13, 24)):
                v = []
                for i in rg:
                    r = R[str(i)]
                    k = f"dot_{n}|rd_zero" if f"dot_{n}|rd_zero" in r else f"dot_rd_zero|{n}"
                    v.append(((r[f"n_{n}"] + r["n_rd_zero"] - 2 * r[k]) / r["n_lkd"]) ** 0.5)
                ds.append(np.mean(v))
            L.append(f"  {n:9s} " + "  ".join(f"{x:.3f}" for x in ds))
    out = "\n".join(L)
    print(out)
    (D0 / "properties.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["model", "scale_check", "compare"])
    ap.add_argument("name", nargs="?")
    a = ap.parse_args()
    if a.phase == "model":
        assert a.name in MODELS, a.name
        model_props(a.name)
    elif a.phase == "scale_check":
        scale_check()
    else:
        compare()
