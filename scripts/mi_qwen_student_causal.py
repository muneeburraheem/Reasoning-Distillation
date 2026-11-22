"""Qwen students: do they USE the teacher's RD-feature directions? (port of Llama 8ab, scripts/mi_llama_students.py)

Pre-registered in docs/rd_newsae.md ("Student causal test"), 2026-11-19. Primary: k = 32, E(s) = dP(marker | rd) -
dP(marker | rand) at marker targets inside <think>, I = E(RD) - E(LKD) for rd_moderate and rd_high.

Text/targets  mi_qwen_student_probe.py outputs (400 held-out AM rows, first 2048 tokens, even = fit, odd = test; teacher
              hidden_states[17], new SAE, RD's encode; rd = top-196 ReasonScore, rand = density-matched)
Site          student hidden_states[13] = output of model.layers[12] (where RD's head reads)
Subspaces     per student, fitted on the fit rows only:
  rd       top-k left singular vectors of the ridge map h (centred) -> standardised rd targets
  rand     the same for the rand targets
  randdir  random k-dim subspace (QR of a Gaussian, seed 0)
  aligner  top-k right singular vectors of RD_moderate's trained head proj.weight . diag(norm.weight)
Intervention  mean-ablation h <- h - U U^T (h - mu), mu = fit-row mean of h, every position
Phases
  tf   teacher-forced on the 200 test rows: dP(marker class) and d log p(marker) at marker targets inside <think>,
       KL(clean || ablated) and NLL at all positions, |dh|/|h|
  gen  MATH-500 through custom_eval (mod_passk settings, 1 sample, 8192), hook registered on the harness's model

Run:
  python scripts/mi_qwen_student_causal.py tf
  python scripts/mi_qwen_student_causal.py gen --student lkd --cond rd|k32
  python scripts/mi_qwen_student_causal.py analyze
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from mi_qwen_student_probe import STUDENTS as PROBE_STUDENTS, OUT as PROBE, TENS as PROBE_TENS, MAXLEN, N_SET  # noqa: E402

OUT = ROOT / "eval_outputs/rd_newsae/student_causal"
STUDENTS = {k: PROBE_STUDENTS[k] for k in ("lkd", "rd_moderate", "rd_high", "rd_low", "rd_oct", "base")}
ALIGNER_PT = ROOT / "checkpoints/rd_newsae_moderate/final_step_94/aligner.pt"
S_LAYER, D = 12, 2048                       # layers[12] output = hidden_states[13]
KS = (8, 32, 128)
TYPES = ("rd", "rand", "randdir", "aligner")
MARKERS = {"wait", "hmm", "maybe", "perhaps", "but", "another", "therefore", "however"}   # = mi_ablation.MARKERS
GEN_CFG = {"lkd": "causal_gen_lkd.yaml", "rd_moderate": "causal_gen_rd_mod.yaml"}


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


def marker_ids(tok):
    return sorted(i for i in range(len(tok)) if tok.decode([i]).strip().lower() in MARKERS)


def load_rows_targets():
    rows = [json.loads(l) for l in (PROBE / "texts.jsonl").open()]
    T = torch.load(PROBE_TENS / "targets.pt", mmap=True)
    return rows, {"Y": T["Y"], "rd": T["rd"], "rand": T["rand"]}


def find_layer(model):
    hits = [m for n, m in model.named_modules() if re.search(rf"(^|\.)layers\.{S_LAYER}$", n)]
    assert len(hits) == 1, f"expected one layers.{S_LAYER}, got {len(hits)}"
    return hits[0]


@torch.no_grad()
def subspaces(name, model, tok, rows, T):
    p = OUT / f"subspaces_{name}.pt"
    if p.exists():
        return torch.load(p)
    fit = [i for i in range(len(rows)) if i % 2 == 0]
    Y = T["Y"]
    ys = torch.cat([Y[i] for i in fit]).float().std(0).clamp(min=1e-6).cuda()
    n, sx = 0, torch.zeros(D, dtype=torch.float64, device="cuda")
    A = torch.zeros(D, D, dtype=torch.float64, device="cuda")
    Bm = torch.zeros(D, 2 * N_SET, dtype=torch.float64, device="cuda")
    sy = torch.zeros(2 * N_SET, dtype=torch.float64, device="cuda")
    for i in fit:
        ids = tok(rows[i]["text"], truncation=True, max_length=MAXLEN)["input_ids"]
        h = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[S_LAYER + 1][0].double()
        y = (Y[i].cuda().float() / ys).double()
        assert h.shape[0] == y.shape[0], (h.shape, y.shape)
        A += h.T @ h; Bm += h.T @ y; sx += h.sum(0); sy += y.sum(0); n += h.shape[0]
    mu = sx / n
    A = A - n * torch.outer(mu, mu)
    Bm = Bm - n * torch.outer(mu, sy / n)
    W = torch.linalg.solve(A + 1e-2 * A.diagonal().mean() * torch.eye(D, device="cuda", dtype=torch.float64), Bm)
    out = {"mu": mu.float().cpu()}
    for t, sl in (("rd", slice(0, N_SET)), ("rand", slice(N_SET, 2 * N_SET))):
        U = torch.linalg.svd(W[:, sl], full_matrices=False)[0]
        for k in KS:
            out[f"{t}|k{k}"] = U[:, :k].float().cpu()
    g = torch.Generator().manual_seed(0)
    for k in KS:
        out[f"randdir|k{k}"] = torch.linalg.qr(torch.randn(D, k, generator=g))[0]
    sd = torch.load(str(ALIGNER_PT), map_location="cpu")
    M = sd["proj.weight"].float() @ torch.diag(sd["norm.weight"].float())       # [196, 2048]
    V = torch.linalg.svd(M, full_matrices=False)[2].T
    for k in KS:
        out[f"aligner|k{k}"] = V[:, :k].contiguous()
    torch.save(out, p)
    return out


class MeanAblate:
    """h <- h - U U^T (h - mu) on the output of one decoder layer, every position."""

    def __init__(self, layer, U=None, mu=None):
        self.U, self.mu, self.size = U, mu, []
        self.hk = layer.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        if self.U is None:
            return o
        hs = o[0] if isinstance(o, tuple) else o
        x = hs.float()
        d = ((x - self.mu) @ self.U) @ self.U.T
        self.size.append(float((d.norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-6)).mean()))
        new = (x - d).to(hs.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

    def remove(self):
        self.hk.remove()


def think_span(tok, text):
    enc = tok(text, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    a, b = text.find("<think>"), text.find("</think>")
    b = len(text) if b < 0 else b
    inside = [a >= 0 and s >= a + 7 and e <= b for s, e in enc["offset_mapping"]]
    return enc["input_ids"], inside


@torch.no_grad()
def tf_phase(students):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir=str(ROOT / "cache"))
    rows, T = load_rows_targets()
    mids = marker_ids(tok)
    log(f"tf: {len(rows)} rows, marker token ids {len(mids)}: {sorted({tok.decode([i]).strip().lower() for i in mids})}")
    mset = torch.zeros(len(tok), dtype=torch.bool)
    mset[mids] = True
    mid_t = torch.tensor(mids).cuda()
    te = [i for i in range(len(rows)) if i % 2 == 1]
    path = OUT / "tf.jsonl"
    done = {json.loads(l)["student"] for l in path.open()} if path.exists() else set()
    for name in students:
        if name in done:
            continue
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[name], dtype=torch.bfloat16, device_map={"": 0},
                                                     cache_dir=str(ROOT / "cache")).eval()
        t0 = time.time()
        sub = subspaces(name, model, tok, rows, T)
        log(f"  {name}: subspaces {time.time() - t0:.0f}s")
        mu = sub["mu"].cuda()
        layer = find_layer(model)
        conds = ["clean"] + [f"{t}|k{k}" for t in TYPES for k in KS]
        recs, t0 = [], time.time()
        for i in te:
            ids, inside = think_span(tok, rows[i]["text"])
            x = torch.tensor([ids]).cuda()
            tgt = x[0, 1:]
            mk = torch.tensor([inside[t + 1] and bool(mset[ids[t + 1]]) for t in range(len(ids) - 1)]).cuda()
            th = torch.tensor(inside[1:]).cuda()
            r = {"row": i, "n_mk": int(mk.sum()), "n_think": int(th.sum()), "cond": {}}
            base_lp = None
            for c in conds:
                hk = MeanAblate(layer, None if c == "clean" else sub[c].cuda(), mu)
                lp = F.log_softmax(model(x).logits[0, :-1].float(), -1)
                hk.remove()
                lpa = lp.gather(1, tgt[:, None])[:, 0]
                pm = torch.logsumexp(lp[:, mid_t], -1).exp()
                if c == "clean":
                    base_lp = lp
                    kl = torch.zeros_like(lpa)
                else:
                    kl = (base_lp.exp() * (base_lp - lp)).sum(-1)
                r["cond"][c] = {"lp_mk": float(lpa[mk].mean()) if mk.any() else None,
                                "pm_mk": float(pm[mk].mean()) if mk.any() else None,
                                "pm_think": float(pm[th].mean()) if th.any() else None,
                                "nll": float(-lpa.mean()), "nll_think": float(-lpa[th].mean()) if th.any() else None,
                                "kl": float(kl.mean()), "size": float(np.mean(hk.size)) if hk.size else 0.0}
                del lp
            del base_lp
            recs.append(r)
        with path.open("a") as f:
            f.write(json.dumps({"student": name, "records": recs}) + "\n")
        log(f"  tf {name}: {len(recs)} rows {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()


def gen_phase(student, cond):
    """custom_eval run with the ablation hook added right after the harness loads the model."""
    import custom_eval.runner as runner
    from custom_eval.config import load_config
    sub = torch.load(OUT / f"subspaces_{student}.pt")
    U, mu = sub[cond].cuda(), sub["mu"].cuda()
    orig = runner.load_model_and_tokenizer
    hooks = []

    def patched(spec, cache_dir):
        model, tok = orig(spec, cache_dir)
        hooks.append(MeanAblate(find_layer(model), U, mu))
        log(f"gen {student} {cond}: hook on layers.{S_LAYER} ({type(model).__name__})")
        return model, tok
    runner.load_model_and_tokenizer = patched
    cfg = load_config(str(ROOT / "configs/mech_interp" / GEN_CFG[student]))
    cfg.output_dir = str(OUT / "gen" / f"{student}__{cond.replace('|', '_')}")
    runner.run_evaluation(cfg)
    assert hooks and hooks[0].size, "hook never fired"
    log(f"gen {student} {cond}: done, mean |dh|/|h| {np.mean(hooks[0].size):.3f}")


def boot(x, b=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    p = max(2 * min((m <= 0).mean(), (m >= 0).mean()), 1 / b)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), p


def analyze():
    TF = {json.loads(l)["student"]: {r["row"]: r for r in json.loads(l)["records"]} for l in (OUT / "tf.jsonl").open()}
    rows = sorted(set.intersection(*[{i for i, r in R.items() if r["n_mk"] > 0} for R in TF.values()]))
    L = [f"# Qwen students: mean-ablation at hs[13], teacher-forced, {len(rows)} test rows with markers",
         "  d = ablated - clean, row means; pm_mk = P(marker class) at marker targets inside <think>; kl at all positions"]

    def d(s, c, m):
        return np.array([TF[s][i]["cond"][c][m] - TF[s][i]["cond"]["clean"][m] for i in rows])

    for s in TF:
        cl = np.mean([TF[s][i]["cond"]["clean"]["pm_mk"] for i in rows])
        L.append(f"\n## {s}  clean P(marker) {cl:.3f}  marker targets {sum(TF[s][i]['n_mk'] for i in rows)}")
        for k in KS:
            parts = []
            for t in TYPES:
                c = f"{t}|k{k}"
                sz = np.mean([TF[s][i]["cond"][c]["size"] for i in rows])
                parts.append(f"{t} {100 * d(s, c, 'pm_mk').mean():+.2f}pp kl {d(s, c, 'kl').mean():.4f} |dh| {sz:.3f}")
            L.append(f"  k{k:<3d} " + " | ".join(parts))
        for k in KS:
            e = d(s, f"rd|k{k}", "pm_mk") - d(s, f"rand|k{k}", "pm_mk")
            m, lo, hi, p = boot(e)
            e2 = d(s, f"rd|k{k}", "pm_mk") - d(s, f"randdir|k{k}", "pm_mk")
            m2, lo2, hi2, p2 = boot(e2)
            L.append(f"  E k{k:<3d} rd-rand {100 * m:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.4f} | "
                     f"rd-randdir {100 * m2:+.2f}pp [{100 * lo2:+.2f},{100 * hi2:+.2f}] p={p2:.4f}")

    L.append("\n# PRIMARY (pre-registered): k32, I = E(RD) - E(LKD), E = dP(marker|rd) - dP(marker|rand)")
    eL = d("lkd", "rd|k32", "pm_mk") - d("lkd", "rand|k32", "pm_mk")
    mL, loL, hiL, pL = boot(eL)
    L.append(f"  E(LKD) {100 * mL:+.2f}pp [{100 * loL:+.2f},{100 * hiL:+.2f}] p={pL:.4f}  -> condition 3 "
             f"{'met' if hiL < 0 else 'NOT met'}")
    verdict = hiL < 0
    for s in ("rd_moderate", "rd_high", "rd_low", "rd_oct", "base"):
        if s not in TF:
            continue
        eS = d(s, "rd|k32", "pm_mk") - d(s, "rand|k32", "pm_mk")
        m, lo, hi, p = boot(eS - eL)
        ratio = abs(m) / max(abs(mL), 1e-9)
        ok = hi < 0 and ratio >= 0.25
        tag = ""
        if s in ("rd_moderate", "rd_high"):
            verdict = verdict and ok
            tag = f"  -> {'meets' if ok else 'fails'} conditions 1-2"
        L.append(f"  I({s:11s}) {100 * m:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.4f}  |I|/|E(LKD)| {ratio:.2f}{tag}")
    L.append(f"  VERDICT: {'PASS' if verdict else 'FAIL'}")

    L.append("\n# secondary: same contrast at k8/k128 and for the other controls (descriptive)")
    for k in KS:
        for ctrl in ("rand", "randdir"):
            eL2 = d("lkd", f"rd|k{k}", "pm_mk") - d("lkd", f"{ctrl}|k{k}", "pm_mk")
            for s in ("rd_moderate", "rd_high"):
                if s in TF:
                    m, lo, hi, p = boot(d(s, f"rd|k{k}", "pm_mk") - d(s, f"{ctrl}|k{k}", "pm_mk") - eL2)
                    L.append(f"  k{k:<3d} rd-{ctrl:7s} {s:11s} - lkd {100 * m:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.4f}")
        for s in ("rd_moderate", "rd_high"):
            if s in TF:
                m, lo, hi, p = boot(d(s, f"aligner|k{k}", "pm_mk") - d("lkd", f"aligner|k{k}", "pm_mk"))
                L.append(f"  k{k:<3d} aligner dP   {s:11s} - lkd {100 * m:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.4f}")
                m, lo, hi, p = boot(d(s, f"rd|k{k}", "kl") - d(s, f"rand|k{k}", "kl")
                                    - (d("lkd", f"rd|k{k}", "kl") - d("lkd", f"rand|k{k}", "kl")))
                L.append(f"  k{k:<3d} KL rd-rand   {s:11s} - lkd {m:+.5f} [{lo:+.5f},{hi:+.5f}] p={p:.4f}")
    out = "\n".join(L)
    print(out)
    (OUT / "analysis.txt").write_text(out + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tf", "gen", "analyze"])
    ap.add_argument("--students", nargs="+", default=list(STUDENTS))
    ap.add_argument("--student")
    ap.add_argument("--cond")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "tf":
        tf_phase(a.students)
    elif a.phase == "gen":
        gen_phase(a.student, a.cond)
    else:
        analyze()


if __name__ == "__main__":
    main()
