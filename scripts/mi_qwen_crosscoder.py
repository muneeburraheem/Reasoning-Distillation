"""Exp 1 + Exp 5 (docs/mi_qwen_students.md): one shared feature dictionary for the students, read off per model.

Site     student hidden_states[13] (output of layers[12]: where RD's head reads and where the RD term changed the
         weights most, Exp 4). Models lkd (LKD-continue), rd_low, rd_mod, rd_high: lkd vs rd_low = the paper's
         comparison, rd_low -> rd_high = the RD term's strength (same run settings, alpha 0.05 -> 0.40).
Data     AM-DeepSeek rows 100,000+ (no student trained on them), "user\\nassistant" text as in training, 1,024 tokens,
         position 0 dropped. Activations are computed on the fly (never written to disk). Each model's activations are
         divided by one fixed scalar so their mean norm is sqrt(2048) (measured on the first 50 batches).
Two dictionaries trained in the same pass, 32,768 latents each, BatchTopK (k = 64 per token) + AuxK for dead latents:
  crosscoder  one encoder reads all four models at once; each latent has one decoder vector PER MODEL. A latent whose
              decoder norm is ~0 for a model is (nearly) absent from that model.
  pooled SAE  an ordinary SAE trained on tokens from all four models; encoded per model, it gives each latent's
              firing rate and strength in each model on identical text.
Phases
  train     ~100M tokens (both dictionaries)
  eval      held-out FVE / L0 / dead latents; decoder-norm contrasts; per-model firing of pooled-SAE latents
  label     step-type enrichment of every latent on the labelled MATH-500 traces (both trace sets, all four models)
  transfer  Exp 5: for each of the teacher's 196 RD features (and 196 random ones), the best-correlated pooled-SAE
            latent in each student (probe texts, test rows)
  analyze
Run: python scripts/mi_qwen_crosscoder.py train|eval|label|transfer|analyze
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from mi_qwen_students_common import D, OUT, REFL, ROOT, boot, load_model, log, tokenizer, traces

D1 = OUT / "crosscoder"
CK = ROOT / "checkpoints/mi_qwen_crosscoder"
MODELS = ("lkd", "rd_low", "rd_mod", "rd_high")
HS = 13                     # hidden_states[13] = output of layers[12]
F_LAT, K, SEQ, BSEQ = 32768, 64, 1024, 8
TOKENS = int(__import__("os").environ.get("CC_TOKENS", 100_000_000))
LR, WARM, DECAY = 1.5e-4, 500, 0.2
ROW0, ROW_HELD = 100_000, 450_000         # train rows 100k.., held-out eval rows 450k..
AUX_K, AUX_COEF, DEAD_AFTER = 512, 1 / 32, 2_000_000


class BatchTopK(nn.Module):
    """x [B, M, D] -> latents [B, F]; decoders [M, F, D]. M = 1 for the pooled SAE."""

    def __init__(self, m):
        super().__init__()
        self.M = m
        dec = torch.randn(m, F_LAT, D)
        dec = dec / dec.norm(dim=-1, keepdim=True) / math.sqrt(m)                     # joint norm over models = 1
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.permute(0, 2, 1).clone())                       # [M, D, F]
        self.b_enc = nn.Parameter(torch.zeros(F_LAT))
        self.b_dec = nn.Parameter(torch.zeros(m, D))
        self.register_buffer("thresh", torch.tensor(0.0))
        self.register_buffer("since", torch.zeros(F_LAT))

    def dec_norm(self):
        return self.W_dec.norm(dim=-1)                                                  # [M, F]

    @torch.no_grad()
    def renorm(self):
        """Each latent's decoder has joint norm 1 over models (unit norm for the pooled SAE). Without this, BatchTopK's
        score activation x decoder norm lets winning latents grow their decoders and win more (dead-latent collapse).
        Only the joint norm is fixed: the per-model split, which the analysis reads, stays free."""
        self.W_dec.div_(self.W_dec.pow(2).sum((0, 2), keepdim=True).sqrt().clamp(min=1e-8))

    def pre(self, x):
        return torch.einsum("bmd,mdf->bf", (x - self.b_dec).to(torch.bfloat16), self.W_enc.to(torch.bfloat16)).float() + self.b_enc

    def decode(self, f):
        return torch.einsum("bf,mfd->bmd", f.to(torch.bfloat16), self.W_dec.to(torch.bfloat16)).float() + self.b_dec

    def encode(self, x, train=False):
        p = torch.relu(self.pre(x))
        s = p * self.W_dec.pow(2).sum((0, 2)).sqrt()                                  # score: activation x joint decoder norm
        if train:
            # batch top-k in two stages: each token's top 4K, then the batch's top K*B among those (exact unless one
            # token would take more than 4K = 256 of the batch's slots)
            k = K * x.shape[0]
            v1, i1 = s.topk(4 * K, dim=-1)
            v, j = v1.flatten().topk(k)
            rows = torch.div(j, 4 * K, rounding_mode="floor")
            cols = i1.flatten()[j]
            f = torch.zeros_like(p)
            f[rows, cols] = p[rows, cols]
            with torch.no_grad():
                self.thresh.mul_(0.99).add_(0.01 * v[-1])
            return f, p
        return p * (s > self.thresh), p

    def loss(self, x):
        f, p = self.encode(x, train=True)
        xh = self.decode(f)
        err = x - xh
        mse = err.pow(2).sum(-1).mean()
        with torch.no_grad():
            self.since += x.shape[0]
            self.since[(f > 0).any(0)] = 0
        dead = self.since > DEAD_AFTER
        aux = torch.zeros((), device=x.device)
        if dead.any():
            pd = p * dead
            v, i = pd.topk(min(AUX_K, int(dead.sum())), dim=-1)
            fa = torch.zeros_like(pd).scatter_(-1, i, v)
            aux = (err.detach() - (self.decode(fa) - self.b_dec)).pow(2).sum(-1).mean()
        return mse + AUX_COEF * aux, mse, f, xh


def rows(start, n):
    from train_sae_norm64 import ArrowAMDeepSeek
    ds = ArrowAMDeepSeek()
    i = start
    while n is None or i < start + n:
        yield i, ds[i]["text"]
        i += 1


def batches(tok, start, n_rows=None, pos=None):
    """pos (dict, optional): pos["row"] = first row not yet used, updated before each yield (for resuming)."""
    buf = []
    for i, text in rows(start, n_rows):
        ids = tok(text, truncation=True, max_length=SEQ)["input_ids"]
        if len(ids) < 64:
            continue
        buf.append(ids)
        if len(buf) == BSEQ:
            L = min(len(b) for b in buf)
            if pos is not None:
                pos["row"] = i + 1
            yield torch.tensor([b[:L] for b in buf]).cuda()
            buf = []


@torch.no_grad()
def acts(ms, x):
    """[N, M, D] at hidden_states[13], position 0 dropped, tokens flattened."""
    out = []
    for m in ms.values():
        h = m.model(x, output_hidden_states=True).hidden_states[HS][:, 1:]
        out.append(h.reshape(-1, D).float())
    return torch.stack(out, 1)


def load_students():
    return {n: load_model(n, n_layers=HS) for n in MODELS}


def train():
    import bitsandbytes as bnb
    tok = tokenizer()
    ms = load_students()
    CK.mkdir(parents=True, exist_ok=True)
    D1.mkdir(parents=True, exist_ok=True)
    pos = {"row": ROW0}
    R = torch.load(CK / "latest.pt", weights_only=False) if (CK / "latest.pt").exists() else None
    it = batches(tok, R["row"] if R else ROW0, pos=pos)
    if R:
        scale, mean = R["scale"].cuda(), None
        log(f"crosscoder: resuming at step {R['step']}, {R['tokens'] / 1e6:.1f}M tokens, row {R['row']}", "crosscoder.log")
    nsum, vsum, ntok = torch.zeros(len(MODELS), device="cuda"), torch.zeros(len(MODELS), D, device="cuda"), 0
    for _ in range(0 if R else 50):
        a = acts(ms, next(it))
        n = a.norm(dim=-1)
        a = a[n.max(1).values < 10 * n.mean()]
        nsum += a.norm(dim=-1).sum(0); vsum += a.sum(0); ntok += a.shape[0]
    if not R:
        scale = math.sqrt(D) / (nsum / ntok)                                        # [M]
        mean = vsum / ntok * scale[:, None]                                         # [M, D], after scaling
        log(f"crosscoder: scale per model {scale.tolist()}, |mean|/sqrt(D) {(mean.norm(dim=-1) / math.sqrt(D)).tolist()},"
        f" tokens/batch ~{ntok // 50}", "crosscoder.log")
    cc, sae = BatchTopK(len(MODELS)).cuda(), BatchTopK(1).cuda()
    opts = [bnb.optim.Adam8bit(m.parameters(), lr=LR, betas=(0.9, 0.999)) for m in (cc, sae)]
    seen, step, t0, t_model = 0, 0, time.time(), 0.0
    hist = []
    if R:
        cc.load_state_dict(R["cc"]); sae.load_state_dict(R["sae"])
        opts[0].load_state_dict(R["opt_cc"]); opts[1].load_state_dict(R["opt_sae"])
        seen, step, hist = R["tokens"], R["step"], R["hist"]
        del R
    else:
        with torch.no_grad():                                                       # b_dec starts at the data mean
            cc.b_dec.copy_(mean); sae.b_dec.copy_(mean.mean(0, keepdim=True))
    steps_est = TOKENS // (BSEQ * (SEQ - 1) * 0.8)
    while seen < TOKENS:
        tm = time.time()
        x = acts(ms, next(it)) * scale[None, :, None]
        torch.cuda.synchronize(); t_model += time.time() - tm
        keep = x.norm(dim=-1).max(1).values < 10 * math.sqrt(D)                      # drop rare massive-norm tokens
        x = x[keep]
        frac = seen / TOKENS
        lr = LR * min(1.0, (step + 1) / WARM) * (1.0 if frac < 1 - DECAY else (1 - frac) / DECAY)
        rec = {}
        # the pooled SAE sees each model's tokens as a separate chunk (memory), one optimizer step per batch
        for name, model, opt, chunks in (("cc", cc, opts[0], [x]), ("sae", sae, opts[1], [x[:, m:m + 1] for m in range(len(MODELS))])):
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            mses, l0s = [], []
            for inp in chunks:
                loss, mse, f, _ = model.loss(inp)
                (loss / len(chunks)).backward()
                mses.append(float(mse.detach())); l0s.append(float((f > 0).sum(-1).float().mean()))
                del loss, f
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.renorm()
            rec[name] = (np.mean(mses) / D, np.mean(l0s), int((model.since > DEAD_AFTER).sum()))
        seen += x.shape[0]
        step += 1
        if step % 200 == 0:
            msg = (f"step {step} tokens {seen / 1e6:.1f}M lr {lr:.2e} | cc mse/D {rec['cc'][0]:.4f} L0 {rec['cc'][1]:.0f} dead {rec['cc'][2]}"
                   f" | sae mse/D {rec['sae'][0]:.4f} L0 {rec['sae'][1]:.0f} dead {rec['sae'][2]} | {time.time() - t0:.0f}s (models {t_model:.0f}s)"
                   f" (~{steps_est} steps)")
            log(msg, "crosscoder.log")
            hist.append({"step": step, "tokens": seen, **{k: v for k, v in rec.items()}})
        if seen >= TOKENS:
            torch.save({"cc": cc.state_dict(), "sae": sae.state_dict(), "scale": scale.cpu(), "tokens": seen, "step": step},
                       CK / "final.pt")
        elif step % 1000 == 0:                                                      # resumable checkpoint, written atomically
            torch.save({"cc": cc.state_dict(), "sae": sae.state_dict(), "opt_cc": opts[0].state_dict(),
                        "opt_sae": opts[1].state_dict(), "scale": scale.cpu(), "tokens": seen, "step": step,
                        "row": pos["row"], "hist": hist}, CK / "latest.tmp")
            (CK / "latest.tmp").replace(CK / "latest.pt")
    (CK / "latest.pt").unlink(missing_ok=True)
    (D1 / "train_history.json").write_text(json.dumps(hist))
    log(f"crosscoder: trained {seen / 1e6:.1f}M tokens, {time.time() - t0:.0f}s", "crosscoder.log")


def load_dicts():
    S = torch.load(CK / "final.pt")
    cc, sae = BatchTopK(len(MODELS)).cuda(), BatchTopK(1).cuda()
    cc.load_state_dict(S["cc"]); sae.load_state_dict(S["sae"])
    return cc.eval(), sae.eval(), S["scale"].cuda()


@torch.no_grad()
def evaluate():
    tok = tokenizer()
    ms = load_students()
    cc, sae, scale = load_dicts()
    n_tok = 0
    se = {"cc": torch.zeros(len(MODELS), device="cuda"), "sae": torch.zeros(len(MODELS), device="cuda")}
    var = torch.zeros(len(MODELS), device="cuda")
    s1, s2 = torch.zeros(len(MODELS), D, device="cuda"), torch.zeros(len(MODELS), device="cuda")
    l0 = {"cc": 0.0, "sae": 0.0}
    fired_cc = torch.zeros(F_LAT, device="cuda")
    freq = torch.zeros(len(MODELS), F_LAT, device="cuda")      # pooled-SAE firing count per model
    mag = torch.zeros(len(MODELS), F_LAT, device="cuda")
    per_seq = []                                               # per batch firing counts for bootstrap (subset of latents later)
    for b, x in enumerate(batches(tok, ROW_HELD, 2_000)):
        a = acts(ms, x) * scale[None, :, None]
        a = a[a.norm(dim=-1).max(1).values < 10 * math.sqrt(D)]
        n_tok += a.shape[0]
        s1 += a.sum(0); s2 += a.pow(2).sum((0, 2))
        f, _ = cc.encode(a)
        se["cc"] += (a - cc.decode(f)).pow(2).sum((0, 2)); l0["cc"] += float((f > 0).sum())
        fired_cc += (f > 0).sum(0)
        counts = torch.zeros(len(MODELS), F_LAT, device="cuda")
        for m in range(len(MODELS)):
            fs, _ = sae.encode(a[:, m:m + 1])
            se["sae"][m] += (a[:, m:m + 1] - sae.decode(fs)).pow(2).sum()
            l0["sae"] += float((fs > 0).sum()) / len(MODELS)
            counts[m] = (fs > 0).sum(0)
            mag[m] += fs.sum(0)
        freq += counts
        per_seq.append(counts.cpu().to(torch.int32))
    var = s2 - s1.pow(2).sum(-1) / n_tok
    res = {"tokens": n_tok, "models": MODELS,
           "fve_cc": (1 - se["cc"] / var).tolist(), "fve_sae": (1 - se["sae"] / var).tolist(),
           "l0_cc": l0["cc"] / n_tok, "l0_sae": l0["sae"] / n_tok,
           "dead_cc": int((fired_cc == 0).sum()), "dead_sae": int((freq.sum(0) == 0).sum())}
    D1.mkdir(parents=True, exist_ok=True)
    (D1 / "eval.json").write_text(json.dumps(res, indent=1))
    torch.save({"freq": freq.cpu(), "mag": mag.cpu(), "per_batch": torch.stack(per_seq), "n_tok": n_tok,
                "dec_norm": cc.dec_norm().cpu(), "dec": cc.W_dec.detach().cpu().to(torch.float16)}, CK / "eval_stats.pt")
    log(f"crosscoder eval: {json.dumps(res)}", "crosscoder.log")


@torch.no_grad()
def label():
    """mean activation of every latent on tokens of reflection steps vs other steps, per model, same text."""
    tok = tokenizer()
    trs = traces(tok)
    ms = {n: load_model(n, n_layers=HS) for n in MODELS}
    cc, sae, scale = load_dicts()
    S = {k: torch.zeros(len(MODELS), F_LAT, dtype=torch.float64, device="cuda") for k in ("refl", "oth", "refl_n", "oth_n")}
    Scc = {k: torch.zeros(F_LAT, dtype=torch.float64, device="cuda") for k in ("refl", "oth")}
    per_trace = []
    nr = no = 0
    for t in trs:
        if not t["bpos"]:
            continue
        lab = torch.zeros(len(t["ids"]), dtype=torch.long)           # 0 prompt/unlabelled, 1 reflection, 2 other
        b = t["bpos"] + [len(t["ids"]) - 1]
        for k, l in enumerate(t["blab"]):
            lab[b[k] + 1:b[k + 1] + 1] = 1 if l == REFL else 2
        lab = lab[1:].cuda()
        x = torch.tensor([t["ids"]]).cuda()
        a = acts(ms, x) * scale[None, :, None]
        r, o = lab == 1, lab == 2
        if not r.any() or not o.any():
            continue
        nr += int(r.sum()); no += int(o.sum())
        row = {"qi": t["qi"], "src": t["src"], "refl_mean": []}
        for m in range(len(MODELS)):
            fs, _ = sae.encode(a[:, m:m + 1])
            S["refl"][m] += fs[r].sum(0).double(); S["oth"][m] += fs[o].sum(0).double()
            S["refl_n"][m] += (fs[r] > 0).sum(0).double(); S["oth_n"][m] += (fs[o] > 0).sum(0).double()
            row["refl_mean"].append(fs[r].mean(0))
        f, _ = cc.encode(a)
        Scc["refl"] += f[r].sum(0).double(); Scc["oth"] += f[o].sum(0).double()
        per_trace.append(row)
    # per-trace mean activation on reflection tokens, kept only for the latents analyze() will look at (top 512 by
    # enrichment), to compare models paired over traces
    enr = (S["refl"].sum(0) / nr + 1e-6) / (S["oth"].sum(0) / no + 1e-6)
    sup = S["refl_n"].sum(0) >= 200
    top = torch.where(sup, enr, torch.zeros_like(enr)).topk(512).indices
    pt = torch.stack([torch.stack([r["refl_mean"][m][top] for m in range(len(MODELS))]) for r in per_trace]).cpu()
    torch.save({"S": {k: v.cpu() for k, v in S.items()}, "Scc": {k: v.cpu() for k, v in Scc.items()}, "n_refl": nr,
                "n_oth": no, "top": top.cpu(), "per_trace": pt, "trace_q": [r["qi"] for r in per_trace],
                "trace_src": [r["src"] for r in per_trace]}, CK / "label_stats.pt")
    log(f"crosscoder label: {len(per_trace)} traces, {nr} reflection / {no} other tokens", "crosscoder.log")


@torch.no_grad()
def transfer():
    """Exp 5: best |corr| between each teacher feature and any pooled-SAE latent, per student, probe test rows."""
    from mi_qwen_student_probe import OUT as PROBE, TENS as PROBE_TENS, MAXLEN
    tok = tokenizer()
    ms = {n: load_model(n, n_layers=HS) for n in MODELS}
    _, sae, scale = load_dicts()
    texts = [json.loads(l) for l in (PROBE / "texts.jsonl").open()]
    T = torch.load(PROBE_TENS / "targets.pt", mmap=True)
    nT = T["Y"][0].shape[1]
    st = {m: {k: torch.zeros(F_LAT, nT, dtype=torch.float64, device="cuda") for k in ("xy",)} | {
        "x": torch.zeros(F_LAT, dtype=torch.float64, device="cuda"), "xx": torch.zeros(F_LAT, dtype=torch.float64, device="cuda")}
        for m in MODELS}
    sy = torch.zeros(nT, dtype=torch.float64, device="cuda"); syy = torch.zeros_like(sy)
    n = 0
    for i in range(1, len(texts), 2):
        ids = tok(texts[i]["text"], truncation=True, max_length=MAXLEN)["input_ids"]
        y = T["Y"][i].cuda().double()[1:]
        a = acts(ms, torch.tensor([ids]).cuda()) * scale[None, :, None]
        assert a.shape[0] == y.shape[0], (a.shape, y.shape)
        sy += y.sum(0); syy += y.pow(2).sum(0); n += y.shape[0]
        for j, m in enumerate(MODELS):
            fs, _ = sae.encode(a[:, j:j + 1])
            fs = fs.double()
            st[m]["x"] += fs.sum(0); st[m]["xx"] += fs.pow(2).sum(0); st[m]["xy"] += fs.T @ y
    out = {}
    for m in MODELS:
        cov = st[m]["xy"] / n - torch.outer(st[m]["x"] / n, sy / n)
        sx = (st[m]["xx"] / n - (st[m]["x"] / n) ** 2).clamp(min=1e-12).sqrt()
        syd = (syy / n - (sy / n) ** 2).clamp(min=1e-12).sqrt()
        c = cov / torch.outer(sx, syd)
        c[sx < 1e-6] = 0
        best, arg = c.max(0)
        out[m] = {"best": best.cpu().tolist(), "arg": arg.cpu().tolist()}
    (D1 / "transfer.json").write_text(json.dumps({"n_tokens": n, "n_rd": len(T["rd"]), **out}))
    log(f"crosscoder transfer: {n} tokens", "crosscoder.log")


def analyze():
    L = ["# Exp 1 · Shared dictionaries for the students (hidden_states[13]); Exp 5 · teacher-feature transfer"]
    E = json.loads((D1 / "eval.json").read_text())
    L.append(f"\n## Quality, {E['tokens']:,} held-out tokens per model")
    L.append("  FVE crosscoder " + "  ".join(f"{m} {v:.3f}" for m, v in zip(MODELS, E["fve_cc"])) + f"  | L0 {E['l0_cc']:.1f}  dead {E['dead_cc']}")
    L.append("  FVE pooled SAE " + "  ".join(f"{m} {v:.3f}" for m, v in zip(MODELS, E["fve_sae"])) + f"  | L0 {E['l0_sae']:.1f}  dead {E['dead_sae']}")
    ES = torch.load(CK / "eval_stats.pt")
    dn = ES["dec_norm"].double()                                    # [M, F]
    live = ES["freq"].sum(0) > 0
    idx = {m: i for i, m in enumerate(MODELS)}

    def rel(a, b):
        return (dn[idx[b]] / (dn[idx[a]] + dn[idx[b]]).clamp(min=1e-9))[live].numpy()
    L.append("\n## Crosscoder: relative decoder norm r = |d_B| / (|d_A| + |d_B|) per live latent (0.5 = equally present)")
    L.append("  pair (A -> B)          n    r<0.1  r<0.3  0.3-0.7  r>0.7  r>0.9   sd(r)")
    for a, b, what in (("lkd", "rd_low", "mostly the RD run"), ("rd_low", "rd_high", "RD term strength"),
                       ("lkd", "rd_mod", "the paper's pair"), ("rd_low", "rd_mod", "")):
        r = rel(a, b)
        L.append(f"  {a:>7s} -> {b:7s} {len(r):6d}  {np.sum(r < .1):5d}  {np.sum(r < .3):5d}  {np.sum((r >= .3) & (r <= .7)):6d}  "
                 f"{np.sum(r > .7):5d}  {np.sum(r > .9):5d}   {r.std():.4f}   {what}")
    dec = ES["dec"].float()
    cosm = torch.nn.functional.cosine_similarity(dec[idx["lkd"]], dec[idx["rd_mod"]], dim=-1)[live]
    L.append(f"  decoder cosine lkd vs rd_mod per latent: median {cosm.median():.4f}, 1st pct {cosm.quantile(0.01):.4f}")
    fr = ES["freq"].double() / ES["n_tok"]
    L.append("\n## Pooled SAE: firing rate per model on identical held-out text")
    pb = ES["per_batch"].double()                                   # [batches, M, F]
    for a, b in (("lkd", "rd_mod"), ("rd_low", "rd_high"), ("lkd", "rd_low")):
        ok = (fr[idx[a]] + fr[idx[b]]) > 1e-4
        lr = torch.log2((fr[idx[b]] + 1e-7) / (fr[idx[a]] + 1e-7))[ok]
        L.append(f"  {a} -> {b}: latents with rate > 1e-4: {int(ok.sum())}; |log2 ratio| > 0.5: {int((lr.abs() > .5).sum())}, "
                 f"> 1: {int((lr.abs() > 1).sum())}; sd {lr.std():.4f}")
    if (CK / "label_stats.pt").exists():
        LS = torch.load(CK / "label_stats.pt")
        S = LS["S"]
        nr, no = LS["n_refl"], LS["n_oth"]
        enr = (S["refl"].sum(0) / nr + 1e-6) / (S["oth"].sum(0) / no + 1e-6)
        sup = S["refl_n"].sum(0) >= 200
        L.append(f"\n## Composition: step-type enrichment of pooled-SAE latents ({nr:,} reflection / {no:,} other tokens, all models)")
        L.append(f"  latents with >= 200 firings on reflection tokens: {int(sup.sum())}; of these enriched x2: "
                 f"{int(((enr > 2) & sup).sum())}, x4: {int(((enr > 4) & sup).sum())}; depleted x2: {int(((enr < .5) & sup).sum())}")
        L.append("  per model, mean activation on reflection tokens of the top-512 reflection-enriched latents, paired over traces:")
        pt = LS["per_trace"].double()                               # [traces, M, 512]
        qd = {}
        for k, q in enumerate(LS["trace_q"]):
            qd.setdefault(q, []).append(k)
        tot = pt.sum(-1)                                            # summed over the 512 latents
        for a, b in (("lkd", "rd_mod"), ("lkd", "rd_low"), ("rd_low", "rd_high"), ("lkd", "rd_high")):
            d = [float((tot[ks, idx[b]] - tot[ks, idx[a]]).mean() / tot[ks, idx[a]].mean()) for ks in qd.values()]
            m, lo, hi, p = boot(d)
            L.append(f"    {b} vs {a}: {100 * m:+.2f}% [{100 * lo:+.2f},{100 * hi:+.2f}] p={p:.4f}")
        Scc = LS["Scc"]
        ecc = (Scc["refl"] / nr + 1e-6) / (Scc["oth"] / no + 1e-6)
        r = (dn[idx["rd_mod"]] / (dn[idx["lkd"]] + dn[idx["rd_mod"]]).clamp(min=1e-9))
        for lo_, hi_, nm in ((0, .4, "LKD-leaning (r<0.4)"), (.4, .6, "shared"), (.6, 1.01, "RD-leaning (r>0.6)")):
            sel = live & (r >= lo_) & (r < hi_)
            if sel.any():
                L.append(f"  crosscoder {nm:20s} n={int(sel.sum()):6d}  median reflection enrichment {ecc[sel].median():.3f}")
    if (D1 / "transfer.json").exists():
        Tj = json.loads((D1 / "transfer.json").read_text())
        nr_ = Tj["n_rd"]
        L.append(f"\n## Exp 5 · Teacher-feature transfer: best correlation of each teacher feature with any student latent ({Tj['n_tokens']:,} tokens)")
        for part, sl in (("rd set", slice(0, nr_)), ("rand set", slice(nr_, None))):
            line = f"  {part:8s} " + "  ".join(f"{m} {np.mean(Tj[m]['best'][sl]):.4f}" for m in MODELS)
            L.append(line)
            for a, b in (("lkd", "rd_mod"), ("rd_low", "rd_high")):
                d = np.array(Tj[b]["best"][sl]) - np.array(Tj[a]["best"][sl])
                m, lo, hi, p = boot(d)
                L.append(f"    {b} - {a}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}; features with a match r > 0.7: "
                         f"{a} {int(np.sum(np.array(Tj[a]['best'][sl]) > .7))}, {b} {int(np.sum(np.array(Tj[b]['best'][sl]) > .7))}")
    out = "\n".join(L)
    print(out)
    (D1 / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "eval", "label", "transfer", "analyze"])
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "label": label, "transfer": transfer, "analyze": analyze}[a.phase]()
