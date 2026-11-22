"""Llama students: do they carry the teacher's ReasonScore SAE features? (representation phase)

RD-Llama trained student block 10's output (hidden_states[11]) through an aligner (LayerNorm +
Linear 2048->4096) into the teacher's frozen L19 SAE, matching the teacher's top-196 ReasonScore
features (h * 64/|h|, ReLU encode, x decoder-column norm; losses/alignment_loss.py). The question
here: after training, can those teacher features be read out of RD's student better than out of
LKD's (same data, same KD, no alignment term) or base Llama-3.2-1B's?

Text   held-out AM-DeepSeek math rows (am_0.5M minus any question in am_0.9M, as RUN_LOG 8u),
       formatted as in training ("question\\n<think>..."), first 512 tokens (the trained context),
       student tokenizer, BOS position dropped. Even rows = train (80% fit / 20% lambda
       selection), odd rows = test. Texts are saved for later phases.
Targets teacher DeepSeek-R1-Distill-Llama-8B in 8-bit (as in training), hidden_states[20], exactly
       the trainer's encode. Two sets: rd = the 196 RD features; rand = 196 features
       density-matched to them (half-decade bins of density on the train text), the control for
       "KD made the student more teacher-like in general".
Probes, per student, fitted on train, scored on test:
  ridge     every layer hidden_states[1..16], LayerNorm (no affine) + bias, closed form, lambda
            chosen on the 20% split; per-feature R^2
  pipeline  RD's own readout at hidden_states[11]: a fresh aligner (LayerNorm + Linear) trained
            with RD's loss through the frozen SAE, one per feature set; plus RD's trained aligner
            on the RD student (what training itself achieved). From a random (Xavier) start every
            fresh aligner collapses to "all features off" (zero gradient through the ReLU), so
            phase pipe starts it from the ridge map LayerNorm(h_student) -> teacher h * 64/|h|
            (fitted on train text) and then trains it with RD's loss
Statistics: per test sequence sums (SSE, sum y, sum y^2, n, nonzero) so R^2 can be bootstrapped
over sequences; RD - LKD paired on the same sequences, and the same difference for the rand set.

Causal phases (directions fixed from the train rows before any test is run):
  subspaces per student at hidden_states[11] (output of block 10), k in (8, 32, 128):
    rd       top-k left singular vectors of the ridge map h -> standardised RD feature targets
    rand     the same for the density-matched random feature set
    randdir  a random k-dim subspace (QR of a Gaussian)
    aligner  top-k right singular vectors of RD's own readout W_enc[rd] . proj . diag(ln_w)
             (the same map for every student)
  intervention: mean-ablation, h <- h - U U^T (h - mu), mu = train mean of h[11]
  tf    teacher-forced on the 400 test rows, every position ablated: at marker targets inside
        <think> (next token in the ReasonScore vocabulary) d log p(actual), d P(marker class); at
        all positions KL(clean || ablated), d NLL; |dh|/|h|
  gen   GSM8K (MATH-500 is at the floor for these students, 1.5-1.8%), first 500 questions x 2
        samples, T 0.6, top_p 0.95, cap 4096, native prompt and the arm's own scoring (strict /
        lenient under the commitment gate), LKD and RD students, conditions none and k=32 for
        rd / rand / randdir / aligner (k fixed in advance)

Run: python scripts/mi_llama_students.py --phases probe pipe tf gen
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "llama_eval"))
from eval_llama_passk import MODELS  # noqa: E402
from eval_llama_teacher_forced_kl import BASE, LLAMA_DIR, TEACHER, heldout_rows, norm  # noqa: E402

SAE_PT = LLAMA_DIR / "sae_input/sae_checkpoint.pt"       # md5 0bc126ce..., = l19_sae_l1_0.0009.pt
RANK_PT = LLAMA_DIR / "sae_input/reason_scores.pt"       # md5 b324df16...
ALIGNER_PT = Path(MODELS["reasondistill_epoch2"]) / "aligner.pt"
STUDENTS = {"base": BASE, "lkd": str(MODELS["logit_kd_epoch2"]), "rd": str(MODELS["reasondistill_epoch2"])}
N_SET, T_LAYER, S_LAYER, NORM, MAXLEN = 196, 19, 10, 64.0, 512
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)   # x mean diagonal of X^T X
OUT = ROOT / "eval_outputs/mi_students/llama"


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


def texts(n, seed=0, scan=60000):
    p = OUT / "texts.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.open()]
    from datasets import load_from_disk
    train = load_from_disk(str(LLAMA_DIR / "cache/am_0.9M_text"))
    train_q = {norm(t.split("\n<think>", 1)[0]) for t in train["text"]}
    del train
    rows = heldout_rows(n, seed, scan, train_q)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_sae():
    sd = torch.load(str(SAE_PT), map_location="cpu")
    we, be, wd = sd["encoder.weight"].float(), sd["encoder.bias"].float(), sd["decoder.weight"].float()
    return we, be, wd.norm(dim=0)


def sae_feats(h, we, be, dn):
    """The trainer's encode: h * 64/|h|, ReLU(W_enc h + b), x decoder-column norm. we/be/dn subset."""
    h = h.float()
    h = h * (NORM / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return F.relu(h @ we.T + be) * dn


@torch.no_grad()
def teacher_targets(tok, rows):
    p = OUT / "targets.pt"
    if p.exists():
        return torch.load(p)
    q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=q, device_map={"": 0},
                                                 dtype=torch.bfloat16, cache_dir=str(LLAMA_DIR / "cache")).eval()
    we, be, dn = (x.cuda() for x in load_sae())
    rd = torch.load(str(RANK_PT), map_location="cpu")["sorted_indices"][:N_SET].long().tolist()
    L = we.shape[0]
    nz, n_tr, H = torch.zeros(L, device="cuda"), 0, []
    for i, r in enumerate(rows):
        ids = tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"]
        h = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[T_LAYER + 1][0, 1:]
        H.append(h.to(torch.bfloat16).cpu())
        if i % 2 == 0:                                         # density on train rows only
            nz += (sae_feats(h, we, be, dn) > 0).float().sum(0)
            n_tr += h.shape[0]
        if i % 100 == 0:
            log(f"  teacher {i}/{len(rows)}")
    dens = (nz / n_tr).cpu().numpy()
    # density-matched random set (half-decade bins, as mi_ablation)
    bins = lambda d: int(np.clip(np.floor(np.log10(max(d, 1e-7)) * 2), -14, 0))  # noqa: E731
    by_bin = {}
    for f in sorted(set(range(L)) - set(rd)):
        if dens[f] > 0:
            by_bin.setdefault(bins(dens[f]), []).append(f)
    rng, rand = random.Random(0), []
    for f in rd:
        b = bins(dens[f])
        cand = next(by_bin[x] for x in sorted(by_bin, key=lambda x: (abs(x - b), x)) if by_bin[x])
        rand.append(rng.choice([c for c in cand if c not in rand] or cand))
    feats = rd + rand
    Y = [sae_feats(h.cuda(), we[feats], be[feats], dn[feats]).cpu() for h in H]
    out = {"rd": rd, "rand": rand, "density": {f: float(dens[f]) for f in feats}, "Y": Y,
           "n_tokens": [y.shape[0] for y in Y]}
    torch.save(out, p)
    log(f"targets: {len(Y)} rows, {sum(out['n_tokens'])} tokens; rd density median "
        f"{np.median([dens[f] for f in rd]):.2e}, rand {np.median([dens[f] for f in rand]):.2e}; "
        f"rd nonzero on train {int((dens[rd] > 0).sum())}/{N_SET}")
    del model
    torch.cuda.empty_cache()
    return out


def lnorm(x):
    return F.layer_norm(x.float(), x.shape[-1:])


def seq_stats(pred, y):
    """Per-feature [SSE, sum y, sum y^2, n, nonzero] for one sequence."""
    n = torch.full_like(y[0], y.shape[0])
    return torch.stack([((pred - y) ** 2).sum(0), y.sum(0), (y ** 2).sum(0), n, (y > 0).float().sum(0)]).double().cpu()


@torch.no_grad()
def student_hidden(model, tok, r):
    ids = tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"]
    hs = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
    return [h[0, 1:] for h in hs]                               # 17 entries, BOS dropped


def probe_student(name, tok, rows, T):
    p = OUT / f"probe_{name}.pt"
    if p.exists():
        return
    model = AutoModelForCausalLM.from_pretrained(STUDENTS[name], dtype=torch.bfloat16, device_map={"": 0},
                                                 cache_dir=str(LLAMA_DIR / "cache")).eval()
    Y = T["Y"]
    tr = [i for i in range(len(rows)) if i % 2 == 0]
    te = [i for i in range(len(rows)) if i % 2 == 1]
    fit, val = tr[:int(0.8 * len(tr))], set(tr[int(0.8 * len(tr)):])
    ys = torch.cat([Y[i] for i in fit]).std(0).clamp(min=1e-6).cuda()   # standardise targets for lambda choice
    nl, d, k = 16, 2049, Y[0].shape[1]
    A = {g: torch.zeros(nl, d, d, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    B = {g: torch.zeros(nl, d, k, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    C = {g: torch.zeros(k, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    X11 = {}                                                    # hidden_states[11] for the pipeline probe
    t0 = time.time()
    for i in tr:
        hs = student_hidden(model, tok, rows[i])
        y = Y[i].cuda() / ys
        g = "val" if i in val else "fit"
        for l in range(nl):
            x = torch.cat([lnorm(hs[l + 1]), torch.ones_like(hs[0][:, :1], dtype=torch.float32)], 1)
            A[g][l] += (x.T @ x).double()
            B[g][l] += (x.T @ y).double()
        C[g] += (y.double() ** 2).sum(0)
        X11[i] = hs[S_LAYER + 1].cpu()
    log(f"  {name}: train stats {time.time() - t0:.0f}s")
    W, chosen = [], []
    for l in range(nl):
        dg = A["fit"][l].diagonal().mean()
        best = None
        for lam in LAMBDAS:
            w = torch.linalg.solve(A["fit"][l] + lam * dg * torch.eye(d, device="cuda", dtype=torch.float64), B["fit"][l])
            sse = C["val"].sum() - 2 * (w * B["val"][l]).sum() + (w * (A["val"][l] @ w)).sum()
            if best is None or sse < best[0]:
                best = (float(sse), lam)
        At, Bt = A["fit"][l] + A["val"][l], B["fit"][l] + B["val"][l]
        w = torch.linalg.solve(At + best[1] * At.diagonal().mean() * torch.eye(d, device="cuda", dtype=torch.float64), Bt)
        W.append((w * ys.double()).float())                    # back to target units
        chosen.append(best[1])
    del A, B
    torch.cuda.empty_cache()
    log(f"  {name}: ridge lambdas {chosen}")

    # pipeline probe: fresh aligner trained with RD's loss through the frozen SAE, one per set
    we, be, dn = (x.cuda() for x in load_sae())
    feats = T["rd"] + T["rand"]
    sets = {"rd": slice(0, N_SET), "rand": slice(N_SET, 2 * N_SET)}
    Xtr = torch.cat([X11[i] for i in tr])
    Ytr = torch.cat([Y[i] for i in tr])
    aligners = {}
    for s, sl in sets.items():
        torch.manual_seed(0)
        lnm = torch.nn.LayerNorm(2048).cuda()
        proj = torch.nn.Linear(2048, 4096, bias=False).cuda()
        torch.nn.init.xavier_uniform_(proj.weight)
        opt = torch.optim.Adam(list(lnm.parameters()) + list(proj.parameters()), lr=1e-3)
        fs = feats[sl]
        we_s, be_s, dn_s = we[fs], be[fs], dn[fs]
        for ep in range(4):
            perm = torch.randperm(Xtr.shape[0])
            tot = 0.0
            for b in range(0, len(perm), 4096):
                j = perm[b:b + 4096]
                pred = sae_feats(proj(lnm(Xtr[j].cuda().float())), we_s, be_s, dn_s)
                loss = ((pred - Ytr[j][:, sl].cuda()) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(j)
            log(f"  {name}: pipeline probe {s} epoch {ep} train mse {tot / len(perm):.4e}")
        aligners[s] = (lnm.eval(), proj.eval())
    rd_aligner = None
    if name == "rd":
        sd = torch.load(str(ALIGNER_PT), map_location="cpu")
        lnm = torch.nn.LayerNorm(2048).cuda()
        lnm.load_state_dict({"weight": sd["norm.weight"], "bias": sd["norm.bias"]})
        proj = torch.nn.Linear(2048, 4096, bias=False).cuda()
        proj.load_state_dict({"weight": sd["proj.weight"]})
        rd_aligner = (lnm.eval(), proj.eval())
    del Xtr, Ytr, X11

    # test pass
    ridge = torch.zeros(len(te), nl, 5, 2 * N_SET, dtype=torch.float64)
    pipe = {s: torch.zeros(len(te), 5, N_SET, dtype=torch.float64) for s in sets}
    own = torch.zeros(len(te), 5, N_SET, dtype=torch.float64) if rd_aligner else None
    with torch.no_grad():
        for n, i in enumerate(te):
            hs = student_hidden(model, tok, rows[i])
            y = Y[i].cuda()
            for l in range(nl):
                x = torch.cat([lnorm(hs[l + 1]), torch.ones_like(hs[0][:, :1], dtype=torch.float32)], 1)
                ridge[n, l] = seq_stats(x @ W[l].cuda(), y)
            for s, sl in sets.items():
                lnm, proj = aligners[s]
                fs = feats[sl]
                pipe[s][n] = seq_stats(sae_feats(proj(lnm(hs[S_LAYER + 1].float())), we[fs], be[fs], dn[fs]), y[:, sl])
            if rd_aligner:
                lnm, proj = rd_aligner
                fs = feats[sets["rd"]]
                own[n] = seq_stats(sae_feats(proj(lnm(hs[S_LAYER + 1].float())), we[fs], be[fs], dn[fs]), y[:, sets["rd"]])
    torch.save({"test_rows": te, "ridge": ridge, "pipe": pipe, "own": own, "lambdas": chosen}, p)
    log(f"  {name}: done {time.time() - t0:.0f}s")
    del model
    torch.cuda.empty_cache()


def pipe_phase(tok, rows, T):
    """Pipeline probe with a least-squares start (see docstring)."""
    p = OUT / "pipe.pt"
    if p.exists():
        return
    q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=q, device_map={"": 0},
                                                   dtype=torch.bfloat16, cache_dir=str(LLAMA_DIR / "cache")).eval()
    studs = {n: AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map={"": 0},
                                                     cache_dir=str(LLAMA_DIR / "cache")).eval() for n, m in STUDENTS.items()}
    tr = [i for i in range(len(rows)) if i % 2 == 0]
    te = [i for i in range(len(rows)) if i % 2 == 1]
    A = {n: torch.zeros(2048, 2048, dtype=torch.float64, device="cuda") for n in studs}
    Bm = {n: torch.zeros(2048, 4096, dtype=torch.float64, device="cuda") for n in studs}
    X = {n: {} for n in studs}
    t0 = time.time()
    with torch.no_grad():
        for i in tr + te:
            ids = torch.tensor([tok(rows[i]["text"], truncation=True, max_length=MAXLEN)["input_ids"]]).cuda()
            z = None
            if i in tr:
                z = teacher(ids, output_hidden_states=True).hidden_states[T_LAYER + 1][0, 1:].float()
                z = z * (NORM / z.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            for n, m in studs.items():
                h = m(ids, output_hidden_states=True).hidden_states[S_LAYER + 1][0, 1:]
                X[n][i] = h.cpu()
                if z is not None:
                    x = lnorm(h)
                    A[n] += (x.T @ x).double()
                    Bm[n] += (x.T @ z).double()
    del teacher, studs
    torch.cuda.empty_cache()
    log(f"  pipe: hidden states and teacher map stats {time.time() - t0:.0f}s")
    we, be, dn = (x.cuda() for x in load_sae())
    feats = T["rd"] + T["rand"]
    sets = {"rd": slice(0, N_SET), "rand": slice(N_SET, 2 * N_SET)}
    Y = T["Y"]
    res = {"test_rows": te}
    for n in X:
        W0 = torch.linalg.solve(A[n] + 1e-2 * A[n].diagonal().mean() * torch.eye(2048, device="cuda", dtype=torch.float64),
                                Bm[n]).float()                  # [2048, 4096]
        Xtr = torch.cat([X[n][i] for i in tr])
        Ytr = torch.cat([Y[i] for i in tr])
        for s, sl in sets.items():
            fs = feats[sl]
            we_s, be_s, dn_s = we[fs], be[fs], dn[fs]
            lnm = torch.nn.LayerNorm(2048, elementwise_affine=True).cuda()
            proj = torch.nn.Linear(2048, 4096, bias=False).cuda()
            with torch.no_grad():
                proj.weight.copy_(W0.T)

            def test_stats():
                st = torch.zeros(len(te), 5, N_SET, dtype=torch.float64)
                with torch.no_grad():
                    for k, i in enumerate(te):
                        st[k] = seq_stats(sae_feats(proj(lnm(X[n][i].cuda().float())), we_s, be_s, dn_s), Y[i][:, sl].cuda())
                return st
            res[f"{n}|{s}|init"] = test_stats()
            opt = torch.optim.Adam(list(lnm.parameters()) + list(proj.parameters()), lr=1e-4)
            torch.manual_seed(0)
            for ep in range(4):
                perm = torch.randperm(Xtr.shape[0])
                tot = 0.0
                for b in range(0, len(perm), 2048):
                    j = perm[b:b + 2048]
                    pred = sae_feats(proj(lnm(Xtr[j].cuda().float())), we_s, be_s, dn_s)
                    loss = ((pred - Ytr[j][:, sl].cuda()) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(j)
                log(f"  pipe {n} {s}: epoch {ep} train mse {tot / len(perm):.4e}")
            res[f"{n}|{s}|trained"] = test_stats()
        del Xtr, Ytr
    torch.save(res, p)
    log("  pipe: done")


KS_C = (8, 32, 128)


def subspaces(name, model, tok, rows, T):
    """Fit the ridge map at h[11] on train rows (raw h, centred) and build the k-dim subspaces."""
    p = OUT / f"subspaces_{name}.pt"
    if p.exists():
        return torch.load(p)
    tr = [i for i in range(len(rows)) if i % 2 == 0]
    Y = T["Y"]
    ys = torch.cat([Y[i] for i in tr]).std(0).clamp(min=1e-6).cuda()
    n, sx = 0, torch.zeros(2048, dtype=torch.float64, device="cuda")
    A = torch.zeros(2048, 2048, dtype=torch.float64, device="cuda")
    Bm = torch.zeros(2048, 2 * N_SET, dtype=torch.float64, device="cuda")
    sy = torch.zeros(2 * N_SET, dtype=torch.float64, device="cuda")
    for i in tr:
        h = student_hidden(model, tok, rows[i])[S_LAYER + 1].double()
        y = (Y[i].cuda() / ys).double()
        A += h.T @ h; Bm += h.T @ y; sx += h.sum(0); sy += y.sum(0); n += h.shape[0]
    mu = sx / n
    A = A - n * torch.outer(mu, mu)
    Bm = Bm - n * torch.outer(mu, sy / n)
    W = torch.linalg.solve(A + 1e-2 * A.diagonal().mean() * torch.eye(2048, device="cuda", dtype=torch.float64), Bm)
    out = {"mu": mu.float().cpu()}
    for t, sl in (("rd", slice(0, N_SET)), ("rand", slice(N_SET, 2 * N_SET))):
        U = torch.linalg.svd(W[:, sl], full_matrices=False)[0]
        for k in KS_C:
            out[f"{t}|k{k}"] = U[:, :k].float().cpu()
    g = torch.Generator().manual_seed(0)
    for k in KS_C:
        out[f"randdir|k{k}"] = torch.linalg.qr(torch.randn(2048, k, generator=g))[0]
    we, _, _ = load_sae()
    sd = torch.load(str(ALIGNER_PT), map_location="cpu")
    M = we[T["rd"]] @ sd["proj.weight"].float() @ torch.diag(sd["norm.weight"].float())   # [196, 2048]
    V = torch.linalg.svd(M, full_matrices=False)[2].T
    for k in KS_C:
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
    a = text.find("<think>")
    b = text.find("</think>")
    b = len(text) if b < 0 else b
    inside = [a >= 0 and s >= a + 7 and e <= b for s, e in enc["offset_mapping"]]
    return enc["input_ids"], inside


@torch.no_grad()
def tf_phase(tok, rows, T):
    from mi_ablation import marker_ids
    mids = marker_ids(tok)
    log(f"tf: marker token ids {len(mids)}")
    mset = torch.zeros(len(tok), dtype=torch.bool)
    mset[mids] = True
    mid_t = torch.tensor(mids).cuda()
    te = [i for i in range(len(rows)) if i % 2 == 1]
    path = OUT / "tf.jsonl"
    done = {json.loads(l)["student"] for l in path.open()} if path.exists() else set()
    for name in STUDENTS:
        if name in done:
            continue
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[name], dtype=torch.bfloat16, device_map={"": 0},
                                                     cache_dir=str(LLAMA_DIR / "cache")).eval()
        sub = subspaces(name, model, tok, rows, T)
        mu = sub["mu"].cuda()
        layer = model.model.layers[S_LAYER]
        conds = ["clean"] + [f"{t}|k{k}" for t in ("rd", "rand", "randdir", "aligner") for k in KS_C]
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
                kl = (base_lp.exp() * (base_lp - lp)).sum(-1)
                r["cond"][c] = {"lp_mk": float(lpa[mk].mean()) if mk.any() else None,
                                "pm_mk": float(pm[mk].mean()) if mk.any() else None,
                                "pm_think": float(pm[th].mean()) if th.any() else None,
                                "nll": float(-lpa.mean()), "nll_think": float(-lpa[th].mean()) if th.any() else None,
                                "kl": float(kl.mean()), "size": float(np.mean(hk.size)) if hk.size else 0.0}
            recs.append(r)
        with path.open("a") as f:
            f.write(json.dumps({"student": name, "records": recs}) + "\n")
        log(f"  tf {name}: {len(recs)} rows {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()


def gen_phase(tok, rows, T, n_q=500, samples=2, cap=4096, slots=64):
    import eval_llama_passk as llama_passk
    from batched_evals.benchmarks import load_benchmark
    from batched_evals.generation import build_prompt, extract_answer
    bench = load_benchmark("gsm8k", cache_dir=str(ROOT / "cache"), split="test", max_samples=None)
    if any("embedded_fallback" in str(ex.metadata.get("source")) for ex in bench.examples):
        raise RuntimeError("gsm8k fell back to a placeholder")
    ex = list(bench.examples)[:n_q]
    path = OUT / "gen.jsonl"
    done = {json.loads(l)["cond"] for l in path.open()} if path.exists() else set()
    for name in ("lkd", "rd"):
        conds = [f"{name}|none"] + [f"{name}|{t}|k32" for t in ("rd", "rand", "randdir", "aligner")]
        if all(c in done for c in conds):
            continue
        stok = AutoTokenizer.from_pretrained(STUDENTS[name])
        if stok.pad_token is None:
            stok.pad_token = stok.eos_token
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[name], dtype=torch.bfloat16, device_map={"": 0}).eval()
        sub = subspaces(name, model, tok, rows, T)
        mu = sub["mu"].cuda()
        P = [stok(build_prompt(e.question, False, "native"))["input_ids"] for e in ex]
        for c in conds:
            if c in done:
                continue
            t0 = time.time()
            torch.manual_seed(0); random.seed(0); np.random.seed(0)
            key = c.split("|", 1)[1]
            hk = MeanAblate(model.model.layers[S_LAYER], None if key == "none" else sub[key].cuda(), mu)
            g = llama_passk.generate_pool(model, stok, P, samples, cap, 0.6, 0.95, slots)
            hk.remove()
            recs = []
            for q, e in enumerate(ex):
                for j in range(samples):
                    x = g[q * samples + j]
                    rr = {"id": e.id, "sample": j, "tokens": x["tokens"], "hit_limit": x["hit_limit"], "loop": x["loop"],
                          "state": "loop" if x["loop"] else llama_passk.end_state(x["text"], x["hit_limit"]), "text": x["text"]}
                    for mode in ("lenient", "strict"):
                        a = extract_answer(x["text"], mode)
                        rr[mode] = bool(a.committed and bench.scorer(a.answer, e.answer))
                    recs.append(rr)
            with path.open("a") as f:
                f.write(json.dumps({"cond": c, "records": recs}) + "\n")
            log(f"  gen {c:22s} strict {np.mean([r['strict'] for r in recs]):.3f} lenient "
                f"{np.mean([r['lenient'] for r in recs]):.3f} tokens {np.mean([r['tokens'] for r in recs]):.0f} "
                f"loop {np.mean([r['loop'] for r in recs]):.2f} cap {np.mean([r['hit_limit'] for r in recs]):.2f} "
                f"({(time.time() - t0) / 60:.0f} min)")
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["probe"])
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--students", nargs="+", default=list(STUDENTS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(STUDENTS["lkd"])
    rows = texts(a.n)
    log(f"texts: {len(rows)} held-out rows")
    if "probe" in a.phases:
        T = teacher_targets(tok, rows)
        for s in a.students:
            probe_student(s, tok, rows, T)
    if "tf" in a.phases or "gen" in a.phases:
        T = torch.load(OUT / "targets.pt")
        if "tf" in a.phases:
            tf_phase(tok, rows, T)
        if "gen" in a.phases:
            gen_phase(tok, rows, T)
    if "pipe" in a.phases:
        pipe_phase(tok, rows, torch.load(OUT / "targets.pt") if (OUT / "targets.pt").exists() else teacher_targets(tok, rows))


if __name__ == "__main__":
    main()
