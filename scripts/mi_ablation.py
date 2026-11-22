"""SAE feature patching and suppression with matched controls (RUN_LOG 8w steps 3-4).

Shares subjects, tokenizer, SAE, calibration (fmax, density) and question splits with
scripts/mi_steering.py; reads its calib.json and the unsteered traces of its full phase.

Feature sets (first k ranked features ACTIVE in calibration, k = 15 / 50 / all 177):
  rd         the ReasonScore set RD aligned to
  rand s0-4  density-matched random SAE features (same rule as mi_steering.directions)
  attr       positive control: the k latents with the largest mean attribution to the
             reasoning-marker probability, estimated on split A of the traces only
Interventions use the SAE in norm-64 space and keep its reconstruction error:
  ablate     h -> h - (|h|/64) * sum_{f in S} a_f W_dec_f                    (a_f = f's activation)
  patch-in   h -> h + (|h|/64) * sum_{f in S} (a_f(src) - a_f) W_dec_f       (src = a marker position)
  randdir    h -> h + |delta_rd| * g/|g|, g Gaussian: same size as the rd intervention, random direction

Phases
  patch  teacher-forced on the 400 unsteered 4096-token traces (steering full phase, base).
         Marker positions: inside <think>, next token is a ReasonScore word (wait / hmm / maybe /
         perhaps / but / another / therefore / however; "let" excluded, too generic); control
         positions: inside <think>, next token not a marker. Up to 6 + 6 per trace. Each
         intervention acts on ONE position (layer `layer`, KV cache of the clean prefix), so the
         read-out is that position's next-token distribution:
           necessity   at marker positions, ablate S: change in log p(actual marker), in
                       P(marker class), and KL(clean || ablated)
           sufficiency at control positions, patch in S's activations from the nearest marker
                       position of the same trace: change in P(marker class), KL
         Attribution (grad x activation of log P(marker class) w.r.t. all latents) on split A
         marker positions selects the attr sets; all comparisons with attr are reported on split B.
         The ReasonScore set's attribution rank is reported against density-matched random sets.
  sgrid  suppression: ablate S at every position while generating (greedy, 1024 tokens, the
         100 grid prompts), sets rd / rand s0-1 / attr at k 15 / 50 / all
  sfull  suppression, 200 questions x 2, T 0.6, 4096 tokens: rd-all, rand-all s0, rd-50, rand-50
         s0, attr-50; base reused from mi_steering full.jsonl (identical settings and seed)

Run: python scripts/mi_ablation.py --subject llama_teacher --phases patch sgrid sfull
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mi_steering as ms  # noqa: E402

MARKERS = {"wait", "hmm", "maybe", "perhaps", "but", "another", "therefore", "however"}
KS = (15, 50, 0)                     # 0 = every active ranked feature
RAND_SEEDS = (0, 1, 2, 3, 4)
PER_TRACE = 6
CHUNK = 8                            # intervention rows per one-token forward


def log(out, msg):
    ms.log(out, msg)


class SAE:
    def __init__(self, S):
        we, be, wd = ms.load_sae(S)
        self.we, self.be, self.wd = we.cuda().bfloat16(), be.cuda().float(), wd.cuda().bfloat16()
        self.norm = S["sae_norm"]

    def encode(self, h, feats=None):                       # h [..., d] (raw residual) -> latents
        hs = (h.float() * (self.norm / h.float().norm(dim=-1, keepdim=True).clamp(min=1e-8))).bfloat16()
        if feats is None:
            return F.relu((hs @ self.we.T).float() + self.be)
        return F.relu((hs @ self.we[feats].T).float() + self.be[feats])

    def delta(self, h, coef, feats):                       # raw-space step for latent changes coef
        return (coef @ self.wd[:, feats].T.float()) * (h.float().norm(dim=-1, keepdim=True) / self.norm)


def marker_ids(tok):
    ids = set()
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if s in MARKERS:
            ids.add(i)
    return sorted(ids)


def feature_sets(S, out, attr=None):
    st = json.loads((out / "calib.json").read_text())
    fmax = {int(k): v for k, v in st["fmax"].items()}
    dens = st["density"]
    act_rank = [f for f in st["rank"] if fmax[f] > 0]
    act_pool = [f for f in st["pool"] if fmax[f] > 0]
    bins = lambda d: int(np.clip(np.floor(np.log10(max(d, 1e-7)) * 2), -14, 0))  # noqa: E731
    by_bin = {}
    for f in act_pool:
        by_bin.setdefault(bins(dens[f]), []).append(f)

    def matched(feats, seed):
        rng = random.Random(seed)
        rf = []
        for f in feats:
            b = bins(dens[f])
            cand = next(by_bin[x] for x in sorted(by_bin, key=lambda x: (abs(x - b), x)) if by_bin[x])
            rf.append(rng.choice([c for c in cand if c not in rf] or cand))
        return rf

    sets = {}
    for k in KS:
        rd = act_rank[:k] if k else act_rank
        sets[f"rd|k{k}"] = rd
        for s in RAND_SEEDS:
            sets[f"rand|k{k}|s{s}"] = matched(rd, 7000 + 100 * k + s)
        if attr is not None:
            sets[f"attr|k{k}"] = attr[:(k or len(act_rank))]
    return sets, act_rank, dens


def load_traces(S, tok, steer_out):
    rec = next(json.loads(l) for l in (steer_out / "full.jsonl").open() if json.loads(l)["cond"].startswith("base|"))
    _, full_rows, _ = ms.splits()
    q = {r["id"]: r["problem"] for r in full_rows}
    T = []
    for x in rec["records"]:
        p = ms.prompt_ids(S, tok, q[x["id"]])
        g = tok(x["text"], add_special_tokens=False)["input_ids"]
        T.append({"id": x["id"], "sample": x["sample"], "prompt": p, "gen": g})
    return T


def positions(T, tok, mids, seed=0):
    """Marker / control positions t (index into prompt+gen): token t+1 is / is not a marker,
    t inside <think> (before </think>)."""
    rng = random.Random(seed)
    end = tok.convert_tokens_to_ids("</think>")
    ms_ = set(mids)
    for tr in T:
        seq = tr["prompt"] + tr["gen"]
        n0 = len(tr["prompt"])
        stop = seq.index(end) if end in seq[n0:] else len(seq)
        mk = [t for t in range(n0 + 8, stop - 1) if seq[t + 1] in ms_]
        ct = [t for t in range(n0 + 8, stop - 1) if seq[t + 1] not in ms_]
        tr["mk"] = sorted(rng.sample(mk, min(PER_TRACE, len(mk))))
        tr["ct"] = sorted(rng.sample(ct, min(PER_TRACE, len(ct)))) if tr["mk"] else []
        tr["mk_all"] = mk


def sliced_cache(pkv, t, rows):
    data = [(l.keys[:, :, :t].expand(rows, -1, -1, -1), l.values[:, :, :t].expand(rows, -1, -1, -1)) for l in pkv.layers]
    return DynamicCache(ddp_cache_data=data)


class PosHook:
    """Replaces the layer output at the single (last) position with h + deltas[row]."""

    def __init__(self, layer):
        self.d = None
        self.h = layer.register_forward_hook(self.hook)

    def hook(self, _m, _i, out):
        if self.d is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[:, -1] = (hs[:, -1].float() + self.d).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.h.remove()


# ----------------------------------------------------------------------------- patch
def run_trace_clean(model, layer, seq):
    cap = {}
    hk = layer.register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].detach()))
    with torch.no_grad():
        o = model(torch.tensor([seq]).cuda(), use_cache=True)
    hk.remove()
    return o.past_key_values, cap["h"], o.logits[0]


def rep_cache(c, rows):
    """Independent copy of a (hybrid) cache with every state repeated `rows` times on dim 0."""
    c2 = copy.copy(c)
    c2.layers = []
    for l in c.layers:
        l2 = copy.copy(l)
        for a in ("keys", "values", "conv_states", "recurrent_states"):
            v = getattr(l, a, None)
            if isinstance(v, torch.Tensor):
                setattr(l2, a, v.repeat_interleave(rows, 0))
        if hasattr(l2, "max_batch_size"):
            l2.max_batch_size = rows
        c2.layers.append(l2)
    return c2


class Prefix:
    """Clean states of one trace and caches of its prefixes.
    slice:       one forward with KV cache, prefixes are slices (pure-attention models)
    incremental: a clean cache advanced position by position (linear-attention states cannot be
                 sliced); positions must be requested in increasing order."""

    def __init__(self, S, model, layer, seq):
        self.mode, self.model, self.seq = S.get("cache", "slice"), model, seq
        if self.mode == "slice":
            self.pkv, self.H, self.LG = run_trace_clean(model, layer, seq)
        else:
            cap = {}
            hk = layer.register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].detach()))
            with torch.no_grad():
                model.model(torch.tensor([seq]).cuda())          # hidden states only (vocab 248k: skip logits)
            hk.remove()
            self.H, self.c, self.cur = cap["h"], None, 0

    def cache(self, t, rows):
        if self.mode == "slice":
            return sliced_cache(self.pkv, t, rows)
        assert t >= self.cur
        if t > self.cur:
            with torch.no_grad():
                o = self.model(torch.tensor([self.seq[self.cur:t]]).cuda(), past_key_values=self.c, use_cache=True,
                               logits_to_keep=1)
            self.c, self.cur = o.past_key_values, t
        return rep_cache(self.c, rows)

    @torch.no_grad()
    def clean_logits(self, t):
        if self.mode == "slice":
            return self.LG[t].float()
        return self.model(torch.tensor([[self.seq[t]]]).cuda(), past_key_values=self.cache(t, 1)).logits[0, -1].float()


def attribution(S, tok, model, sae, T, mids, out):
    """Mean grad x activation of log P(marker class) over split-A marker positions, all latents."""
    layer = model.model.layers[S["layer"]]
    for p in model.parameters():
        p.requires_grad_(False)
    # fla's fused recurrent gated-delta kernel (one-token steps) has no backward: use the torch one here
    swapped = []
    for m in model.modules():
        if hasattr(m, "recurrent_gated_delta_rule"):
            import importlib
            ref = importlib.import_module(type(m).__module__).torch_recurrent_gated_delta_rule
            swapped.append((m, m.recurrent_gated_delta_rule))
            m.recurrent_gated_delta_rule = ref
    L = sae.we.shape[0]
    tot, n = torch.zeros(L, device="cuda"), 0
    mid = torch.tensor(mids).cuda()
    for i, tr in enumerate(T):
        if tr["id"] % 2 or not tr["mk"]:
            continue
        seq = tr["prompt"] + tr["gen"]
        pre = Prefix(S, model, layer, seq)
        H = pre.H
        for t in tr["mk"]:
            with torch.enable_grad():                       # patch() runs under no_grad
                h = H[t].float()
                a = sae.encode(h[None])[0].detach().requires_grad_(True)
                recon = (a @ sae.wd.T.float()) * (h.norm() / sae.norm)
                new = recon + (h - recon).detach()          # = h, but differentiable in a

                def hook(_m, _i, o, new=new):
                    hs = o[0] if isinstance(o, tuple) else o
                    hs = torch.cat([hs[:, :-1], new.to(hs.dtype)[None, None]], 1)
                    return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
                hk = layer.register_forward_hook(hook)
                lo = model(torch.tensor([[seq[t]]]).cuda(), past_key_values=pre.cache(t, 1)).logits[0, -1].float()
                torch.logsumexp(F.log_softmax(lo, -1)[mid], 0).backward()
                hk.remove()
                tot += (a.grad * a).detach()
                n += 1
        del pre, H
        if i % 50 == 0:
            log(out, f"  attribution {i}/{len(T)} traces, {n} positions")
    for m, f in swapped:
        m.recurrent_gated_delta_rule = f
    return (tot / n).cpu(), n


@torch.no_grad()
def patch(S, tok, model, out, steer_out):
    sae = SAE(S)
    mids = marker_ids(tok)
    log(out, f"marker token ids: {len(mids)} ({sorted({tok.decode([i]).strip().lower() for i in mids})})")
    T = load_traces(S, tok, steer_out)
    positions(T, tok, mids)
    log(out, f"traces {len(T)}; marker positions {sum(len(t['mk']) for t in T)}, control {sum(len(t['ct']) for t in T)}")
    ap = out / "attribution.pt"
    if ap.exists():
        att = torch.load(ap)
    else:
        a, n = attribution(S, tok, model, sae, T, mids, out)
        att = {"mean": a, "n": n}
        torch.save(att, ap)
    order = torch.argsort(att["mean"], descending=True).tolist()
    sets, act_rank, dens = feature_sets(S, out, attr=order)
    # where does the ReasonScore set sit in the attribution ranking?
    rk = {f: i for i, f in enumerate(order)}
    rep = {"attr_n": att["n"]}
    for k in KS:
        rd = sets[f"rd|k{k}"]
        rep[f"rd|k{k}"] = {"mean_attr": float(att["mean"][rd].mean()), "median_rank": float(np.median([rk[f] for f in rd])),
                           "in_top200": sum(rk[f] < 200 for f in rd),
                           "rand_mean_attr": [float(att["mean"][sets[f"rand|k{k}|s{s}"]].mean()) for s in RAND_SEEDS],
                           "rand_in_top200": [sum(rk[f] < 200 for f in sets[f"rand|k{k}|s{s}"]) for s in RAND_SEEDS]}
    rep["attr_top50"] = order[:50]
    rep["attr_top50_rd_overlap"] = len(set(order[:50]) & set(act_rank))
    (out / "attribution_summary.json").write_text(json.dumps(rep, indent=1))
    log(out, "attribution: " + json.dumps({k: v for k, v in rep.items() if k != "attr_top50"}))

    names = list(sets)
    mid = torch.tensor(mids).cuda()
    layer = model.model.layers[S["layer"]]
    hook = PosHook(layer)
    path = out / "patch.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["sample"]) for l in path.open()} if path.exists() else set()
    t0 = time.time()
    for i, tr in enumerate(T):
        if not tr["mk"] or (tr["id"], tr["sample"]) in done:
            continue
        seq = tr["prompt"] + tr["gen"]
        pre = Prefix(S, model, layer, seq)
        H = pre.H
        U = sorted({f for v in sets.values() for f in v})
        col = {f: j for j, f in enumerate(U)}
        A = sae.encode(H.float(), torch.tensor(U).cuda())      # [n, |U|] latents of every set feature
        recs = []
        for kind, t in sorted([("mk", t) for t in tr["mk"]] + [("ct", t) for t in tr["ct"]], key=lambda x: x[1]):
            if True:
                h = H[t].float()
                src = min(tr["mk_all"], key=lambda m: abs(m - t)) if kind == "ct" else None
                deltas, labels = [torch.zeros_like(h)], ["clean"]   # clean via cache: KL must be ~0
                for nm in names:
                    f = sets[nm]
                    c_ = [col[x] for x in f]
                    coef = -A[t, c_] if kind == "mk" else (A[src, c_] - A[t, c_])
                    deltas.append(sae.delta(h[None], coef[None], f)[0])
                    labels.append(nm)
                    if nm.startswith("rd|"):                  # same-size random direction
                        g = torch.randn(h.shape[0], generator=torch.Generator().manual_seed(t)).cuda()
                        deltas.append(g / g.norm() * deltas[-1].norm())
                        labels.append("randdir" + nm[2:])
                D = torch.stack(deltas)
                clean = F.log_softmax(pre.clean_logits(t), -1)
                res = {"kind": kind, "t": t, "next": seq[t + 1], "id": tr["id"], "sample": tr["sample"],
                       "clean_lp": float(clean[seq[t + 1]]), "clean_pm": float(clean[mid].exp().sum()),
                       "hnorm": float(h.norm()), "cond": {}}
                for c in range(0, len(D), CHUNK):
                    d = D[c:c + CHUNK]
                    hook.d = d
                    lo = model(torch.tensor([[seq[t]]] * len(d)).cuda(), past_key_values=pre.cache(t, len(d))).logits[:, -1].float()
                    hook.d = None
                    lp = F.log_softmax(lo, -1)
                    kl = (clean.exp()[None] * (clean[None] - lp)).sum(-1)
                    for j in range(len(d)):
                        res["cond"][labels[c + j]] = {"lp": float(lp[j, seq[t + 1]]), "pm": float(lp[j, mid].exp().sum()),
                                                      "kl": float(kl[j]), "dnorm": float(d[j].norm())}
                recs.append(res)
        with path.open("a") as f:
            f.write(json.dumps({"id": tr["id"], "sample": tr["sample"], "split": "A" if tr["id"] % 2 == 0 else "B",
                                "positions": recs}) + "\n")
        del pre, H, A
        if i % 20 == 0:
            log(out, f"  patch {i}/{len(T)} traces ({time.time() - t0:.0f}s)")
    hook.remove()
    (out / "sets.json").write_text(json.dumps(sets))


# ----------------------------------------------------------------------------- suppression
class Ablate:
    """Removes S's SAE contribution at every position of the layer output."""

    def __init__(self, layer, sae, feats):
        self.sae, self.f = sae, torch.tensor(feats).cuda()
        self.h = layer.register_forward_hook(self.hook)

    def hook(self, _m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        a = self.sae.encode(hs, self.f)
        hs = (hs.float() + self.sae.delta(hs, -a, self.f)).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.h.remove()


@torch.no_grad()
def sgrid(S, tok, model, out, rows, bs=20, gen=1024):
    bs = S.get("grid_bs", bs)
    sae = SAE(S)
    sets = json.loads((out / "sets.json").read_text())
    conds = [f"{w}|k{k}" + ("|s%d" % s if w == "rand" else "") for k in KS for w, s in
             (("rd", 0), ("rand", 0), ("rand", 1), ("attr", 0))]
    path = out / "sgrid.jsonl"
    done = {json.loads(l)["cond"] for l in path.open()} if path.exists() else set()
    P = [ms.prompt_ids(S, tok, r["problem"]) for r in rows]
    layer = model.model.layers[S["layer"]]
    for name in conds:
        if name in done:
            continue
        t0 = time.time()
        res = []
        for b in range(0, len(P), bs):
            ids, mask = ms.left_pad(P[b:b + bs], tok.pad_token_id)
            ab = Ablate(layer, sae, sets[name])
            o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=gen, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            ab.remove()
            for row in o[:, ids.shape[1]:].tolist():
                n = next((t for t, x in enumerate(row) if x == tok.eos_token_id), len(row))
                res.append(row[:n] + ([tok.eos_token_id] if n < len(row) else []))
        recs = [{"id": r["id"], "tokens": len(g), "eos": bool(g) and g[-1] == tok.eos_token_id,
                 "closed_think": "</think>" in tok.decode(g), "rep": ms.repetition(g),
                 "nll": ms.nll(model, p + g, len(p)) if g else None, "text": tok.decode(g, skip_special_tokens=True)}
                for r, p, g in zip(rows, P, res)]
        with path.open("a") as f:
            f.write(json.dumps({"cond": name, "kind": name.split("|")[0], "k": int(name.split("|")[1][1:]),
                                "seed": int(name.split("|s")[1]) if "|s" in name else 0, "r": 0.0,
                                "features": sets[name], "records": recs}) + "\n")
        log(out, f"  sgrid {name:16s} tokens {np.mean([x['tokens'] for x in recs]):6.0f} closed "
                 f"{np.mean([x['closed_think'] for x in recs]):.2f} rep>=.85 {np.mean([x['rep'] >= .85 for x in recs]):.2f} "
                 f"nll {np.mean([x['nll'] for x in recs if x['nll'] is not None]):.3f} ({time.time() - t0:.0f}s)")


@torch.no_grad()
def sfull(S, tok, model, out, rows, steer_out, samples=2, gen=4096, slots=10):
    slots = S.get("slots", slots)
    import eval_llama_passk as llama_passk
    llama_passk.PREFILL_TOKENS = 8_000
    sae = SAE(S)
    sets = json.loads((out / "sets.json").read_text())
    path = out / "sfull.jsonl"
    if not path.exists():                                 # base: identical run from the steering phase
        base = next(l for l in (steer_out / "full.jsonl").open() if json.loads(l)["cond"].startswith("base|"))
        path.write_text(base)
    done = {json.loads(l)["cond"] for l in path.open()}
    P = [ms.prompt_ids(S, tok, r["problem"]) for r in rows]
    layer = model.model.layers[S["layer"]]
    for name in ("rd|k0", "rand|k0|s0", "rd|k50", "rand|k50|s0", "attr|k50"):
        if name in done:
            continue
        t0 = time.time()
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        ab = Ablate(layer, sae, sets[name])
        gens = llama_passk.generate_pool(model, tok, P, samples, gen, 0.6, 0.95, slots)
        ab.remove()
        recs = []
        for q, r in enumerate(rows):
            for j in range(samples):
                x = gens[q * samples + j]
                s_, l_ = ms.score(x["text"], r["answer"])
                recs.append({"id": r["id"], "level": r["level"], "sample": j, "tokens": x["tokens"],
                             "hit_limit": x["hit_limit"], "loop": x["loop"], "closed_think": "</think>" in x["text"],
                             "strict": s_, "lenient": l_, "text": x["text"]})
        with path.open("a") as f:
            f.write(json.dumps({"cond": name, "kind": name.split("|")[0], "records": recs}) + "\n")
        log(out, f"  sfull {name:16s} strict {np.mean([x['strict'] for x in recs]):.3f} lenient "
                 f"{np.mean([x['lenient'] for x in recs]):.3f} tokens {np.mean([x['tokens'] for x in recs]):.0f} "
                 f"loop {np.mean([x['loop'] for x in recs]):.2f} cap {np.mean([x['hit_limit'] for x in recs]):.2f} "
                 f"({(time.time() - t0) / 60:.0f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="llama_teacher", choices=list(ms.SUBJECTS))
    ap.add_argument("--phases", nargs="+", default=["patch", "sgrid", "sfull"])
    ap.add_argument("--max_traces", type=int, default=None, help="smoke test")
    ap.add_argument("--n_grid", type=int, default=100)
    ap.add_argument("--n_full", type=int, default=200)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--steer_dir", default=None, help="mi_steering output dir (default eval_outputs/mi_steering/<subject>)")
    args = ap.parse_args()
    steer_out = Path(args.steer_dir) if args.steer_dir else ROOT / "eval_outputs/mi_steering" / args.subject
    out = Path(args.out_dir or ROOT / "eval_outputs/mi_ablation" / args.subject)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "calib.json").exists():
        (out / "calib.json").write_text((steer_out / "calib.json").read_text())
    S, tok, model = ms.load(args.subject)
    grid_rows, full_rows, _ = ms.splits()
    log(out, f"{args.subject}: phases {args.phases}")
    if "patch" in args.phases:
        if args.max_traces:
            global load_traces
            _lt = load_traces
            load_traces = lambda *a: _lt(*a)[:args.max_traces]  # noqa: E731
        patch(S, tok, model, out, steer_out)
    torch.cuda.empty_cache()
    if "sgrid" in args.phases:
        sgrid(S, tok, model, out, grid_rows[:args.n_grid], gen=128 if args.max_traces else 1024)
    torch.cuda.empty_cache()
    if "sfull" in args.phases:
        sfull(S, tok, model, out, full_rows[:args.n_full], steer_out, gen=256 if args.max_traces else 4096)


if __name__ == "__main__":
    main()
