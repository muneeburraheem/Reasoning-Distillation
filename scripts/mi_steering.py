"""SAE feature steering, redone with the defects of unified_sweep fixed (RUN_LOG 8w).

Subjects are entries in SUBJECTS; the Llama teacher is the first. Qwen (after an SAE retrain) and
the students (directions from fitted probes) are added as further entries, same phases.

Fixes relative to KnowledgeDistillation/base/reasoning_neurons_validation/unified_sweep:
  - tokenizer built from tokenizer.json (the hub AutoTokenizer under transformers 5.8 drops spaces
    and BOS), real chat template, prompt ends "<think>\\n"
  - fmax = 0.98 quantile of a feature's NONZERO activations (the old all-token quantile was 0 for
    any feature firing on <2% of tokens, so "top-1" steered nothing); features with <20 nonzero
    calibration activations are reported and skipped, top-k = the first k ACTIVE features by rank
  - the SAE reads h * 64/|h|, so a step v in SAE space is h + (|h|/64) * v in the residual
    (the old script added v unscaled: ~3.5x too large at L19, median |h| 18)
  - strength is set as r = |delta|/|h| along u = v/|v|, v = sum_k fmax_k W_dec[:,k]. The paper's
    gamma * sum_k fmax_k W_dec_k grows with k (one feature at gamma 1 is 0.02-0.32 |h|, fifteen
    summed are 1.1 |h|: the smoke run at gamma 2 was gibberish), so gamma is not comparable
    across k; r is, and the paper-equivalent gamma = r * 64 / |v| is stored
  - controls of the same r: k density-matched random SAE features, and a Gaussian direction
  - positive and negative r

Phases
  calib  greedy 512 tokens on 100 MATH-500 prompts (disjoint from the eval sets): per-feature
         density for all latents, nonzero values for the ranked set and a random pool -> fmax
  grid   100 MATH-500 prompts, greedy, 1,024 new tokens, every (k, r) for the ranked
         features + 2 random-feature seeds + 1 random direction; text, length, </think>, NLL of
         the generated tokens under the unsteered model
  full   conditions chosen from grid by a fixed rule (see choose_full), 200 MATH-500 questions x
         2 samples, T 0.6, top_p 0.95, 4,096 tokens, scored strict / lenient

Run: python scripts/mi_steering.py --subject llama_teacher --phases calib grid full
"""
import argparse
import json
import math
import random
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from custom_eval.math_scoring import extract_boxed_answer, math_equivalent  # noqa: E402

# Hugging Face hub cache (HF_HUB_CACHE / HF_HOME aware; default ~/.cache/huggingface/hub)
from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402
# Llama teacher SAE + ReasonScore (KnowledgeDistillation project); set KD_BASE_DIR on another machine
KD_BASE = Path(os.environ.get("KD_BASE_DIR", "/home/cs/Desktop/KD/KnowledgeDistillation/base"))
QWEN_SAE = Path(HF_HUB_CACHE, "models--Khaquan--qwen-khaquanS-distillations/snapshots/"
                "1a0cc88b882e2001410b28cafefd54816e6ed66e/qwen-3.5-4B-L16_16x-SAE")
SUBJECTS = {
    "llama_teacher": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "layer": 19,                     # output of model.model.layers[19] (= hidden_states[20])
        "sae": KD_BASE / "checkpoints/l19_sae_l1_0.0009.pt",   # md5 0bc126ce..., used for RD
        "ranking": KD_BASE / "reasonscores_base_sae.pt",       # md5 b324df16..., used for RD
        "sae_norm": 64.0,
        "bos": "<｜begin▁of▁sentence｜>", "user": "<｜User｜>", "assistant": "<｜Assistant｜>",
        "eos": "<｜end▁of▁sentence｜>",
    },
    # The SAE/feature set RD-Qwen aligned to, read exactly as the RD trainer read it:
    # hidden_states[16] = output of layers[15], rescaled to norm 64, top-512 of reasonscore.pt.
    # (RUN_LOG 8g/8j: this SAE has FVE < 0 and 31/512 of the set ever nonzero; run as specified.)
    "qwen_teacher": {
        "model": "Qwen/Qwen3.5-4B", "cache_dir": str(ROOT / "cache"), "chat": True,
        "layer": 15,
        "sae": QWEN_SAE / "sae.pt", "ranking": QWEN_SAE / "reasonscore.pt",
        "sae_norm": 64.0, "n_ranked": 512,
        "eos": "<|im_end|>", "pad": "<|endoftext|>",
        "cache": "incremental",          # hybrid linear attention: KV slicing does not apply
        "slots": 24, "grid_bs": 25,
    },
}
PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
KS = (1, 5, 15, 50)
RS = (-0.8, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.8)
SEEDS = (0, 1)
N_RANKED = 196                      # the set RD aligned to
POOL = 4096                         # random-feature control pool
SMOKE = False                       # --smoke: tiny run into eval_outputs/mi_steering/smoke


def log(out, msg):
    print(msg, flush=True)
    with (out / "progress.log").open("a") as f:
        f.write(msg + "\n")


def load(subject):
    S = SUBJECTS[subject]
    global N_RANKED
    N_RANKED = S.get("n_ranked", N_RANKED)
    if S.get("chat"):
        tok = AutoTokenizer.from_pretrained(S["model"], cache_dir=S["cache_dir"])
        tok.eos_token, tok.pad_token = S["eos"], S["pad"]
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(S["model"], "tokenizer.json"))
        tok.eos_token = tok.pad_token = S["eos"]
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(S["model"], dtype=torch.bfloat16, device_map={"": 0},
                                                 cache_dir=S.get("cache_dir")).eval()
    return S, tok, model


def prompt_ids(S, tok, q):
    if S.get("chat"):
        return tok(tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True),
                   add_special_tokens=False)["input_ids"]
    return tok(S["bos"] + S["user"] + PROMPT.format(q=q) + S["assistant"] + "<think>\n",
               add_special_tokens=False)["input_ids"]


def splits():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=str(ROOT / "cache"))
    perm = np.random.RandomState(0).permutation(len(ds)).tolist()
    rows = [{"id": i, "problem": ds[i]["problem"], "answer": ds[i]["answer"], "level": ds[i]["level"]}
            for i in range(len(ds))]
    return [rows[i] for i in perm[:100]], [rows[i] for i in perm[:200]], [rows[i] for i in perm[-100:]]


def load_sae(S):
    sd = torch.load(str(S["sae"]), map_location="cpu")
    return sd["encoder.weight"], sd["encoder.bias"], sd["decoder.weight"]   # [L,d], [L], [d,L]


def left_pad(seqs, pad):
    w = max(map(len, seqs))
    ids = torch.full((len(seqs), w), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, w - len(s):] = torch.tensor(s)
        mask[i, w - len(s):] = 1
    return ids.cuda(), mask.cuda()


class Steer:
    """h -> h + r * |h| * u at every position of one layer's output (u unit norm)."""

    def __init__(self, layer, u, r):
        self.v = None if u is None or r == 0 else (r * u).cuda()
        self.h = layer.register_forward_hook(self.hook) if self.v is not None else None

    def hook(self, _m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + hs.norm(dim=-1, keepdim=True) * self.v.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        if self.h:
            self.h.remove()


# ----------------------------------------------------------------------------- calib
@torch.no_grad()
def calib(S, tok, model, out, rows, bs=25, gen=512):
    enc_w, enc_b, _ = load_sae(S)
    enc_w, enc_b = enc_w.cuda(), enc_b.cuda()
    L = enc_w.shape[0]
    rank = torch.load(str(S["ranking"]), map_location="cpu")["sorted_indices"][:N_RANKED].tolist()
    rng = random.Random(0)
    pool = rng.sample(sorted(set(range(L)) - set(rank)), POOL)
    track = rank + pool
    nz = torch.zeros(L, dtype=torch.long)
    vals = {f: [] for f in track}
    n_tok = 0
    layer = model.model.layers[S["layer"]]
    for b in range(0, len(rows), bs):
        ids, mask = left_pad([prompt_ids(S, tok, r["problem"]) for r in rows[b:b + bs]], tok.pad_token_id)
        cache = []
        hk = layer.register_forward_hook(lambda m, i, o: cache.append((o[0] if isinstance(o, tuple) else o)[:, -1:].float().cpu()))
        o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=gen, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        hk.remove()
        h = torch.cat(cache[1:], dim=1)                       # decode steps only: [B, T, d]
        live = (o[:, ids.shape[1]:-1] != tok.pad_token_id).cpu()[:, :h.shape[1]]
        h = h[:, :live.shape[1]][live]                        # generated, pre-EOS tokens
        for c in h.split(2048):
            c = c.cuda()
            f = F.relu((c * (S["sae_norm"] / c.norm(dim=-1, keepdim=True).clamp(min=1e-8))) @ enc_w.T + enc_b)
            nz += (f > 0).sum(0).cpu()
            ft = f[:, track].cpu()
            for j, fid in enumerate(track):
                v = ft[:, j]
                vals[fid].extend(v[v > 0].tolist())
        n_tok += h.shape[0]
        log(out, f"  calib {b + bs}/{len(rows)} prompts, {n_tok} tokens")
    fmax = {f: (float(np.quantile(v, 0.98)) if len(v) >= 20 else 0.0) for f, v in vals.items()}
    stats = {"n_tokens": n_tok, "rank": rank, "pool": pool, "density": (nz.float() / n_tok).tolist(),
             "fmax": fmax, "n_nonzero": {f: len(v) for f, v in vals.items()}}
    (out / "calib.json").write_text(json.dumps(stats))
    act = [f for f in rank if fmax[f] > 0]
    log(out, f"calib done: {n_tok} tokens; ranked features active (>=20 nonzero) {len(act)}/{N_RANKED}; "
             f"first 50 ranked active {sum(fmax[f] > 0 for f in rank[:50])}/50")
    del enc_w, enc_b
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- directions
def directions(S, out):
    st = json.loads((out / "calib.json").read_text())
    fmax = {int(k): v for k, v in st["fmax"].items()}
    dens = st["density"]
    _, _, W = load_sae(S)
    act_rank = [f for f in st["rank"] if fmax[f] > 0]
    act_pool = [f for f in st["pool"] if fmax[f] > 0]
    bins = lambda d: int(np.clip(np.floor(np.log10(max(d, 1e-7)) * 2), -14, 0))   # half-decade bins
    by_bin = {}
    for f in act_pool:
        by_bin.setdefault(bins(dens[f]), []).append(f)
    D = {}
    for k in KS:
        feats = act_rank[:k]
        v = sum(fmax[f] * W[:, f].float() for f in feats)
        D[("feat", k, 0)] = (v / v.norm(), feats, 64.0 / float(v.norm()))
        for s in SEEDS:
            rng = random.Random(1000 * k + s)
            rf = []
            for f in feats:                                    # density-matched stand-ins
                b = bins(dens[f])
                cand = next(by_bin[x] for x in sorted(by_bin, key=lambda x: (abs(x - b), x)) if by_bin[x])
                rf.append(rng.choice([c for c in cand if c not in rf] or cand))
            r = sum(fmax[f] * W[:, f].float() for f in rf)
            D[("randfeat", k, s)] = (r / r.norm(), rf, 64.0 / float(r.norm()))
            g = torch.Generator().manual_seed(2000 * k + s)
            r = torch.randn(W.shape[0], generator=g)
            D[("randdir", k, s)] = (r / r.norm(), [], None)
    return D


# ----------------------------------------------------------------------------- grid
@torch.no_grad()
def nll(model, seq, n_prompt):
    ids = torch.tensor([seq]).cuda()
    lp = torch.log_softmax(model(ids).logits[0, n_prompt - 1:-1].float(), -1)
    return float(-lp.gather(1, ids[0, n_prompt:, None]).mean())


def repetition(ids, window=400, n=8):
    t = ids[-window:]
    if len(t) < window:
        return 0.0
    g = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    return sum(v for v in g.values() if v > 1) / (len(t) - n + 1)


@torch.no_grad()
def grid(S, tok, model, out, rows, bs=20, gen=1024):
    gen = 128 if SMOKE else gen
    bs = S.get("grid_bs", bs)
    D = directions(S, out)
    path = out / "grid.jsonl"
    done = {json.loads(l)["cond"] for l in path.open()} if path.exists() else set()
    conds = [("base", 0, 0, 0.0)] + [(kind, k, s, g) for k in KS for g in RS
                                     for kind, s in [("feat", 0), ("randdir", 0)] + [("randfeat", s) for s in SEEDS]]
    if SMOKE:
        conds = [c for c in conds if c[0] == "base" or (c[1] == 15 and (c[0] == "feat" or c[3] == 0.2))]
    P = [prompt_ids(S, tok, r["problem"]) for r in rows]
    layer = model.model.layers[S["layer"]]
    for kind, k, s, g in conds:
        name = f"{kind}|k{k}|s{s}|r{g:+.2f}"
        if name in done:
            continue
        t0 = time.time()
        v, feats, per_gamma = (None, [], None) if kind == "base" else D[(kind, k, s)]
        res = []
        for b in range(0, len(P), bs):
            ids, mask = left_pad(P[b:b + bs], tok.pad_token_id)
            st = Steer(layer, v, g)
            o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=gen, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            st.remove()
            for i, row in enumerate(o[:, ids.shape[1]:].tolist()):
                n = next((t for t, x in enumerate(row) if x == tok.eos_token_id), len(row))
                res.append(row[:n] + ([tok.eos_token_id] if n < len(row) else []))
        recs = []
        for r, p, gids in zip(rows, P, res):
            recs.append({"id": r["id"], "tokens": len(gids), "eos": bool(gids) and gids[-1] == tok.eos_token_id,
                         "closed_think": "</think>" in tok.decode(gids), "rep": repetition(gids),
                         "nll": nll(model, p + gids, len(p)) if gids else None,
                         "text": tok.decode(gids, skip_special_tokens=True)})
        with path.open("a") as f:
            f.write(json.dumps({"cond": name, "kind": kind, "k": k, "seed": s, "r": g,
                                "gamma_equiv": None if per_gamma is None else g * per_gamma,
                                "features": feats, "records": recs}) + "\n")
        m = lambda key: np.mean([x[key] for x in recs if x[key] is not None])  # noqa: E731
        log(out, f"  {name:28s} tokens {m('tokens'):6.0f} closed {m('closed_think'):.2f} "
                 f"rep>=.85 {np.mean([x['rep'] >= .85 for x in recs]):.2f} nll {m('nll'):.3f} ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- full
def choose_full(out):
    """Fixed rule, written before the grid ran: k=15 (the old headline arm); for each sign the
    largest |r| in {0.05,0.1,0.2,0.4,0.8} whose looping rate (8-gram repetition >= .85 over the last 400
    tokens) is at most the baseline's + 10 pp AND whose mean NLL of the generated text under the
    unsteered model is at most the baseline's + 1.0 nat/token (coherence), else 1. Each with
    randfeat s0 and randdir s0 at the same r."""
    G, N = {}, {}
    for l in (out / "grid.jsonl").open():
        r = json.loads(l)
        G[r["cond"]] = np.mean([x["rep"] >= .85 for x in r["records"]])
        N[r["cond"]] = np.mean([x["nll"] for x in r["records"] if x["nll"] is not None])
    base, bn = G["base|k0|s0|r+0.00"], N["base|k0|s0|r+0.00"]
    conds = [("base", 0, 0, 0.0)]
    for sign in (1, -1):
        ok = [a for a in (0.05, 0.1, 0.2, 0.4, 0.8) if G[f"feat|k15|s0|r{sign * a:+.2f}"] <= base + 0.10
              and N[f"feat|k15|s0|r{sign * a:+.2f}"] <= bn + 1.0]
        g = sign * (max(ok) if ok else 0.05)
        conds += [("feat", 15, 0, g), ("randfeat", 15, 0, g), ("randdir", 15, 0, g)]
    return conds


def score(text, gold):
    tail = text.split("</think>", 1)[1] if "</think>" in text else ""
    strict = extract_boxed_answer(tail) if tail else None
    lenient = extract_boxed_answer(text)
    return (bool(strict) and math_equivalent(strict, gold)), (bool(lenient) and math_equivalent(lenient, gold))


@torch.no_grad()
def full(S, tok, model, out, rows, samples=2, gen=4096, slots=10):
    gen = 256 if SMOKE else gen
    slots = S.get("slots", slots)
    import eval_llama_passk as llama_passk
    llama_passk.PREFILL_TOKENS = 8_000                     # 8B: 40k-token prefill OOMed in the MLP (2026-10-29)
    D = directions(S, out)
    conds = choose_full(out)
    log(out, "full conditions: " + ", ".join(f"{c[0]} k{c[1]} r{c[3]:+.2f}" for c in conds))
    path = out / "full.jsonl"
    done = {json.loads(l)["cond"] for l in path.open()} if path.exists() else set()
    P = [prompt_ids(S, tok, r["problem"]) for r in rows]
    layer = model.model.layers[S["layer"]]
    for kind, k, s, g in conds:
        name = f"{kind}|k{k}|s{s}|r{g:+.2f}"
        if name in done:
            continue
        t0 = time.time()
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        v = None if kind == "base" else D[(kind, k, s)][0]
        st = Steer(layer, v, g)
        gens = llama_passk.generate_pool(model, tok, P, samples, gen, 0.6, 0.95, slots)
        st.remove()
        recs = []
        for q, r in enumerate(rows):
            for j in range(samples):
                x = gens[q * samples + j]
                st_, le = score(x["text"], r["answer"])
                recs.append({"id": r["id"], "level": r["level"], "sample": j, "tokens": x["tokens"],
                             "hit_limit": x["hit_limit"], "loop": x["loop"], "closed_think": "</think>" in x["text"],
                             "strict": st_, "lenient": le, "text": x["text"]})
        with path.open("a") as f:
            f.write(json.dumps({"cond": name, "kind": kind, "k": k, "seed": s, "r": g, "records": recs}) + "\n")
        log(out, f"  {name:28s} strict {np.mean([x['strict'] for x in recs]):.3f} lenient "
                 f"{np.mean([x['lenient'] for x in recs]):.3f} tokens {np.mean([x['tokens'] for x in recs]):.0f} "
                 f"loop {np.mean([x['loop'] for x in recs]):.2f} cap {np.mean([x['hit_limit'] for x in recs]):.2f} "
                 f"({(time.time() - t0) / 60:.0f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="llama_teacher", choices=list(SUBJECTS))
    ap.add_argument("--phases", nargs="+", default=["calib", "grid", "full"])
    ap.add_argument("--n_grid", type=int, default=100)
    ap.add_argument("--n_full", type=int, default=200)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global SMOKE
    SMOKE = args.smoke
    if SMOKE:
        args.n_grid, args.n_full, args.out_dir = 4, 3, args.out_dir or str(ROOT / "eval_outputs/mi_steering/smoke")
    out = Path(args.out_dir or ROOT / "eval_outputs/mi_steering" / args.subject)
    out.mkdir(parents=True, exist_ok=True)
    S, tok, model = load(args.subject)
    grid_rows, full_rows, calib_rows = splits()
    log(out, f"{args.subject}: phases {args.phases}; grid {args.n_grid}, full {args.n_full}, calib {len(calib_rows)}")
    if "calib" in args.phases:
        calib(S, tok, model, out, calib_rows[:10] if SMOKE else calib_rows)
    if "grid" in args.phases:
        grid(S, tok, model, out, grid_rows[:args.n_grid])
    if "full" in args.phases:
        full(S, tok, model, out, full_rows[:args.n_full])


if __name__ == "__main__":
    main()
