"""Exp 5 · Layer similarity (CKA): is RD's student more teacher-like overall, or only on the RD features?

Linear CKA (centred, feature-space form ||X'Y||_F^2 / (||X'X||_F ||Y'Y||_F)) between every student
layer (hidden_states 1..16) and every teacher layer (hidden_states 1..32), for base / LKD / RD, on
the 400 held-out test texts of RUN_LOG 8aa (first 512 tokens, BOS dropped), 64 random positions per
text (25,600 tokens; the same positions for every model). Teacher bf16.
Bootstrap over texts (300 resamples) for RD - LKD at every student layer against teacher h[20] (the
SAE's layer) and against that student layer's best-matching teacher layer.

Run: python scripts/mi_llama_exp5_cka.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mi_llama_students as M  # noqa: E402

OUT = M.OUT / "exp5"
PER_TEXT = 64


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


@torch.no_grad()
def collect(path, tok, rows, te, pos, quant=False):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0},
                                                 cache_dir=str(M.LLAMA_DIR / "cache")).eval()
    H = None
    for n, i in enumerate(te):
        ids = tok(rows[i]["text"], truncation=True, max_length=M.MAXLEN)["input_ids"]
        hs = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        p = torch.tensor(pos[n]).cuda() + 1                       # +1: BOS dropped in positions
        x = torch.stack([h[0, p] for h in hs[1:]]).half().cpu()  # [layers, PER_TEXT, d]
        if H is None:
            H = torch.zeros(x.shape[0], len(te) * PER_TEXT, x.shape[2], dtype=torch.float16)
        H[:, n * PER_TEXT:(n + 1) * PER_TEXT] = x
    del model
    torch.cuda.empty_cache()
    return H


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M.STUDENTS["lkd"])
    rows = [json.loads(l) for l in (M.OUT / "texts.jsonl").open()]
    te = [i for i in range(len(rows)) if i % 2 == 1]
    rng = np.random.default_rng(0)
    pos = []
    for i in te:
        n = len(tok(rows[i]["text"], truncation=True, max_length=M.MAXLEN)["input_ids"]) - 1
        pos.append(sorted(rng.choice(n, PER_TEXT, replace=n < PER_TEXT).tolist()))
    t0 = time.time()
    Ht = collect(M.TEACHER, tok, rows, te, pos)
    log(f"teacher hidden states {tuple(Ht.shape)} {time.time() - t0:.0f}s")
    Hs = {s: collect(p, tok, rows, te, pos) for s, p in M.STUDENTS.items()}
    log(f"student hidden states {time.time() - t0:.0f}s")

    nt, ns = Ht.shape[0], 16
    C = {s: np.zeros((ns, nt)) for s in Hs}
    for lt in range(nt):
        Y = Ht[lt].cuda().float()
        for s in Hs:
            for ls in range(ns):
                C[s][ls, lt] = cka(Hs[s][ls].cuda().float(), Y)
    log(f"CKA matrices done {time.time() - t0:.0f}s")

    lines = ["# Exp 5 · Layer similarity (linear CKA), 400 held-out texts x 64 positions",
             "## CKA of each student layer with teacher h[20] (SAE layer), and its best teacher layer"]
    lines.append("  student  " + "  ".join(f"{s:>16s}" for s in Hs))
    for ls in range(ns):
        lines.append(f"  h[{ls + 1:2d}]{'*' if ls == 10 else ' '}  " + "  ".join(
            f"{C[s][ls, 19]:.3f} (best h[{int(C[s][ls].argmax()) + 1:2d}] {C[s][ls].max():.3f})" for s in Hs))
    lines.append("  * = guided layer; teacher index = hidden_states index")

    # bootstrap RD - LKD over texts
    lines.append("\n## RD - LKD (and LKD - base), bootstrap over texts: vs teacher h[20] | vs best teacher layer of LKD")
    B = 300
    idx = torch.arange(len(te) * PER_TEXT).view(len(te), PER_TEXT)
    boots = [idx[torch.from_numpy(rng.integers(0, len(te), len(te)))].flatten() for _ in range(B)]
    for ls in range(ns):
        best = int(C["lkd"][ls].argmax())
        parts = []
        for a, b in (("rd", "lkd"), ("lkd", "base")):
            for lt in (19, best):
                Y = Ht[lt].cuda().float()
                Xa, Xb = Hs[a][ls].cuda().float(), Hs[b][ls].cuda().float()
                d0 = C[a][ls, lt] - C[b][ls, lt]
                bs = np.array([cka(Xa[j], Y[j]) - cka(Xb[j], Y[j]) for j in (bb.cuda() for bb in boots)])
                p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
                parts.append(f"{a}-{b} {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={p:.3f}")
        lines.append(f"  h[{ls + 1:2d}] vs h[20]: {parts[0]} | vs h[{best + 1}]: {parts[1]} || lkd-base vs h[20]: {parts[2]}")
        log(lines[-1])
    torch.save({"cka": C, "positions": pos}, OUT / "cka.pt")
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
