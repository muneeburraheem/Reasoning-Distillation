"""Exp 3 · Feature function transfer: does a feature do the same job in the student as in the teacher?

For one feature f at a time, push the residual along f's write direction by a fixed fraction r of
its norm at every position, teacher-forced on held-out text, and record the mean change in next-token
log-probabilities over the whole vocabulary (the feature's "effect vector").
  teacher   L19 output (hidden_states[20]), direction = SAE decoder column W_dec[:, f]
  student   h[11] (output of block 10), direction = D_f, the student's own write direction for f:
            ridge regression of the student's h[11] on all 392 teacher feature activations
            (h - mu ~ Y D), fitted on the train texts, so D_f is the part of the student's state
            that moves with teacher feature f
Matched transfer = Pearson correlation over the vocabulary between the teacher's and the student's
effect vectors for the same f; the null is the mean correlation with the other features' teacher
effect vectors (mismatched). Specific transfer = matched - mismatched.
Features (chosen by density only, before any result): the 30 densest live RD features, and the 30
density-matched random features paired with them. 100 test texts, r = 0.15 (both models).
Compared: base / LKD / RD; RD - LKD paired over features (bootstrap); RD set vs random set.

Run: python scripts/mi_llama_exp3_feature_function.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mi_llama_students as M  # noqa: E402

OUT = M.OUT / "exp3"
N_FEAT, N_TEXT, R = 30, 100, 0.15


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0},
                                                cache_dir=str(M.LLAMA_DIR / "cache")).eval()


@torch.no_grad()
def write_dirs(name, model, tok, rows, T):
    """D [392, 2048]: ridge of centred student h[11] on teacher feature activations (train texts)."""
    p = M.OUT / f"write_dirs_{name}.pt"
    if p.exists():
        return torch.load(p)
    tr = [i for i in range(len(rows)) if i % 2 == 0]
    k = 2 * M.N_SET
    A = torch.zeros(k, k, dtype=torch.float64, device="cuda")
    Bm = torch.zeros(k, 2048, dtype=torch.float64, device="cuda")
    sy = torch.zeros(k, dtype=torch.float64, device="cuda")
    sh = torch.zeros(2048, dtype=torch.float64, device="cuda")
    n = 0
    for i in tr:
        h = M.student_hidden(model, tok, rows[i])[M.S_LAYER + 1].double()
        y = T["Y"][i].cuda().double()
        A += y.T @ y; Bm += y.T @ h; sy += y.sum(0); sh += h.sum(0); n += h.shape[0]
    A -= torch.outer(sy, sy) / n
    Bm -= torch.outer(sy, sh) / n
    D = torch.linalg.solve(A + 1e-3 * A.diagonal().mean() * torch.eye(k, device="cuda", dtype=torch.float64), Bm)
    out = {"D": D.float().cpu(), "mu": (sh / n).float().cpu()}
    torch.save(out, p)
    return out


class Push:
    def __init__(self, layer):
        self.u = None
        self.hk = layer.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        if self.u is None:
            return o
        hs = o[0] if isinstance(o, tuple) else o
        new = (hs.float() + R * hs.float().norm(dim=-1, keepdim=True) * self.u).to(hs.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new


@torch.no_grad()
def effects(model, layer, dirs, texts):
    """dirs [n, d] unit rows -> mean d log p over all positions and texts, [n, V] (float32, CPU)."""
    hk = Push(layer)
    E = None
    npos = 0
    for x in texts:
        x = x.cuda()
        hk.u = None
        clean = F.log_softmax(model(x).logits[0].float(), -1)
        if E is None:
            E = torch.zeros(len(dirs), clean.shape[-1], device="cuda")
        for j in range(len(dirs)):
            hk.u = dirs[j].cuda()
            E[j] += (F.log_softmax(model(x).logits[0].float(), -1) - clean).sum(0)
        npos += x.shape[1]
    hk.hk.remove()
    return (E / npos).cpu()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(M.STUDENTS["lkd"])
    rows = [json.loads(l) for l in (M.OUT / "texts.jsonl").open()]
    T = torch.load(M.OUT / "targets.pt")
    feats = T["rd"] + T["rand"]
    te = [i for i in range(len(rows)) if i % 2 == 1][:N_TEXT]
    texts = [torch.tensor([tok(rows[i]["text"], truncation=True, max_length=M.MAXLEN)["input_ids"]]) for i in te]
    # densest live RD features (density on the train texts), and their matched random partners
    dens = T["density"]
    order = sorted(range(M.N_SET), key=lambda k: -dens[T["rd"][k]])[:N_FEAT]
    cols = order + [M.N_SET + k for k in order]                  # column indices into feats / D
    chosen = [feats[c] for c in cols]
    log(f"features: rd {chosen[:N_FEAT]}\n          rand {chosen[N_FEAT:]}")

    p = OUT / "effects.pt"
    res = torch.load(p) if p.exists() else {"features": chosen, "cols": cols}
    t0 = time.time()
    if "teacher" not in res:
        model = load(M.TEACHER)
        wd = torch.load(str(M.SAE_PT), map_location="cpu")["decoder.weight"].float()
        dirs = torch.stack([wd[:, f] / wd[:, f].norm() for f in chosen])
        res["teacher"] = effects(model, model.model.layers[M.T_LAYER], dirs, texts)
        torch.save(res, p)
        log(f"teacher effects {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()
    for s in M.STUDENTS:
        if s in res:
            continue
        model = load(M.STUDENTS[s])
        W = write_dirs(s, model, tok, rows, T)
        D = W["D"][cols]
        dirs = D / D.norm(dim=-1, keepdim=True)
        res[s] = effects(model, model.model.layers[M.S_LAYER], dirs, texts)
        res[f"{s}_dnorm"] = W["D"][cols].norm(dim=-1)
        torch.save(res, p)
        log(f"{s} effects {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    # analysis
    def corr(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a @ b.T) / (a.norm(dim=-1)[:, None] * b.norm(dim=-1)[None, :])

    Et = res["teacher"]
    lines = ["# Exp 3 · Feature function transfer (teacher-forced, 100 held-out texts, push r = 0.15 of |h|)",
             "matched = corr(teacher effect f, student effect f); mismatched = mean corr with the other features "
             "of the same set; specific = matched - mismatched"]
    spec = {}
    rng = np.random.default_rng(0)
    for s in M.STUDENTS:
        C = corr(res[s], Et).numpy()                                # [student f, teacher g]
        for name, sl in (("rd", slice(0, N_FEAT)), ("rand", slice(N_FEAT, 2 * N_FEAT))):
            c = C[sl, sl]
            matched = np.diag(c)
            mism = (c.sum(1) - matched) / (c.shape[1] - 1)
            spec[(s, name)] = matched - mism
            lines.append(f"  {s:4s} {name:4s} matched {matched.mean():+.3f}  mismatched {mism.mean():+.3f}  "
                         f"specific {spec[(s, name)].mean():+.3f}  (features with specific > 0: {(spec[(s, name)] > 0).sum()}/{N_FEAT})")
    lines.append("\n## paired over features, bootstrap")
    for name in ("rd", "rand"):
        for a, b in (("rd", "lkd"), ("lkd", "base")):
            d = spec[(a, name)] - spec[(b, name)]
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 2000)
            lines.append(f"  {name:4s} specific {a}-{b:4s} {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={pv:.4f}")
    for s in M.STUDENTS:
        d = spec[(s, "rd")] - spec[(s, "rand")]
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lines.append(f"  {s:4s} specific rd-set - rand-set (paired by density partner) {d.mean():+.4f} "
                     f"[{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]")
    lines.append("\n## examples: top 8 promoted tokens, teacher vs LKD vs RD (5 densest RD features)")
    for j in range(5):
        row = [f"  f{chosen[j]}"]
        for s in ("teacher", "lkd", "rd"):
            top = res[s][j].topk(8).indices.tolist()
            row.append(f"{s}: " + " ".join(repr(tok.decode([t]).strip()) for t in top))
        lines.append("\n     ".join(row))
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
