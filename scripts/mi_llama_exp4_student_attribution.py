"""Exp 4 · Student attribution overlap: do the students route reasoning-word prediction through the same
teacher features as the teacher does?

Objective (both models, same held-out text): sum over reasoning-word targets inside <think> of
log P(reasoning-word class) (ReasonScore vocabulary, 35 token ids). One backward per text gives the
gradient g_t with respect to the residual at every position.
  teacher   at L19 output. Feature f's contribution at t is (|h_t|/64) a_f(t) W_dec[:, f] (the SAE's
            own decomposition, a_f from the bf16 teacher), so its attribution is
            sum_t (|h_t|/64) a_f(t) (g_t . W_dec[:, f])
  student   at h[11]. Feature f's contribution is y_f(t) D_f (y = teacher activation on the same
            token, D_f = the student's write direction from Exp 3), so its attribution is
            sum_t y_f(t) (g_t . D_f)
392 features: the 196 RD features and their 196 density-matched random partners. 400 test texts.
Reported: Spearman between teacher and student attribution over features (all / RD set / random set),
top-20 overlap, mean attribution RD set vs random set per model, and the share of the student's
total attribution sum_t g_t . (h_t - mu) carried by the 392 features. Bootstrap over texts;
RD - LKD paired.

Run: python scripts/mi_llama_exp4_student_attribution.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mi_llama_students as M  # noqa: E402
from mi_llama_exp3_feature_function import write_dirs  # noqa: E402

OUT = M.OUT / "exp4"


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0},
                                             cache_dir=str(M.LLAMA_DIR / "cache")).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Leaf:
    """Replace a layer's output with a leaf tensor so its gradient can be read."""

    def __init__(self, layer):
        self.x = None
        self.hk = layer.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        hs = o[0] if isinstance(o, tuple) else o
        self.x = hs.detach().float().requires_grad_(True)
        new = self.x.to(hs.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new


def objective_grad(model, layer, ids, mk, mid_t):
    hk = Leaf(layer)
    with torch.enable_grad():
        lp = F.log_softmax(model(torch.tensor([ids]).cuda()).logits[0, :-1].float(), -1)
        obj = torch.logsumexp(lp[mk][:, mid_t], -1).sum()
        obj.backward()
    hk.hk.remove()
    return hk.x.grad[0].detach(), hk.x[0].detach()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from mi_ablation import marker_ids
    tok = AutoTokenizer.from_pretrained(M.STUDENTS["lkd"])
    rows = [json.loads(l) for l in (M.OUT / "texts.jsonl").open()]
    T = torch.load(M.OUT / "targets.pt")
    feats = T["rd"] + T["rand"]
    te = [i for i in range(len(rows)) if i % 2 == 1]
    mids = marker_ids(tok)
    mset = set(mids)
    mid_t = torch.tensor(mids).cuda()
    spans = {}
    for i in te:
        ids, inside = M.think_span(tok, rows[i]["text"])
        mk = torch.tensor([inside[t + 1] and ids[t + 1] in mset for t in range(len(ids) - 1)])
        if mk.any():
            spans[i] = (ids, mk.cuda())
    use = sorted(spans)
    log(f"texts with reasoning-word targets: {len(use)}")
    p = OUT / "attribution.pt"
    res = torch.load(p) if p.exists() else {"rows": use}
    t0 = time.time()
    if "teacher" not in res:
        model = load(M.TEACHER)
        sd = torch.load(str(M.SAE_PT), map_location="cpu")
        we, be = sd["encoder.weight"][feats].float().cuda(), sd["encoder.bias"][feats].float().cuda()
        wd = sd["decoder.weight"][:, feats].float().cuda()                 # [4096, 392]
        A = torch.zeros(len(use), len(feats))
        tot = torch.zeros(len(use))
        for n, i in enumerate(use):
            ids, mk = spans[i]
            g, h = objective_grad(model, model.model.layers[M.T_LAYER], ids, mk, mid_t)
            g, h = g[1:], h[1:]                                              # BOS dropped
            hn = h.norm(dim=-1, keepdim=True)
            a = F.relu((h * (M.NORM / hn)) @ we.T + be)                      # [t, 392] raw latents
            A[n] = ((hn / M.NORM) * a * (g @ wd)).sum(0).cpu()
            tot[n] = float((g * (h - h.mean(0))).sum())
        res["teacher"], res["teacher_total"] = A, tot
        torch.save(res, p)
        log(f"teacher attribution {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()
    for s in M.STUDENTS:
        if s in res:
            continue
        model = load(M.STUDENTS[s])
        W = write_dirs(s, model, tok, rows, T)
        D, mu = W["D"].cuda(), W["mu"].cuda()                              # [392, 2048]
        A = torch.zeros(len(use), len(feats))
        tot = torch.zeros(len(use))
        for n, i in enumerate(use):
            ids, mk = spans[i]
            g, h = objective_grad(model, model.model.layers[M.S_LAYER], ids, mk, mid_t)
            g, h = g[1:], h[1:]
            y = T["Y"][i].cuda()
            A[n] = (y * (g @ D.T)).sum(0).cpu()
            tot[n] = float((g * (h - mu)).sum())
        res[s], res[f"{s}_total"] = A, tot
        torch.save(res, p)
        log(f"{s} attribution {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    # analysis
    rng = np.random.default_rng(0)
    n = len(use)
    sets = {"all": slice(0, 2 * M.N_SET), "rd": slice(0, M.N_SET), "rand": slice(M.N_SET, 2 * M.N_SET)}

    def rho(a, b, sl):
        return spearmanr(a[:, sl].mean(0), b[:, sl].mean(0)).statistic

    lines = ["# Exp 4 · Student attribution overlap: attribution of log P(reasoning word) to teacher features",
             f"{n} held-out texts with reasoning-word targets inside <think>; bootstrap over texts (500)"]
    At = res["teacher"].numpy()
    for s in M.STUDENTS:
        As = res[s].numpy()
        share = res[s].sum(1).sum() / res[f"{s}_total"].sum()
        parts = []
        for nm, sl in sets.items():
            r0 = rho(As, At, sl)
            bs = [rho(As[j], At[j], sl) for j in (rng.integers(0, n, n) for _ in range(500))]
            parts.append(f"{nm} {r0:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
        top_t = set(np.argsort(-At.mean(0))[:20])
        top_s = set(np.argsort(-As.mean(0))[:20])
        lines.append(f"  {s:4s} Spearman with teacher: " + " | ".join(parts) +
                     f" | top-20 overlap {len(top_t & top_s)} | features' share of student total {share:+.3f}")
    lines.append("\n## RD - LKD in Spearman with teacher (paired bootstrap over texts)")
    for nm, sl in sets.items():
        Ar, Al = res["rd"].numpy(), res["lkd"].numpy()
        d0 = rho(Ar, At, sl) - rho(Al, At, sl)
        bs = []
        for _ in range(500):
            j = rng.integers(0, n, n)
            bs.append(rho(Ar[j], At[j], sl) - rho(Al[j], At[j], sl))
        bs = np.array(bs)
        pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 500)
        lines.append(f"  {nm:4s} {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={pv:.3f}")
    lines.append("\n## mean attribution per feature, RD set vs random set (per text, then bootstrap)")
    for s in ["teacher"] + list(M.STUDENTS):
        A = res[s].numpy()
        d = A[:, :M.N_SET].mean(1) - A[:, M.N_SET:].mean(1)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        lines.append(f"  {s:7s} RD set {A[:, :M.N_SET].mean():+.3e}  random set {A[:, M.N_SET:].mean():+.3e}  "
                     f"difference {d.mean():+.3e} [{np.percentile(bs, 2.5):+.3e},{np.percentile(bs, 97.5):+.3e}]")
    lines.append("\n## teacher's top-10 RD features by attribution, and their rank in each student (of 392)")
    order = np.argsort(-At.mean(0))
    ranks = {s: {f: r for r, f in enumerate(np.argsort(-res[s].numpy().mean(0)))} for s in M.STUDENTS}
    for c in [c for c in order if c < M.N_SET][:10]:
        lines.append(f"  f{feats[c]:<6d} teacher {At.mean(0)[c]:+.3e}  " +
                     "  ".join(f"{s} #{ranks[s][c] + 1}" for s in M.STUDENTS))
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
