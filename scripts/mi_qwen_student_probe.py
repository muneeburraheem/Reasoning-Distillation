"""Check 1 · Headroom: can the existing Qwen students already read the new SAE's RD features?

If LKD's student already carries the teacher's top-196 ReasonScore features about as well as a linear
read-out can get, RD (which trains the student to carry them) has little to add.

Text    400 held-out AM-DeepSeek math rows (row 531,000+, past RD/LKD training rows 0-52,031),
        "user\\nassistant" as in training, first 2048 tokens; even rows = train (80% fit / 20% lambda),
        odd rows = test
Targets teacher Qwen3.5-4B 8-bit (as RD training), hidden_states[17], RD trainer's encode with the
        new SAE (h * 64/|h|, ReLU encode, x decoder-column norm). rd = top-196 ReasonScore,
        rand = 196 density-matched random features (half-decade bins, as mi_students_llama)
Probes  ridge, LayerNorm (no affine) + bias, closed form, lambda on the 20% split, per student layer
        hidden_states[1..24]; students base Qwen3.5-2B / LKD / October RD (aligned to the dead SAE)
Ceiling the same ridge from the teacher's own hidden_states[17] (what a linear read-out of the
        teacher's state reaches; features are ReLU of it, so < 1)
Stats   per test sequence sums so R^2 can be bootstrapped over sequences.

Run: python scripts/mi_qwen_student_probe.py   (then --analyze)
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = ROOT / "eval_outputs/rd_newsae/student_probe"          # texts, progress log, analysis
TENS = ROOT / "checkpoints/rd_newsae_probe"                    # targets.pt, probe_*.pt (large)
SAE_PT = ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/final/sae.pt"
RANK_PT = ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/reasonscore/reasonscore.pt"
SNAP = ROOT / "cache/models--Khaquan--qwen-khaquanS-distillations/snapshots/e3a06f557af2163c6af19a730242bfcdc8b98783"
STUDENTS = {"base": "Qwen/Qwen3.5-2B",
            "lkd": str(SNAP / "qwen-logitKD-final/qwen_logit_kd_continue/epoch_1"),
            "rd_oct": str(SNAP / "qwen-reasondistill-october/qwen_reasondistill_final_continue/epoch_1")}
# new-SAE RD warm-start runs (Vast, 2026-11-13); compared against "lkd" (= LKD-continue, same config but alpha_align 0)
NEW = {f"rd_{lvl}{tag}": str(ROOT / "checkpoints" / f"rd_newsae_{lvl}" / d)
       for lvl in ("low", "moderate", "high") for tag, d in (("", "final_step_94"), ("_h", "step_47_halfway"))}
STUDENTS.update({k: v for k, v in NEW.items() if (Path(v) / "model.safetensors").exists()})
MATH_SOURCES = {"NuminaMath_1.5", "MATH_metamathQA", "MATH_numina", "openR1Math_default", "MATH-lighteval", "Omni-MATH"}
N_TEXT, N_SET, T_HS, S_HS, NORM, MAXLEN = 400, 196, 17, 13, 64.0, 2048
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def log(msg):
    print(msg, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(msg + "\n")


def texts():
    p = OUT / "texts.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.open()]
    from train import ArrowAMDeepSeek
    ds = ArrowAMDeepSeek()
    rows = []
    for i in range(531_000, 700_000, 97):
        if ds.data[i]["messages"][0]["info"]["source"] in MATH_SOURCES:
            rows.append({"row": i, "text": ds[i]["text"]})
        if len(rows) == N_TEXT:
            break
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def sae_feats(h, we, be, dn):
    h = h.float()
    h = h * (NORM / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    return F.relu(h @ we.T + be) * dn


@torch.no_grad()
def teacher_targets(tok, rows):
    p = TENS / "targets.pt"
    if p.exists():
        return torch.load(p)
    from models.student_loader import load_teacher
    model = load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True, cache_dir=str(ROOT / "cache")).eval()
    sd = torch.load(str(SAE_PT), map_location="cpu")
    we, be = sd["encoder.weight"].float().cuda(), sd["encoder.bias"].float().cuda()
    dn = sd["decoder.weight"].float().norm(dim=0).cuda()
    rd = torch.load(str(RANK_PT), map_location="cpu")["sorted_indices"][:N_SET].long().tolist()
    L = we.shape[0]
    nz, n_tr, H = torch.zeros(L, device="cuda"), 0, []
    for i, r in enumerate(rows):
        ids = tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"]
        h = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[T_HS][0]
        H.append(h.to(torch.bfloat16).cpu())
        if i % 2 == 0:
            nz += (sae_feats(h, we, be, dn) > 0).float().sum(0)
            n_tr += h.shape[0]
        if i % 100 == 0:
            log(f"  teacher {i}/{len(rows)}")
    dens = (nz / n_tr).cpu().numpy()
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
    out = {"rd": rd, "rand": rand, "density": {f: float(dens[f]) for f in feats}, "Y": Y, "H": H}
    torch.save(out, p)
    log(f"targets: {len(Y)} rows, {sum(y.shape[0] for y in Y)} tokens; rd density median "
        f"{np.median([dens[f] for f in rd]):.2e}, rand {np.median([dens[f] for f in rand]):.2e}; "
        f"rd nonzero on train {int((dens[rd] > 0).sum())}/{N_SET}")
    del model
    torch.cuda.empty_cache()
    return out


def lnorm(x):
    return F.layer_norm(x.float(), x.shape[-1:])


def seq_stats(pred, y):
    n = torch.full_like(y[0], y.shape[0])
    return torch.stack([((pred - y) ** 2).sum(0), y.sum(0), (y ** 2).sum(0), n, (y > 0).float().sum(0)]).double().cpu()


def ridge_probe(name, layers_of, nl, d, rows, T):
    """layers_of(i) -> list of nl hidden tensors [t, d] for row i. Saves per-test-sequence stats."""
    p = TENS / f"probe_{name}.pt"
    if p.exists():
        return
    Y = T["Y"]
    tr = [i for i in range(len(rows)) if i % 2 == 0]
    te = [i for i in range(len(rows)) if i % 2 == 1]
    val = set(tr[int(0.8 * len(tr)):])
    ys = torch.cat([Y[i] for i in tr if i not in val]).std(0).clamp(min=1e-6).cuda()
    k, d1 = Y[0].shape[1], d + 1
    A = {g: torch.zeros(nl, d1, d1, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    B = {g: torch.zeros(nl, d1, k, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    C = {g: torch.zeros(k, dtype=torch.float64, device="cuda") for g in ("fit", "val")}
    t0 = time.time()
    for i in tr:
        hs = layers_of(i)
        y = Y[i].cuda() / ys
        g = "val" if i in val else "fit"
        for l in range(nl):
            x = torch.cat([lnorm(hs[l]), torch.ones(hs[l].shape[0], 1, device="cuda")], 1)
            A[g][l] += (x.T @ x).double()
            B[g][l] += (x.T @ y).double()
        C[g] += (y.double() ** 2).sum(0)
    W, chosen = [], []
    eye = torch.eye(d1, device="cuda", dtype=torch.float64)
    for l in range(nl):
        dg = A["fit"][l].diagonal().mean()
        best = None
        for lam in LAMBDAS:
            w = torch.linalg.solve(A["fit"][l] + lam * dg * eye, B["fit"][l])
            sse = C["val"].sum() - 2 * (w * B["val"][l]).sum() + (w * (A["val"][l] @ w)).sum()
            if best is None or sse < best[0]:
                best = (float(sse), lam)
        At, Bt = A["fit"][l] + A["val"][l], B["fit"][l] + B["val"][l]
        w = torch.linalg.solve(At + best[1] * At.diagonal().mean() * eye, Bt)
        W.append((w * ys.double()).float())
        chosen.append(best[1])
    del A, B
    torch.cuda.empty_cache()
    stats = torch.zeros(len(te), nl, 5, k, dtype=torch.float64)
    for n, i in enumerate(te):
        hs = layers_of(i)
        y = Y[i].cuda()
        for l in range(nl):
            x = torch.cat([lnorm(hs[l]), torch.ones(hs[l].shape[0], 1, device="cuda")], 1)
            stats[n, l] = seq_stats(x @ W[l], y)
    torch.save({"test_rows": te, "ridge": stats, "lambdas": chosen}, p)
    log(f"  {name}: done {time.time() - t0:.0f}s, lambdas {chosen}")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    TENS.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir=str(ROOT / "cache"))
    rows = texts()
    log(f"texts: {len(rows)} rows from {rows[0]['row']} to {rows[-1]['row']}")
    T = teacher_targets(tok, rows)
    ridge_probe("teacher_ceiling", lambda i: [T["H"][i].cuda()], 1, T["H"][0].shape[1], rows, T)
    for s, path in STUDENTS.items():
        if (TENS / f"probe_{s}.pt").exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0},
                                                     cache_dir=str(ROOT / "cache")).eval()

        @torch.no_grad()
        def layers_of(i, model=model):
            ids = tok(rows[i]["text"], truncation=True, max_length=MAXLEN)["input_ids"]
            hs = model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
            return [h[0] for h in hs[1:]]
        nl = model.config.get_text_config().num_hidden_layers
        ridge_probe(s, layers_of, nl, model.config.get_text_config().hidden_size, rows, T)
        del model
        torch.cuda.empty_cache()


def r2(st):
    sse, sy, sy2, n, nz = st.sum(0)
    return 1 - sse / (sy2 - sy ** 2 / n), nz


def analyze():
    T = torch.load(TENS / "targets.pt")
    P = {s: torch.load(TENS / f"probe_{s}.pt") for s in ["teacher_ceiling"] + list(STUDENTS)}
    rng = np.random.default_rng(0)
    n = P["lkd"]["ridge"].shape[0]
    boots = [torch.from_numpy(rng.integers(0, n, n)) for _ in range(1000)]
    sets = {"rd": slice(0, N_SET), "rand": slice(N_SET, 2 * N_SET)}
    _, nz = r2(P["lkd"]["ridge"][:, 0])
    live = nz >= 20
    lines = ["# Check 1 · Headroom: ridge R^2 of the new SAE's features from each model (400 held-out texts, 2048 tok)",
             f"live features (>= 20 nonzero test tokens): rd {int(live[sets['rd']].sum())}/196, rand {int(live[sets['rand']].sum())}/196",
             "mean R^2 over live features; ceiling = ridge from the teacher's own hidden_states[17]"]

    def m(st, sl):
        r, _ = r2(st)
        return float(r[sl][live[sl]].mean())

    for sname, sl in sets.items():
        lines.append(f"\n## {sname} set")
        lines.append(f"  teacher ceiling            {m(P['teacher_ceiling']['ridge'][:, 0], sl):.3f}")
        for s in STUDENTS:
            R = P[s]["ridge"]
            per = [m(R[:, l], sl) for l in range(R.shape[1])]
            best = int(np.argmax(per))
            lines.append(f"  {s:7s} hs[{S_HS}] {per[S_HS - 1]:.3f}   best hs[{best + 1}] {per[best]:.3f}   "
                         f"all layers " + " ".join(f"{x:.2f}" for x in per))
    lines.append(f"\n## paired differences at hs[{S_HS}] (bootstrap over test texts, 1000)")
    for sname, sl in sets.items():
        for a, b in (("lkd", "base"), ("rd_oct", "lkd"), ("teacher_ceiling", "lkd")):
            la = 0 if a == "teacher_ceiling" else S_HS - 1
            A, Bm = P[a]["ridge"][:, la], P[b]["ridge"][:, S_HS - 1]
            d0 = m(A, sl) - m(Bm, sl)
            bs = np.array([m(A[j], sl) - m(Bm[j], sl) for j in boots])
            pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 1000)
            lines.append(f"  {sname:4s} {a:15s} - {b:5s} {d0:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={pv:.3f}")
    new = [s for s in STUDENTS if s in NEW]
    if new:
        lines.append(f"\n## new-SAE RD vs LKD-continue at hs[{S_HS}]: gain on rd set, on rand set, and rd-gain minus rand-gain")
        for s in new:
            A, Bm = P[s]["ridge"][:, S_HS - 1], P["lkd"]["ridge"][:, S_HS - 1]
            row = []
            for f in (lambda X, Y: m(X, sets["rd"]) - m(Y, sets["rd"]), lambda X, Y: m(X, sets["rand"]) - m(Y, sets["rand"]),
                      lambda X, Y: (m(X, sets["rd"]) - m(Y, sets["rd"])) - (m(X, sets["rand"]) - m(Y, sets["rand"]))):
                d0 = f(A, Bm)
                bs = np.array([f(A[j], Bm[j]) for j in boots])
                pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 1000)
                row.append(f"{d0:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={pv:.3f}")
            lines.append(f"  {s:12s} rd {row[0]} | rand {row[1]} | rd-rand {row[2]} | abs rd R^2 {m(A, sets['rd']):.4f}")
    lines.append("\n## share of the base->ceiling gap that LKD already closed, at hs[13] (rd set)")
    c, b_, l_ = (m(P["teacher_ceiling"]["ridge"][:, 0], sets["rd"]), m(P["base"]["ridge"][:, S_HS - 1], sets["rd"]),
                 m(P["lkd"]["ridge"][:, S_HS - 1], sets["rd"]))
    lines.append(f"  (LKD - base) / (ceiling - base) = {(l_ - b_) / max(c - b_, 1e-9):.2f};  remaining gap ceiling - LKD = {c - l_:.3f}")
    out = "\n".join(lines)
    print(out)
    (OUT / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else run()
