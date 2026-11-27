"""How the alignment term raises reflection: causal tests against the matched alpha = 0 run (docs/mi_qwen_students.md).

Models   rd_zero (RD pipeline, alpha 0) and rd_high (alpha 0.40); bench also lkd, rd_low, rd_mod, rd_low_rr (if present).
Metric   P_R and its log-odds at step boundaries (mi_qwen_students_common), the Exp 3 test traces (213 questions x 2 sets).
Phases
  patch  activation patching rd_high <-> rd_zero: layer L's output replaced by the donor's, at every position (`all`)
         or at the step boundaries only (`bound`), every layer, both directions. share = (patched - receiver) /
         (donor - receiver), ratio of means over questions.
  graft  rd_zero with rd_high's weights in a layer band (0-8, 9-12, 13-23, all; single layers 9-12) and the reverse
         (rd_high with rd_zero's 9-12): P_R, and for the bands that reach hs[13] the probe R^2 of the teacher's RD
         features (mi_qwen_student_probe ridge) and the reflection-latent enrichment (pooled SAE, Exp 1 latent sets)
  bench  P_R on other benchmarks' finished traces of LKD and RD_mod (OlympiadBench, MMLU-think, HellaSwag-think, and
         MATH-500 second samples), up to 300 questions each, every model on the same text
  analyze
Outputs  eval_outputs/mi_qwen_students/alpha_causal/
Run: python scripts/mi_qwen_alpha_causal.py patch|graft|bench|analyze
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import mi_qwen_crosscoder as CC
from mi_qwen_alpha0_props import latent_sets
from mi_qwen_patching import per_q, run
from mi_qwen_students_common import (D, MAXTOK, MODELS, N_LAYERS, OUT, REFL, ROOT, Record, Replace, boot, boot_ratio,
                                     layers, load_model, log, reflection_ids, segment_spans, split, tokenizer,
                                     traces)

DC = OUT / "alpha_causal"
LAY = 12                                                  # layers[12] output = hidden_states[13]
BANDS = {"0-8": range(0, 9), "9-12": range(9, 13), "13-23": range(13, 24), "all": range(24),
         **{f"L{i}": [i] for i in range(9, 13)}}
BENCH_MODELS = ("lkd", "rd_zero", "rd_low_rr", "rd_low", "rd_mod", "rd_high")
N_BENCH = 300


def _setup():
    tok = tokenizer()
    trs_all = traces(tok)
    return tok, trs_all, split(trs_all, "test"), torch.tensor(reflection_ids(tok, trs_all)).cuda()


def _jsonl(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


@torch.no_grad()
def patch():
    _, _, trs, r_ids = _setup()
    ms = {"rd_zero": load_model("rd_zero"), "rd_high": load_model("rd_high")}
    path = DC / "patch.jsonl"
    DC.mkdir(parents=True, exist_ok=True)
    done = {(r["qi"], r["src"]) for r in _jsonl(path)}
    t0 = time.time()
    for n, t in enumerate(trs):
        if (t["qi"], t["src"]) in done:
            continue
        x = torch.tensor([t["ids"]]).cuda()
        rec = {"qi": t["qi"], "src": t["src"], "cond": {}}
        acts = {}
        for name, m in ms.items():
            r = Record(m)
            m.model(x)
            r.remove()
            acts[name] = r.out
            rec["cond"][name] = run(m, t, r_ids)
        bpos = torch.tensor(t["bpos"]).cuda()
        for donor, recv in (("rd_high", "rd_zero"), ("rd_zero", "rd_high")):
            for L in range(N_LAYERS):
                for mode, pos in (("all", None), ("bound", bpos)):
                    hk = Replace(layers(ms[recv])[L], acts[donor][L], pos)
                    rec["cond"][f"{donor}>{recv}|{mode}|{L}"] = run(ms[recv], t, r_ids)
                    hk.remove()
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if n % 50 == 0:
            log(f"  alpha_causal patch {n + 1}/{len(trs)} {time.time() - t0:.0f}s")
    log(f"alpha_causal patch: done {time.time() - t0:.0f}s")


class Graft:
    """recv's layers `idx` take donor's weights until restore()."""

    def __init__(self, recv, donor, idx):
        self.recv, self.idx = recv, list(idx)
        self.keep = {i: {k: v.clone() for k, v in layers(recv)[i].state_dict().items()} for i in self.idx}
        for i in self.idx:
            layers(recv)[i].load_state_dict(layers(donor)[i].state_dict())

    def restore(self):
        for i in self.idx:
            layers(self.recv)[i].load_state_dict(self.keep[i])


GRAFTS = {**{f"zero+high[{b}]": ("rd_zero", "rd_high", b) for b in BANDS}, "high+zero[9-12]": ("rd_high", "rd_zero", "9-12")}
GRAFT_HS13 = ("zero+high[0-8]", "zero+high[9-12]", "high+zero[9-12]")     # bands that change hs[13]


@torch.no_grad()
def scale_of(m):
    """the pooled SAE's activation scale rule (mi_qwen_alpha0_props.fresh_scale) for a model object."""
    tok = tokenizer()
    it = CC.batches(tok, CC.ROW0)
    s, n = 0.0, 0
    for _ in range(50):
        nr = CC.acts({"m": m}, next(it))[:, 0].norm(dim=-1)
        nr = nr[nr < 10 * nr.mean()]
        s += float(nr.sum()); n += nr.shape[0]
    return math.sqrt(D) / (s / n)


@torch.no_grad()
def enrichment(m, trs):
    """reflection-latent enrichment (Exp 1 definition: mean activation of the fixed top-512 on reflection tokens / on
    other tokens) and the same ratio for the 512 random live latents."""
    S = torch.load(CC.CK / "final.pt")
    sae = CC.BatchTopK(1).cuda()
    sae.load_state_dict(S["sae"])
    sae.eval()
    top, rnd = latent_sets()
    scale = scale_of(m)
    sr = torch.zeros(CC.F_LAT, dtype=torch.float64, device="cuda"); so = torch.zeros_like(sr)
    nr = no = 0
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
        fs, _ = sae.encode(CC.acts({"m": m}, torch.tensor([t["ids"]]).cuda()) * scale)
        sr += fs[r].sum(0).double(); so += fs[o].sum(0).double()
        nr += int(r.sum()); no += int(o.sum())
    e = lambda idx: float((sr[idx].sum() / nr) / (so[idx].sum() / no))  # noqa: E731
    return {"scale": scale, "top": e(top.cuda()), "rnd": e(rnd.cuda())}


_PROBE = {}


def probe(name, m):
    """ridge probe of the teacher's SAE features from hs[1..13] (mi_qwen_student_probe), saved as probe_<name>.pt."""
    import mi_qwen_student_probe as PB
    tok = tokenizer()
    if not _PROBE:
        _PROBE["rows"] = PB.texts()
        _PROBE["T"] = PB.teacher_targets(tok, _PROBE["rows"])
    rows, T = _PROBE["rows"], _PROBE["T"]

    @torch.no_grad()
    def layers_of(i):
        ids = tok(rows[i]["text"], truncation=True, max_length=PB.MAXLEN)["input_ids"]
        hs = m.model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
        return [h[0] for h in hs[1:LAY + 2]]
    PB.ridge_probe(name, layers_of, LAY + 1, D, rows, T)


@torch.no_grad()
def graft():
    tok, trs_all, trs, r_ids = _setup()
    DC.mkdir(parents=True, exist_ok=True)
    path = DC / "graft.json"
    res = json.loads(path.read_text()) if path.exists() else {"pr": {}, "enrich": {}}
    ms = {n: load_model(n) for n in ("rd_zero", "rd_high")}
    for n, m in ms.items():
        if n not in res["pr"]:
            res["pr"][n] = [run(m, t, r_ids) for t in trs]
    for c, (recv, donor, band) in GRAFTS.items():
        if c in res["pr"]:
            continue
        g = Graft(ms[recv], ms[donor], BANDS[band])
        res["pr"][c] = [run(ms[recv], t, r_ids) for t in trs]
        g.restore()
        log(f"  alpha_causal graft {c}: P_R done")
        path.write_text(json.dumps(res))
    del ms
    torch.cuda.empty_cache()
    ms = {n: load_model(n, n_layers=LAY + 1) for n in ("rd_zero", "rd_high")}
    for c in ("rd_zero", "rd_high") + GRAFT_HS13:
        if c in res["enrich"]:
            continue
        recv, donor, band = GRAFTS.get(c, (c, None, None))
        g = Graft(ms[recv], ms[donor], BANDS[band]) if donor else None
        res["enrich"][c] = enrichment(ms[recv], trs_all)
        if g:
            probe("graft_" + c.replace("+", "_").replace("[", "_").replace("]", ""), ms[recv])
            g.restore()
        log(f"  alpha_causal graft {c}: enrichment {res['enrich'][c]['top']:.3f}")
        path.write_text(json.dumps(res))
    log("alpha_causal graft: done")


def bench_traces(tok):
    """[{q, qi, src, ids, bpos, blab}] like mi_qwen_students_common.traces, from other benchmarks' finished traces."""
    p = DC / "bench_traces.json"
    if p.exists():
        return json.loads(p.read_text())
    from eval_reflection_benchmarks import BENCH, finished
    out = {}
    for bench, skip in (("olympiad", 0), ("mmlu_think", 0), ("hellaswag_think", 0), ("math500_p8", 1)):
        lkd_f, rd_f, _, sep = BENCH[bench]
        cands = {}
        for src, f in (("LKD", lkd_f), ("RD_mod", rd_f)):
            for rec in json.load(open(f))["records"]:
                fin = [c for c in sorted(rec["candidates"], key=lambda c: c["index"]) if finished(c, sep)]
                if len(fin) > skip:                    # MATH-500: the second finished sample (the first is the Exp 3 text)
                    cands.setdefault(rec["id"], {})[src] = fin[skip]
        qs = sorted(q for q, v in cands.items() if len(v) == 2)
        qs = [qs[i] for i in np.random.default_rng(0).permutation(len(qs))[:N_BENCH]]
        rows = []
        for qi, q in enumerate(qs):
            for src, c in cands[q].items():
                think = c["raw_output"].split("</think>")[0]
                spans = segment_spans(think)
                enc = tok(c["prompt"] + think, return_offsets_mapping=True, add_special_tokens=False)
                ids, offs = enc["input_ids"][:MAXTOK], enc["offset_mapping"][:MAXTOK]
                ends = np.array([e for _, e in offs])
                bpos = []
                for k in range(1, len(spans)):
                    tt = int(np.searchsorted(ends, len(c["prompt"]) + spans[k][0], side="right"))
                    if tt >= len(ids):
                        break
                    bpos.append(tt - 1)
                if bpos:
                    rows.append({"q": q, "qi": qi, "src": src, "ids": ids, "bpos": bpos, "blab": [""] * len(bpos)})
        out[bench] = rows
        log(f"alpha_causal bench traces {bench}: {len(qs)} questions, {len(rows)} traces")
    DC.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def _rerun_ok():
    """the alpha 0.05 rerun is used once its download has passed the sha256 check (marker written after the check)."""
    from pathlib import Path
    return (Path(MODELS["rd_low_rr"]).parent / "VERIFIED").exists()


@torch.no_grad()
def bench():
    tok, trs_all, _, r_ids = _setup()
    B = bench_traces(tok)
    path = DC / "bench.jsonl"
    done = {(r["model"], r["bench"]) for r in _jsonl(path)}
    for name in BENCH_MODELS:
        if name == "rd_low_rr" and not _rerun_ok():
            log("alpha_causal bench: rd_low_rr not downloaded and verified yet, skipped")
            continue
        if all((name, b) in done for b in B):
            continue
        m = load_model(name)
        for b, trs in B.items():
            if (name, b) in done:
                continue
            t0 = time.time()
            recs = [run(m, t, r_ids) for t in trs]
            with path.open("a") as f:
                f.write(json.dumps({"model": name, "bench": b, "records": recs}) + "\n")
            log(f"  alpha_causal bench {name} {b}: {len(recs)} traces {time.time() - t0:.0f}s")
        del m
        torch.cuda.empty_cache()
    log("alpha_causal bench: done")


def _c(a, b, key="pr"):
    """paired contrast a - b over questions."""
    pa, pb = per_q(a, key), per_q(b, key)
    qs = sorted(set(pa) & set(pb))
    return boot([pa[q] - pb[q] for q in qs])


def _fmt(t, f="+.5f"):
    m, lo, hi, p = t
    return f"{m:{f}} [{lo:{f}},{hi:{f}}] p={p:.4f}"


def _share(cond, recv, donor, key="pr"):
    a, b, c = per_q(recv, key), per_q(donor, key), per_q(cond, key)
    qs = sorted(set(a) & set(b) & set(c))
    return boot_ratio([c[q] - a[q] for q in qs], [b[q] - a[q] for q in qs])


def analyze():
    L = ["# How the alignment term raises reflection: causal tests vs the matched alpha = 0 run",
         "  P_R = probability of starting a reflection step at step boundaries; lo = its log-odds. Paired over questions,",
         "  bootstrap 10,000. share = (condition - receiver) / (donor - receiver), ratio of means."]
    P = _jsonl(DC / "patch.jsonl")
    if P:
        cond = lambda k: [{"qi": r["qi"], **r["cond"][k]} for r in P]  # noqa: E731
        L.append(f"\n## 1 · Activation patching rd_high <-> rd_zero ({len(P)} traces, {len({r['qi'] for r in P})} questions)")
        L.append(f"  gap rd_high - rd_zero: P_R {_fmt(_c(cond('rd_high'), cond('rd_zero')))}; "
                 f"log-odds {_fmt(_c(cond('rd_high'), cond('rd_zero'), 'lo'), '+.4f')}")
        for donor, recv in (("rd_high", "rd_zero"), ("rd_zero", "rd_high")):
            L.append(f"\n  {donor} -> {recv}: share of the gap recovered when layer L's output is the donor's")
            L.append("  layer  all positions P_R        boundary only P_R       all positions log-odds")
            for Ly in range(N_LAYERS):
                parts = []
                for mode, key in (("all", "pr"), ("bound", "pr"), ("all", "lo")):
                    r, lo, hi = _share(cond(f"{donor}>{recv}|{mode}|{Ly}"), cond(recv), cond(donor), key)
                    parts.append(f"{r:+.2f} [{lo:+.2f},{hi:+.2f}]")
                L.append(f"  {Ly:5d}  {parts[0]:22s}  {parts[1]:22s}  {parts[2]}")
    if (DC / "graft.json").exists():
        G = json.loads((DC / "graft.json").read_text())
        pr = G["pr"]
        L.append("\n## 2 · Weight graft (P_R on the test traces; share of the rd_high - rd_zero gap)")
        L.append(f"  rd_zero {np.mean([r['pr'] for r in pr['rd_zero']]):.5f}  rd_high {np.mean([r['pr'] for r in pr['rd_high']]):.5f}")
        for c, (recv, donor, _) in GRAFTS.items():
            if c not in pr:
                continue
            r, lo, hi = _share(pr[c], pr[recv], pr[donor])
            rl, lol, hil = _share(pr[c], pr[recv], pr[donor], "lo")
            L.append(f"  {c:18s} P_R {np.mean([x['pr'] for x in pr[c]]):.5f}  vs {recv} {_fmt(_c(pr[c], pr[recv]))}"
                     f"  share {r:+.2f} [{lo:+.2f},{hi:+.2f}]  (log-odds share {rl:+.2f} [{lol:+.2f},{hil:+.2f}])")
        if G["enrich"]:
            L.append("  reflection-latent enrichment at hs[13] (top-512 / random-512), activation scale:")
            for c, e in G["enrich"].items():
                L.append(f"    {c:18s} {e['top']:.3f} / {e['rnd']:.3f}   scale {e['scale']:.4f}")
        import mi_qwen_student_probe as PB
        names = {"rd_zero": "rd_zero", "rd_high": "rd_high",
                 **{c: "graft_" + c.replace("+", "_").replace("[", "_").replace("]", "") for c in GRAFT_HS13}}
        R = {c: torch.load(PB.TENS / f"probe_{n}.pt") for c, n in names.items() if (PB.TENS / f"probe_{n}.pt").exists()}
        if len(R) > 2:
            _, nz = PB.r2(R["rd_zero"]["ridge"][:, 0])
            live = nz >= 20
            sets = {"rd": slice(0, PB.N_SET), "rand": slice(PB.N_SET, 2 * PB.N_SET)}

            def m(st, sl):
                r, _ = PB.r2(st)
                return float(r[sl][live[sl]].mean())
            L.append(f"  probe R^2 of the teacher's features at hs[{PB.S_HS}] (rd set / random set):")
            for c, P_ in R.items():
                st = P_["ridge"][:, PB.S_HS - 1]
                L.append(f"    {c:18s} {m(st, sets['rd']):.4f} / {m(st, sets['rand']):.4f}")
    Bn = _jsonl(DC / "bench.jsonl")
    if Bn:
        by = {}
        for r in Bn:
            by.setdefault(r["bench"], {})[r["model"]] = r["records"]
        L.append("\n## 4 · P_R on other benchmarks (finished traces of LKD and RD_mod, same text for every model)")
        pooled = {}
        for b, M in by.items():
            L.append(f"\n  {b} ({len(per_q(next(iter(M.values()))))} questions)  " +
                     "  ".join(f"{n} {np.mean(list(per_q(M[n]).values())):.4f}" for n in BENCH_MODELS if n in M))
            for base in ("rd_zero", "lkd"):
                for n in BENCH_MODELS:
                    if n in M and base in M and n != base:
                        L.append(f"    {n:9s} - {base:7s} {_fmt(_c(M[n], M[base]))}")
            for n, recs in M.items():
                pooled.setdefault(n, []).extend({**r, "qi": f"{b}:{r['qi']}"} for r in recs)
        L.append(f"\n  pooled over benchmarks ({len(per_q(pooled['rd_zero']))} questions)" if "rd_zero" in pooled else "")
        for n in BENCH_MODELS:
            if n in pooled and n != "rd_zero" and "rd_zero" in pooled:
                L.append(f"    {n:9s} - rd_zero {_fmt(_c(pooled[n], pooled['rd_zero']))}")
    out = "\n".join(L)
    print(out)
    (DC / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["patch", "graft", "bench", "analyze"])
    a = ap.parse_args()
    {"patch": patch, "graft": graft, "bench": bench, "analyze": analyze}[a.phase]()
