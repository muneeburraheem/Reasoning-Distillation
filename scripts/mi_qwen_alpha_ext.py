"""Teacher agreement and structure of the alignment term, against the matched alpha = 0 run (docs/mi_qwen_students.md,
"Teacher agreement and structure of the alignment term"). Expectations were written in the doc before any phase ran.

Models   lkd, rd_zero (alpha 0), rd_low / rd_low_rr / rd_mod / rd_high (alpha 0.05 / 0.05 rerun / 0.15 / 0.40); teacher
         Qwen3.5-4B 8-bit, as in RD training
Features the M1 probe targets (mi_qwen_student_probe): teacher hidden_states[17] through the new SAE, 196 top-ReasonScore
         (rd) + 196 density-matched random (rand) features
Phases
  teacher   X1 teacher reflection log-odds at every step boundary of the Exp 3 test traces; teacher features on the
            off-math corpora (X5)
  students  per model: X1 log-odds per boundary, X5 probes on the off-math corpora
  svd       X4 singular-value shape of the weight change, every matrix of every layer. Control added after the first
            run showed the noise is mostly 1-ulp bf16 flips: train = rd_low final - rd_low halfway (a structured update
            that is mostly not the term)
  arith     X2 rd_zero + k (rd_high - rd_zero) and the control rd_zero + k s (rd_low_rr - rd_low), s = norm match:
            P_R, log-odds and next-token loss on the test traces, M1 probe on the math probe texts
  x5fix     X5 robustness, added after the bench probe chose lambda 0.01 for rd_high but 0.1 for the others at hs[13]:
            both off-math probes refit with lambda fixed at 0.1 and at 0.01 for every model
  analyze
Outputs  eval_outputs/mi_qwen_students/alpha_ext/; probe stats checkpoints/rd_newsae_probe/probe_x*.pt and ext/
Run: python scripts/mi_qwen_alpha_ext.py teacher|students|svd|arith|x5fix|analyze
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mi_qwen_student_probe as PB
from mi_qwen_alpha0_props import PROBE_NAME
from mi_qwen_alpha_causal import _c, _fmt, _jsonl, _setup
from mi_qwen_students_common import D, MODELS, N_LAYERS, OUT, ROOT, load_model, log, p_r

DX = OUT / "alpha_ext"
TX = PB.TENS / "ext"
XMODELS = ("rd_zero", "lkd", "rd_low", "rd_low_rr", "rd_mod", "rd_high")
MIN_B = 10                                                            # X1: traces with at least this many boundaries
CORPORA = {"nr": "general-science reasoning (AM natural_reasoning, rows 531,000+)",
           "bench": "MMLU-think + HellaSwag-think traces (C4)"}
ARITH = {"k0": ("term", 0.0), "k0.5": ("term", 0.5), "k1": ("term", 1.0), "k2": ("term", 2.0), "k3": ("term", 3.0),
         "ctl1": ("noise", 1.0), "ctl2": ("noise", 2.0), "ctl3": ("noise", 3.0)}


def _feats():
    p = TX / "feats.pt"
    if not p.exists():
        TX.mkdir(parents=True, exist_ok=True)
        T = torch.load(PB.TENS / "targets.pt")
        torch.save({"rd": T["rd"], "rand": T["rand"]}, p)
    return torch.load(p)


def nr_rows(tok):
    p = DX / "x5_nr_rows.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.open()]
    from train_sae_norm64 import ArrowAMDeepSeek
    ds, rows = ArrowAMDeepSeek(), []
    for i in range(531_000, 900_000, 97):
        if ds.data[i]["messages"][0]["info"]["source"] == "natural_reasoning":
            rows.append({"row": i, "ids": tok(ds[i]["text"], truncation=True, max_length=PB.MAXLEN)["input_ids"]})
        if len(rows) == PB.N_TEXT:
            break
    DX.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def bench_rows():
    """200 questions each of MMLU-think and HellaSwag-think, one trace per question, source alternating in pairs so the
    probe's even / odd split sees both sources."""
    p = DX / "x5_bench_rows.json"
    if p.exists():
        return json.loads(p.read_text())
    B = json.loads((OUT / "alpha_causal/bench_traces.json").read_text())
    rows = []
    for bench in ("mmlu_think", "hellaswag_think"):
        by = {}
        for t in B[bench]:
            by.setdefault(t["qi"], {})[t["src"]] = t
        qs = [q for q in sorted(by) if len(by[q]) == 2][:200]
        rows += [{"bench": bench, "qi": q, "src": s, "ids": by[q][s]["ids"]}
                 for j, q in enumerate(qs) for s in [("LKD", "RD_mod")[(j // 2) % 2]]]
    DX.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


# ---------------------------------------------------------------- teacher

@torch.no_grad()
def teacher():
    tok, _, trs, r_ids = _setup()
    DX.mkdir(parents=True, exist_ok=True)
    TX.mkdir(parents=True, exist_ok=True)
    fs = _feats()
    idx = fs["rd"] + fs["rand"]
    sd = torch.load(str(PB.SAE_PT), map_location="cpu")
    we, be = sd["encoder.weight"].float()[idx].cuda(), sd["encoder.bias"].float()[idx].cuda()
    dn = sd["decoder.weight"].float().norm(dim=0)[idx].cuda()
    from models.student_loader import load_teacher
    m = load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True, cache_dir=str(ROOT / "cache"))

    def fwd(ids):
        o = m.model(torch.tensor([ids]).cuda(), output_hidden_states=True)
        return o.last_hidden_state[0], PB.sae_feats(o.hidden_states[PB.T_HS][0], we, be, dn).half().cpu()
    t0 = time.time()
    p1 = DX / "x1_lo.jsonl"
    if "teacher" not in {r["model"] for r in _jsonl(p1)}:
        recs = []
        for n, t in enumerate(trs):
            last, _ = fwd(t["ids"])
            _, lo = p_r(m, last, torch.tensor(t["bpos"]).cuda(), r_ids)
            recs.append({"qi": t["qi"], "src": t["src"], "lo": lo.tolist()})
            if n % 100 == 0:
                log(f"  alpha_ext teacher traces {n}: {time.time() - t0:.0f}s")
        with p1.open("a") as f:
            f.write(json.dumps({"model": "teacher", "records": recs}) + "\n")
        log(f"  alpha_ext teacher: mean reflection log-odds {np.mean([np.mean(r['lo']) for r in recs]):.4f}")
    for c, rows in (("nr", nr_rows(tok)), ("bench", bench_rows())):
        p = TX / f"x5{c}_targets.pt"
        if not p.exists():
            torch.save({"Y": [fwd(r["ids"])[1] for r in rows]}, p)
            log(f"  alpha_ext teacher {c}: {len(rows)} rows, {time.time() - t0:.0f}s")
    log("alpha_ext teacher: done")


# ---------------------------------------------------------------- students

def _hs(m, ids):
    return [h[0] for h in m.model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[1:]]


@torch.no_grad()
def students():
    tok, _, trs, r_ids = _setup()
    X5 = {c: (r, [y.float() for y in torch.load(TX / f"x5{c}_targets.pt")["Y"]])
          for c, r in (("nr", nr_rows(tok)), ("bench", bench_rows()))}
    p1 = DX / "x1_lo.jsonl"
    for name in XMODELS:
        pn = PROBE_NAME[name]
        todo = (name not in {r["model"] for r in _jsonl(p1)}, any(not (PB.TENS / f"probe_x5{c}_{pn}.pt").exists() for c in X5))
        if not any(todo):
            continue
        m, t0 = load_model(name), time.time()
        if todo[0]:
            recs = []
            for t in trs:
                h = m.model(torch.tensor([t["ids"]]).cuda()).last_hidden_state[0]
                _, lo = p_r(m, h, torch.tensor(t["bpos"]).cuda(), r_ids)
                recs.append({"qi": t["qi"], "src": t["src"], "lo": lo.tolist()})
            with p1.open("a") as f:
                f.write(json.dumps({"model": name, "records": recs}) + "\n")
            log(f"  alpha_ext {name} X1: {time.time() - t0:.0f}s")
        for c, (rr, Y) in X5.items():
            PB.ridge_probe(f"x5{c}_{pn}", lambda i, rr=rr: _hs(m, rr[i]["ids"]), N_LAYERS, D, rr, {"Y": Y})
        log(f"  alpha_ext {name} X5: {time.time() - t0:.0f}s")
        del m
        torch.cuda.empty_cache()
    log("alpha_ext students: done")


@torch.no_grad()
def x5fix():
    tok = _setup()[0]
    X5 = {c: (r, [y.float() for y in torch.load(TX / f"x5{c}_targets.pt")["Y"]])
          for c, r in (("nr", nr_rows(tok)), ("bench", bench_rows()))}
    keep = PB.LAMBDAS
    for name in XMODELS:
        m = load_model(name)
        for lam, tag in ((0.1, "L1"), (0.01, "L2")):
            PB.LAMBDAS = (lam,)
            for c, (rr, Y) in X5.items():
                PB.ridge_probe(f"x5{c}{tag}_{PROBE_NAME[name]}", lambda i, rr=rr: _hs(m, rr[i]["ids"]), N_LAYERS, D, rr, {"Y": Y})
        PB.LAMBDAS = keep
        log(f"  alpha_ext x5fix {name}: done")
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- X4 weight-change shape

@torch.no_grad()
def svd():
    from safetensors import safe_open
    names = ("rd_zero", "rd_low", "rd_low_rr", "rd_mod", "rd_high")
    Fs = {n: safe_open(str(Path(MODELS[n]) / "model.safetensors"), "pt", device="cpu") for n in names}
    Fs["rd_low_half"] = safe_open(str(Path(MODELS["rd_low"]).parent / "step_47_halfway/model.safetensors"), "pt", device="cpu")
    keys = [k for k in Fs["rd_zero"].keys() if ".layers." in k and len(Fs["rd_zero"].get_slice(k).get_shape()) == 2]
    share = lambda s, k: float((s[:k] ** 2).sum() / (s ** 2).sum())  # noqa: E731
    ov = lambda a, b: float(((a.T @ b) ** 2).sum() / a.shape[1])      # noqa: E731
    out = []
    for key in keys:
        w = {n: Fs[n].get_tensor(key).float().cuda() for n in Fs}
        dl = {"term": w["rd_high"] - w["rd_zero"], "noise": w["rd_low_rr"] - w["rd_low"], "step": w["rd_mod"] - w["rd_low"],
              "train": w["rd_low"] - w["rd_low_half"]}
        sv = {k: torch.linalg.svd(v, full_matrices=False) for k, v in dl.items()}
        r = {"key": key, "layer": int(key.split(".layers.")[1].split(".")[0]),
             "kind": key.split(".layers.")[1].split(".", 1)[1].removesuffix(".weight"), "shape": list(w["rd_zero"].shape),
             "norm": {k: float(v.norm()) for k, v in dl.items()}, "w_norm": float(w["rd_zero"].norm()),
             "share": {k: {str(j): share(s.S, j) for j in (1, 8, 32)} for k, s in sv.items()}}
        for a in ("step", "noise", "train"):
            r[f"ov_{a}_term"] = {"left": ov(sv[a].U[:, :8], sv["term"].U[:, :8]),
                                 "right": ov(sv[a].Vh[:8].T, sv["term"].Vh[:8].T)}
        out.append(r)
        del w, dl, sv
    DX.mkdir(parents=True, exist_ok=True)
    (DX / "x4_svd.json").write_text(json.dumps(out))
    log(f"alpha_ext svd: {len(out)} matrices")


# ---------------------------------------------------------------- X2 weight arithmetic

@torch.no_grad()
def arith():
    from transformers import AutoModelForCausalLM
    tok, _, trs, r_ids = _setup()
    path = DX / "x2_arith.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if all(c in res and (PB.TENS / f"probe_x2_{c}.pt").exists() for c in ARITH):
        return
    m = load_model("rd_zero")
    live = m.state_dict()
    keys = [k for k, v in live.items() if v.is_floating_point()]
    base = {k: live[k].detach().cpu().clone() for k in keys}
    cpu = lambda n: AutoModelForCausalLM.from_pretrained(MODELS[n], dtype=torch.bfloat16, device_map={"": "cpu"},  # noqa: E731
                                                        cache_dir=str(ROOT / "cache")).state_dict()
    delta = {}
    for kind, (a, b) in (("term", ("rd_high", "rd_zero")), ("noise", ("rd_low_rr", "rd_low"))):
        A = cpu(a)
        Bm = base if b == "rd_zero" else cpu(b)
        assert set(keys) <= set(A) and set(keys) <= set(Bm)
        delta[kind] = {k: A[k].float() - Bm[k].float() for k in keys}
        del A, Bm
    nrm = {k: math.sqrt(sum(float(v.norm()) ** 2 for kk, v in d.items() if kk != "lm_head.weight")) for k, d in delta.items()}
    s = nrm["term"] / nrm["noise"]
    res["norms"] = {**nrm, "scale": s}
    log(f"alpha_ext arith: |term| {nrm['term']:.3f}, |noise| {nrm['noise']:.3f}, control scale {s:.2f}")
    rows, T = PB.texts(), torch.load(PB.TENS / "targets.pt")
    T = {"Y": T["Y"]}
    for c, (kind, k) in ARITH.items():
        if c in res and (PB.TENS / f"probe_x2_{c}.pt").exists():
            continue
        f = k * (1.0 if kind == "term" else s)
        for key in keys:
            live[key].copy_((base[key].float() + f * delta[kind][key]).to(torch.bfloat16))
        recs, t0 = [], time.time()
        for t in trs:
            x = torch.tensor([t["ids"]]).cuda()
            h = m.model(x).last_hidden_state[0]
            pr, lo = p_r(m, h, torch.tensor(t["bpos"]).cuda(), r_ids)
            nll = 0.0
            for a in range(0, x.shape[1] - 1, 512):
                e = min(a + 512, x.shape[1] - 1)
                nll += float(F.cross_entropy(m.lm_head(h[a:e]).float(), x[0, a + 1:e + 1], reduction="sum"))
            recs.append({"qi": t["qi"], "src": t["src"], "pr": float(pr.mean()), "lo": float(lo.mean()),
                         "nll": nll / (x.shape[1] - 1)})
        res[c] = recs
        path.write_text(json.dumps(res))
        PB.ridge_probe(f"x2_{c}", lambda i: _hs(m, tok(rows[i]["text"], truncation=True, max_length=PB.MAXLEN)["input_ids"]),
                       N_LAYERS, D, rows, T)
        log(f"  alpha_ext arith {c}: P_R {np.mean([r['pr'] for r in recs]):.5f}, loss {np.mean([r['nll'] for r in recs]):.4f}, "
            f"{time.time() - t0:.0f}s")
    log("alpha_ext arith: done")


# ---------------------------------------------------------------- analysis

def _r2m(st, sl, live):
    r, _ = PB.r2(st)
    return float(r[sl][live[sl]].mean())


SETS = {"rd": slice(0, PB.N_SET), "rand": slice(PB.N_SET, 2 * PB.N_SET)}


_CNT = {}


def _counts(n, B=1000, seed=0):
    """resample counts [B, n]: the resamples of mi_qwen_alpha_mech._boot_diff, so a bootstrap is one matmul."""
    if n not in _CNT:
        rng = np.random.default_rng(seed)
        _CNT[n] = torch.from_numpy(np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])).double()
    return _CNT[n]


def _r2b(st, sl, live):
    """mean R^2 over live features of set sl for every bootstrap resample of the units of st [n, 5, k] -> [B]."""
    sse, sy, sy2, n, _ = (_counts(st.shape[0]) @ st.reshape(st.shape[0], -1)).reshape(-1, *st.shape[1:]).unbind(1)
    r = 1 - sse / (sy2 - sy ** 2 / n)
    return r[:, sl][:, live[sl]].mean(1).numpy()


def _ci(point, bs):
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / len(bs))
    return f"{point:+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] p={p:.3f}"


def _probe_diff(A, B, sl, live):
    return _ci(_r2m(A, sl, live) - _r2m(B, sl, live), _r2b(A, sl, live) - _r2b(B, sl, live))


def _sign_p(k, n):
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n)


def analyze():
    L = ["# Teacher agreement and structure of the alignment term vs the matched alpha = 0 run (expectations in",
         "  docs/mi_qwen_students.md). Paired contrasts vs rd_zero; bootstrap 10,000 over questions unless noted."]
    others = XMODELS[1:]
    # X1
    R = {r["model"]: {(x["qi"], x["src"]): x["lo"] for x in r["records"]} for r in _jsonl(DX / "x1_lo.jsonl")}
    if "teacher" in R and all(n in R for n in XMODELS):
        tch = R["teacher"]

        def rows_of(n):
            out = []
            for key, lo in R[n].items():
                s, t = np.array(lo), np.array(tch[key])
                if len(t) >= MIN_B:
                    out.append({"qi": key[0], "src": key[1], "r": float(np.corrcoef(s, t)[0, 1]),
                                "mad": float(np.abs(s - t).mean()), "lo": float(s.mean())})
            return out
        X = {n: rows_of(n) for n in XMODELS}
        L.append(f"\n## X1 · reflection placement vs the teacher ({len(X['rd_zero'])} test traces with >= {MIN_B} boundaries)")
        L.append(f"  teacher mean reflection log-odds {np.mean([np.mean(v) for v in tch.values()]):.4f}")
        L.append("  model       r(student, teacher)  |student - teacher| lo   r - rd_zero [95% CI]            |diff| - rd_zero")
        for n in XMODELS:
            r_, d_ = np.mean([x["r"] for x in X[n]]), np.mean([x["mad"] for x in X[n]])
            c1 = _fmt(_c(X[n], X["rd_zero"], "r"), "+.4f") if n != "rd_zero" else ""
            c2 = _fmt(_c(X[n], X["rd_zero"], "mad"), "+.4f") if n != "rd_zero" else ""
            L.append(f"  {n:10s}  {r_:.4f}               {d_:.4f}                 {c1:32s} {c2}")
    # X2
    p2 = DX / "x2_arith.json"
    if p2.exists():
        A2 = json.loads(p2.read_text())
        L.append(f"\n## X2 · weight arithmetic rd_zero + k (rd_high - rd_zero); control rd_zero + k s (rd_low_rr - rd_low)")
        L.append(f"  |term| {A2['norms']['term']:.3f}, |noise| {A2['norms']['noise']:.3f}, s = {A2['norms']['scale']:.2f}")
        conds = [c for c in ARITH if c in A2 and (PB.TENS / f"probe_x2_{c}.pt").exists()]
        P = {c: torch.load(PB.TENS / f"probe_x2_{c}.pt")["ridge"] for c in conds}
        Pz = torch.load(PB.TENS / "probe_rd_zero.pt")["ridge"]
        _, nz = PB.r2(Pz[:, 0])
        live = nz >= 20
        L.append("  cond    P_R      log-odds - k0 [95% CI]              loss    loss - k0                 "
                 "probe rd hs13 - k0 [CI]           rand hs13 - k0              rd hs24 - k0")
        for c in conds:
            pr = np.mean([r["pr"] for r in A2[c]]); ls = np.mean([r["nll"] for r in A2[c]])
            d_lo = _fmt(_c(A2[c], A2["k0"], "lo"), "+.4f") if c != "k0" else ""
            d_ls = _fmt(_c(A2[c], A2["k0"], "nll"), "+.4f") if c != "k0" else ""
            pd = [_probe_diff(P[c][:, l], P["k0"][:, l], SETS[s], live) if c != "k0" else ""
                  for s, l in (("rd", 12), ("rand", 12), ("rd", 23))]
            L.append(f"  {c:6s}  {pr:.5f}  {d_lo:34s}  {ls:.4f}  {d_ls:24s}  {pd[0]:32s}  {pd[1]:26s}  {pd[2]}")
        if "k1" in P:
            Ph = torch.load(PB.TENS / "probe_rd_high.pt")["ridge"]
            g = json.loads((OUT / "alpha_causal/graft.json").read_text())["pr"]
            L.append(f"  sanity k0 = rd_zero: probe rd hs13 {_r2m(P['k0'][:, 12], SETS['rd'], live):.4f} vs "
                     f"{_r2m(Pz[:, 12], SETS['rd'], live):.4f}; P_R {np.mean([r['pr'] for r in A2['k0']]):.5f} vs "
                     f"{np.mean([r['pr'] for r in g['rd_zero']]):.5f}")
            L.append(f"  sanity k1 = rd_high: probe rd hs13 {_r2m(P['k1'][:, 12], SETS['rd'], live):.4f} vs "
                     f"{_r2m(Ph[:, 12], SETS['rd'], live):.4f}; P_R {np.mean([r['pr'] for r in A2['k1']]):.5f} vs "
                     f"{np.mean([r['pr'] for r in g['rd_high']]):.5f}")
        L.append("  probe rd-set gain over k0 by layer:")
        L.append("  layer " + " ".join(f"{c:>8s}" for c in conds[1:]))
        for l in range(N_LAYERS):
            L.append(f"  hs[{l + 1:2d}] " + " ".join(f"{_r2m(P[c][:, l], SETS['rd'], live) - _r2m(P['k0'][:, l], SETS['rd'], live):+8.4f}"
                                              for c in conds[1:]))
    # X4
    p4 = DX / "x4_svd.json"
    if p4.exists():
        S4 = [r for r in json.loads(p4.read_text()) if min(r["shape"]) >= 256]
        L.append("\n## X4 · shape of the weight change (matrices with both dims >= 256). term = rd_high - rd_zero,")
        L.append("  noise = rd_low_rr - rd_low, step = rd_mod - rd_low. share_k = fraction of |change|^2 in the top k")
        L.append("  singular directions; overlap = mean squared cosine between top-8 subspaces (random ~ 8/dim)")
        for name, sel in (("layers 9-12", lambda r: 9 <= r["layer"] <= 12), ("layers 0-8", lambda r: r["layer"] <= 8),
                          ("layers 13-23", lambda r: r["layer"] >= 13), ("all layers", lambda r: True)):
            rs = [r for r in S4 if sel(r)]
            L.append(f"\n  {name} ({len(rs)} matrices)")
            L.append(f"    |term| / |noise| median {np.median([r['norm']['term'] / r['norm']['noise'] for r in rs]):.2f}; "
                     f"|term| / |W| median {np.median([r['norm']['term'] / r['w_norm'] for r in rs]):.2e}")
            for k in ("1", "8", "32"):
                t = [r["share"]["term"][k] for r in rs]; nn = [r["share"]["noise"][k] for r in rs]
                st = [r["share"]["step"][k] for r in rs]; tr = [r["share"]["train"][k] for r in rs]
                w, w2 = sum(a > b for a, b in zip(t, nn)), sum(a > b for a, b in zip(t, tr))
                L.append(f"    share_{k:2s} term {np.median(t):.4f}  step {np.median(st):.4f}  noise {np.median(nn):.4f}  "
                         f"train {np.median(tr):.4f}  term > noise in {w}/{len(rs)} (sign p={_sign_p(w, len(rs)):.2g}), "
                         f"term > train in {w2}/{len(rs)} (p={_sign_p(w2, len(rs)):.2g})")
            for side in ("left", "right"):
                a = [r["ov_step_term"][side] for r in rs]; b = [r["ov_noise_term"][side] for r in rs]
                c = [r["ov_train_term"][side] for r in rs]
                w, w2 = sum(x > y for x, y in zip(a, b)), sum(x > y for x, y in zip(a, c))
                L.append(f"    overlap {side:5s} (step, term) {np.median(a):.4f}  (noise, term) {np.median(b):.4f}  "
                         f"(train, term) {np.median(c):.4f}  step > noise in {w}/{len(rs)} (sign p={_sign_p(w, len(rs)):.2g}), "
                         f"step > train in {w2}/{len(rs)} (p={_sign_p(w2, len(rs)):.2g})")
        L.append("\n  per layer: median share_8 term / noise / train, median left overlap with term: step / noise / train")
        for l in range(N_LAYERS):
            rs = [r for r in S4 if r["layer"] == l]
            L.append(f"    L{l:2d}  {np.median([r['share']['term']['8'] for r in rs]):.4f} / "
                     f"{np.median([r['share']['noise']['8'] for r in rs]):.4f} / {np.median([r['share']['train']['8'] for r in rs]):.4f}   "
                     f"{np.median([r['ov_step_term']['left'] for r in rs]):.4f} / {np.median([r['ov_noise_term']['left'] for r in rs]):.4f} / "
                     f"{np.median([r['ov_train_term']['left'] for r in rs]):.4f}")
    # X5
    for c, desc in [(c + t, f"{d}{td}") for c, d in CORPORA.items()
                    for t, td in (("", ""), ("L1", " [lambda fixed 0.1]"), ("L2", " [lambda fixed 0.01]"))]:
        fs = {n: PB.TENS / f"probe_x5{c}_{PROBE_NAME[n]}.pt" for n in XMODELS}
        if not all(f.exists() for f in fs.values()):
            continue
        P = {n: torch.load(f)["ridge"] for n, f in fs.items()}
        _, nz = PB.r2(P["rd_zero"][:, 0])
        live = nz >= 20
        L.append(f"\n## X5 · M1 probe off math: {desc}; {P['rd_zero'].shape[0]} test texts, bootstrap 1,000")
        L.append(f"  live features rd {int(live[SETS['rd']].sum())}/196, rand {int(live[SETS['rand']].sum())}/196")
        for s in SETS:
            L.append(f"  {s} set: R^2 of rd_zero, then each model - rd_zero [95% CI]")
            for l in (1, 4, 7, 10, 13, 16, 19, 22, 24):
                L.append(f"  hs[{l:2d}] {_r2m(P['rd_zero'][:, l - 1], SETS[s], live):.4f}  " +
                         "  ".join(f"{n} {_probe_diff(P[n][:, l - 1], P['rd_zero'][:, l - 1], SETS[s], live)}" for n in others))
    out = "\n".join(L)
    print(out)
    DX.mkdir(parents=True, exist_ok=True)
    (DX / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["teacher", "students", "svd", "arith", "x5fix", "analyze"])
    a = ap.parse_args()
    {"teacher": teacher, "students": students, "svd": svd, "arith": arith, "x5fix": x5fix, "analyze": analyze}[a.phase]()
