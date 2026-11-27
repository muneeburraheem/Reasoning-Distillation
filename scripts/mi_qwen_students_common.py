"""Shared pieces of the Qwen student MI suite (docs/mi_qwen_students.md): models, traces, step boundaries, metric.

Models   anc = LKD epoch 1 (the shared start of LKD-continue and every new-SAE RD run; checkpoints/rd_newsae_student_init
         holds a copy), lkd = LKD-continue, rd_mod / rd_low / rd_high = new-SAE RD (alpha 0.15 / 0.05 / 0.40)
Traces   the step-labelled MATH-500 traces of RUN_LOG 8n / 8ai (eval_outputs/rd_newsae/eval_mod/bonds/steps.json): one
         finished sample per question from LKD and from RD_mod, prompt + thinking, first MAXTOK tokens. Both models are
         always run on BOTH trace sets, so every comparison is on identical text. Even question index = fit, odd = test.
Bounds   a step boundary is the position whose next token is the first token of a step (steps split as in
         eval_bond_annotate.segment); label = the annotator's label of that step. Step 0 is skipped.
Metric   P_R = probability the model gives to starting a reflection step, = sum of softmax over R_ids at a boundary.
         R_ids (fixed on the fit traces only): step-first tokens seen >= 100 times whose steps are labelled
         self-reflection >= 50% of the time.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from mi_qwen_student_probe import STUDENTS as PROBE_STUDENTS  # noqa: E402
from eval_answer_entropy import MODELS as TRACE_SRC  # noqa: E402

OUT = ROOT / "eval_outputs/mi_qwen_students"
STEPS = ROOT / "eval_outputs/rd_newsae/eval_mod/bonds/steps.json"
MODELS = {"anc": str(ROOT / "checkpoints/rd_newsae_student_init"), "lkd": PROBE_STUDENTS["lkd"],
          "rd_mod": PROBE_STUDENTS["rd_moderate"], "rd_low": PROBE_STUDENTS["rd_low"],
          "rd_high": PROBE_STUDENTS["rd_high"],
          # 2026-11-24 Vast runs, same RD pipeline: alpha 0 (ablation) and alpha 0.05 rerun (replication), same machine
          "rd_zero": str(ROOT / "checkpoints/rd_newsae_zero/final_step_94"),
          "rd_low_rr": str(ROOT / "checkpoints/rd_newsae_lowrerun/final_step_94")}
SOURCES = ("LKD", "RD_mod")            # whose traces
N_LAYERS, D, MAXTOK = 24, 2048, 3072
REFL = "self-reflection"


def log(msg, name="progress.log"):
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("a") as f:
        f.write(msg + "\n")


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir=str(ROOT / "cache"))


def load_model(name, n_layers=None):
    """bf16 on GPU; refuses to continue on any missing/unexpected weight. n_layers truncates the stack (no head)."""
    from transformers import AutoModelForCausalLM
    m, info = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16, device_map={"": 0},
                                                   cache_dir=str(ROOT / "cache"), output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    assert not bad, f"{name}: loading problems {bad}"
    if n_layers is not None:
        m.model.layers = m.model.layers[:n_layers]
        m.lm_head = None
    return m.eval().requires_grad_(False)


def layers(model):
    return model.model.layers


def segment_spans(thinking):
    """(start, stripped text) of every step, the same split and filter as eval_bond_annotate.segment."""
    out, pos = [], 0
    for m in list(re.finditer(r"\n\n|\n|(?<=\.) ", thinking)) + [None]:
        end = len(thinking) if m is None else m.start()
        piece = thinking[pos:end]
        if piece and piece.strip():
            out.append((pos + len(piece) - len(piece.lstrip()), piece.strip()))
        if m is not None:
            pos = m.end()
    return out


def traces(tok):
    """[{q, qi, src, ids, bpos, blab}] for both trace sources, cached in OUT/traces.json."""
    p = OUT / "traces.json"
    if p.exists():
        return json.loads(p.read_text())
    steps = json.loads(STEPS.read_text())
    by = {}
    for s in steps:
        if s["model"] in SOURCES:
            by.setdefault((s["model"], s["qid"]), {})[s["step"]] = s
    qids = sorted({q for (_, q) in by})
    out = []
    for src in SOURCES:
        recs = {r["id"]: r for r in json.load(open(ROOT / TRACE_SRC[src][1]))["records"]}
        for qi, q in enumerate(qids):
            st = by[(src, q)]
            cand = next(c for c in recs[q]["candidates"] if c["index"] == st[0]["idx"])
            think = cand["raw_output"].split("</think>")[0]
            spans = segment_spans(think)
            assert len(spans) == st[0]["n_steps"] and all(spans[k][1] == st[k]["text"] for k in st), (src, q)
            text = cand["prompt"] + think
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"][:MAXTOK], enc["offset_mapping"][:MAXTOK]
            ends = np.array([e for _, e in offs])
            base = len(cand["prompt"])
            bpos, blab = [], []
            for k in range(1, len(spans)):
                t = int(np.searchsorted(ends, base + spans[k][0], side="right"))   # first token ending past the start
                if t >= len(ids):
                    break
                bpos.append(t - 1); blab.append(st[k]["label"])
            out.append({"q": q, "qi": qi, "src": src, "ids": ids, "bpos": bpos, "blab": blab})
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def reflection_ids(tok, trs):
    p = OUT / "reflection_tokens.json"
    if p.exists():
        return json.loads(p.read_text())["ids"]
    n, r = {}, {}
    for t in trs:
        if t["qi"] % 2:
            continue
        for b, lab in zip(t["bpos"], t["blab"]):
            nxt = t["ids"][b + 1]
            n[nxt] = n.get(nxt, 0) + 1
            r[nxt] = r.get(nxt, 0) + (lab == REFL)
    ids = sorted(i for i in n if n[i] >= 100 and r[i] / n[i] >= 0.5)
    tot = sum(r.values())
    info = {"ids": ids, "tokens": [tok.decode([i]) for i in ids], "share_of_reflection_steps": sum(r[i] for i in ids) / tot,
            "p_reflection": [r[i] / n[i] for i in ids], "count": [n[i] for i in ids]}
    p.write_text(json.dumps(info, indent=1))
    log(f"R tokens {info['tokens']} cover {info['share_of_reflection_steps']:.2f} of reflection steps (fit traces)")
    return ids


def split(trs, part):
    return [t for t in trs if t["qi"] % 2 == (0 if part == "fit" else 1)]


def subsample(trs, n_q, seed=0):
    """first n_q questions of a fixed shuffle, both trace sources kept."""
    qs = sorted({t["qi"] for t in trs})
    keep = set(np.random.default_rng(seed).permutation(qs)[:n_q].tolist())
    return [t for t in trs if t["qi"] in keep]


@torch.no_grad()
def p_r(model, hidden, bpos, r_ids):
    """P_R at the boundaries from final hidden states [T, D]; also log-odds."""
    lg = model.lm_head(hidden[bpos]).float()
    lp = torch.log_softmax(lg, -1)
    lr = torch.logsumexp(lp[:, r_ids], -1)
    pr = lr.exp()
    return pr, lr - torch.log1p(-pr.clamp(max=1 - 1e-6))


class Replace:
    """Forward hook on one decoder layer: overwrite its output at `pos` (None = all positions) with `src` [T, D]."""

    def __init__(self, layer, src, pos=None):
        self.src, self.pos = src, pos
        self.hk = layer.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        hs = o[0] if isinstance(o, tuple) else o
        hs = hs.clone()
        if self.pos is None:
            hs[0] = self.src.to(hs.dtype)
        else:
            hs[0, self.pos] = self.src[self.pos].to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def remove(self):
        self.hk.remove()


class Add:
    """Forward hook: add vector v [D] to one decoder layer's output at every position."""

    def __init__(self, layer, v):
        self.v = v
        self.hk = layer.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        hs = o[0] if isinstance(o, tuple) else o
        hs = (hs.float() + self.v).to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs

    def remove(self):
        self.hk.remove()


class Record:
    """Forward hooks on every decoder layer: keep each layer's output for batch item 0."""

    def __init__(self, model):
        self.out = {}
        self.hks = [l.register_forward_hook(self._mk(i)) for i, l in enumerate(layers(model))]

    def _mk(self, i):
        def f(_m, _i, o):
            self.out[i] = (o[0] if isinstance(o, tuple) else o)[0].detach()
        return f

    def remove(self):
        for h in self.hks:
            h.remove()


def boot(x, b=10000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (b, len(x)))].mean(1)
    p = max(2 * min((m <= 0).mean(), (m >= 0).mean()), 1 / b)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), p


def boot_ratio(num, den, b=10000, seed=0):
    """ratio of means, bootstrapped over the paired units."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (b, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)
