"""Exp 4 (docs/mi_qwen_students.md): where did RD's extra training change the network, relative to LKD-continue?

Both start from LKD epoch 1 (anc) and see the same 12,032 rows. Also the weight helpers (state, group) of mi_qwen_alpha0.
Phases
  weights  per layer: |dW| of LKD-continue and of each RD run vs anc, cosine between the two updates, |dW_rd - dW_lkd|
  analyze  writes analysis.txt
Run: python scripts/mi_qwen_diffs.py weights|analyze
"""
import argparse
import json
import re
from pathlib import Path

from mi_qwen_students_common import MODELS, N_LAYERS, OUT, log

D4 = OUT / "diffs"


def state(name):
    from safetensors.torch import load_file
    p = Path(MODELS[name])
    sd = {}
    for f in sorted(p.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    return {re.sub(r"^model\.language_model\.", "model.", k): v for k, v in sd.items()}


def group(k):
    m = re.search(r"\blayers\.(\d+)\.", k)
    if m and not k.startswith("mtp"):
        return int(m.group(1))
    return "embed" if "embed" in k else ("final_norm" if k == "model.norm.weight" else "other")


def weights():
    anc = state("anc")
    res = {}
    for name in ("lkd", "rd_low", "rd_mod", "rd_high"):
        sd = state(name)
        assert set(sd) == set(anc), (name, sorted(set(sd) ^ set(anc))[:5])
        res[name] = sd
    rows = {}
    for k, w0 in anc.items():
        g = str(group(k))
        w0 = w0.float()
        dl = res["lkd"][k].float() - w0
        r = rows.setdefault(g, {"w0": 0.0, "lkd": 0.0, **{f"{n}": 0.0 for n in ("rd_low", "rd_mod", "rd_high")},
                                **{f"dot_{n}": 0.0 for n in ("rd_low", "rd_mod", "rd_high")},
                                **{f"diff_{n}": 0.0 for n in ("rd_low", "rd_mod", "rd_high")}})
        r["w0"] += float((w0 ** 2).sum()); r["lkd"] += float((dl ** 2).sum())
        for n in ("rd_low", "rd_mod", "rd_high"):
            dr = res[n][k].float() - w0
            r[n] += float((dr ** 2).sum()); r[f"dot_{n}"] += float((dr * dl).sum())
            r[f"diff_{n}"] += float(((dr - dl) ** 2).sum())
        dlo, dhi = res["rd_low"][k].float() - w0, res["rd_high"][k].float() - w0
        r["dot_lh"] = r.get("dot_lh", 0.0) + float((dlo * dhi).sum())
        r["diff_lh"] = r.get("diff_lh", 0.0) + float(((dhi - dlo) ** 2).sum())
    D4.mkdir(parents=True, exist_ok=True)
    (D4 / "weights.json").write_text(json.dumps(rows, indent=1))
    log("diffs weights: done")


def analyze():
    L = ["# Exp 4 · Where RD's training changed the network (RD_mod vs LKD-continue, both from LKD epoch 1)", ""]
    W = json.loads((D4 / "weights.json").read_text())
    L.append("## Weights: update size relative to |W_anc| (x1e-3), cosine(dW_rd, dW_lkd), |dW_rd - dW_lkd| / |dW_lkd|")
    L.append("  last two columns: rd_high vs rd_low (same RD pipeline, alpha 0.40 vs 0.05) = the part due to the RD term")
    L.append("  layer  type    lkd   rd_low  rd_mod rd_high | cos low  mod  high | diff/lkd low  mod  high | cos(lo,hi) |hi-lo|/|lkd|")
    cfg = json.loads((Path(MODELS["anc"]) / "config.json").read_text())
    types = cfg["layer_types"]
    keys = [str(i) for i in range(N_LAYERS)] + ["embed", "final_norm", "other"]
    for g in keys:
        if g not in W:
            continue
        r = W[g]
        s = lambda n: 1e3 * (r[n] / r["w0"]) ** 0.5  # noqa: E731
        c = lambda n: r[f"dot_{n}"] / (r[n] * r["lkd"]) ** 0.5 if r[n] * r["lkd"] > 0 else float("nan")  # noqa: E731
        dd = lambda n: (r[f"diff_{n}"] / r["lkd"]) ** 0.5 if r["lkd"] > 0 else float("nan")  # noqa: E731
        ty = types[int(g)][:6] if g.isdigit() else ""
        L.append(f"  {g:>10s} {ty:6s} {s('lkd'):6.2f} {s('rd_low'):6.2f} {s('rd_mod'):6.2f} {s('rd_high'):6.2f} |"
                 f" {c('rd_low'):5.3f} {c('rd_mod'):5.3f} {c('rd_high'):5.3f} | {dd('rd_low'):5.3f} {dd('rd_mod'):5.3f} {dd('rd_high'):5.3f} |"
                 f" {r['dot_lh'] / (r['rd_low'] * r['rd_high']) ** 0.5 if r['rd_low'] > 0 else float('nan'):6.4f}"
                 f" {(r['diff_lh'] / r['lkd']) ** 0.5 if r['lkd'] > 0 else float('nan'):6.4f}")
    out = "\n".join(L)
    print(out)
    (D4 / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["weights", "analyze"])
    a = ap.parse_args()
    {"weights": weights, "analyze": analyze}[a.phase]()
