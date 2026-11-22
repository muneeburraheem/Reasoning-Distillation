"""Health check for SAE checkpoints of the new-SAE experiment (and the old HF SAE as a baseline).

Held-out text: 40 AM-DeepSeek math rows from 531,000+ (outside SAE and RD training ranges), 2048 tokens,
Qwen3.5-4B bf16, hidden_states[17] (= output of layer 16, the SAE's site and October RD's teacher site).
Per SAE: fraction of variance explained (FVE), mean L0, dead-latent fraction (never nonzero),
and "reasoning latents": >= 20 nonzero tokens and > 50% of activation mass on reasoning-vocabulary
tokens (scripts/reason_score.py vocabulary). Llama's working SAE: FVE +0.52, L0 127.

Run: python scripts/sae_norm64_health.py <name>=<sae.pt>[:raw] ...
Appends to eval_outputs/rd_newsae/sae/eval.txt
"""
import sys
from pathlib import Path

import os

import torch

DEV = os.environ.get("EVAL_DEVICE", "cuda")   # EVAL_DEVICE=cpu while training holds the GPU
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from models.sae_loader import load_sae  # noqa: E402
from reason_score import REASONING_VOCABULARY  # noqa: E402
from train_sae_norm64 import ArrowAMDeepSeek  # noqa: E402

MATH = {"NuminaMath_1.5", "MATH_metamathQA", "MATH_numina", "openR1Math_default", "MATH-lighteval", "Omni-MATH"}
CACHE = ROOT / "checkpoints/qwen_sae_norm64/eval_hidden.pt"


@torch.no_grad()
def hidden(tok):
    if CACHE.exists():
        return torch.load(CACHE)
    ds = ArrowAMDeepSeek().data
    H, ids_all = [], []
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map={"": 0},
                                                 cache_dir=str(ROOT / "cache")).eval()
    for i in range(531_000, 700_000, 400):
        m = ds[i]["messages"]
        if m[0]["info"]["source"] not in MATH:
            continue
        u = next(x["content"] for x in m if x["role"] == "user"); a = next(x["content"] for x in m if x["role"] == "assistant")
        ids = tok(u + "\n" + a, truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        H.append(model(ids.cuda(), output_hidden_states=True).hidden_states[17][0].float().cpu())
        ids_all.append(ids[0])
        if len(H) == 40:
            break
    out = {"h": torch.cat(H), "ids": torch.cat(ids_all)}
    torch.save(out, CACHE)
    return out


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=str(ROOT / "cache"))
    d = hidden(tok)
    h, ids = d["h"].to(DEV), d["ids"]
    vocab = {w.lower() for w in REASONING_VOCABULARY}
    mk = torch.tensor([tok.decode([t]).strip().lower() in vocab for t in ids.tolist()]).to(DEV)
    lines = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        raw = path.endswith(":raw")
        path = path.removesuffix(":raw")
        sae = load_sae(path, DEV, torch.float32)
        x = h if raw else h * (64.0 / h.norm(dim=-1, keepdim=True))
        nz = torch.zeros(sae.encoder.weight.shape[0], device=DEV)
        mass = torch.zeros_like(nz); mass_mk = torch.zeros_like(nz)
        sse = sst = l0 = 0.0
        mu = x.mean(0)
        for c in range(0, len(x), 4096):
            xc = x[c:c + 4096]
            rec, lat = sae(xc)
            sse += float(((xc - rec) ** 2).sum()); sst += float(((xc - mu) ** 2).sum())
            l0 += float((lat > 0).sum())
            nz += (lat > 0).sum(0); mass += lat.sum(0); mass_mk += lat[mk[c:c + 4096]].sum(0)
        reasoning = int(((nz >= 20) & (mass_mk / mass.clamp(min=1e-9) > 0.5)).sum())
        line = (f"{name:28s} input {'raw ' if raw else 'n64 '} FVE {1 - sse / sst:+.3f}  L0 {l0 / len(x):7.1f}  "
                f"dead {(nz == 0).float().mean().item():.3f}  latents>=20 tokens {int((nz >= 20).sum()):6d}  "
                f"reasoning latents {reasoning:4d}  (tokens {len(x)}, reasoning-vocab tokens {int(mk.sum())})")
        print(line, flush=True)
        lines.append(line)
    with (ROOT / "eval_outputs/rd_newsae/sae/eval.txt").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
