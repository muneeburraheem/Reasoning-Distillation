"""Go/no-go before any RD retraining: are the new SAE's top-512 ReasonScore features live on RD's own
training text, read exactly as the RD trainer reads them?

Path = losses/alignment_loss.py ReasoningAlignmentLoss.encode_teacher_features: teacher (8-bit, as in
training) hidden_states[17] (teacher_align_layer 17 = output of layer 16, the SAE's site), rescale to norm
64, sae.encode, x decoder-column norm, top-512 of reasonscore.pt. Text: 40 rows spread over RD's training
range (0-52,031), "user\\nassistant", 2048 tokens.
Reports target density, features nonzero on >= 20 tokens, share of target mass on reasoning-vocabulary
tokens, for the new SAE and the old HF SAE. Reference: Llama SAE 3.7e-3 density, 191/196 live.

Run: python scripts/sae_norm64_gate.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from transformers import AutoTokenizer  # noqa: E402
from models.sae_loader import load_sae  # noqa: E402
from models.student_loader import load_teacher  # noqa: E402
from reason_score import REASONING_VOCABULARY  # noqa: E402
from train_sae_norm64 import ArrowAMDeepSeek  # noqa: E402

# Hugging Face hub cache (HF_HUB_CACHE / HF_HOME aware; default ~/.cache/huggingface/hub)
from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402
OLD = Path(HF_HUB_CACHE, "models--Khaquan--qwen-khaquanS-distillations/snapshots/"
           "1a0cc88b882e2001410b28cafefd54816e6ed66e/qwen-3.5-4B-L16_16x-SAE")
SAES = {"new (qwen_sae_l16_norm64)": (ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/final/sae.pt", ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/reasonscore/reasonscore.pt"),
        "old HF SAE": (OLD / "sae.pt", OLD / "reasonscore.pt")}


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=str(ROOT / "cache"))
    ds = ArrowAMDeepSeek()
    teacher = load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True, cache_dir=str(ROOT / "cache")).eval()
    vocab = {w.lower() for w in REASONING_VOCABULARY}
    H, MK = [], []
    for i in range(0, 52_032, 1300):
        ids = tok(ds[i]["text"], truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        H.append(teacher(ids.cuda(), output_hidden_states=True).hidden_states[17][0].float())
        MK.append(torch.tensor([tok.decode([t]).strip().lower() in vocab for t in ids[0].tolist()]))
    del teacher; torch.cuda.empty_cache()
    h = torch.cat(H); mk = torch.cat(MK).cuda()
    h = h * (64.0 / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    lines = [f"# gate: {len(h)} tokens of RD training text, teacher 8-bit hidden_states[17], top-512 ReasonScore features"]
    for name, (sp, rp) in SAES.items():
        sae = load_sae(str(sp), "cuda", torch.float32)
        idx = torch.load(rp, map_location="cpu")["sorted_indices"][:512].long().cuda()
        dn = sae.decoder.weight.float().norm(dim=0)
        Y = torch.cat([(sae.encode(h[c:c + 4096]) * dn)[:, idx] for c in range(0, len(h), 4096)])
        nz = (Y > 0).sum(0)
        mass = Y.sum(0)
        share = float(Y[mk].sum() / mass.sum().clamp(min=1e-9))
        lines.append(f"  {name:26s} density {float((Y > 0).float().mean()):.2e}  live (>=20 tokens) {int((nz >= 20).sum())}/512  "
                     f"ever nonzero {int((nz > 0).sum())}/512  mean nonzero value {float(Y[Y > 0].mean()) if (Y > 0).any() else 0:.3f}  "
                     f"mass on reasoning-vocab tokens {share:.2f} (those tokens are {float(mk.float().mean()):.3f} of text)")
    lines.append("  reference Llama SAE: density 3.7e-3, 191/196 live")
    out = "\n".join(lines)
    print(out)
    (ROOT / "eval_outputs/rd_newsae/sae/gate.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
