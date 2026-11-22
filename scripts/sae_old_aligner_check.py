"""Does October RD's logged alignment loss match all-zero targets?
Rows from its own training range (40,000-52,031), 2048 tokens. Loss = the trainer's ReasoningAlignmentLoss."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from datasets import Dataset, concatenate_datasets
from custom_eval.config import ModelSpec
from custom_eval.modeling import load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import mi_steering as ms
D = "cache/a-m-team___am-deep_seek-r1-distilled-1.4_m/am_0.9M-6e8fbe69f20ec71c/0.0.0/53531c06634904118a2dcd83961918c4d69d1cdf/"
ds = concatenate_datasets([Dataset.from_file(D + f"am-deep_seek-r1-distilled-1.4_m-train-{i:05d}-of-00038.arrow") for i in range(38)])
rows = []
for i in range(40_000, 52_032, 600):
    m = ds[i]["messages"]
    u = next(x["content"] for x in m if x["role"] == "user"); a = next(x["content"] for x in m if x["role"] == "assistant")
    rows.append(u + "\n" + a)
REPO = "Khaquan/qwen-khaquanS-distillations"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir="cache")
ids = [tok(t, truncation=True, max_length=2048, return_tensors="pt")["input_ids"].cuda() for t in rows]
S = ms.SUBJECTS["qwen_teacher"]
we, be, wd = ms.load_sae(S); we, be = we.cuda().float(), be.cuda().float(); dn = wd.float().norm(dim=0).cuda()
rk = torch.load(str(S["ranking"]), map_location="cpu")
idx = (rk["sorted_indices"] if isinstance(rk, dict) else rk)[:512].long().cuda()
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map={"": 0}, cache_dir="cache").eval()
T = {16: [], 17: []}
with torch.no_grad():
    for x in ids:
        hs = teacher(x, output_hidden_states=True).hidden_states
        for L in T:
            h = hs[L][0].float(); h = h * (64 / h.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            T[L].append((F.relu(h @ we.T + be) * dn)[:, idx])
del teacher; torch.cuda.empty_cache()

def loss(z, t):
    s = F.softplus(z); v = ((s - t) ** 2).mean(); p = F.binary_cross_entropy_with_logits(z, (t > 0).float())
    r = (((z - z.mean(-1, keepdim=True)) - (t - t.mean(-1, keepdim=True))) ** 2).mean()
    return float(v + 0.30 * p + 0.15 * r)

for name, sub in (("oct_rd", "qwen-reasondistill-october/qwen_reasondistill_final_continue/epoch_1"),):
    model, _ = load_model_and_tokenizer(ModelSpec(name=name, checkpoint=REPO, subfolder=sub, dtype="bfloat16",
                                                  device_map="auto", strip_language_model_prefix=True), cache_dir="cache")
    from huggingface_hub.constants import HF_HUB_CACHE
    sd = torch.load(f"{HF_HUB_CACHE}/models--Khaquan--qwen-khaquanS-distillations/snapshots/e3a06f557af2163c6af19a730242bfcdc8b98783/{sub}/aligner.pt", map_location="cpu")
    ln = torch.nn.LayerNorm(2048).cuda(); ln.weight.data, ln.bias.data = sd["norm.weight"].cuda().float(), sd["norm.bias"].cuda().float()
    W, b = sd["proj.weight"].cuda().float(), sd["proj.bias"].cuda().float()
    with torch.no_grad():
        H = [model(x, output_hidden_states=True).hidden_states for x in ids]
        for SL in (12, 13):
            Z = [ln(h[SL][0].float()) @ W.T + b for h in H]
            z = torch.cat(Z)
            pa = torch.sigmoid(z).mean().item()
            for TL in (16, 17):
                t = torch.cat(T[TL])
                print(f"student hs[{SL}] teacher hs[{TL}]  P(active) pred {pa:.3f}  target active {(t>0).float().mean().item():.2e}  "
                      f"loss vs real targets {loss(z, t):.4f}  vs all-zero {loss(z, torch.zeros_like(t)):.4f}  tokens {z.shape[0]}")
print("logged final alignment loss 0.0717 (last 704 steps), first 0.1657")
