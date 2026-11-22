import sys, torch
sys.path.insert(0, "scripts")
import sae_norm64_gate as gate
from sae_norm64_gate import *
@torch.no_grad()
def run():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir="cache")
    ds = ArrowAMDeepSeek()
    teacher = load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True, cache_dir="cache").eval()
    vocab = {w.lower() for w in REASONING_VOCABULARY}
    H, MK = [], []
    for i in range(0, 52_032, 1300):
        ids = tok(ds[i]["text"], truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        H.append(teacher(ids.cuda(), output_hidden_states=True).hidden_states[17][0].float())
        MK.append(torch.tensor([tok.decode([t]).strip().lower() in vocab for t in ids[0].tolist()]))
    h = torch.cat(H); mk = torch.cat(MK).cuda(); h = h * (64 / h.norm(dim=-1, keepdim=True))
    sp, rp = SAES["new (qwen_sae_l16_norm64)"]
    sae = load_sae(str(sp), "cuda", torch.float32)
    idx = torch.load(rp)["sorted_indices"][:512].long().cuda()
    Y = torch.cat([sae.encode(h[c:c+4096])[:, idx] for c in range(0, len(h), 4096)])
    dens = (Y > 0).float().mean(0)
    base = mk.float().mean()
    enr = (Y[mk].sum(0) / Y.sum(0).clamp(min=1e-9)) / base       # mass share on reasoning tokens / base rate
    # also "just before" a reasoning token (predictive features)
    pre = torch.zeros_like(mk); pre[:-1] = mk[1:]
    enr_pre = (Y[pre].sum(0) / Y.sum(0).clamp(min=1e-9)) / base
    for lo, hi in ((0, 50), (50, 196), (196, 512)):
        e, ep, d = enr[lo:hi], enr_pre[lo:hi], dens[lo:hi]
        print(f"rank {lo:3d}-{hi:3d}: enrichment on-token median {e.median():.1f}x (>2x: {(e>2).float().mean():.0%}), "
              f"just-before median {ep.median():.1f}x (>2x: {(ep>2).float().mean():.0%}), density median {d.median():.2e}, "
              f"dense >5%: {(d>0.05).sum().item()}")
    top = dens.topk(10)
    print("10 densest of top-512: density", [f"{x:.2f}" for x in top.values.tolist()], "enrichment", [f"{enr[i]:.1f}" for i in top.indices.tolist()])
    share = Y.sum(0); print(f"share of total target mass carried by the 20 densest: {share[top.indices].sum()/share.sum():.0%}" if False else "")
    m = Y.sum(0); o = m.argsort(descending=True)
    print(f"share of target mass from the 20 highest-mass features: {m[o[:20]].sum()/m.sum():.0%}; their median enrichment {enr[o[:20]].median():.1f}x")
run()
