import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import mi_steering as ms
from data_loaders.amdeepseek import AMDeepSeekDataset  # noqa
from datasets import load_dataset
S, tok, model = ms.load("qwen_teacher")
we, be, wd = ms.load_sae(S)
we, be = we.cuda().float(), be.cuda().float()
dn = wd.float().norm(dim=0).cuda()
rank = torch.load(str(S["ranking"]), map_location="cpu")
idx = (rank["sorted_indices"] if isinstance(rank, dict) and "sorted_indices" in rank else rank)[:512].long().cuda()
import glob
from datasets import Dataset, concatenate_datasets
D="cache/a-m-team___am-deep_seek-r1-distilled-1.4_m/am_0.9M-6e8fbe69f20ec71c/0.0.0/53531c06634904118a2dcd83961918c4d69d1cdf/"
ds = concatenate_datasets([Dataset.from_file(D + f"am-deep_seek-r1-distilled-1.4_m-train-{i:05d}-of-00038.arrow") for i in range(38)])
rows = []
for i in range(531_000, 700_000, 400):
    m = ds[i]["messages"]
    if m[0]["info"]["source"] in {"NuminaMath_1.5","MATH_metamathQA","MATH_numina","openR1Math_default","MATH-lighteval","Omni-MATH"}:
        u = next(x["content"] for x in m if x["role"]=="user"); a = next(x["content"] for x in m if x["role"]=="assistant")
        rows.append(u + "\n" + a)
    if len(rows) == 40: break
tot = 0; stats = {16: [0, torch.zeros(512).cuda(), 0.0, torch.zeros(40960).cuda()], 17: [0, torch.zeros(512).cuda(), 0.0, torch.zeros(40960).cuda()]}
with torch.no_grad():
    for t in rows:
        ids = tok(t, truncation=True, max_length=2048, return_tensors="pt")["input_ids"].cuda()
        hs = model(ids, output_hidden_states=True).hidden_states
        for L in (16, 17):
            h = hs[L][0].float(); h = h * (64 / h.norm(dim=-1, keepdim=True))
            a = F.relu(h @ we.T + be)
            y = a[:, idx] * dn[idx]
            st = stats[L]; st[0] += (y > 0).sum().item(); st[1] += (y > 0).sum(0); st[2] += y.max().item() if y.numel() else 0; st[3] += (a > 0).sum(0)
        tot += ids.shape[1]
for L, (nz, per, mx, allf) in stats.items():
    print(f"hs[{L}] tokens {tot} RD-set density {nz/(tot*512):.2e}  features >=20 nonzero {(per>=20).sum().item()}/512  any {(per>0).sum().item()}  L0 all-latents {allf.sum().item()/tot:.2f}")
