"""Audit unified_sweep: (1) were prompts encoded space-stripped? (2) fmax per feature, delta norm vs residual norm."""
import json, math, os, sys, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
BASE=os.environ.get("KD_BASE_DIR", "/home/cs/Desktop/KD/KnowledgeDistillation/base")
STU=os.environ.get("LLAMA_KD_DIR", "/home/cs/Desktop/KD/Reasoning-Distillation/KD_TRAIN_Deepseek_Dataset") + "/checkpoints/amdeepseek_reasondistill/epoch_2"
M="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
bad=AutoTokenizer.from_pretrained(M); good=AutoTokenizer.from_pretrained(STU)
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map={"":0}).eval()
R=[json.loads(l) for l in open(f"{BASE}/reasoning_neurons_validation/unified_sweep/results/raw_results.jsonl")]
Q=[r for r in R if r["sweep"]=="qualitative" and r["topk"]==1]
def gtext(ids): return good.decode(ids,skip_special_tokens=True)
# stored baseline text is decoded with the broken decoder (byte-level pieces); map to normal text
def fix(s): return good.convert_tokens_to_string(list(s)) if False else s.replace("Ġ"," ").replace("Ċ","\n")
print("== 1. prompt encoding")
for i in (6,9):
    p=Q[i]["prompt"]; stored=fix(Q[i]["baseline_text"])[:300]
    for name,tk in (("broken",bad),("correct",good)):
        ids=tk(p,return_tensors="pt")["input_ids"].to(0)
        o=model.generate(ids,max_new_tokens=60,do_sample=False,pad_token_id=128001)
        txt=gtext(o[0][ids.shape[1]:])
        print(f"  prompt {i} {name:7s} n_prompt={ids.shape[1]} match_stored={stored.startswith(txt[:150])} :: {txt[:110]!r}")
    print(f"  stored :: {stored[:110]!r}")
print("== 2. steering size")
class SAE(nn.Module):
    def __init__(s): super().__init__(); s.encoder=nn.Linear(4096,65536); s.decoder=nn.Linear(65536,4096)
sae=SAE().to(0); sae.load_state_dict(torch.load(f"{BASE}/checkpoints/l19_sae_l1_0.0009.pt",map_location="cuda:0")); sae.eval()
idx=torch.load(f"{BASE}/reasonscores_base_sae.pt")["sorted_indices"][:50].tolist()
W=sae.decoder.weight.float()[:,idx]
print("  decoder col norms (top50) min/median/max", W.norm(dim=0).min().item(), W.norm(dim=0).median().item(), W.norm(dim=0).max().item())
ds=load_dataset("HuggingFaceH4/MATH-500",split="test")
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
for name,tk in (("broken",bad),("correct",good)):
    acts=[]; norms=[]
    for ex in list(ds)[:N]:
        p=f"Problem: {ex['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        fmt=bad.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        ids=tk(fmt,return_tensors="pt",add_special_tokens=False)["input_ids"].to(0)
        col=[]; h=model.model.layers[19].register_forward_hook(lambda m,i,o: col.append((o[0] if isinstance(o,tuple) else o).detach().float()))
        with torch.no_grad(): model.generate(ids,max_new_tokens=512,do_sample=False,pad_token_id=128001)
        h.remove()
        hs=torch.cat(col,dim=1)[0]; norms.append(hs.norm(dim=-1)[1:].cpu())
        with torch.no_grad(): f=F.relu(sae.encoder(hs*(64/hs.norm(dim=-1,keepdim=True).clamp(min=1e-8))))[:,idx]
        acts.append(f.cpu())
    A=torch.cat(acts); nr=torch.cat(norms)
    fmax=torch.quantile(A,0.98,dim=0)
    print(f"  [{name} tokenizer, {N} prompts, {A.shape[0]} tokens] residual norm L19 median {nr.median():.1f} (p5 {nr.quantile(.05):.1f}, p95 {nr.quantile(.95):.1f}); SAE input norm 64")
    for k in (1,15,50):
        fk=fmax[:k]; d=(W[:,:k].cpu()*(2*fk)).sum(1)
        print(f"   top{k}: features fmax>0 {(fk>0).sum().item()}/{k}; firing density median {(A[:,:k]>0).float().mean(0).median():.4f}; |delta| {d.norm():.1f} = {d.norm()/nr.median():.2f}x median residual norm")
    print("   per-feature density top15:", [round(x,4) for x in (A[:,:15]>0).float().mean(0).tolist()])
