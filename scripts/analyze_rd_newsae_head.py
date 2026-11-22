"""Does the pre-trained RD head carry signal, or did it settle on "everything off" (October's failure)?

Held-out rows 39,000-39,999 (every 25th), the same as train_rd_newsae_head.py. For the fitted head
(checkpoints/rd_newsae_student_init/aligner.pt), a fresh head, and the best constant head (bias = per-feature
logit of the train mean, so it ignores the input), reports: total RD loss and its parts (value MSE,
presence BCE, rank), R^2 of softplus(head) vs target per feature (mean over live features), and AUC of the
head's logit for "feature on" (pooled). Reference: ridge R^2 of these features from the LKD student is
about 0.42 (check 1).

Run: python scripts/analyze_rd_newsae_head.py   (needs the a25867d checkout, see train_rd_newsae_head.py)
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_rd_newsae_head as P  # noqa: E402


@torch.no_grad()
def main():
    tok = P.AutoTokenizer.from_pretrained(str(P.STUDENT))
    ds = P.ArrowAMDeepSeek()
    teacher = P.load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True,
                             cache_dir=str(P.ROOT / "cache")).eval()
    student = P.AutoModelForCausalLM.from_pretrained(str(P.STUDENT), dtype=torch.bfloat16, device_map={"": 0}).eval()
    sae = P.load_sae(str(P.B / "sae.pt"), "cuda", torch.bfloat16)
    neurons = P.load_reasoning_neurons(str(P.B / "reasonscore.pt"), P.K, "cuda")
    dn = sae.decoder.weight.float().norm(dim=0)
    loss_fn = P.ReasoningAlignmentLoss(sae, neurons, dn, 64.0, active_threshold=0.0, presence_weight=0.30, rank_weight=0.15)
    S, Y = [], []
    for i in P.EVAL_ROWS:
        enc = tok(ds[i]["text"], truncation=True, max_length=P.MAXLEN, return_tensors="pt").to("cuda")
        th = teacher(**enc, output_hidden_states=True).hidden_states[P.T_LAYER + 1]
        S.append(student(**enc, output_hidden_states=True).hidden_states[P.S_LAYER + 1][0].float())
        Y.append(loss_fn.encode_teacher_features(th.float())[0])
    S, Y = torch.cat(S), torch.cat(Y)
    d = S.shape[1]
    fitted = P.ReasoningFeatureHead(d, P.K).cuda()
    fitted.load_state_dict(torch.load(P.STUDENT / "aligner.pt"))
    fresh = P.ReasoningFeatureHead(d, P.K).cuda()
    const = P.ReasoningFeatureHead(d, P.K).cuda()
    const.proj.weight.zero_()
    m = Y.mean(0).clamp(min=1e-4)
    const.proj.bias.copy_(torch.log(torch.expm1(m)))         # softplus(bias) = mean target
    live = (Y > 0).sum(0) >= 20
    on = (Y > 0).flatten()
    print(f"held-out tokens {len(Y)}, target density {float((Y > 0).float().mean()):.3e}, live features {int(live.sum())}/{P.K}")
    for name, h in (("fitted", fitted), ("fresh", fresh), ("constant", const)):
        z = h(S)
        p = F.softplus(z)
        val = float(((p - Y) ** 2).mean())
        pres = float(F.binary_cross_entropy_with_logits(z, (Y > 0).float()))
        cz, ct = z - z.mean(-1, keepdim=True), Y - Y.mean(-1, keepdim=True)
        rank = float(((cz - ct) ** 2).mean())
        r2 = 1 - ((p - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0).clamp(min=1e-9)
        s = z.flatten()
        pos, neg = s[on], s[~on]
        idx = torch.randperm(len(neg), device=neg.device)[:200_000]
        auc = float((pos[torch.randint(len(pos), (200_000,), device=pos.device)] > neg[idx]).float().mean())
        print(f"  {name:8s} total {val + 0.30 * pres + 0.15 * rank:.4f} = value {val:.4f} + 0.30*presence {pres:.4f} + 0.15*rank {rank:.4f} | "
              f"R2 (live mean) {float(r2[live].mean()):+.3f} median {float(r2[live].median()):+.3f} | AUC(on) {auc:.3f} | "
              f"mean pred {float(p.mean()):.3f} vs target {float(Y.mean()):.3f}")


if __name__ == "__main__":
    main()
