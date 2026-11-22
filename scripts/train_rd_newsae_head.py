"""Pre-train RD's feature head on the frozen phase-1 LKD student, before phase 2 (RunPod).

Why: phase 2 is ~94 optimizer steps (12,032 samples / accumulation 128) and starts the head from a
random init. A base/LKD student's hs[13] already linearly carries most of the target (check 1: ridge R^2
0.42 of a 0.47 ceiling), so a fresh head spends those steps fitting itself; little gradient reaches
the student. Starting from a fitted head, the remaining error has to be closed by the student.

Exact loss of the Nov 5 code (commit a25867d, losses/alignment_loss.py, ReasoningAlignmentLoss), exact
sites (teacher hidden_states[16 + 1], student hidden_states[12 + 1]), teacher 8-bit, presence 0.30,
rank 0.15, top-196 of the new SAE's ReasonScore. Only the head trains (Adam 1e-3, 1 pass).
Rows 30,000-39,999 (phase-1 range, seen by the LKD student; phase-2 rows 40,000-52,031 untouched),
2048 tokens. Held-out: 40 rows from 39,000+. Reports held-out loss for fresh vs fitted head, and the
same loss with all-zero targets (if they match, the targets carry no signal: the October failure).
Writes checkpoints/rd_newsae_student_init/aligner.pt (loaded by a25867d's scripts/train.py from student_checkpoint).

Run: python scripts/train_rd_newsae_head.py
"""
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# RD new-SAE training used the awesomeReasonDistill code at commit a25867d (docs/rd_newsae.md), not this branch's.
# Check it out next to this repo first:  git worktree add ../rd_train_a25867d a25867d   (or set RD_TRAIN_REPO)
REPO = Path(os.environ.get("RD_TRAIN_REPO", ROOT.parent / "rd_train_a25867d"))
if not (REPO / "losses/alignment_loss.py").exists():
    sys.exit(f"RD training code not found at {REPO}; run: git worktree add {REPO} a25867d")
sys.path.insert(0, str(REPO))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from losses.alignment_loss import ReasoningAlignmentLoss  # noqa: E402
from models.aligner import ReasoningFeatureHead  # noqa: E402
from models.sae_loader import load_reasoning_neurons, load_sae  # noqa: E402
from models.student_loader import load_teacher  # noqa: E402

SHARDS = (ROOT / "cache/a-m-team___am-deep_seek-r1-distilled-1.4_m/am_0.9M-6e8fbe69f20ec71c/0.0.0/"
          "53531c06634904118a2dcd83961918c4d69d1cdf")


class ArrowAMDeepSeek:
    """The cached AM-DeepSeek arrow shards, "user\\nassistant" (as train_sae_norm64.py; standalone to keep the a25867d imports clean)."""

    def __init__(self):
        from datasets import Dataset, concatenate_datasets
        self.data = concatenate_datasets([Dataset.from_file(str(SHARDS / f"am-deep_seek-r1-distilled-1.4_m-train-{i:05d}-of-00038.arrow"))
                                          for i in range(38)])

    def __getitem__(self, idx):
        msg = {m["role"]: m["content"] for m in self.data[idx]["messages"]}
        return {"text": msg.get("user", "") + "\n" + msg.get("assistant", "")}

B = ROOT / "checkpoints/rd_newsae_sae"   # sae.pt, reasonscore.pt
STUDENT = ROOT / "checkpoints/rd_newsae_student_init"
T_LAYER, S_LAYER, K, MAXLEN = 16, 12, 196, 2048
TRAIN_ROWS = range(30_000, 39_000, 9)          # 1,000 rows
EVAL_ROWS = range(39_000, 40_000, 25)          # 40 rows


def log(msg):
    print(msg, flush=True)
    with (ROOT / "logs/rd_newsae/pretrain_head.log").open("a") as f:
        f.write(msg + "\n")


def main():
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(str(STUDENT))
    ds = ArrowAMDeepSeek()
    teacher = load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16, quantize_8bit=True,
                           cache_dir=str(ROOT / "cache")).eval()
    student = AutoModelForCausalLM.from_pretrained(str(STUDENT), dtype=torch.bfloat16, device_map={"": 0}).eval()
    sae = load_sae(str(B / "sae.pt"), "cuda", torch.bfloat16)
    neurons = load_reasoning_neurons(str(B / "reasonscore.pt"), K, "cuda")
    with torch.no_grad():
        dn = sae.decoder.weight.float().norm(dim=0)
    head = ReasoningFeatureHead(student.config.get_text_config().hidden_size, K).cuda()
    loss_fn = ReasoningAlignmentLoss(sae, neurons, dn, 64.0, active_threshold=0.0, presence_weight=0.30, rank_weight=0.15)
    loss_fn.aligner = head
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    @torch.no_grad()
    def hidden(i):
        enc = tok(ds[i]["text"], truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
        th = teacher(**enc, output_hidden_states=True).hidden_states[T_LAYER + 1]
        sh = student(**enc, output_hidden_states=True).hidden_states[S_LAYER + 1]
        return sh, th, enc["attention_mask"].float()

    ev = [hidden(i) for i in EVAL_ROWS]

    @torch.no_grad()
    def evaluate(tag):
        real = zero = 0.0
        dens = 0.0
        for sh, th, m in ev:
            real += float(loss_fn(sh, th, m))
            t = loss_fn.encode_teacher_features(th.float())
            dens += float((t > 0).float().mean())
        # same loss with every target set to 0 (swap the teacher encode)
        orig = loss_fn.encode_teacher_features
        loss_fn.encode_teacher_features = lambda h: torch.zeros(*h.shape[:2], K, device=h.device)
        for sh, th, m in ev:
            zero += float(loss_fn(sh, th, m))
        loss_fn.encode_teacher_features = orig
        n = len(ev)
        log(f"  [{tag}] held-out loss real targets {real / n:.4f} | all-zero targets {zero / n:.4f} | "
            f"target density {dens / n:.3e}")
        return real / n

    log(f"head pre-training: {len(TRAIN_ROWS)} train rows, {len(EVAL_ROWS)} held-out rows, K={K}")
    evaluate("fresh head")
    t0 = time.time()
    run = 0.0
    for n, i in enumerate(TRAIN_ROWS):
        sh, th, m = hidden(i)
        loss = loss_fn(sh, th, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run = 0.98 * run + 0.02 * float(loss) if n else float(loss)
        if n % 100 == 0:
            log(f"  row {n}/{len(TRAIN_ROWS)} running loss {run:.4f} ({time.time() - t0:.0f}s)")
            if n:
                evaluate(f"after {n} rows")
    final = evaluate("fitted head")
    torch.save(head.state_dict(), STUDENT / "aligner.pt")
    log(f"saved {STUDENT / 'aligner.pt'} (keys {list(head.state_dict())}); final held-out loss {final:.4f}")


if __name__ == "__main__":
    main()
