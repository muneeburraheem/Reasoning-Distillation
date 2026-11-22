"""Local smoke test of the Nov 5 RD warm-start training with the new SAE (4090, before RunPod).

Runs scripts/train.py of commit a25867d unchanged, except the AM-DeepSeek loader is swapped for the cached
arrow shards (the repo loader fails offline here; same rows, same "user\\nassistant" text).
Config: configs/train/rd_newsae_smoke.yaml = rd_newsae_low with 128 samples, 512 tokens, accumulation 8.
Checks: student + head load from checkpoints/rd_newsae_student_init (no silent "starting from scratch"), alignment
loss is logged and finite, memory fits, a checkpoint is written. 8-bit AdamW here only (memory).

Run (from the repo root; needs the a25867d checkout, see below):
  python scripts/train_rd_newsae_smoke.py --config configs/train/rd_newsae_smoke.yaml
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# RD new-SAE training used the awesomeReasonDistill code at commit a25867d (docs/rd_newsae.md), not this branch's.
# Check it out next to this repo first:  git worktree add ../rd_train_a25867d a25867d   (or set RD_TRAIN_REPO)
REPO = Path(os.environ.get("RD_TRAIN_REPO", ROOT.parent / "rd_train_a25867d"))
if not (REPO / "scripts/train.py").exists():
    sys.exit(f"RD training code not found at {REPO}; run: git worktree add {REPO} a25867d")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import train as T  # noqa: E402  (a25867d's scripts/train.py)

SHARDS = (ROOT / "cache/a-m-team___am-deep_seek-r1-distilled-1.4_m/am_0.9M-6e8fbe69f20ec71c/0.0.0/"
          "53531c06634904118a2dcd83961918c4d69d1cdf")


class ArrowAMDeepSeek(torch.utils.data.Dataset):
    def __init__(self, split="train", cache_dir=None, max_samples=None, skip_samples=0):
        from datasets import Dataset, concatenate_datasets
        d = concatenate_datasets([Dataset.from_file(str(SHARDS / f"am-deep_seek-r1-distilled-1.4_m-train-{i:05d}-of-00038.arrow"))
                                  for i in range(38)])
        end = len(d) if max_samples is None else min(len(d), skip_samples + max_samples)
        self.data = d.select(range(skip_samples, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msg = {m["role"]: m["content"] for m in self.data[idx]["messages"]}
        return {"text": msg.get("user", "") + "\n" + msg.get("assistant", "")}


T.AMDeepSeekDataset = ArrowAMDeepSeek
# SMOKE ONLY: full AdamW state does not fit the 24 GB 4090 next to the 8-bit teacher (RunPod uses AdamW)
import bitsandbytes as bnb  # noqa: E402
T.torch.optim.AdamW = bnb.optim.AdamW8bit
if __name__ == "__main__":
    T.main()
