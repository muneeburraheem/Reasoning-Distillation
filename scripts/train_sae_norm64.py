"""New-SAE experiment (docs/rd_newsae.md): retrain the Qwen3.5-4B layer-16 SAE with input normalisation.

Runs the repo's own SAE code (scripts/train_sae.py, training/sae_trainer.py, models/sae.py) unchanged;
this wrapper patches three things at runtime only, so nothing in the main repo changes:
  1. Norm64Trainer: rescales every activation to norm 64 before the SAE, as the RD alignment loss
     (losses/alignment_loss.py) and scripts/reason_score.py already do. The old SAE was trained on raw
     activations (norm ~10-11) but read on norm-64 inputs.
  2. The LR schedule length is computed from the measured tokens per step. The repo estimates it as
     max_train_tokens / token_batch_size, but each step only carries one sequence batch (far fewer
     tokens), so the old run's LR reached 0 at step 3,817 and stayed there (step_4000 == step_8000).
  3. ArrowAMDeepSeek: reads the cached AM-DeepSeek arrow shards directly (same rows, same order,
     same "user\\nassistant" text as data_loaders/amdeepseek.py), because the repo loader's offline
     cache lookup fails on this machine. Text is built on the fly; nothing is written to disk.
  --raw (diagnostic pilot): skip fix 1 only.
  --weights-only (pilots): checkpoints keep sae.pt only (no optimizer state), to fit the disk.

Run:  python scripts/train_sae_norm64.py --config configs/train/sae_norm64_<full|pilot_*>.yaml [--weights-only]
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_sae  # noqa: E402  (the repo's launcher)
from training.sae_trainer import SAETrainer, LastFractionLinearDecay  # noqa: E402

TARGET_NORM = 64.0
SHARDS = ROOT / "cache/a-m-team___am-deep_seek-r1-distilled-1.4_m/am_0.9M-6e8fbe69f20ec71c/0.0.0/53531c06634904118a2dcd83961918c4d69d1cdf"
WEIGHTS_ONLY = "--weights-only" in sys.argv
if WEIGHTS_ONLY:
    sys.argv.remove("--weights-only")
RAW = "--raw" in sys.argv          # diagnostic pilot only: skip the norm-64 rescale (fixes 2-3 still apply)
if RAW:
    sys.argv.remove("--raw")


class ArrowAMDeepSeek(torch.utils.data.Dataset):
    def __init__(self, split="train", cache_dir=None, max_samples=None, skip_samples=0):
        from datasets import Dataset, concatenate_datasets
        self.data = concatenate_datasets([Dataset.from_file(str(SHARDS / f"am-deep_seek-r1-distilled-1.4_m-train-{i:05d}-of-00038.arrow"))
                                          for i in range(38)])
        end = len(self.data) if max_samples is None else min(len(self.data), skip_samples + max_samples)
        self.data = self.data.select(range(skip_samples, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = asst = ""
        for m in self.data[idx]["messages"]:
            if m["role"] == "user":
                user = m["content"]
            elif m["role"] == "assistant":
                asst = m["content"]
        return {"text": user + "\n" + asst}


class Norm64Trainer(SAETrainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # tokens per step = non-pad tokens of one sequence batch; measure on 200 batches' worth of rows
        n = min(len(self.dataset), 200 * self.config.sequence_batch_size)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(self.dataset), generator=g)[:n].tolist()
        lens = [min(self.config.max_length, len(self.tokenizer(self.dataset[i]["text"])["input_ids"])) for i in idx]
        per_step = min(self.config.token_batch_size, self.config.sequence_batch_size * sum(lens) / len(lens))
        steps = math.ceil(self.config.max_train_tokens / per_step)
        self.scheduler = LastFractionLinearDecay(self.optimizer, total_steps=steps, base_lr=self.config.lr,
                                                 min_lr=self.config.min_lr, decay_fraction=self.config.lr_decay_fraction)
        print(f"[norm64] measured {per_step:.0f} tokens/step -> schedule {steps} steps "
              f"(repo estimate was {math.ceil(self.config.max_train_tokens / self.config.token_batch_size)})")

    def _layer_activations(self, batch):
        a = super()._layer_activations(batch)
        if RAW:
            return a
        return a * (TARGET_NORM / a.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    def _save_checkpoint(self, name):
        if not WEIGHTS_ONLY:
            return super()._save_checkpoint(name)
        d = Path(self.config.checkpoint_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae.state_dict(), d / "sae.pt")
        print(f"SAE weights saved: {d}")


train_sae.AMDeepSeekDataset = ArrowAMDeepSeek
train_sae.SAETrainer = Norm64Trainer

if __name__ == "__main__":
    train_sae.main()
