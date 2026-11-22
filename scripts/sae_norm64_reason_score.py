"""Runs the repo's scripts/reason_score.py unchanged, with the cached-arrow dataset reader from train_sae_norm64.py
(the repo loader's offline cache lookup fails here). Same rows, order and text.

Run: python scripts/sae_norm64_reason_score.py --sae_checkpoint ... (reason_score.py args)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reason_score  # noqa: E402  (the repo's script)
from train_sae_norm64 import ArrowAMDeepSeek  # noqa: E402

reason_score.AMDeepSeekDataset = ArrowAMDeepSeek
if __name__ == "__main__":
    reason_score.main()
