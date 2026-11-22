"""Checks 2 + 3: does the new SAE's ReasonScore set matter to the Qwen teacher?

Runs the repo's own scripts/mi_steering.py (calib only) and scripts/mi_ablation.py unchanged, on a
subject added here at run time (no repo file edited):
  qwen_teacher_l16n64   Qwen3.5-4B bf16, output of layers[16] (= hidden_states[17], the SAE's site),
                        new SAE checkpoints/qwen_sae_norm64/full_l1_003/final/sae.pt, its ReasonScore ranking, top 196
Feature sets (mi_ablation): rd = first k ACTIVE ranked features (k 15 / 50 / all of the 196),
rand = density-matched random SAE features (5 seeds), attr = positive control (top attribution on
split A, compared on split B), randdir = same-size random direction.
  check 3  attribution (patch phase): grad x activation of log P(reasoning-word class), rd vs rand
  check 2  patch (teacher-forced): ablate the set at marker positions -> d log p(marker), d P(class)
           sfull (generation, 200 MATH-500 x 2, 4096 tokens): accuracy, reflection
The unsteered base traces are the existing eval_outputs/mi_steering/qwen_teacher base run (no SAE
involved; same model, prompts, seed), copied into this folder.

Run: python scripts/mi_qwen_teacher_newsae.py steer --phases calib
     python scripts/mi_qwen_teacher_newsae.py ablate --phases patch sfull
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import mi_steering as ms  # noqa: E402
import mi_ablation as ma  # noqa: E402

SUBJ = "qwen_teacher_l16n64"
ms.SUBJECTS[SUBJ] = {**ms.SUBJECTS["qwen_teacher"], "layer": 16,
                     "sae": ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/final/sae.pt",
                     "ranking": ROOT / "checkpoints/qwen_sae_norm64/full_l1_003/reasonscore/reasonscore.pt", "n_ranked": 196}
OUT = ROOT / "eval_outputs/rd_newsae/teacher_mi"
STEER, ABL = OUT / "steering", OUT / "ablation"


def main():
    which, rest = sys.argv[1], sys.argv[2:]
    STEER.mkdir(parents=True, exist_ok=True)
    if not (STEER / "full.jsonl").exists():
        src = ROOT / "eval_outputs/mi_steering/qwen_teacher/full.jsonl"
        base = next(l for l in src.open() if json.loads(l)["cond"].startswith("base|"))
        (STEER / "full.jsonl").write_text(base)
    if which == "steer":
        sys.argv = ["mi_steering", "--subject", SUBJ, "--out_dir", str(STEER)] + rest
        ms.main()
    else:
        sys.argv = ["mi_ablation", "--subject", SUBJ, "--out_dir", str(ABL), "--steer_dir", str(STEER)] + rest
        ma.main()


if __name__ == "__main__":
    main()
