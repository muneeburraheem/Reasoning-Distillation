"""Where the alignment term acts, against the matched alpha = 0 run (docs/mi_qwen_students.md, "Depth profile of the
alignment term").

Models   rd_zero (alpha 0), rd_low / rd_low_rr / rd_mod / rd_high (alpha 0.05 / 0.05 rerun / 0.15 / 0.40), lkd.
Phase
  analyze   M1: probe R^2 of the teacher's features at every student layer (existing mi_qwen_student_probe files)
Outputs  eval_outputs/mi_qwen_students/alpha_mech/
Run: python scripts/mi_qwen_alpha_mech.py analyze
"""
import argparse

import numpy as np
import torch

from mi_qwen_alpha0_props import PROBE_NAME
from mi_qwen_students_common import OUT

DM = OUT / "alpha_mech"


def _boot_diff(f, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([f(torch.from_numpy(rng.integers(0, n, n))) for _ in range(B)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)


def analyze():
    L = ["# Depth profile of the alignment term vs the matched alpha = 0 run (expectations in docs/mi_qwen_students.md)"]
    # M1
    import mi_qwen_student_probe as PB
    names = ("rd_zero", "rd_low", "rd_low_rr", "rd_mod", "rd_high", "lkd")
    R = {n: torch.load(PB.TENS / f"probe_{PROBE_NAME[n]}.pt")["ridge"] for n in names}
    _, nz = PB.r2(R["rd_zero"][:, 0])
    live = nz >= 20
    sets = {"rd": slice(0, PB.N_SET), "rand": slice(PB.N_SET, 2 * PB.N_SET)}

    def mr(st, sl):
        r, _ = PB.r2(st)
        return float(r[sl][live[sl]].mean())
    nt = R["rd_zero"].shape[0]
    L.append(f"\n## M1 · probe R^2 of the teacher's features by student layer hs[1..24] ({nt} test texts, bootstrap 1,000)")
    for part, sl in sets.items():
        L.append(f"\n  {part} set: R^2, then difference vs rd_zero [95% CI]")
        L.append("  layer  " + "  ".join(f"{n:>9s}" for n in names) + "   " +
                 "  ".join(f"{n + ' - zero':>30s}" for n in names if n != "rd_zero"))
        for l in range(R["rd_zero"].shape[1]):
            row = [mr(R[n][:, l], sl) for n in names]
            ds = []
            for n in names[1:]:
                A, B_ = R[n][:, l], R["rd_zero"][:, l]
                lo, hi, p = _boot_diff(lambda j: mr(A[j], sl) - mr(B_[j], sl), nt)
                ds.append(f"{mr(A, sl) - mr(B_, sl):+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
            L.append(f"  hs[{l + 1:2d}] " + "  ".join(f"{x:9.4f}" for x in row) + "   " + "  ".join(f"{d:>30s}" for d in ds))
    out = "\n".join(L)
    print(out)
    DM.mkdir(parents=True, exist_ok=True)
    (DM / "analysis.txt").write_text(out + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["analyze"])
    a = ap.parse_args()
    {"analyze": analyze}[a.phase]()
