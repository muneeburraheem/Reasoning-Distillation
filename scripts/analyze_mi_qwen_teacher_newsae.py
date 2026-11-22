"""Analysis of mi_qwen_teacher_newsae.py: the repo's analyze_mi_ablation (patch + sfull) pointed at this folder,
plus reflection in the sfull generations (reasoning-marker words per 1k words of thinking, paired per
question vs base and vs the matched random set).

Run: python scripts/analyze_mi_qwen_teacher_newsae.py > eval_outputs/rd_newsae/teacher_mi/analysis.txt
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "qwen_teacher_l16n64"]
import analyze_mi_ablation as ama  # noqa: E402
import analyze_mi_steering as am  # noqa: E402

ama.D = HERE.parent / "eval_outputs/rd_newsae/teacher_mi/ablation"
MK = re.compile(r"\b(wait|hmm|maybe|perhaps|but|another|therefore|however|alternatively|double[- ]check|let me check)\b", re.I)


def reflection():
    p = ama.D / "sfull.jsonl"
    if not p.exists():
        return
    F = {json.loads(l)["cond"]: json.loads(l) for l in p.open()}
    per = {}
    for c, r in F.items():
        q = {}
        for x in r["records"]:
            think = x["text"].split("</think>")[0]
            w = max(len(think.split()), 1)
            q.setdefault(x["id"], []).append(1000 * len(MK.findall(think)) / w)
        per[c] = {i: float(np.mean(v)) for i, v in q.items()}
    print("\n# reflection: reasoning-marker words per 1k words of thinking")
    for c in per:
        print(f"  {c:24s} {np.mean(list(per[c].values())):.2f}")
    base = next(c for c in per if c.startswith("base|"))
    for a, b in [(c, base) for c in per if c != base] + [("rd|k0", "rand|k0|s0"), ("rd|k50", "rand|k50|s0"),
                                                          ("attr|k50", "rand|k50|s0")]:
        if a in per and b in per:
            ids = sorted(set(per[a]) & set(per[b]))
            print(f"  {a:12s} - {b:18s} {am.fmt(am.boot([per[a][i] - per[b][i] for i in ids]))}")


if __name__ == "__main__":
    ama.patch()
    ama.sfull()
    reflection()
