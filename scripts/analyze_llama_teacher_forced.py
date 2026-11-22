"""Analysis of scripts/eval_llama_teacher_forced_kl.py (Llama arm version of RUN_LOG 8f).

Per domain (heldout am_0.5M / insample am_0.9M), region (think / answer / boxed) and position
window (lt512 = inside the 512-token training context, ge512 = beyond it): pooled per-token means
per model, then RD - LKD paired per trace (per-trace means), bootstrap over traces, and the
number of traces where RD is higher.

Run: python scripts/analyze_llama_teacher_forced.py [results.json]
"""
import json
import random
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "eval_outputs/llama_teacher_forced/results.json"
MODELS = ("teacher", "lkd", "rd", "base")
STATS = ("kl1", "kl4", "ent", "nll")


def boot(d, B=10000):
    rng = random.Random(0)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B
    return sum(d) / len(d), m[int(.025 * B)], m[int(.975 * B)], min(1.0, max(p, 1 / B))


def main():
    res = json.load(open(P))["results"]
    print(f"{len(res)} traces: " + ", ".join(f"{d} {sum(r['domain'] == d for r in res)}" for d in ("heldout", "insample")))
    for dom in ("heldout", "insample"):
        R = [r for r in res if r["domain"] == dom]
        for key in ("think/lt512", "think/ge512", "answer/lt512", "answer/ge512", "boxed/lt512", "boxed/ge512"):
            n_pos = sum(r["stats"]["lkd"][key]["n"] for r in R)
            if n_pos == 0:
                continue
            print(f"\n## {dom} {key}  ({n_pos} positions, {sum(r['stats']['lkd'][key]['n'] > 0 for r in R)} traces)")
            print("  " + " ".join(f"{m:>9s}" for m in ("stat",) + MODELS))
            for s in STATS:
                vals = [sum(r["stats"][m][key][s] for r in R) / n_pos for m in MODELS]
                print("  " + f"{s:>9s} " + " ".join(f"{v:9.4f}" if not (m == "teacher" and s.startswith("kl")) else f"{'-':>9s}"
                                                       for m, v in zip(MODELS, vals)))
            T = [r for r in R if r["stats"]["lkd"][key]["n"] >= 5]
            if len(T) < 20:
                continue
            for s in STATS:
                d = [r["stats"]["rd"][key][s] / r["stats"]["rd"][key]["n"] - r["stats"]["lkd"][key][s] / r["stats"]["lkd"][key]["n"] for r in T]
                m, lo, hi, p = boot(d)
                print(f"  RD-LKD {s:5s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}  RD higher on {sum(x > 0 for x in d)}/{len(d)} traces")


if __name__ == "__main__":
    main()
