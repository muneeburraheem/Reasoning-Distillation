"""Llama pass@8 outputs (RUN_LOG 8k) re-read with the Qwen outcome split (8d) and by MATH level.

Per sample, from the stored end state: correct (strict / lenient), committed wrong (an answer was
extracted and is wrong), stopped in think (end-of-sequence before </think>), closed without an
answer, truncated (cap or loop). RD - LKD per-sample rates paired per question, bootstrap over
questions. MATH-500 additionally by level (levels from HuggingFaceH4/MATH-500, index = id), with
pass@1, pass@8 and all-8-correct, the reliability measure of 8m.

Run: python scripts/analyze_llama_outcomes.py
"""
import json
import random
import sys
from math import comb
from pathlib import Path

D = Path("eval_outputs/llama_passk")
M = ("logit_kd_epoch2", "reasondistill_epoch2")


def boot(d, B=5000):
    rng = random.Random(0)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B
    return sum(d) / len(d), m[int(.025 * B)], m[int(.975 * B)], min(1.0, max(p, 1 / B))


def sign(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def outcomes(c):
    committed = bool(c["answer_lenient"])
    return {"strict": c["correct_strict"], "lenient": c["correct_lenient"],
            "committed_wrong": committed and not c["correct_lenient"],
            "stopped_in_think": c["state"] == "stopped_in_think",
            "closed_no_answer": c["state"] == "closed_no_answer",
            "truncated": c["hit_limit"] or c["loop"]}


def table(recs, qids, label):
    print(f"\n## {label}: {len(qids)} questions x 8 samples, LKD / RD, RD-LKD (per-sample rate, paired per question)")
    for k in ("strict", "lenient", "committed_wrong", "stopped_in_think", "closed_no_answer", "truncated"):
        per = {m: {q: sum(outcomes(c)[k] for c in recs[m][q]) / len(recs[m][q]) for q in qids} for m in M}
        mm, lo, hi, p = boot([per[M[1]][q] - per[M[0]][q] for q in qids])
        a, b = (sum(per[m][q] for q in qids) / len(qids) for m in M)
        print(f"  {k:17s} {a:.3f} / {b:.3f}  {100*mm:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] p={p:.4f}")
    n_ok = {m: {q: sum(c["correct_strict"] for c in recs[m][q]) for q in qids} for m in M}
    p8 = [sum(n_ok[m][q] > 0 for q in qids) for m in M]
    a8 = [sum(n_ok[m][q] == 8 for q in qids) for m in M]
    lo_ = sum(n_ok[M[0]][q] > 0 and n_ok[M[1]][q] == 0 for q in qids)
    ro_ = sum(n_ok[M[1]][q] > 0 and n_ok[M[0]][q] == 0 for q in qids)
    print(f"  pass@8 (strict)   {p8[0]/len(qids):.3f} / {p8[1]/len(qids):.3f}  LKD-only {lo_} RD-only {ro_} sign p={sign(lo_, ro_):.3f}"
          f" | all 8 correct {a8[0]} / {a8[1]} | >=4 of 8 {sum(n_ok[M[0]][q] >= 4 for q in qids)} / {sum(n_ok[M[1]][q] >= 4 for q in qids)}")


def main():
    for bench in ("gsm8k", "math500", "aime25"):
        recs = {m: {r["id"]: r["candidates"] for r in json.load(open(D / f"{m}__{bench}.json"))["records"]} for m in M}
        qids = sorted(set(recs[M[0]]) & set(recs[M[1]]), key=lambda q: (len(q), q))
        table(recs, qids, bench)
        if bench == "math500":
            sys.path.insert(0, ".")
            from datasets import load_dataset
            lv = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="cache")["level"]
            for levels in ((1,), (2,), (1, 2), (3,), (4, 5)):
                table(recs, [q for q in qids if lv[int(q)] in levels], f"math500 level {'+'.join(map(str, levels))}")


if __name__ == "__main__":
    main()
