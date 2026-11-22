"""Set B (RUN_LOG 8q): AIME 2024, HMMT Feb 2025, BRUMO 2025, CMIMC 2025; 4 samples, 16k.
New RD vs LKD, paired per problem: per-sample accuracy (strict and lenient), pass@4,
outcome split; per competition and pooled (bootstrap over problems, sign tests).

Run: python scripts/analyze_hard_B.py [LKD=<dir/model-name>] [RD=<dir/model-name>]
     reads <dir/model-name>__<bench>.json; defaults: LKD-continue and October RD in eval_outputs/hard_math/B/.
     "LKD" and "RD" are only the column labels: the new-SAE suite passes RD_mod as RD, and October RD as LKD."""
import json, random, sys
from math import comb
sys.path.insert(0, ".")
from custom_eval.math_scoring import aime_exact_match, extract_boxed_answer, math_equivalent
B = "eval_outputs/hard_math/B/"
FILES = {"L": B + "qwen-logit-kd-continue-epoch_1-think", "R": B + "qwen-reasondistill-october-continue-epoch_1-think"}
for _a in sys.argv[1:]:
    _k, _v = _a.split("=", 1); FILES[{"LKD": "L", "RD": "R"}[_k]] = _v
def sign(a, b):
    n, k = a + b, min(a, b); return 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def boot(d, b=10000):
    rng = random.Random(0); m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / len(d), m[250], m[9750], min(1.0, max(p, 1 / b))
pq = {"L": {}, "R": {}}
for bench in ("aime24", "hmmt25", "brumo25", "cmimc25"):
    eq = aime_exact_match if bench == "aime24" else math_equivalent
    M = {k: {r["id"]: r for r in json.load(open(f"{FILES[k]}__{bench}.json"))["records"]} for k in "LR"}
    ids = sorted(M["L"])
    def q(m, i):
        cs = M[m][i]["candidates"]; gold = M[m][i]["answer"]
        eos = [not c["hit_token_limit"] and "</think>" not in c["raw_output"] for c in cs]
        len_ok = [c["correct"] or (e and bool((b := extract_boxed_answer(c["raw_output"])) and eq(b, gold))) for c, e in zip(cs, eos)]
        return dict(acc=sum(c["correct"] for c in cs) / len(cs), len=sum(len_ok) / len(cs), any=any(c["correct"] for c in cs),
                    trunc=sum(c["hit_token_limit"] for c in cs) / len(cs), eos=sum(eos) / len(cs),
                    cw=sum(not c["hit_token_limit"] and not e and not c["correct"] for c, e in zip(cs, eos)) / len(cs))
    for m in "LR":
        for i in ids: pq[m][(bench, i)] = q(m, i)
    line = f"{bench:8s} n={len(ids):2d}"
    for k in ("acc", "len", "trunc", "cw", "eos"):
        a = sum(pq["L"][(bench, i)][k] for i in ids) / len(ids); r = sum(pq["R"][(bench, i)][k] for i in ids) / len(ids)
        line += f" | {k} {a:.3f}/{r:.3f}"
    g = sum(pq["L"][(bench, i)]["any"] and not pq["R"][(bench, i)]["any"] for i in ids); l = sum(pq["R"][(bench, i)]["any"] and not pq["L"][(bench, i)]["any"] for i in ids)
    print(line + f" | pass@4 LKD {sum(pq['L'][(bench,i)]['any'] for i in ids)/len(ids):.3f} RD {sum(pq['R'][(bench,i)]['any'] for i in ids)/len(ids):.3f} (LKD-only {g}, RD-only {l}, p={sign(g,l):.2f})")
keys = sorted(pq["L"])
print(f"\nPooled, {len(keys)} problems x 4 samples (LKD / RD):")
for k, name in (("acc", "per-sample accuracy strict"), ("len", "per-sample accuracy lenient"), ("trunc", "truncated"), ("cw", "committed wrong"), ("eos", "stopped in think")):
    m, lo, hi, p = boot([pq["R"][x][k] - pq["L"][x][k] for x in keys])
    print(f"  {name:28s} {sum(pq['L'][x][k] for x in keys)/len(keys):.3f} / {sum(pq['R'][x][k] for x in keys)/len(keys):.3f}  RD-LKD {100*m:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] p={p:.3f}")
g = sum(pq["L"][x]["any"] and not pq["R"][x]["any"] for x in keys); l = sum(pq["R"][x]["any"] and not pq["L"][x]["any"] for x in keys)
print(f"  pass@4 (any of 4 correct)    {sum(pq['L'][x]['any'] for x in keys)/len(keys):.3f} / {sum(pq['R'][x]['any'] for x in keys)/len(keys):.3f}  LKD-only {g}, RD-only {l}, sign p={sign(g,l):.3f}")
