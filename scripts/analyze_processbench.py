"""ProcessBench (RUN_LOG 8s): LKD vs October RD, earliest-error identification.

Per split and pooled: accuracy on erroneous solutions, on correct solutions, official F1
(harmonic mean; an invalid answer counts as wrong), the same restricted to items where the
answer is a valid index (-1..n_steps-1), and compliance (share of valid answers).
Also a coarse "error detection" view: on erroneous items, did the model flag any step
(answer >= 0)? on correct items, did it answer -1? Paired: sign test on per-item exact
correctness, bootstrap over items (stratified by split) for the F1 difference.
"""
import json, random, re, sys
from collections import Counter
from math import comb

D = "eval_outputs/processbench/"
NAMES = {"LKD": "qwen-logit-kd-continue-epoch_1-think", "RD": "qwen-reasondistill-october-continue-epoch_1-think"}
TAGS = "--boxed-only" not in sys.argv
# Optional result files: LKD=<results.json> RD=<results.json> ("LKD"/"RD" are only the column labels; the
# new-SAE suite passes RD_mod as RD, and October RD as LKD). Default: the LKD and October RD runs above.
FILES = {m: D + f"{n}__processbench.json" for m, n in NAMES.items()}
FILES.update(a.split("=", 1) for a in sys.argv[1:] if a.split("=", 1)[0] in NAMES)
SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")


def sign(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def parse(c, n_steps):
    """Return (index or None, outcome) where outcome classifies how the answer ended."""
    if c["hit_token_limit"]:
        return None, "truncated"
    raw = c["raw_output"]
    if "</think>" not in raw:
        return None, "stopped_in_think"
    m = re.fullmatch(r"\s*(-?\d+)\s*", str(c.get("final_response") or ""))
    if not m and TAGS:
        # Both students were trained to write <answer>...</answer>; many answer the index
        # there without \boxed{}. Accept a bare integer or exactly one "paragraph N" mention.
        t = re.findall(r"<answer>(.*?)</answer>", raw.split("</think>")[-1], re.S)
        t = re.sub(r"[*$\\]|\\boxed|\{|\}", "", t[-1]) if t else ""
        m = re.fullmatch(r"\s*(-?\d+)\.?\s*", t)
        if not m:
            ps = set(re.findall(r"paragraph[ _]?(-?\d+)", t, re.I))
            m = re.match(r"(-?\d+)", ps.pop()) if len(ps) == 1 else None
    if not m:
        return None, "no_integer_answer"
    v = int(m.group(1))
    return (v, "valid") if -1 <= v < n_steps else (None, "out_of_range")


def load(model):
    out = {}
    for r in json.load(open(FILES[model]))["records"]:
        md = r["metadata"]
        c = r["candidates"][0]
        idx, how = parse(c, md["n_steps"])
        out[r["id"]] = dict(split=md["split"], label=md["label"], pred=idx, how=how,
                            exact=idx is not None and idx == md["label"], tokens=c["generated_tokens"])
    return out


def f1(rows, valid_only=False):
    if valid_only:
        rows = [r for r in rows if r["pred"] is not None]
    err = [r for r in rows if r["label"] != -1]
    ok = [r for r in rows if r["label"] == -1]
    a_err = sum(r["exact"] for r in err) / len(err) if err else float("nan")
    a_ok = sum(r["exact"] for r in ok) / len(ok) if ok else float("nan")
    f = 2 * a_err * a_ok / (a_err + a_ok) if a_err + a_ok > 0 else 0.0
    return a_err, a_ok, f, len(err), len(ok)


def detect(rows):
    err = [r for r in rows if r["label"] != -1]
    ok = [r for r in rows if r["label"] == -1]
    flag = sum(r["pred"] is not None and r["pred"] >= 0 for r in err) / len(err)
    clean = sum(r["pred"] == -1 for r in ok) / len(ok)
    early = [r for r in err if r["pred"] is not None and r["pred"] >= 0]
    pos = Counter("exact" if r["pred"] == r["label"] else "too early" if r["pred"] < r["label"] else "too late" for r in early)
    return flag, clean, pos


def boot_f1(L, R, ids, B=5000, valid_only=False):
    rng = random.Random(0)
    by = {s: [i for i in ids if L[i]["split"] == s] for s in SPLITS}
    ds = []
    for _ in range(B):
        pick = [i for s in SPLITS for i in rng.choices(by[s], k=len(by[s])) if by[s]]
        ds.append(f1([R[i] for i in pick], valid_only)[2] - f1([L[i] for i in pick], valid_only)[2])
    ds.sort()
    p = 2 * min(sum(x <= 0 for x in ds), sum(x >= 0 for x in ds)) / B
    return ds[int(.025 * B)], ds[int(.975 * B)], min(1.0, max(p, 1 / B))


def main():
    L, R = load("LKD"), load("RD")
    assert set(L) == set(R)
    ids = sorted(L)
    for scope in SPLITS + ("all",):
        sid = [i for i in ids if scope == "all" or L[i]["split"] == scope]
        print(f"\n=== {scope}  n={len(sid)}  (correct solutions {sum(L[i]['label'] == -1 for i in sid)})")
        for name, M in (("LKD", L), ("RD", R)):
            rows = [M[i] for i in sid]
            ae, ao, f, _, _ = f1(rows)
            ve, vo, vf, ne, no = f1(rows, True)
            how = Counter(r["how"] for r in rows)
            flag, clean, pos = detect(rows)
            comp = how["valid"] / len(rows)
            print(f"  {name:3s} official: err {ae:.3f} ok {ao:.3f} F1 {f:.3f} | valid-only (n={ne}+{no}): err {ve:.3f} ok {vo:.3f} F1 {vf:.3f}"
                  f" | compliance {comp:.3f}")
            print(f"      outcomes {dict(how)} | flags an error on erroneous {flag:.3f}, answers -1 on correct {clean:.3f}"
                  f" | flagged erroneous: {dict(pos)} | median tokens {sorted(r['tokens'] for r in rows)[len(rows)//2]}")
        a = sum(L[i]["exact"] and not R[i]["exact"] for i in sid)
        b = sum(R[i]["exact"] and not L[i]["exact"] for i in sid)
        lo, hi, p = boot_f1(L, R, sid)
        vl, vh, vp = boot_f1(L, R, sid, valid_only=True)
        cl = sum(L[i]["how"] == "valid" and R[i]["how"] != "valid" for i in sid)
        cr = sum(R[i]["how"] == "valid" and L[i]["how"] != "valid" for i in sid)
        print(f"  paired exact: LKD-only {a}, RD-only {b}, sign p={sign(a, b):.4f} | official F1 RD-LKD 95% [{lo:+.3f}, {hi:+.3f}] p={p:.4f}"
              f" | valid-only F1 [{vl:+.3f}, {vh:+.3f}] p={vp:.4f} | compliance LKD-only {cl}, RD-only {cr}, p={sign(cl, cr):.4f}")


if __name__ == "__main__":
    sys.exit(main())
