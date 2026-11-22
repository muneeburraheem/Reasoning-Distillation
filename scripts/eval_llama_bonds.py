"""CoT bond structure (Chen et al. 2026; RUN_LOG 8n families A-E) for the Llama arm.

Texts: the Llama pass@8 GSM8K outputs (eval_outputs/llama_passk); one finished sample per
question per model (the lowest-index sample that did not hit the cap or loop), only questions
where both models have one. GSM8K, not MATH-500: on MATH-500 most Llama samples truncate.

Steps, labels and embeddings exactly as scripts/eval_bond_annotate.py (Qwen3.5-4B annotator, blind
to the author; base Qwen3.5-2B embeddings). Entropy: every Llama model reads every selected text
(prompt "{question}\\n<think>" + output), as scripts/eval_answer_entropy.py; the self-scored per-token
entropies feed family C. Answer = first \\boxed{} after </think>, else the <answer> content.

Outputs in --out_dir: steps.json, embeddings.npy, raw.json, entropy/rows.json,
entropy/tokens_<model>.npz. Then:
  python scripts/analyze_bonds.py --bonds <out_dir> --ent <out_dir>/entropy
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "llama_eval"))
from eval_answer_entropy import final_box_span  # noqa: E402
from eval_bond_annotate import annotate, embed, segment  # noqa: E402
from eval_llama_passk import MODELS  # noqa: E402

from batched_evals.benchmarks import load_benchmark  # noqa: E402

NAMES = {"LKD": "logit_kd_epoch2", "RD": "reasondistill_epoch2"}


def answer_span(text):
    span = final_box_span(text)
    if span:
        return span
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    return (m.start(1), m.end(1)) if m and m.end(1) > m.start(1) else None


def select(bench, cache_dir):
    qtext = {ex.id: ex.question for ex in load_benchmark(bench, cache_dir=cache_dir, split="test").examples}
    per = {}
    for m, key in NAMES.items():
        d = {}
        for rec in json.load(open(ROOT / f"eval_outputs/llama_passk/{key}__{bench}.json"))["records"]:
            for j, c in enumerate(rec["candidates"]):
                if not c["hit_limit"] and not c["loop"]:
                    d[rec["id"]] = {"idx": j, "text": c["text"], "correct": bool(c["correct_strict"]),
                                    "closed": "</think>" in c["text"], "state": c["state"]}
                    break
        per[m] = d
    common = sorted(set.intersection(*(set(d) for d in per.values())), key=lambda q: (len(q), q))
    return per, common, qtext


@torch.no_grad()
def entropy(per, common, qtext, out):
    texts = [(m, q) for m in per for q in common]
    rows = []
    for scorer, key in NAMES.items():
        tok = AutoTokenizer.from_pretrained(str(MODELS[key]))
        model = AutoModelForCausalLM.from_pretrained(str(MODELS[key]), dtype=torch.bfloat16,
                                                     device_map={"": 0}).eval()
        toks, t0 = {}, time.time()
        for n, (author, q) in enumerate(texts):
            it = per[author][q]
            p_ids = tok(f"{qtext[q]}\n<think>")["input_ids"]
            enc = tok(it["text"], add_special_tokens=False, return_offsets_mapping=True)
            o_ids, offs = enc["input_ids"], enc["offset_mapping"]
            ids = torch.tensor([p_ids + o_ids], device="cuda")
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            P = len(p_ids)
            tgt = torch.tensor(o_ids, device="cuda")
            ent = torch.empty(len(o_ids), device="cuda")
            for a in range(0, len(o_ids), 1024):
                lp = F.log_softmax(model.lm_head(hidden[P - 1 + a:P - 1 + min(a + 1024, len(o_ids))]).float(), -1)
                ent[a:a + 1024] = -(lp.exp() * lp).sum(-1)
            ent = ent.cpu().numpy()
            r = {"scorer": scorer, "author": author, "qid": q, "idx": it["idx"], "level": 1,
                 "correct": it["correct"], "closed": it["closed"], "trace_ent": float(ent.mean()),
                 "n_tokens": len(o_ids)}
            span = answer_span(it["text"])
            if span:
                s, e = span
                k = [i for i, (cs, _) in enumerate(offs) if s <= cs < e]
                if k:
                    r.update(first_ent=float(ent[k[0]]), ans_ent=float(ent[k].mean()), ans_len=len(k))
            if author == scorer:
                toks[f"{q}|{it['idx']}|ent"] = ent.astype(np.float16)
                toks[f"{q}|{it['idx']}|offs"] = np.array([o[0] for o in offs], dtype=np.int32)
            rows.append(r)
            if n == 0 and r["trace_ent"] > 5:
                raise RuntimeError(f"{scorer}: implausible entropy {r['trace_ent']:.2f}")
            if (n + 1) % 500 == 0:
                print(f"  entropy {scorer} {n + 1}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out / f"tokens_{scorer}.npz", **toks)
        (out / "rows.json").write_text(json.dumps(rows))
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="gsm8k")
    ap.add_argument("--max_q", type=int, default=None)
    ap.add_argument("--out_dir", default=str(ROOT / "eval_outputs/llama_bonds"))
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "entropy").mkdir(parents=True, exist_ok=True)
    per, common, qtext = select(args.bench, args.cache_dir)
    if args.max_q:
        common = common[:args.max_q]
    (out / "raw.json").write_text(json.dumps({f"{m}|{q}|{per[m][q]['idx']}": per[m][q]["text"]
                                              for m in per for q in common}))
    rows, pairs = [], []
    for m in per:
        for q in common:
            it = per[m][q]
            s = segment(it["text"].split("</think>")[0])
            for k, cur in enumerate(s):
                rows.append({"model": m, "qid": q, "idx": it["idx"], "step": k, "n_steps": len(s),
                             "correct": it["correct"], "closed": it["closed"], "level": 1, "text": cur})
                pairs.append((s[k - 1] if k else "", cur))
    print(f"{len(common)} questions, {len(rows)} steps: " +
          ", ".join(f"{m}={sum(r['model'] == m for r in rows)}" for m in per), flush=True)
    entropy(per, common, qtext, out / "entropy")
    print("entropy saved", flush=True)
    probs = annotate(pairs, args.cache_dir)
    for r, p in zip(rows, probs):
        r["label"] = ["normal operation", "deep reasoning", "self-reflection", "exploration"][int(p.argmax())]
        r["p"] = [round(float(x), 4) for x in p]
    (out / "steps.json").write_text(json.dumps(rows))
    print("labels saved", flush=True)
    np.save(out / "embeddings.npy", embed([r["text"] for r in rows], args.cache_dir))
    print("embeddings saved", flush=True)


if __name__ == "__main__":
    main()
