"""Budget forcing (RUN_LOG 8i) for the Llama arm: what does each model believe after B tokens?

Phase 1 samples up to B new tokens with the Llama arm's pipeline (scripts/eval_llama_passk.py:
native prompt, slot-refilled sampling, LoopStopper, stop at </answer>, T=0.7, top_p 0.95).
Every sample that did not close </answer> then gets a forced answer: the phase-1 text is kept
and the model continues greedily (up to --answer_tokens) after

  still thinking (cap, loop, EOS in think):  "\\n</think>\\n<answer>\\nThe final answer is \\boxed{"
  closed </think>, no <answer>:              "\\n<answer>\\nThe final answer is \\boxed{"
  inside <answer>, not closed:               "\\nThe final answer is \\boxed{"

Scores per sample, as 8i: hard_strict / hard_lenient (the arm's gated extractor on the phase-1
text, strict and lenient) and forced (natural answer if the sample closed </answer>, otherwise
the content of the prefilled box).

Run:
  python scripts/eval_llama_budget_forcing.py --bench gsm8k --budgets 256 512 1024 2048 \\
      --out_dir eval_outputs/llama_budget_forcing
  python scripts/eval_llama_budget_forcing.py --bench math500 --levels 1 2 --budgets 512 1024 2048 4096 \\
      --out_dir eval_outputs/llama_budget_forcing
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "llama_eval"))
from analyze_llama_processbench import answers, same  # noqa: E402
from eval_budget_forcing import left_pad, prefilled_box  # noqa: E402
from eval_llama_passk import MODELS, end_state, generate_pool  # noqa: E402

from batched_evals.benchmarks import load_benchmark  # noqa: E402
from batched_evals.generation import build_prompt  # noqa: E402

BOX = "The final answer is \\boxed{"
SUFFIX = {"think": "\n</think>\n<answer>\n" + BOX, "closed": "\n<answer>\n" + BOX, "in_answer": "\n" + BOX}


def suffix_for(state):
    if state in ("truncated_in_think", "stopped_in_think", "loop"):
        return SUFFIX["think"]
    if state == "closed_no_answer":
        return SUFFIX["closed"]
    return SUFFIX["in_answer"]  # answer_cut / answered_unclosed


@torch.no_grad()
def force(model, tok, seqs, answer_tokens, chunk):
    outs = []
    for s in range(0, len(seqs), chunk):
        ids, mask = left_pad(seqs[s:s + chunk], tok.pad_token_id, model.device)
        o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=answer_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs += [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in o]
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="gsm8k")
    ap.add_argument("--levels", nargs="+", type=int, default=None)
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--answer_tokens", type=int, default=32)
    ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--per_call", type=int, default=100)
    ap.add_argument("--force_chunk", type=int, default=32)
    ap.add_argument("--max_q", type=int, default=None, help="smoke test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    ap.add_argument("--out_dir", default=str(ROOT / "eval_outputs/llama_budget_forcing"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with (out / "progress.log").open("a") as f:
            f.write(msg + "\n")

    examples = list(load_benchmark(args.bench, cache_dir=args.cache_dir, split="test").examples)
    level = {}
    if args.bench == "math500":
        from datasets import load_dataset
        lv = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=args.cache_dir)["level"]
        level = {str(i): int(l) for i, l in enumerate(lv)}
        assert len(level) == len(examples) == 500
        if args.levels:
            examples = [ex for ex in examples if level[ex.id] in args.levels]
    examples = examples[:args.max_q] if args.max_q else examples
    tag = args.bench + ("_L" + "".join(map(str, args.levels)) if args.levels else "")
    log(f"{tag}: {len(examples)} questions x {args.samples} samples, budgets {args.budgets}")

    # Budgets outer, models inner: each budget finishes for both models first.
    for B in args.budgets:
        for name in args.models:
            path = out / f"{name}__{tag}__B{B}.json"
            if path.exists():
                log(f"skip {path.name}")
                continue
            tok = AutoTokenizer.from_pretrained(str(MODELS[name]))
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(str(MODELS[name]), dtype=torch.bfloat16,
                                                         device_map={"": 0}).eval()
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            records, t0 = [], time.time()
            for b in range(0, len(examples), args.per_call):
                batch = examples[b:b + args.per_call]
                p_ids = [tok(build_prompt(ex.question, False, "native"))["input_ids"] for ex in batch]
                rows = generate_pool(model, tok, p_ids, args.samples, B, 0.7, 0.95, args.slots)
                seqs, idx = [], []
                for i, row in enumerate(rows):
                    row["state"] = "loop" if row["loop"] else end_state(row["text"], row["hit_limit"])
                    if row["state"] != "answered":
                        idx.append(i)
                        seqs.append(p_ids[i // args.samples] + tok(row["text"] + suffix_for(row["state"]),
                                                                   add_special_tokens=False)["input_ids"])
                for i, o in zip(idx, force(model, tok, seqs, args.answer_tokens, args.force_chunk)):
                    rows[i]["forced_answer"], rows[i]["forced_raw"] = prefilled_box(o), o
                for q, ex in enumerate(batch):
                    cands = []
                    for row in rows[q * args.samples:(q + 1) * args.samples]:
                        a = answers(row["text"])
                        forced = a["strict"] if row["state"] == "answered" else row.get("forced_answer", "")
                        cands.append({"state": row["state"], "tokens": row["tokens"],
                                      "hard_strict": same(a["strict"], ex.answer),
                                      "hard_lenient": same(a["lenient"], ex.answer),
                                      "forced_answer": forced, "forced": same(forced, ex.answer),
                                      "text": row["text"], "forced_raw": row.get("forced_raw")})
                    records.append({"id": ex.id, "answer": ex.answer, "level": level.get(ex.id),
                                    "candidates": cands})
                acc = {k: np.mean([c[k] for r in records for c in r["candidates"]])
                       for k in ("hard_strict", "hard_lenient", "forced")}
                log(f"  {name} {tag} B={B} {len(records)}/{len(examples)} "
                    + " ".join(f"{k}={v:.3f}" for k, v in acc.items()) + f" ({time.time() - t0:.0f}s)")
            path.write_text(json.dumps({"model": name, "bench": tag, "budget": B, "samples": args.samples,
                                        "seed": args.seed, "records": records}))
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
