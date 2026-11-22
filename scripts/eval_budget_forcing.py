"""Budget forcing on MATH-500: what does each model believe after B thinking tokens?

Phase 1 samples up to B new tokens exactly as the eval harness does (same chat
template, sampling settings and FinalAnswerStopper). Each sample then ends in
one of four states:

  natural       closed </think> and wrote a complete \\boxed{} answer
  closed_cut    closed </think> but hit B before finishing the answer
  eos_in_think  emitted end-of-turn before ever closing </think>
  budget        still thinking when B ran out

Phase 2 forces an answer for every sample that is not `natural`: the phase-1
text is kept, a stray end-of-turn is dropped, and the model continues greedily
(up to --answer_tokens) after the suffix

  thinking states:  "\\n</think>\\n\\nThe final answer is \\boxed{"
  closed_cut:       "\\n\\nThe final answer is \\boxed{"

Per sample we store both readings, so one run gives three scores:
  hard_strict   harness convention: only `natural` samples can be correct
  hard_lenient  + eos_in_think samples whose last \\boxed{} in thinking is correct
  forced        natural answer if natural, otherwise the forced answer

Greedy decoding in phase 2 reads off the model's most likely answer given its
own reasoning; phase 1 keeps temperature sampling.

Run:
  python scripts/eval_budget_forcing.py --budgets 512 1024 2048 4096 --samples 4 \\
      --out_dir eval_outputs/budget_forcing
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_eval.answer_stopping import FinalAnswerStopper
from custom_eval.benchmarks import load_benchmark
from custom_eval.config import ModelSpec
from custom_eval.generation import _format_prompt, extract_final_response
from custom_eval.math_scoring import extract_boxed_answer, math_equivalent
from custom_eval.modeling import load_model_and_tokenizer

REPO = "Khaquan/qwen-khaquanS-distillations"
MODELS = {
    "lkd": "qwen-logitKD-final/qwen_logit_kd_continue/epoch_1",
    "rd": "qwen_reasondistill_final_continue/epoch_1",
    "rd_oct": "qwen-reasondistill-october/qwen_reasondistill_final_continue/epoch_1",
    # new-SAE RD moderate (docs/rd_newsae.md), local checkpoint
    "rd_mod": str(ROOT / "checkpoints/rd_newsae_moderate/final_step_94"),
}


def spec_for(name):
    """ModelSpec for MODELS[name]: an absolute path is a local checkpoint, otherwise a subfolder of REPO."""
    sub = MODELS[name]
    local = sub.startswith("/")
    return ModelSpec(name=name, checkpoint=sub if local else REPO, subfolder=None if local else sub,
                     dtype="bfloat16", device_map="auto", strip_language_model_prefix=True)
THINK_SUFFIX = "\n</think>\n\nThe final answer is \\boxed{"
ANSWER_SUFFIX = "\n\nThe final answer is \\boxed{"
FORCE_CHUNK = 16


def prefilled_box(continuation):
    """Content of the box we opened: up to the brace that balances it.

    The model often closes our box and then opens another, so the harness's
    last-box extractor would read the wrong (often clipped) one.
    """
    depth, i = 1, 0
    while i < len(continuation):
        ch = continuation[i]
        if ch == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            return continuation[:i].strip()
        i += 1
    return ""  # never closed within answer_tokens


def left_pad(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def phase1(model, tok, prompt_ids, budget, gen):
    ids, mask = left_pad(prompt_ids, tok.pad_token_id, model.device)
    width = ids.shape[1]
    stopper = FinalAnswerStopper(tokenizer=tok, input_width=width, batch_size=len(prompt_ids),
                                 thinking=True, eos_token_ids=tok.eos_token_id)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                         do_sample=True, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id,
                         stopping_criteria=StoppingCriteriaList([stopper]), **gen)
    eos = set(tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id])
    rows = []
    for r in range(out.shape[0]):
        gen_ids = out[r, width:].tolist()
        stop = stopper.stop_lengths[r]
        if stop is not None:
            gen_ids = gen_ids[:stop]
        end = next((i for i, t in enumerate(gen_ids) if t in eos), None)
        ended_eos = end is not None
        gen_ids = gen_ids[:end] if ended_eos else gen_ids
        rows.append({"ids": gen_ids, "eos": ended_eos, "answer_stop": stop is not None})
    return rows


@torch.no_grad()
def phase2(model, tok, seqs, answer_tokens):
    ids, mask = left_pad(seqs, tok.pad_token_id, model.device)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=answer_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]


def classify(text, row):
    if "</think>" in text:
        answer = extract_final_response(text, "math500", True, "qwen")
        return ("natural", answer) if answer else ("closed_cut", "")
    return ("eos_in_think" if row["eos"] else "budget"), ""


def run(model, tok, examples, budget, samples, batch_q, answer_tokens, gen, log):
    records = []
    for b in range(0, len(examples), batch_q):
        batch = examples[b:b + batch_q]
        prompts = [_format_prompt(ex.question, "math500", tok, "qwen", True) for ex in batch]
        p_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        flat = [p for p in p_ids for _ in range(samples)]
        rows = phase1(model, tok, flat, budget, gen)
        force_idx, force_seqs = [], []
        for i, row in enumerate(rows):
            row["text"] = tok.decode(row["ids"], skip_special_tokens=False)
            row["state"], row["natural_answer"] = classify(row["text"], row)
            if row["state"] != "natural":
                suffix = ANSWER_SUFFIX if row["state"] == "closed_cut" else THINK_SUFFIX
                force_idx.append(i)
                force_seqs.append(flat[i] + row["ids"] + tok(suffix, add_special_tokens=False)["input_ids"])
        # Phase-2 prefill over full-length sequences is the memory peak; at
        # B=2048 a chunk of 64 OOMed on the 24 GB card.
        for s in range(0, len(force_seqs), FORCE_CHUNK):
            outs = phase2(model, tok, force_seqs[s:s + FORCE_CHUNK], answer_tokens)
            for i, o in zip(force_idx[s:s + FORCE_CHUNK], outs):
                rows[i]["forced_answer"] = prefilled_box(o)
                rows[i]["forced_raw"] = o
        for q, ex in enumerate(batch):
            cands = []
            for row in rows[q * samples:(q + 1) * samples]:
                nat_ok = bool(row["natural_answer"]) and math_equivalent(row["natural_answer"], ex.answer)
                in_think = extract_boxed_answer(row["text"]) if row["state"] == "eos_in_think" else None
                len_ok = nat_ok or bool(in_think and math_equivalent(in_think, ex.answer))
                forced = row["natural_answer"] if row["state"] == "natural" else row.get("forced_answer", "")
                cands.append({
                    "state": row["state"], "tokens": len(row["ids"]),
                    "natural_answer": row["natural_answer"], "forced_answer": forced,
                    "hard_strict": nat_ok, "hard_lenient": len_ok,
                    "forced": bool(forced) and math_equivalent(forced, ex.answer),
                    "text": row["text"], "forced_raw": row.get("forced_raw"),
                })
            records.append({"id": ex.id, "answer": ex.answer, "metadata": ex.metadata,
                            "candidates": cands})
        done = len(records)
        acc = {k: np.mean([c[k] for r in records for c in r["candidates"]])
               for k in ("hard_strict", "hard_lenient", "forced")}
        log(f"  B={budget} {done}/{len(examples)} " + " ".join(f"{k}={v:.3f}" for k, v in acc.items()))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch_q", type=int, default=16)
    ap.add_argument("--answer_tokens", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out_dir", default="eval_outputs/budget_forcing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "progress.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    bench = load_benchmark("math500", cache_dir=args.cache_dir, split="test",
                           max_samples=args.max_samples)
    examples = list(bench.examples)
    gen = dict(temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0)
    loaded = {}
    # Budgets outer, models inner: each budget finishes for both models before the next.
    for budget in args.budgets:
        for name in args.models:
            path = out_dir / f"{name}__B{budget}.json"
            if path.exists():
                log(f"skip {path} (exists)")
                continue
            if name not in loaded:
                loaded.clear()
                torch.cuda.empty_cache()
                loaded[name] = load_model_and_tokenizer(spec_for(name), cache_dir=args.cache_dir)
            model, tok = loaded[name]
            model.eval()
            # Same seed per (model, budget) so the two models see identical RNG streams.
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            t0 = time.time()
            log(f"{name} B={budget} start")
            records = run(model, tok, examples, budget, args.samples, args.batch_q,
                          args.answer_tokens, gen, log)
            path.write_text(json.dumps({"model": name, "budget": budget, "samples": args.samples,
                                        "elapsed_s": round(time.time() - t0, 1),
                                        "records": records}))
            log(f"{name} B={budget} saved in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
