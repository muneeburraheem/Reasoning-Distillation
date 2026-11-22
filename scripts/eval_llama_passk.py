"""pass@k for the Llama arm (Llama-3.2-1B students of DeepSeek-R1-Distill-Llama-8B).

Uses the Llama arm's own native format and scoring, copied into
llama_eval/batched_evals/ from KD_TRAIN_Deepseek_Dataset/batched_evals (unchanged):
prompt "{question}\\n<think>", generation stops at "</answer>", answers are read
under the commitment gate in both modes (lenient = the arm's headline, strict =
its lower bound), and matched with numeric_or_exact_match.

Per sample it also records what the existing outputs lack: generated tokens,
whether the cap was hit, and how the sample ended:

  answered           closed </answer>
  answered_unclosed  opened <answer>, ended before closing it (not at the cap)
  answer_cut         opened <answer>, hit the cap before closing it
  closed_no_answer   closed </think>, never opened <answer>
  stopped_in_think   end-of-sequence before </think>
  truncated_in_think hit the cap before </think>
  loop               stopped early by LoopStopper (counted as truncated; tokens censored)

LoopStopper: every 64 steps, a row whose last 400 generated tokens are >= 85%
repeated 8-grams is stopped. Validated on 4,800 earlier Llama-arm samples
(native_pass4, LKD/RD epoch 2, GSM8K + MATH-500): fired on 0 of 2,115 samples
that finished, and on 42.5% of the 2,685 that were truncated. It only saves
compute; a stopped row is a sample that was not going to finish.

Run:
  python scripts/eval_llama_passk.py --out_dir eval_outputs/llama_passk
"""

import argparse
import json
import random
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama_eval"))

from batched_evals.benchmarks import load_benchmark
from batched_evals.generation import NATIVE_STOP_STRINGS, build_prompt, extract_answer
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Llama-1B student checkpoints (local, not on the Hub); set LLAMA_KD_DIR on another machine
CKPT = Path(os.environ.get("LLAMA_KD_DIR", "/home/cs/Desktop/KD/Reasoning-Distillation/KD_TRAIN_Deepseek_Dataset")) / "checkpoints"
MODELS = {
    "logit_kd_epoch2": CKPT / "amdeepseek_logit_kd/epoch_2",
    "reasondistill_epoch2": CKPT / "amdeepseek_reasondistill/epoch_2",
}
# benchmark: (max_new_tokens, concurrent rows, questions per pool call)
PREFILL_TOKENS = 160_000
BENCH = {"aime25": (8192, 48, 30), "math500": (8192, 48, 50), "gsm8k": (4096, 64, 100)}


class LoopStopper(StoppingCriteria):
    def __init__(self, width, rows, stop_ids, window=400, ngram=8, threshold=0.85, every=64):
        self.width, self.window, self.n, self.thr, self.every = width, window, ngram, threshold, every
        self.stop_ids = stop_ids
        self.stopped = torch.zeros(rows, dtype=torch.bool)
        self.stop_len = [None] * rows
        self.step = 0

    def __call__(self, input_ids, scores=None, **kwargs):
        self.step += 1
        if self.step % self.every == 0 and self.step >= self.window:
            tail = input_ids[:, -self.window:].tolist()
            for r, ids in enumerate(tail):
                # A finished row is padded with end tokens; that padding is not a loop.
                if self.stopped[r] or ids[-1] in self.stop_ids:
                    continue
                grams = Counter(tuple(ids[i:i + self.n]) for i in range(len(ids) - self.n + 1))
                rep = sum(v for v in grams.values() if v > 1) / (len(ids) - self.n + 1)
                if rep >= self.thr:
                    self.stopped[r] = True
                    self.stop_len[r] = input_ids.shape[1] - self.width
        return self.stopped.to(input_ids.device)


def end_state(text, hit_limit):
    if "</answer>" in text:
        return "answered"
    if "<answer>" in text:
        return "answer_cut" if hit_limit else "answered_unclosed"
    if "</think>" in text:
        return "closed_no_answer"
    return "truncated_in_think" if hit_limit else "stopped_in_think"


def advance(model, tok, prompt_ids, active, max_new, temperature, top_p, chunk, stop_ids, done):
    """Advance one group of rows by up to `chunk` tokens; return the unfinished ones."""
    seqs = [prompt_ids[a["q"]] + a["gen"] for a in active]
    width = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(seqs):
        ids[i, width - len(x):] = torch.tensor(x)
        mask[i, width - len(x):] = 1
    ids, mask = ids.to(model.device), mask.to(model.device)
    steps = min(chunk, max(max_new - len(a["gen"]) for a in active))
    loops = LoopStopper(width, len(seqs), stop_ids)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps,
                         do_sample=True, temperature=temperature, top_p=top_p,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                         stop_strings=NATIVE_STOP_STRINGS, tokenizer=tok,
                         stopping_criteria=StoppingCriteriaList([loops]))
    new = out[:, width:].tolist()
    still = []
    for i, a in enumerate(active):
        row = new[i]
        if loops.stopped[i]:
            row = row[:loops.stop_len[i]]
        n = next((t for t, x in enumerate(row) if x in stop_ids), len(row))
        ended_eos = n < len(row) and not loops.stopped[i]
        a["gen"].extend(row[:n])
        a["gen"] = a["gen"][:max_new]
        text = tok.decode(a["gen"], skip_special_tokens=True)
        if loops.stopped[i]:
            done[(a["q"], a["j"])] = {"ids": a["gen"], "loop": True, "hit": True}
        elif "</answer>" in text or ended_eos:
            done[(a["q"], a["j"])] = {"ids": a["gen"], "loop": False, "hit": False}
        elif len(a["gen"]) >= max_new:
            done[(a["q"], a["j"])] = {"ids": a["gen"], "loop": False, "hit": True}
        else:
            still.append(a)
    return still


@torch.inference_mode()
def generate_pool(model, tok, prompt_ids, k, max_new, temperature, top_p, slots, chunk=512):
    """Sample k completions per prompt with a refilled pool of `slots` rows.

    Plain batched generate() keeps every row until the longest one ends, so a
    batch costs `max_new` steps however early most rows finish. Here each row
    is advanced `chunk` tokens at a time; rows that closed </answer>, hit the
    end token, looped or reached the cap leave, and queued samples take their
    slots. A row is continued from its own token ids, so its sampling
    distribution is identical to one uninterrupted generate().
    """
    stop_ids = {tok.pad_token_id} | set(tok.eos_token_id if isinstance(tok.eos_token_id, list)
                                         else [tok.eos_token_id])
    queue = [(q, j) for q in range(len(prompt_ids)) for j in range(k)]
    done = {}
    active = []  # dicts: q, j, gen (token ids)
    tok.padding_side = "left"
    while queue or active:
        while queue and len(active) < slots:
            q, j = queue.pop(0)
            active.append({"q": q, "j": j, "gen": []})
        # Re-prefill cost is rows x length; cap it so long rows run in smaller groups.
        active.sort(key=lambda a: len(prompt_ids[a["q"]]) + len(a["gen"]))
        groups, cur = [], []
        for a in active:
            L = len(prompt_ids[a["q"]]) + len(a["gen"])
            if cur and (len(cur) + 1) * L > PREFILL_TOKENS:
                groups.append(cur); cur = []
            cur.append(a)
        if cur:
            groups.append(cur)
        still = []
        for group in groups:
            still.extend(advance(model, tok, prompt_ids, group, max_new, temperature, top_p,
                                 chunk, stop_ids, done))
        active = still
    rows = []
    for q in range(len(prompt_ids)):
        for j in range(k):
            d = done[(q, j)]
            rows.append({"text": tok.decode(d["ids"], skip_special_tokens=True).strip(),
                         "tokens": len(d["ids"]), "hit_limit": d["hit"], "loop": d["loop"]})
    return rows


def run(name, path, bench_name, args, log, partial_path=None):
    max_new, slots, q_per_call = BENCH[bench_name]
    bench = load_benchmark(bench_name, cache_dir=args.cache_dir, split="test",
                           max_samples=args.max_samples)
    srcs = {str(ex.metadata.get("source")) for ex in bench.examples}
    if any("embedded_fallback" in s for s in srcs):
        raise RuntimeError(f"{bench_name}: real dataset failed to load, harness fell back to a placeholder")
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records, t0 = [], time.time()
    examples = list(bench.examples)
    for b in range(0, len(examples), q_per_call):
        batch = examples[b:b + q_per_call]
        prompts = [build_prompt(ex.question, False, "native") for ex in batch]
        p_ids = [tok(p)["input_ids"] for p in prompts]
        rows = generate_pool(model, tok, p_ids, args.k, max_new, args.temperature, args.top_p, slots)
        for q, ex in enumerate(batch):
            cands = []
            for row in rows[q * args.k:(q + 1) * args.k]:
                state = "loop" if row["loop"] else end_state(row["text"], row["hit_limit"])
                cand = {"tokens": row["tokens"], "hit_limit": row["hit_limit"], "loop": row["loop"],
                        "state": state, "text": row["text"]}
                for mode in ("lenient", "strict"):
                    e = extract_answer(row["text"], mode)
                    cand[f"answer_{mode}"] = e.answer if e.committed else ""
                    cand[f"cue_{mode}"] = e.cue
                    cand[f"correct_{mode}"] = bool(e.committed and bench.scorer(e.answer, ex.answer))
                cands.append(cand)
            records.append({"id": ex.id, "answer": ex.answer, "candidates": cands})
        done = len(records)
        acc = np.mean([c["correct_lenient"] for r in records for c in r["candidates"]])
        trunc = np.mean([c["hit_limit"] for r in records for c in r["candidates"]])
        log(f"  {name} {bench_name} {done}/{len(examples)} per-sample acc={acc:.3f} "
            f"trunc={trunc:.3f} {time.time() - t0:.0f}s")
        if partial_path is not None:
            partial_path.write_text(json.dumps({"model": name, "benchmark": bench_name,
                                                "status": "partial", "records": records}))
    del model
    torch.cuda.empty_cache()
    return records, max_new, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["aime25", "math500", "gsm8k"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    ap.add_argument("--out_dir", default=str(ROOT / "eval_outputs/llama_passk"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with (out / "progress.log").open("a") as f:
            f.write(msg + "\n")

    # Benchmarks outer, models inner: each benchmark finishes for both models first.
    for bench_name in args.benchmarks:
        for name in args.models:
            path = out / f"{name}__{bench_name}.json"
            if path.exists():
                log(f"skip {path.name} (exists)")
                continue
            log(f"{name} {bench_name} start")
            partial = out / f"{name}__{bench_name}.partial.json"
            records, max_new, secs = run(name, MODELS[name], bench_name, args, log, partial)
            path.write_text(json.dumps({"model": name, "checkpoint": str(MODELS[name]),
                                        "benchmark": bench_name, "k": args.k,
                                        "temperature": args.temperature, "top_p": args.top_p,
                                        "max_new_tokens": max_new, "seed": args.seed,
                                        "elapsed_s": round(secs, 1), "records": records}))
            partial.unlink(missing_ok=True)
            log(f"{name} {bench_name} saved in {secs:.0f}s")


if __name__ == "__main__":
    main()
