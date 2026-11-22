"""ProcessBench (RUN_LOG 8s) for the Llama arm.

Same items as the Qwen runs; everything else is the Llama arm's own pipeline, as in
scripts/eval_llama_passk.py: native prompt "{question}\\n<think>" (the training layout), slot-refilled
sampling with LoopStopper, stop at "</answer>", T=0.7, top_p 0.95, commitment-gated extraction.

processbench    the 1,600 items of custom_eval/benchmarks/processbench.py, official critique
                prompt followed by "\\n<think>".

Run:
  python scripts/eval_llama_processbench.py --task processbench --out_dir eval_outputs/llama_processbench
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

from eval_llama_passk import MODELS, end_state, generate_pool  # noqa: E402

MAX_NEW = {"gsm8k": 4096, "math": 8192, "olympiadbench": 8192, "omnimath": 8192}


def processbench_items(args):
    from custom_eval.benchmarks.processbench import load
    rows = []
    for ex in load(cache_dir=args.cache_dir).examples:
        if args.splits and ex.metadata["split"] not in args.splits:
            continue
        rows.append({"id": ex.id, "split": ex.metadata["split"], "condition": "critique",
                     "label": ex.metadata["label"], "prompt": f"{ex.question}\n<think>"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["processbench"], required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--max_items", type=int, default=None, help="smoke test: shortest N per condition")
    ap.add_argument("--slots", type=int, default=48)
    ap.add_argument("--per_call", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with (out / "progress.log").open("a") as f:
            f.write(msg + "\n")

    rows = processbench_items(args)
    for name in args.models:
        path = MODELS[name]
        tok = AutoTokenizer.from_pretrained(str(path))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = None
        for cond in sorted({r["condition"] for r in rows}, key=str):
            todo_all = [r for r in rows if r["condition"] == cond]
            ids = {r["id"]: tok(r["prompt"])["input_ids"] for r in todo_all}
            # Fixed length order: smoke tests take the shortest items, reruns resume cleanly.
            todo_all.sort(key=lambda r: (len(ids[r["id"]]), r["id"]))
            if args.max_items:
                todo_all = todo_all[:args.max_items]
            fpath = out / f"{name}__{cond}.jsonl"
            done = {json.loads(l)["id"] for l in fpath.open()} if fpath.exists() else set()
            todo = [r for r in todo_all if r["id"] not in done]
            if not todo:
                log(f"skip {fpath.name} (complete)")
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16,
                                                             device_map={"": 0}).eval()
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            t0 = time.time()
            log(f"{name} {cond}: {len(todo)} to go")
            # One cap per call: group by split (GSM8K 4,096, the rest 8,192, as in llama_passk).
            for split in sorted({r["split"] for r in todo}):
                part = [r for r in todo if r["split"] == split]
                for b in range(0, len(part), args.per_call):
                    batch = part[b:b + args.per_call]
                    gen = generate_pool(model, tok, [ids[r["id"]] for r in batch], 1, MAX_NEW[split],
                                        args.temperature, args.top_p, args.slots)
                    with fpath.open("a") as f:
                        for r, g in zip(batch, gen):
                            f.write(json.dumps({
                                "id": r["id"], "split": split, "condition": cond, "label": r.get("label"),
                                "prefix_tokens": len(ids[r["id"]]), "tokens": g["tokens"],
                                "hit_limit": g["hit_limit"], "loop": g["loop"],
                                "state": "loop" if g["loop"] else end_state(g["text"], g["hit_limit"]),
                                "text": g["text"]}) + "\n")
                    log(f"  {name} {cond} {split} {min(b + args.per_call, len(part))}/{len(part)} "
                        f"({time.time() - t0:.0f}s)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
