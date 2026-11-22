"""Teacher-forced KL for the Llama arm (the RUN_LOG 8f measurement, Llama students).

Teacher DeepSeek-R1-Distill-Llama-8B in 8-bit (as in training), students LKD / RD epoch 2,
plus base Llama-3.2-1B as reference. All read with the student tokenizer, as in training.

Texts. The Llama students trained on ALL of am_0.9M, so nothing there is held out:
  heldout   am_0.5M (sibling subset of the same AM-DeepSeek-R1 release), math rows
            (openR1Math_extended), streamed from the Hub; any row whose question also occurs
            in am_0.9M (whitespace/case-normalised exact match) is dropped. Formatted like
            training: "question\\n" + assistant, with 0.5M's "<think>\\n" written "<think>"
            as in 0.9M.
  insample  am_0.9M rows from offset 800,000 (training text; reference for the held-out gap).
Up to 2,048 tokens. Training used max_length 512, so every statistic is also split by
position: pos < 512 (trained context) vs >= 512.

Per target token (regions think / answer / boxed, as 8f; the multi-token </think> is skipped):
  kl1 / kl4  KL(teacher || model) at T=1 and T=4 (training temperature; raw, not x tau^2)
  ent        model entropy (T=1);  nll  NLL of the actual next token

Run:
  python scripts/eval_llama_teacher_forced_kl.py --out eval_outputs/llama_teacher_forced/results.json
"""

import argparse
import io
import json
import random
import re
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_llama_passk import MODELS  # noqa: E402
from eval_teacher_forced_kl import boxed_spans  # noqa: E402

# Llama-1B student checkpoints (local, not on the Hub); set LLAMA_KD_DIR on another machine
LLAMA_DIR = Path(os.environ.get("LLAMA_KD_DIR", "/home/cs/Desktop/KD/Reasoning-Distillation/KD_TRAIN_Deepseek_Dataset"))
TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE = "meta-llama/Llama-3.2-1B"
REGIONS = ("think", "answer", "boxed")
STATS = ("kl1", "kl4", "ent", "nll")
TRAIN_LEN = 512
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731


def heldout_rows(n, seed, scan, train_questions):
    import zstandard
    from huggingface_hub import HfFileSystem
    f = HfFileSystem().open("datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/am_0.5M.jsonl.zst",
                            "rb", block_size=8 << 20)
    lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
    pool, dup = [], 0
    for i, line in enumerate(lines):
        if i >= scan:
            break
        msgs = json.loads(line)["messages"]
        info = msgs[0].get("info") if isinstance(msgs[0].get("info"), dict) else {}
        if info.get("source") != "openR1Math_extended":
            continue
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        if norm(user) in train_questions:
            dup += 1
            continue
        if asst.startswith("<think>\n"):
            asst = "<think>" + asst[len("<think>\n"):]
        pool.append({"row": i, "user": user, "text": user + "\n" + asst})
    print(f"held-out: scanned {scan}, {len(pool)} math rows kept, {dup} dropped as in am_0.9M", flush=True)
    return sorted(random.Random(seed).sample(pool, n), key=lambda r: r["row"])


def labels_for(text, user, offsets):
    resp = len(user) + 1
    close = text.find("</think>", resp)
    end_close = close + len("</think>") if close >= 0 else -1
    spans = boxed_spans(text, close) if close >= 0 else []
    out = []
    for s, e in offsets:
        if s < resp or e <= s:
            out.append(None)
        elif close >= 0 and close <= s < end_close or close >= 0 and s < close < e:
            out.append(None)  # pieces of the multi-token </think>
        elif close < 0 or s < close:
            out.append("think")
        elif any(a <= s < b for a, b in spans):
            out.append("boxed")
        else:
            out.append("answer")
    return out


def load_models(cache_dir):
    q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    models = {"teacher": AutoModelForCausalLM.from_pretrained(
        TEACHER, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16, cache_dir=cache_dir)}
    for name, key in (("lkd", "logit_kd_epoch2"), ("rd", "reasondistill_epoch2")):
        models[name] = AutoModelForCausalLM.from_pretrained(str(MODELS[key]), dtype=torch.bfloat16,
                                                            device_map={"": 0})
    models["base"] = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0},
                                                          cache_dir=cache_dir)
    return {k: m.eval() for k, m in models.items()}


@torch.no_grad()
def score(models, ids, labels, chunk=256):
    logits = {k: m(input_ids=ids, use_cache=False).logits[0, :-1] for k, m in models.items()}
    targets = ids[0, 1:]
    lab = labels[1:]
    keys = [(r, w) for r in REGIONS for w in ("lt512", "ge512")]
    acc = {k: {f"{r}/{w}": {s: 0.0 for s in STATS} | {"n": 0} for r, w in keys} for k in models}
    for a in range(0, targets.shape[0], chunk):
        rows = [i for i in range(a, min(a + chunk, targets.shape[0])) if lab[i] is not None]
        if not rows:
            continue
        idx = torch.tensor(rows, device=ids.device)
        lt1 = F.log_softmax(logits["teacher"][idx].float(), -1)
        lt4 = F.log_softmax(logits["teacher"][idx].float() / 4, -1)
        for k in models:
            lp1 = F.log_softmax(logits[k][idx].float(), -1)
            lp4 = F.log_softmax(logits[k][idx].float() / 4, -1)
            vals = torch.stack([(lt1.exp() * (lt1 - lp1)).sum(-1), (lt4.exp() * (lt4 - lp4)).sum(-1),
                                -(lp1.exp() * lp1).sum(-1),
                                -lp1.gather(-1, targets[idx][:, None])[:, 0]], 1).tolist()
            for j, pos in enumerate(rows):
                # pos is the predicting position; the target sits at pos + 1
                r = acc[k][f"{lab[pos]}/{'lt512' if pos + 1 < TRAIN_LEN else 'ge512'}"]
                r["n"] += 1
                for s, v in zip(STATS, vals[j]):
                    r[s] += v
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval_outputs/llama_teacher_forced/results.json"))
    ap.add_argument("--n_heldout", type=int, default=1000)
    ap.add_argument("--n_insample", type=int, default=300)
    ap.add_argument("--scan", type=int, default=60000)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=str(LLAMA_DIR / "cache"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_from_disk
    train = load_from_disk(str(LLAMA_DIR / "cache/am_0.9M_text"))
    train_q = {norm(t.split("\n<think>", 1)[0]) for t in train["text"]}
    jobs = [("heldout", r) for r in heldout_rows(args.n_heldout, args.seed, args.scan, train_q)]
    rng = random.Random(args.seed)
    for i in sorted(rng.sample(range(800_000, len(train)), args.n_insample)):
        t = train[i]["text"]
        jobs.append(("insample", {"row": i, "user": t.split("\n<think>", 1)[0], "text": t}))
    del train, train_q

    tok = AutoTokenizer.from_pretrained(str(MODELS["logit_kd_epoch2"]))
    models = load_models(args.cache_dir)
    results, t0 = [], time.time()
    for n, (domain, r) in enumerate(jobs):
        enc = tok(r["text"], truncation=True, max_length=args.max_length, return_offsets_mapping=True)
        labels = labels_for(r["text"], r["user"], enc["offset_mapping"])
        ids = torch.tensor([enc["input_ids"]], device="cuda")
        acc = score(models, ids, labels)
        results.append({"row": r["row"], "domain": domain, "tokens": len(enc["input_ids"]),
                        "has_close": "</think>" in tok.decode(enc["input_ids"]), "stats": acc})
        if n == 0:
            th = {k: v["think/lt512"]["nll"] / max(v["think/lt512"]["n"], 1) for k, v in acc.items()}
            print("first-trace think NLL:", th, flush=True)
            if any(x > 5 for x in th.values()):
                raise RuntimeError(f"implausible NLL, weights likely not loaded: {th}")
        if (n + 1) % 20 == 0 or n + 1 == len(jobs):
            out.write_text(json.dumps({"args": vars(args), "results": results}))
            print(f"{n + 1}/{len(jobs)} {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
