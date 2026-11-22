"""Teacher-forced comparison of teacher and students on held-out AM-DeepSeek traces.

KD trained the students on the teacher's next-token distributions over
AM-DeepSeek text, never on the teacher's own generations. This scores that
training signal directly: on held-out traces (rows no training stage touched),
formatted exactly as in training ("question\\nanswer", no chat template,
truncated at 2048 tokens), every model predicts every next token and we record,
per position and grouped by region of the target token:

  kl1 / kl4      KL(teacher || model) at temperature 1 and 4 (4 = training KD
                 temperature; raw, not multiplied by tau^2)
  ent            entropy of the model's distribution (temperature 1)
  nll            negative log-likelihood of the actual next token
  p_eos          P(<|im_end|>) + P(<|endoftext|>) - mass on stopping
  p_close        P(</think>)

Regions: think (inside <think>...</think>), close (the position whose target is
</think>), answer (after </think>), boxed (tokens inside \\boxed{...} in the
answer). The teacher is loaded in 8-bit exactly as in training
(models/student_loader.load_teacher), because that is the distribution the
students were fitted to.

Run:
  python scripts/eval_teacher_forced_kl.py --no_base --students lkd rd_oct rd_mod --out eval_outputs/rd_newsae/eval_mod/teacher_forced/results.json
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_eval.config import ModelSpec
from custom_eval.modeling import load_model_and_tokenizer
from data_loaders.amdeepseek import AMDeepSeekDataset
from models.student_loader import load_teacher

MATH_SOURCES = {"NuminaMath_1.5", "MATH_metamathQA", "MATH_numina",
                "openR1Math_default", "MATH-lighteval", "Omni-MATH"}
# Training used rows 0-52,031 (stage 1: 0-40k, continuation: 40k-52,032).
MATH_RANGE = (531_000, 700_000)
NAT_RANGE = (160_000, 225_000)
REPO = "Khaquan/qwen-khaquanS-distillations"
STUDENTS = {
    "lkd": "qwen-logitKD-final/qwen_logit_kd_continue/epoch_1",
    "rd": "qwen_reasondistill_final_continue/epoch_1",
}
REGIONS = ("think", "close", "answer", "boxed")
STATS = ("kl1", "kl4", "ent", "nll", "p_eos", "p_close")


def select_rows(ds, n_math, n_nat, seed):
    rng = random.Random(seed)
    math_rows = []
    for i in range(MATH_RANGE[0], MATH_RANGE[1], 20):
        if ds[i]["messages"][0]["info"]["source"] in MATH_SOURCES:
            math_rows.append(i)
    nat_rows = list(range(NAT_RANGE[0], NAT_RANGE[1], 20))
    return (sorted(rng.sample(math_rows, n_math)),
            sorted(rng.sample(nat_rows, n_nat)))


def format_row(row):
    user = asst = ""
    for msg in row["messages"]:
        if msg["role"] == "user":
            user = msg["content"]
        elif msg["role"] == "assistant":
            asst = msg["content"]
    return user, user + "\n" + asst  # identical to AMDeepSeekDataset._format_example


def boxed_spans(text, start):
    spans = []
    for m in re.finditer(r"\\boxed\s*\{", text[start:]):
        i = start + m.end()
        depth = 1
        j = i
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            spans.append((i, j - 1))
    return spans


def region_labels(text, user, offsets):
    """Region of each token, as the TARGET of the preceding position."""
    resp = len(user) + 1
    close = text.find("</think>", resp)
    spans = boxed_spans(text, close) if close >= 0 else []
    labels = []
    for s, e in offsets:
        if s < resp:
            labels.append(None)
        elif close >= 0 and s == close:
            labels.append("close")
        elif close < 0 or s < close:
            labels.append("think")
        elif any(a <= s < b for a, b in spans):
            labels.append("boxed")
        else:
            labels.append("answer")
    return labels, "close" in labels  # </think> survived truncation


def load_models(cache_dir, with_base, students=None):
    models = {"teacher": load_teacher("Qwen/Qwen3.5-4B", 0, torch.bfloat16,
                                      quantize_8bit=True, cache_dir=cache_dir)}
    for name in students or STUDENTS:
        if students:
            from eval_budget_forcing import spec_for
            spec = spec_for(name)
        else:
            spec = ModelSpec(name=name, checkpoint=REPO, subfolder=STUDENTS[name], dtype="bfloat16",
                             device_map="auto", strip_language_model_prefix=True)
        model, _ = load_model_and_tokenizer(spec, cache_dir=cache_dir)
        models[name] = model.eval()
    if with_base:
        models["base"] = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-2B", dtype=torch.bfloat16, device_map={"": 0},
            cache_dir=cache_dir).eval()
    return models


@torch.no_grad()
def score(models, ids, labels, eos_ids, close_id, chunk=256):
    logits = {k: m(input_ids=ids, use_cache=False).logits[0, :-1] for k, m in models.items()}
    targets = ids[0, 1:]
    lab = labels[1:]
    acc = {k: {r: {s: 0.0 for s in STATS} | {"n": 0} for r in REGIONS} for k in models}
    for a in range(0, targets.shape[0], chunk):
        b = min(a + chunk, targets.shape[0])
        rows = [i for i in range(a, b) if lab[i] is not None]
        if not rows:
            continue
        idx = torch.tensor(rows, device=ids.device)
        lt1 = F.log_softmax(logits["teacher"][idx].float(), -1)
        lt4 = F.log_softmax(logits["teacher"][idx].float() / 4, -1)
        for k in models:
            lp1 = F.log_softmax(logits[k][idx].float(), -1)
            lp4 = F.log_softmax(logits[k][idx].float() / 4, -1)
            vals = {
                "kl1": (lt1.exp() * (lt1 - lp1)).sum(-1),
                "kl4": (lt4.exp() * (lt4 - lp4)).sum(-1),
                "ent": -(lp1.exp() * lp1).sum(-1),
                "nll": -lp1.gather(-1, targets[idx][:, None])[:, 0],
                "p_eos": lp1[:, eos_ids].exp().sum(-1),
                "p_close": lp1[:, close_id].exp(),
            }
            for j, pos in enumerate(rows):
                r = acc[k][lab[pos]]
                r["n"] += 1
                for s in STATS:
                    r[s] += vals[s][j].item()
            del lp1, lp4
        del lt1, lt4
    del logits
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_outputs/rd_newsae/eval_mod/teacher_forced/results.json")
    ap.add_argument("--n_math", type=int, default=1000)
    ap.add_argument("--n_nat", type=int, default=200)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--students", nargs="+", default=None,
                    help="names from budget_forcing.MODELS (default: lkd, rd from STUDENTS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="./cache")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(AMDeepSeekDataset.DATASET_ID, AMDeepSeekDataset.CONFIG_NAME,
                      split="train", cache_dir=args.cache_dir,
                      features=AMDeepSeekDataset._features())
    math_rows, nat_rows = select_rows(ds, args.n_math, args.n_nat, args.seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir=args.cache_dir)
    eos_ids = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    close_id = tok.convert_tokens_to_ids("</think>")
    models = load_models(args.cache_dir, not args.no_base, args.students)

    results = []
    t0 = time.time()
    jobs = [(r, "math") for r in math_rows] + [(r, "natural_reasoning") for r in nat_rows]
    for n, (row_id, domain) in enumerate(jobs):
        row = ds[row_id]
        user, text = format_row(row)
        enc = tok(text, truncation=True, max_length=args.max_length,
                  return_offsets_mapping=True)
        labels, has_close = region_labels(text, user, enc["offset_mapping"])
        ids = torch.tensor([enc["input_ids"]], device="cuda")
        acc = score(models, ids, labels, eos_ids, close_id)
        results.append({"row": row_id, "domain": domain,
                        "source": row["messages"][0]["info"]["source"],
                        "tokens": len(enc["input_ids"]), "has_close": has_close,
                        "stats": acc})
        if n == 0:
            # Guard against silently random weights (harness gotcha #2).
            nll = {k: v["think"]["nll"] / max(v["think"]["n"], 1) for k, v in acc.items()}
            print("first-trace think NLL:", nll, flush=True)
            if any(x > 5 for x in nll.values()):
                raise RuntimeError(f"implausible NLL, weights likely not loaded: {nll}")
        if (n + 1) % 20 == 0 or n + 1 == len(jobs):
            out.write_text(json.dumps({"args": vars(args), "results": results}))
            print(f"{n + 1}/{len(jobs)} {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
