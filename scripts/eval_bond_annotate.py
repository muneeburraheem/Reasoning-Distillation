"""Label reasoning steps with the bond types of Chen et al. 2026 (arXiv 2601.06002), and embed them.

Follows the paper's Appendix 13.2-13.3:
  - steps: the thinking text split on "\\n\\n", "\\n" and ". "
  - each (previous step, current step) pair is classified with the paper's
    published annotation prompt into normal operation / deep reasoning /
    self-reflection / exploration
  - each step is embedded alone as the mean of a reference model's
    penultimate-layer token states

Differences from the paper, forced by what is available locally: the
annotator is Qwen3.5-4B with thinking off (paper: Qwen2.5-32B-Instruct), and
the label is the argmax over the four labels' first-token log-probabilities
after the prefilled "### Behavior:" (deterministic; all four first tokens are
distinct single tokens). The annotator never sees which model wrote a step.
Reference encoder: base Qwen3.5-2B.

Texts: one finished sample per question per model (the lowest-index sample
that did not hit the cap), only questions where every model has one.

Run: python scripts/eval_bond_annotate.py --out_dir eval_outputs/rd_newsae/eval_mod/bonds
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
from eval_answer_entropy import MODELS  # noqa: E402

LABELS = ["normal operation", "deep reasoning", "self-reflection", "exploration"]
PROMPT = """You are an expert annotator. Classify the CURRENT STEP into exactly one of the following categories of reasoning/behavior:
• normal operation — Straightforward, direct operations (e.g., arithmetic, factual recall, simple step-by-step logic) without introducing new logical nodes.
• deep reasoning — Multi-step causal, deductive, or analogical thinking that extends the reasoning chain by introducing new logical nodes or hidden assumptions.
• self-reflection — commenting on its own thought process (e.g., confidence, strategy, uncertainty, mistakes, or reconsideration of earlier reasoning) and tracing back to previous logical nodes.
• exploration — generating new possibilities, hypotheses, or questions, branching into alternative paths rather than following a single conclusion.
Decision rules:
(1) If multiple categories seem to overlap, choose the most specific match based on intent:
– If the text is about reasoning itself → self-reflection.
– If the text is branching or speculating → exploration.
– If the text is extending the reasoning chain with deeper causality or hidden steps → deep reasoning.
– Otherwise, if it's just direct calculation or straightforward logic → normal operation.
(2) Do not label based on correctness of the reasoning — only on the behavioral style of thinking.
(3) Ignore surface complexity (e.g., long math steps may still be normal operation if they are straightforward).
(4) If mixed, choose the dominant intent; break ties with this priority: self-reflection > exploration > deep reasoning > normal operation.
Output format (strict): Return exactly one line and nothing else:
### Behavior: {normal operation | deep reasoning | self-reflection | exploration}

PREVIOUS STEP: {prev}

CURRENT STEP: {cur}"""
MAX_STEP_CHARS = 1500


def segment(thinking):
    parts = re.split(r"\n\n|\n|(?<=\.) ", thinking)
    return [p.strip() for p in parts if p and p.strip()]


def select():
    per_model = {}
    for name, (_, path) in MODELS.items():
        d = {}
        for rec in json.load(open(ROOT / path))["records"]:
            for c in sorted(rec["candidates"], key=lambda c: c["index"]):
                if not c["hit_token_limit"]:
                    think = c["raw_output"].split("</think>")[0]
                    d[rec["id"]] = {"idx": c["index"], "correct": bool(c["correct"]),
                                    "closed": "</think>" in c["raw_output"],
                                    "level": rec["metadata"].get("level"), "steps": segment(think)}
                    break
        per_model[name] = d
    common = sorted(set.intersection(*(set(d) for d in per_model.values())))
    return {m: {q: per_model[m][q] for q in common} for m in per_model}, common


@torch.no_grad()
def annotate(items, cache_dir, batch=48):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=cache_dir)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16,
                                                 device_map={"": 0}, cache_dir=cache_dir).eval()
    label_ids = [tok.encode(" " + l.split()[0].split("-")[0], add_special_tokens=False)[0] for l in LABELS]
    assert len(set(label_ids)) == 4, label_ids
    prompts = []
    for prev, cur in items:
        user = PROMPT.replace("{prev}", prev[:MAX_STEP_CHARS] or "(start of reasoning)").replace("{cur}", cur[:MAX_STEP_CHARS])
        chat = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(chat + "### Behavior:")
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    probs = np.zeros((len(prompts), 4), dtype=np.float32)
    t0 = time.time()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        h = model.model(**enc, use_cache=False).last_hidden_state[:, -1]
        lp = F.log_softmax(model.lm_head(h).float(), -1)[:, label_ids]
        probs[idx] = F.softmax(lp, -1).cpu().numpy()
        if (b // batch) % 200 == 0:
            print(f"  annotate {b}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return probs


@torch.no_grad()
def embed(steps, cache_dir, batch=64):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", cache_dir=cache_dir)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype=torch.bfloat16,
                                                 device_map={"": 0}, cache_dir=cache_dir).eval()
    order = sorted(range(len(steps)), key=lambda i: len(steps[i]))
    out = None
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([steps[i][:MAX_STEP_CHARS] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(model.device)
        hs = model.model(**enc, output_hidden_states=True, use_cache=False).hidden_states[-2].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = ((hs * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
        if out is None:
            out = np.zeros((len(steps), v.shape[1]), dtype=np.float16)
        out[idx] = v
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_q", type=int, default=None)
    ap.add_argument("--out_dir", default=str(ROOT / "eval_outputs/rd_newsae/eval_mod/bonds"))
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, common = select()
    if args.max_q:
        common = common[: args.max_q]
    rows, pairs = [], []
    for m in data:
        for q in common:
            s = data[m][q]["steps"]
            for k, cur in enumerate(s):
                rows.append({"model": m, "qid": q, "idx": data[m][q]["idx"], "step": k, "n_steps": len(s),
                             "correct": data[m][q]["correct"], "closed": data[m][q]["closed"],
                             "level": data[m][q]["level"], "text": cur})
                pairs.append((s[k - 1] if k else "", cur))
    print(f"{len(common)} questions, {len(rows)} steps: " +
          ", ".join(f"{m}={sum(r['model'] == m for r in rows)}" for m in data), flush=True)
    probs = annotate(pairs, args.cache_dir)
    for r, p in zip(rows, probs):
        r["label"] = LABELS[int(p.argmax())]
        r["p"] = [round(float(x), 4) for x in p]
    (out / "steps.json").write_text(json.dumps(rows))
    print("labels saved", flush=True)
    emb = embed([r["text"] for r in rows], args.cache_dir)
    np.save(out / "embeddings.npy", emb)
    print("embeddings saved", flush=True)


if __name__ == "__main__":
    main()
