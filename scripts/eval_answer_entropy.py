"""Teacher-forced entropy on the pass@8 MATH-500 traces: answer tokens and full trace.

Texts: up to `--per_q` finished samples per question per model from the pass@8
runs (a sample is finished if it did not hit the cap; it may close </think> or
stop inside it). Every scorer model reads every model's texts (3 x 3), so an
entropy difference can be separated from a difference in the text itself.

For each (scorer, text) pair, at temperature 1:
  first_ent   entropy of the next-token distribution at the first token of the
              final answer (the first token inside the final \\boxed{...})
  ans_ent     mean entropy over all tokens inside that box
  ans_nll     total negative log-likelihood of the answer string
  trace_ent   mean entropy over the generated reasoning (all generated tokens)
Self-scored texts also keep per-token entropy with character offsets
(`--out_dir/tokens_<model>.npz`) for the step-level entropy dynamics analysis.

Run: python scripts/eval_answer_entropy.py --per_q 4 --out_dir eval_outputs/rd_newsae/eval_mod/answer_entropy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custom_eval.config import ModelSpec  # noqa: E402
from custom_eval.modeling import load_model_and_tokenizer  # noqa: E402

REPO = "Khaquan/qwen-khaquanS-distillations"
MODELS = {
    "LKD": ("qwen-logitKD-final/qwen_logit_kd_continue/epoch_1",
            "eval_outputs/sweep_pass_k/20260929T073130Z-1261224/qwen-logit-kd-continue-epoch_1-think__math500.json"),
    "RD_new": ("qwen-reasondistill-october/qwen_reasondistill_final_continue/epoch_1",
               "eval_outputs/sweep_pass_k_rd_october/qwen-reasondistill-october-continue-epoch_1-think__math500.json"),
}
# Reported set: LKD, October RD ("RD_new") and new-SAE RD moderate (docs/rd_newsae.md; local checkpoint, its
# own pass@8 outputs). Also read by eval_bond_annotate.py, which imports MODELS from here. (Formerly selected
# with RD_MOD_SET=1; the default set with the first RD run is archived.)
MODELS["RD_mod"] = (str(ROOT / "checkpoints/rd_newsae_moderate/final_step_94"),
                    "eval_outputs/rd_newsae/eval_mod/passk/qwen-rd-newsae-moderate-final-think__math500.json")


def final_box_span(text):
    """Character span of the committed answer's box content.

    The FIRST \\boxed{} after </think> (later boxes are often repeats; fixed
    2026-10-21 - the first version took the last box); if the trace never
    closed </think>, the last box in the thinking.
    """
    k = text.find("</think>")
    if k >= 0:
        starts = [k + m.start() for m in re.finditer(r"\\boxed\s*\{", text[k:])][:1]
    else:
        starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)][-1:]
    if not starts:
        return None
    m = re.match(r"\\boxed\s*\{", text[starts[0]:])
    i = starts[0] + m.end()
    depth, j = 1, i
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return (i, j) if j > i else None
        j += 1
    return None


def select_texts(per_q):
    texts = []
    for name, (_, path) in MODELS.items():
        for rec in json.load(open(ROOT / path))["records"]:
            taken = 0
            for c in rec["candidates"]:
                if c["hit_token_limit"] or taken >= per_q:
                    continue
                span = final_box_span(c["raw_output"])
                if span is None:
                    continue
                texts.append({"author": name, "qid": rec["id"], "idx": c["index"],
                              "level": rec["metadata"].get("level"), "correct": bool(c["correct"]),
                              "closed": "</think>" in c["raw_output"],
                              "prompt": c["prompt"], "out": c["raw_output"], "span": span})
                taken += 1
    return texts


@torch.no_grad()
def score(model, tok, item, keep_tokens, chunk=1024):
    p_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
    enc = tok(item["out"], add_special_tokens=False, return_offsets_mapping=True)
    o_ids, offs = enc["input_ids"], enc["offset_mapping"]
    ids = torch.tensor([p_ids + o_ids], device=model.device)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    # position t predicts token t+1; generated token k sits at index len(p) + k
    P = len(p_ids)
    pos = torch.arange(P - 1, P - 1 + len(o_ids), device=model.device)
    ent = torch.empty(len(o_ids), device=model.device)
    nll = torch.empty(len(o_ids), device=model.device)
    tgt = torch.tensor(o_ids, device=model.device)
    for a in range(0, len(o_ids), chunk):
        lp = F.log_softmax(model.lm_head(hidden[pos[a:a + chunk]]).float(), -1)
        ent[a:a + chunk] = -(lp.exp() * lp).sum(-1)
        nll[a:a + chunk] = -lp.gather(-1, tgt[a:a + chunk, None])[:, 0]
    ent, nll = ent.cpu().numpy(), nll.cpu().numpy()
    s, e = item["span"]
    # tokens that START inside the box (a token like "{\\" straddling the
    # opening brace is excluded; fixed 2026-10-21)
    in_ans = [k for k, (cs, ce) in enumerate(offs) if s <= cs < e]
    res = {"trace_ent": float(ent.mean()), "n_tokens": len(o_ids)}
    if in_ans:
        res.update(first_ent=float(ent[in_ans[0]]), ans_ent=float(ent[in_ans].mean()),
                   ans_nll=float(nll[in_ans].sum()), ans_len=len(in_ans))
    if keep_tokens:
        res["_ent"] = ent.astype(np.float16)
        res["_offs"] = np.array([o[0] for o in offs], dtype=np.int32)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_q", type=int, default=2)
    ap.add_argument("--max_q", type=int, default=None)
    ap.add_argument("--out_dir", default=str(ROOT / "eval_outputs/rd_newsae/eval_mod/answer_entropy"))
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = select_texts(args.per_q)
    if args.max_q:
        keep = sorted({t["qid"] for t in texts})[: args.max_q]
        texts = [t for t in texts if t["qid"] in keep]
    print(f"{len(texts)} texts: " + ", ".join(f"{m}={sum(t['author'] == m for t in texts)}" for m in MODELS), flush=True)
    rows = []
    for scorer, (sub, _) in MODELS.items():
        local = sub.startswith("/")  # absolute path = local checkpoint
        spec = ModelSpec(name=scorer, checkpoint=sub if local else REPO, subfolder=None if local else sub,
                         dtype="bfloat16", device_map="auto", strip_language_model_prefix=True)
        model, tok = load_model_and_tokenizer(spec, cache_dir=args.cache_dir)
        model.eval()
        toks = {}
        t0 = time.time()
        for n, item in enumerate(texts):
            self_scored = item["author"] == scorer
            r = score(model, tok, item, keep_tokens=self_scored)
            if self_scored:
                key = f"{item['qid']}|{item['idx']}"
                toks[key + "|ent"], toks[key + "|offs"] = r.pop("_ent"), r.pop("_offs")
            rows.append({"scorer": scorer, **{k: item[k] for k in ("author", "qid", "idx", "level", "correct", "closed")}, **r})
            if n == 0 and r["trace_ent"] > 5:
                raise RuntimeError(f"{scorer}: implausible entropy {r['trace_ent']:.2f}; weights may not have loaded")
            if (n + 1) % 200 == 0:
                print(f"  {scorer} {n + 1}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(out / f"tokens_{scorer}.npz", **toks)
        (out / "rows.json").write_text(json.dumps(rows))
        del model
        torch.cuda.empty_cache()
        print(f"{scorer} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
