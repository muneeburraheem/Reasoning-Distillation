"""Harder math benchmarks (RUN_LOG 8p): recent competitions, OlympiadBench, Omni-MATH.

Each loader fixes the expected item count, drops items listed in
contamination_excluded.json (>= 10% of their word 13-grams appear in the
student training prompts, AM-DeepSeek rows 0-52,031), and raises rather than
falling back to anything else.

  aime24         HuggingFaceH4/aime_2024, 30, integer answers (aime_exact_match)
  hmmt25         MathArena/hmmt_feb_2025, 30, LaTeX answers (math_equivalent)
  brumo25        MathArena/brumo_2025, 30
  cmimc25        MathArena/cmimc_2025, 40
  olympiadbench  Hothan/OlympiadBench OE_TO_maths_en_COMP (text-only, English,
                 open-ended maths), 674 - 16 excluded = 658
  omnimath_hard  KbsdJames/Omni-MATH, difficulty >= 6, contaminated removed,
                 empty gold answers removed,
                 then a fixed random 500 (seed 0)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples
from custom_eval.math_scoring import aime_exact_match, aime_integer, math_equivalent, require_math_verify

EXCLUDED = json.loads((Path(__file__).parent / "contamination_excluded.json").read_text())


def _load(path, config, split, expected, cache_dir):
    ds = load_dataset(path, config, split=split, cache_dir=cache_dir) if config else \
        load_dataset(path, split=split, cache_dir=cache_dir)
    if len(ds) != expected:
        raise ValueError(f"Expected {expected} examples from {path}, got {len(ds)}.")
    return ds, {"path": path, "config": config, "split": split,
                "fingerprint": getattr(ds, "_fingerprint", None)}


def _strip_dollars(ans):
    ans = str(ans).strip()
    while ans.startswith("$") and ans.endswith("$") and len(ans) > 1:
        ans = ans[1:-1].strip()
    return ans


def _competition(name, path, expected, scorer, integer=False):
    def load(cache_dir=None, max_samples=None, **_):
        require_math_verify()
        ds, source = _load(path, None, "train", expected, cache_dir)
        drop = set(EXCLUDED.get(name, []))
        examples = []
        for i, row in enumerate(ds):
            if i in drop:
                continue
            gold = str(row["answer"]).strip()
            if integer and aime_integer(gold) is None:
                raise ValueError(f"{name}: non-integer gold answer {gold!r}")
            examples.append(EvalExample(f"{name}/{row.get('id', row.get('problem_idx', i))}", row["problem"], gold,
                                        {"source": source, "type": row.get("problem_type")}))
        return Benchmark(name, limit_examples(examples, max_samples), scorer)
    return load


load_aime24 = _competition("aime24", "HuggingFaceH4/aime_2024", 30, aime_exact_match, integer=True)
load_hmmt25 = _competition("hmmt25", "MathArena/hmmt_feb_2025", 30, math_equivalent)
load_brumo25 = _competition("brumo25", "MathArena/brumo_2025", 30, math_equivalent)
load_cmimc25 = _competition("cmimc25", "MathArena/cmimc_2025", 40, math_equivalent)


def load_olympiadbench(cache_dir=None, max_samples=None, **_):
    require_math_verify()
    ds, source = _load("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train", 674, cache_dir)
    drop = set(EXCLUDED["olympiadbench"])
    examples = []
    for row in ds:
        if str(row["id"]) in drop:
            continue
        if row["modality"] != "Text-only" or row["language"] != "English":
            raise ValueError(f"OlympiadBench item {row['id']} is not text-only English")
        fa = row["final_answer"]
        gold = _strip_dollars(fa[0] if isinstance(fa, list) else fa)
        question = row["question"] if not row.get("context") or row["context"] == "None" else row["context"] + "\n\n" + row["question"]
        examples.append(EvalExample(f"olympiadbench/{row['id']}", question, gold, {
            "source": source, "subfield": row.get("subfield"), "answer_type": row.get("answer_type"),
            "is_multiple_answer": row.get("is_multiple_answer"), "unit": row.get("unit")}))
    return Benchmark("olympiadbench", limit_examples(examples, max_samples), math_equivalent)


def load_omnimath_hard(cache_dir=None, max_samples=None, **_):
    require_math_verify()
    ds, source = _load("KbsdJames/Omni-MATH", None, "test", 4428, cache_dir)
    drop = set(EXCLUDED["omnimath"])
    # Items with an empty gold answer cannot be scored; drop before sampling.
    pool = [i for i, row in enumerate(ds) if float(row["difficulty"]) >= 6 and i not in drop
            and str(row["answer"]).strip()]
    pick = sorted(random.Random(0).sample(pool, 500))
    examples = [EvalExample(f"omnimath/{i}", ds[i]["problem"], str(ds[i]["answer"]).strip(), {
        "source": source, "difficulty": float(ds[i]["difficulty"]), "domain": ds[i]["domain"],
        "origin": ds[i]["source"]}) for i in pick]
    return Benchmark("omnimath_hard", limit_examples(examples, max_samples), math_equivalent)
