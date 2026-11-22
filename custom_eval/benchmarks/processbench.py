"""ProcessBench (Zheng et al. 2024, Qwen/ProcessBench): find the earliest erroneous step.

Each item is a problem plus a step-by-step solution; the gold label is the index of the
first wrong paragraph, or -1 if the solution is fully correct. The prompt is the paper's
official critique template. Subset (RUN_LOG 8s): all 400 GSM8K items and a fixed random
400 (seed 0) from each of MATH, OlympiadBench and Omni-MATH = 1,600 items.
Scoring: exact match of the boxed integer index. The paper's headline metric is F1, the
harmonic mean of accuracy on erroneous and on correct solutions (computed in analysis).
"""

import random
import re

from datasets import load_dataset

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples

TEMPLATE = """The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}."""
SIZES = {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}
PER_SPLIT = 400


def index_match(prediction, answer):
    m = re.fullmatch(r"\s*(-?\d+)\s*", str(prediction or ""))
    return bool(m) and int(m.group(1)) == int(answer)


def load(cache_dir=None, max_samples=None, **_):
    examples = []
    for split, n in SIZES.items():
        ds = load_dataset("Qwen/ProcessBench", split=split, cache_dir=cache_dir)
        if len(ds) != n:
            raise ValueError(f"ProcessBench {split}: expected {n}, got {len(ds)}")
        pick = sorted(random.Random(0).sample(range(n), PER_SPLIT)) if n > PER_SPLIT else list(range(n))
        for i in pick:
            row = ds[i]
            tagged = "\n\n".join(f"<paragraph_{k}>\n{s}\n</paragraph_{k}>" for k, s in enumerate(row["steps"]))
            examples.append(EvalExample(
                f"processbench/{row['id']}", TEMPLATE.format(problem=row["problem"], tagged_response=tagged),
                str(row["label"]),
                {"source": {"path": "Qwen/ProcessBench", "split": split}, "split": split,
                 "label": int(row["label"]), "n_steps": len(row["steps"]), "generator": row["generator"]}))
    return Benchmark("processbench", limit_examples(examples, max_samples), index_match)
