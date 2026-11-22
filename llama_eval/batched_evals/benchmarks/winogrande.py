"""WinoGrande, scored the way lm-evaluation-harness scores it.

WinoGrande is not an ordinary multiple-choice task. Each sentence contains a
blank (`_`) and two candidate fillers; the question is which filler makes the
rest of the sentence coherent. lm-eval's "partial evaluation" substitutes each
option into the blank and scores the *shared tail* of the sentence against it:

    sentence : "Sarah was a much better surgeon than Maria so _ always got the
                easier cases."
    option 1 : "Sarah"   ->  context "…than Maria so Sarah",  cont " always got…"
    option 2 : "Maria"   ->  context "…than Maria so Maria",  cont " always got…"

So the option varies the context and the continuation is common — the reverse of
ARC-C/OpenBookQA, which is why this loader sets `ll_contexts`/`ll_continuations`
rather than relying on `choices`. Because both continuations are the same
string, length normalisation cancels and acc_norm == acc here by construction;
lm-eval reports plain `acc` for this task and either column is the same number.

Random baseline is 50%, not 25% — it is a two-way choice.
"""

from batched_evals.benchmarks.base import Benchmark, EvalExample
from batched_evals.benchmarks.loaders import limit_examples, try_load_dataset
from batched_evals.scoring import choice_match


LABELS = ["1", "2"]


def load(cache_dir=None, split="validation", max_samples=None, **_):
    ds, source = try_load_dataset(
        [
            {"path": "allenai/winogrande", "name": "winogrande_xl", "splits": [split, "validation", "train"]},
            {"path": "winogrande", "name": "winogrande_xl", "splits": [split, "validation", "train"]},
        ],
        cache_dir=cache_dir,
        split=split,
    )
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            sentence = row["sentence"]
            options = [str(row["option1"]), str(row["option2"])]
            answer = str(row["answer"]).strip()
            # Rows with no gold label (the unlabelled test split) cannot be scored.
            if answer not in LABELS or "_" not in sentence:
                continue
            index = sentence.index("_")
            prefix, suffix = sentence[:index], sentence[index + 1 :]
            if not suffix.strip():
                continue
            choices_block = "\n".join(
                f"{label}. {opt}" for label, opt in zip(LABELS, options)
            )
            question = (
                f"Fill in the blank.\n\nSentence: {sentence}\n\nChoices:\n{choices_block}"
            )
            examples.append(
                EvalExample(
                    str(row.get("qID", i)),
                    question,
                    answer,
                    {"source": source},
                    choices=options,
                    choice_labels=list(LABELS),
                    bare_question=sentence,
                    ll_contexts=[f"{prefix}{opt}" for opt in options],
                    ll_continuations=[suffix, suffix],
                )
            )
    else:
        examples = [
            EvalExample(
                "winogrande_fallback_0",
                "Fill in the blank.\n\nSentence: Sarah was a much better surgeon than Maria so _ always got the easier cases.\n\nChoices:\n1. Sarah\n2. Maria",
                "2",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("winogrande", limit_examples(examples, max_samples), choice_match)
