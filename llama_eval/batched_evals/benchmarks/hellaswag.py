from batched_evals.benchmarks.base import Benchmark, EvalExample
from batched_evals.benchmarks.loaders import limit_examples, try_load_dataset
from batched_evals.scoring import choice_match


LABELS = ["A", "B", "C", "D"]


def load(cache_dir=None, split="validation", max_samples=None, **_):
    ds, source = try_load_dataset(
        [{"path": "Rowan/hellaswag", "splits": [split, "validation", "train"]}],
        cache_dir=cache_dir,
        split=split,
    )
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            endings = row["endings"]
            choices = "\n".join(f"{label}. {ending}" for label, ending in zip(LABELS, endings))
            context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip() or row.get("ctx", "")
            question = f"Choose the most plausible ending.\n\nContext: {context}\n\nChoices:\n{choices}"
            # The public test split ships blank labels — those rows cannot be
            # scored, so drop them rather than crashing on int('').
            raw_label = str(row.get("label", "")).strip()
            if not raw_label:
                continue
            answer = LABELS[int(raw_label)]
            examples.append(
                EvalExample(
                    str(row.get("ind", i)),
                    question,
                    answer,
                    {"source": source},
                    choices=[str(ending) for ending in endings],
                    choice_labels=list(LABELS),
                    bare_question=context,
                    # HellaSwag is a continuation task: score each ending against
                    # the story context directly, as lm-eval does.
                    ll_context=context,
                )
            )
    else:
        examples = [
            EvalExample(
                "hellaswag_fallback_0",
                "Choose the most plausible ending.\n\nContext: A person opens an umbrella because\n\nChoices:\nA. it is raining outside.\nB. the sun turned into music.\nC. they are reading a book.\nD. the floor is asleep.",
                "A",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("hellaswag", limit_examples(examples, max_samples), choice_match)

