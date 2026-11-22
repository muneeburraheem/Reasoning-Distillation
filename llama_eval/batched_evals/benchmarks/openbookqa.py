from batched_evals.benchmarks.base import Benchmark, EvalExample
from batched_evals.benchmarks.loaders import limit_examples, try_load_dataset
from batched_evals.scoring import choice_match


def _format_choices(labels, texts):
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def load(cache_dir=None, split="test", max_samples=None, **_):
    ds, source = try_load_dataset(
        [{"path": "allenai/openbookqa", "name": "main", "splits": [split, "test", "validation"]}],
        cache_dir=cache_dir,
        split=split,
    )
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            stem = row["question_stem"]
            question = f"{stem}\n\nChoices:\n{_format_choices(labels, texts)}"
            examples.append(
                EvalExample(
                    str(row.get("id", i)),
                    question,
                    str(row["answerKey"]),
                    {"source": source},
                    choices=list(texts),
                    choice_labels=[str(label) for label in labels],
                    # Same shape as ARC-Challenge: the stem is the likelihood
                    # context and each option text is scored as a continuation.
                    bare_question=stem,
                )
            )
    else:
        examples = [
            EvalExample(
                "openbookqa_fallback_0",
                "The sun is responsible for\n\nChoices:\nA. puppies learning new tricks\nB. children growing up and getting old\nC. flowers wilting in a vase\nD. plants sprouting, blooming and wilting",
                "D",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("openbookqa", limit_examples(examples, max_samples), choice_match)
