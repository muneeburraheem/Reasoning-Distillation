from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalExample:
    id: str
    question: str
    answer: str
    metadata: dict
    # Populated by multiple-choice benchmarks so they can also be scored by
    # log-likelihood ranking instead of by generating text. `question` keeps the
    # rendered "Choices:" block for generative scoring; `bare_question` is the
    # stem alone, which is what the likelihood context uses.
    choices: list[str] | None = None
    choice_labels: list[str] | None = None
    bare_question: str | None = None
    # Verbatim likelihood context, for benchmarks that are a plain continuation
    # rather than a question (HellaSwag scores endings against the story so far,
    # so wrapping it in "Question:/Answer:" would be wrong). When None the
    # default Question/Answer template is applied to bare_question.
    ll_context: str | None = None
    # Per-choice likelihood context and continuation, for benchmarks where the
    # options vary the *context* rather than the continuation. WinoGrande is
    # scored this way ("partial evaluation" in lm-eval): each option is
    # substituted into the blank and the shared sentence tail is scored against
    # it, so every option gets its own context and they share a continuation.
    # When set these win over ll_context / choices for pairing only; `choices`
    # still holds the human-readable options for display and generative scoring.
    ll_contexts: list[str] | None = None
    ll_continuations: list[str] | None = None

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices and self.choice_labels)


@dataclass
class Benchmark:
    name: str
    examples: list[EvalExample]
    scorer: Callable[[str, str], bool]

