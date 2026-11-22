"""Batched generation.

Two batching axes, both active at once:
  1. across questions  — `batch_size` prompts are left-padded and generated
     in a single `model.generate` call;
  2. across pass@k     — sampled candidates come from `num_return_sequences=k`
     in that same call instead of k sequential calls.

Greedy decoding (temperature == 0) with k > 1 would produce k identical
candidates, so the harness generates one greedy candidate and replicates it —
same scores as the sequential harness, without paying k× compute.

CoT runs batch stage-wise: stage 1 (reasoning) for the whole batch, then
stage 2 (final answer) for all batch×k continuations in one call.
"""

import re
from dataclasses import dataclass
from typing import Callable

import torch

from batched_evals.scoring import extract_boxed


FINAL_RE = re.compile(r"final\s*response\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
# Native AM-DeepSeek trace format: <think>reasoning</think><answer>answer</answer>.
# The closing tag is REQUIRED: an unclosed block is a trace that ran out of
# budget mid-answer, which the commitment gate below must not treat as an answer.
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
OPEN_ANSWER_RE = re.compile(r"<answer>", re.IGNORECASE)
CLOSE_ANSWER_RE = re.compile(r"</answer>", re.IGNORECASE)

# Sequences that end a generation early; passed to generate(stop_strings=...).
NATIVE_STOP_STRINGS = ["</answer>"]

PROMPT_STYLES = {"marker", "native"}

# How hard the extractor works to find an answer inside a committed region.
# Both modes apply the commitment gate; they differ only in cue breadth and in
# how much of the committed region is handed to the matcher. `lenient` is the
# validated headline, `strict` the lower bound. See `extract_answer`.
SCORING_MODES = {"strict", "lenient"}


@dataclass
class GeneratedCandidate:
    prompt: str
    raw_output: str
    final_response: str
    # False when the completion never signalled an answer. The runner scores
    # these incorrect WITHOUT consulting the matcher — see `extract_answer`.
    committed: bool = True
    cue: str = "unknown"


@dataclass(frozen=True)
class Extraction:
    """What a completion committed to, and which cue said so."""

    committed: bool
    answer: str
    cue: str


UNCOMMITTED = Extraction(False, "", "uncommitted")

# A cue takes (already-stripped text, mode) and returns an Extraction if it
# fires, or None to fall through to the next cue.
Cue = Callable[[str, str], "Extraction | None"]


def _last_line(region: str) -> str:
    lines = [line.strip() for line in region.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _cue_answer_block(text: str, mode: str) -> Extraction | None:
    """The trained format, closing tag present."""
    match = ANSWER_BLOCK_RE.search(text)
    if not match:
        return None
    region = match.group(1).strip()
    if not region:
        return None
    boxed = extract_boxed(region, require_closed=True)
    if boxed is not None:
        return Extraction(True, boxed.strip(), "answer_block_boxed")
    if mode == "lenient":
        # Hand over the whole block: answer blocks are often a short paragraph
        # whose result sits mid-block or ahead of a trailing sentence, and the
        # matcher's last-number rule can find it anywhere inside.
        return Extraction(True, region, "answer_block_region")
    return Extraction(True, _last_line(region), "answer_block_lastline")


def _cue_boxed(text: str, mode: str) -> Extraction | None:
    """A closed \\boxed{} anywhere in the completion."""
    boxed = extract_boxed(text, require_closed=True)
    if boxed is None:
        return None
    return Extraction(True, boxed.strip(), "boxed")


def _cue_marker(text: str, mode: str) -> Extraction | None:
    """The legacy `final response:` marker asked for by prompt_style="marker"."""
    match = FINAL_RE.search(text)
    if not match:
        return None
    return Extraction(True, match.group(1).strip().splitlines()[0].strip(), "marker")


def _cue_stray_close(text: str, mode: str) -> Extraction | None:
    """A stray </answer> with no opening tag (lenient only).

    hard_label degenerates this way — it emits the closing tag without ever
    opening one, so the text immediately before it is its answer region. Without
    this cue those completions fall through to `uncommitted` and are scored
    wrong even when the reasoning above them is correct and complete (measured:
    ~9.6 pt of GSM8K for hard_label_epoch1).
    """
    if mode != "lenient":
        return None
    close = CLOSE_ANSWER_RE.search(text)
    if not close or OPEN_ANSWER_RE.search(text):
        return None
    region = text[: close.start()].strip()
    if not region:
        return None
    return Extraction(True, region, "stray_close")


# Precedence order — most specific cue first. A completion that both closes an
# <answer> block and contains a \boxed{} elsewhere is read from the block.
ANSWER_CUES: list[Cue] = [
    _cue_answer_block,
    _cue_boxed,
    _cue_marker,
    _cue_stray_close,
]


def register_cue(cue: Cue, *, index: int | None = None) -> None:
    """Add a commitment cue, e.g. when a prompt-format sweep adds an arm.

    The gate only credits formats it has a cue for, so a new prompt arm that
    asks for some other answer marker scores zero by construction until its cue
    is registered here. `index` inserts ahead of the existing cues when the new
    format must win a precedence conflict.
    """
    if index is None:
        ANSWER_CUES.append(cue)
    else:
        ANSWER_CUES.insert(index, cue)


def extract_answer(text: str, mode: str = "lenient") -> Extraction:
    """Extract an answer under a commitment gate.

    THE GATE: a candidate scores only if the completion actually signalled an
    answer — a closed <answer></answer>, a closed \\boxed{}, `final response:`,
    or (lenient) a stray </answer>. With no cue the result is `UNCOMMITTED` and
    the caller must score it incorrect without guessing.

    This replaces an unconditional `return text.splitlines()[-1].strip()`
    fallback. Composed with `scoring.numeric_or_exact_match`, which applies its
    own last-number rule whenever the gold answer is numeric, that fallback
    meant *the last number in a degenerate repetition loop was accepted as the
    model's answer*. It produced every non-zero AIME25 cell in the 2026-08 panel
    and 40-100% of every model's MATH500 hits; base_student's GSM8K went 1.82% →
    0.00% once gated. The policy here is the one validated in
    rescore_2026-08-17/ (permutation null, tail agreement, specificity).

    `scoring.numeric_or_exact_match` is deliberately NOT changed — it is the
    validated component (0/1849 self-match failures). What changes is which
    string reaches it, and whether it is consulted at all.
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"mode must be one of {sorted(SCORING_MODES)} (got {mode!r}).")

    text = (text or "").strip()
    if not text:
        return Extraction(False, "", "empty")

    for cue in ANSWER_CUES:
        extraction = cue(text, mode)
        if extraction is not None:
            return extraction

    return UNCOMMITTED


def extract_final_response(text: str, mode: str = "lenient") -> str:
    """The extracted answer alone; "" when the completion never committed.

    Thin wrapper over `extract_answer` for callers that only want the string.
    Prefer `extract_answer` where the commitment flag matters — an empty string
    happens to score incorrect against every current matcher, but relying on
    that is relying on a coincidence.
    """
    return extract_answer(text, mode).answer


def _make_candidate(prompt: str, raw: str, mode: str) -> GeneratedCandidate:
    extraction = extract_answer(raw, mode)
    return GeneratedCandidate(
        prompt=prompt,
        raw_output=raw,
        final_response=extraction.answer,
        committed=extraction.committed,
        cue=extraction.cue,
    )


def build_prompt(question: str, cot: bool, prompt_style: str = "marker") -> str:
    """Render a question into a prompt.

    prompt_style="native" reproduces the training distribution of the
    AM-DeepSeek-R1-Distilled students: the raw question, a newline, then an open
    <think> tag, exactly as the fine-tuning text was laid out
    (`user_content + "\\n" + assistant_content`). Prefilling <think> costs the
    model nothing it would not have generated itself and is identical for every
    checkpoint, so it advantages no scheme.

    prompt_style="marker" is the original instruction-following format, kept so
    earlier runs stay reproducible.
    """
    if prompt_style == "native":
        return f"{question}\n<think>"
    if cot:
        return (
            f"{question}\n\n"
            "Think through the problem. Do not give the final answer yet."
        )
    return (
        f"{question}\n\n"
        "Return only the final answer on a new line exactly as:\n"
        "final response: {answer}"
    )


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int = 1,
    stop_strings: list[str] | None = None,
) -> list[list[str]]:
    """One generate call for all prompts; returns per-prompt candidate lists.

    Decoder-only models need left padding for batched generation — right
    padding would place pad tokens between the prompt and the continuation.
    """
    do_sample = temperature > 0.0
    effective_k = num_return_sequences if do_sample else 1

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = original_side

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": effective_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if stop_strings:
        # generate() needs the tokenizer to match stop strings against text.
        generation_kwargs["stop_strings"] = stop_strings
        generation_kwargs["tokenizer"] = tokenizer

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
    texts = [t.strip() for t in texts]

    # generate() orders outputs prompt-major: [p0_k0..p0_k-1, p1_k0, ...]
    grouped = [
        texts[i * effective_k : (i + 1) * effective_k] for i in range(len(prompts))
    ]
    if not do_sample and num_return_sequences > 1:
        grouped = [group * num_return_sequences for group in grouped]
    return grouped


def generate_candidates_batch(
    model,
    tokenizer,
    questions: list[str],
    cot: bool,
    max_toks: int | None,
    pass_at_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_style: str = "marker",
    scoring_mode: str = "lenient",
) -> list[list[GeneratedCandidate]]:
    """Generate pass@k candidates for a batch of questions.

    Returns one candidate list per question, in input order.
    """
    prompts = [build_prompt(q, cot, prompt_style) for q in questions]

    if prompt_style == "native":
        # The trained format already interleaves reasoning and answer in one
        # completion (<think>…</think><answer>…</answer>), so there is no second
        # stage to run — max_new_tokens must cover the whole trace, and
        # generation stops as soon as </answer> is emitted.
        grouped = _generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=pass_at_k,
            stop_strings=NATIVE_STOP_STRINGS,
        )
        return [
            [_make_candidate(prompt, raw, scoring_mode) for raw in candidates]
            for prompt, candidates in zip(prompts, grouped)
        ]

    if not cot:
        grouped = _generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=pass_at_k,
        )
        return [
            [_make_candidate(prompt, raw, scoring_mode) for raw in candidates]
            for prompt, candidates in zip(prompts, grouped)
        ]

    # CoT stage 1: k reasoning traces per question, batched together.
    reasoning_grouped = _generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_toks,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=pass_at_k,
    )

    # CoT stage 2: every (question, reasoning) pair becomes one final-answer
    # prompt; all batch×k of them run in a single call.
    final_prompts = []
    for question, reasonings in zip(questions, reasoning_grouped):
        for reasoning in reasonings:
            final_prompts.append(
                f"{question}\n\nReasoning:\n{reasoning}\n\n"
                "Now return only the final answer on a new line exactly as:\n"
                "final response: {answer}"
            )
    # Chunk stage 2 to batch_size-sized calls so peak memory stays comparable
    # to stage 1 (stage-2 prompts are batch×k and carry the reasoning text).
    chunk = max(len(questions), 1)
    final_texts = []
    for start in range(0, len(final_prompts), chunk):
        final_grouped = _generate_batch(
            model,
            tokenizer,
            final_prompts[start : start + chunk],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )
        final_texts.extend(group[0] for group in final_grouped)

    results = []
    flat_index = 0
    for question, prompt, reasonings in zip(questions, prompts, reasoning_grouped):
        candidates = []
        for reasoning in reasonings:
            final_text = final_texts[flat_index]
            raw_output = f"{reasoning}\n{final_text}".strip()
            candidates.append(
                _make_candidate(
                    f"{prompt}\n\n--- final prompt ---\n{final_prompts[flat_index]}",
                    raw_output,
                    scoring_mode,
                )
            )
            flat_index += 1
        results.append(candidates)
    return results
