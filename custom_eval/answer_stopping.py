"""Stop generation once a final answer is complete, outside any thinking block.

Extracted from september-eval-scripts/gsm8k_generation_length.py so the eval
harness and the generation-length script share one definition of "finished".
"""

import re

import torch


def final_answer_boundary(text, thinking):
    """First nonempty answer block or balanced box outside a thinking block.

    This is a format heuristic, not a correctness judgment. An opening <think>
    may already be in the prompt; thinking=True accounts for that case.

    Returns (end_offset, kind) or None. An unbalanced or empty box never
    matches, so a clipped final box cannot be mistaken for an answer.
    """
    in_think = thinking
    answer_start = None
    for match in re.finditer(r'<(/?)(think|answer)\s*>|\\boxed\s*\{', text, re.IGNORECASE):
        if match.group(2):
            closing, tag = match.group(1), match.group(2).lower()
            if tag == 'think':
                in_think = not bool(closing)
                answer_start = None
            elif not in_think:
                if not closing:
                    answer_start = match.end()
                elif answer_start is not None:
                    content = text[answer_start:match.start()].strip()
                    answer_start = None
                    if content and content.lower() != 'answer':
                        return match.end(), 'answer_tag'
            continue
        if in_think or answer_start is not None:
            continue
        depth = 1
        escaped = False
        for i in range(match.end(), len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    content = text[match.end():i].strip()
                    if content and content.lower() != 'answer':
                        return i + 1, 'boxed'
                    break
    return None


class FinalAnswerStopper:
    """Per-row stopping callback; never halt other unfinished batch members.

    Full text is decoded only when the newest token can close a tag or box.
    Parsing completion-only text prevents matches against prompt examples.
    Store actual stop lengths because HF pads stopped rows until the batch ends.
    """

    def __init__(self, tokenizer, input_width, batch_size, thinking, eos_token_ids):
        self.tokenizer = tokenizer
        self.input_width = input_width
        self.thinking = thinking
        self.eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
        self.stop_lengths = [None] * batch_size
        self.stop_kinds = [None] * batch_size
        self.eos_seen = [False] * batch_size
        self.token_text = {}

    def should_check(self, row, token):
        if self.stop_lengths[row] is not None or self.eos_seen[row]:
            return False
        if token in self.eos:
            self.eos_seen[row] = True
            return False
        if token not in self.token_text:
            self.token_text[token] = self.tokenizer.decode(
                [token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # These ASCII characters must be present in the token completing a marker.
        return '>' in self.token_text[token] or '}' in self.token_text[token]

    def observe(self, row, generated):
        text = self.tokenizer.decode(generated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        boundary = final_answer_boundary(text, self.thinking)
        if boundary is not None and self.stop_lengths[row] is None:
            self.stop_lengths[row] = len(generated)
            self.stop_kinds[row] = boundary[1]

    def __call__(self, input_ids, scores=None, **kwargs):
        latest = input_ids[:, -1].tolist()
        for row, token in enumerate(latest):
            if self.should_check(row, token):
                self.observe(row, input_ids[row, self.input_width:].tolist())
        return torch.tensor([n is not None for n in self.stop_lengths],
                            device=input_ids.device, dtype=torch.bool)
