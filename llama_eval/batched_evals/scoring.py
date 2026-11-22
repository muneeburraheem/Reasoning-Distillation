import math
import re
from fractions import Fraction


# NOTE: a naive r"\\boxed\{([^{}]+)\}" cannot match nested braces, so every
# \boxed{\frac{1}{2}} / \boxed{x^{2}} silently failed to extract. The numeric
# fallback then compared the first number found anywhere in the string, which
# both passed wrong answers (\frac{1}{2} "matched" gold \frac{1}{3}, since both
# start with 1) and failed right ones. Brace matching is done by scanning.
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FRAC_RE = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")

# LaTeX noise that carries no mathematical content.
LATEX_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\[,;:!]"), ""),
    (re.compile(r"\\dfrac|\\tfrac"), r"\\frac"),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mbox\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\^\{?\\circ\}?"), ""),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"\\\$|\$"), ""),
    (re.compile(r"\\ "), " "),
    (re.compile(r"\{,\}"), ""),  # LaTeX thousands separator: 1{,}000
]


def extract_boxed(text: str, *, require_closed: bool = False) -> str | None:
    """Return the content of the last \\boxed{...}, honouring nested braces.

    By default a truncated `\\boxed{` with no closing brace still yields its
    tail, so a cut-off trace gives up whatever it had written. That is the right
    behaviour for normalisation, and exactly the wrong one for a commitment cue:
    a model that ran out of budget mid-box never committed to an answer. Pass
    require_closed=True to get None in that case.
    """
    key = "\\boxed"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    while i < len(text) and text[i] != "{":
        if not text[i].isspace():
            return None
        i += 1
    if i >= len(text):
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    # Unclosed brace (truncated generation): take the rest, unless the caller is
    # using this as a commitment cue.
    return None if require_closed else text[i + 1 :]


def normalize_text(text: str) -> str:
    text = str(text).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed
    for pattern, replacement in LATEX_STRIP:
        text = pattern.sub(replacement, text)
    text = text.lower().strip()
    text = re.sub(r"^(final\s*response\s*:)", "", text).strip()
    text = re.sub(r"^(the\s+)?answer\s*(is)?\s*:?\s*", "", text).strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(" .")


def normalize_choice(text: str) -> str:
    text = normalize_text(text)
    match = re.search(r"\b([a-e])\b", text)
    if match:
        return match.group(1).upper()
    if text and text[0] in "abcde":
        return text[0].upper()
    return text.upper()


def _parse_value(text: str):
    """Parse a *complete* answer string to a number, or None.

    Strict: the whole string must be the value. This is what makes comparison
    trustworthy — no grabbing a stray digit out of surrounding prose.
    """
    text = text.strip()
    if not text:
        return None

    frac = FRAC_RE.fullmatch(text)
    if frac:
        try:
            num = _parse_value(frac.group(1))
            den = _parse_value(frac.group(2))
            if num is not None and den not in (None, 0):
                return num / den
        except Exception:
            return None
        return None

    try:
        return float(Fraction(text))  # ints, decimals, "3/4"
    except Exception:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _last_number(text: str):
    """Last number in a string — the standard GSM8K-style extraction.

    Used only when the gold answer is itself a plain number, so a prose answer
    like "so she makes $18" still scores. Taking the *last* number matters:
    the first one is usually an input from the question restated.
    """
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _numeric_value(text: str):
    """Backwards-compatible helper: strict parse, else last number."""
    normalized = normalize_text(text)
    value = _parse_value(normalized)
    return value if value is not None else _last_number(normalized)


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_text(prediction) == normalize_text(answer)


def numeric_or_exact_match(prediction: str, answer: str) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    if pred == gold:
        return True

    pred_val = _parse_value(pred)
    gold_val = _parse_value(gold)
    if pred_val is not None and gold_val is not None:
        return math.isclose(pred_val, gold_val, rel_tol=1e-6, abs_tol=1e-6)

    # Gold is numeric but the prediction is prose: fall back to its last number.
    if gold_val is not None:
        loose = _last_number(pred)
        if loose is not None:
            return math.isclose(loose, gold_val, rel_tol=1e-6, abs_tol=1e-6)

    return False


def choice_match(prediction: str, answer: str) -> bool:
    return normalize_choice(prediction) == normalize_choice(answer)
