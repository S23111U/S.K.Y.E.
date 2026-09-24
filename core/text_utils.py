"""Text-level cleanup shared by the guardrails and the raw-output sanitiser:
splitting into sentences, stripping stock filler phrases, detecting
repetition/role leakage/verbosity/over-saturated "personality" tokens. Pure
functions — no model, no shared engine state — so this is safe to import
from anywhere without pulling in the rest of core_v2.py.
"""

import re

MAX_SENTENCES = 8
MAX_WORDS = 200

STYLE_TOKENS = ["sir", "certainly", "of course", "acknowledged"]

FORBIDDEN_ROLE_TOKENS = [
    "user:",
    "assistant:",
    "system:",
    "tutor:",
    "chatran:",
]

LOW_INFO_PHRASES = [
    "everything is functioning flawlessly",
    "absolutely",
    "that's what i'm here for",
    "consider it handled",
    "farewell",
    "standing by for your directive",
    "all systems are operational",
    "right away",
]

# Intentionally fuzzy to catch hallucinated function calls and broken JSON blocks.
CALL_FUNC_RE = re.compile(
    r"CALL_FUNC\w*\s*:.*",
    re.DOTALL | re.IGNORECASE,
)

# Catch dangling JSON tail fragments
JSON_TAIL_RE = re.compile(r"(\s*[{}]\s*:\s*[{}]\s*){1,}", re.IGNORECASE)

# Standalone status/filler phrases that should always be stripped
ALWAYS_STRIP_PHRASES = [
    r"operation completed",
    r"systems? green and stable",
    r"systems? (?:are )?(?:fully )?operational",
    r"standing by for your directive",
    r"everything is functioning flawlessly",
    r"all systems (?:are )?(?:fully )?online",
    r"all systems functioning within normal parameters",
    r"task executed successfully",
    r"awaiting your next command",
    r"executing now",
    r"on it(?:, sir)?",
    r"as you wish(?:, sir)?",
]


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def suppress_low_info_phrases(text: str) -> str:
    for phrase in LOW_INFO_PHRASES:
        lowered = text.lower()
        if lowered.count(phrase) > 1:
            parts = re.split(rf"({re.escape(phrase)})", text, flags=re.IGNORECASE)
            kept, seen = [], False
            for part in parts:
                if part.lower() == phrase.lower():
                    if not seen:
                        kept.append(part)
                        seen = True
                else:
                    kept.append(part)
            text = "".join(kept).strip()
    return text


def strip_status_filler(text: str) -> str:
    for pattern in ALWAYS_STRIP_PHRASES:
        text = re.sub(
            rf"[,.]?\s*{pattern}\s*[,.]?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\s{2,}", " ", text)
    # NB: '.' is deliberately absent from the *trailing* class. It is still
    # stripped from the front. Leaving it here removed the full stop from the
    # end of every reply.
    text = re.sub(r"^[\s.,;]+|[\s,;]+$", "", text)
    return text.strip()


def role_integrity_failed(text: str) -> bool:
    lower = text.lower()
    if any(tok in lower for tok in FORBIDDEN_ROLE_TOKENS):
        return True
    if len(re.findall(r"\b[A-Z][a-z]+:\s", text)) >= 2:
        return True
    return False


def _normalise_for_comparison(text: str) -> str:
    result = strip_status_filler(text)
    for phrase in LOW_INFO_PHRASES:
        result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip().lower()


def repetition_detected(text: str, history: list[str]) -> bool:
    sentences = split_sentences(text)
    normalised_sentences = [
        _normalise_for_comparison(s) for s in sentences if _normalise_for_comparison(s)
    ]
    if len(normalised_sentences) != len(set(normalised_sentences)):
        return True
    norm = _normalise_for_comparison(text)
    if len(norm) < 20:
        return False
    return any(norm == _normalise_for_comparison(prev) for prev in history)


def exceeds_verbosity(text: str) -> bool:
    return len(text.split()) > MAX_WORDS or len(split_sentences(text)) > MAX_SENTENCES


def personality_saturated(text: str) -> bool:
    words = text.lower().split()
    hits = sum(words.count(tok) for tok in STYLE_TOKENS)
    return hits / max(len(words), 1) > 0.08


def sanitise_raw(text: str) -> str:
    """
    Aggressive pre-guardrail cleanup of raw model output.

    Everything removed here was an artefact of the fine-tuned adapter, which is
    gone. The <draft>/<critique>/<final_answer> strippers went with it; archived
    logs still contain those tags, so ingest_history.py and proposed_facts.py
    keep their own handling. The role-leak chain and trailing-stub strippers
    went too: the persona now handles tone, and both regexes damaged ordinary
    prose ("The reason: it works." lost its clause, and a persona-sanctioned
    trailing "Sir" was cut down to a dangling comma).
    """
    text = re.sub(r"<\|end\|>|<\|assistant\|>|<unk>|<s>|</s>", "", text)
    text = CALL_FUNC_RE.sub("", text)
    text = JSON_TAIL_RE.sub("", text)
    text = strip_status_filler(text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"([\s:;}\]]+)$", "", text)
    return text.strip()
