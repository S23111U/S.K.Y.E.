"""Hard-coded Python fail-safes on length, tone and repetition, run once over
the model's full reply. Built to compensate for a fine-tune that has since
been removed; which of these can still fire is under review (see README) —
the caller logs every one that does."""

import re

from core.text_utils import (
    MAX_SENTENCES, exceeds_verbosity, personality_saturated,
    repetition_detected, role_integrity_failed, split_sentences,
    suppress_low_info_phrases,
)


def apply_runtime_guardrails(text: str, history: list[str]):
    triggered = []

    if role_integrity_failed(text):
        triggered.append("role_integrity_reset")
        return "Apologies. I'm SKYE — how may I assist you?", triggered

    if repetition_detected(text, history):
        triggered.append("repetition_abort")
        return "Apologies. Let me respond more clearly and concisely.", triggered

    if exceeds_verbosity(text):
        triggered.append("verbosity_truncate")
        truncated = " ".join(split_sentences(text)[:MAX_SENTENCES])
        return truncated, triggered

    if personality_saturated(text):
        triggered.append("personality_dampen")
        text = re.sub(r"\b(Sir|sir)\b[, ]*", "", text).strip()

    cleaned = suppress_low_info_phrases(text)
    if cleaned != text:
        triggered.append("low_info_suppression")

    return cleaned, triggered
