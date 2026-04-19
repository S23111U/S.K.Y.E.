from mlx_lm import load, generate
import re
import json
from datetime import datetime

# =========================================================
# SESSION / TELEMETRY STATE
# =========================================================
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TURN_INDEX = 0
TELEMETRY_LOG = []

# =========================================================
# GUARDRAIL CONFIG
# =========================================================
# Give SKYE enough room to answer properly before truncating.
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

# Phrases that are pure filler / low-information; suppress any repetition.
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

# Regex to strip any CALL_FUNC / CALL_FUNCT artifact that leaks into the final text.
# Intentionally fuzzy — catches CALL_FUNC, CALL_FUNCT, CALL_FUNCTION, etc.
# Uses .* so it strips everything to the end, even if the JSON is cut off by the token limit.
CALL_FUNC_RE = re.compile(
    r"CALL_FUNC\w*\s*:.*",
    re.DOTALL | re.IGNORECASE,
)

# Catch dangling JSON tail fragments like: } : } : }
JSON_TAIL_RE = re.compile(r"(\s*[{}]\s*:\s*[{}]\s*){1,}", re.IGNORECASE)

# Role-play / dialog leak patterns: ': Certainly. : Right away. :'
ROLE_LEAK_CHAIN_RE = re.compile(
    r"(\s*:\s+[A-Za-z ,.'!?-]{1,60}[.!?]){1,}",
)

# Standalone status/filler phrases that should always be stripped from responses.
# These are appended by the model's trained persona but carry zero information.
ALWAYS_STRIP_PHRASES = [
    r"operation completed",
    r"systems? green and stable",
    r"systems? (?:are )?(?:fully )?operational",
    r"standing by for your directive",
    r"everything is functioning flawlessly",
    r"all systems (?:are )?(?:fully )?online",
    r"all systems functioning within normal parameters",
]

# =========================================================
# TEXT UTILITIES
# =========================================================
def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def suppress_low_info_phrases(text: str) -> str:
    """Keep the first occurrence of each low-info phrase; remove duplicates."""
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
                    # else: silently drop the duplicate
                else:
                    kept.append(part)
            text = "".join(kept).strip()
    return text


def strip_status_filler(text: str) -> str:
    """Always strip trailing/embedded status phrases that add no information."""
    for pattern in ALWAYS_STRIP_PHRASES:
        text = re.sub(
            rf"[,.]?\s*{pattern}\s*[,.]?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    # Clean up any leftover punctuation artifacts
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[\s.,;]+|[\s.,;]+$", "", text)
    return text.strip()


def role_integrity_failed(text: str) -> bool:
    lower = text.lower()
    if any(tok in lower for tok in FORBIDDEN_ROLE_TOKENS):
        return True
    # Detect simulated dialog turns like "Name: ..." repeated ≥2 times
    if len(re.findall(r"\b[A-Z][a-z]+:\s", text)) >= 2:
        return True
    return False


def _normalise_for_comparison(text: str) -> str:
    """Strip filler/status phrases before comparing for repetition."""
    result = strip_status_filler(text)
    for phrase in LOW_INFO_PHRASES:
        result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip().lower()


def repetition_detected(text: str, history: list[str]) -> bool:
    sentences = split_sentences(text)
    # Intra-response sentence duplication (ignore filler)
    normalised_sentences = [_normalise_for_comparison(s) for s in sentences if _normalise_for_comparison(s)]
    if len(normalised_sentences) != len(set(normalised_sentences)):
        return True
    # Cross-turn repetition: only flag if the *meaningful* content matches exactly
    norm = _normalise_for_comparison(text)
    if len(norm) < 20:          # too short to be meaningful — don't flag
        return False
    return any(norm == _normalise_for_comparison(prev) for prev in history)


def exceeds_verbosity(text: str) -> bool:
    return (
        len(text.split()) > MAX_WORDS
        or len(split_sentences(text)) > MAX_SENTENCES
    )


def personality_saturated(text: str) -> bool:
    words = text.lower().split()
    hits = sum(words.count(tok) for tok in STYLE_TOKENS)
    return hits / max(len(words), 1) > 0.08


# =========================================================
# PRE-GUARDRAIL SANITISATION
# =========================================================
def sanitise_raw(text: str) -> str:
    """
    Remove clear model artifacts before any guardrail evaluation:
      - Internal logic tags
      - Special tokens
      - CALL_FUNC bleed
      - Role-leak repetition patterns (': Certainly. : Right away. ...')
    """
    text = re.sub(r"<draft>.*?</draft>", "", text, flags=re.DOTALL)
    text = re.sub(r"<critique>.*?</critique>", "", text, flags=re.DOTALL)
    text = re.sub(r"<final_answer>|</final_answer>", "", text)
    
    # 1. Strip special tokens
    text = re.sub(r"<\|end\|>|<\|assistant\|>|<unk>|<s>|</s>", "", text)

    # 2. Remove CALL_FUNC/CALL_FUNCT/CALL_FUNCTION blocks (greedy on inner JSON)
    text = CALL_FUNC_RE.sub("", text)

    # 3. Remove dangling JSON tail fragments like '} : } : }'
    text = JSON_TAIL_RE.sub("", text)

    # 4. Remove role-leak chains: ': That would be 4. : Certainly. : ...'
    text = ROLE_LEAK_CHAIN_RE.sub("", text)

    # 5. Strip always-remove status filler phrases
    text = strip_status_filler(text)

    # 6. Collapse whitespace artifacts
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)

    # 7. Strip trailing colons, braces, or spaces from cut-off generations
    text = re.sub(r"([\s:;.}\]]+)$", "", text)

    return text.strip()


# =========================================================
# RUNTIME GUARDRAILS
# =========================================================
def apply_runtime_guardrails(text: str, history: list[str]):
    triggered = []

    if role_integrity_failed(text):
        triggered.append("role_integrity_reset")
        return (
            "Apologies. I'm SKYE — how may I assist you?",
            triggered,
        )

    if repetition_detected(text, history):
        triggered.append("repetition_abort")
        return (
            "Apologies. Let me respond more clearly and concisely.",
            triggered,
        )

    if exceeds_verbosity(text):
        triggered.append("verbosity_truncate")
        truncated = " ".join(split_sentences(text)[:MAX_SENTENCES])
        return truncated, triggered

    if personality_saturated(text):
        triggered.append("personality_dampen")
        text = re.sub(r"\b(Sir|sir)\b[, ]*", "", text)
        text = text.strip()

    cleaned = suppress_low_info_phrases(text)
    if cleaned != text:
        triggered.append("low_info_suppression")

    return cleaned, triggered


# =========================================================
# TELEMETRY
# =========================================================
def response_stats(text: str) -> dict:
    return {
        "word_count": len(text.split()),
        "sentence_count": len(split_sentences(text)),
        "used_call_func": "CALL_FUNC" in text,
    }


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading SKYE with adapters...")
model, tokenizer = load(
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    adapter_path="SKYE",
)
print("✓ SKYE ONLINE\n")


# =========================================================
# CHAT LOOP (NON-STREAMING)
# =========================================================
def chat():
    global TURN_INDEX
    messages = []

    print("S.K.Y.E. ONLINE. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        assistant_history = [
            m["content"] for m in messages if m["role"] == "assistant"
        ]

        # ---- NON-STREAMING GENERATION ----
        raw_response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False,
        ).split("<|eot_id|>")[0].strip()

        # ---- SANITISE → GUARDRAILS → DISPLAY ----
        sanitised = sanitise_raw(raw_response)
        guarded, guardrails = apply_runtime_guardrails(sanitised, assistant_history)
        final = guarded.strip()

        print(f"\nSKYE: {final}\n")

        # TELEMETRY
        TELEMETRY_LOG.append(
            {
                "session_id": SESSION_ID,
                "turn_index": TURN_INDEX,
                "user_input": user_input,
                "assistant_output": final,
                "guardrails_triggered": guardrails,
                "response_stats": response_stats(final),
                "timestamp": datetime.now().isoformat(),
            }
        )

        TURN_INDEX += 1
        messages.append({"role": "assistant", "content": final})

    # Save telemetry
    with open(f"logs/telemetry_{SESSION_ID}.jsonl", "w") as f:
        for event in TELEMETRY_LOG:
            f.write(json.dumps(event) + "\n")

    print("\nTelemetry saved. Systems OFFLINE.\n")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    chat()
