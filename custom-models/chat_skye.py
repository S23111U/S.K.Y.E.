from mlx_lm import load, stream_generate
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
MAX_SENTENCES = 5
MAX_WORDS = 120

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
    "acknowledged",
    "farewell",
]


# =========================================================
# TEXT UTILITIES
# =========================================================
def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def suppress_low_info_phrases(text):
    lowered = text.lower()
    for phrase in LOW_INFO_PHRASES:
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
            return "".join(kept).strip()
    return text


def role_integrity_failed(text):
    lower = text.lower()
    if any(tok in lower for tok in FORBIDDEN_ROLE_TOKENS):
        return True
    if re.findall(r"\b[A-Z][a-z]+:\s", text):
        return True
    return False


def repetition_detected(text, history):
    sentences = split_sentences(text)
    if len(sentences) != len(set(sentences)):
        return True
    norm = text.lower().strip()
    return any(norm == prev.lower().strip() for prev in history)


def exceeds_verbosity(text):
    return len(text.split()) > MAX_WORDS or len(split_sentences(text)) > MAX_SENTENCES


def personality_saturated(text):
    words = text.lower().split()
    hits = sum(words.count(tok) for tok in STYLE_TOKENS)
    return hits / max(len(words), 1) > 0.08


# =========================================================
# RUNTIME GUARDRAILS
# =========================================================
def apply_runtime_guardrails(text, history):
    triggered = []

    if role_integrity_failed(text):
        triggered.append("role_integrity_reset")
        return (
            "Apologies, Sir. I'm SKYE. Let's continue—how may I assist you?",
            triggered,
        )

    if repetition_detected(text, history):
        triggered.append("repetition_abort")
        return (
            "Apologies, Sir. Let me respond more clearly and concisely.",
            triggered,
        )

    if exceeds_verbosity(text):
        triggered.append("verbosity_truncate")
        return " ".join(split_sentences(text)[:3]), triggered

    if personality_saturated(text):
        triggered.append("personality_dampen")
        text = re.sub(r"\b(Sir|sir)\b[, ]*", "", text)
        return text.strip(), triggered

    cleaned = suppress_low_info_phrases(text)
    if cleaned != text:
        triggered.append("low_info_suppression")

    return cleaned, triggered


# =========================================================
# CLEANUP & FUNCTION CALL HANDLING
# =========================================================
def clean_response(text):
    text = re.sub(r"<\|end\|>|<\|assistant\|>|<unk>|<s>|</s>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*$", "", text)
    return text.strip()


# =========================================================
# TELEMETRY
# =========================================================
def response_stats(text):
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
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
    adapter_path="adapters_SKYE_V2",
)
print("✓ SKYE ONLINE\n")


# =========================================================
# CHAT LOOP (STREAMING)
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

        assistant_history = [m["content"] for m in messages if m["role"] == "assistant"]

        print("\nSKYE: ", end="", flush=True)

        raw_response = ""

        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=200,
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                raw_response += chunk.text

        print()  # newline after stream

        # ---- FINALIZATION PHASE ----
        guarded, guardrails = apply_runtime_guardrails(raw_response, assistant_history)

        clean = clean_response(guarded)

        # If guardrails changed output, show correction
        if clean.strip() != raw_response.strip():
            print(f"\n[SKYE corrected response]\n{clean}\n")

        # TELEMETRY
        TELEMETRY_LOG.append(
            {
                "session_id": SESSION_ID,
                "turn_index": TURN_INDEX,
                "user_input": user_input,
                "assistant_output": clean,
                "guardrails_triggered": guardrails,
                "response_stats": response_stats(clean),
                "timestamp": datetime.now().isoformat(),
            }
        )

        TURN_INDEX += 1
        messages.append({"role": "assistant", "content": clean})

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
