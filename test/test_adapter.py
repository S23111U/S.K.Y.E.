from mlx_lm import load, generate
import re
import json


def extract_complete_json(text, start_pos):
    """Extract a complete JSON object starting at start_pos, handling nested braces."""
    if start_pos >= len(text):
        return None, None

    # Find the opening brace
    brace_start = text.find("{", start_pos)
    if brace_start == -1:
        return None, None

    # Count braces to find the matching closing brace
    brace_count = 0
    i = brace_start

    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                # Found matching closing brace
                return text[brace_start : i + 1], i + 1
        i += 1

    # If we didn't find a closing brace, try to find one in the next few characters
    # (model might have generated incomplete JSON)
    remaining = text[brace_start : brace_start + 200]  # Look ahead 200 chars
    if "}" in remaining:
        end_pos = remaining.find("}") + 1
        return text[brace_start : brace_start + end_pos], brace_start + end_pos

    return None, None


def clean_response(text):
    """Clean response by removing special tokens and handling repetitive patterns."""
    if not text:
        return ""

    # Remove special tokens
    text = re.sub(r"<\|end\|>", "", text)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<unk>", "", text)
    text = re.sub(r"<s>|</s>", "", text)

    # Remove repetitive function calls - stop after first valid CALL_FUNC
    # Use a more flexible pattern that handles the CALL_FUNC prefix
    call_pattern = re.compile(r"CALL_FUNC\s*[:\-]*\s*", re.IGNORECASE)
    match = call_pattern.search(text)

    if match:
        # Extract text before the call
        before_call = text[: match.start()].strip()

        # Try to extract complete JSON after CALL_FUNC
        json_start = match.end()
        json_str, json_end_pos = extract_complete_json(text, json_start)

        if json_str:
            try:
                json_str = json_str.replace("'", '"')
                func_data = json.loads(json_str)
                if "name" in func_data:
                    # Valid function call
                    call_prefix = text[match.start() : match.end()].strip()
                    complete_call = f"{call_prefix}{json_str}"
                    if before_call:
                        return (before_call + " " + complete_call).strip()
                    return complete_call.strip()
            except:
                pass

        # If we couldn't parse, at least return up to where we found CALL_FUNC
        # The detect_and_stop_at_function_call will handle fixing incomplete JSON
        if json_end_pos:
            return text[:json_end_pos].strip()
        else:
            return text[: match.end()].strip()

    # For non-function-call responses, clean up repetitive patterns
    # Remove repetitive phrases that appear multiple times
    text = re.sub(r"\s+", " ", text).strip()

    # Stop at first occurrence of common ending phrases if they repeat
    ending_phrases = [
        "Operation completed",
        "Task executed successfully",
        "Standing by for your directive",
        "Awaiting your next command",
        "Understood, Sir",
        "Acknowledged",
    ]

    for phrase in ending_phrases:
        if text.count(phrase) > 1:
            # Find first occurrence and stop there
            idx = text.find(phrase)
            if idx > 0:
                text = text[: idx + len(phrase)].strip()
                break

    # Remove any remaining <|assistant|> tokens that might appear mid-text
    text = re.sub(r"<\|assistant\|>", "", text)

    # Fix double commas/spaces
    text = re.sub(r",\s*,", ",", text)  # Fix ", ," -> ","
    text = re.sub(r"\s{2,}", " ", text)  # Fix multiple spaces

    return text.strip()


def detect_and_stop_at_function_call(text):
    """Detect function calls and stop generation at first valid one, handling incomplete JSON."""
    # Find CALL_FUNC pattern
    call_pattern = re.compile(r"CALL_FUNC\s*[:\-]*\s*", re.IGNORECASE)
    match = call_pattern.search(text)

    if match:
        call_prefix = text[match.start() : match.end()].strip()
        json_start = match.end()

        # Try to extract complete JSON starting after CALL_FUNC
        json_str, json_end_pos = extract_complete_json(text, json_start)

        if json_str:
            # Try to validate the JSON
            try:
                json_str = json_str.replace("'", '"')
                func_data = json.loads(json_str)
                if "name" in func_data:
                    # Valid function call found
                    return f"{call_prefix}{json_str}".strip()
            except json.JSONDecodeError:
                pass

        # If we couldn't extract complete JSON, try to extract what we have and fix it
        # Look for JSON-like content after CALL_FUNC
        remaining_text = text[json_start : json_start + 200]  # Look ahead 200 chars

        # Try to find the JSON object (even if incomplete)
        brace_start = remaining_text.find("{")
        if brace_start != -1:
            # Extract everything from the opening brace
            json_candidate = remaining_text[brace_start:]

            # Count braces to see how many are missing
            open_count = json_candidate.count("{")
            close_count = json_candidate.count("}")
            missing_braces = open_count - close_count

            # If we have some JSON structure, try to complete it
            if open_count > 0 and '"name"' in json_candidate:
                # Add missing closing braces
                if missing_braces > 0:
                    json_candidate = json_candidate.rstrip() + "}" * missing_braces

                # Try to parse the completed JSON
                try:
                    json_candidate = json_candidate.replace("'", '"')
                    func_data = json.loads(json_candidate)
                    if "name" in func_data:
                        return f"{call_prefix}{json_candidate}".strip()
                except:
                    # If parsing still fails, return what we have with braces fixed
                    return f"{call_prefix}{json_candidate}".strip()

        # Fallback: return text up to CALL_FUNC + some reasonable amount
        # This handles cases where JSON is very malformed
        return (
            text[: json_start + 100].strip()
            if len(text) > json_start + 100
            else text.strip()
        )

    return text


print("Testing adapter BEFORE fusing...")
model, tokenizer = load(
    "mlx-community/Phi-3-mini-4k-instruct-4bit", adapter_path="adapters_SKYE_V2"
)

# Get EOS token ID for stopping
eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
if eos_token_id is None:
    # Try to get from tokenizer config
    try:
        eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    except:
        eos_token_id = None

# Test prompts with conversation context
test_conversations = [
    [{"role": "user", "content": "who is elon musk?"}],
    [{"role": "user", "content": "what time is it?"}],
    [{"role": "user", "content": "set an alarm for 5pm"}],
    [{"role": "user", "content": "who are you?"}],
    [{"role": "user", "content": "what's your name?"}],
]

for i, messages in enumerate(test_conversations):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate with proper stopping criteria
    try:
        # Try with temperature parameter (MLX uses 'temperature' not 'temp')
        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=150,  # Reduced from 200
            verbose=False,
            temperature=0.7,  # Lower temperature for more focused, less repetitive responses
        )
    except TypeError:
        # Fallback if temperature parameter not supported
        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=150,
            verbose=False,
        )

    # Clean up the response
    clean = clean_response(response)

    # Additional cleanup: stop at first valid function call if present
    clean = detect_and_stop_at_function_call(clean)

    # Final fix: ensure JSON in function calls is properly closed
    if "CALL_FUNC" in clean:
        call_start = clean.find("CALL_FUNC")
        json_part = clean[call_start:]

        # Count braces in the JSON part
        open_braces = json_part.count("{")
        close_braces = json_part.count("}")

        # If we have more opening braces than closing, add the missing ones
        if open_braces > close_braces:
            missing = open_braces - close_braces
            # Remove any trailing whitespace/newlines before adding braces
            clean = clean.rstrip() + "}" * missing

    print(f"\nYou: {messages[-1]['content']}")
    print(f"SKYE: {clean}")
