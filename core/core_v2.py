import os
import sys
import re
import json
import socket
import threading
import time
import webbrowser
import wikipedia
import asyncio
from datetime import datetime
from mlx_lm import load, stream_generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from helper_functions.current_time import TellTime
from helper_functions.weather import Get_Info
from helper_functions.set_alarm import set_alarm
from helper_functions.set_reminder import set_reminder
from helper_functions.GenAI import GenAI_search
from helper_functions.greet import Greetings
from helper_functions.news import fetch_news_summary
from clients.browser import start_http_server, open_browser, start_ws_server
from core.protocol import frame

# =========================================================
# SYSTEM CONFIG & PATHS
# =========================================================
HOST, PORT = "0.0.0.0", 12345
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

PERSONA = open(os.path.join(ROOT, "prompts", "skye_persona.txt")).read()

# Safety: functions that must never be executed
SAFETY_BLOCKLIST = {
    "delete_files",
    "shutdown",
    "reboot",
    "format_drive",
    "hack_into_server",
    "hack",
}

# =========================================================
# SESSION / TELEMETRY STATE
# =========================================================
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TURN_INDEX = 0
TELEMETRY_FILE = os.path.join(LOG_DIR, f"telemetry_{SESSION_ID}.jsonl")
MODEL_LOCK = threading.Lock()


# =========================================================
# LONG-TERM MEMORY
# =========================================================
from memory.manager import MemoryManager

MEMORY = MemoryManager(ROOT)

# =========================================================
# GUARDRAIL CONFIG
# =========================================================
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

# Detect actual execution payloads (both CALL_FUNC and 'Executing function' from older logs)
CALL_EXEC_PATTERN = re.compile(
    r"(CALL_FUNC|Executing function)\w*\s*[:\-]*\s*(.*)", re.DOTALL | re.IGNORECASE
)


# =========================================================
# TEXT UTILITIES & GUARDRAILS
# =========================================================
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


# =========================================================
# RAW OUTPUT SANITISATION
# =========================================================
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


# =========================================================
# RUNTIME GUARDRAILS
# =========================================================
def apply_runtime_guardrails(text: str, history: list[str]):
    """
    Hard-coded Python fail-safes on length, tone and repetition.

    Built to compensate for the fine-tune that has since been removed. Which of
    these can still fire is under review; the caller logs every one that does.
    """
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


# =========================================================
# TELEMETRY
# =========================================================
def save_telemetry(
    user_input, final_output, guardrails, call_func_used, tool_name=None, tool_args=None
):
    global TURN_INDEX
    event = {
        "session_id": SESSION_ID,
        "turn_index": TURN_INDEX,
        "user_input": user_input,
        "assistant_output": final_output,
        "guardrails_triggered": guardrails,
        "response_stats": {
            "word_count": len(final_output.split()),
            "sentence_count": len(split_sentences(final_output)),
            "used_call_func": call_func_used,
            "tool_dispatched": tool_name,
            "tool_args": tool_args,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(TELEMETRY_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    TURN_INDEX += 1


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading SKYE...")
with MODEL_LOCK:
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    tokenizer.eos_token_ids = {
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    }
print("✓ SKYE ONLINE\n")


# =========================================================
# TOOL REGISTRY
# =========================================================
TOOLS = {}


def register_tool(name, func):
    TOOLS[name] = func


register_tool("tell_time", lambda **kw: TellTime())
register_tool(
    "get_weather",
    lambda **kw: Get_Info()[1] + ": " + str(Get_Info()[2]) + "°C, " + Get_Info()[3],
)
register_tool(
    "set_alarm",
    lambda **kw: threading.Thread(target=set_alarm, args=(kw.get("time"),), daemon=True)
    or f"Alarm set for {kw.get('time')}.",
)
register_tool(
    "set_reminder",
    lambda **kw: threading.Thread(
        target=set_reminder, args=(kw.get("time"), kw.get("task")), daemon=True
    )
    or f"Reminder set: {kw.get('task')} at {kw.get('time')}.",
)
register_tool("web_search", lambda **kw: wikipedia_summary_or_genai(kw.get("query")))
register_tool("open_youtube", lambda **kw: web_open_and_ack("youtube", kw.get("query")))
register_tool("open_spotify", lambda **kw: web_open_and_ack("spotify", kw.get("query")))
register_tool("open_calendar", lambda **kw: web_open_and_ack("google", "calendar"))
register_tool("fetch_news", lambda **kw: fetch_news_summary(kw.get("query")))


def wikipedia_summary_or_genai(query):
    if not query:
        return "No query provided."
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return GenAI_search(query)


def web_open_and_ack(site_key, query=None):
    sites = {
        "youtube": "https://youtube.com",
        "wikipedia": "https://wikipedia.com",
        "google": "https://google.com",
        "spotify": "https://open.spotify.com",
    }
    url = sites.get(site_key, "https://google.com")
    if query:
        if site_key == "youtube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"Opening YouTube for {query}."
        if site_key == "spotify":
            webbrowser.open(f"https://open.spotify.com/search/{query}")
            return f"Opening Spotify for {query}."
    webbrowser.open(url)
    return f"Opening {site_key}."


# =========================================================
# FUNCTION-CALL PARSER
# =========================================================
def normalize_json_text(s):
    s = s.replace("'", '"')
    s = s.replace("params", "arguments")
    s = re.sub(r"\}\s*[\.\,]+\s*$", "}", s)
    try:
        stack = 0
        start = None
        for i, ch in enumerate(s):
            if ch == "{":
                if start is None:
                    start = i
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0 and start is not None:
                    return s[start : i + 1]
    except Exception:
        pass
    return s


def extract_function_call(raw_text):
    if not raw_text:
        return None, None, raw_text
    matches = list(CALL_EXEC_PATTERN.finditer(raw_text))
    if not matches:
        return None, None, raw_text

    for m in matches:
        narration = raw_text[: m.start()].strip()
        json_part = normalize_json_text(m.group(2))
        try:
            payload = json.loads(json_part)
            name = payload.get("name") or payload.get("func") or payload.get("function")
            if not name or not isinstance(name, str):
                continue
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            return name, arguments, narration
        except Exception as e:
            continue
    return None, None, raw_text


def call_function_safe(name, args):
    if not name:
        return "Apologies, Sir. Could not detect a function."
    if name in SAFETY_BLOCKLIST:
        return f"Negative, Sir. `{name}` is not authorized."
    fn = TOOLS.get(name)
    if not fn:
        return f"Apologies, Sir. I lack the tool `{name}`."
    try:
        result = fn(**args) if callable(fn) else fn
        return result if result is not None else "Executed."
    except Exception as e:
        print(f"\n[Tool Execution Error ({name})]: {e}")
        return f"Apologies, Sir. `{name}` failed."


# =========================================================
# CORE NLP MASTER LOGIC
# =========================================================
# CONVERSATION STATE
# Shared chat history across an active session (for CLI or single-user Socket)
SHARED_MESSAGES = []
LAST_TOOL_RESULT = ""


def stream_skye_response(user_input: str):
    """Yields protocol frames as the reply is generated.

    Two phases. The first ~40 characters are buffered without being emitted,
    because a reply that begins with CALL_FUNC is a tool call and must never be
    typed at the user as raw JSON. Once the buffer proves it is ordinary prose,
    it is flushed and every later chunk streams live.

    Guardrails need the whole response, so they run once at the end. The `done`
    frame carries that cleaned text; clients replace the streamed tokens with it.
    """
    global SHARED_MESSAGES, LAST_TOOL_RESULT

    _t = {}
    _t["start"] = time.time()

    # Fast path: Rules
    rule_reply = rule_based_response(user_input)
    if rule_reply:
        save_telemetry(user_input, rule_reply, ["rule_bypass"], False)
        yield frame("done", text=rule_reply)
        return

    # Format user prompt, injecting past tool data if present
    if LAST_TOOL_RESULT:
        augmented_input = f"[System Note: Tool execution returned: {LAST_TOOL_RESULT}]\n\n{user_input}"
        LAST_TOOL_RESULT = ""
    else:
        augmented_input = user_input

    # Retrieve the persistent profile and any relevant semantic memories
    profile = MEMORY.get_persistent_profile()
    memories = MEMORY.search(user_input, top_k=3)
    _t["memory"] = time.time()

    system_text = PERSONA
    if profile:
        profile_str = json.dumps(profile, ensure_ascii=False)
        system_text += f"\n[User Profile Data]: {profile_str}"

    if memories:
        memories_str = "\n".join([f"- {m}" for m in memories])
        system_text += f"\n[Relevant Past Memories]:\n{memories_str}"

    # Ensure a fresh system message is at the top or update existing
    if not SHARED_MESSAGES or SHARED_MESSAGES[0]["role"] != "system":
        SHARED_MESSAGES.insert(0, {"role": "system", "content": system_text})
    else:
        SHARED_MESSAGES[0]["content"] = system_text

    SHARED_MESSAGES.append({"role": "user", "content": augmented_input})

    # Keep system message + last 6 turns. Older context lives in semantic memory.
    if len(SHARED_MESSAGES) > 7:
        SHARED_MESSAGES = [SHARED_MESSAGES[0]] + SHARED_MESSAGES[-6:]

    with MODEL_LOCK:
        prompt = tokenizer.apply_chat_template(
            SHARED_MESSAGES, tokenize=False, add_generation_prompt=True
        )

    assistant_history = [
        m["content"] for m in SHARED_MESSAGES if m["role"] == "assistant"
    ]

    yield frame("start")

    # Generate
    with MODEL_LOCK:
        buffer, _raw_full, streaming, suppressed = "", "", False, False
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=150,
        ):
            piece = chunk.text
            _raw_full += piece

            if suppressed:
                # Tool call: keep generating so the JSON completes and the tool
                # can actually be dispatched, but emit nothing to the client.
                continue

            if not streaming:
                buffer += piece
                # Wait until we can tell a tool call from ordinary prose.
                if len(buffer) < 40 and "CALL_FUNC" not in buffer:
                    continue
                if "CALL_FUNC" in buffer:
                    suppressed = True          # tool call: never stream it
                    continue
                streaming = True
                yield frame("token", text=buffer)
            else:
                yield frame("token", text=piece)

        raw_response = _raw_full.split("<|eot_id|>")[0].strip()

        _t["generate"] = time.time()

    # Tool Extraction
    tool_name, tool_args, pure_narration = extract_function_call(raw_response)

    # If a tool matched cleanly, pure_narration is set. Otherwise, we sanitise the raw string
    # to kill any broken CALL_FUNC artifacts.
    text_to_clean = pure_narration if tool_name else raw_response

    # Guardrails
    sanitised = sanitise_raw(text_to_clean)
    guarded, guardrails = apply_runtime_guardrails(sanitised, assistant_history)
    if guardrails:
        print(f"[guardrails fired: {guardrails}]")
    final_narration = guarded.strip()

    # Tool Execution
    if tool_name:
        tool_reply = call_function_safe(tool_name, tool_args)
        tool_reply_str = (
            json.dumps(tool_reply, ensure_ascii=False)
            if isinstance(tool_reply, (dict, list))
            else str(tool_reply)
        )

        # Merge narration and result for USER display
        if final_narration:
            final_output = f"{final_narration}\n\n{tool_reply_str}".strip()
        else:
            final_output = tool_reply_str

        # Stage the truncated execution trace for the NEXT prompt cycle
        trunc_len = 1000
        LAST_TOOL_RESULT = (
            tool_reply_str[:trunc_len] + "... [truncated due to length]"
            if len(tool_reply_str) > trunc_len
            else tool_reply_str
        )
    else:
        final_output = final_narration

    # Store State cleanly!
    # If a guardrail fired or a tool failed (i.e. we sent an apology), we DO NOT
    # save this to SHARED_MESSAGES. This prevents "context poisoning" where the
    # model learns from its own failures and gets trapped repeating apologies!
    if guardrails or tool_name == "error" or "Apologies" in final_output:
        if len(SHARED_MESSAGES) > 0 and SHARED_MESSAGES[-1]["role"] == "user":
            SHARED_MESSAGES.pop()
            LAST_TOOL_RESULT = ""
    else:
        # Append ONLY the model's generated narration! We never append the raw tool
        # result block into the Assistant's own mouth, otherwise it will mimic it later.
        SHARED_MESSAGES.append({"role": "assistant", "content": final_narration})

    save_telemetry(
        user_input, final_output, guardrails, bool(tool_name), tool_name, tool_args
    )

    if tool_name:
        yield frame("tool", name=tool_name)

    print(
        f"[timing] memory {(_t['memory']-_t['start'])*1000:.0f}ms | "
        f"generate {(_t['generate']-_t['memory'])*1000:.0f}ms | "
        f"prompt_tokens ~{len(prompt)//4} | out_tokens ~{len(_raw_full)//4}"
    )

    yield frame("done", text=final_output)


def get_skye_response(user_input: str) -> str:
    """Thin wrapper over stream_skye_response for callers that want a string."""
    final = ""
    for f in stream_skye_response(user_input):
        data = json.loads(f.decode())
        if data["type"] == "done":
            final = data["text"]
    return final


def rule_based_response(speech: str):
    s = speech.lower()
    if any(
        g in s
        for g in ["hi skye", "hello skye", "good morning skye", "good evening skye"]
    ):
        return Greetings()
    if s.startswith("open "):
        target = s.split("open ", 1)[1].strip()
        sites = {
            "youtube": "https://youtube.com",
            "google": "https://google.com",
            "spotify": "https://open.spotify.com",
        }
        if target in sites:
            webbrowser.open(sites[target])
            return f"Opening {target}, Sir..."
    return None


# =========================================================
# MODES: CLI VS SOCKET
# =========================================================
def start_cli_mode():
    print(f"\nS.K.Y.E. INTERACTIVE CLI (SESSION: {SESSION_ID})\nType 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            print("\nSKYE: ", end="", flush=True)
            final = ""
            for f in stream_skye_response(user_input):
                data = json.loads(f.decode())
                if data["type"] == "token":
                    print(data["text"], end="", flush=True)
                elif data["type"] == "tool":
                    print(f"[{data['name']}]", end="", flush=True)
                elif data["type"] == "done":
                    final = data["text"]
            print()
            if final:
                # Overwrite the streamed text with the guardrailed version.
                print(f"\r\033[KSKYE: {final}\n")
        except KeyboardInterrupt:
            break
    print(f"\nTelemetry saved to logs/telemetry_{SESSION_ID}.jsonl\nSystems OFFLINE.")


def handle_client(conn):
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            user_speech = data.decode().strip()
            print(f"[Client]: {user_speech}")
            try:
                reply = ""
                for f in stream_skye_response(user_speech):
                    conn.sendall(f)
                    payload = json.loads(f.decode())
                    if payload["type"] == "done":
                        reply = payload["text"]
                print(f"[Reply]: {reply}")
            except Exception as e:
                print(f"[Socket send error]: {e}")
                break


def start_server_mode():
    try:
        print("[Bridge Integration]: Booting UI Servers concurrently...")
        threading.Thread(target=start_http_server, daemon=True).start()
        threading.Thread(target=open_browser, daemon=True).start()

        def run_async_ws():
            asyncio.run(start_ws_server())

        threading.Thread(target=run_async_ws, daemon=True).start()
    except Exception as bridge_err:
        print(f"[Bridge Error]: Could not automatically open browser UI: {bridge_err}")

    with socket.socket() as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"\nS.K.Y.E. SOCKET SERVER LISTENING ON {HOST}:{PORT}")
        print(f"SESSION: {SESSION_ID} (Telemetry recording active)")
        while True:
            conn, addr = server_socket.accept()
            print(f"Client connected: {addr}")
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    print("=======================================")
    print("S.K.Y.E. INTERACTION MODES")
    print("=======================================")
    print("[1] CLI Chat (Interactive Terminal)")
    print("[2] Socket Server (Listen on 0.0.0.0:12345)")
    print("=======================================")

    choice = input("Select mode [1/2]: ").strip()
    if choice == "2":
        start_server_mode()
    else:
        start_cli_mode()
