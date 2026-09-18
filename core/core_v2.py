import os
import sys
import re
import json
import base64
import glob
import importlib.util
import platform
import random
import socket
import subprocess
import threading
import time
import webbrowser
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
import mlx.core as mx
from mlx_lm import load, stream_generate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from plyer import notification
from helper_functions.greet import Greetings
from clients.browser import start_http_server, open_browser, start_ws_server
from core.protocol import frame, FrameReader
from core import stt
from core import tts
from scripts.ingest_history import ingest_all_logs
from memory.tasks import format_due

# `memory/short-term/` has a hyphen, so it can't be a normal dotted import
# target (`memory.short-term` isn't valid Python syntax) — loaded by file
# path instead. Only its `run_fact_extraction` function is used here.
_proposed_facts_spec = importlib.util.spec_from_file_location(
    "proposed_facts", os.path.join(ROOT, "memory", "short-term", "proposed_facts.py")
)
proposed_facts = importlib.util.module_from_spec(_proposed_facts_spec)
_proposed_facts_spec.loader.exec_module(proposed_facts)

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

# Connections currently open, so the scheduler thread can push a proactive
# check-in to whatever browser tab(s) are live without one having just sent
# a message — see _broadcast_proactive() and handle_client().
ACTIVE_CONNECTIONS = []
ACTIVE_CONNECTIONS_LOCK = threading.Lock()

# MLX streams are thread-local (mlx >= 0.31.2) — a stream created on one
# thread cannot be used from another. Since every client connection is
# handled on its own thread (see handle_client/start_server_mode), all MLX
# work on that thread must run inside `with mx.stream(MLX_STREAM):`, which
# transparently gives each calling thread its own valid stream.
MLX_STREAM = mx.new_thread_local_stream(mx.default_device())


# =========================================================
# LONG-TERM MEMORY
# =========================================================
from memory.manager import MemoryManager
from memory.tasks import TaskStore

MEMORY = MemoryManager(ROOT)
TASKS = TaskStore(ROOT)

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
# BACKCHANNEL FILLERS
# =========================================================
# Real conversation doesn't go silent while the other person thinks — a
# quick "mm-hmm" or "let me check" fills the gap. LLM generation (plus, for
# tool calls, the tool's own execution time) takes a second or more with
# nothing spoken, which reads as a stall rather than a person listening.
# stream_skye_response() emits a "filler" frame for handle_client() to speak
# immediately, in a background thread, while the real reply is generated.
#
# Placement matters more than the phrases themselves: a coin-flip on every
# turn produces fillers back-to-back one moment and none for five turns the
# next, which reads as a tic, not a person. Real backchannel is driven by
# (a) whether a pause is actually about to happen — a one-word "thanks" gets
# answered instantly, a real question or a tool call doesn't — and (b) not
# doing it again right after the last one. Both are checked below before the
# phrase pool is even chosen.
FILLER_ACK = ["Mm-hmm.", "I hear you.", "Right.", "Okay, go on.", "Got it."]
FILLER_THINKING = [
    "Hmm, let me check that.",
    "One moment.",
    "Let me look into that.",
    "Give me a second.",
]
# Rough signal that the request will dispatch a tool (and so take longer than
# a plain conversational reply) — not exhaustive, just enough to pick the
# right tone of filler.
FILLER_TOOL_HINTS = (
    "weather", "news", "alarm", "remind", "search", "time",
    "open ", "spotify", "youtube", "calendar",
)
QUESTION_LEAD_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|which|whose|is|are|was|were|do|does|did|"
    r"can|could|would|will|should|has|have)\b",
    re.IGNORECASE,
)
# A remark this short ("thanks", "okay then", "sounds good") gets answered
# almost instantly — there's no gap for a filler to fill, so one would land
# after the real reply instead of before it.
FILLER_MIN_WORDS = 4
# Minimum turns that must pass before a *conversational* filler can fire
# again (tool-bound requests are exempt — those pauses are real every time).
# This is what actually stops the "randomly appearing" feel: a filler at
# every turn is a tic, one every third or so is closer to how backchannel
# actually happens.
FILLER_COOLDOWN_TURNS = 2

_last_filler = None
_last_filler_turn = -FILLER_COOLDOWN_TURNS


def pick_filler(user_input: str):
    """Returns a filler phrase to speak while the real reply is generated, or None."""
    global _last_filler, _last_filler_turn

    words = user_input.split()
    lower = user_input.lower()
    is_tool_like = any(hint in lower for hint in FILLER_TOOL_HINTS)
    is_question = user_input.rstrip().endswith("?") or bool(QUESTION_LEAD_RE.match(user_input))

    if not is_tool_like:
        if len(words) < FILLER_MIN_WORDS:
            return None
        if TURN_INDEX - _last_filler_turn < FILLER_COOLDOWN_TURNS:
            return None

    if is_tool_like:
        pool, probability = FILLER_THINKING, 1.0
    elif is_question:
        pool, probability = FILLER_THINKING, 0.6
    else:
        pool, probability = FILLER_ACK, 0.45

    if random.random() > probability:
        return None

    choices = [f for f in pool if f != _last_filler] or pool
    _last_filler = random.choice(choices)
    _last_filler_turn = TURN_INDEX
    return _last_filler


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
    model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")
    tokenizer.eos_token_ids = {
        tokenizer.convert_tokens_to_ids("<turn|>"),
        tokenizer.convert_tokens_to_ids("<eos>"),
    }
    # mlx's first-ever generation call bakes in some internal state (compiled
    # kernels / cache templates) that isn't safely reusable across threads
    # unless that first call happened on the main thread. Every real request
    # runs stream_generate from a per-connection worker thread (handle_client),
    # which crashes with "There is no Stream(gpu, N) in current thread"
    # without this warmup.
    _warmup_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for _ in stream_generate(model, tokenizer, prompt=_warmup_prompt, max_tokens=1):
        pass
print("✓ SKYE ONLINE\n")


# =========================================================
# BACKGROUND SCHEDULER — nightly memory consolidation
# =========================================================
# Nothing outside this process ever scheduled `nightly_agi_cron.py` (no cron
# entry, no launchd plist exists anywhere in the repo) — the fact-extraction
# and semantic-ingestion pipeline it runs has simply never executed. Rather
# than relying on an external scheduler firing at a fixed clock time (which
# silently misses a whole day if the laptop is asleep or the server isn't
# running then), a lightweight in-process thread checks once a minute
# whether more than a day has passed since the last run, and if so runs it
# there and then — resilient to the server being started at an arbitrary
# time, at the cost of "once a day, whenever it's next up" rather than a
# fixed hour.
CONSOLIDATION_STATE_FILE = os.path.join(LOG_DIR, ".consolidation_state.json")
CONSOLIDATION_INTERVAL = timedelta(hours=20)
SCHEDULER_TICK_SECONDS = 60


def _consolidation_due() -> bool:
    if not os.path.exists(CONSOLIDATION_STATE_FILE):
        return True
    try:
        with open(CONSOLIDATION_STATE_FILE, "r") as f:
            state = json.load(f)
        last_run = datetime.fromisoformat(state["last_run"])
    except Exception:
        return True
    return datetime.now() - last_run >= CONSOLIDATION_INTERVAL


def _mark_consolidation_ran():
    with open(CONSOLIDATION_STATE_FILE, "w") as f:
        json.dump({"last_run": datetime.now().isoformat()}, f)


DIAGNOSTICS_FILE = os.path.join(LOG_DIR, "diagnostics.json")


def _run_diagnostics_pass(log_files):
    """Aggregates guardrail-trigger and tool success/failure counts from a
    batch of telemetry files into a small running report.

    Pure counting over data `save_telemetry()` already logs on every turn —
    no LLM call, no change to the live turn path. Tool success/failure is
    read from the same `"Apologies" in assistant_output` convention
    `stream_skye_response()` already relies on elsewhere (for deciding
    whether to keep a turn in SHARED_MESSAGES), rather than plumbing a new
    explicit flag through call_function_safe().
    """
    report = {"guardrails": {}, "tools": {}, "turns_analyzed": 0, "total_words": 0, "total_sentences": 0}
    if os.path.exists(DIAGNOSTICS_FILE):
        try:
            with open(DIAGNOSTICS_FILE, "r") as f:
                report.update(json.load(f))
        except Exception:
            pass

    for filename in log_files:
        try:
            with open(filename, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    report["turns_analyzed"] += 1
                    for g in event.get("guardrails_triggered") or []:
                        report["guardrails"][g] = report["guardrails"].get(g, 0) + 1
                    stats = event.get("response_stats") or {}
                    tool_name = stats.get("tool_dispatched")
                    if tool_name:
                        entry = report["tools"].setdefault(tool_name, {"success": 0, "failure": 0})
                        if "Apologies" in (event.get("assistant_output") or ""):
                            entry["failure"] += 1
                        else:
                            entry["success"] += 1
                    report["total_words"] += stats.get("word_count", 0) or 0
                    report["total_sentences"] += stats.get("sentence_count", 0) or 0
        except Exception as e:
            print(f"[Diagnostics ERROR]: Could not read {filename}: {e}")

    with open(DIAGNOSTICS_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Scheduler]: Diagnostics updated — {report['turns_analyzed']} turns analyzed (cumulative).")
    return report


def _run_daily_consolidation():
    print("[Scheduler]: Running nightly memory consolidation...")
    try:
        # Captured before run_fact_extraction moves these files into
        # processed_logs/, so diagnostics analyzes the same batch fact
        # extraction just did.
        log_files = glob.glob(os.path.join(LOG_DIR, "telemetry_*.jsonl"))
        _run_diagnostics_pass(log_files)

        with MODEL_LOCK, mx.stream(MLX_STREAM):
            old_claims = proposed_facts.run_fact_extraction(model, tokenizer)
        ingest_all_logs(memory=MEMORY)
        # Superseding must happen AFTER ingestion, not before: ingest_all_logs
        # re-embeds every dialogue pair in these same files unconditionally,
        # including the turn where the now-wrong claim was originally stated —
        # superseding it first would just have it silently reappear moments
        # later from that same ingestion pass.
        for claim in old_claims:
            superseded_count = MEMORY.supersede_similar(claim)
            print(f"[Scheduler]: Correction '{claim}' superseded {superseded_count} memory row(s).")

        _mark_consolidation_ran()
        print("[Scheduler]: Nightly memory consolidation complete.")
    except Exception as e:
        print(f"[Scheduler ERROR]: Nightly consolidation failed: {e}")


def _notify_os(title: str, message: str):
    """OS-level notification — the fallback that fires regardless of whether
    a browser tab is even open (or asleep), so a time-critical alarm is never
    silently missed just because no one's listening for voice.

    plyer's macOS backend needs `pyobjus` (an Objective-C bridge that's
    effectively unmaintained and frequently fails to build against current
    macOS/Python), so macOS is handled directly via `osascript` instead —
    it ships with every Mac, no extra dependency required. Windows keeps
    using plyer, which is what this was originally written and run against.
    """
    system = platform.system()
    if system == "Darwin":
        script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
    else:
        notification.notify(title=title, message=message, app_name="SKYE", timeout=10)


def _fire_due_tasks():
    for task in TASKS.get_due():
        message = task["description"]
        print(f"[Scheduler]: Task due — {message}")
        try:
            _notify_os("SKYE", message)
        except Exception as e:
            print(f"[Scheduler ERROR]: OS notification failed: {e}")
        _broadcast_proactive(message)
        if task["recurrence"]:
            TASKS.reschedule(task["id"])
        else:
            TASKS.mark_status(task["id"], "done")


def _scheduler_loop():
    while True:
        if _consolidation_due():
            _run_daily_consolidation()
        _fire_due_tasks()
        time.sleep(SCHEDULER_TICK_SECONDS)


# =========================================================
# TOOL EXECUTION — MCP client
# =========================================================
# Every tool used to be a plain Python function in an in-process `TOOLS`
# dict, called directly by call_function_safe(). That dict is gone —
# mcp_server/server.py now serves the exact same set of tools (faithfully
# ported, see that file) over MCP, running as its own subprocess.
#
# MCP client sessions are async and need to stay alive for the life of the
# stdio connection to that subprocess, but stream_skye_response()/
# handle_client() are synchronous throughout (always have been — only
# clients/browser.py's WS<->TCP bridge uses asyncio, for an unrelated
# reason). Rather than convert the whole turn-handling path to async, one
# dedicated background thread runs its own event loop for the life of the
# process; call_mcp_tool() bridges a synchronous call site to it via
# asyncio.run_coroutine_threadsafe(), the standard pattern for exactly this.
MCP_SERVER_PATH = os.path.join(ROOT, "mcp_server", "server.py")
TOOL_ROUTES = {}  # tool name -> ClientSession
_mcp_loop = None
_mcp_exit_stack = None


async def _mcp_connect():
    global _mcp_exit_stack
    _mcp_exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH])
    read, write = await _mcp_exit_stack.enter_async_context(stdio_client(server_params))
    session = await _mcp_exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await session.list_tools()
    for t in tools.tools:
        TOOL_ROUTES[t.name] = session
    print(f"[MCP] Connected to local tool server — {len(TOOL_ROUTES)} tools: {', '.join(TOOL_ROUTES)}")


def start_mcp_bridge():
    global _mcp_loop
    _mcp_loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(_mcp_loop)
        _mcp_loop.run_forever()

    threading.Thread(target=_run_loop, daemon=True).start()
    # Block startup until the local tool server is actually up and
    # list_tools() has populated TOOL_ROUTES — every mode (CLI and socket
    # server) needs working tools from the moment it starts accepting input.
    asyncio.run_coroutine_threadsafe(_mcp_connect(), _mcp_loop).result(timeout=30)


def call_mcp_tool(name, arguments):
    session = TOOL_ROUTES[name]

    async def _call():
        return await session.call_tool(name, arguments)

    result = asyncio.run_coroutine_threadsafe(_call(), _mcp_loop).result(timeout=30)
    text = result.content[0].text if result.content else None
    if result.is_error:
        raise RuntimeError(text or "unknown MCP tool error")
    return text


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
    if name not in TOOL_ROUTES:
        return f"Apologies, Sir. I lack the tool `{name}`."
    try:
        result = call_mcp_tool(name, args)
        return result if result is not None else "Executed."
    except Exception as e:
        print(f"\n[Tool Execution Error ({name})]: {e}")
        return f"Apologies, Sir. `{name}` failed."


# Every mode (CLI and socket server) dispatches tools through
# call_function_safe(), so the MCP bridge needs to be up before either
# starts accepting input — done here, once, at import time, same as the
# model load above it.
print("Connecting to local tool server (MCP)...")
start_mcp_bridge()
print("✓ TOOLS ONLINE\n")


# =========================================================
# CORE NLP MASTER LOGIC
# =========================================================
# CONVERSATION STATE
# Shared chat history across an active session (for CLI or single-user Socket)
#
# Persisted to disk so a server restart (a crash, a code update, closing the
# laptop) doesn't wipe the immediate conversational thread — long-term memory
# (persistent_profile.json, semantics.db, tasks.db) already survives a
# restart; this brings the short-term window in line with that. Deliberately
# NOT persisted: SESSION_ID/TURN_INDEX (a new boot should get a fresh
# telemetry session, that's existing intended behavior) and LAST_TOOL_RESULT
# (a single-turn staging value already cleared immediately after use).
CONVERSATION_STATE_FILE = os.path.join(ROOT, "memory", "conversation_state.json")


def _load_conversation_state() -> list:
    if os.path.exists(CONVERSATION_STATE_FILE):
        try:
            with open(CONVERSATION_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Conversation state] Failed to load, starting fresh: {e}")
    return []


def _save_conversation_state():
    try:
        with open(CONVERSATION_STATE_FILE, "w") as f:
            json.dump(SHARED_MESSAGES, f)
    except Exception as e:
        print(f"[Conversation state] Failed to save: {e}")


SHARED_MESSAGES = _load_conversation_state()
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

    # Past this point a real generation (and maybe a tool call) is about to
    # happen, which is exactly the gap a filler should cover. handle_client
    # speaks this in a background thread the moment it sees the frame, so it
    # overlaps with generation instead of adding to the wait.
    filler_text = pick_filler(user_input)
    if filler_text:
        yield frame("filler", text=filler_text)

    # Format user prompt, injecting past tool data if present
    if LAST_TOOL_RESULT:
        augmented_input = f"[System Note: Tool execution returned: {LAST_TOOL_RESULT}]\n\n{user_input}"
        LAST_TOOL_RESULT = ""
    else:
        augmented_input = user_input

    # Retrieve the persistent profile, relevant semantic memories, and any
    # pending tasks/reminders
    profile = MEMORY.get_persistent_profile()
    memories = MEMORY.search(user_input, top_k=3)
    upcoming_tasks = TASKS.get_upcoming(5)
    _t["memory"] = time.time()

    system_text = PERSONA
    if profile:
        profile_str = json.dumps(profile, ensure_ascii=False)
        system_text += f"\n[User Profile Data]: {profile_str}"

    if memories:
        memories_str = "\n".join([f"- {m}" for m in memories])
        system_text += f"\n[Relevant Past Memories]:\n{memories_str}"

    if upcoming_tasks:
        tasks_str = "\n".join(
            f"- {t['description']} ({format_due(t['due_at'])})" for t in upcoming_tasks
        )
        system_text += f"\n[Upcoming Tasks]:\n{tasks_str}"

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
            SHARED_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
            # Was 150 — too tight for the persona's new "explain properly"
            # length target (~4-6 sentences); that could get cut off
            # mid-thought before this was raised. MAX_WORDS/MAX_SENTENCES
            # guardrails below still apply on top of this.
            max_tokens=320,
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

        raw_response = _raw_full.split("<turn|>")[0].strip()

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

    _save_conversation_state()

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


def _speak_filler(conn, send_lock, phrase):
    """Synthesizes and sends a filler phrase's audio in the background.

    Runs on its own thread so it overlaps with LLM generation on the main
    connection thread instead of adding to the wait before it. The two
    threads share one socket, so every send is serialized through
    `send_lock` — each frame() call is still written whole, just never
    interleaved byte-for-byte with a frame from the other thread.
    """
    try:
        for pcm, _ in tts.synthesize_reply(phrase, speed_range=(1.0, 1.12)):
            with send_lock:
                conn.sendall(
                    frame(
                        "audio_chunk",
                        pcm=base64.b64encode(pcm).decode("ascii"),
                        sample_rate=24000,
                        final=False,
                    )
                )
    except Exception as e:
        print(f"[Filler TTS error]: {e}")


def _broadcast_proactive(text: str):
    """Pushes an unprompted spoken message to every currently-connected client.

    Same shape as a normal turn's outbound frames (a text frame the client
    displays, then audio_chunk/turn_end for the synthesized voice) except
    nothing preceded it — no inbound message ever triggered this. Only
    reaches a browser tab because clients/browser.py's bridge now pumps
    TCP->WS continuously instead of only right after forwarding a message.
    """
    with ACTIVE_CONNECTIONS_LOCK:
        targets = list(ACTIVE_CONNECTIONS)
    for conn, send_lock in targets:
        try:
            with send_lock:
                conn.sendall(frame("proactive", text=text))
            for pcm, is_final in tts.synthesize_reply(text):
                with send_lock:
                    conn.sendall(
                        frame(
                            "audio_chunk",
                            pcm=base64.b64encode(pcm).decode("ascii"),
                            sample_rate=24000,
                            final=is_final,
                        )
                    )
            with send_lock:
                conn.sendall(frame("turn_end"))
        except Exception as e:
            print(f"[Proactive broadcast error]: {e}")


def handle_client(conn):
    reader = FrameReader()
    send_lock = threading.Lock()
    with ACTIVE_CONNECTIONS_LOCK:
        ACTIVE_CONNECTIONS.append((conn, send_lock))
    try:
        _handle_client_loop(conn, reader, send_lock)
    finally:
        with ACTIVE_CONNECTIONS_LOCK:
            if (conn, send_lock) in ACTIVE_CONNECTIONS:
                ACTIVE_CONNECTIONS.remove((conn, send_lock))


def _handle_client_loop(conn, reader, send_lock):
    with conn, mx.stream(MLX_STREAM):
        while True:
            data = conn.recv(4096)
            if not data:
                break
            for msg in reader.feed(data):
                if msg.get("type") == "text":
                    user_speech = msg.get("text", "").strip()
                elif msg.get("type") == "audio":
                    pcm = base64.b64decode(msg["pcm"])
                    transcript = stt.transcribe_pcm(
                        pcm, sample_rate=msg.get("sample_rate", 16000)
                    ).strip()
                    with send_lock:
                        conn.sendall(frame("transcript", text=transcript))
                    continue
                else:
                    continue
                if not user_speech:
                    continue

                print(f"[Client]: {user_speech}")
                try:
                    reply = ""
                    for f in stream_skye_response(user_speech):
                        with send_lock:
                            conn.sendall(f)
                        payload = json.loads(f.decode())
                        if payload["type"] == "filler":
                            threading.Thread(
                                target=_speak_filler,
                                args=(conn, send_lock, payload["text"]),
                                daemon=True,
                            ).start()
                        elif payload["type"] == "done":
                            reply = payload["text"]
                    if reply.strip():
                        for pcm, is_final in tts.synthesize_reply(reply):
                            with send_lock:
                                conn.sendall(
                                    frame(
                                        "audio_chunk",
                                        pcm=base64.b64encode(pcm).decode("ascii"),
                                        sample_rate=24000,
                                        final=is_final,
                                    )
                                )
                    with send_lock:
                        conn.sendall(frame("turn_end"))
                    print(f"[Reply]: {reply}")
                except Exception as e:
                    print(f"[Socket send error]: {e}")
                    return


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

    threading.Thread(target=_scheduler_loop, daemon=True).start()

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
