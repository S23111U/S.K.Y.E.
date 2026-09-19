import os
import sys
import re
import json
import base64
import glob
import importlib.util
import platform
import queue
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
from clients.browser import start_http_server, open_browser, start_ws_server
from core.protocol import frame, FrameReader
from core import stt
from core import tts_client as tts
from core.mood import MoodClassifier, MOOD_PARAMS
from core.skills import SKILLS, STICKY_TURNS, manifest_text, route_skill
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

# Per-turn latency breakdown, filled in by stream_skye_response() and completed
# (TTS times added, then written out) by _run_turn(). Exists because "replies
# feel slower as the conversation goes on" was a feeling with nothing to
# measure it against.
TIMING_FILE = os.path.join(LOG_DIR, f"timing_{SESSION_ID}.jsonl")
LAST_TURN_TIMING = {}
# (skill, turns left): keeps the last turn's skill for a follow-up like "undo that".
_SKILL_STICKY = [None, 0]

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
print("Loading emotion model...")
MOOD = MoodClassifier()
print("✓ EMOTION MODEL ONLINE")
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
# Fillers are chosen by the *user's* mood, so an empathetic beat ("Oh, I'm
# sorry to hear that.") comes before the answer to bad news, and a pleased one
# before the answer to good news. Each phrase is spoken in that mood's delivery.
FILLERS = {
    "calm": [
        "Hmm, let me check that.", "One moment.", "Let me look into that.",
        "Give me a second.", "Let me think.", "Right, one moment.", "Good question.",
    ],
    "happy": ["Oh, wonderful.", "Ah, splendid.", "That's good to hear."],
    "sad": ["Oh, I'm sorry to hear that.", "Ah, that's unfortunate.", "Oh dear."],
    "concerned": ["Hmm, I understand.", "I see.", "Let me see."],
}
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
_last_filler = None


def pick_filler(user_input: str, mood: str = "calm"):
    """Returns (phrase, mood) to speak while the real reply is generated, or None.

    Nearly every turn that goes to the LLM has a real pause (1.5-3.5 s), and a
    listener who says nothing for that long reads as a stall, not a person. So
    fillers are the rule, not the exception: always for tool requests and for
    emotional messages (the empathetic beat matters most there), very likely for
    questions, and often for longer statements. The only brake is not repeating
    a plain "let me check" filler on consecutive turns.
    """
    global _last_filler

    words = user_input.split()
    lower = user_input.lower()
    is_tool_like = any(hint in lower for hint in FILLER_TOOL_HINTS)
    is_question = user_input.rstrip().endswith("?") or bool(QUESTION_LEAD_RE.match(user_input))
    emotional = mood != "calm"

    # Probabilities (were 1.0 / 1.0 / 0.85 / 0.6, which put a filler on ~89% of
    # turns in a real session and started to sound like a tic). These give ~68%
    # on the same session: still highest where the pause is real (tools) or the
    # beat matters (emotion), and lower for plain statements.
    if emotional:
        if len(words) < 2:
            return None
        probability = 0.85
    elif is_tool_like:
        probability = 0.9
    elif is_question and len(words) >= 3:
        probability = 0.7
    elif len(words) >= FILLER_MIN_WORDS:
        probability = 0.5
    else:
        return None

    if random.random() > probability:
        return None

    pool = FILLERS[mood]
    choices = [f for f in pool if f != _last_filler] or pool
    _last_filler = random.choice(choices)
    return _last_filler, mood


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
WEB_SOURCES_MARKER = "[WEB SOURCES]"

WEB_SYNTH_SYSTEM = (
    "You are S.K.Y.E., a dry, precise assistant. Answer the user's question "
    "using ONLY the numbered sources below. Sources are noisy and may disagree "
    "or be out of date: prefer the most recent, favour facts several sources "
    "agree on, and never fill a gap from your own memory. Every name, score "
    "and date you state must appear in the sources. If the sources conflict "
    "or do not cover the question, say so plainly instead of guessing. For "
    "time-sensitive facts, say how recent the information is. Compare every event "
    "date with today's date: an event before today has already happened and is "
    "never 'upcoming' or 'next'; if no future fixture appears in the sources, "
    "say none was found. Plain spoken "
    "prose, three to five sentences, no markdown, no lists."
)


def synthesize_web_answer(question: str, query: str, sources: str) -> str:
    """Second, grounded generation pass over raw web_search sources."""
    messages = [
        {"role": "system", "content": WEB_SYNTH_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nSearch query used: {query}\n\n{sources}"},
    ]
    with MODEL_LOCK:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out = ""
        for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=260):
            out += chunk.text
    return sanitise_raw(out.split("<turn|>")[0].strip())


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


def _warm_llm():
    """One throwaway generation with a realistic (persona-sized) prompt, so the
    first real reply doesn't pay for cold Metal kernels — the very first turn of
    a session took 22 s against ~2 s afterwards."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": PERSONA}, {"role": "user", "content": "hello"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    with MODEL_LOCK, mx.stream(MLX_STREAM):
        for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=12):
            pass


_warm_llm()

print("Loading TTS (Chatterbox Turbo)...")
tts.start()
print("✓ TTS ONLINE\n")


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


TIMING_RETENTION_DAYS = 14
TTS_LOG_MAX_BYTES = 2 * 1024 * 1024


def _prune_logs():
    """Housekeeping for diagnostic-only files. timing_*.jsonl (per-turn latency
    breakdowns) is read by nothing in SKYE — only by a human debugging — so it
    is deleted after two weeks; telemetry_*.jsonl is NOT touched (memory
    ingestion and fact extraction read it). The TTS subprocess log is trimmed
    to its last 200 KB once it passes 2 MB."""
    try:
        cutoff = time.time() - TIMING_RETENTION_DAYS * 86400
        for f in glob.glob(os.path.join(LOG_DIR, "timing_*.jsonl")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
        tts_log = os.path.join(LOG_DIR, "tts_server.log")
        if os.path.isfile(tts_log) and os.path.getsize(tts_log) > TTS_LOG_MAX_BYTES:
            with open(tts_log, "r+b") as fh:   # the writer appends, so truncating is safe
                fh.seek(-200 * 1024, os.SEEK_END)
                tail = fh.read()
                fh.seek(0)
                fh.truncate()
                fh.write(tail)
    except OSError as e:
        print(f"[Scheduler]: log pruning skipped: {e}")


_prune_logs()   # also at boot: the nightly job only runs if SKYE is up at that hour


def _run_daily_consolidation():
    print("[Scheduler]: Running nightly memory consolidation...")
    _prune_logs()
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
    LAST_TURN_TIMING.clear()
    user_mood, emotion, emotion_score = MOOD.user_mood(user_input)
    LAST_TURN_TIMING.update(
        turn=TURN_INDEX, tool=None, user_mood=user_mood, emotion=emotion,
        emotion_score=emotion_score, filler=None,
    )

    skill = route_skill(user_input, _SKILL_STICKY[0] if _SKILL_STICKY[1] > 0 else None)
    if skill:
        _SKILL_STICKY[0], _SKILL_STICKY[1] = skill, STICKY_TURNS
    else:
        _SKILL_STICKY[1] = max(0, _SKILL_STICKY[1] - 1)
    LAST_TURN_TIMING["skill"] = skill

    ack = acknowledgement_reply(user_input)
    rule_reply = ack or rule_based_response(user_input)
    if rule_reply:
        # Whatever the last tool returned is not relevant to a greeting/thanks.
        LAST_TOOL_RESULT = ""
        if ack:
            LAST_TURN_TIMING["user_mood"] = user_mood = "happy"
        save_telemetry(user_input, rule_reply, ["rule_bypass"], False)
        yield frame("done", text=rule_reply)
        return

    # Past this point a real generation (and maybe a tool call) is about to
    # happen, which is exactly the gap a filler should cover. handle_client
    # speaks this in a background thread the moment it sees the frame, so it
    # overlaps with generation instead of adding to the wait.
    if skill:
        # The browser recolours itself for the skill that is answering.
        yield frame("skill", skill=skill, ui=SKILLS[skill]["ui"])
    filler = pick_filler(user_input, user_mood)
    if filler:
        LAST_TURN_TIMING["filler"] = filler[0]
        yield frame("filler", text=filler[0], mood=filler[1])

    # Format user prompt, injecting past tool data if present
    if LAST_TOOL_RESULT:
        augmented_input = (
            "[Context only — the result of your previous tool call. Answer the message "
            "below on its own terms; do not repeat or summarise this unless the user "
            f"asks about it: {LAST_TOOL_RESULT}]\n\n{user_input}"
        )
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
    if skill:
        system_text += f"\n\n[Tools for this request]\n{manifest_text(skill)}"
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
        _last_chunk = None
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
            _last_chunk = chunk

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
        _t["tool"] = time.time()
        tool_reply_str = (
            json.dumps(tool_reply, ensure_ascii=False)
            if isinstance(tool_reply, (dict, list))
            else str(tool_reply)
        )

        if tool_reply_str.startswith(WEB_SOURCES_MARKER):
            tool_reply_str = synthesize_web_answer(
                user_input, (tool_args or {}).get("query", user_input), tool_reply_str
            )
            final_narration = ""
        _t["synth"] = time.time()

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
        # A tool turn has no narration; recording it as an empty assistant
        # message taught the model that "empty" is what it says — after a few
        # tool turns in a row it started answering "Undo that." with nothing.
        # The call it actually made is the honest history, and keeps the
        # CALL_FUNC format in view.
        history_text = final_narration
        if tool_name and not history_text:
            history_text = f"CALL_FUNC: {json.dumps({'name': tool_name, 'arguments': tool_args}, ensure_ascii=False)}"
        SHARED_MESSAGES.append({"role": "assistant", "content": history_text})

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

    _end = time.time()
    _tool_end = _t.get("tool", _t["generate"])
    LAST_TURN_TIMING.update(
        turn=TURN_INDEX - 1,
        tool=tool_name,
        history_msgs=len(SHARED_MESSAGES),
        prompt_tokens=getattr(_last_chunk, "prompt_tokens", None),
        prompt_tps=round(getattr(_last_chunk, "prompt_tps", 0) or 0, 1),
        gen_tokens=getattr(_last_chunk, "generation_tokens", None),
        gen_tps=round(getattr(_last_chunk, "generation_tps", 0) or 0, 1),
        memory_ms=round((_t["memory"] - _t["start"]) * 1000),
        generate_ms=round((_t["generate"] - _t["memory"]) * 1000),
        tool_ms=round((_tool_end - _t["generate"]) * 1000),
        web_synth_ms=round((_t["synth"] - _tool_end) * 1000) if "synth" in _t else 0,
        total_llm_ms=round((_end - _t["start"]) * 1000),
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


# Thanks-type phrases are unambiguous, so a short sentence containing one is an
# acknowledgement. Bare approval words ("okay", "great") are not: "Okay there are
# a few corrections" is the start of a request, so they only count when they
# are the whole message.
THANKS_RE = re.compile(
    r"\b(?:thank(?:s| you)|cheers|much appreciated|appreciate (?:it|that)|"
    r"(?:that'?s|that is|sounds|it'?s) (?:great|good|perfect|fine|interesting|helpful|brilliant|superb))\b",
    re.IGNORECASE,
)
APPROVAL_ONLY_RE = re.compile(
    r"^\W*(?:(?:ok|okay|alright|all right|got it|understood|superb|perfect|brilliant|excellent|awesome|cool|nice|great|"
    r"fair enough|sounds good)\W*)+(?:(?:skye|sky|sir)\W*)?$",
    re.IGNORECASE,
)
# Anything that makes it a request rather than a bare acknowledgement.
ACK_BLOCK_RE = re.compile(
    r"\?|\b(?:can|could|would|will|please|set|open|play|search|find|tell|show|list|remind|"
    r"what|who|when|where|why|how|which|explain|also|and then|but|now)\b",
    re.IGNORECASE,
)
ACK_REPLIES = ["You're welcome, Sir.", "Happy to help.", "Not at all.", "Any time, Sir.", "My pleasure."]


def acknowledgement_reply(speech: str):
    """A short direct reply to a bare "thank you" / "okay great", or None.

    These used to go to the LLM, which — with the previous tool result still
    attached to the message — dutifully recapped the last answer instead of
    just replying. Handling them here also makes them instant.
    """
    if APPROVAL_ONLY_RE.match(speech):
        return random.choice(ACK_REPLIES)
    if len(speech.split()) > 8 or ACK_BLOCK_RE.search(speech) or not THANKS_RE.search(speech):
        return None
    return random.choice(ACK_REPLIES)


def greeting() -> str:
    """Time-of-day greeting for the rule-based fast path ("hi skye")."""
    hour = datetime.now().hour
    if hour == 0 or hour > 22:
        return "It's quite late, Good Evening Sir"
    if hour < 12:
        return "Good Morning, Sir"
    if hour <= 15:
        return "Good Afternoon, Sir"
    return "Good Evening, Sir"


def rule_based_response(speech: str):
    s = speech.lower()
    if any(
        g in s
        for g in ["hi skye", "hello skye", "good morning skye", "good evening skye"]
    ):
        return greeting()
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


def _speak_filler(conn, send_lock, phrase, mood, cancel):
    """Sends a filler phrase's (pre-synthesized, cached) audio in the background.

    The audio comes from tts.cached_phrase(), so this costs no GPU while the
    LLM is generating — it used to run TTS concurrently with generation and
    slow both. Still on its own thread so the (small) disk read and socket
    write never delay the main turn. Every send goes through `send_lock` so a
    frame is never interleaved byte-for-byte with one from the turn thread.
    """
    try:
        pcm = tts.cached_phrase(phrase, mood)
        if cancel.is_set():
            return
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


class CancelEvent(threading.Event):
    """A cancel flag that remembers when and why it was set, so telemetry can
    show whether an interruption was a real barge-in or (say) SKYE's own voice
    leaking into the microphone."""

    def __init__(self):
        super().__init__()
        self.set_at = None
        self.info = None

    def set(self, info=None):
        if not self.is_set():
            self.set_at, self.info = time.time(), info
            super().set()


def _run_turn(conn, send_lock, user_speech, cancel):
    """One full turn: LLM (+ tools), then speech. `cancel` is set when the user
    barges in or sends a newer request; it stops speech synthesis between
    audio chunks (generation itself always completes, so conversation state is
    never left half-written)."""
    print(f"[Client]: {user_speech}")
    t_turn = time.time()
    reply = ""
    for f in stream_skye_response(user_speech):
        with send_lock:
            conn.sendall(f)
        payload = json.loads(f.decode())
        if payload["type"] == "filler":
            threading.Thread(
                target=_speak_filler,
                args=(conn, send_lock, payload["text"], payload.get("mood", "calm"), cancel),
                daemon=True,
            ).start()
        elif payload["type"] == "done":
            reply = payload["text"]

    timing = dict(LAST_TURN_TIMING)
    tts_t, first_audio_ms, chunks, mood = {}, None, 0, None
    if reply.strip() and not cancel.is_set():
        mood = MOOD.final_mood(timing.get("user_mood", "calm"), reply)
        with send_lock:
            conn.sendall(frame("mood", mood=mood, emotion=timing.get("emotion")))
        for pcm, is_final in tts.synthesize_reply(
            reply, params=MOOD_PARAMS[mood], cancel=cancel, timings=tts_t
        ):
            if cancel.is_set():
                break
            if first_audio_ms is None:
                first_audio_ms = round((time.time() - t_turn) * 1000)
            chunks += 1
            with send_lock:
                conn.sendall(
                    frame(
                        "audio_chunk",
                        pcm=base64.b64encode(pcm).decode("ascii"),
                        sample_rate=24000,
                        final=is_final,
                    )
                )
    # Always closes the turn — including a cancelled one, so the client knows
    # to stop discarding audio and the next turn's chunks aren't dropped.
    with send_lock:
        conn.sendall(frame("turn_end"))

    timing.update(
        cancelled=cancel.is_set(),
        mood=mood,
        cancel_after_ms=(
            round((cancel.set_at - t_turn) * 1000) if cancel.set_at else None
        ),
        cancel_info=cancel.info,
        tts_first_chunk_ms=tts_t.get("first_chunk_ms"),
        tts_total_ms=tts_t.get("total_ms"),
        tts_chunks_sent=chunks,
        first_audio_ms=first_audio_ms,
        turn_total_ms=round((time.time() - t_turn) * 1000),
    )
    try:
        with open(TIMING_FILE, "a") as tf:
            tf.write(json.dumps(timing) + "\n")
    except OSError:
        pass
    print(f"[Reply{' (cancelled)' if cancel.is_set() else ''}]: {reply}")


def _handle_client_loop(conn, reader, send_lock):
    """Reads frames continuously on this thread while turns run on a worker.

    Previously one thread did both: it read a message, ran the whole turn
    (LLM *and* synthesizing every sentence), and only then read the next
    message. So a barge-in's audio sat unread until the old reply had finished
    synthesizing, and the interrupted reply kept synthesizing (and being sent)
    after the user had already started talking. Now this thread only reads —
    audio is transcribed and 'cancel'/'text' frames are acted on immediately —
    and turns are queued to a single worker so they still run one at a time.
    """
    turns = queue.Queue()
    state = {"cancel": CancelEvent()}

    def worker():
        with mx.stream(MLX_STREAM):
            while True:
                item = turns.get()
                if item is None:
                    return
                text, cancel = item
                try:
                    _run_turn(conn, send_lock, text, cancel)
                except Exception as e:
                    print(f"[Socket send error]: {e}")
                    return

    threading.Thread(target=worker, daemon=True).start()

    try:
        with conn, mx.stream(MLX_STREAM):
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                for msg in reader.feed(data):
                    kind = msg.get("type")
                    if kind == "cancel":
                        # Barge-in: stop speaking the current reply.
                        state["cancel"].set({k: v for k, v in msg.items() if k != "type"})
                    elif kind == "audio":
                        pcm = base64.b64decode(msg["pcm"])
                        transcript = stt.transcribe_pcm(
                            pcm, sample_rate=msg.get("sample_rate", 16000)
                        ).strip()
                        with send_lock:
                            conn.sendall(frame("transcript", text=transcript))
                    elif kind == "text":
                        user_speech = msg.get("text", "").strip()
                        if not user_speech:
                            continue
                        # A new request supersedes whatever is still being
                        # spoken from the previous one.
                        state["cancel"].set({"reason": "superseded"})
                        state["cancel"] = CancelEvent()
                        turns.put((user_speech, state["cancel"]))
    finally:
        state["cancel"].set()
        turns.put(None)


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

    # Synthesize every filler phrase now (a no-op once cached on disk) so the
    # first one a user triggers is instant instead of waiting on the engine.
    threading.Thread(
        target=lambda: [tts.cached_phrase(p, m) for m, ps in FILLERS.items() for p in ps],
        daemon=True,
    ).start()

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
