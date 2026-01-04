import os
import sys
import re
import json
import socket
import threading
import webbrowser
import wikipedia
from datetime import datetime
from mlx_lm import load, generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from helper_functions.current_time import TellTime
from helper_functions.weather import Get_Info
from helper_functions.set_alarm import set_alarm
from helper_functions.set_reminder import set_reminder
from helper_functions.GenAI import GenAI_search
from helper_functions.greet import Greetings

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(ROOT, "custom-models", "SKYE")
HOST, PORT = "0.0.0.0", 12345
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "skye_history.jsonl")

# Safety: functions that must never be executed
SAFETY_BLOCKLIST = {
    "delete_files",
    "shutdown",
    "reboot",
    "format_drive",
    "hack_into_server",
    "hack",
}

# ---------------- LOAD MODEL ----------------
print("Loading S.K.Y.E. ...")
model, tokenizer = load(MODEL_PATH)
print("✓ Model loaded from:", MODEL_PATH)

# ---------------- TOOL REGISTRY ----------------
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
register_tool("web_search", lambda **kw: (wikipedia_summary_or_genai(kw.get("query"))))
register_tool("open_youtube", lambda **kw: web_open_and_ack("youtube", kw.get("query")))
register_tool("open_spotify", lambda **kw: web_open_and_ack("spotify", kw.get("query")))
register_tool("open_calendar", lambda **kw: web_open_and_ack("google", "calendar"))


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


# ---------------- UTILS ----------------
def log_interaction(user_text, skye_text, metadata=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_text,
        "skye": skye_text,
    }
    if metadata:
        entry.update(metadata)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def clean_response(text):
    """Sanitize tokens and trim to a readable short response."""
    if text is None:
        return ""
    
    # Remove special tokens
    text = re.sub(r"<\|end\|>", "", text)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<unk>|<s>|</s>", "", text)
    text = text.strip()
    
    # If there's a function call, preserve it but clean up surrounding text
    if "CALL_FUNC" in text:
        # Don't truncate if there's a function call - let detect_function_call handle it
        return text.strip()
    
    # For regular responses, remove repetitive patterns
    # Stop at first occurrence of common ending phrases if they repeat
    ending_phrases = [
        "Operation completed",
        "Task executed successfully",
        "Standing by for your directive",
        "Awaiting your next command",
        "Understood, Sir",
        "Acknowledged",
        "Affirmative",
    ]
    
    for phrase in ending_phrases:
        if text.count(phrase) > 1:
            # Find first occurrence and stop there
            idx = text.find(phrase)
            if idx > 0:
                text = text[:idx + len(phrase)].strip()
                break
    
    # Remove any mid-text <|assistant|> tokens
    text = re.sub(r"<\|assistant\|>", "", text)
    
    # For non-function responses, limit to first 2-3 sentences
    sentences = [s.strip() for s in re.split(r"[.?!]\s*", text) if s.strip()]
    if sentences and "CALL_FUNC" not in text:
        out = ". ".join(sentences[:3]).strip()
        if not out.endswith((".", "!", "?")):
            out += "."
        return out
    
    return text[:200].strip()


# ---------------- FUNCTION-CALL PARSER ----------------
# This regex looks for the first JSON-looking object after CALL_FUNC or Executing function
CALL_PATTERN = re.compile(
    r"(CALL_FUNC|Executing function)\s*[:\-]*\s*({.*?})", re.DOTALL
)


def normalize_json_text(s):
    # replace params -> arguments, single quotes -> double quotes, stray trailing punctuation
    s = s.replace("'", '"')
    s = s.replace("params", "arguments")
    # remove trailing dots after JSON
    s = re.sub(r"\}\s*[\.\,]+\s*$", "}", s)
    # Collapse repeated braces/garbage (take first well-formed JSON substring)
    # Attempt to find matching braces pair
    try:
        # find first { ... } balanced substring
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
                    candidate = s[start : i + 1]
                    return candidate
    except Exception:
        pass
    return s


def detect_function_call(skye_text):
    """Return (name, arguments dict, narration_before_call) or (None, None, full_text)
    Only returns the FIRST valid function call to prevent repetitive calls.
    """
    if not skye_text:
        return None, None, skye_text
    
    # Find ALL matches first
    matches = list(CALL_PATTERN.finditer(skye_text))
    if not matches:
        return None, None, skye_text
    
    # Process matches in order, return first valid one
    for m in matches:
        narration = skye_text[: m.start()].strip()
        json_part = m.group(2)
        json_part = normalize_json_text(json_part)
        
        try:
            payload = json.loads(json_part)
            name = payload.get("name") or payload.get("func") or payload.get("function")
            
            # Validate that we have a function name
            if not name or not isinstance(name, str):
                continue
                
            arguments = payload.get("arguments") or {}
            # safety normalization: ensure arguments is a dict
            if not isinstance(arguments, dict):
                arguments = {}
            
            # Found first valid function call - return it
            # Clean up narration to remove any repetitive patterns
            narration = re.sub(r"<\|assistant\|>", "", narration).strip()
            
            # Remove repetitive ending phrases from narration
            ending_phrases = [
                "Operation completed",
                "Task executed successfully",
                "Standing by for your directive",
                "Awaiting your next command",
            ]
            for phrase in ending_phrases:
                if phrase in narration and narration.count(phrase) > 1:
                    idx = narration.find(phrase)
                    if idx > 0:
                        narration = narration[:idx + len(phrase)].strip()
                        break
            
            return name, arguments, narration
        except Exception as e:
            # Try next match if this one failed
            print(f"[Parser] JSON parse error: {e} | raw: {json_part[:200]}")
            continue
    
    # No valid function calls found
    return None, None, skye_text


# ---------------- SAFE CALLER ----------------
def call_function_safe(name, args):
    if not name:
        return "Apologies, Sir. Could not detect a function to execute."
    if name in SAFETY_BLOCKLIST:
        return f"Negative, Sir. The command `{name}` is not authorized."
    fn = TOOLS.get(name)
    if not fn:
        return f"Apologies, Sir. I do not have a tool named `{name}` registered."
    try:
        # if the tool spawns a thread internally it should return immediately
        result = fn(**args) if callable(fn) else fn
        # If function returns a thread-like object, ignore
        return result if result is not None else "Executed."
    except Exception as e:
        print(f"[Executor] Error calling {name}: {e}")
        return f"Apologies, Sir. Execution of `{name}` failed."


# ---------------- S.K.Y.E. CHAT WRAPPER ----------------
def skye_chat(prompt, max_tokens=120):
    """Generate using fused model; returns cleaned SKYE text."""
    try:
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        try:
            # Try with temperature parameter (MLX uses 'temperature' not 'temp')
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted, 
                max_tokens=max_tokens, 
                verbose=False,
                temperature=0.7,  # Lower temperature for more focused, less repetitive responses
            )
        except TypeError:
            # Fallback if temperature parameter not supported
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted, 
                max_tokens=max_tokens, 
                verbose=False,
            )
        
        # Clean response
        cleaned = clean_response(response)
        
        # Early stopping: if we detect a function call, stop any further generation
        # This prevents the model from continuing after a valid function call
        if "CALL_FUNC" in cleaned:
            # Extract text up to and including the first valid function call
            call_match = CALL_PATTERN.search(cleaned)
            if call_match:
                # Validate the JSON in the function call
                json_part = call_match.group(2)
                json_part = normalize_json_text(json_part)
                try:
                    payload = json.loads(json_part)
                    if payload.get("name"):
                        # Valid function call found - truncate at this point
                        cleaned = cleaned[:call_match.end()].strip()
                except:
                    pass  # If JSON is invalid, keep full text for error handling
        
        return cleaned
    except Exception as e:
        print(f"[S.K.Y.E. Error]: {e}")
        return "My apologies, Sir. Systems encountered an error."


# ---------------- MAIN HANDLER ----------------
def ai_response(user_input):
    """Master handler: rule-based first, then model. Executes CALL_FUNC if present."""
    rule = (
        rule_based_response(user_input) if "rule_based_response" in globals() else None
    )
    if rule:
        # log and return rule output
        log_interaction(user_input, rule, {"source": "rule"})
        return rule

    skye_out = skye_chat(user_input)
    name, args, narration = detect_function_call(skye_out)

    if name:
        # execute safe
        exec_result = call_function_safe(name, args)
        # combine narration + exec result (prefer narration if short)
        reply = ""
        if narration:
            reply += narration.strip()
            if not reply.endswith("."):
                reply += "."
            reply += " "
        # Append execution result (stringify if non-string)
        if isinstance(exec_result, (dict, list)):
            exec_text = json.dumps(exec_result, ensure_ascii=False)
        else:
            exec_text = str(exec_result)
        reply += exec_text
        # Log and return
        log_interaction(
            user_input,
            skye_out,
            {"detected_function": name, "args": args, "result": exec_text},
        )
        return reply
    else:
        # no callable found; just return model output
        log_interaction(user_input, skye_out, {"detected_function": None})
        return skye_out


# ---------------- minimal rule_based_response ----------------
def rule_based_response(speech):
    s = speech.lower()
    if any(g in s for g in ["hi skye", "hello skye", "good morning skye"]):
        return Greetings()
    if s.startswith("open "):
        target = s.split("open ", 1)[1].strip()
        # open site if known
        sites = {
            "youtube": "https://youtube.com",
            "google": "https://google.com",
            "spotify": "https://open.spotify.com",
        }
        if target in sites:
            webbrowser.open(sites[target])
            return f"Opening {target}, Sir..."
    return None


# ---------------- SOCKET SERVER ----------------
def handle_client(conn):
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            user_speech = data.decode().strip()
            print(f"[Client]: {user_speech}")
            reply = ai_response(user_speech)
            print(f"[Reply]: {reply}")
            try:
                # Send reply + delimiter to mark end of message
                conn.sendall(reply.encode() + b"...")
            except Exception as e:
                print(f"[Socket send error]: {e}")
                break


def serverstart():
    with socket.socket() as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"S.K.Y.E. server listening on {HOST}:{PORT}")
        while True:
            conn, addr = server_socket.accept()
            print(f"Client connected: {addr}")
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()


# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    serverstart()
