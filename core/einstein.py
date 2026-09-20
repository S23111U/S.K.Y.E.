"""Einstein mode: hand one hard question to Gemini with extended thinking.

Everything else in SKYE stays on the device; this is the one deliberate,
opt-in exception, so it is narrow. Only the question (with the trigger phrase
removed) goes out, plus a short background note the user can switch off with
EINSTEIN_CONTEXT=off: a few interests/study areas from the profile and the
previous question, so a follow-up makes sense. No memories, Notion data,
calendar or location are ever sent.

The reply has two parts: a short spoken SUMMARY and a long DETAIL that the
browser shows on screen.
"""

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Tried in order; a model that is over quota or busy just falls through.
MODELS = [m.strip() for m in os.getenv(
    "EINSTEIN_MODELS", "gemini-3.6-flash,gemini-3-flash-preview,gemini-2.5-flash").split(",") if m.strip()]
TIMEOUT_S = 120
PROFILE_KEYS = ("academic_background", "interest", "current_topic_of_interest")

# Names that switch the mode on, and phrases that switch it off.
_NAMES = r"(?:einstein(?:'s)?(?: mode)?|genius mode|deep[- ]think(?:ing)? mode|thinking mode|professor mode)"
MODE_RE = re.compile(rf"\b{_NAMES}\b", re.IGNORECASE)
_OFF_VERBS = r"(?:switch|turn|shut|put|set|take|exit|leave|stop|disable|end|cancel|close|quit|drop|kill|finish|deactivate)"
OFF_RE = re.compile(
    rf"\b(?:{_OFF_VERBS}\b.{{0,30}}\b{_NAMES}|"
    rf"{_NAMES}\b.{{0,15}}\b(?:off|over|done|finished|stop|disabled|deactivated)|"
    rf"(?:no more|enough|done with|finished with) {_NAMES})",
    re.IGNORECASE,
)
_ON_WORDS = re.compile(r"\b(?:switch|turn|go|change|move)\s+(?:on|to|into)\b|\b(?:activate|enable|use|start|begin)\b", re.IGNORECASE)


def is_off(text: str) -> bool:
    """"Switch Einstein mode off" — but not "switch to Einstein mode"."""
    return bool(OFF_RE.search(text)) and not _ON_WORDS.search(text)


# Only meaningful while the mode is on ("back to normal", "normal mode").
NORMAL_RE = re.compile(
    r"\b(?:back to normal|go back to normal|normal mode|regular mode|standard mode|quick mode|"
    r"switch back|(?:that'?s|that is) enough (?:thinking|of that)|stop (?:thinking|overthinking))\b",
    re.IGNORECASE,
)
TRIGGER_RE = re.compile(
    rf"\b(?:{_NAMES}|einstein|think (?:really |very )?(?:hard|deeply|carefully)|think it through|"
    r"deep(?:ly)? (?:think|dive|analy[sz]e)|in[- ]depth (?:explanation|analysis))\b",
    re.IGNORECASE,
)
FILLERS = [
    "Let me think this through properly. Give me a moment.",
    "Okay, let me really think about this.",
    "Right, give me a moment to think this through.",
    "Let me work through this carefully.",
]
FILLER = FILLERS[0]

SYSTEM = (
    "You are the reasoning engine behind a voice assistant. Think carefully, then give a rigorous, "
    "well-structured, detailed answer: define terms, show the key steps or reasoning, give examples, "
    "and note caveats or where experts disagree. Write in clear Markdown-light text (short headings "
    "and bullet points are fine). Do not use LaTeX or dollar signs: write maths in plain text or Unicode "
    "(for example F(n) = F(n-1) + F(n-2), φ, √5).\n\n"
    "Format your final answer EXACTLY as:\n"
    "SUMMARY: <2 to 4 plain spoken sentences that capture the answer; no markdown, no lists>\n"
    "DETAIL:\n<the full detailed answer>"
)


def background(profile: dict | None, previous_question: str | None) -> str:
    if os.getenv("EINSTEIN_CONTEXT", "on").lower() == "off":
        return ""
    bits = []
    user = (profile or {}).get("user", {})
    for key in PROFILE_KEYS:
        vals = user.get(key)
        vals = [vals] if isinstance(vals, str) else list(vals or [])
        bits += [str(v)[:120] for v in vals[:2]]
    out = []
    if bits:
        out.append("About the asker: " + "; ".join(bits) + ".")
    if previous_question:
        out.append("Their previous question was: " + previous_question[:200])
    return " ".join(out)


def clean_question(text: str) -> str:
    q = TRIGGER_RE.sub("", text)
    q = re.sub(r"^[\s,.:;!-]+|\s+$", "", q)
    q = re.sub(r"^(?:and |so |now |please |can you |could you )+", "", q, flags=re.IGNORECASE)
    return q.strip() or text.strip()


def _config(model: str) -> dict:
    # Gemini 3 takes a thinking level; 2.5 takes a token budget.
    if model.startswith("gemini-2.5"):
        return {"thinkingConfig": {"thinkingBudget": 8192}}
    return {"thinkingConfig": {"thinkingLevel": "high"}}


def _parse(text: str):
    m = re.search(r"SUMMARY:\s*(.*?)\s*DETAIL:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    first = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(first[:3]), text.strip()


def think(question: str, context: str = ""):
    """(summary, detail, model) — raises RuntimeError with a spoken-safe reason."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("no key")
    prompt = (f"[Background]\n{context}\n\n" if context else "") + f"[Question]\n{question}"
    last = "unavailable"
    for model in MODELS:
        try:
            r = requests.post(
                ENDPOINT.format(model=model), headers={"x-goog-api-key": key}, timeout=TIMEOUT_S,
                json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                      "contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": _config(model)})
        except requests.RequestException as e:
            last = str(e)[:80]
            continue
        if r.status_code != 200:
            last = f"{model} {r.status_code}"
            continue
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, ValueError):
            last = f"{model} empty"
            continue
        if text.strip():
            summary, detail = _parse(text)
            return summary, detail, model
    raise RuntimeError(last)
