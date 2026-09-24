"""Backchannel fillers: a short spoken phrase ("Let me check that.") while the
real reply is still generating, so a 1.5-3.5 s pause doesn't read as a stall.

Real conversation doesn't go silent while the other person thinks — a quick
"mm-hmm" fills the gap. core_v2.py's stream_skye_response() calls pick_filler()
and, if it returns one, emits a "filler" frame for the socket handler to speak
immediately in the background while generation continues.

Placement matters more than the phrases themselves: a coin-flip on every turn
produces fillers back-to-back one moment and none for five turns the next,
which reads as a tic, not a person. Real backchannel is driven by (a) whether
a pause is actually about to happen — a one-word "thanks" gets answered
instantly, a real question or a tool call doesn't — and (b) not doing it
again right after the last one. Fillers are chosen by the *user's* mood, so
an empathetic beat ("Oh, I'm sorry to hear that.") comes before the answer to
bad news, and a pleased one before the answer to good news.
"""

import random
import re

FILLERS = {
    "calm": [
        "Hmm, let me check that.", "One moment.", "Let me look into that.",
        "Give me a second.", "Let me think.", "Right, one moment.", "Good question.",
        "Let me see.", "Okay, give me a moment.", "Alright, let me think about that.",
        "Interesting, one second.", "Sure, let me work that out.", "Hold on a moment.",
        "Let me have a think.", "Right, let me see.", "Bear with me a second.",
        "Ooh, let me think.", "Okay, let me pull that together.",
    ],
    "happy": [
        "Oh, wonderful.", "Ah, splendid.", "That's good to hear.", "Oh, lovely.",
        "Ah, brilliant.", "Oh, nice.", "Ha, great.",
    ],
    "sad": [
        "Oh, I'm sorry to hear that.", "Ah, that's unfortunate.", "Oh dear.",
        "Oh no, I'm sorry.", "Ah, that sounds hard.", "I'm sorry about that.",
    ],
    "concerned": [
        "Hmm, I understand.", "I see.", "Let me see.", "Hmm, let me look at that.",
        "Okay, let's have a look.", "Right, I hear you.",
    ],
}
# For requests that will run a tool (weather, calendar, Notion, search...).
TOOL_FILLERS = [
    "Let me check that for you.", "Checking now.", "On it.", "One moment, checking.",
    "Let me look that up.", "Pulling that up now.", "Sure, checking.", "Give me a second to check.",
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

    pool = TOOL_FILLERS if (mood == "calm" and is_tool_like) else FILLERS[mood]
    choices = [f for f in pool if f != _last_filler] or pool
    _last_filler = random.choice(choices)
    return _last_filler, mood
