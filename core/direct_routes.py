"""Requests that map one-to-one onto a tool are handled here, without asking the
model to pick the tool.

The 4B model is unreliable at this: it answered "The alarm is set" without
setting anything, invented the time, and made up a joke tool. A regex that
recognises "set a timer for 10 minutes" and calls set_timer is exact, instant,
and cannot claim something it did not do. Anything that does not match falls
through to the model and the skills as before.

match(text) -> (tool_name, arguments) or None.
"""

import re

I = re.IGNORECASE
_END = r"[\s.!?]*$"

_APPS = ("index 0", "index zero", "index", "safari", "music", "apple music", "notes", "reminders", "messages",
         "mail", "calendar", "clock", "notion", "system settings", "settings", "finder", "facetime", "maps", "photos")
_SITES = ("youtube", "google", "gmail", "github", "notion", "wikipedia", "reddit", "netflix", "amazon", "linkedin", "chatgpt")

_RULES = [
    # "I want to study on Index 0" (Whisper may hear "index zero" / "index O")
    ("start_studying", re.compile(r"\b(?:study|studying|learn|learning|revise|revising)\b.{0,40}\bindex(?:\s+(?:0|zero|o))?\b|\bindex(?:\s+(?:0|zero|o))?\b.{0,25}\b(?:study|learn|revise)", I), lambda m: {}),

    ("morning_briefing", re.compile(r"\b(?:morning briefing|daily briefing|brief me|(?:my|the) briefing|start my day|(?:what|how)(?:'s| is| does)? my day (?:look|looking|going|like)|rundown of my day|what(?:'s| is) on today)\b", I), lambda m: {}),

    ("cancel_alarm", re.compile(r"\b(?:cancel|delete|remove|turn off|clear|stop)\b\s+(?:the |my |that )?(?P<w>alarm|timer)s?\b" + r"[\s\w]*" + _END, I), lambda m: {"which": m.group("w").lower()}),
    ("change_alarm", re.compile(r"\b(?:change|move|update|shift|push|reschedule)\b\s+(?:the |my )?alarm\b.*?\b(?:to|for)\s+(?P<t>.+?)" + _END, I), lambda m: {"which": "alarm", "time": m.group("t")}),

    ("set_timer", re.compile(r"\b(?:set|start|create|begin|put on|make)\b[^.?!]*?\btimer\b(?:\s+(?:for|of))?\s+(?P<d>\d.*?|an?\s.*?|half.*?)" + _END, I), lambda m: {"duration": m.group("d")}),
    ("set_timer", re.compile(r"^\W*(?:(?:please\s+)?(?:set|start|create|make)\s+)?(?:me\s+)?(?:a\s+)?(?P<d>\d+(?:\.\d+)?[\s-]*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?))\s+timer" + _END, I), lambda m: {"duration": m.group("d").replace("-", " ")}),
    ("set_timer", re.compile(r"^\W*(?:a\s+)?timer\s+(?:for|of)\s+(?P<d>.+?)" + _END, I), lambda m: {"duration": m.group("d")}),

    ("set_alarm", re.compile(r"\b(?:set|create|add|make)\b[^.?!]*?\balarm\b[^.?!]*?\b(?:for|at)\s+(?P<t>.+?)" + _END, I), lambda m: {"time": m.group("t")}),
    ("set_alarm", re.compile(r"\bwake me(?: up)?\s+(?:at|by)\s+(?P<t>.+?)" + _END, I), lambda m: {"time": m.group("t")}),

    ("set_reminder", re.compile(r"\bremind me\s+(?P<t>(?:at|in|on)\s+.+?)\s+to\s+(?P<task>.+?)" + _END, I), lambda m: {"time": m.group("t"), "task": m.group("task")}),
    ("set_reminder", re.compile(r"\bremind me\s+to\s+(?P<task>.+?)\s+(?P<t>(?:at|in|on|tomorrow|tonight|today|this)\b.+?)" + _END, I), lambda m: {"time": m.group("t"), "task": m.group("task")}),

    ("unread_messages", re.compile(r"\b(?:unread|new|missed) (?:messages?|texts?|imessages?)\b|\bany (?:new )?(?:messages?|texts?)\b|\bdo i have (?:any )?(?:new )?(?:messages?|texts?)\b", I), lambda m: {}),
    ("read_messages", re.compile(r"\bwhat did (?P<c>[\w' .-]+?) (?:text|message|say to) me\b|\b(?:messages?|texts?|imessages?) from (?P<c2>[\w' .-]+?)" + _END, I), lambda m: {"contact": (m.group("c") or m.group("c2") or "").strip()}),
    ("read_messages", re.compile(r"^(?!.*\b(?:send|write|reply|draft|compose)\b).*\b(?:read|check|show|open)\b.{0,20}\b(?:my |the )?(?:latest |last |recent |newest )?(?:messages?|texts?|imessages?)" + _END, I), lambda m: {}),

    ("music_now_playing", re.compile(r"\b(?:what(?:'s| is) (?:playing|this song)|what song is (?:this|playing)|which song is this)\b", I), lambda m: {}),
    ("music_control", re.compile(r"^\W*(?:please\s+)?(?:(?P<a>pause|resume|skip)\b(?:\s+(?:the\s+)?(?:music|song|track|this song|it))?|(?P<a2>next|previous|stop)\s+(?:the\s+)?(?:music|song|track)|go back(?: a song)?|(?:play )?(?:the )?next (?:song|track)|(?:play )?(?:the )?previous (?:song|track)|(?:set )?(?:the )?volume (?:to |at )?(?P<v>\d+))" + _END, I),
     lambda m: {"action": (f"volume {m.group('v')}" if m.group("v") else (m.group("a") or m.group("a2") or ("previous" if "back" in m.group(0).lower() or "previous" in m.group(0).lower() else "next")).lower())}),
    ("music_play", re.compile(r"^\W*(?:please\s+)?play\s+(?!.*\b(?:youtube|spotify|netflix|game|video)\b)(?:some\s+)?(?P<q>.+?)(?:\s+(?:on|in|using|with)\s+apple music)?" + _END, I), lambda m: {"query": re.sub(r"^(?:something by|songs? by|music by|the song|the album|the artist)\s+", "", m.group("q"), flags=I)}),

    ("open_in_safari", re.compile(r"^\W*(?:please\s+)?(?:open|go to|launch|show me|take me to)\s+(?P<t>.+?)\s+in safari" + _END, I), lambda m: {"target": m.group("t")}),
    ("open_app", re.compile(r"^\W*(?:please\s+)?(?:open|launch|start|switch to)\s+(?:the\s+)?(?P<a>" + "|".join(re.escape(a) for a in _APPS) + r")(?:\s+app)?" + _END, I), lambda m: {"name": m.group("a")}),
    ("open_in_safari", re.compile(r"^\W*(?:please\s+)?(?:open|go to)\s+(?P<t>" + "|".join(_SITES) + r"|[\w-]+\.(?:com|org|net|io|edu|au|co)(?:/\S*)?)" + _END, I), lambda m: {"target": m.group("t")}),

    ("create_note", re.compile(r"\b(?:create|make|start|write|take)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?note\s*(?:called|named|titled|about|on)?\s*[:,]?\s*(?P<t>.+?)(?:\s+(?:saying|that says|with the text|and write|:)\s+(?P<b>.+?))?" + _END, I), lambda m: {"title": m.group("t"), "body": m.group("b") or ""}),
]


def match(text: str):
    t = (text or "").strip()
    if not t or len(t.split()) > 30:
        return None
    for tool, rx, args in _RULES:
        m = rx.search(t)
        if m:
            return tool, args(m)
    return None
