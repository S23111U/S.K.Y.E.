"""Always-on tools: visible to the model on every turn (see
prompts/skye_persona.txt's own tool list), not gated behind a skill match —
the small model needs a short, permanent core it can rely on rather than
guessing whether "what time is it" counts as some skill's territory.

This set barely changes, so — unlike a routed skill — its tool list in
prompts/skye_persona.txt is still hand-written rather than generated; see
skills/base.py's `base_manifest` field for why.
"""

import re

from skills.base import Skill
from skills.core_tools.alarms import cancel_alarm, change_alarm, set_alarm, set_reminder, set_timer
from skills.core_tools.briefing import morning_briefing
from skills.core_tools.media import analyze_video, show_media
from skills.core_tools.opens import open_calendar_web, open_spotify, open_youtube
from skills.core_tools.tasks_diag import complete_task, get_diagnostics, list_tasks
from skills.core_tools.time_weather import get_weather, tell_date, tell_time
from skills.core_tools.web_search import fetch_news, web_search
from skills.routing_helpers import END, I

SKILL = Skill(
    name="core_tools",
    always_on=True,
    tools=[
        tell_time, tell_date, get_weather,
        set_alarm, set_timer, change_alarm, cancel_alarm, set_reminder,
        web_search, fetch_news,
        open_youtube, open_spotify, open_calendar_web,
        list_tasks, complete_task, get_diagnostics,
        show_media, analyze_video,
        morning_briefing,
    ],
    direct_routes=[
        ("morning_briefing", re.compile(
            r"\b(?:morning briefing|daily briefing|brief me|(?:my|the) briefing|start my day|"
            r"(?:what|how)(?:'s| is| does)? my day (?:look|looking|going|like)|rundown of my day|what(?:'s| is) on today)\b", I),
         lambda m: {}),

        ("cancel_alarm", re.compile(r"\b(?:cancel|delete|remove|turn off|clear|stop)\b\s+(?:the |my |that )?(?P<w>alarm|timer)s?\b[\s\w]*" + END, I),
         lambda m: {"which": m.group("w").lower()}),
        ("change_alarm", re.compile(r"\b(?:change|move|update|shift|push|reschedule)\b\s+(?:the |my )?alarm\b.*?\b(?:to|for)\s+(?P<t>.+?)" + END, I),
         lambda m: {"which": "alarm", "time": m.group("t")}),

        ("set_timer", re.compile(r"\b(?:set|start|create|begin|put on|make)\b[^.?!]*?\btimer\b(?:\s+(?:for|of))?\s+(?P<d>\d.*?|an?\s.*?|half.*?)" + END, I),
         lambda m: {"duration": m.group("d")}),
        ("set_timer", re.compile(r"^\W*(?:(?:please\s+)?(?:set|start|create|make)\s+)?(?:me\s+)?(?:a\s+)?(?P<d>\d+(?:\.\d+)?[\s-]*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?))\s+timer" + END, I),
         lambda m: {"duration": m.group("d").replace("-", " ")}),
        ("set_timer", re.compile(r"^\W*(?:a\s+)?timer\s+(?:for|of)\s+(?P<d>.+?)" + END, I),
         lambda m: {"duration": m.group("d")}),

        ("set_alarm", re.compile(r"\b(?:set|create|add|make)\b[^.?!]*?\balarm\b[^.?!]*?\b(?:for|at)\s+(?P<t>.+?)" + END, I),
         lambda m: {"time": m.group("t")}),
        ("set_alarm", re.compile(r"\bwake me(?: up)?\s+(?:at|by)\s+(?P<t>.+?)" + END, I),
         lambda m: {"time": m.group("t")}),

        ("set_reminder", re.compile(r"\bremind me\s+(?P<t>(?:at|in|on)\s+.+?)\s+to\s+(?P<task>.+?)" + END, I),
         lambda m: {"time": m.group("t"), "task": m.group("task")}),
        ("set_reminder", re.compile(r"\bremind me\s+to\s+(?P<task>.+?)\s+(?P<t>(?:at|in|on|tomorrow|tonight|today|this)\b.+?)" + END, I),
         lambda m: {"time": m.group("t"), "task": m.group("task")}),

        ("analyze_video", re.compile(
            r"\b(?:watch|analy[sz]e|summari[sz]e|break down|go through|look at|explain|review|check out|tell me about|what(?:'s| is| does)|transcribe)\b.{0,60}?\b(?:video|youtube|clip|tutorial)\b|"
            r"\b(?:video|youtube)\b.{0,20}\b(?:i(?:'m| am) (?:watching|on)|(?:that|which) (?:is )?(?:open|playing)|open right now)\b|(?:youtu\.be|youtube\.com)/\S+", I),
         lambda m: {"request": re.sub(r"https?://\S+", "", m.string).strip(),
                    "url": (re.search(r"https?://\S*(?:youtu\.be|youtube\.com)/\S+", m.string) or [""])[0]
                    if re.search(r"https?://\S*(?:youtu\.be|youtube\.com)/\S+", m.string) else ""}),

        ("show_media", re.compile(r"\b(?:show|display|put up|pull up)\b.{0,20}(?P<u>https?://\S+)", I), lambda m: {"url": m.group("u")}),
    ],
)
