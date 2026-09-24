"""Morning briefing: weather, today's calendar, today's to-dos and today's
alarms/reminders/timers, all in one spoken summary. Cross-cutting (touches
three other skills), so it lives here rather than under any one of them."""

import os
import sys
from datetime import datetime

from skills import canvas
from skills.core_tools.time_weather import get_weather
from skills.shared import get_tasks


def morning_briefing() -> str:
    """The day at a glance: greeting, weather, calendar events, to-dos and alarms."""
    now = datetime.now()
    greet = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 18 else "Good evening"
    parts = [f"{greet}."]
    blocks = []
    def _today_calendar():
        from skills.calendar.tools import list_events
        return list_events("today")

    def _today_todos():
        from skills.todo.tools import list_todos
        return list_todos("for today") if os.getenv("NOTION_TOKEN") else ""

    for label, fn in (
        ("weather", lambda: get_weather("")),
        ("calendar", _today_calendar),
        ("todos", _today_todos),
    ):
        try:
            out = fn()
        except Exception as e:
            print(f"[briefing] {label} failed: {e}", file=sys.stderr)
            continue
        out, payload = canvas.split(out or "")
        if payload:
            blocks.extend(payload.get("blocks", []))
        if out and "could not" not in out.lower() and "not connected" not in out.lower():
            parts.append(out if out.endswith((".", "!", "?")) else out + ".")
    end = now.replace(hour=23, minute=59)
    todays = [t for t in get_tasks().get_upcoming(20) if datetime.fromisoformat(t["due_at"]) <= end]
    if todays:
        parts.append("Also today: " + "; ".join(f"{t['description']} at {datetime.fromisoformat(t['due_at']).strftime('%-I:%M %p')}" for t in todays[:4]) + ".")
    text = " ".join(parts)
    return canvas.attach(text, "Your day", *blocks) if blocks else text

