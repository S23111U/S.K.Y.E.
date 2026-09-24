"""The "calendar" skill: SKYE's Google Calendar (primary calendar)."""

import re

from skills.base import Skill
from skills.calendar.tools import (
    add_event, cancel_event, event_alerts, find_free_time, list_events,
    move_event, next_event, undo_calendar,
)
from skills.routing_helpers import END, I, PERIOD, period_group

SKILL = Skill(
    name="calendar",
    utterances=["what's on my calendar tomorrow", 'schedule a meeting on Friday', 'am I free this afternoon',
                'move my appointment to Monday', 'cancel my event', 'when is my next meeting',
                'what does my week look like', 'am I busy on Saturday', 'do I have anything on this weekend'],
    pattern=re.compile(
        r"\b(calendar|schedule|scheduled|agenda|appointments?|meetings?|events?|free time|free slot|"
        r"am i free|what'?s next|next event)\b|"
        r"\b(?:book|put|add|move|reschedule|cancel|push)\b.{0,50}\b(?:calendar|meeting|appointment|event)\b|"
        r"\bwhat do i have (?:on|today|tomorrow|this|next)\b|\b(?:re)?schedule\b|"
        r"\b(?:move|push|shift)\b.{0,50}\b(?:to|until|for)\b.{0,25}(?:tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|\d\s?(?:am|pm)|\d{1,2}:\d{2}|noon)|"
        r"\bcancel (?:the|my|that|our)\b(?!.{0,30}\b(?:alarm|reminder|subscription|order)\b)",
        re.IGNORECASE,
    ),
    manifest=(
        "Calendar tools (his Google Calendar; alarms and reminders are different): "
        'list_events{"period","until"} · next_event{} · add_event{"title","when","duration_minutes"} · '
        'move_event{"title","when"} · cancel_event{"title"} · find_free_time{"day","minutes"} · '
        "undo_calendar{}. period is today, tomorrow, this week, next week, a weekday or a date; for a span put the first day in period and the last day in until. If he asks about his calendar without a day, use today; never ask which day. Anything about his calendar or events uses these tools, never web_search. "
        'when is the day and time as he said it, such as "tomorrow at 3pm". Use undo_calendar when '
        "he says undo or scratch that after a calendar change. To just open the calendar website use open_calendar_web."
    ),
    examples=[
        ("what's on my calendar today?", '{"name": "list_events", "arguments": {"period": "today"}}'),
        ("schedule a dentist appointment tomorrow at 3pm", '{"name": "add_event", "arguments": {"title": "Dentist appointment", "when": "tomorrow at 3pm"}}'),
        ("move my standup to Friday at 10", '{"name": "move_event", "arguments": {"title": "standup", "when": "Friday at 10am"}}'),
    ],
    tools=[list_events, next_event, add_event, move_event, cancel_event, find_free_time, undo_calendar],
    internal_tools=[event_alerts],
    direct_routes=[
        ("next_event", re.compile(r"\b(?:what(?:'s| is)|when(?:'s| is)) my next (?:event|meeting|appointment)\b|\bnext (?:event|meeting|appointment)\b.{0,15}\?", I), lambda m: {}),
        ("list_events", re.compile(
            r"\b(?:what(?:'s| is| do i have)? (?:on )?(?:my )?(?:calendar|schedule|agenda)|what do i have|"
            r"(?:check|show|read|tell me)(?: me)? (?:my )?(?:calendar|schedule|agenda)|am i (?:busy|free)|"
            r"(?:do i have|any) (?:any )?(?:events?|meetings?|appointments?|plans?|lectures?)) ?(?:on |for |this |next )?" + PERIOD + r"\b", I),
         lambda m: {"period": period_group(m)}),
        ("list_events", re.compile(r"\bwhat(?:'s| is)? on my (?:calendar|schedule)" + END, I), lambda m: {"period": "today"}),
    ],
)
