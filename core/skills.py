"""Skills: which specialised tools SKYE gets to see for a given request.

A 4B model given ~20 tools at once picks badly, so the base persona lists only
the always-on tools and a skill's tools are added *for that turn only*, when
the user's message is about that skill. A skill also names the UI mode the
browser should switch to while it is answering (browser.html `setMode`), which
is how the finance / knowledge / health colours will follow the conversation.

Adding a skill = one entry here (pattern, tools, examples, UI mode) plus the
tools themselves on the MCP server. Routing is by regex on purpose: it is
instant, deterministic and easy to debug from the timing log ("skill" field).
"""

import re

SKILLS = {
    "finance": {
        "ui": "finance",
        "pattern": re.compile(
            r"\b(spent|spend|spending|expenses?|budget|paid|bought|purchased|costs?|dollars?|bucks|"
            r"balance|income|salary|rent|groceries|savings|money|afford|transactions?)\b|\$\s?\d",
            re.IGNORECASE,
        ),
        "manifest": (
            "Finance tools (his Notion finance tracker): "
            'expense_summary{"period","category"} · list_expenses{"period","category"} · '
            'add_expense{"name","amount","category"} · budget_status{} · account_balance{} · '
            "undo_last_entry{}. For any question about how much he spent (in total, on a category, "
            "or in a period) use expense_summary; use list_expenses only when he asks to see or "
            "list individual expenses. period can be \"this month\", \"last month\", a month name, "
            '"today", "this week" or "all time". Use add_expense when he says he spent or bought '
            "something, with amount as a plain number. Use undo_last_entry when he says undo, "
            "remove that or scratch that entry."
        ),
        "examples": [
            ("how much have I spent this month?", '{"name": "expense_summary", "arguments": {"period": "this month"}}'),
            ("how much did I spend on food last month?", '{"name": "expense_summary", "arguments": {"period": "last month", "category": "food"}}'),
            ("I spent 12 dollars on lunch at Subway", '{"name": "add_expense", "arguments": {"name": "Subway lunch", "amount": "12", "category": "Food"}}'),
        ],
    },
    "todo": {
        "ui": "default",
        "pattern": re.compile(
            r"\b(to-?do|to do|todos?|my list|task list|priority)\b|"
            r"\badd .{0,40}\bto (?:my |the )?(?:list|to-?do)|\bmark .{0,40}\b(?:done|complete|finished)\b",
            re.IGNORECASE,
        ),
        "manifest": (
            "To-do tools (his Notion to-do list; alarms and reminders are different): "
            'list_todos{"status"} · add_todo{"name","priority"} · '
            'update_todo{"name","status","priority"} · undo_last_entry{}. '
            "status is one of Pending, For today, In Progress, Done; priority is High, Medium or Low."
        ),
        "examples": [
            ("what's on my to-do list?", '{"name": "list_todos", "arguments": {}}'),
            ("what do I have to do today?", '{"name": "list_todos", "arguments": {"status": "for today"}}'),
            ("mark the DSA assignment as done", '{"name": "update_todo", "arguments": {"name": "DSA assignment", "status": "Done"}}'),
        ],
    },
    "calendar": {
        "ui": "calendar",
        "pattern": re.compile(
            r"\b(calendar|schedule|scheduled|agenda|appointments?|meetings?|events?|free time|free slot|"
            r"am i free|what'?s next|next event)\b|"
            r"\b(?:book|put|add|move|reschedule|cancel|push)\b.{0,50}\b(?:calendar|meeting|appointment|event)\b|"
            r"\bwhat do i have (?:on|today|tomorrow|this|next)\b|\b(?:re)?schedule\b|"
            r"\b(?:move|push|shift)\b.{0,50}\b(?:to|until|for)\b.{0,25}(?:tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|\d\s?(?:am|pm)|\d{1,2}:\d{2}|noon)|"
            r"\bcancel (?:the|my|that|our)\b(?!.{0,30}\b(?:alarm|reminder|subscription|order)\b)",
            re.IGNORECASE,
        ),
        "manifest": (
            "Calendar tools (his Google Calendar; alarms and reminders are different): "
            'list_events{"period"} · next_event{} · add_event{"title","when","duration_minutes"} · '
            'move_event{"title","when"} · cancel_event{"title"} · find_free_time{"day","minutes"} · '
            "undo_calendar{}. period is today, tomorrow, this week, next week, a weekday or a date. "
            'when is the day and time as he said it, such as "tomorrow at 3pm". Use undo_calendar when '
            "he says undo or scratch that after a calendar change. To just open the calendar website use open_calendar_web."
        ),
        "examples": [
            ("what's on my calendar today?", '{"name": "list_events", "arguments": {"period": "today"}}'),
            ("schedule a dentist appointment tomorrow at 3pm", '{"name": "add_event", "arguments": {"title": "Dentist appointment", "when": "tomorrow at 3pm"}}'),
            ("move my standup to Friday at 10", '{"name": "move_event", "arguments": {"title": "standup", "when": "Friday at 10am"}}'),
        ],
    },
    "knowledge": {
        "ui": "knowledge",
        "pattern": re.compile(
            r"\b(my notes|notes? (?:on|about|for|from)|lectures?|tutorials?|study|studying|revise|revision|"
            r"week \d+|comp ?\d{4}|info ?\d{4}|learning notes|notebook|what did i (?:write|note))\b",
            re.IGNORECASE,
        ),
        "manifest": (
            "Notes tools (his Notion learning notes: courses, lectures, tutorials): "
            'search_notes{"query"} · read_note{"query"}. To answer a question about his notes, '
            'read_note with the course and week, such as "DSA week 3"; what it returns will be '
            "summarised for him."
        ),
        "examples": [
            ("what did I write about CSMA in networks week 3?", '{"name": "read_note", "arguments": {"query": "networks lecture week 3"}}'),
        ],
    },
}

UNDO_RE = re.compile(r"\b(undo|scratch that|take that back|remove that)\b", re.IGNORECASE)
STICKY_TURNS = 2


def route_skill(text: str, sticky: str | None = None):
    """Skill name for a message, or None. `sticky` is the skill of the last
    turn(s): "undo that" or a bare follow-up should stay in it."""
    scores = {name: len(s["pattern"].findall(text)) for name, s in SKILLS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if sticky and UNDO_RE.search(text):
        return sticky
    return None


def manifest_text(skill: str) -> str:
    s = SKILLS[skill]
    lines = [s["manifest"], ""]
    for user, call in s["examples"]:
        lines += [f"User: {user}", f"SKYE: CALL_FUNC: {call}", ""]
    return "\n".join(lines).rstrip()
