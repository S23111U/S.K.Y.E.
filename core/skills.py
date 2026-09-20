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
        "utterances": ['how much money did I spend last month', 'what is my account balance', 'add an expense of twelve for lunch', 'am I within budget', 'what did I buy this week', 'how much did I pay for rent'],
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
        "utterances": ["what's on my to-do list", 'add buy milk to my list', 'mark that task as done', 'what do I need to do today', 'what are my priorities', 'remind me what is pending on my list'],
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
        "utterances": ["what's on my calendar tomorrow", 'schedule a meeting on Friday', 'am I free this afternoon', 'move my appointment to Monday', 'cancel my event', 'when is my next meeting', 'what does my week look like', 'am I busy on Saturday', 'do I have anything on this weekend'],
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
            'list_events{"period","until"} · next_event{} · add_event{"title","when","duration_minutes"} · '
            'move_event{"title","when"} · cancel_event{"title"} · find_free_time{"day","minutes"} · '
            "undo_calendar{}. period is today, tomorrow, this week, next week, a weekday or a date; for a span put the first day in period and the last day in until. If he asks about his calendar without a day, use today; never ask which day. Anything about his calendar or events uses these tools, never web_search. "
            'when is the day and time as he said it, such as "tomorrow at 3pm". Use undo_calendar when '
            "he says undo or scratch that after a calendar change. To just open the calendar website use open_calendar_web."
        ),
        "examples": [
            ("what's on my calendar today?", '{"name": "list_events", "arguments": {"period": "today"}}'),
            ("schedule a dentist appointment tomorrow at 3pm", '{"name": "add_event", "arguments": {"title": "Dentist appointment", "when": "tomorrow at 3pm"}}'),
            ("move my standup to Friday at 10", '{"name": "move_event", "arguments": {"title": "standup", "when": "Friday at 10am"}}'),
        ],
    },
    # Handled directly by core/einstein.py (Gemini extended thinking), not by a
    # local tool call, so it has no manifest; the pattern is its explicit trigger.
    "einstein": {
        "ui": "einstein",
        "direct": True,
        "pattern": re.compile(
            r"\b(?:einstein|genius mode|deep[- ]think(?:ing)? mode|thinking mode|professor mode)\b|\bthink (?:really |very )?(?:hard|deeply|carefully)\b|\bthink it through\b|"
            r"\bdeep(?:ly)? (?:think|dive|analy[sz]e)\b|\bin[- ]depth (?:explanation|analysis)\b",
            re.IGNORECASE,
        ),
        "manifest": "",
        "examples": [],
    },
    "mac": {
        "ui": "default",
        "weight": 2,   # its phrases are specific ("add ... to my note"); beat a stray finance/todo word
        "utterances": ["play some music on apple music", "add a note to my apple notes", "create a note about the meeting",
                       "open the reminders app", "open safari and search for something", "what reminders do I have in the reminders app",
                       "change my alarm to a different time", "cancel my timer"],
        "pattern": re.compile(
            r"\b(?:apple music|apple notes|reminders app|safari|(?:the |my )?music|(?:a |an )?(?:new )?note(?: called| about| on)|"
            r"(?:add|append) .{0,40} to (?:my |the )?note|alarm|timer|snooze)\b",
            re.IGNORECASE,
        ),
        "manifest": (
            "Mac tools (his MacBook apps): music_play{\"query\"} · music_control{\"action\"} · music_now_playing{} · "
            "open_in_safari{\"target\"} · open_app{\"name\"} · create_note{\"title\",\"body\"} · add_to_note{\"title\",\"text\"} · "
            "find_notes{\"query\"} · read_apple_note{\"title\"} · list_mac_reminders{} · complete_mac_reminder{\"title\"} · "
            "set_timer{\"duration\",\"label\"} · set_alarm{\"time\"} · change_alarm{\"which\",\"time\"} · cancel_alarm{\"which\"}. "
            "music_control action is pause, resume, next, previous or \"volume 40\". "
            "Never say something was done unless you called the tool for it."
        ),
        "examples": [
            ("add eggs to my groceries note", '{"name": "add_to_note", "arguments": {"title": "groceries", "text": "eggs"}}'),
            ("open the news in safari", '{"name": "open_in_safari", "arguments": {"target": "news"}}'),
        ],
    },
    "knowledge": {
        "utterances": ['what did I write in my lecture notes', 'summarise my notes on algorithms week three', 'find my study notes about networks', 'read my tutorial notes'],
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


EMBED_MIN = 0.62      # cosine to the best matching example
EMBED_MARGIN = 0.06   # ...and this far ahead of the next skill
_centroids = None


def _embed_route(text, embed):
    """Fallback for phrasings the regex misses ("what does my week look like"):
    nearest skill by sentence embedding, using the model SKYE already has loaded.
    Deliberately conservative — a wrong skill is worse than no skill."""
    global _centroids
    import numpy as np
    if _centroids is None:
        _centroids = {}
        for name, s in SKILLS.items():
            ex = s.get("utterances")
            if ex:
                _centroids[name] = np.asarray(embed(ex), dtype=np.float32)
    q = np.asarray(embed([text]), dtype=np.float32)[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = {}
    for name, m in _centroids.items():
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        scores[name] = float((m @ q).max())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] >= EMBED_MIN and (len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= EMBED_MARGIN):
        return ranked[0][0]
    return None


def route_skill(text: str, sticky: str | None = None, embed=None):
    """Skill name for a message, or None. `sticky` is the skill of the last
    turn(s): "undo that" or a bare follow-up should stay in it. `embed` (list of
    strings -> vectors) enables the embedding fallback."""
    if SKILLS["einstein"]["pattern"].search(text):
        return "einstein"
    scores = {name: len(s["pattern"].findall(text)) * s.get("weight", 1) for name, s in SKILLS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if sticky and UNDO_RE.search(text):
        return sticky
    if embed is not None and len(text.split()) >= 3:
        return _embed_route(text, embed)
    return None


def manifest_text(skill: str) -> str:
    s = SKILLS[skill]
    lines = [s["manifest"], ""]
    for user, call in s["examples"]:
        lines += [f"User: {user}", f"SKYE: CALL_FUNC: {call}", ""]
    return "\n".join(lines).rstrip()
