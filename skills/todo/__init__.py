"""The "todo" skill: SKYE's Notion to-do list."""

import re

from skills.base import Skill
from skills.todo.tools import add_todo, list_todos, todo_nudge, update_todo

SKILL = Skill(
    name="todo",
    ui="default",
    utterances=["what's on my to-do list", 'add buy milk to my list', 'mark that task as done',
                'what do I need to do today', 'what are my priorities', 'remind me what is pending on my list'],
    pattern=re.compile(
        r"\b(to-?do|to do|todos?|my list|task list|priority)\b|"
        r"\badd .{0,40}\bto (?:my |the )?(?:list|to-?do)|\bmark .{0,40}\b(?:done|complete|finished)\b",
        re.IGNORECASE,
    ),
    manifest=(
        "To-do tools (his Notion to-do list; alarms and reminders are different): "
        'list_todos{"status"} · add_todo{"name","priority"} · '
        'update_todo{"name","status","priority"} · undo_last_entry{}. '
        "status is one of Pending, For today, In Progress, Done; priority is High, Medium or Low."
    ),
    examples=[
        ("what's on my to-do list?", '{"name": "list_todos", "arguments": {}}'),
        ("what do I have to do today?", '{"name": "list_todos", "arguments": {"status": "for today"}}'),
        ("mark the DSA assignment as done", '{"name": "update_todo", "arguments": {"name": "DSA assignment", "status": "Done"}}'),
    ],
    tools=[list_todos, add_todo, update_todo],
    internal_tools=[todo_nudge],
    direct_routes=[
        ("list_todos", re.compile(r"\b(?:what(?:'s| is)|show|read|list|check)\b.{0,25}\b(?:to-?do(?:s| list)?|task list)\b", re.IGNORECASE), lambda m: {}),
    ],
)
