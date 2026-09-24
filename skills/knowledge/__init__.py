"""The "knowledge" skill: SKYE's Notion learning notes (courses, lectures,
tutorials, read only)."""

import re

from skills.base import Skill
from skills.knowledge.tools import read_note, search_notes

SKILL = Skill(
    name="knowledge",
    utterances=['what did I write in my lecture notes', 'summarise my notes on algorithms week three',
                'find my study notes about networks', 'read my tutorial notes'],
    pattern=re.compile(
        r"\b(my notes|notes? (?:on|about|for|from)|lectures?|tutorials?|study|studying|revise|revision|"
        r"week \d+|comp ?\d{4}|info ?\d{4}|learning notes|notebook|what did i (?:write|note))\b",
        re.IGNORECASE,
    ),
    manifest=(
        "Notes tools (his Notion learning notes: courses, lectures, tutorials): "
        'search_notes{"query"} · read_note{"query"}. To answer a question about his notes, '
        'read_note with the course and week, such as "DSA week 3"; what it returns will be '
        "summarised for him."
    ),
    examples=[
        ("what did I write about CSMA in networks week 3?", '{"name": "read_note", "arguments": {"query": "networks lecture week 3"}}'),
    ],
    tools=[search_notes, read_note],
)
