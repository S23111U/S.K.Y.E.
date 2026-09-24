"""Pending-task list and SKYE's own performance diagnostics."""

import json
import os

from memory.tasks import format_due
from skills.shared import ROOT, get_tasks


def list_tasks() -> str:
    """Lists upcoming pending tasks and reminders."""
    upcoming = get_tasks().get_upcoming(10)
    if not upcoming:
        return "No pending tasks or reminders."
    return "; ".join(f"{t['description']} ({format_due(t['due_at'])})" for t in upcoming)


def complete_task(description: str) -> str:
    """Marks the most recent pending task matching a description as done."""
    match = get_tasks().find_recent_pending(description)
    if not match:
        return "No matching pending task found."
    get_tasks().mark_status(match["id"], "done")
    return f"Marked '{match['description']}' as done."


def get_diagnostics() -> str:
    """Reports SKYE's own performance diagnostics — guardrail triggers and tool reliability."""
    diagnostics_file = os.path.join(ROOT, "logs", "diagnostics.json")
    if not os.path.exists(diagnostics_file):
        return "No diagnostics recorded yet — nothing has gone through a consolidation pass."
    with open(diagnostics_file, "r") as f:
        report = json.load(f)
    turns = report.get("turns_analyzed", 0)
    if not turns:
        return "No diagnostics recorded yet."
    parts = [f"{turns} turns analyzed so far."]
    guardrails = report.get("guardrails") or {}
    if guardrails:
        top = sorted(guardrails.items(), key=lambda kv: kv[1], reverse=True)[:3]
        parts.append(
            "Most common guardrail triggers: "
            + ", ".join(f"{name} ({count})" for name, count in top) + "."
        )
    tools = report.get("tools") or {}
    if tools:
        tool_bits = [f"{name} ({t['success']} ok, {t['failure']} failed)" for name, t in tools.items()]
        parts.append("Tool reliability: " + ", ".join(tool_bits) + ".")
    return " ".join(parts)
