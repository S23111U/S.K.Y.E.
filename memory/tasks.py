"""Structured task/reminder store for SKYE.

Deliberately separate from `memory/manager.py`'s semantic memory: tasks have
a due time, optional recurrence, and a status (pending/done/dismissed), none
of which fit the free-text "fact" shape `persistent_profile.json` and
`semantics.db` use. Same style as MemoryManager though — plain sqlite3, no
ORM, matching the project's minimal-dependency approach.
"""

import os
import sqlite3
from datetime import datetime, timedelta


def next_occurrence(time_str, fmt="%I:%M %p"):
    """Resolves a clock-time string like '7:30 AM' to the next datetime it
    occurs at — today if that time hasn't passed yet, otherwise tomorrow.

    Lives here (not in core_v2.py or mcp_server/server.py) because both the
    main server and the MCP tool-execution process need it: it's how
    set_alarm/set_reminder turn a spoken time into a due_at for TaskStore,
    and neither process should import the other (core_v2.py loads the LLM
    at import time; mcp_server/server.py must stay a lightweight, separate
    process, not a second copy of that).
    """
    parsed = datetime.strptime(time_str, fmt)
    due_at = datetime.now().replace(
        hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0
    )
    if due_at <= datetime.now():
        due_at += timedelta(days=1)
    return due_at


def format_due(due_at_iso: str) -> str:
    """Human-readable due time — gets read aloud by TTS and injected into the
    system prompt, so a raw ISO timestamp would be a real regression (SKYE
    speaking "twenty twenty-six dash zero nine..." verbatim).
    """
    due = datetime.fromisoformat(due_at_iso)
    today = datetime.now().date()
    if due.date() == today:
        return f"today at {due.strftime('%-I:%M %p')}"
    if due.date() == today + timedelta(days=1):
        return f"tomorrow at {due.strftime('%-I:%M %p')}"
    return due.strftime("%A at %-I:%M %p")


class TaskStore:
    def __init__(self, root_dir):
        self.db_path = os.path.join(root_dir, "memory", "tasks.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        # WAL mode: this file is now read/written from two separate
        # processes (the main server and the MCP tool-execution process),
        # not just multiple threads in one process — WAL is sqlite's
        # standard answer for concurrent multi-process access without
        # "database is locked" errors.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                due_at TEXT NOT NULL,
                recurrence TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_task(self, description, due_at, recurrence=None, source="explicit"):
        due_iso = due_at.isoformat() if isinstance(due_at, datetime) else due_at
        cursor = self.conn.execute(
            "INSERT INTO tasks (description, due_at, recurrence, source) VALUES (?, ?, ?, ?)",
            (description, due_iso, recurrence, source),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_due(self, now: datetime = None):
        now = now or datetime.now()
        rows = self.conn.execute(
            "SELECT id, description, due_at, recurrence, source FROM tasks "
            "WHERE status = 'pending' AND due_at <= ? ORDER BY due_at ASC",
            (now.isoformat(),),
        ).fetchall()
        return [
            {"id": r[0], "description": r[1], "due_at": r[2], "recurrence": r[3], "source": r[4]}
            for r in rows
        ]

    def get_upcoming(self, limit=5, now: datetime = None):
        now = now or datetime.now()
        rows = self.conn.execute(
            "SELECT id, description, due_at, recurrence FROM tasks "
            "WHERE status = 'pending' ORDER BY due_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "description": r[1], "due_at": r[2], "recurrence": r[3]}
            for r in rows
        ]

    def mark_status(self, task_id, status):
        self.conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        self.conn.commit()

    def reschedule(self, task_id):
        """Advances a recurring task's due_at by one period and resets it to pending.

        No-op (falls back to mark_status('done')) for tasks without a
        recognized recurrence — a one-off task simply completes.
        """
        row = self.conn.execute(
            "SELECT due_at, recurrence FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return
        due_at, recurrence = row
        step = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1)}.get(recurrence)
        if not step:
            self.mark_status(task_id, "done")
            return
        next_due = datetime.fromisoformat(due_at) + step
        self.conn.execute(
            "UPDATE tasks SET due_at = ?, status = 'pending' WHERE id = ?",
            (next_due.isoformat(), task_id),
        )
        self.conn.commit()

    def find_recent_pending(self, description_contains=None):
        """Best-effort lookup for the `complete_task` tool: most recent pending
        task, optionally filtered by a substring match against its description.
        """
        query = "SELECT id, description, due_at FROM tasks WHERE status = 'pending'"
        params = ()
        if description_contains:
            query += " AND description LIKE ?"
            params = (f"%{description_contains}%",)
        query += " ORDER BY created_at DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        return {"id": row[0], "description": row[1], "due_at": row[2]} if row else None
