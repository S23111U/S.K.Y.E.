"""Read-only access to iMessage / SMS (the Messages app's local database).

Read only, on purpose: there is no send here. Nothing is copied anywhere; the
database is opened read-only and only the few most recent rows are returned.
macOS protects that file, so the app running SKYE (Terminal, VS Code...) needs
Full Disk Access once: System Settings > Privacy & Security > Full Disk Access.
Until then the tools say exactly that instead of failing.

Message text never goes into SKYE's long-term memory (scripts/ingest_history.py
skips these tools' turns).
"""

import os
import re
import sqlite3
import subprocess
import time

DB_PATH = os.path.expanduser(os.getenv("MESSAGES_DB", "~/Library/Messages/chat.db"))
COCOA_EPOCH = 978307200          # 2001-01-01 in unix seconds
SNIPPET_CHARS = 160
NEED_ACCESS = ("I need permission to read Messages. In System Settings, Privacy and Security, Full Disk Access, "
               "turn it on for the app that runs SKYE, then restart SKYE.")


class MessagesError(Exception):
    pass


def _connect():
    try:
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    except sqlite3.OperationalError as e:
        raise MessagesError(str(e))


def _body(text, blob):
    """Newer macOS leaves `text` empty and keeps the message in a typed-stream blob."""
    if text:
        return text
    if not blob:
        return ""
    raw = bytes(blob)
    i = raw.find(b"NSString")
    if i < 0:
        return ""
    raw = raw[i + len(b"NSString"):]
    j = raw.find(b"\x01+")
    if j < 0:
        return ""
    raw = raw[j + 2:]
    n = raw[0]
    start = 1
    if n == 0x81:                 # two-byte length
        n = int.from_bytes(raw[1:3], "little")
        start = 3
    try:
        return raw[start:start + n].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _ago(unix):
    d = max(0, time.time() - unix)
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} minutes ago"
    if d < 86400:
        h = int(d // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    days = int(d // 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


_names = {}


def _who(handle):
    """A phone number or email -> the contact's name if Contacts knows it."""
    if not handle:
        return "someone"
    if handle in _names:
        return _names[handle]
    name = handle
    digits = re.sub(r"\D", "", handle)
    try:
        key = digits[-9:] if len(digits) >= 9 else handle
        script = f'''tell application "Contacts"
  repeat with p in people
    repeat with ph in (value of phones of p)
      if (do shell script "echo " & quoted form of (ph as text) & " | tr -cd '0-9'") ends with "{key}" then return name of p
    end repeat
  end repeat
  return ""
end tell''' if digits else f'tell application "Contacts" to return name of first person whose value of emails contains "{handle}"'
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip()
    except Exception:
        pass
    _names[handle] = name
    return name


def _handles_for(contact):
    """Handles (numbers/emails) whose contact name matches, plus a raw number/email if given."""
    c = contact.strip()
    if re.fullmatch(r"[+\d\s()-]{6,}", c) or "@" in c:
        return [c]
    try:
        r = subprocess.run(["osascript", "-e", f'''tell application "Contacts"
  set out to {{}}
  repeat with p in (every person whose name contains "{c.replace('"', '')}")
    repeat with ph in (value of phones of p)
      set end of out to (ph as text)
    end repeat
    repeat with em in (value of emails of p)
      set end of out to (em as text)
    end repeat
  end repeat
  set AppleScript's text item delimiters to "|"
  return out as text
end tell'''], capture_output=True, text=True, timeout=15)
        return [h for h in r.stdout.strip().split("|") if h]
    except Exception:
        return []


def _rows(sql, params=()):
    try:
        con = _connect()
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()
    except (MessagesError, sqlite3.OperationalError, PermissionError) as e:
        raise MessagesError(str(e))


def _snip(text):
    t = re.sub(r"\s+", " ", text or "").strip()
    return (t[:SNIPPET_CHARS] + "...") if len(t) > SNIPPET_CHARS else (t or "an attachment")


def read_messages(contact="", limit=4):
    limit = max(1, min(int(limit or 4), 6))
    base = ("SELECT m.text, m.attributedBody, m.is_from_me, m.date, h.id FROM message m "
            "LEFT JOIN handle h ON m.handle_id = h.rowid WHERE m.associated_message_type = 0 ")
    if contact.strip():
        handles = _handles_for(contact)
        if not handles:
            return f"I could not find a contact called {contact}."
        marks = ",".join("?" * len(handles))
        rows = _rows(base + f"AND h.id IN ({marks}) ORDER BY m.date DESC LIMIT ?", (*handles, limit))
        if not rows:
            return f"I found no messages with {contact}."
        parts = []
        for text, blob, mine, date, hid in reversed(rows):
            parts.append(f"{'You said' if mine else 'They said'}: {_snip(_body(text, blob))}")
        return f"Latest with {contact}, {_ago(rows[0][3] / 1e9 + COCOA_EPOCH)}. " + " ".join(p + "." for p in parts)
    rows = _rows(base + "AND m.is_from_me = 0 ORDER BY m.date DESC LIMIT ?", (limit,))
    if not rows:
        return "You have no recent messages."
    parts = [f"{_who(hid)}, {_ago(date / 1e9 + COCOA_EPOCH)}: {_snip(_body(text, blob))}" for text, blob, _, date, hid in rows]
    return "Your latest messages. " + ". ".join(parts) + "."


def unread_messages():
    rows = _rows("SELECT h.id, COUNT(*) FROM message m LEFT JOIN handle h ON m.handle_id = h.rowid "
                 "WHERE m.is_read = 0 AND m.is_from_me = 0 AND m.associated_message_type = 0 GROUP BY h.id ORDER BY COUNT(*) DESC LIMIT 6")
    if not rows:
        return "You have no unread messages."
    total = sum(c for _, c in rows)
    who = [f"{_who(h)} ({c})" if c > 1 else _who(h) for h, c in rows[:4]]
    return f"You have {total} unread message{'s' if total != 1 else ''}, from " + ", ".join(who) + "."
