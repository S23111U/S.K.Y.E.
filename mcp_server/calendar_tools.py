"""Google Calendar tools (primary calendar) for SKYE.

Same shape as notion_tools.py: every function returns a sentence ready to be
spoken, failures become sentences, and changes are reversible through a small
undo log (`undo_calendar`). Login is a one-time browser consent done by
scripts/google_login.py; the token is kept in memory/google_token.json.
"""

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import dateparser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE = os.path.join(ROOT, "gcp-oauth.keys.json")
TOKEN_FILE = os.path.join(ROOT, "memory", "google_token.json")
UNDO_FILE = os.path.join(ROOT, "memory", "calendar_undo.json")
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
UNDO_MAX_AGE_S = 3600
DEFAULT_MINUTES = 60
LIST_CAP = 6

_service = None


class CalendarError(Exception):
    pass


def connected() -> bool:
    return os.path.exists(TOKEN_FILE)


def _svc():
    global _service
    if _service:
        return _service
    if not connected():
        raise CalendarError("not connected")
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds.valid:
        if creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                raise CalendarError(f"login expired ({e})")
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        else:
            raise CalendarError("login expired")
    _service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _service


def _call(req):
    try:
        return req.execute()
    except Exception as e:  # googleapiclient.errors.HttpError and network errors
        raise CalendarError(str(e))


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def _now():
    return datetime.now().astimezone()


def _parse_when(text, day_only=False):
    """Natural language -> aware datetime in the future-leaning local zone."""
    dt = dateparser.parse(
        text or "",
        settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(minutes=1) if day_only else datetime.now())},
    )
    if dt is None:
        return None
    return dt.astimezone()  # naive -> local zone


def _day_range(text):
    """(start, end, spoken label) for 'today', 'tomorrow', 'this week', a weekday or a date."""
    t = (text or "").strip().lower()
    now = _now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if t in ("", "today"):
        return midnight, midnight + timedelta(days=1), "today"
    if t == "tomorrow":
        return midnight + timedelta(days=1), midnight + timedelta(days=2), "tomorrow"
    if t in ("this week", "week"):
        s = midnight - timedelta(days=midnight.weekday())
        return max(s, midnight), s + timedelta(days=7), "this week"
    if t == "next week":
        s = midnight - timedelta(days=midnight.weekday()) + timedelta(days=7)
        return s, s + timedelta(days=7), "next week"
    if t in ("next 7 days", "coming week", "upcoming"):
        return now, now + timedelta(days=7), "over the next week"
    d = _parse_when(t, day_only=True)
    if d is None:
        return None
    s = d.replace(hour=0, minute=0, second=0, microsecond=0)
    label = s.strftime("%A") if s - midnight < timedelta(days=7) else s.strftime("%B %-d")
    return s, s + timedelta(days=1), label


def _clock(dt):
    t = dt.strftime("%-I:%M %p")
    return t.replace(":00 ", " ")


def _day_word(dt):
    today = _now().date()
    d = dt.date()
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    if 0 < (d - today).days < 7:
        return dt.strftime("%A")
    return dt.strftime("%B %-d")


def _start_of(ev):
    s = ev["start"]
    if "dateTime" in s:
        return datetime.fromisoformat(s["dateTime"]).astimezone(), False
    return datetime.fromisoformat(s["date"]).astimezone(), True


def _end_of(ev):
    e = ev["end"]
    if "dateTime" in e:
        return datetime.fromisoformat(e["dateTime"]).astimezone()
    return datetime.fromisoformat(e["date"]).astimezone()


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _spoken_event(ev, with_day=False):
    start, all_day = _start_of(ev)
    title = ev.get("summary", "an untitled event")
    if all_day:
        return f"{title} all day" + (f" {_day_word(start)}" if with_day else "")
    when = _clock(start)
    return f"{title} at {when}" + (f" {_day_word(start)}" if with_day else "")


# --------------------------------------------------------------------------
# Fetching and matching
# --------------------------------------------------------------------------
def _events(start, end, query=None):
    params = dict(calendarId="primary", timeMin=start.isoformat(), timeMax=end.isoformat(),
                  singleEvents=True, orderBy="startTime", maxResults=50)
    if query:
        params["q"] = query
    items = _call(_svc().events().list(**params)).get("items", [])
    return [e for e in items if e.get("status") != "cancelled"]


def _find(title, when=""):
    """The event best matching a title: upcoming first, within 60 days."""
    words = [w for w in re.findall(r"[a-z0-9']+", (title or "").lower()) if w not in
             {"the", "a", "an", "my", "meeting", "event", "appointment"}] or re.findall(r"[a-z0-9']+", (title or "").lower())
    if when:
        s, e, _ = _day_range(when) or (_now() - timedelta(hours=1), _now() + timedelta(days=60), "")
    else:
        s, e = _now() - timedelta(hours=1), _now() + timedelta(days=60)
    cands = []
    for ev in _events(s, e):
        name = ev.get("summary", "").lower()
        score = sum(1 for w in words if w in name)
        if words and score == len(words):
            cands.append((score, ev))
    if not cands:
        for ev in _events(s, e):
            name = ev.get("summary", "").lower()
            if words and any(w in name for w in words if len(w) > 3):
                cands.append((1, ev))
    return cands[0][1] if cands else None


# --------------------------------------------------------------------------
# Undo log
# --------------------------------------------------------------------------
def _undo_load():
    try:
        return json.load(open(UNDO_FILE))
    except (OSError, ValueError):
        return []


def _undo_push(entry):
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S][-19:]
    stack.append(dict(entry, at=time.time()))
    os.makedirs(os.path.dirname(UNDO_FILE), exist_ok=True)
    json.dump(stack, open(UNDO_FILE, "w"))


def undo_calendar():
    """Reverses the last calendar change SKYE made: removes an added event,
    puts a moved one back, or restores a cancelled one."""
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S]
    if not stack:
        return "There is nothing recent to undo on your calendar."
    entry = stack.pop()
    svc = _svc()
    try:
        if entry["kind"] == "create":
            _call(svc.events().delete(calendarId="primary", eventId=entry["id"]))
            msg = f"Removed {entry['label']} from your calendar."
        elif entry["kind"] == "move":
            _call(svc.events().patch(calendarId="primary", eventId=entry["id"],
                                     body={"start": entry["start"], "end": entry["end"]}))
            msg = f"Put {entry['label']} back where it was."
        else:  # delete
            _call(svc.events().insert(calendarId="primary", body=entry["body"]))
            msg = f"Restored {entry['label']}."
    except CalendarError as e:
        return f"I could not undo that: {e}"
    json.dump(stack, open(UNDO_FILE, "w"))
    return msg


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def _clean_range_word(text):
    t = (text or "").strip().lower()
    t = re.sub(r"^(?:from|between|on|for|till|until|to)\s+", "", t)
    return t


def list_events(period="", until=""):
    """Events for a day, a week, or a span: `period` is the first day and
    `until` (optional) the last day, inclusive."""
    p = _clean_range_word(period)
    # a span spoken in one string: "today to september 30", "from monday until friday"
    m = re.split(r"\s+(?:to|till|until|through|and)\s+", p, maxsplit=1)
    if len(m) == 2 and not until and re.search(r"\d|mon|tue|wed|thu|fri|sat|sun|tomorrow|today|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", m[1]):
        p, until = m
    rng = _day_range(p)
    if rng is None:
        return "I could not work out which date you meant. Could you say it again?"
    start, end, label = rng
    if until:
        u = _day_range(_clean_range_word(until))
        if u is None:
            return f"I could not work out the date {until}. That day may not exist."
        end = max(end, u[1])
        label = f"from {_day_word(start)} to {_day_word(end - timedelta(days=1))}"
    evs = _events(start, end)
    if not evs:
        return f"You have nothing on your calendar {label}." if label.startswith(("today", "tomorrow", "this", "next", "over", "from")) \
            else f"You have nothing on your calendar on {label}."
    multi_day = (end - start) > timedelta(days=1)
    shown = evs[:LIST_CAP]
    spoken = _join(_spoken_event(e, with_day=multi_day) for e in shown)
    n = len(evs)
    head = f"You have {n} event{'s' if n != 1 else ''} {label}" if label.startswith(("today", "tomorrow", "this", "next", "over", "from")) \
        else f"You have {n} event{'s' if n != 1 else ''} on {label}"
    tail = f" And {n - LIST_CAP} more." if n > LIST_CAP else "."
    return f"{head}: {spoken}{tail}"


def next_event():
    now = _now()
    for ev in _events(now, now + timedelta(days=30))[:5]:
        start, all_day = _start_of(ev)
        if all_day and start.date() != now.date():
            continue
        if not all_day and _end_of(ev) <= now:
            continue
        mins = int((start - now).total_seconds() // 60)
        if not all_day and 0 < mins < 60:
            return f"Your next event is {ev.get('summary', 'untitled')} in {mins} minutes, at {_clock(start)}."
        return f"Your next event is {_spoken_event(ev, with_day=True)}."
    return "You have nothing coming up in the next month."


def add_event(title, when, duration_minutes=""):
    title = (title or "").strip()
    if not title:
        return "What should I call the event?"
    start = _parse_when(when)
    if start is None:
        return "I did not catch when that should be. Could you say the day and time?"
    if start < _now() - timedelta(minutes=5):
        return f"That time has already passed. Did you mean {_day_word(start + timedelta(days=1))}?"
    try:
        mins = int(float(duration_minutes)) if str(duration_minutes).strip() else DEFAULT_MINUTES
    except ValueError:
        mins = DEFAULT_MINUTES
    end = start + timedelta(minutes=mins)
    clash = [e for e in _events(start, end) if not _start_of(e)[1]]
    body = {"summary": title, "start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}
    ev = _call(_svc().events().insert(calendarId="primary", body=body))
    _undo_push({"kind": "create", "id": ev["id"], "label": title})
    msg = f"Added {title} {_day_word(start)} at {_clock(start)}."
    if clash:
        msg += f" Note it overlaps with {clash[0].get('summary', 'another event')}."
    return msg


def move_event(title, when):
    ev = _find(title)
    if not ev:
        return f"I could not find an event called {title}."
    start = _parse_when(when)
    if start is None:
        return "I did not catch the new time. Could you say the day and time?"
    old_start, all_day = _start_of(ev)
    if all_day:
        return f"{ev.get('summary')} is an all-day event, so I will leave it. You can move it yourself."
    length = _end_of(ev) - old_start
    _undo_push({"kind": "move", "id": ev["id"], "label": ev.get("summary", "the event"),
                "start": ev["start"], "end": ev["end"]})
    _call(_svc().events().patch(calendarId="primary", eventId=ev["id"], body={
        "start": {"dateTime": start.isoformat()}, "end": {"dateTime": (start + length).isoformat()}}))
    return f"Moved {ev.get('summary')} to {_day_word(start)} at {_clock(start)}."


def cancel_event(title, when=""):
    ev = _find(title, when)
    if not ev:
        return f"I could not find an event called {title}."
    keep = {k: ev[k] for k in ("summary", "start", "end", "description", "location") if k in ev}
    _undo_push({"kind": "delete", "label": ev.get("summary", "the event"), "body": keep})
    _call(_svc().events().delete(calendarId="primary", eventId=ev["id"]))
    return f"Cancelled {ev.get('summary')} {_day_word(_start_of(ev)[0])}."


def find_free_time(day="", minutes=""):
    """Free gaps between 9 AM and 6 PM on a day that fit the requested length."""
    start, end, label = _day_range(day or "today") or _day_range("today")
    try:
        need = int(float(minutes)) if str(minutes).strip() else 60
    except ValueError:
        need = 60
    now = _now()
    win_s, win_e = start.replace(hour=9), start.replace(hour=18)
    win_s = max(win_s, now) if start.date() == now.date() else win_s
    busy = [(max(_start_of(e)[0], win_s), min(_end_of(e), win_e)) for e in _events(start, end) if not _start_of(e)[1]]
    busy.sort()
    gaps, cur = [], win_s
    for b_s, b_e in busy:
        if b_s > cur and (b_s - cur) >= timedelta(minutes=need):
            gaps.append((cur, b_s))
        cur = max(cur, b_e)
    if win_e > cur and (win_e - cur) >= timedelta(minutes=need):
        gaps.append((cur, win_e))
    if not gaps:
        return f"You have no free {need} minute slot between 9 and 6 {label}."
    shown = _join(f"{_clock(a)} to {_clock(b)}" for a, b in gaps[:3])
    return f"You are free {label} from {shown}."
