"""Shared Notion plumbing for the finance/todo/knowledge skills: the raw REST
client, workspace discovery (databases found by title and shape, no IDs
committed to the repo), formatting helpers, fuzzy name matching, period/date
parsing, and the undo log. Not a skill itself (no SKILL descriptor) — just
infrastructure three skill packages build on.

Purpose-named tools instead of a raw Notion API surface. A small local model
can't be trusted to pick page IDs or guess which of ~20 generic tools to use,
so each skill's tools do one job apiece and this module knows where the data
lives: links between databases (expense -> category/account) are resolved in
code, and every number the user hears is computed here from the raw rows —
deliberately not from Notion's own rollups/formulas, which disagreed with the
underlying rows in testing (Budget Summary said $0 spent while the Expenses
database held real numbers for the month).

Writes are narrow and reversible: each is recorded in
memory/notion_undo.json, and undo_last_entry() archives what SKYE created or
restores what it changed. Learning notes are read-only; nothing here can
delete data.
"""

import difflib
import json
import os
import re
import threading
import time
from datetime import date, datetime, timedelta

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.notion.com/v1"
UNDO_FILE = os.path.join(ROOT, "memory", "notion_undo.json")
UNDO_MAX_AGE_S = 24 * 3600
DUPLICATE_WINDOW_S = 120

_lock = threading.Lock()
_last_call = [0.0]
_cache: dict = {}


class NotionError(Exception):
    pass


# --------------------------------------------------------------------------
# API layer
# --------------------------------------------------------------------------
def api(method, path, body=None):
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise NotionError("NOTION_TOKEN is not set")
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
    for _ in range(4):
        with _lock:  # Notion allows ~3 requests/second
            wait = 0.34 - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        r = requests.request(method, API + path, headers=headers, json=body, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code >= 400:
            raise NotionError(f"{r.status_code}: {r.text[:200]}")
        return r.json()
    raise NotionError("rate limited")


def paginate(method, path, body=None, limit=None):
    out, cursor = [], None
    while True:
        b = dict(body or {}, page_size=100)
        p = path
        if cursor:
            if method == "POST":
                b["start_cursor"] = cursor
            else:
                p += ("&" if "?" in p else "?") + f"start_cursor={cursor}"
        r = api(method, p, b if method == "POST" else None)
        out += r.get("results", [])
        if not r.get("has_more") or (limit and len(out) >= limit):
            return out[:limit] if limit else out
        cursor = r["next_cursor"]


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def title(obj):
    if obj["object"] == "database":
        return "".join(t["plain_text"] for t in obj.get("title", []))
    for v in obj.get("properties", {}).values():
        if v["type"] == "title":
            return "".join(t["plain_text"] for t in v["title"])
    return ""


def text(prop):
    return "".join(t["plain_text"] for t in prop.get("title", prop.get("rich_text", [])))


# --------------------------------------------------------------------------
# Workspace discovery (by title + shape, no IDs stored)
# --------------------------------------------------------------------------
def databases():
    def load():
        dbs = paginate("POST", "/search", {"filter": {"property": "object", "value": "database"}})
        found = {"notes": []}
        for db in dbs:
            t, props = title(db), db["properties"]
            if t == "Expenses" and {"Amount", "Category", "Date"} <= props.keys():
                found["expenses"] = db
            elif t == "Categories" and "Monthly Budget" in props:
                found["categories"] = db
            elif t == "Accounts" and "Initial Balance" in props:
                found["accounts"] = db
            elif t == "Income" and {"Amount", "Account"} <= props.keys():
                found["income"] = db
            elif "Status" in props and "Priority" in props and props["Status"]["type"] == "status":
                found["todo"] = db
            elif t == "Notes" and "Notebook" in props:
                found["notes"].append(db)
            elif t == "Notebooks" and "Notes" in props:
                found.setdefault("notebooks", []).append(db)
        return found

    return cached("dbs", 3600, load)


def need(name):
    db = databases().get(name)
    if not db:
        raise NotionError(f"the {name} database is not shared with the SKYE integration")
    return db


# --------------------------------------------------------------------------
# Formatting for speech
# --------------------------------------------------------------------------
def money(x):
    x = round(float(x), 2)
    dollars, cents = int(x), int(round((x - int(x)) * 100))
    d = f"{dollars:,} dollar{'s' if dollars != 1 else ''}"
    return d if cents == 0 else f"{d} and {cents} cent{'s' if cents != 1 else ''}"


def day(iso):
    d = datetime.strptime(iso[:10], "%Y-%m-%d")
    return d.strftime("%B ") + str(d.day)


def join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Matching helpers
# --------------------------------------------------------------------------
def match_name(query, names, aliases=None, cutoff=0.6):
    """Best name for `query` among `names`, or None."""
    q = (query or "").strip().lower()
    if not q:
        return None
    lower = {n.lower(): n for n in names}
    if q in lower:
        return lower[q]
    if aliases and q in aliases and aliases[q].lower() in lower:
        return lower[aliases[q].lower()]
    for n_low, n in lower.items():
        if q in n_low or n_low in q:
            return n
    if aliases:
        for word in re.findall(r"[a-z]+", q):
            if word in aliases and aliases[word].lower() in lower:
                return lower[aliases[word].lower()]
    close = difflib.get_close_matches(q, list(lower), n=1, cutoff=cutoff)
    return lower[close[0]] if close else None


def fuzzy_pick(query, items, key):
    """(best, runner_up_is_close). Scores token overlap plus string similarity."""
    q = (query or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for it in items:
        name = key(it).lower()
        n_tokens = set(re.findall(r"[a-z0-9]+", name))
        overlap = len(q_tokens & n_tokens) / max(len(q_tokens), 1)
        ratio = difflib.SequenceMatcher(None, q, name).ratio()
        contains = 1.0 if (q and (q in name or name in q)) else 0.0
        scored.append((0.5 * overlap + 0.3 * ratio + 0.2 * contains, it))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < 0.35:
        return None, False
    close = len(scored) > 1 and scored[1][0] > scored[0][0] - 0.06
    return scored[0][1], close


def parse_amount(value):
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d[\d,]*\.?\d*", str(value or ""))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


# --------------------------------------------------------------------------
# Dates / periods
# --------------------------------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}


def _month_start(d):
    return d.replace(day=1)


def _next_month(d):
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def period(text):
    """(start, end_exclusive, spoken_label); start/end None means all time."""
    t = (text or "").strip().lower()
    today = date.today()
    if t in ("", "this month", "month", "current month", "so far"):
        s = _month_start(today)
        return s, _next_month(s), s.strftime("%B")
    if t == "last month":
        e = _month_start(today)
        s = _month_start(e - timedelta(days=1))
        return s, e, s.strftime("%B")
    if t == "today":
        return today, today + timedelta(days=1), "today"
    if t == "yesterday":
        return today - timedelta(days=1), today, "yesterday"
    if t in ("this week", "week"):
        s = today - timedelta(days=today.weekday())
        return s, s + timedelta(days=7), "this week"
    if t == "last week":
        s = today - timedelta(days=today.weekday() + 7)
        return s, s + timedelta(days=7), "last week"
    if t in ("this year", "year"):
        return date(today.year, 1, 1), date(today.year + 1, 1, 1), str(today.year)
    if t in ("all", "all time", "ever", "total", "overall"):
        return None, None, "all time"
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if m:
        s = date(int(m.group(1)), int(m.group(2)), 1)
        return s, _next_month(s), s.strftime("%B %Y")
    for name, num in MONTHS.items():
        if name in t or name[:3] == t[:3] and t[:3] in ("jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"):
            year = re.search(r"\b(20\d{2})\b", t)
            y = int(year.group(1)) if year else (today.year if num <= today.month else today.year - 1)
            s = date(y, num, 1)
            return s, _next_month(s), s.strftime("%B") + (f" {y}" if y != today.year else "")
    s = _month_start(today)
    return s, _next_month(s), s.strftime("%B")


def parse_date(text):
    t = (text or "").strip().lower()
    today = date.today()
    if t in ("", "today", "now"):
        return today
    if t == "yesterday":
        return today - timedelta(days=1)
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if m:
        return date.fromisoformat(m.group(0))
    for name, num in MONTHS.items():
        m = re.search(rf"\b{name[:3]}\w*\s+(\d{{1,2}})\b", t)
        if m:
            d = date(today.year, num, int(m.group(1)))
            return d if d <= today + timedelta(days=1) else d.replace(year=today.year - 1)
    return today


# --------------------------------------------------------------------------
# Undo log
# --------------------------------------------------------------------------
def _undo_load():
    try:
        return json.load(open(UNDO_FILE))
    except (OSError, ValueError):
        return []


def undo_push(entry):
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S][-19:]
    stack.append(dict(entry, at=time.time()))
    os.makedirs(os.path.dirname(UNDO_FILE), exist_ok=True)
    json.dump(stack, open(UNDO_FILE, "w"))


def recent_undo_entries():
    """create_expense/add_todo etc. use this to check for a just-added
    duplicate before writing a new one."""
    return [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S]


def undo_last_entry() -> str:
    """Undoes the last thing SKYE added to or changed in Notion (an expense, a to-do)."""
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S]
    if not stack:
        return "There is nothing recent to undo."
    entry = stack.pop()
    try:
        if entry["kind"] == "create":
            api("PATCH", f"/pages/{entry['page_id']}", {"archived": True})
            msg = f"Removed {entry['label']}."
        else:
            api("PATCH", f"/pages/{entry['page_id']}", {"properties": entry["previous"]})
            msg = f"Restored {entry['label']} to how it was."
    except NotionError as e:
        return f"I could not undo that: {e}"
    os.makedirs(os.path.dirname(UNDO_FILE), exist_ok=True)
    json.dump(stack, open(UNDO_FILE, "w"))
    return msg
