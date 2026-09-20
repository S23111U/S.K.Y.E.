"""SKYE's Notion tools: finance tracker, to-do list, learning notes.

Purpose-named tools instead of a raw Notion API surface. A small local model
can't be trusted to pick page IDs or guess which of ~20 generic tools to use,
so each tool here does one job and this module knows where the data lives:
databases are found by title and shape at runtime (no IDs in the repo), links
between them (expense -> category/account) are resolved in code, and every
number the user hears is computed here from the raw rows.

Deliberately not used: Notion's own rollups/formulas for totals. They disagreed
with the underlying rows (Budget Summary said $0 spent while the Expenses
database held $1,429.50 for the month), so the source of truth is the rows.

Writes are narrow (add an expense or to-do, change a to-do's status/priority)
and reversible: each is recorded in memory/notion_undo.json, and
undo_last_entry() archives what SKYE created or restores what it changed.
Learning notes are read-only. Nothing here can delete data.

Every function returns a complete sentence (or two) ready to be spoken.
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
# Prefix core_v2.py keys on to know a tool result is raw material to summarise
# from, not a finished reply (shared with mcp_server/server.py).
SOURCES_MARKER = "[WEB SOURCES]"
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
def _api(method, path, body=None):
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


def _paginate(method, path, body=None, limit=None):
    out, cursor = [], None
    while True:
        b = dict(body or {}, page_size=100)
        p = path
        if cursor:
            if method == "POST":
                b["start_cursor"] = cursor
            else:
                p += ("&" if "?" in p else "?") + f"start_cursor={cursor}"
        r = _api(method, p, b if method == "POST" else None)
        out += r.get("results", [])
        if not r.get("has_more") or (limit and len(out) >= limit):
            return out[:limit] if limit else out
        cursor = r["next_cursor"]


def _cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def _title(obj):
    if obj["object"] == "database":
        return "".join(t["plain_text"] for t in obj.get("title", []))
    for v in obj.get("properties", {}).values():
        if v["type"] == "title":
            return "".join(t["plain_text"] for t in v["title"])
    return ""


def _text(prop):
    return "".join(t["plain_text"] for t in prop.get("title", prop.get("rich_text", [])))


# --------------------------------------------------------------------------
# Workspace discovery (by title + shape, no IDs stored)
# --------------------------------------------------------------------------
def _databases():
    def load():
        dbs = _paginate("POST", "/search", {"filter": {"property": "object", "value": "database"}})
        found = {"notes": []}
        for db in dbs:
            t, props = _title(db), db["properties"]
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

    return _cached("dbs", 3600, load)


def _need(name):
    db = _databases().get(name)
    if not db:
        raise NotionError(f"the {name} database is not shared with the SKYE integration")
    return db


# --------------------------------------------------------------------------
# Formatting for speech
# --------------------------------------------------------------------------
def _money(x):
    x = round(float(x), 2)
    dollars, cents = int(x), int(round((x - int(x)) * 100))
    d = f"{dollars:,} dollar{'s' if dollars != 1 else ''}"
    return d if cents == 0 else f"{d} and {cents} cent{'s' if cents != 1 else ''}"


def _day(iso):
    d = datetime.strptime(iso[:10], "%Y-%m-%d")
    return d.strftime("%B ") + str(d.day)


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Matching helpers
# --------------------------------------------------------------------------
CATEGORY_ALIASES = {
    "groceries": "Food", "grocery": "Food", "lunch": "Food", "dinner": "Food", "breakfast": "Food",
    "coffee": "Food", "takeaway": "Food", "restaurant": "Food", "snack": "Food", "meal": "Food",
    "uber": "Transportation", "taxi": "Transportation", "train": "Transportation", "bus": "Transportation",
    "opal": "Transportation", "transport": "Transportation", "fuel": "Transportation", "petrol": "Transportation",
    "phone": "Utilities", "internet": "Utilities", "electricity": "Utilities", "water": "Utilities",
    "bills": "Utilities", "bill": "Utilities", "wifi": "Utilities", "laundry": "Utilities",
    "book": "Knowledge", "books": "Knowledge", "course": "Knowledge", "tuition": "Knowledge", "fees": "Knowledge",
    "netflix": "Entertainment", "movie": "Entertainment", "movies": "Entertainment", "game": "Entertainment",
    "games": "Entertainment", "spotify": "Entertainment", "concert": "Entertainment",
    "clothes": "Shopping", "clothing": "Shopping", "shoes": "Shopping",
    "stock": "Investing", "stocks": "Investing", "etf": "Investing", "shares": "Investing",
}


def _match_name(query, names, aliases=None, cutoff=0.6):
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


def _fuzzy_pick(query, items, key):
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


def _parse_amount(value):
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


def _period(text):
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


def _parse_date(text):
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


def _undo_push(entry):
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S][-19:]
    stack.append(dict(entry, at=time.time()))
    os.makedirs(os.path.dirname(UNDO_FILE), exist_ok=True)
    json.dump(stack, open(UNDO_FILE, "w"))


def undo_last_entry():
    """Removes the last thing SKYE added to Notion, or restores what it last changed."""
    stack = [e for e in _undo_load() if time.time() - e["at"] < UNDO_MAX_AGE_S]
    if not stack:
        return "There is nothing recent to undo."
    entry = stack.pop()
    try:
        if entry["kind"] == "create":
            _api("PATCH", f"/pages/{entry['page_id']}", {"archived": True})
            msg = f"Removed {entry['label']}."
        else:
            _api("PATCH", f"/pages/{entry['page_id']}", {"properties": entry["previous"]})
            msg = f"Restored {entry['label']} to how it was."
    except NotionError as e:
        return f"I could not undo that: {e}"
    os.makedirs(os.path.dirname(UNDO_FILE), exist_ok=True)
    json.dump(stack, open(UNDO_FILE, "w"))
    return msg


# --------------------------------------------------------------------------
# Finance
# --------------------------------------------------------------------------
def _categories():
    def load():
        rows = _paginate("POST", f"/databases/{_need('categories')['id']}/query")
        return [{"id": r["id"], "name": _title(r), "budget": r["properties"]["Monthly Budget"]["number"] or 0} for r in rows]
    return _cached("categories", 300, load)


def _accounts():
    return _cached("accounts", 300, lambda: [
        {"id": r["id"], "name": _title(r), "initial": r["properties"]["Initial Balance"]["number"] or 0}
        for r in _paginate("POST", f"/databases/{_need('accounts')['id']}/query")
    ])


def _expenses(start=None, end=None):
    conds = []
    if start:
        conds.append({"property": "Date", "date": {"on_or_after": start.isoformat()}})
    if end:
        conds.append({"property": "Date", "date": {"before": end.isoformat()}})
    body = {"filter": {"and": conds}} if len(conds) > 1 else ({"filter": conds[0]} if conds else {})
    cat_names = {c["id"]: c["name"] for c in _categories()}
    out = []
    for r in _paginate("POST", f"/databases/{_need('expenses')['id']}/query", body):
        p = r["properties"]
        rel = p["Category"]["relation"]
        out.append({
            "id": r["id"], "name": _text(p["Expenses"]), "amount": p["Amount"]["number"] or 0,
            "date": (p["Date"]["date"] or {}).get("start", "")[:10],
            "category": cat_names.get(rel[0]["id"], "Uncategorised") if rel else "Uncategorised",
        })
    return out


def _scope(label):
    return {"all time": "Over all time", "today": "Today", "yesterday": "Yesterday",
            "this week": "This week", "last week": "Last week"}.get(label, f"In {label}")


def expense_summary(period="", category=""):
    """Total spending for a period, by category, computed from the raw expense rows."""
    start, end, label = _period(period)
    rows = _expenses(start, end)
    cat = None
    if category:
        names = [c["name"] for c in _categories()]
        cat = _match_name(category, names, CATEGORY_ALIASES)
        if not cat:
            return f"I could not find a category called {category}. Your categories are {_join(names)}."
        rows = [r for r in rows if r["category"] == cat]
    if not rows:
        return f"No expenses are recorded {'on ' + cat + ' ' if cat else ''}for {label}."
    total = sum(r["amount"] for r in rows)
    lead = (f"{_scope(label)}{' on ' + cat if cat else ''}, you have spent {_money(total)} "
            f"across {len(rows)} expense{'s' if len(rows) != 1 else ''}.")
    if cat:
        return lead
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]
    ranked = sorted(by_cat.items(), key=lambda kv: -kv[1])
    return f"{lead} By category: {_join(f'{n} {_money(v)}' for n, v in ranked[:5])}."


def list_expenses(period="", category="", limit="5"):
    """Most recent individual expenses."""
    start, end, label = _period(period) if period else (None, None, "recently")
    rows = _expenses(start, end)
    if category:
        cat = _match_name(category, [c["name"] for c in _categories()], CATEGORY_ALIASES)
        if cat:
            rows = [r for r in rows if r["category"] == cat]
    rows.sort(key=lambda r: r["date"], reverse=True)
    n = max(1, min(int(_parse_amount(limit) or 5), 10))
    rows = rows[:n]
    if not rows:
        return "There are no matching expenses."
    parts = [f"{r['name']}, {_money(r['amount'])}, {_day(r['date'])}, {r['category']}" for r in rows]
    lead = "Your latest expense" if len(rows) == 1 else f"Your latest {len(rows)} expenses"
    return f"{lead}: " + "; ".join(parts) + "."


def budget_status():
    """This month's spending against each category's monthly budget."""
    start = _month_start(date.today())
    rows = _expenses(start, _next_month(start))
    spent = {}
    for r in rows:
        spent[r["category"]] = spent.get(r["category"], 0) + r["amount"]
    budgeted = [c for c in _categories() if c["budget"] > 0]
    total_spent = sum(spent.values())
    total_budget = sum(c["budget"] for c in budgeted)
    if not budgeted:
        return f"No monthly budgets are set. You have spent {_money(total_spent)} so far this month."
    parts = []
    for c in budgeted:
        s = spent.get(c["name"], 0)
        state = "over budget" if s > c["budget"] else "within budget"
        parts.append(f"{c['name']} {_money(s)} of {_money(c['budget'])}, {state}")
    unbudgeted = sum(v for k, v in spent.items() if k not in {c["name"] for c in budgeted})
    tail = f" Spending in categories without a budget comes to {_money(unbudgeted)}." if unbudgeted else ""
    return (f"This month you have spent {_money(total_spent)} in total, against {_money(total_budget)} budgeted. "
            + _join(parts) + "." + tail)


def account_balance():
    """Current balance: initial balance + income - expenses (transfers are unused)."""
    accounts = _accounts()
    if not accounts:
        return "I could not find an account."
    inc = sum(r["properties"]["Amount"]["number"] or 0
              for r in _paginate("POST", f"/databases/{_need('income')['id']}/query"))
    exp = sum(r["amount"] for r in _expenses())
    lines = []
    for a in accounts:
        bal = a["initial"] + inc - exp if len(accounts) == 1 else None
        lines.append(f"{a['name']} balance is {_money(bal)}" if bal is not None else a["name"])
    return _join(lines) + f", after {_money(exp)} spent and {_money(inc)} received in total."


def add_expense(name="", amount="", category="", date_text=""):
    """Logs an expense in the tracker; returns what was recorded."""
    name = (name or "").strip()
    value = _parse_amount(amount)
    if not name or value is None or value <= 0:
        return "I need what it was for and how much it cost to log an expense."
    cats = _categories()
    cat = _match_name(category, [c["name"] for c in cats], CATEGORY_ALIASES) or _match_name(name, [c["name"] for c in cats], CATEGORY_ALIASES)
    if not cat:
        return f"Which category should that go under? Your categories are {_join([c['name'] for c in cats])}."
    when = _parse_date(date_text)
    accounts = _accounts()
    if not accounts:
        return "I could not find an account to log that against."

    # Guard against the same entry arriving twice (a re-sent or merged utterance).
    for e in _undo_load():
        if (e["kind"] == "create" and e.get("sig") == [name.lower(), round(value, 2), when.isoformat()]
                and time.time() - e["at"] < DUPLICATE_WINDOW_S):
            return f"That looks like a duplicate of {e['label']}, so I have not added it again."

    props = {
        "Expenses": {"title": [{"text": {"content": name}}]},
        "Amount": {"number": round(value, 2)},
        "Date": {"date": {"start": when.isoformat()}},
        "Category": {"relation": [{"id": next(c["id"] for c in cats if c["name"] == cat)}]},
        "Account": {"relation": [{"id": accounts[0]["id"]}]},
        "Notes": {"rich_text": [{"text": {"content": "Logged by SKYE"}}]},
    }
    page = _api("POST", "/pages", {"parent": {"database_id": _need("expenses")["id"]}, "properties": props})
    label = f"{name}, {_money(value)}"
    _undo_push({"kind": "create", "page_id": page["id"], "label": label,
                "sig": [name.lower(), round(value, 2), when.isoformat()]})
    return f"Logged {name}, {_money(value)}, under {cat} on {_day(when.isoformat())}."


# --------------------------------------------------------------------------
# To-do list
# --------------------------------------------------------------------------
STATUS_SYNONYMS = {
    "done": "Done", "complete": "Done", "completed": "Done", "finished": "Done", "finish": "Done", "tick": "Done",
    "in progress": "In Progress", "started": "In Progress", "start": "In Progress", "doing": "In Progress", "working": "In Progress",
    "today": "For today", "for today": "For today", "pending": "Pending", "later": "Pending", "not started": "Pending",
}
STATUS_ORDER = {"For today": 0, "In Progress": 1, "Pending": 2, "Done": 3}
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _todos():
    db = _need("todo")
    out = []
    for r in _paginate("POST", f"/databases/{db['id']}/query"):
        p = r["properties"]
        out.append({
            "id": r["id"], "name": _title(r),
            "status": (p["Status"]["status"] or {}).get("name", "Pending"),
            "priority": (p["Priority"]["select"] or {}).get("name", "Medium"),
        })
    return out


def _norm_status(text):
    t = (text or "").strip().lower()
    if not t:
        return None
    if t in STATUS_SYNONYMS:
        return STATUS_SYNONYMS[t]
    return _match_name(t, ["Pending", "For today", "In Progress", "Done"], STATUS_SYNONYMS, 0.7)


def _norm_priority(text):
    return _match_name((text or "").strip(), ["High", "Medium", "Low"], {"urgent": "High", "important": "High", "normal": "Medium"}, 0.7)


def list_todos(status=""):
    """Open items on the to-do list, most urgent first."""
    items = _todos()
    want = _norm_status(status)
    items = [t for t in items if t["status"] == want] if want else [t for t in items if t["status"] != "Done"]
    if not items:
        return "Your to-do list is clear." if not want else f"Nothing is marked {want}."
    items.sort(key=lambda t: (STATUS_ORDER.get(t["status"], 9), PRIORITY_ORDER.get(t["priority"], 9)))
    n = len(items)
    shown = items[:6]   # a long list read aloud is a monologue; the rest can be asked for
    groups = {}
    for t in shown:
        groups.setdefault(t["status"], []).append(t)
    parts = [f"{st}: " + _join(f"{t['name']} ({t['priority'].lower()} priority)" for t in ts) for st, ts in groups.items()]
    more = f" And {n - len(shown)} more." if n > len(shown) else ""
    return f"You have {n} open to-do{'s' if n != 1 else ''}. " + ". ".join(parts) + "." + more


def add_todo(name="", priority="Medium", status="Pending"):
    """Adds an item to the to-do list."""
    name = (name or "").strip()
    if not name:
        return "What should I add to your to-do list?"
    pri = _norm_priority(priority) or "Medium"
    st = _norm_status(status) or "Pending"
    for t in _todos():
        if t["name"].lower() == name.lower() and t["status"] != "Done":
            return f"{t['name']} is already on your to-do list."
    props = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Priority": {"select": {"name": pri}},
        "Status": {"status": {"name": st}},
    }
    page = _api("POST", "/pages", {"parent": {"database_id": _need("todo")["id"]}, "properties": props})
    _undo_push({"kind": "create", "page_id": page["id"], "label": f"the to-do {name}"})
    return f"Added {name} to your to-do list as {pri.lower()} priority, {st.lower()}."


def update_todo(name="", status="", priority=""):
    """Changes a to-do's status and/or priority (e.g. mark it done)."""
    st, pri = _norm_status(status), _norm_priority(priority)
    if not name or not (st or pri):
        return "Tell me which to-do to change and what to change it to, such as done or high priority."
    item, ambiguous = _fuzzy_pick(name, _todos(), key=lambda t: t["name"])
    if not item:
        return f"I could not find a to-do matching {name}."
    if ambiguous:
        return f"More than one to-do matches {name}. Which one did you mean?"
    props, previous = {}, {}
    if st:
        props["Status"], previous["Status"] = {"status": {"name": st}}, {"status": {"name": item["status"]}}
    if pri:
        props["Priority"], previous["Priority"] = {"select": {"name": pri}}, {"select": {"name": item["priority"]}}
    _api("PATCH", f"/pages/{item['id']}", {"properties": props})
    _undo_push({"kind": "update", "page_id": item["id"], "label": f"the to-do {item['name']}", "previous": previous})
    changes = _join([x for x in (f"status {st}" if st else "", f"{pri.lower()} priority" if pri else "") if x])
    return f"Updated {item['name']}: {changes}."


# --------------------------------------------------------------------------
# Learning notes (read-only)
# --------------------------------------------------------------------------
def _ancestor_course(db):
    """Title of the nearest ancestor page of a Notes database (its course)."""
    parent = db["parent"]
    for _ in range(8):
        t = parent["type"]
        if t == "workspace":
            return "Learning Notes"
        try:
            obj = _api("GET", f"/pages/{parent[t]}") if t == "page_id" else _api("GET", f"/blocks/{parent[t]}")
        except NotionError:
            return "Learning Notes"
        if obj["object"] == "page":
            return _title(obj) or "Learning Notes"
        parent = obj["parent"]
    return "Learning Notes"


def _notes_index():
    def load():
        notebooks = {}
        for nb_db in _databases().get("notebooks", []):
            for r in _paginate("POST", f"/databases/{nb_db['id']}/query"):
                notebooks[r["id"]] = _title(r)
        index = []
        for db in _databases()["notes"]:
            course = _ancestor_course(db)
            for r in _paginate("POST", f"/databases/{db['id']}/query"):
                rel = r["properties"]["Notebook"]["relation"]
                index.append({
                    "id": r["id"], "name": _title(r), "course": course,
                    "notebook": notebooks.get(rel[0]["id"], "") if rel else "",
                    "edited": r["last_edited_time"][:10],
                })
        return index
    return _cached("notes_index", 600, load)


def _search_key(n):
    return f"{n['course']} {n['notebook']} {n['name']}"


def _label(n):
    return ", ".join(x for x in (n["course"], n["notebook"], n["name"]) if x)


def search_notes(query=""):
    """Finds learning notes by course, notebook or title."""
    index = _notes_index()
    if not index:
        return "I could not find any learning notes."
    q = (query or "").lower()
    scored = []
    for n in index:
        hay = set(re.findall(r"[a-z0-9]+", _search_key(n).lower()))
        toks = set(re.findall(r"[a-z0-9]+", q)) - {"notes", "note", "my", "the", "for", "on", "of", "about", "in"}
        hits = len(toks & hay)
        scored.append((hits / max(len(toks), 1) + 0.3 * difflib.SequenceMatcher(None, q, _search_key(n).lower()).ratio(), n))
    scored.sort(key=lambda x: -x[0])
    top = [n for s, n in scored[:5] if s > 0.35]
    if not top:
        return f"I found no notes matching {query}."
    return "Matching notes: " + "; ".join(_label(n) for n in top) + "."


def _block_text(block_id, depth=0, budget=None):
    budget = budget if budget is not None else [5000]
    lines = []
    for b in _paginate("GET", f"/blocks/{block_id}/children"):
        if budget[0] <= 0:
            break
        t = b["type"]
        payload = b.get(t, {})
        text = "".join(x["plain_text"] for x in payload.get("rich_text", []))
        if t == "code":
            text = "code: " + text
        if text.strip():
            prefix = "- " if t in ("bulleted_list_item", "numbered_list_item", "to_do") else ""
            lines.append(prefix + text.strip())
            budget[0] -= len(text)
        if b.get("has_children") and depth < 2 and t != "child_page":
            lines.append(_block_text(b["id"], depth + 1, budget))
    return "\n".join(x for x in lines if x)


def read_note(query=""):
    """Reads a learning note; returned as a source for the model to summarise."""
    index = _notes_index()
    item, ambiguous = _fuzzy_pick(query, index, key=_search_key)
    if not item:
        return f"I could not find a note matching {query}."
    if ambiguous:
        top = sorted(index, key=lambda n: -difflib.SequenceMatcher(None, (query or "").lower(), _search_key(n).lower()).ratio())[:3]
        return "More than one note matches. Did you mean " + _join([_label(n) for n in top]) + "?"
    body = _block_text(item["id"])
    if not body.strip():
        return f"The note {_label(item)} is empty."
    return (f"{SOURCES_MARKER} Today is {date.today():%Y-%m-%d}.\n"
            f"[1] {_label(item)} (your Notion learning notes, last edited {item['edited']}): {body}")
