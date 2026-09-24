"""To-do list tools — reads/writes SKYE's Notion to-do database. See
skills/notion_client.py for the shared plumbing this builds on."""

from skills.notion_client import api, fuzzy_pick, join, match_name, need, paginate, title, undo_push

STATUS_SYNONYMS = {
    "done": "Done", "complete": "Done", "completed": "Done", "finished": "Done", "finish": "Done", "tick": "Done",
    "in progress": "In Progress", "started": "In Progress", "start": "In Progress", "doing": "In Progress", "working": "In Progress",
    "today": "For today", "for today": "For today", "pending": "Pending", "later": "Pending", "not started": "Pending",
}
STATUS_ORDER = {"For today": 0, "In Progress": 1, "Pending": 2, "Done": 3}
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def todos():
    """All to-do rows, raw (used by list_todos and by other skills — the mac
    skill's INDEX 0 study suggestions, and the proactive to-do nudge)."""
    db = need("todo")
    out = []
    for r in paginate("POST", f"/databases/{db['id']}/query"):
        p = r["properties"]
        out.append({
            "id": r["id"], "name": title(r),
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
    return match_name(t, ["Pending", "For today", "In Progress", "Done"], STATUS_SYNONYMS, 0.7)


def _norm_priority(text):
    return match_name((text or "").strip(), ["High", "Medium", "Low"], {"urgent": "High", "important": "High", "normal": "Medium"}, 0.7)


def list_todos(status="") -> str:
    """Lists open items on the Notion to-do list, most urgent first. status: optional filter such as "for today" or "in progress"."""
    items = todos()
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
    parts = [f"{st}: " + join(f"{t['name']} ({t['priority'].lower()} priority)" for t in ts) for st, ts in groups.items()]
    more = f" And {n - len(shown)} more." if n > len(shown) else ""
    return f"You have {n} open to-do{'s' if n != 1 else ''}. " + ". ".join(parts) + "." + more


def add_todo(name="", priority="Medium", status="Pending") -> str:
    """Adds an item to the Notion to-do list. priority: High, Medium or Low."""
    name = (name or "").strip()
    if not name:
        return "What should I add to your to-do list?"
    pri = _norm_priority(priority) or "Medium"
    st = _norm_status(status) or "Pending"
    for t in todos():
        if t["name"].lower() == name.lower() and t["status"] != "Done":
            return f"{t['name']} is already on your to-do list."
    props = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Priority": {"select": {"name": pri}},
        "Status": {"status": {"name": st}},
    }
    page = api("POST", "/pages", {"parent": {"database_id": need("todo")["id"]}, "properties": props})
    undo_push({"kind": "create", "page_id": page["id"], "label": f"the to-do {name}"})
    return f"Added {name} to your to-do list as {pri.lower()} priority, {st.lower()}."


def update_todo(name="", status="", priority="") -> str:
    """Changes a Notion to-do's status (Pending, For today, In Progress, Done) or priority. name: which to-do."""
    st, pri = _norm_status(status), _norm_priority(priority)
    if not name or not (st or pri):
        return "Tell me which to-do to change and what to change it to, such as done or high priority."
    item, ambiguous = fuzzy_pick(name, todos(), key=lambda t: t["name"])
    if not item:
        return f"I could not find a to-do matching {name}."
    if ambiguous:
        return f"More than one to-do matches {name}. Which one did you mean?"
    props, previous = {}, {}
    if st:
        props["Status"], previous["Status"] = {"status": {"name": st}}, {"status": {"name": item["status"]}}
    if pri:
        props["Priority"], previous["Priority"] = {"select": {"name": pri}}, {"select": {"name": item["priority"]}}
    api("PATCH", f"/pages/{item['id']}", {"properties": props})
    undo_push({"kind": "update", "page_id": item["id"], "label": f"the to-do {item['name']}", "previous": previous})
    changes = join([x for x in (f"status {st}" if st else "", f"{pri.lower()} priority" if pri else "") if x])
    return f"Updated {item['name']}: {changes}."


def todo_nudge() -> str:
    """Internal: a short nudge about what is still open for today (empty when nothing)."""
    import os
    import sys
    if not os.getenv("NOTION_TOKEN"):
        return ""
    try:
        items = [t for t in todos() if t["status"] in ("For today", "In Progress") or
                 (t["priority"] == "High" and t["status"] != "Done")]
    except Exception as e:
        print(f"[nudge] {e}", file=sys.stderr)
        return ""
    items = [t for t in items if t["status"] != "Done"]
    if not items:
        return ""
    n = len(items)
    names = ", ".join(t["name"] for t in items[:2])
    return f"You still have {n} thing{'s' if n != 1 else ''} open for today, such as {names}. Want to get started on one?"
