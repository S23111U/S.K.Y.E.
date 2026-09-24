"""Learning-notes tools (read only) — SKYE's Notion notebooks: courses,
lectures, tutorials. See skills/notion_client.py for the shared plumbing."""

import difflib
import re
from datetime import date

from skills.notion_client import NotionError, api, cached, databases, fuzzy_pick, join, paginate, title

SOURCES_MARKER = "[WEB SOURCES]"   # shared shape with web_search — core_v2 summarises rather than reads it verbatim


def _ancestor_course(db):
    """Title of the nearest ancestor page of a Notes database (its course)."""
    parent = db["parent"]
    for _ in range(8):
        t = parent["type"]
        if t == "workspace":
            return "Learning Notes"
        try:
            obj = api("GET", f"/pages/{parent[t]}") if t == "page_id" else api("GET", f"/blocks/{parent[t]}")
        except NotionError:
            return "Learning Notes"
        if obj["object"] == "page":
            return title(obj) or "Learning Notes"
        parent = obj["parent"]
    return "Learning Notes"


def _notes_index():
    def load():
        notebooks = {}
        for nb_db in databases().get("notebooks", []):
            for r in paginate("POST", f"/databases/{nb_db['id']}/query"):
                notebooks[r["id"]] = title(r)
        index = []
        for db in databases()["notes"]:
            course = _ancestor_course(db)
            for r in paginate("POST", f"/databases/{db['id']}/query"):
                rel = r["properties"]["Notebook"]["relation"]
                index.append({
                    "id": r["id"], "name": title(r), "course": course,
                    "notebook": notebooks.get(rel[0]["id"], "") if rel else "",
                    "edited": r["last_edited_time"][:10],
                })
        return index
    return cached("notes_index", 600, load)


def _search_key(n):
    return f"{n['course']} {n['notebook']} {n['name']}"


def _label(n):
    return ", ".join(x for x in (n["course"], n["notebook"], n["name"]) if x)


def search_notes(query="") -> str:
    """Finds learning notes by course, notebook or title, e.g. "DSA week 3"."""
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
    for b in paginate("GET", f"/blocks/{block_id}/children"):
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


def read_note(query="") -> str:
    """Reads a learning note so it can be summarised or questioned. query: course and note title, e.g. "DSA week 3"."""
    index = _notes_index()
    item, ambiguous = fuzzy_pick(query, index, key=_search_key)
    if not item:
        return f"I could not find a note matching {query}."
    if ambiguous:
        top = sorted(index, key=lambda n: -difflib.SequenceMatcher(None, (query or "").lower(), _search_key(n).lower()).ratio())[:3]
        return "More than one note matches. Did you mean " + join([_label(n) for n in top]) + "?"
    body = _block_text(item["id"])
    if not body.strip():
        return f"The note {_label(item)} is empty."
    return (f"{SOURCES_MARKER} Today is {date.today():%Y-%m-%d}.\n"
            f"[1] {_label(item)} (your Notion learning notes, last edited {item['edited']}): {body}")
