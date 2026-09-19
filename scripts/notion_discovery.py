"""Maps everything the SKYE Notion integration can see.

Prints a readable report of every top-level page and database: databases with
their columns (type, select/status options, relation targets), row counts and
a few sample rows; pages with their block structure. Use it to decide which
purpose-built tools SKYE needs and to write the workspace map.

Read-only. Writes the full JSON to the path given as argv[1] (default
notion_discovery.json in the current directory — keep that out of git, it
contains your data).
"""

import json
import os
import re
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = re.search(r"^NOTION_TOKEN=(.*)$", open(os.path.join(ROOT, ".env")).read(), re.M).group(1).strip().strip("\"'")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
_last = [0.0]


def api(method, path, body=None):
    # Notion allows ~3 requests/second; stay under it.
    wait = 0.34 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    for attempt in range(4):
        r = requests.request(method, "https://api.notion.com/v1" + path, headers=HEADERS, json=body, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        return r.json()
    return {"error": "rate limited"}


def paginate(method, path, body=None, limit=None):
    out, cursor = [], None
    while True:
        b = dict(body or {}, page_size=100)
        p = path
        if cursor:
            if method == "POST":
                b["start_cursor"] = cursor
            else:
                p = path + ("&" if "?" in path else "?") + f"start_cursor={cursor}"
        r = api(method, p, b if method == "POST" else None)
        out += r.get("results", [])
        if not r.get("has_more") or (limit and len(out) >= limit):
            return out[:limit] if limit else out
        cursor = r.get("next_cursor")


def title_of(obj):
    if obj["object"] == "database":
        return "".join(t["plain_text"] for t in obj.get("title", []))
    for v in obj.get("properties", {}).values():
        if v["type"] == "title":
            return "".join(t["plain_text"] for t in v["title"])
    return ""


def flat(prop):
    t = prop["type"]
    v = prop[t]
    if t in ("title", "rich_text"):
        return "".join(x["plain_text"] for x in v)
    if t in ("select", "status"):
        return v["name"] if v else None
    if t == "multi_select":
        return [x["name"] for x in v]
    if t == "date":
        return v["start"] if v else None
    if t == "people":
        return [x.get("name") for x in v]
    if t == "relation":
        return f"{len(v)} linked"
    if t == "formula":
        return v.get(v["type"])
    if t == "rollup":
        return v.get(v["type"])
    return v


def describe_schema(db):
    cols = {}
    for name, p in db["properties"].items():
        t = p["type"]
        info = {"type": t}
        if t in ("select", "multi_select", "status"):
            info["options"] = [o["name"] for o in p[t]["options"]]
        elif t == "relation":
            info["relates_to"] = p["relation"].get("database_id")
        elif t == "formula":
            info["expression"] = p["formula"].get("expression", "")[:120]
        elif t == "rollup":
            info["rollup"] = f"{p['rollup'].get('function')} of {p['rollup'].get('rollup_property_name')}"
        cols[name] = info
    return cols


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "notion_discovery.json"
    print("Searching workspace...", flush=True)
    items = paginate("POST", "/search")
    dbs = {x["id"]: x for x in items if x["object"] == "database"}
    pages = {x["id"]: x for x in items if x["object"] == "page"}
    print(f"  {len(dbs)} databases, {len(pages)} pages (rows count as pages)\n", flush=True)

    # Rows belong to databases; the rest are standalone pages.
    row_pages = {i: p for i, p in pages.items() if p["parent"]["type"] == "database_id"}
    standalone = {i: p for i, p in pages.items() if i not in row_pages}

    # Parent chain -> which top-level page each database lives under.
    title_cache = {i: title_of(x) for i, x in {**dbs, **pages}.items()}

    def resolve_top(obj, depth=0):
        par = obj["parent"]
        t = par["type"]
        if t == "workspace" or depth > 8:
            return title_of(obj) or "(untitled)"
        pid = par.get(t)
        if pid in pages:
            return resolve_top(pages[pid], depth + 1)
        if pid in dbs:
            return resolve_top(dbs[pid], depth + 1)
        if t in ("page_id", "block_id"):
            r = api("GET", f"/pages/{pid}") if t == "page_id" else api("GET", f"/blocks/{pid}")
            if r.get("object") in ("page", "block"):
                if r["object"] == "page":
                    pages[pid] = r
                    title_cache[pid] = title_of(r)
                    return resolve_top(r, depth + 1)
                return resolve_top({"parent": r["parent"], "properties": {}}, depth + 1) if r.get("parent") else "?"
        return "?"

    report = {"databases": [], "pages": []}
    for did, db in dbs.items():
        rows = paginate("POST", f"/databases/{did}/query")
        cols = describe_schema(db)
        samples = []
        for row in rows[:3]:
            samples.append({k: flat(v) for k, v in row["properties"].items() if flat(v) not in (None, "", [])})
        report["databases"].append({
            "id": did, "title": title_of(db), "lives_under": resolve_top(db),
            "columns": cols, "row_count": len(rows), "sample_rows": samples,
        })
        print(f"DATABASE  {title_of(db) or '(untitled)'}   [{did}]", flush=True)
        print(f"  under: {report['databases'][-1]['lives_under']}   rows: {len(rows)}")
        for name, info in cols.items():
            extra = ""
            if "options" in info:
                extra = "  options: " + ", ".join(info["options"][:10])
            elif "relates_to" in info:
                extra = f"  -> {title_cache.get(info['relates_to'], info['relates_to'])}"
            elif "expression" in info:
                extra = f"  = {info['expression']}"
            elif "rollup" in info:
                extra = f"  = {info['rollup']}"
            print(f"    - {name}: {info['type']}{extra}")
        for s in samples:
            print(f"    e.g. {json.dumps(s, ensure_ascii=False)[:230]}")
        print()

    print("=" * 70 + "\nSTANDALONE PAGES (structure)\n" + "=" * 70, flush=True)
    for pid, pg in standalone.items():
        if pg["parent"]["type"] not in ("workspace", "page_id"):
            continue
        blocks = paginate("GET", f"/blocks/{pid}/children")
        kinds = {}
        for b in blocks:
            kinds[b["type"]] = kinds.get(b["type"], 0) + 1
        headings = [
            "".join(t["plain_text"] for t in b[b["type"]]["rich_text"])
            for b in blocks if b["type"].startswith("heading")
        ][:12]
        children = [
            (b["type"], b.get("child_page", b.get("child_database", {})).get("title", ""))
            for b in blocks if b["type"] in ("child_page", "child_database")
        ]
        report["pages"].append({"id": pid, "title": title_of(pg), "parent": pg["parent"]["type"],
                                "blocks": kinds, "headings": headings, "children": children})
        print(f"PAGE  {title_of(pg) or '(untitled)'}   [{pid}]   parent: {pg['parent']['type']}")
        print(f"  blocks: {kinds}")
        if headings:
            print(f"  headings: {headings}")
        for kind, name in children:
            print(f"  contains {kind}: {name}")
        print()

    json.dump(report, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
