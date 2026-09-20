"""Rich content for the screen, attached to a tool's spoken text.

A tool returns its sentence as usual and may append blocks with attach(); core
splits them off (core/core_v2.py `_split_canvas`), speaks only the sentence and
sends the blocks to the browser as a `canvas` frame. Block types the browser
draws: markdown, links, chart (bar/line), table, agenda, image, video (file or
YouTube link), audio.
"""

import json

MARKER = "\n[CANVAS]"


def attach(text, title="", *blocks):
    payload = {"title": title, "blocks": [b for b in blocks if b]}
    return text + MARKER + json.dumps(payload, ensure_ascii=False)


def bar_chart(title, labels, values, unit=""):
    return {"type": "chart", "kind": "bar", "title": title, "labels": labels, "values": values, "unit": unit}


def links(items):
    return {"type": "links", "items": [{"title": t, "url": u, "source": s} for t, u, s in items]}


def agenda(items):
    return {"type": "agenda", "items": [{"time": t, "title": n} for t, n in items]}


def split(text):
    """(spoken text, payload dict or None) — the inverse of attach()."""
    if MARKER not in text:
        return text, None
    spoken, _, raw = text.partition(MARKER)
    try:
        return spoken, json.loads(raw)
    except ValueError:
        return spoken, None
