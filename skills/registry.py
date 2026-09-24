"""Discovers every skill under skills/ and answers the questions the rest of
the app asks of "all the skills" collectively: what to route a message to,
what to tell the model about a skill, what the always-on base tools are, and
what every tool function is (for the MCP server to register).

A skill is any *subpackage* directly under skills/ (skills/finance/,
skills/mac/, ...) whose __init__.py defines `SKILL = Skill(...)`. Nothing
needs to be registered anywhere else — dropping in a new folder is enough;
see skills/README.md.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import Callable

from skills.base import Skill

SKILLS: dict[str, Skill] = {}
ALWAYS_ON: list[Skill] = []


def _discover():
    import skills as _pkg
    for finder, name, ispkg in pkgutil.iter_modules(_pkg.__path__):
        if not ispkg:
            continue    # base.py/palette.py/registry.py live alongside the skill folders, not as skills themselves
        mod = importlib.import_module(f"skills.{name}")
        skill = getattr(mod, "SKILL", None)
        if skill is None:
            raise RuntimeError(f"skills/{name}/__init__.py has no SKILL = Skill(...) — see skills/README.md")
        if skill.always_on:
            ALWAYS_ON.append(skill)
        else:
            SKILLS[skill.name] = skill


_discover()

UNDO_RE = re.compile(r"\b(undo|scratch that|take that back|remove that)\b", re.IGNORECASE)
STICKY_TURNS = 2
EMBED_MIN = 0.62       # cosine to the best matching example
EMBED_MARGIN = 0.06    # ...and this far ahead of the next skill
_centroids = None


def all_tools() -> list[Callable]:
    """Every tool function across every skill (routed, always-on or
    internal-only) — what the MCP server registers. See mcp_server/server.py."""
    tools = []
    for skill in (*SKILLS.values(), *ALWAYS_ON):
        tools.extend(skill.tools)
        tools.extend(skill.internal_tools)
    return tools


def all_direct_routes() -> list[tuple]:
    """Every skill's direct_routes, concatenated — see core/direct_routes.py."""
    routes = []
    for skill in (*SKILLS.values(), *ALWAYS_ON):
        routes.extend(skill.direct_routes)
    return routes


def _embed_route(text: str, embed) -> str | None:
    """Fallback for phrasings the regex misses ("what does my week look
    like"): nearest skill by sentence embedding, using the model SKYE
    already has loaded. Deliberately conservative — a wrong skill is worse
    than no skill."""
    global _centroids
    import numpy as np
    if _centroids is None:
        _centroids = {}
        for name, skill in SKILLS.items():
            if skill.utterances:
                _centroids[name] = np.asarray(embed(skill.utterances), dtype=np.float32)
    q = np.asarray(embed([text]), dtype=np.float32)[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = {}
    for name, m in _centroids.items():
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        scores[name] = float((m @ q).max())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] >= EMBED_MIN and (len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= EMBED_MARGIN):
        return ranked[0][0]
    return None


def route_skill(text: str, sticky: str | None = None, embed=None) -> str | None:
    """Skill name for a message, or None. `sticky` is the skill of the last
    turn(s): "undo that" or a bare follow-up should stay in it. `embed`
    (list of strings -> vectors) enables the embedding fallback."""
    direct_hit = next((s for s in SKILLS.values() if s.direct and s.pattern and s.pattern.search(text)), None)
    if direct_hit:
        return direct_hit.name
    scores = {name: len(s.pattern.findall(text)) * s.weight for name, s in SKILLS.items() if s.pattern}
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    if sticky and UNDO_RE.search(text):
        return sticky
    if embed is not None and len(text.split()) >= 3:
        return _embed_route(text, embed)
    return None


def manifest_text(skill_name: str) -> str:
    s = SKILLS[skill_name]
    lines = [s.manifest, ""]
    for user, call in s.examples:
        lines += [f"User: {user}", f"SKYE: CALL_FUNC: {call}", ""]
    return "\n".join(lines).rstrip()


def ui_for(skill_name: str) -> str | None:
    return SKILLS[skill_name].ui
