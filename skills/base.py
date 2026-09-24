"""The plugin contract every skill implements.

Before this, adding one skill meant hand-editing four separate files: a
tool-wrapper function per tool in mcp_server/server.py, a routing entry in
core/skills.py, a palette colour in clients/browser.html, and (for anything
Mac-native) yet another file for the actual implementation. It was easy for a
tool to exist in one place and never get wired into another — several of the
gaps found in testing were exactly that.

Now a skill is one Python package under skills/ that exports a single `SKILL
= Skill(...)` object. Everything else — registering its tools with the MCP
server, adding it to the router, giving it a UI colour, listing it in the
model's manifest — is done once, generically, by reading this object. See
skills/README.md for the walkthrough of adding a new one.
"""

from __future__ import annotations

import functools
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Pattern


@dataclass
class Skill:
    name: str
    """Short identifier, e.g. "finance". Used as the MCP tool-namespace key,
    the skill router's dict key, and (unless `ui` overrides it) the browser
    palette name."""

    tools: list[Callable] = field(default_factory=list)
    """The actual tool functions. Each needs a docstring (becomes the MCP
    tool's description) and typed keyword arguments with defaults (becomes
    its schema) — exactly how an @mcp.tool() function already looks; nothing
    extra is required. Wrap one in @safe_tool(...) to turn its exceptions
    into a spoken sentence instead of a crash (see below)."""

    pattern: Pattern | None = None
    """Regex: a message matching this routes to the skill, adding `manifest`
    to what the model sees for that one turn. None if the skill has no
    router entry of its own (e.g. it only exposes always-on tools)."""

    manifest: str = ""
    """What the model is told about this skill's tools when it's routed to:
    the tool signatures and any usage notes. Left "" for a skill handled
    entirely outside the normal tool-call flow (see einstein/)."""

    examples: list[tuple[str, str]] = field(default_factory=list)
    """Few-shot (user text, CALL_FUNC JSON string) pairs appended after the
    manifest — small models route much more reliably with 1-3 of these."""

    ui: str | None = None
    """Browser palette key this skill switches the UI to while answering.
    Defaults to `name` if not given; None means "don't recolour"."""

    color: tuple[int, int, int] | None = None
    """(r, g, b) 0-255 for the browser's THREE.Color primary — the *only*
    place this skill's colour needs to be chosen; the rest of the palette
    (secondary/atmosphere/fog/CSS hex) is derived from it automatically. See
    clients/palette.py."""

    utterances: list[str] = field(default_factory=list)
    """A handful of example phrasings for the embedding-based routing
    fallback (catches things the regex misses, e.g. "what does my week look
    like"). Optional — the regex is the primary router."""

    weight: int = 1
    """Multiplies this skill's regex-match score against others when a
    message could plausibly match more than one (see core/direct_routes.py's
    equivalent for the deterministic layer). Raise it only for a skill whose
    phrases are unusually specific and keep losing to a shorter, vaguer
    pattern elsewhere."""

    direct: bool = False
    """True for a skill that intercepts the turn itself before the normal
    tool-call flow even starts (currently only einstein/, which hands the
    question to Gemini directly rather than emitting a CALL_FUNC)."""

    always_on: bool = False
    """True for tools that should be visible to the model on every turn
    rather than only when routed to (time, weather, alarms, web search...).
    An always-on skill's `pattern`/`manifest`/`ui`/`examples` are unused —
    only its `tools` and `base_manifest` matter."""

    base_manifest: str = ""
    """For an always_on skill only: not read by anything yet — the persona's
    permanent tool list (prompts/skye_persona.txt) is still hand-written,
    deliberately, since it barely changes and a subtly-wrong generated prompt
    is a worse failure mode than a one-line manual edit. Documents intent for
    now; wire it in (see skills/registry.py) if that list starts changing
    often enough to be worth the risk."""

    internal_tools: list[Callable] = field(default_factory=list)
    """Tools that exist for SKYE's own scheduler to call directly (a to-do
    nudge, a pre-event alert) — registered with the MCP server the same as
    any other tool so the scheduler can reach them, but never mentioned in a
    manifest, since the model choosing to call one on its own makes no sense."""

    direct_routes: list[tuple[str, Pattern, Callable]] = field(default_factory=list)
    """(tool_name, regex, lambda match -> kwargs dict) triples: requests that
    map onto exactly one of this skill's tools so plainly that asking the
    model to pick is just a chance to get it wrong (a made-up time, a
    hallucinated "done" with no tool call...). Checked before the model ever
    sees the turn — see core/direct_routes.py."""

    def __post_init__(self):
        if self.ui is None and not self.always_on:
            self.ui = self.name


def safe_tool(friendly_error: str = "I could not do that just now.", *, catch: tuple[type[Exception], ...] = (Exception,)):
    """Decorates a tool function so any exception it raises becomes a spoken
    sentence instead of crashing the MCP call. `friendly_error` is the
    fallback message; raise a `SkillError(message)` from inside the function
    for a specific one instead (its text is used verbatim)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except SkillError as e:
                return str(e)
            except catch as e:
                print(f"[{fn.__module__}.{fn.__name__}] failed: {e}", file=sys.stderr)
                return friendly_error
        return wrapped
    return deco


class SkillError(Exception):
    """Raise with a spoken-ready message from inside a @safe_tool function
    for a specific failure reason (e.g. "I need permission to..."); anything
    else raised falls back to the decorator's generic friendly_error."""
