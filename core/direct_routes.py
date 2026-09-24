"""Requests that map one-to-one onto a tool are handled here, without asking
the model to pick the tool.

The 4B model is unreliable at this: it answered "The alarm is set" without
setting anything, invented the time, and made up a joke tool. A regex that
recognises "set a timer for 10 minutes" and calls set_timer is exact, instant,
and cannot claim something it did not do. Anything that does not match falls
through to the model and the skills as before.

The rules themselves live with the skill they belong to (each skill's
`direct_routes` — see skills/base.py); this module just concatenates them
all and does the matching. Adding a skill's own direct routes never means
touching this file.

match(text) -> (tool_name, arguments) or None.
"""

from skills.registry import all_direct_routes

_RULES = None   # built lazily: skills/registry.py's own module-level discovery
                # must finish importing every skill first


def _rules():
    global _RULES
    if _RULES is None:
        _RULES = all_direct_routes()
    return _RULES


def match(text: str):
    t = (text or "").strip()
    if not t or len(t.split()) > 30:
        return None
    for tool, rx, args in _rules():
        m = rx.search(t)
        if m:
            return tool, args(m)
    return None
