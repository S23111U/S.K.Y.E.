"""Confirm-before-acting: a small held-tool-call gate for anything risky enough
(sending, calling, deleting, running commands) that SKYE should read it back
before doing it. The model asks for the tool as usual; core_v2.py checks the
tool's name against CONFIRM_TOOLS and, if it matches, holds the call in
_PENDING and speaks a question instead of running it. The *next* turn is then
checked against _PENDING first: "yes" runs the held call, anything else (or a
bare "no") drops it. Tools get added to CONFIRM_TOOLS as they are built;
SKYE_CONFIRM_TOOLS=name,name in .env adds more without a code change.

_PENDING is a module-level dict shared by reference with core_v2.py — always
mutated with .update()/.clear(), never reassigned, so importing it here works
correctly across the module boundary without a `global` declaration on either
side.
"""

import os
import re

CONFIRM_TOOLS = {t.strip() for t in os.getenv("SKYE_CONFIRM_TOOLS", "").split(",") if t.strip()}
CONFIRM_TTL_S = 90
_PENDING = {}
YES_RE = re.compile(r"^\W*(?:yes|yeah|yep|yup|sure|ok|okay|correct|confirm(?:ed)?|go ahead|do it|send it|please do|that'?s right|affirmative|proceed)\b[\w\s,.!]{0,30}$", re.IGNORECASE)
NO_RE = re.compile(r"^\W*(?:no|nope|nah|cancel|don'?t|do not|stop|never ?mind|abort|not now|wait)\b", re.IGNORECASE)

NO_ONLY_RE = re.compile(r"^\W*(?:no|nope|nah|cancel(?: that| it)?|never ?mind|not now|don'?t)\W*$", re.IGNORECASE)


def _describe_action(name, args):
    bits = ", ".join(f"{k} {v}" for k, v in (args or {}).items() if v not in ("", None))
    return f"{name.replace('_', ' ')}" + (f" with {bits}" if bits else "")
