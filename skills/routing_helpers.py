"""Small regex building blocks shared by more than one skill's `direct_routes`
(see skills/base.py). Not a skill itself — just building blocks.
"""

import re

I = re.IGNORECASE
END = r"[\s.!?]*$"
LEAD = r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?|please\s+)?"

PERIOD = (r"(?P<p>today|tomorrow|tonight|this week|next week|this weekend|this month|last month|this year|all time|"
          r"yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
          r"january|february|march|april|may|june|july|august|september|october|november|december)")


def period_group(m: re.Match) -> str:
    p = (m.groupdict().get("p") or "").lower()
    return "today" if p == "tonight" else p


def category_from(text: str) -> str:
    c = re.search(r"\bon\s+([a-z][a-z ]{1,25}?)(?:\s+(?:this|last|in|today|for|during|so)\b|[.?!]*$)", text, I)
    return c.group(1).strip() if c and c.group(1).strip().lower() not in ("my", "the", "it") else ""
