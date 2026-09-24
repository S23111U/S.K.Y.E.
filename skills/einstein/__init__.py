"""The "einstein" skill: opt-in deep-thinking mode via Gemini extended
thinking. Unlike every other skill, this one never reaches the normal
CALL_FUNC/MCP tool-call flow — `direct=True` tells core_v2's engine to hand
the turn straight to core/einstein.py instead (see core_v2.py's
_einstein_turn). That logic stays in core/ rather than moving under
skills/ since core_v2.py's special-cased turn handling needs it directly,
not through a tool call — see core/einstein.py for why (it's also the shared
backend behind the always-on analyze_video tool, see skills/core_tools/media.py).
"""

from core.einstein import TRIGGER_RE
from skills.base import Skill

SKILL = Skill(
    name="einstein",
    direct=True,
    pattern=TRIGGER_RE,
)
