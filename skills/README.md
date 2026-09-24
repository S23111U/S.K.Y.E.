# Adding a skill

A skill is one Python package under `skills/` that exports a single
`SKILL = Skill(...)` object. Nothing else needs to be touched — no wrapper
function in the MCP server, no routing entry in a central file, no manual
manifest edit. Drop in the folder, and:

- `skills/registry.py` discovers it automatically (`pkgutil.iter_modules`
  over the `skills/` package — any subpackage with a `SKILL` object counts).
- `mcp_server/server.py` registers all of its `tools` (and `internal_tools`)
  with the MCP server.
- `core/direct_routes.py` picks up its `direct_routes`.
- The router (`skills.registry.route_skill`) starts matching its `pattern`
  and, once matched, shows the model its `manifest`.

This replaces what used to be four separate places to edit by hand for one
new feature (a tool wrapper in `mcp_server/server.py`, a routing entry in
`core/skills.py`, a palette colour in `clients/browser.html`, and the actual
implementation) — which is exactly how features like reading Messages ended
up implemented but never wired into the router.

## Minimal example

```
skills/
  weather2/
    __init__.py   # SKILL = Skill(...)
    tools.py       # the actual tool function(s)
```

`skills/weather2/tools.py`:

```python
def wind_speed(city: str = "") -> str:
    """Returns the current wind speed for a city, or his home city if none is given."""
    ...
    return f"It's {mph} mph in {city}."
```

`skills/weather2/__init__.py`:

```python
import re
from skills.base import Skill
from skills.weather2.tools import wind_speed

SKILL = Skill(
    name="weather2",
    pattern=re.compile(r"\bwind speed\b", re.IGNORECASE),
    manifest='Wind tool: wind_speed{"city"}.',
    examples=[("how windy is it in Sydney?", '{"name": "wind_speed", "arguments": {"city": "Sydney"}}')],
    tools=[wind_speed],
)
```

That's the whole skill. See `skills/finance/`, `skills/todo/` or
`skills/knowledge/` for real, larger examples — finance/todo also use
`direct_routes` for the phrasings common enough to not need the model at all.

## Rules that aren't optional

- **The tool function's docstring and signature ARE its MCP schema.** There
  is no separate wrapper layer anymore — whatever you write is exactly what
  the model sees and exactly what argument names it must use. Get the
  docstring right (one sentence, plain language, mention what an empty
  argument defaults to) and never rename a parameter "for clarity" without
  checking nothing already depends on the old name.
- **Every tool takes typed keyword arguments with string defaults** (e.g.
  `city: str = ""`), matching the existing MCP tool convention — this is what
  makes the schema auto-derivable.
- **Heavy resources are lazy.** `skills/registry.py` (and therefore every
  skill's `__init__.py`) is imported by the *lightweight* main process
  (`core/core_v2.py`) just for routing metadata, as well as by the *heavy*
  MCP subprocess that actually runs tools. Never construct a real resource
  (a `TaskStore`, a `MemoryManager`, a second embedding model, a DB
  connection) at module import time — do it inside the tool function, the
  first time it's actually called. If you need `TaskStore`/`MemoryManager`,
  use `skills.shared.get_tasks()` / `get_memory()`, which construct them once
  on first use and cache the instance.
- **Wrap risky tools in `@safe_tool(...)`** (from `skills/base.py`) so a
  crash inside the tool becomes a spoken sentence instead of an MCP error the
  model doesn't know how to recover from:

  ```python
  from skills.base import safe_tool, SkillError

  @safe_tool("I could not reach the calendar just now.")
  def next_event() -> str:
      """Returns the title and time of his next calendar event."""
      if not connected():
          raise SkillError("I need calendar permission first — check System Settings.")
      ...
  ```

  Raise `SkillError(message)` for a specific, known failure; anything else
  falls back to the decorator's generic message.

## `Skill` fields (see `skills/base.py` for the full docstrings)

| Field | Purpose |
|---|---|
| `name` | Identifier — MCP namespace key, router key, default UI palette name. |
| `tools` | The tool functions, shown to the model only when this skill is routed to. |
| `pattern` | Regex; a match routes the turn to this skill and adds its `manifest`. |
| `manifest` | Tool signatures + usage notes shown to the model for that turn. |
| `examples` | 1-3 (user text, CALL_FUNC JSON) few-shot pairs — small models route much better with these. |
| `utterances` | Example phrasings for the embedding-based routing fallback (optional; catches phrasings the regex misses). |
| `weight` | Multiplier for this skill's regex score when a message could match more than one skill; raise only if this skill's phrases are unusually specific and keep losing to a vaguer one. |
| `ui` / `color` | Browser palette key, or an (r,g,b) to auto-derive one (`skills/palette.py`). |
| `always_on` | True for tools shown to the model on *every* turn, not just when routed to (time, weather, alarms...) — no `pattern`/`manifest`/`ui` needed. |
| `internal_tools` | Registered with MCP but never shown to the model — for SKYE's own scheduler to call directly (a nudge, a pre-event alert). |
| `direct_routes` | `(tool_name, regex, lambda match -> kwargs)` triples for phrasings unambiguous enough to skip the model entirely — checked before any turn reaches it (`core/direct_routes.py`). |
| `direct` | True only for a skill that intercepts the turn *before* the normal tool-call flow (currently only `einstein/`). |

## Testing a new skill

1. `python3 -c "import ast; ast.parse(open('skills/yourskill/__init__.py').read())"` for a fast syntax check.
2. Boot the real server (`python core/core_v2.py`, mode `2`) and try a phrase
   that should hit your `pattern`, one that should hit a `direct_routes`
   entry (if any), and one that shouldn't match at all (make sure it doesn't
   steal traffic from another skill).
3. Check `mcp_server/server.py`'s startup log for your tool names, and that
   there are no duplicate tool names across skills.
