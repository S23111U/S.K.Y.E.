"""SKYE's local tool-execution MCP server.

Every tool SKYE can call lives under skills/ — one Python package per skill,
each exporting a `SKILL = Skill(...)` descriptor (see skills/base.py and
skills/README.md). This file's only job is to find them all and register
every one of their tool functions with the MCP server; it has no tool
implementations of its own, and adding a new skill never means editing this
file.

Deliberately does NOT import core_v2.py: that module loads Gemma/Whisper at
import time, and this process has no business paying for any of that — it
only needs the skills' own (lazily-constructed — see skills/shared.py)
TaskStore/MemoryManager instances and the tool packages' own third-party
clients (requests, geocoder, tavily...).
"""

import os
import sys

# stdio transport requires stdout to carry ONLY JSON-RPC protocol messages —
# any stray print/log line from a dependency corrupts the stream from the
# client's point of view (seen directly: MemoryManager's SentenceTransformer
# load emits something to stdout during import, before mcp.run() even takes
# over, and the client choked on it as a malformed first message). Redirect
# the *name* `sys.stdout` to stderr for the whole module-init phase; real
# stdout is saved and only handed back for the actual mcp.run() call below.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skills import registry

load_dotenv(os.path.join(ROOT, ".env"))

mcp = MCPServer("skye-tools")
for _fn in registry.all_tools():
    mcp.tool()(_fn)


def _prewarm():
    """Builds each skill's slow lookups (Notion database discovery, the notes
    index, ...) in the background at startup — the first Notion call
    otherwise takes ~20 s. Every skill's own module decides what, if
    anything, needs this; server.py just gives them the chance."""
    import threading

    def run():
        if os.getenv("NOTION_TOKEN"):
            try:
                from skills.notion_client import databases
                from skills.finance.tools import _categories
                from skills.knowledge.tools import _notes_index
                databases()
                _categories()
                _notes_index()
                print("[notion] warmed", file=sys.stderr)
            except Exception as e:
                print(f"[notion] prewarm failed: {e}", file=sys.stderr)

    threading.Thread(target=run, daemon=True).start()


_prewarm()

if __name__ == "__main__":
    sys.stdout = _real_stdout
    mcp.run(transport="stdio")
