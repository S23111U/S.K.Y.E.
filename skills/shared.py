"""Shared, lazily-constructed resources for skill implementations.

Skill *packages* (skills/finance/, skills/mac/, ...) are imported from two
very different places: the MCP tool-server subprocess, which actually runs
the tools and needs real TaskStore/MemoryManager instances, and the main
process (core_v2.py), which only wants routing metadata (a skill's pattern,
manifest, UI colour) and must not pay for anything heavier than that — same
reasoning mcp_server/server.py already documented for not importing
core_v2.py, in the other direction. So nothing here (or in any skill module)
may construct a real resource at import time; everything is built lazily,
the first time a tool actually runs, which only ever happens inside the
subprocess.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_tasks = None
_memory = None


def get_tasks():
    global _tasks
    if _tasks is None:
        from memory.tasks import TaskStore
        _tasks = TaskStore(ROOT)
    return _tasks


def get_memory():
    global _memory
    if _memory is None:
        from memory.manager import MemoryManager
        _memory = MemoryManager(ROOT)
    return _memory
