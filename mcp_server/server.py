"""SKYE's local tool-execution MCP server.

Every tool SKYE can call used to be a plain Python function looked up in an
in-process dict (`core_v2.py`'s old `TOOLS`/`register_tool()`). This is the
same set of tools, faithfully ported, served over MCP instead — a real
protocol boundary, run as its own subprocess, so tool execution is no longer
tied to being "just a function call inside the same process as the LLM."

Deliberately does NOT import core_v2.py: that module loads Gemma/XTTS/Whisper
at import time, and this process has no business paying for any of that —
it only needs its own TaskStore/MemoryManager instances (pointed at the same
sqlite files the main server uses; WAL mode in both makes that safe) and the
handful of third-party clients (requests, geocoder, wikipedia) the ported
tools already depended on.
"""

import os
import sys
from datetime import datetime

# stdio transport requires stdout to carry ONLY JSON-RPC protocol messages —
# any stray print/log line from a dependency corrupts the stream from the
# client's point of view (seen directly: MemoryManager's SentenceTransformer
# load emits something to stdout during import, before mcp.run() even takes
# over, and the client choked on it as a malformed first message). Redirect
# the *name* `sys.stdout` to stderr for the whole module-init phase; real
# stdout is saved and only handed back for the actual mcp.run() call below.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

import requests
import geocoder
import webbrowser
import wikipedia
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory.tasks import TaskStore, next_occurrence, format_due
from memory.manager import MemoryManager

load_dotenv(os.path.join(ROOT, ".env"))

TASKS = TaskStore(ROOT)
MEMORY = MemoryManager(ROOT)

mcp = MCPServer("skye-tools")


@mcp.tool()
def tell_time() -> str:
    """Returns the current local time, e.g. '4:07 PM'."""
    now = datetime.now()
    current_time = now.strftime("%I:%M %p")
    if current_time[0] == "0":
        current_time = current_time[1:]
    return current_time


@mcp.tool()
def get_weather() -> str:
    """Returns current weather for the user's location, geolocated via IP."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    g = geocoder.ip("me")
    latitude, longitude = g.latlng
    url = (
        f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}"
        f"&appid={api_key}&units=metric"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        city = data["name"]
        temp = data["main"]["temp"]
        sky = data["weather"][0]["description"]
        return f"{city}: {temp}°C, {sky}"
    return "N/A: N/A, N/A"


@mcp.tool()
def set_alarm(time: str) -> str:
    """Sets an alarm for a given clock time, e.g. '7:30 AM'."""
    try:
        TASKS.add_task(f"Alarm: {time}", next_occurrence(time), source="explicit")
    except Exception as e:
        return f"Could not set alarm for {time} — {e}"
    return f"Alarm set for {time}."


@mcp.tool()
def set_reminder(time: str, task: str) -> str:
    """Sets a reminder for a given clock time with a description of the task."""
    try:
        TASKS.add_task(task, next_occurrence(time), source="explicit")
    except Exception as e:
        return f"Could not set reminder '{task}' — {e}"
    return f"Reminder set: {task} at {time}."


@mcp.tool()
def fetch_news(query: str) -> str:
    """Fetches recent news headlines matching a query."""
    api_key = os.getenv("NEWSDATA_API_KEY")
    url = f"https://newsdata.io/api/1/latest?apikey={api_key}&q={query}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return f"Failed to retrieve news for {query}."
        articles = response.json().get("results", [])
        if not articles:
            return f"No news found for {query}."
        summary = f"Top {min(5, len(articles))} News Titles for '{query}':\n"
        for i, article in enumerate(articles[:5]):
            title = article.get("title", "No Title")
            description = article.get("description", "") or ""
            if len(description) > 100:
                description = description[:100] + "..."
            summary += f"{i+1}. {title} - {description}\n"
        return summary
    except Exception as e:
        return f"Error fetching news: {e}"


def _web_open_and_ack(site_key, query=None):
    sites = {
        "youtube": "https://youtube.com",
        "wikipedia": "https://wikipedia.com",
        "google": "https://google.com",
        "spotify": "https://open.spotify.com",
    }
    url = sites.get(site_key, "https://google.com")
    if query:
        if site_key == "youtube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"Opening YouTube for {query}."
        if site_key == "spotify":
            webbrowser.open(f"https://open.spotify.com/search/{query}")
            return f"Opening Spotify for {query}."
    webbrowser.open(url)
    return f"Opening {site_key}."


@mcp.tool()
def open_youtube(query: str) -> str:
    """Opens YouTube search results for a query in the default browser."""
    return _web_open_and_ack("youtube", query)


@mcp.tool()
def open_spotify(query: str) -> str:
    """Opens Spotify search results for a query in the default browser."""
    return _web_open_and_ack("spotify", query)


@mcp.tool()
def open_calendar_web() -> str:
    """Opens the Google Calendar website in the default browser."""
    return _web_open_and_ack("google", "calendar")


@mcp.tool()
def web_search(query: str) -> str:
    """Searches the web for a query and returns a short summary."""
    if not query:
        return "No query provided."
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "Web search is currently unavailable."


@mcp.tool()
def list_tasks() -> str:
    """Lists upcoming pending tasks and reminders."""
    upcoming = TASKS.get_upcoming(10)
    if not upcoming:
        return "No pending tasks or reminders."
    return "; ".join(f"{t['description']} ({format_due(t['due_at'])})" for t in upcoming)


@mcp.tool()
def complete_task(description: str) -> str:
    """Marks the most recent pending task matching a description as done."""
    match = TASKS.find_recent_pending(description)
    if not match:
        return "No matching pending task found."
    TASKS.mark_status(match["id"], "done")
    return f"Marked '{match['description']}' as done."


@mcp.tool()
def get_diagnostics() -> str:
    """Reports SKYE's own performance diagnostics — guardrail triggers and tool reliability."""
    diagnostics_file = os.path.join(ROOT, "logs", "diagnostics.json")
    if not os.path.exists(diagnostics_file):
        return "No diagnostics recorded yet — nothing has gone through a consolidation pass."
    import json

    with open(diagnostics_file, "r") as f:
        report = json.load(f)
    turns = report.get("turns_analyzed", 0)
    if not turns:
        return "No diagnostics recorded yet."
    parts = [f"{turns} turns analyzed so far."]
    guardrails = report.get("guardrails") or {}
    if guardrails:
        top = sorted(guardrails.items(), key=lambda kv: kv[1], reverse=True)[:3]
        parts.append(
            "Most common guardrail triggers: "
            + ", ".join(f"{name} ({count})" for name, count in top) + "."
        )
    tools = report.get("tools") or {}
    if tools:
        tool_bits = [f"{name} ({t['success']} ok, {t['failure']} failed)" for name, t in tools.items()]
        parts.append("Tool reliability: " + ", ".join(tool_bits) + ".")
    return " ".join(parts)


if __name__ == "__main__":
    sys.stdout = _real_stdout
    mcp.run(transport="stdio")
