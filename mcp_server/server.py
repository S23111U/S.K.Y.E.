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
handful of third-party clients (requests, geocoder, tavily) the ported
tools already depended on.
"""

import os
import re
import sys
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from datetime import datetime

import numpy as np

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
from tavily import TavilyClient
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import notion_tools
import calendar_tools
import canvas
import mac_tools
import messages_tools
from memory.tasks import TaskStore, next_occurrence, format_due, parse_when, parse_duration
from memory.manager import MemoryManager

load_dotenv(os.path.join(ROOT, ".env"))

TASKS = TaskStore(ROOT)
MEMORY = MemoryManager(ROOT)

# Prefix core_v2.py keys on to know a tool result is raw material to synthesise
# from, not a finished reply to show the user.
WEB_SOURCES_MARKER = notion_tools.SOURCES_MARKER

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
def get_weather(city: str = "") -> str:
    """Returns current weather. With no city it uses the user's location (from IP); pass city, e.g. "Salida, Colorado, USA", for anywhere else."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    place = None
    if city.strip():
        parts = [p.strip() for p in city.split(",") if p.strip()]
        # OpenWeather's geocoder wants "city,state,country" codes, not full names,
        # so try the whole thing, then progressively less.
        for q in (city, ",".join(parts[:2]), parts[0]):
            r = requests.get("http://api.openweathermap.org/geo/1.0/direct",
                             params={"q": q, "limit": 1, "appid": api_key}, timeout=10)
            if r.status_code == 200 and r.json():
                place = r.json()[0]
                break
        if not place:
            return f"I could not find a place called {city}."
        latitude, longitude = place["lat"], place["lon"]
        name = place["name"] + (f", {place['state']}" if place.get("state") and place["state"] != place["name"] else "")
    else:
        g = geocoder.ip("me")
        latitude, longitude = g.latlng
        name = None
    response = requests.get(
        "http://api.openweathermap.org/data/2.5/weather",
        params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}, timeout=10)
    if response.status_code != 200:
        return "I could not get the weather just now."
    data = response.json()
    name = name or data["name"]
    temp = round(data["main"]["temp"])
    feels = round(data["main"]["feels_like"])
    sky = data["weather"][0]["description"]
    out = f"In {name} it is {temp} degrees Celsius with {sky}"
    if abs(feels - temp) >= 3:
        out += f", feeling like {feels}"
    return out + "."


@mcp.tool()
def set_alarm(time: str) -> str:
    """Sets an alarm for a clock time such as "7:30 AM", "7am", "6:30 tomorrow"."""
    due = parse_when(time, alarm=True)
    if due is None:
        return "I did not catch that time. Could you say it like 7 30 AM?"
    TASKS.add_task(f"Alarm: {due.strftime('%-I:%M %p')}", due, source="explicit")
    return f"Alarm set for {_when_words(due)}."


@mcp.tool()
def set_timer(duration: str, label: str = "") -> str:
    """Starts a countdown timer. duration: "10 minutes", "an hour and a half", "90 seconds"."""
    delta = parse_duration(duration)
    if delta is None:
        return "I did not catch how long the timer should be."
    TASKS.add_task(f"Timer: {label.strip()}" if label.strip() else "Timer", datetime.now() + delta, source="explicit")
    total = int(delta.total_seconds())
    h, m, sec = total // 3600, (total % 3600) // 60, total % 60
    words = " ".join(f"{n} {u}{'s' if n != 1 else ''}" for n, u in ((h, "hour"), (m, "minute"), (sec, "second")) if n)
    return f"Timer set for {words}."


@mcp.tool()
def change_alarm(which: str, time: str) -> str:
    """Moves an existing alarm, timer or reminder (matched by a word from its name, or "alarm") to a new time."""
    due = parse_when(time, alarm="alarm" in which.lower() or not which.strip())
    if due is None:
        return "I did not catch the new time."
    matches = TASKS.find_pending(which if which.lower() not in ("", "alarm", "the alarm", "my alarm") else "Alarm")
    if not matches:
        return f"I could not find {which or 'an alarm'} to change."
    t = matches[0]
    TASKS.set_due(t["id"], due)
    if t["description"].startswith("Alarm"):
        with_time = f"Alarm: {due.strftime('%-I:%M %p')}"
        TASKS.conn.execute("UPDATE tasks SET description = ? WHERE id = ?", (with_time, t["id"]))
        TASKS.conn.commit()
    return f"Moved it to {_when_words(due)}."


@mcp.tool()
def cancel_alarm(which: str = "") -> str:
    """Cancels an alarm, timer or reminder matched by a word from its name. With no name, the next alarm."""
    matches = TASKS.find_pending(which if which.lower() not in ("", "alarm", "the alarm", "my alarm", "timer", "the timer") else
                                 ("Timer" if "timer" in which.lower() else "Alarm"))
    if not matches:
        return f"I could not find {which or 'an alarm'} to cancel."
    TASKS.mark_status(matches[0]["id"], "dismissed")
    return f"Cancelled {matches[0]['description']}."


@mcp.tool()
def set_reminder(time: str, task: str) -> str:
    """Sets a reminder for a time ("7pm", "tomorrow at 8", "in 20 minutes") with what to remember. SKYE speaks it when due and it is added to the Reminders app."""
    due = parse_when(time)
    if due is None:
        return "I did not catch when. Could you say the time again?"
    TASKS.add_task(task, due, source="explicit")
    extra = ""
    try:
        mac_tools.add_mac_reminder(task, time)
    except Exception as e:
        print(f"[mac] Reminders sync skipped: {e}", file=sys.stderr)
        extra = ""
    return f"Reminder set: {task}, {_when_words(due)}."


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
        # Returned as dated sources (same shape as web_search) so core_v2
        # summarises the headlines instead of reading the raw list aloud.
        lines = [f"{WEB_SOURCES_MARKER} Today is {datetime.now():%Y-%m-%d}."]
        for i, article in enumerate(articles[:5], 1):
            description = (article.get("description") or "")[:300]
            published = (article.get("pubDate") or "")[:16]
            source = article.get("source_name") or article.get("source_id") or "news"
            lines.append(
                f"[{i}] {article.get('title', '')} ({source}, published {published}): {description}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching news: {e}"


def _when_words(dt):
    return mac_tools._when_words(dt)


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


# Social pages return image captions and hashtag soup, not usable text.
_NOISY_DOMAINS = ["instagram.com", "facebook.com", "tiktok.com", "x.com", "twitter.com"]

_TIME_SENSITIVE_RE = re.compile(
    r"\b(latest|recent|recently|newest|current|currently|today|tonight|yesterday|"
    r"this (week|month|season)|last (race|match|game)|next|upcoming|schedule|"
    r"news|score|result|results|won|winner|standings)\b",
    re.IGNORECASE,
)


# --- query cache -----------------------------------------------------------
# Repeat/rephrased questions within a short window reuse the previous sources
# instead of spending another Tavily call. Keyed on the *query embedding*, not
# on stored snippets, so a cached answer can only ever be returned for a
# question that is essentially the same one. Time-sensitive queries expire fast.
CACHE_SIMILARITY = 0.92
CACHE_TTL_SENSITIVE = 3 * 3600
CACHE_TTL_GENERAL = 24 * 3600

MEMORY.conn.execute(
    "CREATE TABLE IF NOT EXISTS web_cache ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, "
    "embedding BLOB NOT NULL, result TEXT NOT NULL, created_at REAL NOT NULL)"
)
MEMORY.conn.commit()


def _cache_lookup(query: str, ttl: int):
    q = MEMORY.model.encode(query).astype(np.float32)
    rows = MEMORY.conn.execute(
        "SELECT embedding, result, created_at FROM web_cache WHERE created_at > ?",
        (time.time() - ttl,),
    ).fetchall()
    best, best_score = None, CACHE_SIMILARITY
    for emb, result, _ in rows:
        e = np.frombuffer(emb, dtype=np.float32)
        score = float(e @ q / (np.linalg.norm(e) * np.linalg.norm(q) + 1e-9))
        if score >= best_score:
            best, best_score = result, score
    return best


def _cache_store(query: str, result: str):
    emb = MEMORY.model.encode(query).astype(np.float32).tobytes()
    MEMORY.conn.execute(
        "INSERT INTO web_cache (query, embedding, result, created_at) VALUES (?, ?, ?, ?)",
        (query, emb, result, time.time()),
    )
    MEMORY.conn.commit()


_EVENT_RE = re.compile(
    r"\b(series|season|tournament|cup|league|race|match|matches|fixture|standings|"
    r"wickets|score|final|election|championship)\b",
    re.IGNORECASE,
)


def _with_year(query: str) -> str:
    """The small model often drops the year, and event queries without one
    return old seasons (seen: 2012 and 2022 matches for a 2026 series)."""
    if re.search(r"\b(19|20)\d{2}\b", query) or not _EVENT_RE.search(query):
        return query
    return f"{query} {datetime.now().year}"


def _clean_snippet(text: str) -> str:
    """Drops table-markup lines (pipes/dashes) that make useless snippets."""
    keep = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.count("|") >= 2 or set(line) <= set("-=| "):
            continue
        keep.append(line)
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def _tavily_backend(query: str, time_sensitive: bool):
    """Returns [(title, url, published, snippet)]. Raises on any API failure
    (including exhausted credits), which triggers the fallback backend."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    kwargs = dict(max_results=6, search_depth="basic", exclude_domains=_NOISY_DOMAINS)
    results = []
    if time_sensitive:
        # News topic carries publish dates. Try the tightest window first and
        # widen only if it comes back thin, so "latest" means latest.
        for window in ("week", "month"):
            results = client.search(query, topic="news", time_range=window, **kwargs).get("results") or []
            if len(results) >= 3:
                break
    if not results:
        results = client.search(query, **kwargs).get("results") or []

    def _published(r):
        try:
            return parsedate_to_datetime(r.get("published_date") or "").timestamp()
        except Exception:
            return 0.0

    # Time-sensitive: newest first (relevance-ranking put an older race ahead
    # of the newest one). Otherwise: by relevance.
    key = _published if time_sensitive else (lambda r: r.get("score", 0))
    ranked = sorted(results, key=key, reverse=True)
    return [
        (r.get("title", ""), r.get("url", ""), (r.get("published_date") or "")[:16], r.get("content"))
        for r in ranked
    ]


def _best_window(text: str, query: str, size: int = 600) -> str:
    """The stretch of a full page that overlaps the query most — pages are
    mostly navigation boilerplate, so the start is rarely the useful part."""
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    best, best_score = text[:size], -1
    for i in range(0, max(len(text) - size, 0) + 1, size // 2):
        chunk = text[i:i + size]
        score = sum(1 for w in words if w in chunk.lower())
        if score > best_score:
            best, best_score = chunk, score
    return best


def _free_backend(query: str, time_sensitive: bool):
    """Keyless fallback: DuckDuckGo (ddgs) finds URLs, Scrapling reads the top
    pages. Lower quality than Tavily — it only has to beat 'unavailable'."""
    from ddgs import DDGS
    import logging
    from scrapling.fetchers import Fetcher

    logging.getLogger("scrapling").setLevel(logging.WARNING)

    ddg = DDGS()
    hits = []
    if time_sensitive:
        try:
            hits = [
                (h.get("title", ""), h.get("url", ""), (h.get("date") or "")[:16], h.get("body", ""))
                for h in ddg.news(query, max_results=6, timelimit="m")
            ]
        except Exception as e:
            print(f"[web_search] ddgs news failed: {e}", file=sys.stderr)
    if not hits:
        hits = [
            (h.get("title", ""), h.get("href", ""), "", h.get("body", ""))
            for h in ddg.text(query, max_results=6)
        ]
    hits = [h for h in hits if not any(d in h[1] for d in _NOISY_DOMAINS)]

    out = []
    for i, (title, url, published, body) in enumerate(hits):
        snippet = body
        if i < 2 and url:
            try:
                page = Fetcher.get(url, stealthy_headers=True, timeout=10)
                if page.status == 200:
                    # Paragraph/list text only: skips nav, scripts and ad slots.
                    text = re.sub(
                        r"\s+", " ",
                        " ".join(el.get_all_text() for el in page.css("p, li, h1, h2, h3")),
                    )
                    if len(text) > 200:
                        snippet = _best_window(text, query)
            except Exception as e:
                print(f"[web_search] fetch failed for {url}: {type(e).__name__}", file=sys.stderr)
        out.append((title, url, published, snippet))
    return out


@mcp.tool()
def web_search(query: str) -> str:
    """Searches the live web for current information and returns dated sources."""
    if not query:
        return "No query provided."

    time_sensitive = bool(_TIME_SENSITIVE_RE.search(query))
    cached = _cache_lookup(query, CACHE_TTL_SENSITIVE if time_sensitive else CACHE_TTL_GENERAL)
    if cached:
        print(f"[web_search] cache hit: {query!r}", file=sys.stderr)
        return re.sub(r"Today is \d{4}-\d{2}-\d{2}", f"Today is {datetime.now():%Y-%m-%d}", cached, count=1)

    # Tavily's own generated `answer` is deliberately not used: it summarises
    # noisy snippets and was observed contradicting itself across queries.
    # Raw dated sources go back instead; core_v2 synthesises from them.
    search_query = _with_year(query)
    results, backend = [], None
    if os.getenv("TAVILY_API_KEY"):
        try:
            results, backend = _tavily_backend(search_query, time_sensitive), "tavily"
        except Exception as e:
            print(f"[web_search] Tavily failed ({type(e).__name__}: {e}); using free fallback", file=sys.stderr)
    if not results:
        try:
            results, backend = _free_backend(search_query, time_sensitive), "ddgs+scrapling"
        except Exception as e:
            print(f"[web_search] fallback failed: {type(e).__name__}: {e}", file=sys.stderr)
            return "Web search is currently unavailable."

    sources = []
    for title, url, published, snippet in results:
        snippet = _clean_snippet(snippet)
        if len(snippet) >= 40:
            sources.append((title, url, published, snippet[:500]))
    if not sources:
        return "No results found."

    print(f"[web_search] backend={backend} sources={len(sources)} query={query!r}", file=sys.stderr)
    stamp = datetime.now().strftime("%Y-%m-%d")
    lines = [f"{WEB_SOURCES_MARKER} Today is {stamp}."]
    for i, (title, url, published, snippet) in enumerate(sources[:5], 1):
        dated = f", published {published}" if published else ""
        lines.append(f"[{i}] {title} ({urlparse(url).netloc}{dated}): {snippet}")
        # RAG pipeline: kept in long-term memory, but see MemoryManager.search()
        # — web rows are not passively injected into ordinary chat.
        MEMORY.add_memory(f"[Web, {stamp}] {title} ({url}): {snippet}")
    result = "\n".join(lines)
    result = canvas.attach(result, "Sources", canvas.links([(t, u, urlparse(u).netloc) for t, u, _, _ in sources[:5]]))
    _cache_store(query, result)
    return result


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



# --- Notion: finance tracker, to-do list, learning notes (see notion_tools.py) ---
def _prewarm_notion():
    """Builds the slow lookups (database discovery, the notes index) in the
    background at startup — the first notes search otherwise takes ~20 s."""
    if not os.getenv("NOTION_TOKEN"):
        return
    try:
        notion_tools._databases()
        notion_tools._categories()
        notion_tools._notes_index()
        print("[notion] warmed", file=sys.stderr)
    except Exception as e:
        print(f"[notion] prewarm failed: {e}", file=sys.stderr)


threading.Thread(target=_prewarm_notion, daemon=True).start()



def _notion(fn, *args):
    """Runs a Notion tool; any failure becomes a spoken sentence, not a crash."""
    try:
        return fn(*args)
    except notion_tools.NotionError as e:
        print(f"[notion] {fn.__name__} failed: {e}", file=sys.stderr)
        return "I could not reach Notion just now."


@mcp.tool()
def expense_summary(period: str = "", category: str = "") -> str:
    """Total spent in a period (default this month) with a breakdown by category. period: "this month", "last month", "september", "today", "this week", "all time". category: optional, to total one category."""
    return _notion(notion_tools.expense_summary, period, category)


@mcp.tool()
def list_expenses(period: str = "", category: str = "", limit: str = "5") -> str:
    """Lists the most recent individual expenses, optionally for a period or category."""
    return _notion(notion_tools.list_expenses, period, category, limit)


@mcp.tool()
def add_expense(name: str = "", amount: str = "", category: str = "", date: str = "") -> str:
    """Logs an expense in the finance tracker. name: what it was for. amount: a number of dollars. category: e.g. Food, Rent, Transportation. date: optional, defaults to today."""
    return _notion(notion_tools.add_expense, name, amount, category, date)


@mcp.tool()
def budget_status() -> str:
    """This month's spending against each category's monthly budget."""
    return _notion(notion_tools.budget_status)


@mcp.tool()
def account_balance() -> str:
    """The current balance of the bank account in the finance tracker."""
    return _notion(notion_tools.account_balance)


@mcp.tool()
def list_todos(status: str = "") -> str:
    """Lists open items on the Notion to-do list, most urgent first. status: optional filter such as "for today" or "in progress"."""
    return _notion(notion_tools.list_todos, status)


@mcp.tool()
def add_todo(name: str = "", priority: str = "Medium", status: str = "Pending") -> str:
    """Adds an item to the Notion to-do list. priority: High, Medium or Low."""
    return _notion(notion_tools.add_todo, name, priority, status)


@mcp.tool()
def update_todo(name: str = "", status: str = "", priority: str = "") -> str:
    """Changes a Notion to-do's status (Pending, For today, In Progress, Done) or priority. name: which to-do."""
    return _notion(notion_tools.update_todo, name, status, priority)


@mcp.tool()
def search_notes(query: str = "") -> str:
    """Finds learning notes by course, notebook or title, e.g. "DSA week 3"."""
    return _notion(notion_tools.search_notes, query)


@mcp.tool()
def read_note(query: str = "") -> str:
    """Reads a learning note so it can be summarised or questioned. query: course and note title, e.g. "DSA week 3"."""
    return _notion(notion_tools.read_note, query)


@mcp.tool()
def undo_last_entry() -> str:
    """Undoes the last thing added to or changed in Notion by SKYE (an expense, a to-do)."""
    return _notion(notion_tools.undo_last_entry)


# --- Google Calendar (see calendar_tools.py) ---
def _calendar(fn, *args):
    """Runs a calendar tool; any failure becomes a spoken sentence, not a crash."""
    if not calendar_tools.connected():
        return "Your calendar is not connected yet. Run the Google login script first."
    try:
        return fn(*args)
    except calendar_tools.CalendarError as e:
        print(f"[calendar] {fn.__name__} failed: {e}", file=sys.stderr)
        if "login expired" in str(e):
            return "Your Google login has expired. Run the login script again."
        return "I could not reach your calendar just now."


@mcp.tool()
def list_events(period: str = "today", until: str = "") -> str:
    """Lists calendar events. period: "today", "tomorrow", "this week", "next week", a weekday or a date. For a span give the first day as period and the last day as until, such as period "today", until "September 30"."""
    return _calendar(calendar_tools.list_events, period, until)


@mcp.tool()
def next_event() -> str:
    """Tells the next upcoming calendar event."""
    return _calendar(calendar_tools.next_event)


@mcp.tool()
def add_event(title: str, when: str, duration_minutes: str = "") -> str:
    """Adds an event to the calendar. when: day and time such as "tomorrow at 3pm" or "Friday 10am". duration_minutes optional, default 60."""
    return _calendar(calendar_tools.add_event, title, when, duration_minutes)


@mcp.tool()
def move_event(title: str, when: str) -> str:
    """Moves an existing calendar event to a new day and time."""
    return _calendar(calendar_tools.move_event, title, when)


@mcp.tool()
def cancel_event(title: str, when: str = "") -> str:
    """Cancels (deletes) a calendar event by its title; when is optional to disambiguate."""
    return _calendar(calendar_tools.cancel_event, title, when)


@mcp.tool()
def find_free_time(day: str = "today", minutes: str = "60") -> str:
    """Finds free gaps in the calendar between 9 AM and 6 PM on a day, long enough for the given minutes."""
    return _calendar(calendar_tools.find_free_time, day, minutes)


@mcp.tool()
def undo_calendar() -> str:
    """Undoes the last calendar change SKYE made: removes an added event, restores a moved or cancelled one."""
    return _calendar(calendar_tools.undo_calendar)


# --- macOS apps (see mac_tools.py) ---
def _mac(fn, *args):
    """Runs a Mac tool; failures become a spoken sentence."""
    try:
        return fn(*args)
    except mac_tools.MacError as e:
        print(f"[mac] {fn.__name__} failed: {e}", file=sys.stderr)
        msg = str(e)
        return ("I need permission first: " + msg.split("permission needed: ", 1)[1] + ".") if msg.startswith("permission needed") \
            else "I could not do that on the Mac just now."
    except Exception as e:
        print(f"[mac] {fn.__name__} crashed: {e}", file=sys.stderr)
        return "I could not do that on the Mac just now."


@mcp.tool()
def open_in_safari(target: str) -> str:
    """Opens a website in Safari. target: a site name ("youtube", "gmail"), an address, or search words."""
    return _mac(mac_tools.open_in_safari, target)


@mcp.tool()
def open_app(name: str) -> str:
    """Opens a Mac app: INDEX 0, Safari, Music, Notes, Reminders, Messages, Mail, Calendar, Clock, Notion, System Settings, FaceTime."""
    return _mac(mac_tools.open_app, name)


@mcp.tool()
def start_studying() -> str:
    """Opens INDEX 0 for studying and suggests what to study (from the to-do list and roadmap)."""
    return _mac(mac_tools.start_studying)


@mcp.tool()
def music_play(query: str) -> str:
    """Plays a song, artist or album from the Apple Music library (opens Apple Music search if it is not in the library)."""
    return _mac(mac_tools.music_play, query)


@mcp.tool()
def music_control(action: str) -> str:
    """Controls Apple Music: pause, resume, next, previous, or "volume 40"."""
    return _mac(mac_tools.music_control, action)


@mcp.tool()
def music_now_playing() -> str:
    """Says what is playing in Apple Music."""
    return _mac(mac_tools.music_now_playing)


@mcp.tool()
def list_mac_reminders() -> str:
    """Lists open items in the Apple Reminders app."""
    return _mac(mac_tools.list_mac_reminders)


@mcp.tool()
def complete_mac_reminder(title: str) -> str:
    """Marks an Apple Reminders item as done."""
    return _mac(mac_tools.complete_mac_reminder, title)


@mcp.tool()
def create_note(title: str, body: str = "") -> str:
    """Creates a note in the Apple Notes app."""
    return _mac(mac_tools.create_note, title, body)


@mcp.tool()
def add_to_note(title: str, text: str) -> str:
    """Appends text to an existing Apple Notes note."""
    return _mac(mac_tools.add_to_note, title, text)


@mcp.tool()
def find_notes(query: str) -> str:
    """Finds Apple Notes notes by title."""
    return _mac(mac_tools.find_notes, query)


@mcp.tool()
def read_apple_note(title: str) -> str:
    """Reads the start of an Apple Notes note aloud."""
    return _mac(mac_tools.read_note, title)


@mcp.tool()
def morning_briefing() -> str:
    """The day at a glance: greeting, weather, calendar events, to-dos and alarms."""
    now = datetime.now()
    greet = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 18 else "Good evening"
    parts = [f"{greet}."]
    blocks = []
    for label, fn in (
        ("weather", lambda: get_weather("")),
        ("calendar", lambda: _calendar(calendar_tools.list_events, "today")),
        ("todos", lambda: _notion(notion_tools.list_todos, "for today") if os.getenv("NOTION_TOKEN") else ""),
    ):
        try:
            out = fn()
        except Exception as e:
            print(f"[briefing] {label} failed: {e}", file=sys.stderr)
            continue
        out, payload = canvas.split(out or "")
        if payload:
            blocks.extend(payload.get("blocks", []))
        if out and "could not" not in out.lower() and "not connected" not in out.lower():
            parts.append(out if out.endswith((".", "!", "?")) else out + ".")
    end = datetime.now().replace(hour=23, minute=59)
    todays = [t for t in TASKS.get_upcoming(20) if datetime.fromisoformat(t["due_at"]) <= end]
    if todays:
        parts.append("Also today: " + "; ".join(f"{t['description']} at {datetime.fromisoformat(t['due_at']).strftime('%-I:%M %p')}" for t in todays[:4]) + ".")
    text = " ".join(parts)
    return canvas.attach(text, "Your day", *blocks) if blocks else text


@mcp.tool()
def event_alerts() -> str:
    """Internal: announcements for calendar events starting soon (empty when none)."""
    if not calendar_tools.connected():
        return ""
    try:
        return " ".join(calendar_tools.due_alerts(10))
    except Exception as e:
        print(f"[calendar] alerts failed: {e}", file=sys.stderr)
        return ""


# --- Messages (read only, see messages_tools.py) ---
def _messages(fn, *args):
    try:
        return fn(*args)
    except messages_tools.MessagesError as e:
        print(f"[messages] {fn.__name__} failed: {e}", file=sys.stderr)
        # "file is not a database" is the signature macOS leaves when Full
        # Disk Access is missing but the file handle still opens.
        return messages_tools.NEED_ACCESS if re.search(r"unable to open|not permitted|authorization|denied|not a database", str(e), re.I) \
            else "I could not read your messages just now."
    except Exception as e:
        print(f"[messages] {fn.__name__} crashed: {e}", file=sys.stderr)
        return "I could not read your messages just now."


@mcp.tool()
def read_messages(contact: str = "", limit: int = 4) -> str:
    """Reads recent iMessage/SMS messages (read only). With a contact name, the latest exchange with them; otherwise the newest incoming messages."""
    return _messages(messages_tools.read_messages, contact, limit)


@mcp.tool()
def unread_messages() -> str:
    """Says how many unread messages there are and from whom."""
    return _messages(messages_tools.unread_messages)


@mcp.tool()
def todo_nudge() -> str:
    """Internal: a short nudge about what is still open for today (empty when nothing)."""
    if not os.getenv("NOTION_TOKEN"):
        return ""
    try:
        items = [t for t in notion_tools._todos() if t["status"] in ("For today", "In Progress") or
                 (t["priority"] == "High" and t["status"] != "Done")]
    except Exception as e:
        print(f"[nudge] {e}", file=sys.stderr)
        return ""
    items = [t for t in items if t["status"] != "Done"]
    if not items:
        return ""
    n = len(items)
    names = ", ".join(t["name"] for t in items[:2])
    return f"You still have {n} thing{'s' if n != 1 else ''} open for today, such as {names}. Want to get started on one?"


@mcp.tool()
def show_media(url: str, title: str = "") -> str:
    """Shows an image, video (file or YouTube link) or audio file on the screen from a web address."""
    u = url.strip()
    if not re.match(r"^https?://", u, re.I):
        return "I need a full web address starting with http to show that."
    yt = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})", u)
    if yt:
        block = {"type": "video", "youtube": yt.group(1)}
    elif re.search(r"\.(?:mp4|webm|mov)(?:\?|$)", u, re.I):
        block = {"type": "video", "url": u}
    elif re.search(r"\.(?:mp3|wav|m4a|ogg)(?:\?|$)", u, re.I):
        block = {"type": "audio", "url": u}
    else:
        block = {"type": "image", "url": u, "caption": title}
    return canvas.attach(f"Showing {title or 'that'} on the screen.", title or "Media", block)


@mcp.tool()
def analyze_video(request: str, url: str = "") -> str:
    """Watches a YouTube video (the one open in Safari or Chrome if no link is given) with Gemini and does what the user asks: summarise it, explain a part, list steps. Sends the video link to Google."""
    from core import einstein
    u = (url or "").strip()
    if not u:
        try:
            u = mac_tools.front_browser_url()
        except Exception:
            u = ""
    m = re.search(r"(?:youtube\.com/watch\?[^\s]*v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})", u)
    if not m:
        return "I can only watch YouTube videos. Open one in Safari or give me the link."
    clean = f"https://www.youtube.com/watch?v={m.group(1)}"
    try:
        summary, detail, model = einstein.think(request or "Summarise this video.", "", video_url=clean)
    except RuntimeError as e:
        print(f"[video] failed: {e}", file=sys.stderr)
        return "I could not analyse that video just now. It may be private, too long or the service is busy."
    return canvas.attach(summary, "Video", {"type": "video", "youtube": m.group(1)}, {"type": "markdown", "text": detail})


if __name__ == "__main__":
    sys.stdout = _real_stdout
    mcp.run(transport="stdio")

