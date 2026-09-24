"""Live web search: Tavily first, a keyless DuckDuckGo+Scrapling fallback,
a short-lived cache, and a marker (SOURCES_MARKER) that tells core_v2's
engine to run a grounded-synthesis pass over the results rather than
speaking them verbatim.
"""

import os
import re
import sys
import time
from email.utils import parsedate_to_datetime
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import requests

from skills import canvas
from skills.shared import get_memory

# Prefix telling core_v2.py a tool result is raw material to synthesise from,
# not a finished reply — shared with fetch_news, which returns the same shape.
SOURCES_MARKER = "[WEB SOURCES]"

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
_cache_ready = False


def _ensure_cache_table():
    global _cache_ready
    if _cache_ready:
        return
    get_memory().conn.execute(
        "CREATE TABLE IF NOT EXISTS web_cache ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, "
        "embedding BLOB NOT NULL, result TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    get_memory().conn.commit()
    _cache_ready = True


def _cache_lookup(query: str, ttl: int):
    _ensure_cache_table()
    memory = get_memory()
    q = memory.model.encode(query).astype(np.float32)
    rows = memory.conn.execute(
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
    _ensure_cache_table()
    memory = get_memory()
    emb = memory.model.encode(query).astype(np.float32).tobytes()
    memory.conn.execute(
        "INSERT INTO web_cache (query, embedding, result, created_at) VALUES (?, ?, ?, ?)",
        (query, emb, result, time.time()),
    )
    memory.conn.commit()


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
    from tavily import TavilyClient
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
    memory = get_memory()
    stamp = datetime.now().strftime("%Y-%m-%d")
    lines = [f"{SOURCES_MARKER} Today is {stamp}."]
    for i, (title, url, published, snippet) in enumerate(sources[:5], 1):
        dated = f", published {published}" if published else ""
        lines.append(f"[{i}] {title} ({urlparse(url).netloc}{dated}): {snippet}")
        # RAG pipeline: kept in long-term memory, but see MemoryManager.search()
        # — web rows are not passively injected into ordinary chat.
        memory.add_memory(f"[Web, {stamp}] {title} ({url}): {snippet}")
    result = "\n".join(lines)
    result = canvas.attach(result, "Sources", canvas.links([(t, u, urlparse(u).netloc) for t, u, _, _ in sources[:5]]))
    _cache_store(query, result)
    return result


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
        lines = [f"{SOURCES_MARKER} Today is {datetime.now():%Y-%m-%d}."]
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
