"""Opens a site in the default browser and speaks a short acknowledgement —
used for things that don't need a dedicated app (YouTube/Spotify search
results, the Google Calendar website)."""

import webbrowser

_SITES = {
    "youtube": "https://youtube.com",
    "wikipedia": "https://wikipedia.com",
    "google": "https://google.com",
    "spotify": "https://open.spotify.com",
}


def _open_and_ack(site_key, query=None):
    url = _SITES.get(site_key, "https://google.com")
    if query:
        if site_key == "youtube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"Opening YouTube for {query}."
        if site_key == "spotify":
            webbrowser.open(f"https://open.spotify.com/search/{query}")
            return f"Opening Spotify for {query}."
    webbrowser.open(url)
    return f"Opening {site_key}."


def open_youtube(query: str) -> str:
    """Opens YouTube search results for a query in the default browser."""
    return _open_and_ack("youtube", query)


def open_spotify(query: str) -> str:
    """Opens Spotify search results for a query in the default browser."""
    return _open_and_ack("spotify", query)


def open_calendar_web() -> str:
    """Opens the Google Calendar website in the default browser."""
    return _open_and_ack("google", "calendar")
