"""Generic screen/video utilities — not tied to any one domain, so they live
here rather than under a specific skill. analyze_video shares its Gemini
backend with Einstein mode (core/einstein.py) but is reachable on its own."""

import re
import sys

from skills import canvas


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


def analyze_video(request: str, url: str = "") -> str:
    """Watches a YouTube video (the one open in Safari or Chrome if no link is given) with Gemini and does what the user asks: summarise it, explain a part, list steps. Sends the video link to Google."""
    from core import einstein
    u = (url or "").strip()
    if not u:
        try:
            from skills.mac.apps import front_browser_url
            u = front_browser_url()
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
