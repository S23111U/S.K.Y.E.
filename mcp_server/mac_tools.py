"""macOS integration: Safari, Apple Music, Reminders, Notes and app launching.

Everything goes through `osascript` (AppleScript) or `open`, so nothing extra is
installed and no model is loaded. The first use of each app makes macOS ask
"'Terminal' wants to control 'Music'"; approve it once (System Settings >
Privacy & Security > Automation). Until then a tool answers with a sentence
saying which permission is missing rather than failing silently.

Every function returns a sentence ready to be spoken.
"""

import os
import re
import subprocess
from datetime import datetime, timedelta
from urllib.parse import quote

TIMEOUT_S = 45   # the first call after an app launch, or a permission prompt, is slow
ROADMAP_DIR = os.path.expanduser(os.getenv("ROADMAP_DIR", "~/Desktop/Coding/Roadmap"))

# Only these can be launched by voice.
APPS = {
    "index 0": "INDEX 0", "index zero": "INDEX 0", "index": "INDEX 0",
    "safari": "Safari", "music": "Music", "apple music": "Music", "notes": "Notes",
    "reminders": "Reminders", "messages": "Messages", "mail": "Mail", "calendar": "Calendar",
    "clock": "Clock", "notion": "Notion", "system settings": "System Settings", "settings": "System Settings",
    "finder": "Finder", "facetime": "FaceTime", "maps": "Maps", "photos": "Photos",
}


class MacError(Exception):
    pass


def _osa(script, *args):
    try:
        r = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise MacError("timed out")
    if r.returncode != 0:
        err = r.stderr.strip()
        if "-1743" in err or "not allowed" in err.lower():
            app = re.search(r'application "?([A-Za-z ]+)"?', script)
            raise MacError(f"permission needed: allow control of {app.group(1) if app else 'the app'} in System Settings, Privacy and Security, Automation")
        raise MacError(err[:160])
    return r.stdout.strip()


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0:
        raise MacError(r.stderr.strip()[:160])


# --------------------------------------------------------------------------
# Safari and apps
# --------------------------------------------------------------------------
SITES = {"youtube": "https://youtube.com", "google": "https://google.com", "gmail": "https://mail.google.com",
         "github": "https://github.com", "notion": "https://notion.so", "wikipedia": "https://wikipedia.org",
         "reddit": "https://reddit.com", "netflix": "https://netflix.com", "amazon": "https://amazon.com.au",
         "linkedin": "https://linkedin.com", "chatgpt": "https://chatgpt.com", "maps": "https://maps.google.com"}


def open_in_safari(target):
    t = (target or "").strip().lower().rstrip(".")
    if not t:
        return "What should I open in Safari?"
    if t in SITES:
        url = SITES[t]
    elif re.match(r"^(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?$", t):
        url = t if t.startswith("http") else "https://" + t
    else:
        url = "https://www.google.com/search?q=" + quote(target)
    _run(["open", "-a", "Safari", url])
    return f"Opened {target.strip()} in Safari."


def open_app(name):
    key = re.sub(r"^(?:the\s+)", "", (name or "").strip().lower())
    app = APPS.get(key)
    if not app:
        return f"I am not set up to open {name}."
    if not os.path.exists(f"/Applications/{app}.app") and not os.path.exists(f"/System/Applications/{app}.app") \
            and not os.path.exists(f"/System/Applications/Utilities/{app}.app") and not os.path.exists(f"/System/Library/CoreServices/{app}.app"):
        return f"I could not find {app} on this Mac."
    _run(["open", "-a", app])
    return f"Opening {app}."


def start_studying():
    """Opens INDEX 0 and suggests what to study, from the to-do list, the notes
    index and the roadmap folder. It only reads names, never the app's content."""
    msg = open_app("index 0")
    ideas = []
    try:
        import notion_tools
        todos = [t for t in notion_tools._todos() if t["status"] != "Done"]
        study = [t["name"] for t in todos if re.search(
            r"\b(dsa|comp\d*|info\d*|assignment|tutorial|lecture|revis|study|review|week|exam|quiz|practice|leetcode)\b", t["name"], re.I)]
        ideas += study[:3]
    except Exception:
        pass
    road = []
    try:
        road = [re.sub(r"[_-]+", " ", f[:-5]).strip() for f in sorted(os.listdir(ROADMAP_DIR)) if f.endswith(".html")]
    except OSError:
        pass
    parts = [msg]
    if ideas:
        parts.append("From your to-do list, you could work on " + _join(ideas) + ".")
    if road:
        parts.append("Your roadmap folder also has " + _join(road[:3]) + ".")
    if not ideas and not road:
        parts.append("Tell me the topic and I can suggest where to start.")
    return " ".join(parts)


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Apple Music
# --------------------------------------------------------------------------
_PLAY = '''on run argv
  set q to item 1 of argv
  tell application "Music"
    set hits to (every track of library playlist 1 whose name contains q or artist contains q or album contains q)
    if (count of hits) is 0 then return "NONE"
    try
      delete (first user playlist whose name is "SKYE Queue")
    end try
    set pl to make new user playlist with properties {name:"SKYE Queue"}
    set n to 0
    repeat with t in hits
      duplicate t to pl
      set n to n + 1
      if n >= 40 then exit repeat
    end repeat
    set first_name to (name of item 1 of hits) & " by " & (artist of item 1 of hits)
    play pl
    return first_name
  end tell
end run'''


def music_play(query):
    q = (query or "").strip()
    if not q:
        return "What should I play?"
    # "Bohemian Rhapsody by Queen" was searched for as one literal string
    # against each track's name/artist/album, which no single field ever
    # contains verbatim, so a real match always came back NONE. Try the title
    # on its own first — the piece actually likely to appear in a track's
    # name — and only fall back to the raw phrase (which still works for
    # "some jazz"-style queries that were never a "song by artist" in the
    # first place).
    m = re.match(r"^(?P<title>.+?)\s+by\s+(?P<artist>.+)$", q, re.IGNORECASE)
    candidates = [m.group("title").strip(), q] if m else [q]
    out = "NONE"
    for c in candidates:
        out = _osa(_PLAY, c)
        if out != "NONE":
            break
    if out == "NONE":
        _run(["open", f"music://music.apple.com/search?term={quote(q)}"])
        return f"I could not find {q} in your library, so I opened Apple Music search. Pick it there and it will play."
    return f"Playing {out}."


def music_control(action):
    a = (action or "").lower().strip()
    cmd = {"pause": "pause", "stop": "pause", "resume": "play", "play": "play", "continue": "play",
           "next": "next track", "skip": "next track", "previous": "previous track", "back": "previous track"}.get(a)
    if a.startswith("volume"):
        n = re.search(r"\d+", a)
        if not n:
            return "What volume, from 0 to 100?"
        _osa(f'tell application "Music" to set sound volume to {min(100, int(n.group()))}')
        return f"Volume set to {min(100, int(n.group()))}."
    if not cmd:
        return f"I do not know how to {action} the music."
    _osa(f'tell application "Music" to {cmd}')
    return {"pause": "Paused.", "play": "Playing.", "next track": "Next track.", "previous track": "Previous track."}[cmd]


def music_now_playing():
    out = _osa('tell application "Music" to if player state is playing then return (name of current track) & " by " & (artist of current track)\nreturn "NOTHING"')
    return "Nothing is playing right now." if out == "NOTHING" else f"This is {out}."


# --------------------------------------------------------------------------
# Reminders (syncs to the iPhone through iCloud)
# --------------------------------------------------------------------------
def _as_date(var, dt):
    return (f"set {var} to current date\nset day of {var} to 1\nset year of {var} to {dt.year}\n"
            f"set month of {var} to {dt.month}\nset day of {var} to {dt.day}\n"
            f"set time of {var} to {dt.hour * 3600 + dt.minute * 60}")


def add_mac_reminder(title, when=""):
    from memory.tasks import parse_when
    title = (title or "").strip()
    if not title:
        return "What should the reminder say?"
    dt = parse_when(when) if when else None
    if when and dt is None:
        return "I did not catch when that should be."
    if dt:
        script = f'tell application "Reminders"\n{_as_date("d", dt)}\nmake new reminder with properties {{name:"{_esc(title)}", due date:d}}\nend tell'
    else:
        script = f'tell application "Reminders" to make new reminder with properties {{name:"{_esc(title)}"}}'
    _osa(script)
    return f"Added {title} to your Reminders" + (f" for {_when_words(dt)}." if dt else ".")


def list_mac_reminders():
    out = _osa('''tell application "Reminders"
  set out to {}
  repeat with r in (every reminder whose completed is false)
    set end of out to (name of r)
    if (count of out) >= 8 then exit repeat
  end repeat
  set AppleScript's text item delimiters to "|"
  return out as text
end tell''')
    if not out:
        return "Your Reminders app has nothing open."
    items = out.split("|")
    return f"In Reminders you have {_join(items)}."


def complete_mac_reminder(title):
    out = _osa(f'''tell application "Reminders"
  set hits to (every reminder whose completed is false and name contains "{_esc(title)}")
  if (count of hits) is 0 then return "NONE"
  set completed of (item 1 of hits) to true
  return name of (item 1 of hits)
end tell''')
    return f"I could not find a reminder called {title}." if out == "NONE" else f"Marked {out} as done in Reminders."


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
def _esc(s):
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def _html(s):
    # Notes.app already shows the "name:" property as the note's title (its
    # own first line) — an explicit <h1> here duplicated it verbatim ("milk"
    # and "eggs" showed up under the title repeated twice, confirmed live).
    # A short comma list ("milk, eggs, bread") reads far better as one item
    # per line than as one dense run-on line, so split on it; a longer,
    # sentence-like body (has its own punctuation) is left as whole lines.
    lines = []
    for line in (s or "").split("\n"):
        if line.count(",") >= 1 and not re.search(r"[.!?]", line) and len(line) < 200:
            lines.extend(p.strip() for p in line.split(",") if p.strip())
        elif line.strip():
            lines.append(line)
    return "".join(f"<div>{_esc(line) or '<br>'}</div>" for line in lines) or "<div><br></div>"


def create_note(title, body=""):
    title = (title or "").strip()
    if not title:
        return "What should the note be called?"
    _osa(f'tell application "Notes" to make new note with properties {{name:"{_esc(title)}", body:"{_html(body)}"}}')
    return f"Created a note called {title}."


def add_to_note(title, text):
    out = _osa(f'''tell application "Notes"
  set hits to (every note whose name contains "{_esc(title)}")
  if (count of hits) is 0 then return "NONE"
  set n to item 1 of hits
  set body of n to (body of n) & "{_html(text)}"
  return name of n
end tell''')
    return f"I could not find a note called {title}." if out == "NONE" else f"Added that to your note {out}."


def find_notes(query):
    out = _osa(f'''tell application "Notes"
  set names to name of (every note whose name contains "{_esc(query)}")
  if (count of names) is 0 then return ""
  set AppleScript's text item delimiters to "|"
  set k to count of names
  if k > 6 then set k to 6
  return (items 1 thru k of names) as text
end tell''')
    if not out:
        return f"I found no notes matching {query}."
    items = out.split("|")
    return f"I found {len(items)} note{'s' if len(items) != 1 else ''}: {_join(items)}."


def read_note(title):
    out = _osa(f'''tell application "Notes"
  set hits to (every note whose name contains "{_esc(title)}")
  if (count of hits) is 0 then return "NONE"
  return plaintext of item 1 of hits
end tell''')
    if out == "NONE":
        return f"I could not find a note called {title}."
    return re.sub(r"\s+", " ", out)[:700]


def _when_words(dt):
    today = datetime.now().date()
    day = "today" if dt.date() == today else "tomorrow" if dt.date() == today + timedelta(days=1) else dt.strftime("%A")
    return f"{day} at {dt.strftime('%-I:%M %p').replace(':00 ', ' ')}"


def front_browser_url():
    """The address of the tab in front, in Safari or Chrome, whichever is running."""
    for app, expr in (("Safari", "URL of front document"), ("Google Chrome", "URL of active tab of front window")):
        try:
            running = _osa(f'tell application "System Events" to return (name of processes) contains "{app}"')
            if running == "true":
                url = _osa(f'tell application "{app}" to return {expr}')
                if url:
                    return url
        except MacError:
            continue
    return ""
