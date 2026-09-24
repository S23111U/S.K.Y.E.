"""SKYE taking the initiative: check-ins, task follow-ups and to-do nudges.

Rules, not the model, decide *when* to speak, so it is predictable and free.
Everything that could make it annoying is a setting with a conservative default:

  - never during quiet hours (default 22:30-07:00) or while asleep/muted
  - never within a few minutes of the last thing he said (mid-conversation)
  - at least MIN_GAP_MIN between two unprompted messages, and MAX_PER_DAY a day
  - "leave me alone" / "be quiet for two hours" snoozes it by voice

What it does:
  - task follow-up: "I'm going to work on X" is noted; a while later, "How is X going?"
  - to-do nudge: midday and late afternoon, a reminder of what is still open for today
  - catch-up: if it has been a long quiet stretch in the day, a friendly check-in
  - morning: the first time he is around before noon, offers the briefing
  - routine log: first/last time SKYE was used each day (memory/routine.json), the raw
    material for learning when he wakes and sleeps

State lives in memory/proactive_state.json so a restart does not repeat itself.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(ROOT, "memory", "proactive_state.json")
ROUTINE_FILE = os.path.join(ROOT, "memory", "routine.json")

ENABLED = os.getenv("SKYE_PROACTIVE", "on").lower() != "off"
QUIET_START = os.getenv("SKYE_QUIET_START", "22:30")
QUIET_END = os.getenv("SKYE_QUIET_END", "07:00")
MIN_GAP_MIN = int(os.getenv("SKYE_PROACTIVE_GAP_MIN", "60"))
MAX_PER_DAY = int(os.getenv("SKYE_PROACTIVE_MAX_PER_DAY", "6"))
IDLE_BEFORE_SPEAKING_S = 240          # not within 4 minutes of his last message
FOLLOWUP_AFTER_MIN = int(os.getenv("SKYE_FOLLOWUP_MIN", "45"))
CATCHUP_IDLE_H = 3
TODO_NUDGE_HOURS = (12, 16)           # earliest hour for the first and second nudge

I = re.IGNORECASE
STARTING_RE = re.compile(
    r"\b(?:i(?:'m| am)?\s*(?:going to|gonna|about to|will|'ll|am going to)|let me|time to|i(?:'m| am) (?:starting|beginning)|i(?:'ll| will) start|i'm now)\s+"
    r"(?:start\s+|begin\s+|be\s+)?(?P<act>(?:work(?:ing)? on|study(?:ing)?|do(?:ing)?|finish(?:ing)?|revis(?:e|ing)|practi[sc](?:e|ing)|read(?:ing)?|write|writing|code|coding|clean(?:ing)?|cook(?:ing)?|exercise|work(?:ing)? out|go for a|hit the)\b.{2,60}?)\s*(?:[.!?,]|\band\b|\bso\b|$)", I)
DONE_RE = re.compile(r"\b(?:i(?:'m| am| have|'ve)?\s*(?:done|finished|completed)|(?:it's|that's|all) (?:done|finished)|just finished|wrapped up)\b", I)
LEAVE_ME_RE = re.compile(r"\b(?:stop|quit|no more)\s+(?:checking (?:in|up)|nudging|bugging|pinging)\b|\bleave me alone\b|\bdon'?t (?:check|nudge|bother|interrupt)\b|\bdo not disturb\b|\bturn off (?:check-?ins?|nudges?|reminders? to check)\b", I)
RESUME_RE = re.compile(r"\b(?:turn|switch) on (?:check-?ins?|nudges?)\b|\bcheck in on me again\b|\byou can (?:check in|nudge) (?:on me )?again\b", I)
FOR_RE = re.compile(r"\bfor\s+(?:the next\s+)?(?P<n>\d+|an?|one|two|three|four|five|six)\s*(?P<u>hours?|hrs?|minutes?|mins?)\b", I)
_GERUND = {"work": "working", "study": "studying", "do": "doing", "finish": "finishing", "revise": "revising", "practice": "practising",
           "practise": "practising", "read": "reading", "write": "writing", "code": "coding", "clean": "cleaning", "cook": "cooking", "exercise": "exercising", "go": "going", "hit": "hitting"}
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _load(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w"))


def _minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Proactive:
    def __init__(self):
        s = _load(STATE_FILE, {})
        self.day = s.get("day")
        self.sent_today = s.get("sent_today", 0)
        self.last_sent = s.get("last_sent", 0)
        self.done_today = set(s.get("done_today", []))      # which one-per-day nudges already fired
        self.snooze_until = s.get("snooze_until", 0)
        self.activity = s.get("activity")                    # {"text","started","followups"}
        self.last_input = time.time()
        self.asleep = False
        self.first_input_today = False

    # ---------- persistence ----------
    def _persist(self):
        _save(STATE_FILE, {"day": self.day, "sent_today": self.sent_today, "last_sent": self.last_sent,
                           "done_today": sorted(self.done_today), "snooze_until": self.snooze_until, "activity": self.activity})

    def _roll_day(self, now):
        today = now.strftime("%Y-%m-%d")
        if self.day != today:
            self.day, self.sent_today, self.done_today = today, 0, set()
            self.first_input_today = False
            self._persist()

    # ---------- what he says ----------
    def observe(self, text: str):
        """Called for every message he sends. Returns a spoken reply for the
        snooze/resume commands (handled by the caller), else None."""
        now = datetime.now()
        self._roll_day(now)
        self.last_input = time.time()
        self.first_input_today = True
        self._log_routine(now)
        t = text or ""
        if RESUME_RE.search(t):
            self.snooze_until = 0
            self._persist()
            return "Okay, I will check in on you now and then."
        quiet_for = re.search(r"\b(?:be|stay) quiet\b", t, I) and FOR_RE.search(t)   # a bare "be quiet" just means stop talking
        if LEAVE_ME_RE.search(t) or quiet_for:
            m = FOR_RE.search(t)
            if m:
                n = _WORDS.get(m.group("n").lower()) or int(m.group("n"))
                mins = n * (60 if m.group("u").lower().startswith("h") else 1)
                self.snooze_until = time.time() + mins * 60
                self._persist()
                return f"Understood, I will stay quiet for {n} {'hour' if m.group('u').lower().startswith('h') else 'minute'}{'s' if n != 1 else ''}."
            self.snooze_until = time.time() + 12 * 3600
            self._persist()
            return "Understood. I will leave you alone until you tell me otherwise or tomorrow."
        if DONE_RE.search(t) and self.activity:
            self.activity = None
            self._persist()
            return None
        m = STARTING_RE.search(t)
        if m and not self.activity:
            act = re.sub(r"\s+", " ", m.group("act")).strip(" .,")
            act = re.sub(r"\s+(?:now|today|right now|tonight)$", "", act, flags=I)
            first, _, rest = act.partition(" ")
            act = (_GERUND.get(first.lower(), first) + " " + rest).strip()
            act = re.sub(r"\bmy\b", "your", act, flags=I)
            act = re.sub(r"\bmyself\b", "yourself", act, flags=I)
            self.activity = {"text": act, "started": time.time(), "followups": 0}
            self._persist()
        return None

    def _log_routine(self, now):
        r = _load(ROUTINE_FILE, {})
        d = now.strftime("%Y-%m-%d")
        rec = r.get(d) or {"first": now.strftime("%H:%M"), "last": now.strftime("%H:%M")}
        rec["last"] = now.strftime("%H:%M")
        r[d] = rec
        for k in sorted(r)[:-120]:      # keep about four months
            del r[k]
        _save(ROUTINE_FILE, r)

    # ---------- when SKYE speaks ----------
    def _quiet_now(self, now):
        m = now.hour * 60 + now.minute
        a, b = _minutes(QUIET_START), _minutes(QUIET_END)
        return (m >= a or m < b) if a > b else (a <= m < b)

    def _allowed(self, now, connected):
        if not ENABLED or not connected or self.asleep:
            return False
        if time.time() < self.snooze_until or self._quiet_now(now):
            return False
        if time.time() - self.last_input < IDLE_BEFORE_SPEAKING_S:
            return False
        if self.sent_today >= MAX_PER_DAY or time.time() - self.last_sent < MIN_GAP_MIN * 60:
            return False
        return True

    def _spoke(self, key=None):
        self.sent_today += 1
        self.last_sent = time.time()
        if key:
            self.done_today.add(key)
        self._persist()

    def tick(self, connected: bool, todo_nudge=None, briefing_offer=None):
        """Returns text to say now, or None. `todo_nudge()` returns a sentence or "".
        A returned tuple (text, pending) means the reply should arm a yes/no action."""
        now = datetime.now()
        self._roll_day(now)
        if not self._allowed(now, connected):
            return None

        # 1. follow up on what he said he was starting
        a = self.activity
        if a and a["followups"] < 2 and time.time() - a["started"] > (FOLLOWUP_AFTER_MIN + a["followups"] * 60) * 60:
            a["followups"] += 1
            self._spoke()
            return (f"How are you getting on with {a['text']}?" if a["followups"] == 1
                    else f"Still {a['text']}? Tell me if you want help getting through it.")

        # 2. the first time he is around in the morning: offer the briefing
        if briefing_offer and "briefing" not in self.done_today and 7 <= now.hour < 11 and not self.first_input_today:
            self._spoke("briefing")
            return briefing_offer

        # 3. to-do nudges, midday and late afternoon
        for i, hour in enumerate(TODO_NUDGE_HOURS):
            key = f"todo{i}"
            if now.hour >= hour and key not in self.done_today and now.hour < 20 and todo_nudge:
                self.done_today.add(key)          # even if there is nothing to say: do not re-check all afternoon
                try:
                    text = todo_nudge()
                except Exception:
                    text = ""
                if text:
                    self._spoke()
                    return text
                self._persist()
                return None

        # 4. a friendly check-in after a long quiet stretch
        if "catchup" not in self.done_today and 10 <= now.hour < 20 and time.time() - self.last_input > CATCHUP_IDLE_H * 3600:
            self._spoke("catchup")
            return "Just checking in. How is your day going?"
        return None
