import sys
from datetime import datetime

from memory.tasks import parse_when, parse_duration
from skills.shared import get_tasks


def _when_words(dt):
    from skills.mac.apps import when_words
    return when_words(dt)


def set_alarm(time: str) -> str:
    """Sets an alarm for a clock time such as "7:30 AM", "7am", "6:30 tomorrow"."""
    due = parse_when(time, alarm=True)
    if due is None:
        return "I did not catch that time. Could you say it like 7 30 AM?"
    get_tasks().add_task(f"Alarm: {due.strftime('%-I:%M %p')}", due, source="explicit")
    return f"Alarm set for {_when_words(due)}."


def set_timer(duration: str, label: str = "") -> str:
    """Starts a countdown timer. duration: "10 minutes", "an hour and a half", "90 seconds"."""
    delta = parse_duration(duration)
    if delta is None:
        return "I did not catch how long the timer should be."
    get_tasks().add_task(f"Timer: {label.strip()}" if label.strip() else "Timer", datetime.now() + delta, source="explicit")
    total = int(delta.total_seconds())
    h, m, sec = total // 3600, (total % 3600) // 60, total % 60
    words = " ".join(f"{n} {u}{'s' if n != 1 else ''}" for n, u in ((h, "hour"), (m, "minute"), (sec, "second")) if n)
    return f"Timer set for {words}."


def change_alarm(which: str, time: str) -> str:
    """Moves an existing alarm, timer or reminder (matched by a word from its name, or "alarm") to a new time."""
    due = parse_when(time, alarm="alarm" in which.lower() or not which.strip())
    if due is None:
        return "I did not catch the new time."
    matches = get_tasks().find_pending(which if which.lower() not in ("", "alarm", "the alarm", "my alarm") else "Alarm")
    if not matches:
        return f"I could not find {which or 'an alarm'} to change."
    t = matches[0]
    tasks = get_tasks()
    tasks.set_due(t["id"], due)
    if t["description"].startswith("Alarm"):
        with_time = f"Alarm: {due.strftime('%-I:%M %p')}"
        tasks.conn.execute("UPDATE tasks SET description = ? WHERE id = ?", (with_time, t["id"]))
        tasks.conn.commit()
    return f"Moved it to {_when_words(due)}."


def cancel_alarm(which: str = "") -> str:
    """Cancels an alarm, timer or reminder matched by a word from its name. With no name, the next alarm."""
    matches = get_tasks().find_pending(which if which.lower() not in ("", "alarm", "the alarm", "my alarm", "timer", "the timer") else
                                 ("Timer" if "timer" in which.lower() else "Alarm"))
    if not matches:
        return f"I could not find {which or 'an alarm'} to cancel."
    get_tasks().mark_status(matches[0]["id"], "dismissed")
    return f"Cancelled {matches[0]['description']}."


def set_reminder(time: str, task: str) -> str:
    """Sets a reminder for a time ("7pm", "tomorrow at 8", "in 20 minutes") with what to remember. SKYE speaks it when due and it is added to the Reminders app."""
    due = parse_when(time)
    if due is None:
        return "I did not catch when. Could you say the time again?"
    get_tasks().add_task(task, due, source="explicit")
    try:
        from skills.mac.apps import add_mac_reminder
        add_mac_reminder(task, time)
    except Exception as e:
        print(f"[mac] Reminders sync skipped: {e}", file=sys.stderr)
    return f"Reminder set: {task}, {_when_words(due)}."
