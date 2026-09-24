"""The "mac" skill: everything that talks to a native app on the Mac —
Safari, Apple Music, Notes, Reminders, launching other apps, and reading
iMessage/SMS (read-only). One skill because they're all reached through the
same mechanism (AppleScript via osascript) and tend to come up in the same
kind of request ("add this to my note", "what's playing", "any new texts").
"""

import re

from skills.base import Skill
from skills.mac.apps import (
    add_to_note, complete_mac_reminder, create_note, find_notes, list_mac_reminders,
    music_control, music_now_playing, music_play, open_app, open_in_safari,
    read_apple_note, start_studying,
)
from skills.mac.messages import read_messages, unread_messages
from skills.routing_helpers import END, I, LEAD

_APPS = ("index 0", "index zero", "index", "safari", "music", "apple music", "notes", "reminders", "messages",
         "mail", "calendar", "clock", "notion", "system settings", "settings", "finder", "facetime", "maps", "photos")
_SITES = ("youtube", "google", "gmail", "github", "notion", "wikipedia", "reddit", "netflix", "amazon", "linkedin", "chatgpt")

SKILL = Skill(
    name="mac",
    weight=2,   # its phrases are specific ("add ... to my note"); beat a stray finance/todo word
    utterances=[
        "play some music on apple music", "add a note to my apple notes", "create a note about the meeting",
        "open the reminders app", "open safari and search for something", "what reminders do I have in the reminders app",
        "do I have any new messages", "what did someone text me",
    ],
    pattern=re.compile(
        r"\b(?:apple music|apple notes|reminders app|safari|(?:the |my )?music|(?:a |an )?(?:new )?note(?: called| about| on)|"
        r"(?:add|append) .{0,40} to (?:my |the )?note|messages?|texts?|imessages?)\b",
        re.IGNORECASE,
    ),
    manifest=(
        "Mac tools (his MacBook apps): music_play{\"query\"} · music_control{\"action\"} · music_now_playing{} · "
        "open_in_safari{\"target\"} · open_app{\"name\"} · create_note{\"title\",\"body\"} · add_to_note{\"title\",\"text\"} · "
        "find_notes{\"query\"} · read_apple_note{\"title\"} · list_mac_reminders{} · complete_mac_reminder{\"title\"} · "
        "read_messages{\"contact\",\"limit\"} · unread_messages{}. "
        "music_control action is pause, resume, next, previous or \"volume 40\". Messages/texts means his iMessage/SMS — "
        "read_messages with no contact gives the newest ones, with a contact name gives that exchange. "
        "Never say something was done unless you called the tool for it."
    ),
    examples=[
        ("add eggs to my groceries note", '{"name": "add_to_note", "arguments": {"title": "groceries", "text": "eggs"}}'),
        ("open the news in safari", '{"name": "open_in_safari", "arguments": {"target": "news"}}'),
        ("do I have any new messages?", '{"name": "unread_messages", "arguments": {}}'),
    ],
    tools=[
        open_in_safari, open_app, start_studying,
        music_play, music_control, music_now_playing,
        list_mac_reminders, complete_mac_reminder,
        create_note, add_to_note, find_notes, read_apple_note,
        read_messages, unread_messages,
    ],
    direct_routes=[
        # "I want to study on Index 0" (Whisper may hear "index zero" / "index O")
        ("start_studying", re.compile(
            r"\b(?:study|studying|learn|learning|revise|revising)\b.{0,40}\bindex(?:\s+(?:0|zero|o))?\b|"
            r"\bindex(?:\s+(?:0|zero|o))?\b.{0,25}\b(?:study|learn|revise)", I), lambda m: {}),

        ("unread_messages", re.compile(
            r"\b(?:unread|new|missed) (?:messages?|texts?|imessages?)\b|\bany (?:new )?(?:messages?|texts?)\b|"
            r"\bdo i have (?:any )?(?:new )?(?:messages?|texts?)\b", I), lambda m: {}),
        ("read_messages", re.compile(
            r"\bwhat did (?P<c>[\w' .-]+?) (?:text|message|say to) me\b|\b(?:messages?|texts?|imessages?) from (?P<c2>[\w' .-]+?)" + END, I),
         lambda m: {"contact": (m.group("c") or m.group("c2") or "").strip()}),
        ("read_messages", re.compile(
            r"^(?!.*\b(?:send|write|reply|draft|compose)\b).*\b(?:read|check|show|open)\b.{0,20}\b(?:my |the )?(?:latest |last |recent |newest )?(?:messages?|texts?|imessages?)" + END, I),
         lambda m: {}),

        ("music_now_playing", re.compile(r"\b(?:what(?:'s| is) (?:playing|this song)|what song is (?:this|playing)|which song is this)\b", I), lambda m: {}),
        ("music_control", re.compile(
            r"^\W*(?:please\s+)?(?:(?P<a>pause|resume|skip)\b(?:\s+(?:the\s+)?(?:music|song|track|this song|it))?|"
            r"(?P<a2>next|previous|stop)\s+(?:the\s+)?(?:music|song|track)|go back(?: a song)?|(?:play )?(?:the )?next (?:song|track)|"
            r"(?:play )?(?:the )?previous (?:song|track)|(?:set )?(?:the )?volume (?:to |at )?(?P<v>\d+))" + END, I),
         lambda m: {"action": (f"volume {m.group('v')}" if m.group("v") else
                    (m.group("a") or m.group("a2") or ("previous" if "back" in m.group(0).lower() or "previous" in m.group(0).lower() else "next")).lower())}),
        ("music_play", re.compile(
            r"^\W*(?:please\s+)?play\s+(?!.*\b(?:youtube|spotify|netflix|game|video)\b)(?:some\s+)?(?P<q>.+?)(?:\s+(?:on|in|using|with)\s+apple music)?" + END, I),
         lambda m: {"query": re.sub(r"^(?:something by|songs? by|music by|the song|the album|the artist)\s+", "", m.group("q"), flags=I)}),

        ("open_in_safari", re.compile(r"^\W*" + LEAD + r"(?:open|go to|launch|show me|take me to)\s+(?P<t>.+?)\s+in safari" + END, I), lambda m: {"target": m.group("t")}),
        ("open_in_safari", re.compile(r"^\W*" + LEAD + r"open safari (?:and )?search(?:\s+for)?\s+(?P<t>.+?)" + END, I), lambda m: {"target": m.group("t")}),
        ("open_in_safari", re.compile(r"^\W*" + LEAD + r"search(?:\s+for)?\s+(?P<t>.+?)\s+(?:in|on)\s+safari" + END, I), lambda m: {"target": m.group("t")}),
        ("open_in_safari", re.compile(r"^\W*" + LEAD + r"search\s+safari\s+for\s+(?P<t>.+?)" + END, I), lambda m: {"target": m.group("t")}),
        ("open_app", re.compile(r"^\W*" + LEAD + r"(?:open|launch|start|switch to)\s+(?:the\s+)?(?P<a>" + "|".join(re.escape(a) for a in _APPS) + r")(?:\s+app)?" + END, I), lambda m: {"name": m.group("a")}),
        ("open_in_safari", re.compile(r"^\W*" + LEAD + r"(?:open|go to)\s+(?P<t>" + "|".join(_SITES) + r"|[\w-]+\.(?:com|org|net|io|edu|au|co)(?:/\S*)?)" + END, I), lambda m: {"target": m.group("t")}),

        ("add_to_note", re.compile(
            r"\b(?:add|put|write)\s+(?P<b>.+?)\s+(?:to|in|into)\s+(?:my|the|that)\s+(?:same\s+)?(?P<t>[\w' -]+?)\s+note\b|"
            r"\bin\s+(?:the|my|that)\s+(?:same\s+)?(?P<t2>[\w' -]+?)\s+note,?\s+add\s+(?P<b2>.+?)" + END + r"|"
            r"\bin\s+(?:the|my|that)\s+(?:same\s+)?note\s+(?P<t3>[\w' -]+?),?\s+add\s+(?P<b3>.+?)" + END, I),
         lambda m: {"title": (m.group("t") or m.group("t2") or m.group("t3") or "").strip(),
                    "text": (m.group("b") or m.group("b2") or m.group("b3") or "").strip()}),
        ("create_note", re.compile(
            r"\b(?:create|make|start|write|take)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?note\s*(?:called|named|titled|about|on)?\s*[:,]?\s*(?P<t>.+?)"
            r"(?:\s*(?:,\s*)?(?:saying|that says|with the text|and write|and add|containing|with|:)\s+(?P<b>.+?)(?:\s+to (?:it|the note))?)?" + END, I),
         lambda m: {"title": re.sub(r"^[\s,.:;-]+|[\s,.:;-]+$", "", m.group("t")), "body": (m.group("b") or "").strip()}),
    ],
)
