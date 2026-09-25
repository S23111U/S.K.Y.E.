"""FaceTime and phone calling.

Part of the "mac" skill (see skills/mac/__init__.py). A call is placed the
same way clicking a phone number in Contacts or Safari places one: by opening
a facetime:// / facetime-audio:// / tel: URL and letting macOS take it from
there. A phone number routes through Continuity Calling (relayed via a paired,
nearby iPhone) when that's set up; otherwise it falls back to FaceTime audio,
which only works if the other side also has FaceTime. Contact name -> number
resolution reuses skills/mac/messages.py's Contacts lookup rather than
duplicating it.
"""

import re
from urllib.parse import quote

from skills.mac.apps import _mac_safe, _run
from skills.mac.messages import _handles_for


def _best_handle(contact):
    """The contact's best number/email to call, and which kind it is."""
    handles = _handles_for((contact or "").strip())
    if not handles:
        return None, None
    phones = [h for h in handles if "@" not in h and re.search(r"\d{3}", h)]
    emails = [h for h in handles if "@" in h]
    if phones:
        return phones[0], "phone"
    if emails:
        return emails[0], "email"
    return None, None


@_mac_safe
def facetime_call(contact, audio_only=""):
    """Starts a FaceTime call to a contact by name, phone number, or email.
    Video by default; pass audio_only="yes" for a FaceTime audio call."""

    contact = (contact or "").strip()
    if not contact:
        return "Who should I FaceTime?"
    handle, kind = _best_handle(contact)
    if not handle:
        return f"I could not find a phone number or email for {contact} in your Contacts."
    audio = str(audio_only).strip().lower() in ("yes", "true", "audio", "1")
    _run(["open", f"{'facetime-audio' if audio else 'facetime'}://{quote(handle)}"])
    return f"Starting a FaceTime {'audio ' if audio else ''}call with {contact}."


@_mac_safe
def phone_call(contact):
    """Places a phone call to a contact by name or number, using Continuity
    Calling through a nearby paired iPhone if one's set up, else FaceTime audio."""

    contact = (contact or "").strip()
    if not contact:
        return "Who should I call?"
    handle, kind = _best_handle(contact)
    if not handle:
        return f"I could not find a phone number for {contact} in your Contacts."
    _run(["open", f"{'tel' if kind == 'phone' else 'facetime-audio'}://{quote(handle)}"])
    return f"Calling {contact}."
