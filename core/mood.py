"""Works out how SKYE should *sound*, from how the user *feels*.

The user's message is classified by a small dedicated emotion model
(j-hartmann/emotion-english-distilroberta-base, ~8 ms on CPU) and mapped to one
of four delivery moods:

  happy      user is pleased / grateful      -> brighter, a little faster
  sad        user has bad news / is upset     -> softer, slower, lower
  concerned  user is frustrated, confused,    -> gentle and unhurried
             or SKYE is apologising
  calm       everything else (the default)

An earlier version compared the *reply text* to example sentences; it read
ordinary news lists as sad and capability lists as happy. The user's own words
are the reliable signal, so the reply only contributes an apology check.

Chatterbox Turbo has no emotion control (its `exaggeration` input is ignored)
and sampling temperature alone was inaudible, so each mood also carries `rate`
(playback-rate change: pitch and pace together, as in real speech) and `gain`,
applied in tts_server. A recording of the user in that tone, saved as
assets/reference_voice_<mood>.wav (happy / sad / concerned), is used for the
voice conditioning when present and is the strongest lever of all.
"""

import re

MOOD_PARAMS = {
    "calm":      dict(temperature=0.70, top_p=0.95, repetition_penalty=1.20, rate=1.00, gain=1.00),
    "happy":     dict(temperature=0.80, top_p=0.98, repetition_penalty=1.15, rate=1.10, gain=1.00, voice="happy"),
    "sad":       dict(temperature=0.60, top_p=0.90, repetition_penalty=1.25, rate=0.88, gain=0.80, voice="sad"),
    "concerned": dict(temperature=0.65, top_p=0.92, repetition_penalty=1.22, rate=0.94, gain=0.90, voice="concerned"),
}

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Emotion labels need to clear these confidence levels to change the mood.
JOY_MIN, SAD_MIN, UPSET_MIN = 0.45, 0.50, 0.55
# Questions and commands are usually neutral even when the classifier wobbles
# ("set an alarm for 7 AM" scored fear 0.50), so they need much more evidence.
INSTRUCTION_MIN = 0.80
INSTRUCTION_RE = re.compile(
    r"^\s*(?:hey\s+)?(?:(?:skye|sky)[\s,]*)?(?:please\s+)?"
    r"(?:set|open|play|search|look|find|tell|show|list|remind|what|who|when|where|why|how|which|"
    r"can|could|would|do|does|did|is|are|will|should|mention|explain|give)\b",
    re.IGNORECASE,
)
CONFUSED_RE = re.compile(
    r"\b(?:(?:don't|do not|can't|cannot|couldn't|still)\s+(?:understand|get|follow|figure)|"
    r"confus(?:ed|ing)|no idea|lost me|makes no sense)\b",
    re.IGNORECASE,
)
APOLOGY_RE = re.compile(
    r"^\s*(?:my apologies|apologies|i apologi[sz]e|i(?:'m| am) sorry|i(?:'m| am) afraid|unfortunately)",
    re.IGNORECASE,
)


class MoodClassifier:
    def __init__(self):
        # Imported here so the module can be inspected without loading torch.
        from transformers import pipeline

        self._clf = pipeline(
            "text-classification", model=EMOTION_MODEL, top_k=None, device="cpu"
        )

    def emotions(self, text: str) -> dict:
        return {r["label"]: float(r["score"]) for r in self._clf(text[:512])[0]}

    def user_mood(self, text: str):
        """(mood, top_emotion_label, score) for the user's message."""
        scores = self.emotions(text)
        top = max(scores, key=scores.get)
        joy, sad = scores.get("joy", 0), scores.get("sadness", 0)
        upset = max(scores.get(k, 0) for k in ("anger", "fear", "disgust"))
        sur = scores.get("surprise", 0)
        instruction = bool(INSTRUCTION_RE.match(text))
        strict = INSTRUCTION_MIN if instruction else None

        if joy >= (strict or JOY_MIN) or (
            sur >= 0.70 and joy >= 0.08 and text.rstrip().endswith("!") and not instruction
        ):
            mood = "happy"
        elif sad >= (strict or SAD_MIN):
            mood = "sad"
        elif upset >= (strict or UPSET_MIN):
            mood = "concerned"
        elif CONFUSED_RE.search(text):
            mood = "concerned"
        else:
            mood = "calm"
        return mood, top, round(scores[top], 2)

    @staticmethod
    def final_mood(user_mood: str, reply: str) -> str:
        """The user's mood decides, except that apologising is always gentle."""
        if user_mood == "calm" and APOLOGY_RE.match(reply or ""):
            return "concerned"
        return user_mood
