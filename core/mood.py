"""Chooses how SKYE should *sound* for a given exchange.

Replaces the old random per-sentence speed/temperature jitter, which had no
relationship to what was being said (good news could come out flat by chance).
A mood is picked from the meaning of the user's message and SKYE's reply —
by comparing them to a few example sentences per mood with the sentence
embedder that long-term memory already loads, so it costs milliseconds and no
extra LLM call — and each mood maps to fixed sampling parameters for the TTS
engine, with the default being calm.

Chatterbox Turbo has no emotion dial (its `exaggeration` input is ignored), so
the parameters only shift delivery variability. The strongest lever is a
reference clip recorded in that tone: drop assets/reference_voice_<mood>.wav
(happy, sad, concerned) next to reference_voice_short.wav and tts_server picks
it up for that mood automatically.
"""

import numpy as np

MOOD_PARAMS = {
    "calm":      dict(temperature=0.70, top_p=0.95, repetition_penalty=1.20),
    "happy":     dict(temperature=0.95, top_p=0.98, repetition_penalty=1.15, voice="happy"),
    "sad":       dict(temperature=0.55, top_p=0.90, repetition_penalty=1.25, voice="sad"),
    "concerned": dict(temperature=0.62, top_p=0.92, repetition_penalty=1.22, voice="concerned"),
}

PROTOTYPES = {
    "happy": [
        "I finished the task!", "That's wonderful news!", "Great, it worked!",
        "Congratulations, you did it!", "I completed everything on my list.",
        "Excellent, all done.", "I'm so excited about this.", "Fantastic, that went perfectly.",
    ],
    "sad": [
        "I have some sad news.", "Unfortunately that didn't work out.",
        "I'm sorry for your loss.", "That's really disappointing.", "Things didn't go well.",
        "The flight has been cancelled.", "I lost the game.", "I'm feeling really down today.",
    ],
    "concerned": [
        "I don't understand.", "I'm sorry, I couldn't do that.", "I'm not sure about that.",
        "Apologies, something went wrong.", "I could not find that information.",
        "Could you clarify what you mean?", "I'm confused about this.", "I can't figure this out.",
    ],
    "calm": [
        "The meeting is at three o'clock.", "It is currently four thirty.", "Here is how it works.",
        "I have set the alarm for seven.", "The weather is mild today.", "Sure, I can do that.",
        "A stack is a data structure.", "Let me explain how that works.",
    ],
}

# A non-calm mood must beat calm by this much to win; keeps ordinary factual
# replies from being coloured by a faint resemblance to an emotional sentence.
MARGIN = 0.06


class MoodClassifier:
    def __init__(self, encode):
        """`encode`: list[str] -> array of sentence embeddings."""
        self._encode = encode
        self._proto = {
            m: self._unit(np.asarray(encode(sents), dtype=np.float32))
            for m, sents in PROTOTYPES.items()
        }

    @staticmethod
    def _unit(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def scores(self, text: str) -> dict:
        v = self._unit(np.asarray(self._encode([text]), dtype=np.float32))[0]
        return {m: float((p @ v).max()) for m, p in self._proto.items()}

    def classify(self, user_input: str, reply: str) -> str:
        """The reply carries most of the weight (it is what gets spoken); the
        user's message tips borderline cases ("I finished the task" → happy)."""
        r, u = self.scores(reply), self.scores(user_input or reply)
        combined = {m: 0.65 * r[m] + 0.35 * u[m] for m in r}
        best = max(combined, key=combined.get)
        if best != "calm" and combined[best] - combined["calm"] < MARGIN:
            return "calm"
        return best
