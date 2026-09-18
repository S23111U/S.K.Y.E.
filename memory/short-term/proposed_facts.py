import os
import glob
import json
import logging
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# --- CONFIGS ---
# Anchored to the repo root rather than the current working directory, since
# this module is now also imported and called in-process by core_v2.py's
# scheduler (running from wherever the server process happened to start),
# not just invoked as a standalone script from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(ROOT, "logs")
PROCESSED_DIR = os.path.join(ROOT, "logs", "processed_logs")
MODEL_PATH = "mlx-community/gemma-4-e4b-it-4bit"
PERSISTENT_PROFILE = os.path.join(ROOT, "memory", "persistent_profile.json")

# --- MODEL PROMPT ---
SYSTEM_PROMPT = """You are a silent, efficient log-analysis system.
Your task is to read a conversation log and identify any new, updated, or retracted facts about the user ("Sir") or their preferences.

Three kinds of things are worth extracting:
1. Explicit facts or strong preferences the user stated outright (e.g. "I love watching F1").
2. Implied routine or habits — a regular wake/sleep time, recurring requests at similar times of
   day, a habitual task, a recurring topic that comes up unprompted more than once. These should
   only be proposed when the conversation actually implies a pattern, not guessed from a single
   one-off mention. Use the "routine" path prefix for these (e.g. "routine.morning_checkin",
   "routine.typical_wake_time") to keep them distinct from plain preferences.
3. Explicit corrections or retractions — the user directly contradicting something they said
   before, or something the assistant said back to them, in THIS SAME conversation (e.g. "actually
   I moved to Melbourne, not Sydney" or "I don't like cats anymore"). Only propose these for a
   genuine, unambiguous contradiction, never a guess.

Format your findings as a list of JSON objects. Each object must have "action", "path", and "value",
plus "old_claim" for the CORRECT action only.
- 'action' can be "ADD" (append to a list), "SET"/"UPDATE" (overwrite), "REMOVE" (retract one
  previously-ADDed value from a list), or "CORRECT" (overwrite AND flag old semantic memory of the
  contradicted claim for removal).
- 'path' is the JSON path (e.g., "user.interests", "user.name", "routine.typical_wake_time").
- 'value' is the data to be added, updated, or (for REMOVE) the value being retracted.
- 'old_claim' (CORRECT only) is a short plain-English restatement of the specific wrong claim being
  corrected, written so it would closely match how that claim would appear in past conversation —
  this is used to find and retire the old, now-wrong memory of it.

If nothing is found, return an empty list [].

Example Conversation:
[{"user": "By the way, I love watching F1.", "assistant": "Noted."},
 {"user": "Also, I don't like cats anymore.", "assistant": "Understood — updating that."},
 {"user": "And correction — I actually moved to Melbourne last month, not Sydney.", "assistant": "Updated."}]
Example Output:
[{"action": "ADD", "path": "user.interests", "value": "F1"},
 {"action": "REMOVE", "path": "user.interests", "value": "cats"},
 {"action": "CORRECT", "path": "user.current_location", "value": "Melbourne", "old_claim": "User lives in Sydney."}]

("don't like cats anymore" retracts a preference from an EARLIER conversation
not shown here — REMOVE does not require the original ADD to appear in this
same log, since profile facts accumulate across many days.)"""

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _merge_proposal(profile, proposal):
    """Applies one proposal to `profile` in place.

    Returns the proposal's "old_claim" string for a CORRECT action, else
    None. The caller collects these across all files and hands them to
    MemoryManager.supersede_similar() itself — this function only ever
    touches persistent_profile.json, never semantics.db, so proposed_facts.py
    stays decoupled from MemoryManager exactly as it was before.
    """
    action = proposal.get("action")
    path = proposal.get("path", "")
    value = proposal.get("value")

    # Simple path handling: "user.name" -> profile["user"]["name"]
    keys = path.split(".")
    d = profile
    for k in keys[:-1]:
        d = d.setdefault(k, {})

    key = keys[-1]
    if action == "ADD":
        # ADD always accumulates into a list. An existing scalar is promoted
        # rather than overwritten, so the first ADD no longer stores a bare
        # string and later ADDs no longer clobber it.
        existing = d.get(key)
        if isinstance(existing, list):
            merged = list(existing)
        elif existing is None:
            merged = []
        else:
            merged = [existing]
        incoming = value if isinstance(value, list) else [value]
        for item in incoming:
            if item not in merged:
                merged.append(item)
        d[key] = merged
    elif action in ("SET", "UPDATE"):
        d[key] = value
    elif action == "REMOVE":
        # Retracts one value from an ADD-accumulated list. A no-op if the
        # value was never there (e.g. it was ADDed in a log file already
        # processed under an old prompt version, before REMOVE existed) —
        # that's fine, nothing to retract.
        existing = d.get(key)
        if isinstance(existing, list):
            to_remove = value if isinstance(value, list) else [value]
            d[key] = [item for item in existing if item not in to_remove]
    elif action == "CORRECT":
        d[key] = value
        return proposal.get("old_claim")
    return None


def run_fact_extraction(model, tokenizer):
    """Reads unprocessed telemetry logs, extracts facts/routine observations via
    the given (already-loaded) model, and merges them into persistent_profile.json.

    Takes an already-loaded model/tokenizer rather than loading its own, so the
    in-process nightly scheduler in core_v2.py can reuse the server's resident
    Gemma instance instead of paying for a second full model load. `main()`
    below still loads its own copy for standalone/manual runs.

    Returns a list of "old_claim" strings from any CORRECT proposals found —
    the caller (core_v2.py's _run_daily_consolidation) is responsible for
    superseding matching semantic memory, and deliberately does so *after*
    ingest_all_logs() runs, not here: ingestion re-embeds every dialogue pair
    in these same log files unconditionally, including the turn where the
    now-wrong claim was originally stated, so superseding it before that
    would just have it silently reappear moments later.
    """
    logging.info("--- Starting S.K.Y.E. Fact Proposal Analyzer ---")

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    old_claims = []

    log_files = glob.glob(os.path.join(LOGS_DIR, "telemetry_*.jsonl"))
    if not log_files:
        logging.info("No new log files found")
        return old_claims
    logging.info(f"Found new log files - Length: {len(log_files)}")

    for filename in log_files:
        logging.info(f"Analyzing {filename}...")
        try:
            with open(filename, "r") as f:
                conversation_data = []
                for line in f:
                    try:
                        conversation_data.append(json.loads(line))
                    except Exception:
                        pass

                log_text = ""
                for item in conversation_data:
                    user_input = item.get("user_input", "")
                    assistant_output = item.get("assistant_output", "")
                    # Strip XML for clean extraction
                    clean_output = assistant_output.split("</critique>")[-1].replace("<final_answer>", "").replace("</final_answer>", "").strip()
                    log_text += f"User: {user_input}\nAssistant: {clean_output}\n"

                if not log_text.strip():
                    logging.warning(f"Log file {filename} is empty or unparsable. Skipping.")
                    continue

                final_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Conversation:\n{log_text}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                raw_response = generate(
                    model, tokenizer, final_prompt, max_tokens=500,
                    sampler=make_sampler(temp=0.0),
                )

                try:
                    start_index = raw_response.find("[")
                    end_index = raw_response.rfind("]")

                    if (
                        start_index != -1
                        and end_index != -1
                        and end_index > start_index
                    ):
                        jstring = raw_response[start_index : end_index + 1]
                        proposals = json.loads(jstring)
                    else:
                        proposals = []

                except Exception as e:
                    logging.warning(
                        f"Could not parse JSON from model response. Error: {e}"
                    )
                    logging.debug(f"Raw model response was: {raw_response}")
                    proposals = []

                if proposals:
                    try:
                        profile = {}
                        if os.path.exists(PERSISTENT_PROFILE):
                            with open(PERSISTENT_PROFILE, "r") as pf:
                                profile = json.load(pf)

                        for proposal in proposals:
                            old_claim = _merge_proposal(profile, proposal)
                            if old_claim:
                                old_claims.append(old_claim)

                        with open(PERSISTENT_PROFILE, "w") as pf:
                            json.dump(profile, pf, indent=4)

                        logging.info(f"Successfully updated persistent profile with {len(proposals)} new findings.")

                    except Exception as e:
                        logging.warning(f"Error updating profile - {PERSISTENT_PROFILE}, error: {e}")

                base_name = os.path.basename(filename)
                os.rename(filename, os.path.join(PROCESSED_DIR, base_name))
                logging.info(f"File moved successfully - {filename}")

        except Exception as e:
            logging.warning(
                f"Could not read/ parse the content of file - {filename}. ERROR: {e}"
            )

    return old_claims


def main():
    logging.info("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    logging.info(f"Model loaded successfully - {MODEL_PATH}")
    old_claims = run_fact_extraction(model, tokenizer)
    if old_claims:
        logging.info(f"Detected {len(old_claims)} correction(s) — run alongside "
                      "core_v2.py's scheduler for these to actually supersede semantic memory; "
                      "standalone runs only update persistent_profile.json.")


if __name__ == "__main__":
    main()
