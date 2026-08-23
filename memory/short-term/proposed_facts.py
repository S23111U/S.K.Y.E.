import os
import glob
import json
import logging
from mlx_lm import load, generate

# --- CONFIGS ---
LOGS_DIR = "logs"
PROCESSED_DIR = "logs/processed_logs"
PROPOSED_FACTS_FILE = "proposed_facts.jsonl"
MODEL_PATH = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
PERSISTENT_PROFILE = "memory/persistent_profile.json"

# --- MODEL PROMPT ---
SYSTEM_PROMPT = """You are a silent, efficient log-analysis system.
Your task is to read a conversation log and identify any new or updated facts about the user ("Sir") or their preferences.
Only extract explicit facts or strong preferences.

Format your findings as a list of JSON objects. Each object must have "action", "path", and "value".
- 'action' can be "ADD", "UPDATE", or "SET".
- 'path' is the JSON path (e.g., "user.interests", "user.name").
- 'value' is the data to be added or updated.

If no facts are found, return an empty list [].

Example Conversation:
[{"user": "By the way, I love watching F1.", "assistant": "Great choice! Formula 1 is a thrilling sport."}]
Example Output:
[{"action": "ADD", "path": "user.interests", "value": "F1"}]"""

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- MAIN LOGIC ---
def main():
    logging.info("--- Starting S.K.Y.E. Fact Proposal Analyzer ---")

    # Loading the model
    logging.info("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    logging.info(f"Model loaded successfully - {MODEL_PATH}")

    # Searching for the logs folder
    if not os.path.exists(LOGS_DIR):
        logging.info("Creating directory...")
        os.makedirs(LOGS_DIR)
        logging.info(f"Created directory - {LOGS_DIR}")

    # Searching for processed dir folder
    if not os.path.exists(PROCESSED_DIR):
        logging.info("Creating directory...")
        os.makedirs(PROCESSED_DIR)
        logging.info(f"Created directory - {PROCESSED_DIR}")

    # Searching for the files in logs folder
    log_files = glob.glob(os.path.join(LOGS_DIR, "telemetry_*.jsonl"))
    if not log_files:
        logging.info("No new log files found")
        return
    logging.info(f"Found new log files - Length: {len(log_files)}")

    # Feeding the data to AI model for analysis
    for filename in log_files:
        logging.info(f"Analyzing {filename}...")
        try:
            with open(filename, "r") as f:
                conversation_data = []
                for line in f:
                    try:
                        conversation_data.append(json.loads(line))
                    except:
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

                final_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConversation:\n{log_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                raw_response = generate(
                    model, tokenizer, final_prompt, temp=0.0, max_tokens=500
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
                        # Load existing profile or start fresh
                        profile = {}
                        if os.path.exists(PERSISTENT_PROFILE):
                            with open(PERSISTENT_PROFILE, "r") as pf:
                                profile = json.load(pf)
                        
                        # Merge proposals into profile
                        for proposal in proposals:
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
                                # ADD always accumulates into a list. An existing
                                # scalar is promoted rather than overwritten, so the
                                # first ADD no longer stores a bare string and later
                                # ADDs no longer clobber it.
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


if __name__ == "__main__":
    main()
