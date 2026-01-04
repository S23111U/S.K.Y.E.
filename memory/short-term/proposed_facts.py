import os
import glob
import json
import logging
from mlx_lm import load, generate

# --- CONFIGS ---
LOGS_DIR = "logs"
PROCESSED_DIR = "logs/processed_logs"
PROPOSED_FACTS_FILE = "proposed_facts.jsonl"
MODEL_PATH = "mlx-community/Phi-3-mini-4k-instruct-4bit"

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
    log_files = glob.glob(os.path.join(LOGS_DIR, "skye_*.json"))
    if not log_files:
        logging.info("No new log files found")
        return
    logging.info(f"Found new log files - Length: {len(log_files)}")

    # Feeding the data to AI model for analysis
    for filename in log_files:
        logging.info(f"Analyzing {filename}...")
        try:
            with open(filename, "r") as f:
                conversation_data = json.load(f)

                log_text = ""
                for item in conversation_data:
                    log_text += (
                        f"{item.get('role', 'unknown')}: {item.get('content', '')}\n"
                    )

                if not log_text:
                    logging.warning(f"Log file {filename} is empty. Skipping.")
                    continue

                final_prompt = f"""<|system|>{SYSTEM_PROMPT}<|end|><|user|>Conversation:{log_text}<|end|><|assistant|>"""

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
                        with open(PROPOSED_FACTS_FILE, "a") as f:
                            for proposal in proposals:
                                json.dump(proposal, f)
                                f.write("\n")
                            logging.info(
                                f"Successfully saved {len(proposals)} new proposals."
                            )

                    except Exception as e:
                        logging.warning(
                            f"Error writing to file - {PROPOSED_FACTS_FILE}, error: {e}"
                        )

                base_name = os.path.basename(filename)
                os.rename(filename, os.path.join(PROCESSED_DIR, base_name))
                logging.info(f"File moved successfully - {filename}")

        except Exception as e:
            logging.warning(
                f"Could not read/ parse the content of file - {filename}. ERROR: {e}"
            )


if __name__ == "__main__":
    main()
