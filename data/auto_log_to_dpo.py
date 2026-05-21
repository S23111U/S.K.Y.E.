import os
import json
import glob
import re

# =========================================================
# [PILLAR 2: AUTO-LOG-TO-DPO]
# =========================================================
# This script scans telemetry logs for guardrail triggers and
# synthesizes DPO triplets to teach the model to avoid those mistakes.

TELEMETRY_LOGS = ["logs/telemetry_*.jsonl", "logs/processed_logs/telemetry_*.jsonl"]
DPO_TRAIN_FILE = "prepared_data_mlx/train.jsonl"
DPO_VALID_FILE = "prepared_data_mlx/valid.jsonl"

def clean_xml(text):
    """Basic cleanup for comparison/synthesis."""
    return text.strip()

def synthesize_chosen(prompt, rejected, triggers):
    """
    Synthesizes a valid training target sequence using the real, 
    corrected text instead of a bracket placeholder.
    """
    # If the text was truncated due to verbosity, use the actual truncated version
    if "verbosity_truncate" in triggers:
        sentences = re.split(r"(?<=[.!?])\s+", rejected)
        chosen_text = " ".join(sentences[:3])
        return f"<draft>User prompt is short/direct. I must minimize response length to maximize impact.</draft><critique>Previous attempt exceeded token parameters. I will truncate text explicitly.</critique><final_answer>{chosen_text}</final_answer>"
    
    if "repetition_abort" in triggers:
        return f"<draft>System caught a semantic repetition loop.</draft><critique>I must break the repeating structure immediately and provide clear information.</critique><final_answer>Let me restate that more clearly. I have successfully tracked your request parameters and am executing it now.</final_answer>"

    # Default fallback: Strip out any formatting artifacts and wrap with reasoning block templates
    clean_text = rejected.replace("[Corrected unique response placeholder]", "Understood. Proceeding with parameters.")
    return f"<draft>Analyzing request tokens.</draft><critique>Ensuring response follows persona rules and structural guardrails.</critique><final_answer>{clean_text}</final_answer>"

def convert_logs_to_dpo():
    import shutil  # Added for moving files safely

    # We only want to target active logs in the main logs directory
    active_log_pattern = "logs/telemetry_*.jsonl"
    log_files = glob.glob(active_log_pattern)
        
    new_triplets = []
    
    # Read and extract training pairs from active logs
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    triggers = data.get("guardrails_triggered", [])
                    
                    if triggers:
                        prompt = data.get("user_input", "")
                        rejected = data.get("assistant_output", "")
                        chosen = synthesize_chosen(prompt, rejected, triggers)
                        
                        new_triplets.append({
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": chosen}
                            ]
                        })
                except:
                    continue
                    
    if not new_triplets:
        print("No new active guardrail failures detected in logs. Nothing to learn.")
        return

    print(f"Synthesized {len(new_triplets)} correction pairs from active logs.")
    
    # Write to training files
    os.makedirs(os.path.dirname(DPO_TRAIN_FILE), exist_ok=True)
    with open(DPO_TRAIN_FILE, "w") as f:
        for r in new_triplets:
            f.write(json.dumps(r) + "\n")
            
    with open(DPO_VALID_FILE, "w") as f:
        for r in new_triplets:
            f.write(json.dumps(r) + "\n")

    # --- CLEANUP STEP: Move used logs to the archive folder ---
    archive_dir = "logs/processed_logs"
    os.makedirs(archive_dir, exist_ok=True)
    
    for log_file in log_files:
        try:
            destination = os.path.join(archive_dir, os.path.basename(log_file))
            shutil.move(log_file, destination)
            print(f"[Cleanup]: Successfully archived used log file: {log_file}")
        except Exception as e:
            print(f"[Cleanup Warning]: Could not move log file {log_file}: {e}")

if __name__ == "__main__":
    convert_logs_to_dpo()
