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
    Heuristically generates a 'chosen' response based on what went wrong.
    In a more advanced setup, this would use a larger 'Teacher' model.
    """
    # Pillar 1 & 2 logic:
    # If verbosity_truncate triggered, chosen should be a shorter version.
    if "verbosity_truncate" in triggers:
        # Just use the first few sentences as a placeholder for 'perfectly concise'
        sentences = re.split(r"(?<=[.!?])\s+", rejected)
        chosen_text = " ".join(sentences[:3])
        return f"<draft>User is asking a question. I must be concise.</draft><critique>Previous attempt was too long. Truncating to core facts.</critique><final_answer>{chosen_text}</final_answer>"
    
    if "repetition_abort" in triggers:
        return f"<draft>User is talking. I must not repeat myself.</draft><critique>I detected repetitive loops in my previous thought. I will speak uniquely.</critique><final_answer>Apologies for the repetition. To be clear: [Corrected unique response placeholder]</final_answer>"

    # Default fallback: If it triggered, we just want a standard healthy draft
    return f"<draft>Analyzing request.</draft><critique>Ensuring I follow butler-persona and safety rules.</critique><final_answer>{rejected}</final_answer>"

def convert_logs_to_dpo():
    log_files = []
    for pattern in TELEMETRY_LOGS:
        log_files.extend(glob.glob(pattern))
        
    new_triplets = []
    
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    triggers = data.get("guardrails_triggered", [])
                    
                    # We ONLY care about turns where the model FAILED and needed a guardrail
                    if triggers:
                        prompt = data.get("user_input", "")
                        rejected = data.get("assistant_output", "")
                        chosen = synthesize_chosen(prompt, rejected, triggers)
                        
                        # Format for mlx_lm chat dataset
                        new_triplets.append({
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": chosen}
                            ]
                        })
                except:
                    continue
                    
    if not new_triplets:
        print("No new guardrail failures detected in logs. Nothing to learn.")
        return

    print(f"Synthesized {len(new_triplets)} DPO correction pairs from logs.")
    
    # Write back to train and valid
    # Write back to train and valid
    os.makedirs(os.path.dirname(DPO_TRAIN_FILE), exist_ok=True)
    with open(DPO_TRAIN_FILE, "w") as f:
        for r in new_triplets:
            f.write(json.dumps(r) + "\n")
            
    with open(DPO_VALID_FILE, "w") as f:
        for r in new_triplets:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    convert_logs_to_dpo()
