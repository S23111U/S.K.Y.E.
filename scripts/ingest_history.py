import os
import json
import glob
import sys

# Add root to sys path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from core.memory_manager import MemoryManager

# =========================================================
# [PILLAR 3: SEMANTIC INGESTOR]
# =========================================================

def ingest_all_logs():
    print("[MLOps]: Starting historical log ingestion for Tier 2 semantics...")
    
    memory = MemoryManager(ROOT)
    log_files = glob.glob(os.path.join(ROOT, "logs", "*.jsonl"))
    
    if not log_files:
        print("[MLOps]: No logs found to ingest.")
        return

    # In a production scenario, we would track which logs have been processed.
    # For now, we do a full clean re-ingestion to build a healthy semantic base.
    total_chunks = 0
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user_input = data.get("user_input", "")
                    assistant_output = data.get("assistant_output", "")
                    
                    if not user_input or not assistant_output:
                        continue
                        
                    # We chunk the dialogue pair into a single semantic unit
                    # Strip XML for cleaner memory
                    clean_output = assistant_output.split("</critique>")[-1].replace("<final_answer>", "").replace("</final_answer>", "").strip()
                    memory_chunk = f"User asked: {user_input}\nSKYE responded: {clean_output}"
                    
                    memory.add_memory(memory_chunk)
                    total_chunks += 1
                except:
                    continue
                    
    print(f"[MLOps]: Successfully ingested {total_chunks} conversation chunks into Tier 2 vector memory.")

if __name__ == "__main__":
    ingest_all_logs()
