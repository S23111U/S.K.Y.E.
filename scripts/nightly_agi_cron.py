import os
import subprocess
import time
import sys

# Add root to sys path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

def run_command(cmd_list, description):
    print(f"\n[MLOps]: Starting {description}...")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print(f"[MLOps]: {description} completed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[MLOps ERROR]: {description} failed: {e.stderr}")
        return None

def main():
    print("====================================================")
    print("S.K.Y.E. AUTONOMOUS MASTER ORCHESTRATOR (CRON)")
    print("====================================================")
    
    # 1. PILLAR 3: Fact Extraction (Long-Term Memory)
    # This reads logs, identifies user facts, and updates persistence_profile.json
    run_command([sys.executable, "memory/short-term/proposed_facts.py"], "Pillar 3: Fact Extraction")
    
    # 2. PILLAR 3: Semantic Ingestion (Long-Term Semantics)
    # This chunks and embeds conversation logs for Tier 2 Vector/Keyword search
    run_command([sys.executable, "scripts/ingest_history.py"], "Pillar 3: Semantic Ingestion")
    
    # 3. PILLAR 1 & 2: Autonomous Dataset Generation
    # This generates DPO pairs from guardrail failures in daily logs
    run_command([sys.executable, "data/auto_log_to_dpo.py"], "Pillar 1/2: Auto Dataset Generation")
    
    # 4. TRAINING PHASE: MLX DPO Fine-Tuning
    # We trigger the DPO training loop. 
    # NOTE: We keep it lightweight using DPO mode for tone/correction.
    train_cmd = [
        "mlx_lm.lora",
        "--model", "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "--train",
        "--train-mode", "dpo",
        "--data", "prepared_data_mlx/dpo",
        "--iters", "200",  # Shorter nightly burst to prevent overfitting
        "--adapter-path", "SKYE"
    ]
    run_command(train_cmd, "Neural Matrix Training (DPO)")
    
    # 5. DEPLOYMENT PHASE: Signal Production Server
    print("\n[MLOps]: Signaling Production Engine for hot-swap...")
    signal_file = os.path.join(ROOT, ".reload_model_signal")
    with open(signal_file, "w") as f:
        f.write("RELOAD")
    
    print("====================================================")
    print("AUTONOMOUS CYCLE COMPLETE. S.K.Y.E. IS NOW SMARTER.")
    print("====================================================")

if __name__ == "__main__":
    main()
