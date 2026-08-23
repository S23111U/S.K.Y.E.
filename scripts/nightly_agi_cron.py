import os
import subprocess
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
    print("S.K.Y.E. NIGHTLY MEMORY CONSOLIDATION (CRON)")
    print("====================================================")

    # 1. PILLAR 3: Fact Extraction (Long-Term Memory)
    # This reads logs, identifies user facts, and updates persistent_profile.json
    run_command(
        [sys.executable, "memory/short-term/proposed_facts.py"],
        "Pillar 3: Fact Extraction",
    )

    # 2. PILLAR 3: Semantic Ingestion (Long-Term Semantics)
    # This chunks and embeds conversation logs for Tier 2 Vector/Keyword search
    run_command(
        [sys.executable, "scripts/ingest_history.py"],
        "Pillar 3: Semantic Ingestion",
    )

    print("====================================================")
    print("NIGHTLY CYCLE COMPLETE.")
    print("====================================================")


if __name__ == "__main__":
    main()
