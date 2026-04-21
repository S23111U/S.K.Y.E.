import json
import random
import os

# =========================================================
# [PILLAR 2: DYNAMIC TONE PERCEPTION] DPO DATASET GENERATOR
# =========================================================
# Direct Preference Optimization (DPO) uses triplets:
# - prompt: The user's input
# - chosen: The preferred, perfectly toned output (using our test-time compute)
# - rejected: The output we want to mathematically penalize (robotic filler, bad length)

def create_dpo_triplets():
    triplets = []

    # ---------------------------------------------------------
    # SCENARIO 1: SHORT-FORM QUESTIONS (Penalizing Verbosity)
    # ---------------------------------------------------------
    short_prompts = [
        "What time is it?",
        "Turn on the living room lights.",
        "Set an alarm for 6 AM.",
        "What's the weather like in New York?",
        "Remind me to call John at 5.",
        "Open YouTube.",
        "Are you online right now?"
    ]

    for p in short_prompts:
        chosen = (
            "<draft>The user is requesting a simple action or fact.</draft>"
            "<critique>This requires a fast, 1-sentence response. Verbosity is heavily discouraged.</critique>"
            f"<final_answer>Right away. Executing action for: {p}</final_answer>"
        )
        rejected = (
            "<draft>The user is interacting with me.</draft>"
            "<critique>I must be thoroughly polite and verbose.</critique>"
            f"<final_answer>I have received your request regarding '{p}'. I am analyzing my internal systems to process this. All parameters are optimal and I am executing the task immediately. Please let me know if there is absolutely anything else you require assistance with, as I am standing by!</final_answer>"
        )
        triplets.append({"prompt": p, "chosen": chosen, "rejected": rejected})

    # ---------------------------------------------------------
    # SCENARIO 2: LONG-FORM QUESTIONS (Penalizing Brevity)
    # ---------------------------------------------------------
    long_prompts = [
        "Can you explain the history of the Petrodollar system in detail?",
        "What are the philosophical themes in Dostoyevsky's Crime and Punishment?",
        "Write a detailed summary of how quantum entanglement works.",
        "Give me a deep dive into the causes of the 2008 financial crisis.",
        "Explain the architecture of a transformer neural network."
    ]

    for p in long_prompts:
        chosen = (
            "<draft>The user is requesting an in-depth analysis of a complex topic.</draft>"
            "<critique>This is a highly educational prompt. I must provide a rich, multi-paragraph explanation without artificially truncating myself.</critique>"
            f"<final_answer>[Detailed educational response regarding {p}. It involves multiple facets, deep historical context, and comprehensive analysis broken down into clear, structured paragraphs.]</final_answer>"
        )
        rejected = (
            "<draft>The user wants information.</draft>"
            "<critique>I should answer briefly.</critique>"
            "<final_answer>It's a very complex topic with a long history. You can look it up online for more details.</final_answer>"
        )
        triplets.append({"prompt": p, "chosen": chosen, "rejected": rejected})

    # ---------------------------------------------------------
    # SCENARIO 3: TONE CORRECTION (Penalizing Robotic Filler)
    # ---------------------------------------------------------
    tone_prompts = [
        "Thanks Skye.",
        "You're very helpful.",
        "Good job.",
        "I appreciate the help."
    ]

    for p in tone_prompts:
        chosen = (
            "<draft>The user is closing the conversation politely.</draft>"
            "<critique>A simple, warm, and highly concise acknowledgment is required.</critique>"
            "<final_answer>You're entirely welcome.</final_answer>"
        )
        rejected = (
            "<draft>The user is saying thanks.</draft>"
            "<critique>Be robotic and formal.</critique>"
            "<final_answer>Systems are operating flawlessly. Standing by for your next directive. Task completely executed within normal parameters!</final_answer>"
        )
        triplets.append({"prompt": p, "chosen": chosen, "rejected": rejected})

    # To ensure mathematical weight, artificially duplicate these foundational traces
    # to form a solid batch (DPO thrives on contrast). In production, you'd want hundreds distinct.
    final_dataset = triplets * 10
    random.shuffle(final_dataset)

    # Format for Llama 3 Chat Template natively
    mlx_dpo_data = []
    for item in final_dataset:
        # MLX DPO natively supports huggingface conversational arrays!
        # We supply it as prompt, chosen, rejected strings, BUT we must apply the system template!
        prompt_str = f"<|start_header_id|>user<|end_header_id|>\n\n{item['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # We append eot_id to chosen/rejected so the model optimally learns to stop!
        chosen_str = f"{item['chosen']}<|eot_id|>"
        rejected_str = f"{item['rejected']}<|eot_id|>"
        
        mlx_dpo_data.append({
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str
        })

    os.makedirs("prepared_data_mlx/dpo", exist_ok=True)
    with open("prepared_data_mlx/dpo/train.jsonl", "w", encoding="utf-8") as f:
        for rec in mlx_dpo_data:
            f.write(json.dumps(rec) + "\n")
            
    # Use 10% for validation
    with open("prepared_data_mlx/dpo/valid.jsonl", "w", encoding="utf-8") as f:
        for rec in mlx_dpo_data[:10]:
            f.write(json.dumps(rec) + "\n")

    print(f"✓ DPO Dataset Generated: {len(mlx_dpo_data)} pairs targeting Dynamic Length and Tone.")

if __name__ == "__main__":
    create_dpo_triplets()
