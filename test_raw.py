from mlx_lm import load, generate

print("Loading SKYE (Raw Llama 3) with adapters...")
model, tokenizer = load(
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    adapter_path="SKYE",
)
print("✓ MODEL ONLINE\n")

messages = []

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        break

    messages.append({"role": "user", "content": user_input})
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # We do NOT run any sanitisation, guardrails, or stripping. 
    # Just raw token generation.
    print("\n--- RAW OUTPUT START ---")
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=600,
        verbose=False,
    ).split("<|eot_id|>")[0].strip()
    print(response)
    print("--- RAW OUTPUT END ---\n")

    messages.append({"role": "assistant", "content": response})
