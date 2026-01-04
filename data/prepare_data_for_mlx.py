import json
import os

# Create output directory
os.makedirs("prepared_data_mlx", exist_ok=True)


def convert_to_mlx_format():
    with open("data/dataset.jsonl", "r") as input_file:
        lines = input_file.readlines()

    # Calculate split point (80% train, 20% validation)
    total = len(lines)
    train_size = int(0.8 * total)

    print(f"Total examples: {total}")
    print(f"Training examples: {train_size}")
    print(f"Validation examples: {total - train_size}")

    # Process training data
    with open("prepared_data_mlx/train.jsonl", "w") as train_file:
        for line in lines[:train_size]:
            data = json.loads(line)

            # Build user message
            instruction = data["instruction"]
            input_text = data.get("input", "")
            output_text = data["output"]

            # Combine instruction and input if input exists
            if input_text and input_text.strip():
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            # Create chat format
            mlx_entry = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output_text},
                ]
            }

            train_file.write(json.dumps(mlx_entry) + "\n")

    # Process validation data
    with open("prepared_data_mlx/valid.jsonl", "w") as valid_file:
        for line in lines[train_size:]:
            data = json.loads(line)

            instruction = data["instruction"]
            input_text = data.get("input", "")
            output_text = data["output"]

            if input_text and input_text.strip():
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            mlx_entry = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output_text},
                ]
            }

            valid_file.write(json.dumps(mlx_entry) + "\n")

    print("\n✓ Conversion complete!")
    print("Files created:")
    print("  - prepared_data_mlx/train.jsonl")
    print("  - prepared_data_mlx/valid.jsonl")

    # Show sample of converted data
    print("\n--- Sample Converted Entry ---")
    with open("prepared_data_mlx/train.jsonl", "r") as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    convert_to_mlx_format()
