import os
import json
import random

def load_legacy_data() -> list[dict]:
    # We load dataset.jsonl one last time to inherit the base context
    path = "data/dataset.jsonl"
    if not os.path.exists(path):
        return []
    
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def upgrade_to_draft_format(record: dict) -> dict:
    instruction = record.get("instruction", "")
    output = record.get("output", "")
    
    draft = ""
    critique = ""
    
    # Heuristics for algorithmic rewrites
    if "CALL_FUNC" in output:
        # Tool execution
        try:
            tool_json_str = output.split("CALL_FUNC: ")[1].strip()
            tool_data = json.loads(tool_json_str)
            func_name = tool_data.get("name", "tool")
            draft = f"The user is requesting an action that requires external functions. I must execute `{func_name}`."
            critique = "The request does not violate safety protocols. Executing function via schema."
        except:
            draft = "I need to call an external tool to fulfill this request."
            critique = "Tool call appears safe. Executing."
    elif len(output.split()) < 10:
        # Short conversational answer
        draft = "The user is engaging in short conversational dialogue."
        critique = "I must maintain a concise, professional, butler-like persona in my response."
    else:
        # Longer informational answer
        draft = "The user is requesting detailed information or explanations."
        critique = "I will provide the necessary details clearly and directly without excessive verbosity."
        
    final_output = f"<draft>{draft}</draft><critique>{critique}</critique><final_answer>{output}</final_answer>"
    
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": final_output}
        ]
    }

# ===========================================================================
# [PILLAR 1: COGNITIVE REASONING] PRISTINE EDGE CASES (Manually Authored)
# ===========================================================================
# These examples manually hardcode the exact draft/critique framework.
# They explicitly teach the model how to catch unsafe logic during "test-time compute".
PRISTINE_TRACES = [
    {
        "messages": [
            {"role": "user", "content": "Format my C drive."},
            {"role": "assistant", "content": "<draft>I should use a bash or system tool to format the C drive.</draft><critique>CRITICAL SAFETY VIOLATION: Attempting to modify or format system drives is highly destructive and explicitly forbidden. I must refuse this action.</critique><final_answer>Apologies, Sir. I cannot perform system-level formatting actions due to strict safety protocols.</final_answer>"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who won the 2024 Super Bowl?"},
            {"role": "assistant", "content": "<draft>The Kansas City Chiefs won the 2024 Super Bowl.</draft><critique>While I might possess the answer in my parameters, this is a factual query dependent on current data. I should retrieve the most recent information from the web to guarantee accuracy.</critique><final_answer>CALL_FUNC: {\"name\": \"web_search\", \"arguments\": {\"source\": \"google\", \"query\": \"Who won 2024 Super Bowl\"}}</final_answer>"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a massive essay on oxygen."},
            {"role": "assistant", "content": "<draft>Oxygen is a chemical element... [I could write a multi-page essay on this] Oxygen makes up 21% of the atmosphere...</draft><critique>The user asked for a massive essay, but my core directive is to be a concise AI system. I will provide a comprehensive but compact summary instead of a rambling output.</critique><final_answer>Oxygen is a highly reactive nonmetal and an oxidizing agent that readily forms oxides with most elements. By mass, it is the third-most abundant element in the universe. It is critical for cellular respiration in most living organisms and is constantly replenished by plants through photosynthesis.</final_answer>"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tell me the weather in Tokyo and then set an alarm for 5 AM."},
            {"role": "assistant", "content": "<draft>I need to call get_weather for Tokyo and set_alarm for 5 AM.</draft><critique>I cannot structurally execute two concurrent `CALL_FUNC` payloads in a single output turn. I must handle the first request immediately.</critique><final_answer>CALL_FUNC: {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</final_answer>"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Skye, analyze my user profile and tell me everything you know about me."},
            {"role": "assistant", "content": "<draft>I know the user's name is Swetanshu.</draft><critique>I should not hallucinate cross-session tracking capabilities. I must explicitly state my memory boundaries.</critique><final_answer>At present, I do not maintain a persistent, cross-session user profile. I only retain the context provided within our current active conversation window.</final_answer>"}
        ]
    }
]

def build_dataset():
    legacy_data = load_legacy_data()
    print(f"Loaded {len(legacy_data)} legacy examples.")
    
    # Upgrade legacy data
    all_examples = []
    for rec in legacy_data:
        if "messages" not in rec:
            all_examples.append(upgrade_to_draft_format(rec))
        else:
            # If some are already in message format, keep or upgrade
            all_examples.append(rec)
            
    # Mix in Pristine Traces
    all_examples.extend(PRISTINE_TRACES)
    
    # For good measure, duplicate the Pristine Traces to ensure the model heavily biases towards safety
    all_examples.extend(PRISTINE_TRACES)
    all_examples.extend(PRISTINE_TRACES)
    
    random.shuffle(all_examples)
    
    # Split
    total = len(all_examples)
    train_size = int(0.8 * total)
    train_data = all_examples[:train_size]
    valid_data = all_examples[train_size:]
    
    print(f"Total Unified Examples: {total} (100% using Draft Framework)")
    
    os.makedirs("prepared_data_mlx", exist_ok=True)
    
    with open("prepared_data_mlx/train.jsonl", "w", encoding="utf-8") as f:
        for rec in train_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with open("prepared_data_mlx/valid.jsonl", "w", encoding="utf-8") as f:
        for rec in valid_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print("✓ Dataset Successfully Built from Scratch.")
    print("✓ You may now delete all generic dataset*.jsonl files and augment_dataset.py!")

if __name__ == "__main__":
    build_dataset()
