"""
Fine-tuning dataset generator.
This script uses eval results to produce training data
for a base model vs fine-tuned model comparison.
"""
import json
from pathlib import Path

# High-quality example Q&A pairs
# We will use these for fine-tuning
dataset = [
    {
        "instruction": "Sort a dictionary by value in Python",
        "response": "Use sorted(dict.items(), key=lambda x: x[1]). "
                   "For descending order add reverse=True: "
                   "sorted(dict.items(), key=lambda x: x[1], reverse=True)"
    },
    {
        "instruction": "What do async/await do in Python?",
        "response": "They are used for asynchronous programming. For I/O-bound operations "
                   "(API calls, file reading) you wait without creating a thread. "
                   "It runs on the asyncio event loop and is not suitable for CPU-bound work."
    },
    {
        "instruction": "What is an LLM context window?",
        "response": "It is the maximum number of tokens the model can process at once. "
                   "GPT-4o has a 128K and Claude a 200K token context. "
                   "When the context fills up, the model 'forgets' older information."
    },
]

# Save in JSONL format (the standard for HuggingFace fine-tuning)
output = Path("data/finetune_dataset.jsonl")
output.parent.mkdir(exist_ok=True)

with open(output, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ {len(dataset)} examples saved: {output}")
print("In week 2 we will fine-tune Llama with this data!")
