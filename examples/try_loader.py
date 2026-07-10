# See step 1 for yourself:  python examples/try_loader.py
#
# This script does NOT score anything — it just reads the JSONL file and shows what was loaded.

from openeval.dataset import load_cases

cases = load_cases("examples/sample_cases.jsonl")

print(f"\n✅ {len(cases)} cases loaded\n")
for i, c in enumerate(cases, 1):
    print(f"{i}. question: {c.question}")
    print(f"   answer  : {c.answer[:60]}...")
    print(f"   context : {'yes' if c.context else 'no'}\n")
