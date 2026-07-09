# Adım 1'i kendi gözünle test et:  python examples/try_loader.py
#
# Bu script puanlama YAPMAZ — sadece JSONL dosyasını okuyup ne yüklendiğini gösterir.

from openeval.dataset import load_cases

cases = load_cases("examples/sample_cases.jsonl")

print(f"\n✅ {len(cases)} vaka yüklendi\n")
for i, c in enumerate(cases, 1):
    print(f"{i}. soru   : {c.question}")
    print(f"   cevap  : {c.answer[:60]}...")
    print(f"   context: {'var' if c.context else 'yok'}\n")
