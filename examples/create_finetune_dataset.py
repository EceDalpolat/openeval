"""
Fine-tuning dataset olusturucu.
Bu script eval sonuclarini kullanarak
base model vs fine-tuned model karsilastirmasi icin
training data uretir.
"""
import json
from pathlib import Path

# Yuksek kaliteli ornek Q&A ciftleri
# Bunlari fine-tuning icin kullanacagiz
dataset = [
    {
        "instruction": "Python'da bir sözlüğü değere göre sırala",
        "response": "sorted(dict.items(), key=lambda x: x[1]) kullanılır. "
                   "Büyükten küçüğe için reverse=True ekle: "
                   "sorted(dict.items(), key=lambda x: x[1], reverse=True)"
    },
    {
        "instruction": "async/await Python'da ne işe yarar?",
        "response": "Asenkron programlama için kullanılır. I/O bound işlemlerde "
                   "(API çağrısı, dosya okuma) thread oluşturmadan bekleme yapılır. "
                   "asyncio event loop üzerinde çalışır, CPU bound işlemler için uygun değildir."
    },
    {
        "instruction": "LLM context window nedir?",
        "response": "Modelin tek seferde işleyebildiği maksimum token sayısıdır. "
                   "GPT-4o 128K, Claude 200K token context'e sahiptir. "
                   "Context dolunca model eski bilgileri 'unutur'."
    },
]

# JSONL formatında kaydet (HuggingFace fine-tuning için standart)
output = Path("data/finetune_dataset.jsonl")
output.parent.mkdir(exist_ok=True)

with open(output, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ {len(dataset)} örnek kaydedildi: {output}")
print("Hafta 2'de bu veriyle Llama'yı fine-tune edeceğiz!")
