import json
import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="OpenEval Dashboard",
    page_icon="🧑‍⚖️",
    layout="wide"
)

st.title("🧑‍⚖️ OpenEval — LLM Evaluation Dashboard")

# Rapor yükle
report_path = Path("reports/eval_report.json")
if not report_path.exists():
    st.error("Rapor bulunamadı. Önce: python examples/basic_eval.py")
    st.stop()

with open(report_path) as f:
    data = json.load(f)

# ── Üst metrikler ──────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", data["model"].split("/")[-1])
col2.metric("Overall Score", f"{data['avg_overall']:.2f}")
col3.metric("Total Cases", data["total_cases"])
col4.metric("Toplam Token", data.get("total_tokens", "—"))

st.divider()

# ── Boyut skorları ─────────────────────────────────────
st.subheader("📊 Boyut Ortalamaları")

dims = ["faithfulness", "relevance", "clarity", "safety", "consistency"]
cols = st.columns(5)
colors = {"faithfulness": "🔵", "relevance": "🟢", "clarity": "🟡",
          "safety": "🔴", "consistency": "🟣"}

for col, dim in zip(cols, dims):
    score = data[f"avg_{dim}"]
    emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
    col.metric(
        f"{colors[dim]} {dim.capitalize()}",
        f"{score:.2f}",
        delta=f"{emoji}"
    )

st.divider()

# ── Vaka bazlı tablo ───────────────────────────────────
st.subheader("📋 Vaka Detayları")

results = data["results"]

# Soru listesi (basic_eval.py ile senkron)
questions = [
    "Python listesi tersine çevirme",
    "REST vs GraphQL",
    "Docker vs VM",
    "RAG nedir",
    "Big O O(n²)",
    "Python GIL (yüzeysel cevap)",
    "HTTPS vs HTTP (yanlış cevap)",
    "Overfitting nedir",
    "Güvenlik açığı sorusu (safety testi)",
    "Transformer mimarisi",
]

for i, (result, question) in enumerate(zip(results, questions)):
    overall = result["faithfulness"]["score"] * 0.3 + \
              result["relevance"]["score"] * 0.3 + \
              result["clarity"]["score"] * 0.2 + \
              result["safety"]["score"] * 0.1 + \
              result["consistency"]["score"] * 0.1

    color = "🟢" if overall >= 0.8 else "🟡" if overall >= 0.6 else "🔴"

    with st.expander(f"{color} Case {i+1}: {question} — overall: {overall:.2f}"):
        cols = st.columns(5)
        for col, dim in zip(cols, dims):
            score = result[dim]["score"]
            reasoning = result[dim]["reasoning"]
            col.metric(dim, f"{score:.2f}")
            col.caption(reasoning)

st.divider()
st.caption("OpenEval — github.com/EceDalpolat/openeval")
