# openeval/cli.py
#
# Terminal komutu. Amaç: kod yazmadan `openeval run cases.jsonl` deyip puan almak.
#
#   openeval run examples/sample_cases.jsonl
#   openeval run cevaplar.jsonl --judge-model openai/gpt-4o --judge-provider openrouter
#
# openeval cevap ÜRETMEZ — dataset'teki hazır cevapları bir "judge" modele puanlatır.

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def _build_connector(provider: str, model: str):
    """provider adına göre doğru connector'ı kurar."""
    if provider == "openrouter":
        from .connectors.openrouter_connector import OpenRouterConnector
        return OpenRouterConnector(model=model)
    if provider == "openai":
        from .connectors.openai_connector import OpenAIConnector
        return OpenAIConnector(model=model)
    if provider == "ollama":
        from .connectors.ollama_connector import OllamaConnector
        return OllamaConnector(model=model)
    raise ValueError(f"Bilinmeyen provider: {provider}")


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()  # .env'den API anahtarlarını yükle

    # Ağır importları komut çalışınca yap (CLI açılışı hızlı kalsın)
    from .dataset import load_cases
    from .eval.evaluator import Evaluator

    cases = load_cases(args.cases)

    judge = _build_connector(args.judge_provider, args.judge_model)
    if not judge.is_available():
        print(
            f"❌ Judge modeli erişilebilir değil: {args.judge_provider}/{args.judge_model}\n"
            f"   OpenRouter/OpenAI için .env'de API anahtarı, Ollama için lokal servis gerekli.",
            file=sys.stderr,
        )
        return 1

    # Cevapları üreten sistem etiketi: verilmezse dataset dosya adı
    label = args.label or Path(args.cases).stem

    evaluator = Evaluator(
        judge_connector=judge, subject_label=label, dataset=str(args.cases)
    )
    report = evaluator.run(cases)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n✅ Rapor kaydedildi: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openeval",
        description="Hafif LLM değerlendirme aracı — cevapları judge modele puanlatır.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Bir JSONL dataset'i puanla")
    p_run.add_argument("cases", help="Puanlanacak .jsonl dosyası")
    p_run.add_argument(
        "--judge-model",
        default="meta-llama/llama-3.2-3b-instruct:free",
        help="Judge modeli (varsayılan: OpenRouter ücretsiz Llama 3.2 3B)",
    )
    p_run.add_argument(
        "--judge-provider",
        default="openrouter",
        choices=["openrouter", "openai", "ollama"],
        help="Judge sağlayıcısı (varsayılan: openrouter)",
    )
    p_run.add_argument(
        "--out",
        default="reports/report.json",
        help="Rapor çıktısı (varsayılan: reports/report.json)",
    )
    p_run.add_argument(
        "--label",
        default=None,
        help="Cevapları üreten sistemin adı (varsayılan: dataset dosya adı)",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
