"""Compare the base and fine-tuned models on the held-out test sets.

Usage:
    python -m src.evaluate --adapter outputs/adapter --update-readme
    python -m src.evaluate --mock       # no models; writes to results/mock/
"""
import argparse
import difflib
import json
import random
import re
from pathlib import Path
from statistics import mean

import textstat

from src.config import (BERTSCORE_MODEL, DATA_DIR, EMBEDDING_MODEL, RESULTS_DIR, ROOT,
                        SEED, TECHNICAL_EVAL)
from src.inference import MockRewriter, Rewriter
from src.prepare_data import numbers_in

COPY_THRESHOLD = 0.95   # character similarity at which an output counts as a copy
METRIC_LABELS = [
    ("fk_grade", "FK grade (lower is simpler)"),
    ("bertscore_f1", "BERTScore F1 vs source"),
    ("embedding_similarity", "Embedding similarity vs source"),
    ("number_retention", "Numbers kept"),
    ("copy_rate", "Near-copy rate"),
    ("length_ratio", "Length vs source"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def fk_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


def is_copy(source: str, output: str) -> bool:
    return difflib.SequenceMatcher(None, source, output).ratio() >= COPY_THRESHOLD


def number_retention(sources: list[str], outputs: list[str]) -> float | None:
    """Share of numbers in the sources that survive into the outputs."""
    kept = total = 0
    for source, output in zip(sources, outputs):
        expected = numbers_in(source)
        kept += len(expected & numbers_in(output))
        total += len(expected)
    return kept / total if total else None


def word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap, the download-free meaning proxy used by --mock."""
    first, second = (set(re.findall(r"\w+", s.lower())) for s in (a, b))
    return len(first & second) / max(len(first | second), 1)


def embedding_similarity(sources: list[str], outputs: list[str]) -> float:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    first = model.encode(sources, normalize_embeddings=True, batch_size=64)
    second = model.encode(outputs, normalize_embeddings=True, batch_size=64)
    return float((first * second).sum(axis=1).mean())


def bertscore_f1(sources: list[str], outputs: list[str]) -> float:
    from bert_score import score

    _, _, f1 = score(outputs, sources, model_type=BERTSCORE_MODEL, lang="en",
                     batch_size=32, verbose=False)
    return float(f1.mean())


def compute_metrics(sources: list[str], outputs: list[str], mock: bool) -> dict:
    # An empty answer must not count as "very readable", so it scores as its source.
    outputs = [output or source for source, output in zip(sources, outputs)]
    metrics = {
        "fk_grade": mean(fk_grade(o) for o in outputs),
        "number_retention": number_retention(sources, outputs),
        "copy_rate": mean(is_copy(s, o) for s, o in zip(sources, outputs)),
        "length_ratio": mean(len(o.split()) / max(len(s.split()), 1)
                             for s, o in zip(sources, outputs)),
    }
    if mock:
        metrics["embedding_similarity"] = mean(map(word_overlap, sources, outputs))
        metrics["bertscore_f1"] = None
    else:
        metrics["embedding_similarity"] = embedding_similarity(sources, outputs)
        metrics["bertscore_f1"] = bertscore_f1(sources, outputs)
    return metrics


def evaluate_set(rows: list[dict], rewriter, mock: bool) -> tuple[dict, list[dict]]:
    """Score one test set for the source, both models and any human references."""
    sources = [row["source"] for row in rows]
    base = rewriter.rewrite(sources, use_adapter=False)
    tuned = rewriter.rewrite(sources, use_adapter=True)
    report = {
        "n": len(rows),
        "source_fk_grade": mean(fk_grade(s) for s in sources),
        "base": compute_metrics(sources, base, mock),
        "fine_tuned": compute_metrics(sources, tuned, mock),
    }
    if rows[0].get("references"):
        pairs = [(row["source"], ref) for row in rows for ref in row["references"]]
        report["human"] = compute_metrics([p[0] for p in pairs],
                                          [p[1] for p in pairs], mock)
    records = [{"source": s, "base": b, "fine_tuned": t}
               for s, b, t in zip(sources, base, tuned)]
    return report, records


def format_value(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def metrics_table(report: dict) -> str:
    systems = [("base", "Base"), ("fine_tuned", "Fine-tuned")]
    if "human" in report:
        systems.append(("human", "Human references"))
    lines = ["| Metric | " + " | ".join(label for _, label in systems) + " |",
             "|---|" + "---:|" * len(systems)]
    for key, label in METRIC_LABELS:
        values = [format_value(report[system][key]) for system, _ in systems]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def results_markdown(metrics: dict) -> str:
    parts = []
    for key, title in (("asset", "ASSET test set"),
                       ("technical", "Hand-written technical passages")):
        if key in metrics:
            report = metrics[key]
            parts.append(f"**{title}** ({report['n']} sentences, source FK grade "
                         f"{report['source_fk_grade']:.2f})\n\n{metrics_table(report)}")
    return "\n\n".join(parts)


def examples_markdown(asset: list[dict], technical: list[dict]) -> str:
    """Pick 10 examples by fixed seed, so they are not hand-selected."""
    rng = random.Random(SEED)
    chosen = rng.sample(asset, min(7, len(asset)))
    chosen += rng.sample(technical, min(10 - len(chosen), len(technical)))
    blocks = []
    for number, row in enumerate(chosen, 1):
        blocks.append(f"**{number}. Original:** {row['source']}\n\n"
                      f"- **Base:** {row['base']}\n"
                      f"- **Fine-tuned:** {row['fine_tuned']}")
    return "\n\n".join(blocks)


def update_readme(results: str, examples: str) -> None:
    """Replace the marked README sections with the measured results."""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    for name, content in (("RESULTS", results), ("EXAMPLES", examples)):
        text = re.sub(rf"(<!-- {name}:START -->).*?(<!-- {name}:END -->)",
                      lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}",
                      text, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="local path or Hub repo of the adapter")
    parser.add_argument("--base-model", help="override the base model")
    parser.add_argument("--limit", type=int, help="use only the first N rows per set")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--mock", action="store_true", help="no models or downloads")
    args = parser.parse_args()

    technical = read_jsonl(TECHNICAL_EVAL)[:args.limit]
    if args.mock:
        rewriter, asset = MockRewriter(), None
        args.output_dir = RESULTS_DIR / "mock"
    else:
        if not args.adapter:
            raise SystemExit("Pass --adapter, or use --mock.")
        extra = {"base_model": args.base_model} if args.base_model else {}
        rewriter = Rewriter(adapter=args.adapter, **extra)
        asset = read_jsonl(DATA_DIR / "test.jsonl")[:args.limit]

    metrics, records = {}, {}
    if asset:
        metrics["asset"], records["asset"] = evaluate_set(asset, rewriter, args.mock)
    metrics["technical"], records["technical"] = evaluate_set(technical, rewriter,
                                                              args.mock)

    results = results_markdown(metrics)
    examples = examples_markdown(records.get("asset", []), records["technical"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write = lambda name, text: (args.output_dir / name).write_text(text, encoding="utf-8")
    write("metrics.json", json.dumps(metrics, indent=2))
    write("outputs.json", json.dumps(records, indent=2, ensure_ascii=False))
    write("results_table.md", results + "\n")
    write("examples.md", examples + "\n")
    print(results)

    if args.update_readme and not args.mock:
        update_readme(results, examples)
        print("README updated.")


if __name__ == "__main__":
    main()
