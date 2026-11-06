"""Download, clean and split the complex -> simple sentence pairs.

Usage:
    python -m src.prepare_data          # real data, writes data/processed/
    python -m src.prepare_data --mock   # built-in sample, no downloads
"""
import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import textstat

from src.config import (DATA_DIR, DATASET_REPO, DATASET_REVISION, EMBEDDING_MODEL,
                        MOCK_DATA_DIR, RESULTS_DIR, SEED, DataConfig)

SAMPLE_PAIRS = [
    ("The committee subsequently determined that the proposal was insufficiently detailed.",
     "The committee later decided the proposal did not have enough detail."),
    ("Photosynthesis is the biochemical process whereby plants convert light into chemical energy.",
     "Photosynthesis is how plants turn light into energy."),
    ("The municipality was established in 1854 following the amalgamation of two adjacent settlements.",
     "The town was created in 1854 when two nearby villages joined."),
    ("Prolonged exposure to elevated temperatures may precipitate deterioration of the material.",
     "Long exposure to high heat can damage the material."),
]


def detokenize(text: str) -> str:
    """Undo the spaced-out tokenisation of the corpus ('word , word ( x )')."""
    text = re.sub(r"\s*<SEP>\s*", " ", text)                 # corpus sentence marker
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)        # " quoted " -> "quoted"
    text = re.sub(r"\(\s*[;,:]\s*", "(", text)               # '( ; NATO' -> '(NATO'
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s*\(\s*\)", "", text)                   # empty brackets
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|n't)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def has_bad_encoding(text: str) -> bool:
    """True for broken characters or any script beyond extended Latin."""
    if "�" in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return True
    return any(ord(c) > 0x24F and c not in "–—‘’“”" for c in text)


def normalise(text: str) -> str:
    """Lower-case alphanumerics only, used to compare sentences across splits."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))


def rejection_reason(source: str, target: str, cfg: DataConfig) -> str | None:
    """Return why a pair should be dropped, or None to keep it."""
    source_words, target_words = len(source.split()), len(target.split())
    if has_bad_encoding(source) or has_bad_encoding(target):
        return "bad_encoding"
    if not cfg.min_source_words <= source_words <= cfg.max_source_words:
        return "source_length"
    if target_words < cfg.min_target_words:
        return "target_too_short"
    if not cfg.min_length_ratio <= target_words / source_words <= cfg.max_length_ratio:
        return "length_ratio"
    if numbers_in(target) - numbers_in(source):
        return "new_number_in_target"   # the simple side added a fact
    if numbers_in(source) - numbers_in(target):
        return "number_dropped"         # run 1 learned this habit from such pairs
    if difflib.SequenceMatcher(None, source, target).ratio() > cfg.max_char_similarity:
        return "near_copy"
    drop = (textstat.flesch_kincaid_grade(source)
            - textstat.flesch_kincaid_grade(target))
    if drop < cfg.min_grade_drop:
        return "not_simpler"
    return None


def clean_pairs(rows, cfg: DataConfig, blocked: set[str]) -> tuple[list[dict], Counter]:
    """Apply text cleaning and rule filters. `blocked` holds held-out sentences."""
    kept, stats, seen = [], Counter(), set()
    for row in rows:
        source, target = detokenize(row["source"]), detokenize(row["target"])
        key = normalise(source)
        if key in blocked:
            reason = "overlaps_held_out"
        elif key in seen:
            reason = "duplicate"
        else:
            reason = rejection_reason(source, target, cfg)
        if reason:
            stats[reason] += 1
            continue
        seen.add(key)
        kept.append({"source": source, "target": target})
    return kept, stats


def filter_by_meaning(pairs: list[dict], cfg: DataConfig) -> tuple[list[dict], float]:
    """Drop likely misaligned pairs using local sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(key: str):
        return model.encode([p[key] for p in pairs], batch_size=128,
                            normalize_embeddings=True, show_progress_bar=True)

    similarity = (encode("source") * encode("target")).sum(axis=1)
    kept = [p for p, s in zip(pairs, similarity)
            if s >= cfg.min_embedding_similarity]
    return kept, float(similarity.mean())


def load_split(split: str):
    from datasets import load_dataset

    url = (f"hf://datasets/{DATASET_REPO}@{DATASET_REVISION}/"
           f"wiki_auto_asset_turk/{split}-*.parquet")
    return load_dataset("parquet", data_files={split: url}, split=split)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_real(cfg: DataConfig) -> dict:
    # ASSET: held-out test set with 10 human simplifications per sentence.
    test = [{"source": detokenize(r["source"]),
             "references": [detokenize(x) for x in r["references"]]}
            for r in load_split("test_asset")]
    blocked = {normalise(r["source"]) for r in test}

    train_rows = load_split("train").shuffle(seed=SEED)
    val_rows = load_split("validation").shuffle(seed=SEED)

    # Rule filters are cheap, so stop once there are enough candidates.
    candidates, stats, scanned = {}, Counter(), 0
    for start in range(0, len(train_rows), 10_000):
        chunk = train_rows.select(range(start, min(start + 10_000, len(train_rows))))
        kept, chunk_stats = clean_pairs(chunk, cfg, blocked)
        for pair in kept:
            candidates.setdefault(normalise(pair["source"]), pair)
        stats += chunk_stats
        scanned += len(chunk)
        if len(candidates) >= cfg.n_candidates:
            break
    pairs = list(candidates.values())[:cfg.n_candidates]

    meaning_kept, mean_similarity = filter_by_meaning(pairs, cfg)
    if len(meaning_kept) < cfg.n_train:
        raise ValueError(f"Only {len(meaning_kept)} pairs left; need {cfg.n_train}.")
    train = meaning_kept[:cfg.n_train]

    # Validation must not overlap training either.
    train_keys = {normalise(p["source"]) for p in train}
    val_pool, _ = clean_pairs(val_rows.select(range(5_000)), cfg, blocked | train_keys)
    val, _ = filter_by_meaning(val_pool, cfg)
    val = val[:cfg.n_val]

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    write_jsonl(DATA_DIR / "test.jsonl", test)
    return {
        "dataset": DATASET_REPO, "revision": DATASET_REVISION,
        "rows_scanned": scanned, "rejected_by_rule": dict(stats.most_common()),
        "passed_rules_embedded": len(pairs),
        "rejected_by_meaning": len(pairs) - len(meaning_kept),
        "mean_embedding_similarity": round(mean_similarity, 3),
        "train": len(train), "val": len(val), "test": len(test),
    }


def prepare_mock() -> dict:
    pairs = [{"source": s, "target": t} for s, t in SAMPLE_PAIRS]
    write_jsonl(MOCK_DATA_DIR / "train.jsonl", pairs[:3])
    write_jsonl(MOCK_DATA_DIR / "val.jsonl", pairs[3:])
    return {"mock": True, "train": 3, "val": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="no downloads")
    args = parser.parse_args()
    stats = prepare_mock() if args.mock else prepare_real(DataConfig())
    if not args.mock:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
