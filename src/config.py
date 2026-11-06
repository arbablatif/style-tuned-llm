"""Central settings, so every script agrees on models, paths and hyperparameters."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MOCK_DATA_DIR = ROOT / "data" / "mock"
TECHNICAL_EVAL = ROOT / "data" / "technical_eval.jsonl"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "adapter"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO_NAME = "style-tuned-llm-qwen2.5-1.5b-lora"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERTSCORE_MODEL = "distilbert-base-uncased"

# Parquet files are read directly and pinned to a commit, so the data cannot
# change underneath us and no dataset loading script is executed.
DATASET_REPO = "GEM/wiki_auto_asset_turk"
DATASET_REVISION = "ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99"
SEED = 42


@dataclass(frozen=True)
class DataConfig:
    n_candidates: int = 30_000   # pairs embedded for the meaning filter
    n_train: int = 20_000
    n_val: int = 500
    min_source_words: int = 8
    max_source_words: int = 60
    min_target_words: int = 4
    min_length_ratio: float = 0.4   # target words / source words
    max_length_ratio: float = 1.1   # simplifications should not add much
    max_char_similarity: float = 0.92  # above this, nothing was really rewritten
    min_grade_drop: float = 0.5     # target must read easier than its source
    min_embedding_similarity: float = 0.65  # below this, likely misaligned


@dataclass(frozen=True)
class TrainConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 1.0
    batch_size: int = 8
    grad_accum: int = 2             # effective batch size 16
    max_length: int = 256
    eval_steps: int = 250
    max_minutes: float = 45.0       # stop early, leaving time inside the 1-hour budget


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    batch_size: int = 16
