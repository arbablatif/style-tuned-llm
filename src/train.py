"""Fine-tune the base model with 4-bit QLoRA.

Usage:
    python -m src.train --push-repo USER/REPO   # real run on a CUDA GPU (Colab T4)
    python -m src.train --mock                  # checks data and prompts, no model
    python -m src.train --smoke                 # few CPU steps on a tiny random model
"""
import argparse
import json
import math
import time
from pathlib import Path

from src.config import (ADAPTER_DIR, BASE_MODEL, DATA_DIR, MOCK_DATA_DIR, OUTPUT_DIR,
                        RESULTS_DIR, SEED, TrainConfig)
from src.prompts import build_training_example

SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def read_examples(path: Path) -> list[dict]:
    """Load cleaned pairs and convert them to chat training examples."""
    with path.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    return [build_training_example(p["source"], p["target"]) for p in pairs]


def load_model(name: str, quantize: bool):
    """Load the frozen base model, in 4-bit NF4 when `quantize` is set."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not quantize:
        return AutoModelForCausalLM.from_pretrained(name)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # the T4 has no fast bfloat16
    )
    return AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quantization, dtype=torch.float16, device_map={"": 0})


def restore_fp32_adapters(model) -> int:
    """Return trainable LoRA weights to float32, and report how many changed.

    TRL casts QLoRA adapters to bfloat16, but the T4 trains with fp16 mixed
    precision, whose gradient scaler cannot handle bfloat16 gradients.
    """
    import torch

    changed = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            changed += 1
    return changed


def make_guard_callback(max_minutes: float):
    """Stop cleanly at the time budget, and fail fast if the loss becomes NaN."""
    from transformers import TrainerCallback

    class Guard(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if (time.time() - self.started) / 60 >= max_minutes:
                print(f"Time budget of {max_minutes} minutes reached; stopping.")
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(loss):
                raise RuntimeError("Loss became NaN or infinite; lower the learning rate.")

    return Guard()


def write_model_card(directory: Path, base_model: str) -> None:
    (directory / "README.md").write_text(
        f"""---
base_model: {base_model}
library_name: peft
license: apache-2.0
tags: [lora, qlora, text-simplification]
---
# style-tuned-llm LoRA adapter

Rewrites complex sentences into plain English. This is a LoRA adapter for
`{base_model}`, trained with 4-bit QLoRA on filtered sentence pairs from
`GEM/wiki_auto_asset_turk`.

Trained on encyclopedia sentences, one sentence at a time. It can drop details,
so check its output before relying on it. The training data has its own licence
terms; check them before any commercial use.
""",
        encoding="utf-8",
    )


def train(args) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = TrainConfig()
    smoke = args.smoke
    if not smoke and not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required. In Colab: Runtime > Change runtime "
                         "type > T4 GPU. Use --smoke or --mock to test without one.")

    data_dir = MOCK_DATA_DIR if smoke else DATA_DIR
    model_name = SMOKE_MODEL if smoke else args.base_model
    output_dir = OUTPUT_DIR / ("smoke" if smoke else "run")
    adapter_dir = OUTPUT_DIR / "smoke_adapter" if smoke else ADAPTER_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"    # right padding for training
    train_set = Dataset.from_list(read_examples(data_dir / "train.jsonl"))
    val_set = Dataset.from_list(read_examples(data_dir / "val.jsonl"))

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    # A smaller batch keeps the effective batch size by accumulating more steps.
    batch_size = args.batch_size or cfg.batch_size
    grad_accum = max(1, cfg.batch_size * cfg.grad_accum // batch_size)
    settings = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1 if smoke else batch_size,
        per_device_eval_batch_size=1 if smoke else batch_size,
        gradient_accumulation_steps=1 if smoke else grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0 if smoke else 40,
        num_train_epochs=cfg.epochs,
        max_steps=2 if smoke else args.max_steps,
        max_length=cfg.max_length,
        completion_only_loss=True,      # learn the rewrite, not the prompt
        logging_steps=1 if smoke else 25,
        eval_strategy="steps",
        eval_steps=1 if smoke else cfg.eval_steps,
        save_strategy="steps",
        save_steps=1 if smoke else cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,    # keep the lowest validation loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not smoke,
        bf16=False,                     # TRL defaults this on; the T4 lacks it
        gradient_checkpointing=not smoke,   # trades speed for memory
        optim="adamw_torch" if smoke else "paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        use_cpu=smoke,
    )
    trainer = SFTTrainer(
        model=load_model(model_name, quantize=not smoke),
        args=settings,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[make_guard_callback(cfg.max_minutes)],
    )
    print(f"Adapter tensors restored to float32: {restore_fp32_adapters(trainer.model)}")
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} of {total:,} "
          f"({100 * trainable / total:.2f}%)")

    started = time.time()
    trainer.train()
    minutes = (time.time() - started) / 60

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_model_card(adapter_dir, model_name)

    summary = {
        "base_model": model_name,
        "train_pairs": len(train_set),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_minutes": round(minutes, 1),
        "steps": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_metric,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }
    results_dir = RESULTS_DIR / "smoke" if smoke else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "train_log.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"Saved adapter to {adapter_dir} after {minutes:.1f} minutes.")

    if args.push_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_repo, exist_ok=True)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=args.push_repo,
                          ignore_patterns=["training_args.bin"])
        print(f"Pushed adapter to https://huggingface.co/{args.push_repo}")


def mock_run() -> None:
    """Check the data-to-prompt path without loading any model."""
    examples = read_examples(MOCK_DATA_DIR / "train.jsonl")
    print(f"[mock] {len(examples)} examples formatted. First example:")
    print(json.dumps(examples[0], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 means full epochs")
    parser.add_argument("--batch-size", type=int, help="lower to 4 if out of memory")
    parser.add_argument("--push-repo", help="Hub repo for the adapter, e.g. user/name")
    args = parser.parse_args()
    if args.mock:
        mock_run()
    else:
        train(args)


if __name__ == "__main__":
    main()
