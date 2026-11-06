"""The one prompt used for training, evaluation and the demo.

Base and fine-tuned models must see identical prompts for a fair comparison.
"""
SYSTEM_PROMPT = (
    "You rewrite dense technical text into clear, plain English. Use short "
    "sentences and common words. Keep the original meaning, names and numbers. "
    "Reply with the rewritten text only."
)


def build_messages(text: str) -> list[dict]:
    """Return the chat messages asking the model to rewrite `text`."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite this in plain English:\n{text}"},
    ]


def build_training_example(source: str, target: str) -> dict:
    """Prompt/completion format: TRL then computes loss on the completion only."""
    return {
        "prompt": build_messages(source),
        "completion": [{"role": "assistant", "content": target}],
    }
