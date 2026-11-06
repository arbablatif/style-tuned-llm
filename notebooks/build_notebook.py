"""Generate train_colab.ipynb, so the notebook stays small and reviewable."""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/arbablatif/style-tuned-llm.git"

CELLS = [
    ("md", """# style-tuned-llm: QLoRA training on a free Colab T4

Fine-tunes Qwen2.5-1.5B-Instruct to rewrite complex text in plain English.

This is run 2. Its training data also rejects pairs that drop a number, and its
adapter goes to a separate `-v2` Hub repo, so run 1 is kept for comparison.

**Before you start:** choose *Runtime > Change runtime type > T4 GPU*, then run the
cells in order. Training is capped at 45 minutes. The whole notebook needs no paid API."""),
    ("md", "## 1. Check the GPU"),
    ("code", """import torch
assert torch.cuda.is_available(), "No GPU: choose Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))"""),
    ("md", "## 2. Get the code and install pinned packages"),
    ("code", f"""%cd /content
# Remove any earlier copy: a failed clone would silently leave old code in place.
!rm -rf /content/style-tuned-llm
!git clone {REPO_URL}
%cd /content/style-tuned-llm
!git log -1 --oneline
!pip install -q -r requirements-colab.txt"""),
    ("md", """## 3. Log in to Hugging Face

Use a token with **write** access, from https://huggingface.co/settings/tokens.
The token is not stored in this notebook."""),
    ("code", """from huggingface_hub import notebook_login
notebook_login()"""),
    ("code", """from huggingface_hub import whoami
from src.config import ADAPTER_REPO_NAME
HUB_REPO = f"{whoami()['name']}/{ADAPTER_REPO_NAME}"
print("The adapter will be pushed to:", HUB_REPO)"""),
    ("md", "## 4. Prepare the data"),
    ("code", """!python -m src.prepare_data"""),
    ("code", """# Stop here if this is old code, rather than spend 45 minutes training on it.
import json
stats = json.load(open("results/data_stats.json"))
dropped = stats["rejected_by_rule"].get("number_dropped", 0)
assert dropped > 0, "The number filter did not run: push the latest commit, then start again."
print(f"Number filter is active: {dropped:,} pairs rejected.")"""),
    ("md", """## 5. Train and push the adapter

If you see `CUDA out of memory`, add `--batch-size 4` and run this cell again."""),
    ("code", """!python -m src.train --push-repo {HUB_REPO}"""),
    ("md", """## 6. Check for overfitting

Validation loss should fall and then level off. If it rises while training
loss keeps falling, the model is overfitting."""),
    ("code", """import json
log = json.load(open("results/train_log.json"))
print({k: v for k, v in log.items() if k != "log_history"})
train = {e["step"]: e["loss"] for e in log["log_history"] if "loss" in e}
for entry in log["log_history"]:
    if "eval_loss" in entry:
        step = entry["step"]
        print(f"step {step:>5}  train {train.get(step, float('nan')):.3f}  "
              f"validation {entry['eval_loss']:.3f}")"""),
    ("md", "## 7. Evaluate base vs fine-tuned"),
    ("code", """!python -m src.evaluate --adapter outputs/adapter"""),
    ("code", """from IPython.display import Markdown, display
display(Markdown(open("results/examples.md", encoding="utf-8").read()))"""),
    ("md", """## 8. Download the results

**Do this before trying the demo.** Colab deletes these files when the session ends.
Save `style_tuned_results.zip`, then send it back or unzip it over your local repo."""),
    ("code", """!zip -q -r style_tuned_results.zip results -x "results/v1/*" "results/v1_repeat/*"
from google.colab import files
files.download("style_tuned_results.zip")"""),
    ("md", """## 9. Try the demo (optional)

Open the public `gradio.live` link printed below. Press the stop button on this
cell when you finish. Anyone with the link can use the demo while it runs."""),
    ("code", """!python app.py --adapter outputs/adapter --share"""),
]


def main() -> None:
    build = {"md": new_markdown_cell, "code": new_code_cell}
    notebook = new_notebook(
        cells=[build[kind](source) for kind, source in CELLS],
        metadata={
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    )
    nbformat.validate(notebook)
    path = Path(__file__).parent / "train_colab.ipynb"
    nbformat.write(notebook, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
