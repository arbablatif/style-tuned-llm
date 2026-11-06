"""Gradio demo: base and fine-tuned rewrites side by side.

Usage:
    python app.py --adapter USER/REPO          # add --share for a public Colab link
    python app.py --mock                       # no model download
"""
import argparse
import os

import gradio as gr
import textstat
from dotenv import load_dotenv

from src.inference import MockRewriter, Rewriter, rewrite_document

EXAMPLE = ("Regularization mitigates overfitting by penalizing excessive parameter "
           "magnitudes, thereby constraining model complexity.")


def build_demo(rewriter) -> gr.Blocks:
    def compare(text: str):
        if not text.strip():
            return "", "", ""
        base = rewrite_document(rewriter, text, use_adapter=False)
        tuned = rewrite_document(rewriter, text, use_adapter=True)
        grades = (f"Flesch-Kincaid grade: original "
                  f"{textstat.flesch_kincaid_grade(text):.1f}, base "
                  f"{textstat.flesch_kincaid_grade(base):.1f}, fine-tuned "
                  f"{textstat.flesch_kincaid_grade(tuned):.1f}")
        return base, tuned, grades

    with gr.Blocks(title="Plain English Rewriter") as demo:
        gr.Markdown("# Plain English Rewriter\nA LoRA fine-tune of a 1.5B model. "
                    "Check the output: it can drop or change details.")
        source = gr.Textbox(label="Technical text", lines=5, value=EXAMPLE)
        button = gr.Button("Rewrite", variant="primary")
        with gr.Row():
            base = gr.Textbox(label="Base model", lines=6)
            tuned = gr.Textbox(label="Fine-tuned model", lines=6)
        grades = gr.Markdown()
        button.click(compare, inputs=source, outputs=[base, tuned, grades])
    return demo


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=os.getenv("ADAPTER_ID"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.mock and not args.adapter:
        raise SystemExit("Pass --adapter, set ADAPTER_ID, or use --mock.")
    rewriter = MockRewriter() if args.mock else Rewriter(adapter=args.adapter)
    build_demo(rewriter).launch(share=args.share)


if __name__ == "__main__":
    main()
