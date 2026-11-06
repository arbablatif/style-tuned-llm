"""Rewrite text with the base or fine-tuned model.

Usage:
    python -m src.inference --adapter USER/REPO --text "Dense sentence here."
    python -m src.inference --mock --text "Dense sentence here."
"""
import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.cache import DiskCache
from src.config import BASE_MODEL, CACHE_DIR, GenerationConfig
from src.prompts import SYSTEM_PROMPT, build_messages

MOCK_REPLACEMENTS = {
    "utilize": "use", "whereby": "where", "thereby": "so", "subsequently": "later",
    "employs": "uses", "demonstrated": "showed", "mitigates": "reduces",
    "necessitates": "needs", "permits": "allows",
}


def split_sentences(text: str) -> list[str]:
    """The model is trained on single sentences, so long input is split first."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", text.strip())
    return [part for part in parts if part]


class MockRewriter:
    """Rule-based stand-in: no downloads, no GPU. Its output is not a real result."""

    name = "mock"

    def rewrite(self, texts: list[str], use_adapter: bool = True) -> list[str]:
        if not use_adapter:
            return list(texts)
        pattern = re.compile("|".join(MOCK_REPLACEMENTS), re.IGNORECASE)
        return [pattern.sub(lambda m: MOCK_REPLACEMENTS[m.group().lower()], text)
                for text in texts]


class Rewriter:
    """One base model with an optional adapter that can be switched off.

    Switching the adapter off gives the exact base model, so both systems are
    compared under the same loading and quantization settings.
    """

    def __init__(self, adapter: str | None = None, base_model: str = BASE_MODEL,
                 quantize: bool | None = None, cache_dir: Path = CACHE_DIR):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        self.torch = torch
        self.base_model, self.adapter = base_model, adapter
        self.config = GenerationConfig()
        self.cache = DiskCache(cache_dir)
        self.quantize = torch.cuda.is_available() if quantize is None else quantize

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.padding_side = "left"    # left padding for batched generation
        if self.quantize:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=quantization, dtype=torch.float16,
                device_map={"": 0})
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()

    def _adapter_version(self) -> str:
        """Changes when a local adapter is retrained, so old cache entries miss."""
        weights = Path(self.adapter or "") / "adapter_model.safetensors"
        return str(weights.stat().st_mtime_ns) if weights.exists() else "hub"

    def _key(self, text: str, use_adapter: bool) -> dict:
        return {
            "base": self.base_model, "quantized": self.quantize,
            "adapter": self.adapter if use_adapter else None,
            "adapter_version": self._adapter_version() if use_adapter else None,
            "prompt": SYSTEM_PROMPT, "max_new_tokens": self.config.max_new_tokens,
            "text": text,
        }

    def _generate(self, texts: list[str], use_adapter: bool) -> list[str]:
        prompts = [self.tokenizer.apply_chat_template(
            build_messages(text), tokenize=False, add_generation_prompt=True)
            for text in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        settings = dict(max_new_tokens=self.config.max_new_tokens, do_sample=False,
                        temperature=None, top_p=None, top_k=None,
                        pad_token_id=self.tokenizer.pad_token_id)
        with self.torch.inference_mode():
            if self.adapter and not use_adapter:
                with self.model.disable_adapter():
                    generated = self.model.generate(**inputs, **settings)
            else:
                generated = self.model.generate(**inputs, **settings)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    def rewrite(self, texts: list[str], use_adapter: bool = True) -> list[str]:
        """Rewrite each text, generating only what the cache does not hold."""
        results = [self.cache.get(self._key(t, use_adapter)) for t in texts]
        missing = [i for i, result in enumerate(results) if result is None]
        # Similar lengths per batch waste less padding.
        missing.sort(key=lambda i: len(texts[i]))
        size = self.config.batch_size
        for start in range(0, len(missing), size):
            batch = missing[start:start + size]
            outputs = self._generate([texts[i] for i in batch], use_adapter)
            for i, output in zip(batch, outputs):
                results[i] = output
                self.cache.set(self._key(texts[i], use_adapter), output)
        return results


def rewrite_document(rewriter, text: str, use_adapter: bool = True) -> str:
    """Rewrite paragraphs sentence by sentence, keeping the paragraph breaks."""
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        sentences = split_sentences(paragraph)
        paragraphs.append(" ".join(rewriter.rewrite(sentences, use_adapter)))
    return "\n\n".join(paragraphs)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--adapter", default=os.getenv("ADAPTER_ID"))
    parser.add_argument("--base-only", action="store_true", help="skip the adapter")
    parser.add_argument("--mock", action="store_true", help="no model download")
    args = parser.parse_args()

    if args.mock:
        rewriter = MockRewriter()
    else:
        if not args.adapter and not args.base_only:
            raise SystemExit("Pass --adapter, set ADAPTER_ID, or use --base-only.")
        rewriter = Rewriter(adapter=None if args.base_only else args.adapter)
    print(rewrite_document(rewriter, args.text, use_adapter=not args.base_only))


if __name__ == "__main__":
    main()
