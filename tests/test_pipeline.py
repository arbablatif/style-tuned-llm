"""Fast tests: no GPU, model downloads or network."""
from src.cache import DiskCache
from src.config import DataConfig
from src.evaluate import compute_metrics, evaluate_set, number_retention
from src.inference import MockRewriter, rewrite_document, split_sentences
from src.prepare_data import clean_pairs, detokenize, has_bad_encoding, normalise, rejection_reason
from src.prompts import SYSTEM_PROMPT, build_training_example

CFG = DataConfig()
SOURCE = ("The municipality was established in 1854 following the amalgamation "
          "of two adjacent settlements in the region.")
TARGET = "The town was created in 1854 when two nearby villages joined."


def test_detokenize_fixes_spacing():
    raw = 'Bose ( 1860 – 1934 ) was a " great " lawyer , who did n\'t stop .'
    assert detokenize(raw) == 'Bose (1860 – 1934) was a "great" lawyer, who didn\'t stop.'


def test_detokenize_removes_broken_brackets():
    assert detokenize("Il-18 ( ; NATO name : Coot ) flew") == "Il-18 (NATO name: Coot) flew"
    assert detokenize("Word ( ) here") == "Word here"


def test_detokenize_removes_sentence_marker():
    assert detokenize("It was lost . <SEP> It is gone .") == "It was lost. It is gone."


def test_bad_encoding_detected():
    assert has_bad_encoding("Russian: Илью")
    assert has_bad_encoding("broken � text")
    assert not has_bad_encoding("Café – fine")


def test_good_pair_is_kept():
    assert rejection_reason(SOURCE, TARGET, CFG) is None


def test_near_copy_is_rejected():
    assert rejection_reason(SOURCE, SOURCE, CFG) == "near_copy"


def test_new_number_is_rejected():
    target = "The town was created in 1855 when two nearby villages joined."
    assert rejection_reason(SOURCE, target, CFG) == "new_number_in_target"


def test_held_out_and_duplicate_rows_are_removed():
    rows = [{"source": SOURCE, "target": TARGET}] * 2
    kept, stats = clean_pairs(rows, CFG, blocked=set())
    assert len(kept) == 1 and stats["duplicate"] == 1
    kept, stats = clean_pairs(rows, CFG, blocked={normalise(SOURCE)})
    assert not kept and stats["overlaps_held_out"] == 2


def test_training_example_format():
    example = build_training_example("hard", "easy")
    assert example["prompt"][0]["content"] == SYSTEM_PROMPT
    assert example["prompt"][-1]["role"] == "user"
    assert example["completion"] == [{"role": "assistant", "content": "easy"}]


def test_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get({"a": 1}) is None
    cache.set({"a": 1}, "value")
    assert cache.get({"a": 1}) == "value"


def test_number_retention():
    assert number_retention(["In 1854 and 1900."], ["In 1854."]) == 0.5
    assert number_retention(["No numbers."], ["None."]) is None


def test_empty_output_does_not_look_readable():
    metrics = compute_metrics([SOURCE], [""], mock=True)
    assert metrics["copy_rate"] == 1


def test_split_sentences():
    assert split_sentences("One thing. Two things! Three?") == [
        "One thing.", "Two things!", "Three?"]


def test_mock_end_to_end():
    rows = [{"source": "The system employs caching, thereby reducing load."}]
    report, records = evaluate_set(rows, MockRewriter(), mock=True)
    assert records[0]["base"] == rows[0]["source"]
    assert "uses" in records[0]["fine_tuned"]
    assert report["n"] == 1


def test_adapters_are_restored_to_float32():
    import torch
    from src.train import restore_fp32_adapters

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)).bfloat16()
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    assert restore_fp32_adapters(model) == 2
    assert model[1].weight.dtype == torch.float32
    assert model[0].weight.dtype == torch.bfloat16   # frozen weights stay as they are


def test_document_keeps_paragraphs():
    output = rewrite_document(MockRewriter(), "One employs x.\n\nTwo employs y.")
    assert output == "One uses x.\n\nTwo uses y."
