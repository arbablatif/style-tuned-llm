# Style-Tuned LLM: QLoRA Fine-Tuning Qwen2.5-1.5B for Plain-English Rewriting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arbablatif/style-tuned-llm/blob/main/notebooks/train_colab.ipynb)

QLoRA fine-tuning of Qwen2.5-1.5B-Instruct, a small open-source LLM, to rewrite complex text in plain English, trained on a free Colab T4 GPU.

## Problem

Technical writing is often hard to read for the people who need it. Large hosted models can simplify it, but they cost money per call and send your text to a third party. This project tests whether a 1.5B-parameter model, fine-tuned for under an hour on a free GPU, can do the job more reliably than the same model with prompting alone.

## Approach

- **Base model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache-2.0.
- **Training data:** 20,000 cleaned complex and simple sentence pairs from [GEM/wiki_auto_asset_turk](https://huggingface.co/datasets/GEM/wiki_auto_asset_turk). No LLM API was used to create data.
- **Training:** 4-bit QLoRA with PEFT and TRL. LoRA rank 16 on all linear layers, up to one epoch, with loss on the rewrite only and a 45-minute cap.
- **Evaluation:** the base model and adapter run in the same 4-bit setup, with the same prompt and greedy decoding. Both are tested on 359 unseen ASSET sentences and 20 hand-written technical passages.

### Data cleaning

These are measured results from `python -m src.prepare_data` for run 2 ([results/data_stats.json](results/data_stats.json)). Of 100,000 shuffled pairs scanned, 30,000 passing candidates were embedded, and the first 20,000 that passed the meaning check became the training set.

| Filter | Pairs removed |
|---|---:|
| Length ratio outside 0.4 to 1.1 | 26,107 |
| Broken encoding or non-Latin script | 9,513 |
| Target drops a number from the source (added in run 2) | 8,836 |
| Near-copy of the source | 8,488 |
| Target not simpler by Flesch-Kincaid grade | 7,036 |
| Target contains a number not in the source | 5,499 |
| Source under 8 or over 60 words | 2,909 |
| Sentence also in the test set | 4 |
| Target too short | 1 |
| Low embedding similarity (likely misaligned) | 2,994 of the 30,000 candidates |

The final split has 20,000 training pairs, 500 validation pairs and 359 test sentences, each test sentence having 10 human simplifications. A check confirmed that no source sentence appears in more than one split.

## Architecture

```mermaid
flowchart LR
    A[Wiki-Auto pairs] --> B[Clean and filter]
    B --> C[20k train / 500 validation]
    D[ASSET test set] -->|removed from training| B
    C --> E[QLoRA training on Colab T4]
    F[Qwen2.5-1.5B in 4-bit] --> E
    E --> G[LoRA adapter on Hugging Face Hub]
    G --> H[Evaluation: base vs fine-tuned]
    D --> H
    I[Technical passages] --> H
    G --> J[Gradio demo and inference script]
    F --> J
```

## Results

<!-- RESULTS:START -->
**ASSET test set** (359 sentences, source FK grade 11.65)

| Metric | Base | Fine-tuned | Human references |
|---|---:|---:|---:|
| FK grade (lower is simpler) | 8.38 | 7.53 | 8.40 |
| BERTScore F1 vs source | 0.88 | 0.91 | 0.93 |
| Embedding similarity vs source | 0.82 | 0.85 | 0.90 |
| Numbers kept | 0.89 | 0.92 | 0.79 |
| Near-copy rate | 0.00 | 0.03 | 0.05 |
| Length vs source | 1.16 | 0.81 | 0.87 |

**Hand-written technical passages** (20 sentences, source FK grade 18.51)

| Metric | Base | Fine-tuned |
|---|---:|---:|
| FK grade (lower is simpler) | 10.62 | 11.28 |
| BERTScore F1 vs source | 0.81 | 0.85 |
| Embedding similarity vs source | 0.66 | 0.73 |
| Numbers kept | 1.00 | 1.00 |
| Near-copy rate | 0.00 | 0.00 |
| Length vs source | 1.46 | 0.84 |

<!-- RESULTS:END -->

Meaning scores are measured against the source, so copying the input gets a perfect score. Compare them with the human references rather than aiming for the highest value.

### What the results show

Run 1 lost too many numbers, so run 2 changed the data: it rejects training pairs whose simple side drops a number. The tables above are run 2. The fix worked on ASSET, and the model is now a better rewriter of encyclopedia-style text, but it still changes facts and shows no readability gain on technical text.

| ASSET metric | Run 1 | Repeat of run 1 | Run 2 |
|---|---:|---:|---:|
| Numbers kept | 0.68 | 0.68 | **0.92** |
| Sentences losing a number (of 112) | 44 | not measured | **12** |
| Embedding similarity vs source | 0.82 | 0.82 | **0.85** |
| BERTScore F1 vs source | 0.90 | 0.90 | **0.91** |
| FK grade | 7.41 | 7.41 | 7.53 |
| Length vs source | 0.81 | 0.81 | 0.81 |

The repeat is an accidental rerun of the run 1 code and data ([results/v1_repeat/](results/v1_repeat/)). Its scores moved by 0.005 or less, so run 2's changes are far larger than run-to-run noise. Run 1 is archived in [results/v1/](results/v1/).

- **Numbers recovered:** numbers kept rose from 68% to 92%, above the base model's 89% and the human 79%. Sentences losing a number fell from 44 to 12, against 16 for the base model.
- **Meaning improved a little:** embedding similarity rose from 0.82 to 0.85, now above the base model but still below the human 0.90.
- **Simplicity was kept:** the grade is 0.84 below the base model (95% bootstrap interval 0.49 to 1.19). The change from run 1 is within noise (-0.35 to 0.09). Output is 81% of the source length, while the base model writes 116%.
- **No readability gain on technical text:** the fine-tuned grade is 11.28 against 10.62 for the base model, and the difference is within noise (95% interval -2.39 to 1.06) on only 20 sentences. Meaning scores are higher (0.73 vs 0.66), and output is shorter (84% vs 146% of the source length).
- **It still invents facts:** 9 ASSET outputs contain a number not written as digits in the source. Three are harmless ("twenty-one" became "21"). Six are invented, such as an unsupported date and "18 years old" for "legal adulthood". On a technical passage it added "about 5 mmHg", which is not in the source. The filter removed training pairs that add numbers, so this habit may come from the base model rather than the data, but that was not tested.
- **It still drops or changes details:** in example 1, "arrhythmia or sudden cardiac death" becomes "heart failure". In example 3, "anthropology" is lost. The number filter can't catch errors like these.
- **The technical "numbers kept" score is not meaningful:** that set contains only one number, so its 1.00 reflects a single case.

### Training run

Training reached the 45-minute cap on a Tesla T4 and stopped at step 1,142 of 1,250, about 91% of one epoch. It trained 18.5 million of 1.56 billion parameters (1.2%). The kept checkpoint is step 1,000, the same step as run 1.

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 250 | 0.867 | 0.789 |
| 500 | 0.858 | 0.816 |
| 750 | 0.810 | 0.788 |
| 1,000 | 0.845 | **0.782** (best, kept) |

Validation loss stayed flat rather than rising steadily, so there is no clear overfitting. Run 1's losses aren't directly comparable, because the filter changed the validation set too.

### Side-by-side examples

Seven ASSET and three technical examples, chosen by a fixed random seed rather than by hand.

<!-- EXAMPLES:START -->
**1. Original:** People with cardiomyopathy are often at risk of arrhythmia or sudden cardiac death or both.

- **Base:** People who have a heart condition called cardiomyopathy usually face a higher chance of having irregular heartbeat problems or suddenly dying from their heart.
- **Fine-tuned:** People who have cardiomyopathy may be at risk for heart failure.

**2. Original:** After graduation he returned to Yerevan to teach at the local Conservatory and later he was appointed artistic director of the Armenian Philarmonic Orchestra.

- **Base:** After finishing school, he went back home to Yerevan where he started teaching at a local music conservatory. Later on, he became the leader of the Armenian Philharmonic Orchestra.
- **Fine-tuned:** He returned to Yerevan to teach at the local Conservatory and later became artistic director of the Armenian Philharmonic Orchestra.

**3. Original:** Graham attended Wheaton College from 1939 to 1943, when he graduated with a BA in anthropology.

- **Base:** Graham went to college at Wheaton College from 1939 to 1943. He finished his studies there and got a Bachelor of Arts degree in anthropology.
- **Fine-tuned:** He studied at Wheaton College from 1939 to 1943. He got his Bachelor of Arts degree there.

**4. Original:** On occasion the male "den master" will also allow a second male into the den; the reason for this is unclear.

- **Base:** Sometimes, when there are two males living together, one of them lets another male join their group. No one knows why they do this.
- **Fine-tuned:** Sometimes the male "den master" allows another male to enter the den.

**5. Original:** Moderate to severe damage extended up the Atlantic coastline and as far inland as West Virginia.

- **Base:** Damage from moderate to severe levels spread along parts of the Atlantic coast and went all the way into West Virginia.
- **Fine-tuned:** Damage was moderate to severe along the Atlantic coast and as far inland as West Virginia.

**6. Original:** The characters are foul-mouthed extensions of their earlier characters Pete and Dud.

- **Base:** The characters speak dirty like they were before, just as Pete and Dud did.
- **Fine-tuned:** They were made from Pete and Dud.

**7. Original:** It continues as the Bohemian Switzerland in the Czech Republic.

- **Base:** It's like Bohemian Switzerland in a place called the Czech Republic.
- **Fine-tuned:** The area is called "Bohemian Switzerland" in the Czech Republic.

**8. Original:** Eventual consistency permits replicas to diverge transiently, provided that all replicas converge to an identical state once updates cease to propagate.

- **Base:** Eventually, replicas can become a little different for a bit, as long as they end up being exactly the same when no more changes are happening.
- **Fine-tuned:** Eventually consistent systems allow replicas to differ for a while, but they will eventually become equal.

**9. Original:** Corrosion of reinforcing steel is initiated when chloride ions penetrate the concrete cover and compromise the passive oxide layer surrounding the bar.

- **Base:** When salty water gets inside the cement that holds up a building or bridge, it can eat away at the metal bars that hold those structures together.
- **Fine-tuned:** The corrosion starts when chloride ions enter the concrete cover and break down the passive oxide layer around the bar.

**10. Original:** The load balancer distributes inbound traffic across a horizontally scaled pool of stateless instances, removing unhealthy nodes based on periodic liveness probes.

- **Base:** The load balancer sends traffic to different computer parts (nodes) that don't remember each other. It checks if these parts are working properly by asking them questions at regular intervals. If it finds out a part isn't working right, it tells the load balancer not to send any more traffic there.
- **Fine-tuned:** Load balancing is done by distributing incoming requests to different servers.

<!-- EXAMPLES:END -->

## How to run

### Train and evaluate in Colab

1. Click the **Open in Colab** badge above.
2. Choose **Runtime > Change runtime type > T4 GPU**.
3. Run the cells in order, and log in to Hugging Face when asked.

### Run locally (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then set ADAPTER_ID
```

```bash
# Rewrite text with the adapter (slow without a GPU)
python -m src.inference --adapter USER/style-tuned-llm-qwen2.5-1.5b-lora-v2 --text "Your text."

# Gradio demo
python app.py --adapter USER/style-tuned-llm-qwen2.5-1.5b-lora-v2
```

### Run without a GPU, model or API

`--mock` uses a simple word-replacement stand-in, and its output goes to `results/mock/`. It tests the plumbing and says nothing about model quality.

```bash
python -m src.prepare_data --mock
python -m src.train --mock
python -m src.evaluate --mock
python -m src.inference --mock --text "The system employs caching."
python app.py --mock

python -m src.train --smoke      # two real training steps on a tiny model, on CPU
pytest
```

The project makes no paid API calls. Model outputs are cached to disk in `cache/`, so repeating an evaluation doesn't regenerate them.

## Project layout

```text
notebooks/train_colab.ipynb   Colab workflow (generated by build_notebook.py)
src/prepare_data.py           Download, clean and split the data
src/train.py                  QLoRA training and Hub upload
src/evaluate.py               Base vs fine-tuned metrics and examples
src/inference.py              Loading, caching and the rewrite command
app.py                        Gradio demo
data/technical_eval.jsonl     20 hand-written technical passages
LEARNING.md                   Plain-language notes on the concepts
```

## Limitations

- **Domain mismatch:** the training data is encyclopedia text, not technical documentation. The technical passages are reported separately for this reason, and there are only 20 of them.
- **Noisy data:** Wiki-Auto pairs are aligned automatically. Filters reduce errors but don't remove them, and some simple sentences add or drop facts other than numbers.
- **Detail loss:** the fine-tuned model can drop details, change facts and invent figures. Keeping numbers is not the same as keeping meaning.
- **Sentence-level training:** long input is split into sentences and each is rewritten separately, so links between sentences can be lost.
- **Single runs:** each result comes from one training run. The repeat of run 1 suggests low noise, but it reused the same seed and data, so it doesn't measure sensitivity to either.
- **Automatic metrics:** Flesch-Kincaid only measures word and sentence length, and the meaning scores can miss a wrong detail. Number retention is a narrow check. Nothing here replaces human review.
- **Not for high-stakes text:** don't use it for medical, legal or safety text without expert review.
- **Licensing:** the base model is Apache-2.0, but licence information for the data differs between sources. The dataset card lists non-commercial terms for parts of it, so check this before any commercial use.

## License

The code is MIT licensed. See [LICENSE](LICENSE). The base model and training data are not covered and keep their own terms, which include non-commercial conditions on parts of the data. See Limitations.

## Related projects

- [rag-evaluation-lab](https://github.com/arbablatif/rag-evaluation-lab): controlled comparison of RAG chunking, embeddings and top-k
- [research-paper-rag](https://github.com/arbablatif/research-paper-rag): RAG assistant with page-level citations and refusals

Project page: [https://arbablatif.github.io/projects/style-tuned-llm/](https://arbablatif.github.io/projects/style-tuned-llm/). More work at [arbablatif.github.io](https://arbablatif.github.io/).
