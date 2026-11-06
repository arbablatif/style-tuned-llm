# Learning notes: how this project works

These notes explain the main ideas in plain language, so you can walk a client through them.

## The one-minute version

A small open-source model can already follow an instruction like "rewrite this simply", but it does so inconsistently. We show it about 20,000 examples of a complex sentence next to a simpler version. Rather than changing the whole model, we train a small add-on called a LoRA adapter. The adapter is a few tens of megabytes, trains on a free GPU in under an hour, and steers the model toward that writing style.

## Fine-tuning or prompting?

Prompting tells a model what you want each time. Fine-tuning changes its habits.

| Use prompting when | Use fine-tuning when |
|---|---|
| The task changes often | The task is narrow and repeated |
| A large hosted model is acceptable | You need a small, cheap or private model |
| You have few examples | You have many good examples |

Fine-tuning teaches **style and behaviour** well. It is a poor way to add **new facts**; retrieval (RAG) is usually better for that.

## What is LoRA?

LoRA stands for Low-Rank Adaptation.

A model is mostly large grids of numbers called weights. Normal fine-tuning updates all of them. For a 1.5 billion parameter model, that means holding the weights, their gradients and the optimizer's state in memory, which is far too much for a free GPU.

LoRA freezes the original weights and learns a small correction next to them. For a weight grid `W`, it learns two thin grids, `A` and `B`, and the layer behaves like `W + B×A`.

**Why this saves so much:** a 1,536 × 1,536 grid has about 2.4 million numbers. With rank 16, `A` and `B` together have only about 49,000, which is 2% of that.

**An analogy:** the base model is a printed textbook. Full fine-tuning reprints the book. LoRA adds a thin set of sticky notes that changes how you read it. You can remove the notes and get the original back, or keep several sets for different jobs.

**The settings used here:**

- **Rank (`r = 16`):** how much the adapter can learn. Higher rank means more capacity, more memory and more risk of overfitting. Writing style is a fairly simple change, so a modest rank is enough.
- **Alpha (`32`):** how strongly the correction is applied. Setting it to twice the rank is a common starting point.
- **Target modules (`all-linear`):** which layers receive adapters. Covering all linear layers usually works better than attention layers alone.
- **Dropout (`0.05`):** randomly switches off a small part of the adapter during training, which discourages memorising.

## What is QLoRA?

QLoRA is LoRA on top of a compressed base model.

Each base weight normally takes 16 bits. QLoRA stores it in 4 bits, so the base model needs about a quarter of the memory. The frozen weights stay compressed, and only the small adapter is trained at higher precision.

Three parts make it work:

1. **NF4:** a 4-bit format designed for how neural network weights are typically distributed, so it loses less detail than a simple 4-bit format.
2. **Double quantization:** also compresses the bookkeeping numbers that quantization creates.
3. **Paged optimizer:** moves optimizer memory off the GPU during spikes, which helps avoid out-of-memory crashes.

**The trade-off:** compression loses a little precision and adds some computing work. In exchange, a free 16 GB T4 can do the job.

**A fair-comparison detail:** this project loads one 4-bit model and switches the adapter on and off. Any difference in the scores therefore comes from the adapter, not from a different loading method.

## What is overfitting?

Overfitting is when a model memorises its training examples instead of learning the general skill. It is like a student who memorises past exam answers and then struggles with new questions.

**How to spot it:** watch two numbers during training.

- **Training loss** is the error on examples the model learns from.
- **Validation loss** is the error on examples it never trains on.

If training loss keeps falling while validation loss rises, the model has started to memorise.

**What this project does about it:**

- Trains for one pass over the data, so each example is seen only once.
- Uses a modest rank and a little dropout.
- Measures validation loss every 250 steps and keeps the best checkpoint.
- Removes every test sentence from the training and validation data, so the final score is on unseen sentences.

**A related risk: overfitting to the domain.** The model can do well on Wikipedia sentences and still do worse on software documentation or legal text. A held-out test set from the same source cannot reveal that. This is why the project also tests 20 hand-written technical passages.

## How to choose a base model

Work through these questions in order:

1. **Licence:** can you use it for your purpose? Qwen2.5-1.5B-Instruct uses Apache-2.0, which is permissive. Llama models have their own community licence and need an access request. Qwen2.5-3B has a separate, more restrictive licence.
2. **Hardware:** will it train and run where you need it? Allow for model weights, activations, adapter and optimizer. A 1.5B model leaves comfortable room on a T4.
3. **Instruct or base version:** an instruct model already follows requests, so it needs less data to learn a task.
4. **Task difficulty:** rewriting one sentence is a fairly narrow task, so a small model is a sensible start. Complex reasoning needs a larger one.
5. **Language and tokenizer:** check that it handles your languages and specialist vocabulary well.
6. **Evidence:** run the untouched model on 20 real examples before training. If it can't do the task at all, a different model may help more than fine-tuning.

**Rule of thumb:** start with the smallest model that might work. Move up only if measurements show you need to, because small models are cheaper to train, serve and test.

## Why the data work matters

Fine-tuning copies the patterns in its data, including bad ones.

- **The first dataset tried was rejected.** It was lowercased and had its numbers removed, so training on it would have taught the model to delete numbers.
- **Near-copies are removed.** If the "simple" sentence equals the original, the model learns to copy.
- **Sentences with new numbers are removed.** They suggest the simple side added a fact, which would teach the model to invent details.
- **Misaligned pairs are removed.** Embedding similarity finds pairs that aren't really about the same thing.

## How to read the results

No single score is enough, because each can be gamed:

| Metric | What it tells you | How it can mislead |
|---|---|---|
| **Flesch-Kincaid grade** | Approximate school grade needed to read the text, based on sentence and word length | Short nonsense scores very well |
| **BERTScore** | How closely the words' meanings match the source | Copying the input scores perfectly |
| **Embedding similarity** | Whether the overall meaning stayed the same | Can miss a wrong detail, such as a changed number |
| **Numbers kept** | Whether figures survived | A simple check that ignores names and logic |
| **Near-copy rate** | Whether the model actually rewrote anything | A low rate doesn't prove the rewrite is good |

**The tension to explain to clients:** good simplification removes detail, so its meaning scores should fall a little. A perfect meaning score usually means the model copied the input. That's why the results include the human-written simplifications as a yardstick. A good model should land near the human scores, not simply at the top of every column.

Always read the examples too. Automatic scores cannot confirm that a rewrite is correct.

## What this project actually found

The honest summary for a client: **fine-tuning changed the style, and it also copied the data's flaws.**

- **It worked on style.** On Wikipedia-like sentences, the output became shorter and about one school grade easier to read.
- **It learned a bad habit.** It kept 68% of numbers, compared with 89% for the base model. The cause is probably the data: we rejected examples that added numbers but not ones that dropped them, and the training targets kept only 73.5%. The model learned to do what the examples did.
- **It didn't transfer cleanly.** On technical text there was no measurable readability gain, and it often kept the jargon. The test set is only 20 sentences, so this is an early signal rather than a firm conclusion.
- **There was no clear overfitting.** Validation loss stayed nearly flat from step 250 onward. That points to the data as the limit, not training length.

**The lesson:** the next improvement should come from better data, not more training. First reject pairs that drop numbers, then add real before-and-after technical examples.

## Design decisions

| Decision | Why |
|---|---|
| Qwen2.5-1.5B-Instruct | Permissive licence, no access request, fits the 1-hour limit on a T4 |
| Wiki-Auto data from GEM | Public, keeps case and numbers, ships as Parquet files |
| ASSET as the test set | Separate data with 10 human simplifications per sentence |
| PEFT and TRL rather than Unsloth | Fewer moving parts on Colab's changing software |
| fp16 rather than bf16 | The T4 has no fast bf16 support |
| Loss on the rewrite only | Training on the fixed prompt would waste effort |
| Greedy decoding in evaluation | The same input gives the same output, so the comparison is repeatable |
| Sentence-by-sentence rewriting | Matches how the model was trained |
| 45-minute training cap | Keeps the run inside the 1-hour limit even on a slow GPU |

## Questions a client may ask

**Could this run on our own servers?** Yes. It's a small open model, so text doesn't need to go to an outside API. A production version needs its own speed and quality testing.

**Is it safe for medical or legal text?** Not without human review. Simplification can drop a condition or change a meaning, and in those fields that can cause real harm.

**Can it learn our house style?** Yes, if you have a few thousand good before-and-after examples. Those examples would be the main cost.

**Can we use this adapter commercially?** Check the licences first. The base model is Apache-2.0, but licence information for the training data varies between sources, and the dataset's card lists non-commercial terms for parts of it. For commercial work, retrain on data you have clear rights to.
