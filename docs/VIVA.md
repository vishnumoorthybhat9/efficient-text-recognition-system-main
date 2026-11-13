# Viva preparation

Questions an examiner is likely to ask, with answers grounded in this
codebase. Every number here can be reproduced with the commands in
`docs/RESULTS.md`.

---

## The result

**Q. What did you actually find?**

On 2,998 held-out real handwriting samples, the same model reached CER 0.1033
with self-supervised pretraining and 0.4339 without — a 76.2 % relative
reduction. On 1,000 synthetic samples the figures were 0.1152 and 0.3377, a
65.9 % reduction. The effect reproduces on two independent datasets.

**Q. How do you know the pretraining caused that, and not something else?**

The two arms are the same architecture with the same 10,850,900 parameters,
trained on the same 2,000 labeled images, with the same seed, augmentation,
optimiser, learning rate, schedule and epoch count. They differ in one config
key: `model.init_encoder_from`.

That is not just a claim — `tests/test_config.py` loads both configs and fails
if they differ anywhere except the encoder initialisation and the output
directory. If someone later changes one arm's learning rate, the test breaks.

**Q. How do you know the weights really transferred?**

`tests/test_models.py::test_simclr_weights_transfer_exactly` saves a SimCLR
encoder, loads it into the recogniser, and asserts tensor-by-tensor equality
for all 26 tensors. The training log also records the transfer report
(`transferred 26/26 encoder tensors (100%)`), which is stored in the
checkpoint's history.

**Q. Could the pretraining have seen your test data?**

No. The unlabeled pool is written to its own directory from an independent
text sample (synthetic) or from training images that appear in no labeled
split (handwriting). No image file is shared with the validation or test
splits. For the handwriting set the transcriptions exist but are never read.

**Q. Why does it help so much?**

Look at the training curves (`artifacts/figures/training_curves.png`). The
from-scratch arms sit at CER ≈ 1.0 for 10–14 epochs — that is the CTC
cold-start: before the model can emit anything useful it must discover an
alignment, and until then it predicts only blanks. Roughly half the epoch
budget is spent escaping that plateau.

The pretrained arms skip it. The encoder already produces features that
separate glyph shapes, so only the recognition head has to be learned. The
pretraining does not merely add accuracy — it removes the hardest part of
low-label CTC optimisation.

---

## Architecture

**Q. Why self-attention instead of an attention decoder?**

A Bahdanau-style decoder emits its own output sequence whose length is
unrelated to the input width. CTC requires one output frame per input frame in
monotonic order — the two are incompatible. Using a decoder would mean
dropping CTC entirely for cross-entropy with teacher forcing, a different
system from the one specified.

Self-attention placed *between* the BiLSTM and the CTC classifier keeps the
sequence length at 63 while giving every frame a global receptive field.
`tests/test_models.py::test_attention_is_length_preserving` pins that property.

**Q. Why 63 time steps?**

Input is 32×256. The encoder downsamples width by 4, and its final valid 2×2
convolution trims one column: 256/4 − 1 = 63. Because CTC needs a blank
between repeated characters, the usable label length is capped at ~37
characters, and `validate_manifest` enforces that before training rather than
letting it appear as an `inf` loss.

**Q. Why not just use a transformer like TrOCR?**

We do include it, as a baseline. Two reasons it is not the main model: it has
61.6 M parameters against our 10.85 M and runs ~20× slower (28 ms vs 1.3 ms
per image), and fine-tuning it does not fit in 4 GB of VRAM at a useful batch
size. The synopsis explicitly asks for *lower computational complexity*.

**Q. What does CTC actually do?**

It marginalises over all alignments between a 63-frame output and a shorter
label, so no per-character bounding boxes are needed. The decode rule is:
collapse consecutive duplicates, then remove blanks. The blank is what
separates a genuine double letter — `[1,1,0,1]` decodes to `"aa"` but
`[1,1,1]` decodes to `"a"`. That is tested explicitly in
`tests/test_decoding.py`.

---

## Self-supervised learning

**Q. Explain SimCLR in one minute.**

Take an unlabeled image, make two randomly augmented views, push both through
the encoder and a small projection head, and train so that the two views of
the same image are closer to each other than to any other image in the batch
(NT-Xent loss). No labels are involved. Afterwards the projection head is
thrown away and only the encoder is kept.

**Q. How do you know it learned anything, rather than collapsing?**

Contrastive accuracy — the fraction of views matched to their correct pair —
rose from 0.30 to 0.996 against a chance level of 0.0079 (= 1/(2B−1) for batch
64). Crucially, held-out accuracy tracked it at 0.9957. A collapsed
representation would score at chance on held-out pairs.

**Q. Why not the standard SimCLR augmentations?**

The canonical recipe uses random resized crop, colour jitter and horizontal
flip. For text those are wrong: flipping a word changes what it says, and
aggressive cropping removes characters. Both teach invariance to the very
signal the downstream task needs.

We substitute blur, noise, stroke erosion/dilation, JPEG artefacts, slant,
brightness/contrast and width cropping bounded at 90 % — nuisances a scanner
or camera genuinely introduces. `horizontal_flip` does not merely default to
off; requesting it raises an exception.

**Q. Why discard the projection head?**

The contrastive objective is invariant to the augmentations by construction.
If it acted directly on the encoder output it would strip information the
recogniser still needs — stroke shape, position. The head absorbs that
invariance so the encoder does not have to. This is the finding from the
original paper.

---

## Data

**Q. Where did your data come from?**

Two sources. A synthetic corpus rendered by `scripts/make_synthetic_data.py`
from the 57 typefaces installed on this machine — real glyph shapes, labels
correct by construction, no download needed. And the Kaggle Handwriting
Recognition dataset: 413,701 photographed handwritten name fields, of which we
imported ~26,000 labeled and ~30,000 unlabeled.

**Q. Is synthetic data legitimate?**

For demonstrating and training the pipeline, yes — the glyphs are real and the
labels are exact. But we checked whether it transfers, and it does not: the
synthetic-trained models score CER ≈ 1.0 on real handwriting. That is why the
real dataset was imported, and why the headline result is the handwriting one.
Reporting synthetic-only OCR numbers as though they generalise would be
misleading.

**Q. Did you clean the data?**

Yes, and the importer reports exactly what it dropped. On the Kaggle set it
discarded 888 rows the annotators had marked `UNREADABLE` — training on those
would teach the model to emit that literal word — plus 30 undecodable images
and one label containing a character outside the charset.

---

## Evaluation

**Q. How are CER and WER defined here?**

`(S + D + I) / N` — substitutions, deletions and insertions in the optimal
alignment, over the reference length; characters for CER, whitespace-delimited
words for WER. Micro-averaged: total edits over total reference length, so
long lines carry proportionate weight. Macro averages are also computed and
stored.

**Q. Can CER exceed 1.0?**

Yes, and ours does for the out-of-domain baselines. Insertions are not bounded
by the reference length, so a model that outputs 50 characters for a 3-character
reference scores well above 1.0. We report it unclamped and clamp only
*accuracy*, which is stated wherever accuracy is shown.

**Q. Your TrOCR baseline scored very badly. Is that fair?**

No, and we say so explicitly. Two things are happening, neither of which is
"TrOCR is weak":

1. **Case.** TrOCR-printed was fine-tuned on all-caps receipts, so it outputs
   caps. Its strict CER on synthetic is 0.6342 but its case-insensitive CER is
   **0.0505** — it reads the text almost perfectly. That is why both columns
   appear in the results table.
2. **Annotation convention.** The handwriting crops include a printed form
   label (`NOM :`); the ground truth is only the handwritten name. TrOCR
   transcribes everything visible — `'now : LORD'` for a reference of
   `'LORAS'`. Our models learned the convention from 2,000 labeled examples;
   TrOCR had no way to know it.

We also found and fixed a decoding artefact that was unfairly penalising it:
it looped emitting `000 000 …` until the token cap. Blocking repeated trigrams
improved its CER from 2.2142 to 1.6162. The honest claim is not that we beat a
transformer, but that a small task-adapted model beats a large general model
used zero-shot, at ~20× the speed.

**Q. How do you know your timing numbers are real?**

CUDA is synchronised around every timed region — otherwise the clock only
measures how long it took to *enqueue* work. Warm-up batches are discarded,
since the first forward pass pays for kernel autotuning. Image loading is timed
separately from inference, because it is a cost every model shares.

---

## Engineering

**Q. What was the hardest bug?**

A performance one. The preprocessing pipeline was taking 22.4 ms per image when
the actual OpenCV work took 0.13 ms. The cause: the stopwatch used to *time*
each stage called `torch.cuda.synchronize()` with no device argument, so timing
a CPU operation forced a full GPU sync. Making `synchronize(None)` a no-op made
preprocessing 168× faster.

The interesting part is that nothing was failing — it just silently cost
roughly 90 % of data-loading time.

**Q. Which bugs did your tests catch?**

Three worth mentioning. The polarity check for light-on-dark text compared the
border median against the global median, a condition that can never be true
because the background is the majority of the pixels either way — the feature
had never worked. The skew estimator returned 0.0 for every negative rotation,
so it only ever corrected half its input range. And Sauvola binarisation
shattered glyphs into fragments, which made a single cropped line segment into
four pieces.

**Q. Why SQLite and not Postgres?**

Single-user desktop application, no concurrent writers beyond one background
training thread. WAL journal mode handles that. Postgres would add an install
step and a service to manage for no benefit at this scale.

**Q. Why is training a thread rather than a subprocess?**

The trainers accept a `should_stop` callback, so cancellation is cooperative —
a cancelled run saves its checkpoint and history before returning. PyTorch
releases the GIL inside its kernels, so the API stays responsive. A subprocess
would need IPC for progress reporting and still could not be killed safely
mid-checkpoint-write.

---

## Limitations — be ready to volunteer these

- One seed per arm. The effect is large and reproduces on two datasets, but no
  variance over seeds is reported.
- The handwriting dataset is single-word name fields, not full lines of prose,
  so results are not comparable to published IAM line-level numbers.
- SimCLR ran for 12 epochs because of the 4 GB / 35 W GPU; the objective was
  still improving when the budget ran out.
- TrOCR is zero-shot, not fine-tuned.
- English only, as scoped in the synopsis.
- The IAM, IIIT5K, MJSynth and SynthText importers are written but unexercised —
  no copy of those datasets was available on this machine.

---

## One-sentence summary

> We built a complete OCR system and used it to run a controlled experiment
> showing that self-supervised pretraining on unlabeled images cuts character
> error rate by 65–76 % in a low-label regime, on both synthetic and real
> handwriting, with no change to the model's size or inference cost.
