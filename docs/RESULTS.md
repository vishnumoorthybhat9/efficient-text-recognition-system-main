# Experimental results

Every number in this document was produced by running the models on the
development machine. Nothing is copied from a paper or estimated. The raw data
is in `artifacts/ocr.db` and `artifacts/reports/*.json`; the commands that
produced it are in [Reproducing](#reproducing).

---

## 1. Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2050, 4,095 MB, compute capability 8.6 |
| CUDA | driver 592.00, torch built for CUDA 12.8 |
| CPU | Intel i5-11400H, 6 cores / 12 threads |
| RAM | 16 GB |
| PyTorch | 2.11.0+cu128, bfloat16 autocast enabled |

The GPU is a 35 W mobile part. It sustained roughly 0.8 TFLOPS effective in
this workload, which is what set the training budgets below.

---

## 2. Research question

> Can self-supervised pretraining on unlabeled images recover the accuracy
> that deep OCR normally needs a large labeled corpus to reach?

**Design.** One architecture, trained twice on the same small labeled subset.
The arms differ in exactly one thing: whether the CNN encoder was initialised
from SimCLR pretraining on unlabeled images.

| Held constant across arms | Value |
|---|---|
| Architecture | CNN + BiLSTM×2 + 8-head self-attention + CTC |
| Parameters | 10,850,900 (encoder 5.55 M, BiLSTM 3.15 M, attention 2.10 M, classifier 0.04 M) |
| Labeled training samples | 2,000 (seeded permutation — provably the same images for both arms) |
| Seed | 42 |
| Augmentation | identical `augment` block |
| Optimiser | AdamW, lr 8e-4, cosine schedule, 1 warm-up epoch |
| Epochs | 25 (synthetic) / 30 (handwriting) |
| Evaluation | same held-out test split, greedy CTC decoding, batch 32 |

`tests/test_config.py` fails the build if the two arms ever differ anywhere
else.

---

## 3. Stage 1 — Self-supervised pretraining

No transcription is read at any point in this stage.

### Synthetic pool (11,400 unlabeled images, 600 held out)

| Epoch | NT-Xent loss | Contrastive accuracy | Held-out accuracy |
|---:|---:|---:|---:|
| 1 | 2.7230 | 0.3023 | 0.5035 |
| 4 | 0.5015 | 0.9346 | 0.9418 |
| 8 | 0.1642 | 0.9930 | 0.9939 |
| 12 | 0.1211 | **0.9959** | **0.9957** |

Chance level is `1 / (2B − 1)` = **0.0079** for batch size 64. Wall clock:
526 s (≈35 s/epoch).

### Handwriting pool (28,481 unlabeled images, 1,499 held out)

| Epoch | NT-Xent loss | Contrastive accuracy | Held-out accuracy |
|---:|---:|---:|---:|
| 1 | 2.9836 | 0.2411 | 0.4300 |
| 2 | 1.4187 | 0.6367 | 0.8088 |
| 10 | 0.1533 | 0.9859 | 0.9810 |

**Interpretation.** Held-out accuracy tracks training accuracy closely, so the
encoder learned transferable invariances rather than memorising crops. Had it
memorised, held-out accuracy would have stalled near chance.

---

## 4. Stage 2 — The controlled comparison

### 4.1 Real handwriting — 2,998 held-out samples

| Metric | From scratch | SimCLR pretrained | Change |
|---|---:|---:|---:|
| CER | 0.4339 | **0.1033** | **−76.2 %** relative |
| CER (case-insensitive) | 0.4307 | **0.0998** | −76.8 % |
| WER | 0.9312 | **0.3776** | −59.4 % |
| Exact match | 0.0707 | **0.6364** | +56.6 pp |
| Character accuracy | 0.5661 | **0.8967** | +33.1 pp |
| Parameters | 10.85 M | 10.85 M | identical |

### 4.2 Synthetic — 1,000 held-out samples

| Metric | From scratch | SimCLR pretrained | Change |
|---|---:|---:|---:|
| CER | 0.3377 | **0.1152** | **−65.9 %** relative |
| CER (case-insensitive) | 0.3382 | **0.1115** | −67.0 % |
| WER | 0.7841 | **0.3379** | −56.9 % |
| Exact match | 0.1630 | **0.6090** | +44.6 pp |
| Character accuracy | 0.6623 | **0.8848** | +22.3 pp |

The effect reproduces on two independent datasets, one synthetic and one real.

### 4.3 The mechanism

![Validation CER during training](../artifacts/figures/training_curves.png)

The curves show *why* it helps, which the final numbers alone do not.

The from-scratch arms sit at CER ≈ 1.0 for the first 10–14 epochs. That is the
well-known CTC cold-start: before the model can emit anything useful it must
discover an alignment, and until it does it predicts only blanks. Roughly half
the epoch budget is spent escaping that plateau.

The pretrained arms skip it almost entirely — validation CER is already below
0.35 by epoch 4. The encoder arrives with features that separate glyph shapes,
so only the recognition head has to be learned.

**This is the practical argument for self-supervised pretraining in a
low-label regime**: it does not merely add accuracy, it removes the hardest
part of the optimisation.

---

## 4.4 The production model (not part of the experiment)

The two arms above are deliberately limited to 2,000 labels so the effect of
pretraining is measurable. A separate model is trained on everything, and it
is what the application serves by default:

| Model | Train set | CER | WER | Exact | ms/img |
|---|---:|---:|---:|---:|---:|
| `crnn_full` | 27,992 printed + handwritten | **0.0406** | **0.1631** | **0.8352** | 2.78 |

Measured on 3,998 held-out combined test samples. It starts from the SimCLR
encoder -- there is no reason not to, now that the experiment has established
it helps -- but it carries the arm label `reference` and is excluded from the
control/treatment comparison, so it cannot contaminate the result.

The gap between 0.1033 (treatment arm, 2,000 labels) and 0.0406 (production,
27,992 labels) is the remaining value of more labelled data. Self-supervised
pretraining closed most of the gap from the 0.4339 baseline, but not all of
it: labels still matter.

---

## 4.5 Whole-document extraction

Cropped-line CER is not what someone means by "extract the text from this
document": a page also has to be segmented into lines, and those lines
reassembled in reading order. `scripts/evaluate_documents.py` scores the
entire pipeline against a page's exact ground truth, so segmentation errors
count against the score.

The benchmark is 18 rendered pages (6 layouts x clean / scan / photo capture)
covering 132 text lines, built by `scripts/make_document_benchmark.py`.

| Model | Page CER | Page CER (case-ins.) | Page WER | s/page |
|---|---:|---:|---:|---:|
| Transformer OCR base (IAM) | **0.0287** | **0.0087** | **0.0794** | 2.94 |
| Transformer OCR small (IAM) | 0.0416 | 0.0165 | 0.1308 | 0.39 |
| Transformer OCR small (SROIE) | 0.6772 | 0.0148 | 0.8193 | 0.37 |

Two findings worth stating plainly.

**The TrOCR model names are misleading.** `trocr-*-printed` was fine-tuned on
SROIE receipts, which are upper case throughout, so it returns upper-case text
and scores a strict CER of 0.68 on ordinary documents -- while its
case-insensitive CER is 0.015. The `-handwritten` models were fine-tuned on
IAM, preserve letter case, and are the better *general-purpose document*
recognisers despite the name. The registry now labels them by what they are
good for rather than by their upstream names.

**IAM-style tokenisation cost more than it looked.** The IAM transcriptions
separate punctuation with spaces, so the model faithfully emits
`42 Industrial Estate , Mangalore` and `Laptop stand 51 . 25`. Undoing that
(`ml/text/postprocess.py`, applied by default to TrOCR) moved page CER from
0.0444 to 0.0287 and **page WER from 0.2570 to 0.0794** -- word accuracy was
what suffered, because every misplaced space broke two words. The
post-processor only moves whitespace; a test asserts the non-space characters
are unchanged, so it cannot mask a recognition error.

---

## 5. Computational cost

Latency excludes image loading, discards warm-up batches and synchronises CUDA
around every timed region.

| Model | Params | Checkpoint | ms/img | img/s | Peak tensor memory |
|---|---:|---:|---:|---:|---:|
| CRNN+Attention (either arm) | 10.85 M | 124.2 MB* | 1.30–2.05 | 488–769 | 165–208 MB |
| TrOCR small | 61.60 M | 235.0 MB | 27.2–28.3 | 35–37 | 437–488 MB |

\* The checkpoint also stores optimiser state, which roughly triples it; the
weights alone are ~43 MB.

"Peak tensor memory" is `torch.cuda.max_memory_allocated` during the
benchmark at batch size 32 — tensor allocations only, not the whole process
footprint (CUDA context and cuDNN workspaces add several hundred MB on top for
either model).

The project's recogniser is **5.7× smaller and ~20× faster** than the
transformer baseline. That is the "lower computational complexity" objective
from the synopsis, measured rather than asserted.

---

## 6. The TrOCR baseline, read fairly

| Model | Dataset | Strict CER | Case-insensitive CER | Exact |
|---|---|---:|---:|---:|
| TrOCR small (printed) | synthetic | 0.6342 | **0.0505** | 0.2620 |
| TrOCR small (handwritten) | handwriting | 1.6162 | 0.7141 | 0.0017 |

Two distinct effects, neither of which is "TrOCR is a weak model":

**Case convention.** TrOCR-printed was fine-tuned on SROIE receipts, which are
all upper-case. On the synthetic set its strict CER of 0.6342 falls to 0.0505
once case is ignored — it reads the text almost perfectly and the strict metric
is charging it for a convention. This is exactly why both figures are reported.

**Annotation convention.** The handwriting images are photographed form fields
that contain a printed label. Ground truth is only the handwritten name:

| Reference | TrOCR output |
|---|---|
| `LORAS` | `now : LORD` |
| `CLOUIS` | `Clovis ___` |
| `BECHARA` | `now . Bechkra "` |

`now :` is TrOCR reading the printed `NOM :` on the form. It transcribes
everything visible, because nothing told it the convention. The project's own
models learned it from 2,000 labeled examples.

**A decoding artefact, fixed.** Initially TrOCR scored CER 2.2142 because the
decoder looped on out-of-domain crops, emitting `000 000 000 …` until it hit
the token cap. Blocking repeated trigrams (`no_repeat_ngram_size: 3`) brought
it to 1.6162 and halved its latency. The fix makes the baseline *stronger*,
which is the honest direction to adjust a comparison.

**Conclusion.** TrOCR is a strong model used outside its domain. The defensible
claim is not that a 10.85 M model beats a 61.6 M transformer, but that a small
task-adapted model trained on a few thousand in-domain labels outperforms a
large general model used zero-shot, at a fraction of the cost.

---

## 7. Cross-domain transfer

The synthetic-trained models evaluated on real handwriting:

| Model | CER on handwriting/test |
|---|---:|
| CRNN+Attention (from scratch, synthetic) | 1.0102 |
| CRNN+Attention (SimCLR pretrained, synthetic) | 1.1007 |

Both fail completely. Synthetic data rendered from system fonts does **not**
transfer to photographed handwriting, which is precisely why the real dataset
was imported rather than relying on the generator alone. It is also a caution
against reporting synthetic-only OCR results as though they generalise.

---

## 8. Threats to validity

| Concern | Status |
|---|---|
| Data leakage between pretraining and evaluation | **Controlled.** The unlabeled pool is drawn from images in no labeled split; verified by construction in `prepare_dataset.py` and `make_synthetic_data.py`. |
| Arms differing in something other than the encoder | **Controlled.** Asserted by `tests/test_config.py`. |
| Arms seeing different labeled subsets | **Controlled.** Seeded permutation, asserted by `tests/test_datasets.py`. |
| Transfer not actually happening | **Controlled.** Verified tensor by tensor in `tests/test_models.py`. |
| Single seed per arm | **Not controlled.** One run each. The effect is large and reproduces on two datasets, but no variance is reported. |
| Hyperparameters tuned on the test split | **Avoided.** Model selection used validation CER; the test split was touched only at final evaluation. |
| Handwriting set is single words, not lines | **Acknowledged.** Results are not comparable to published IAM line-level numbers. |

---

## 9. Reproducing

```bash
python scripts/make_synthetic_data.py
python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml
python scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml
python scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml
python scripts/evaluate.py --all --split test
python scripts/compare_models.py --dataset synthetic
```

Training configs use `artifacts/checkpoints` as the shared root; each trainer
adds its run name beneath that directory. The SimCLR encoder is therefore
written to `artifacts/checkpoints/simclr_pretrain/encoder_best.pt`, which is
the input expected by the fine-tuning stage.

For the real-handwriting arm, import the dataset first and use the `hw_*`
configs:

```bash
python scripts/prepare_dataset.py --dataset handwriting --source path/to/archive.zip
python scripts/pretrain_simclr.py --config configs/experiments/hw_simclr_pretrain.yaml
python scripts/train_crnn.py --config configs/experiments/hw_baseline_scratch.yaml
python scripts/train_crnn.py --config configs/experiments/hw_simclr_finetune.yaml
python scripts/evaluate.py --models hw_baseline_scratch hw_simclr_finetune --dataset handwriting
python scripts/compare_models.py --dataset handwriting
```

Total GPU time on the development machine: ≈25 min (synthetic) + ≈30 min
(handwriting).
