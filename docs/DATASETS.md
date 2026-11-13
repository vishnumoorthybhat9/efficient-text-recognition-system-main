# Dataset guide

Where to get data, how to prepare it, and what the project does with it.

---

## 1. The manifest format

Every dataset — whatever its origin — is converted to one two-column TSV:

```
images/train/000001.png	Invoice total
images/train/000002.png	INV-4471
```

Rules:

- Image paths are **relative to the directory containing the manifest**, so a
  prepared dataset is movable between machines.
- Labels may contain spaces; only the **first** tab separates the columns.
- Lines beginning with `#` are comments.

An unlabeled manifest (for self-supervised pretraining) is the same thing with
the label column omitted — one path per line.

Because everything downstream reads this one format, switching the whole
experiment to a different corpus is a change of config paths, nothing more.

---

## 2. Synthetic corpus (no download)

```bash
python scripts/make_synthetic_data.py
```

| Preset | Train | Val | Test | Unlabeled | Time |
|---|---:|---:|---:|---:|---:|
| `--preset quick` | 1,000 | 200 | 200 | 1,500 | ~20 s |
| `--preset default` | 8,000 | 1,000 | 1,000 | 12,000 | ~2 min |
| `--preset large` | 30,000 | 3,000 | 3,000 | 45,000 | ~8 min |

Renders text lines from the typefaces installed on your machine — 39 printed
and 18 handwriting faces were found on the development machine. Varied
backgrounds (plain, gradient, noise, paper texture), lighting, slant, stroke
weight and per-character baseline jitter.

Text is sampled from four families so the charset is genuinely exercised:
words, phrases, alphanumeric codes (`INV-4471`) and numeric/punctuated strings
(dates, amounts, percentages, times).

**Provenance is recorded.** `dataset_info.json` stores the seed, counts, the
exact fonts used and how often each was used, the charset and the validation
report.

**What it is.** Genuine training data: real glyph shapes from real typefaces,
labels correct by construction.

**What it is not.** A substitute for IAM. The "handwritten" styles are script
*typefaces* with jitter — more regular than real cursive. We measured this:
synthetic-trained models score CER ≈ 1.0 on real handwriting.

## Document benchmark (evaluation only)

Generate the small whole-page benchmark with exact reference text:

```bash
python scripts/make_document_benchmark.py
```

It creates 18 page images (six documents with clean, scanned and photo-like
captures) and stores their transcriptions in
`data/processed/docbench/benchmark.json`. In the Datasets page this appears
as an **Evaluation only** benchmark with a `test` split. It intentionally has
no train or validation split; use `python scripts/evaluate_documents.py` to
measure OCR results, and do not train on these test pages.

---

## 3. Kaggle Handwriting Recognition — used for the headline result

**Source:** <https://www.kaggle.com/datasets/landlord/handwriting-recognition>
**Size:** ~1.3 GB zipped · 413,701 images · no registration

Photographed handwritten name fields with transcriptions in
`written_name_*_v2.csv`.

```bash
python scripts/prepare_dataset.py --dataset handwriting \
    --source "path/to/archive.zip" \
    --train 20000 --val 3000 --test 3000 --unlabeled 30000
```

The archive is read **in place** — no 1.3 GB extraction — and only the
requested subset is written out.

**What the importer filters, and why:**

| Dropped | Count on this run | Reason |
|---|---:|---|
| `UNREADABLE` labels | 888 | Marked illegible by the annotators; training on them teaches the model to emit that literal word |
| Undecodable / blank images | 30 | Corrupt or uniform |
| Labels outside the charset | 1 | Contained a backtick |

**A characteristic worth knowing:** many crops include the printed form label
(`NOM :`, `PRENOM :`) above or beside the handwriting, while the ground truth
is only the handwritten name. The model must learn to ignore the printed part.
This is why the zero-shot TrOCR baseline scores poorly here — see
`docs/RESULTS.md` §6.

The unlabeled pool is drawn from **training images that appear in no labeled
split**, so pretraining cannot have seen a validation or test image. Their
transcriptions exist but are deliberately never read.

---

## 4. IAM Handwriting Database

**Source:** <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>
**Size:** ~1.5 GB (lines) · **registration required** · free for non-commercial research

Download `lines.tgz` and `ascii.tgz`, extract both into `data/raw/iam/` so you
have:

```
data/raw/iam/lines/a01/a01-000u/a01-000u-00.png
data/raw/iam/ascii/lines.txt
```

```bash
python scripts/prepare_dataset.py --dataset iam --source data/raw/iam
```

`lines.txt` is whitespace-delimited with the transcription as the last field
and `|` as the word separator. Lines the IAM authors marked `err` (segmentation
errors) are skipped, which is standard practice.

IAM ships without a canonical train/val/test split for this use, so the
importer creates an 80/10/10 split from a seeded shuffle.

> **Status: implemented but unexercised.** No copy of IAM was available on the
> development machine, so this importer has not been run against real data.

### Automatic IAM setup for final-model training

`TRAIN_MODELS.bat` uses the line-level `Teklia/IAM-line` dataset from Hugging
Face and writes `data/processed/iam/` plus `data/processed/combined_iam/`.
It combines IAM with `data/processed/combined/` when available, or falls
back to the generated `data/processed/synthetic/` dataset. Generate that
fallback first if needed:

```bash
python scripts/make_synthetic_data.py
```

The standalone preparation command makes the same choice automatically:

```bash
python scripts/prepare_iam_hf.py
```

If preparation fails, check `artifacts/logs/prepare_iam.log` for the actual
cause; a missing base manifest is a local dataset setup issue, not an internet
connection error.

---

## 5. IIIT 5K-Word

**Source:** <https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset>
**Size:** ~100 MB · 5,000 cropped scene-text words

Extract into `data/raw/iiit5k/` so you have `traindata.mat`, `testdata.mat`
and the `train/` and `test/` image directories.

```bash
pip install scipy    # needed to read the .mat annotations
python scripts/prepare_dataset.py --dataset iiit5k --source data/raw/iiit5k
```

IIIT5K ships only train and test; the importer carves a 10 % validation split
off train so the trainer has something to early-stop on.

> **Status: implemented but unexercised.**

---

## 6. MJSynth and SynthText

**MJSynth (Synth90k):** <https://www.robots.ox.ac.uk/~vgg/data/text/> · ~10 GB · 9 M synthetic word images
**SynthText in the Wild:** <https://www.robots.ox.ac.uk/~vgg/data/scenetext/> · ~41 GB · 800 k scene images with word boxes

Both are far larger than this project needs; cap the import:

```bash
python scripts/prepare_dataset.py --dataset mjsynth --source data/raw/mjsynth --train 200000
```

> **Status: listed in the Datasets page with official links and preparation
> steps. The loaders are not implemented** — the two datasets above cover the
> printed and handwritten cases the synopsis requires, and a 41 GB download was
> not justified. Adding one means writing a function in
> `ml/datasets/external.py` that yields `(image_path, label)` pairs; the rest
> of the pipeline is unchanged.

---

## 7. Validating a prepared dataset

Every importer runs `validate_manifest` and prints a report. You can also do it
directly:

```python
from ml.datasets.manifest import validate_manifest
from ml.text.charset import Charset
from ml.config import load_config

cfg = load_config("default.yaml")
report = validate_manifest(
    "data/processed/handwriting/train.tsv",
    Charset.from_config(cfg.charset),
    max_label_length=37,
)
print(report.summary())
```

It reports missing files, empty labels, characters outside the charset, and —
importantly — labels longer than the CTC time-step budget. An over-long label
makes the loss `inf`, which otherwise surfaces as a mysteriously diverging
training run rather than as a data problem.

The API exposes the same information at `GET /api/datasets/{name}?split=train`,
and the Datasets page renders it with a warning banner when unsupported
characters are present.

---

## 8. Using a dataset in the experiment

Point a config's `data.*` entries at it. The `hw_*` configs are a worked
example:

```yaml
extends: default.yaml
name: hw_baseline_scratch

data:
  train_manifest: data/processed/handwriting/train.tsv
  val_manifest: data/processed/handwriting/val.tsv
  test_manifest: data/processed/handwriting/test.tsv
  unlabeled_manifest: data/processed/handwriting/unlabeled.txt
  max_train_samples: 2000
```

Then run the three stages as usual. Nothing else changes.

---

## 9. Charset

84 CTC classes: digits, lowercase, uppercase, common punctuation and space,
plus the blank at index 0.

The charset is **stored inside every checkpoint**. Loading a model with a
mismatched charset would silently produce plausible-looking nonsense, so the
mapping travels with the weights rather than coming from whatever config
happens to be on disk.

To change it, edit `charset.characters` in `configs/default.yaml` — and retrain,
because the output layer size changes.
