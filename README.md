# Efficient Text Recognition from Images Using Deep Learning and Self-Supervised Learning

**B.Tech Major Project · Project Code 42 · Department of Information Science & Engineering**

A complete OCR system that tests one question properly: **can self-supervised
pretraining on unlabeled images recover the accuracy that deep OCR normally
needs a large labeled corpus to reach?**

The answer, measured on this machine, is yes — and by a wide margin.

| Dataset | From scratch (CER) | SimCLR pretrained (CER) | Relative reduction | Held-out samples |
|---|---:|---:|---:|---:|
| Real handwriting | 0.4339 | **0.1033** | **76.2 %** | 2,998 |
| Synthetic (printed + handwriting fonts) | 0.3377 | **0.1152** | **65.9 %** | 1,000 |

Both arms are the same architecture with the same 10.85 M parameters, trained
on the same 2,000 labeled samples with the same seed, augmentation, optimiser
and schedule. The *only* difference is whether the CNN encoder started from
SimCLR pretraining on unlabeled images.

### The model the application actually uses

The two arms above are deliberately starved of labels so the effect is
measurable. They are **not** the right model for reading a real document, so a
production model is trained separately on everything available:

| Model | Train set | CER | WER | Exact match | ms/img |
|---|---:|---:|---:|---:|---:|
| **CRNN+Attention (full labeled data)** | 27,992 printed + handwritten | **0.0406** | **0.1631** | **0.8352** | 2.78 |

Measured on 3,998 held-out combined test samples. This is the default in the
model dropdown; `crnn_full` is a *reference* model and is excluded from the
controlled comparison.

![Effect of self-supervised pretraining](artifacts/figures/simclr_effect_handwriting.png)

---

## Table of contents

1. [What it does](#what-it-does)
2. [**Setting up on a new machine**](#setting-up-on-a-new-machine)
3. [Using it](#using-it)
4. [Reproducing the experiment](#reproducing-the-experiment)
5. [Results](#results)
6. [Architecture](#architecture)
7. [Datasets](#datasets)
8. [Project layout](#project-layout)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Honest limitations](#honest-limitations)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

---

## What it does

A React dashboard on a FastAPI backend, with the deep-learning stack in a
standalone `ml/` library.

- **Recognise** — drag in an image, get text with per-line boxes, confidence,
  and a full timing breakdown. Copy or download as `.txt` / `.json`.
- **Preprocessing Lab** — every enhancement stage rendered as an image, so the
  controls can be judged rather than taken on trust.
- **Compare models** — one image through three fixed columns: the available
  CNN-based recogniser, a CNN+Transformer hybrid, and Mamba, with identical
  preprocessing.
- **Evaluation** — CER, WER, exact match, latency, throughput, memory and
  parameter counts on a held-out split, plus the controlled SimCLR comparison.
- **Training** — run OCR training jobs from the browser, with live progress
  and working cancellation. Nothing trains unless you ask it to.
- **History** — every recognition stored in SQLite; attach a ground truth
  afterwards to score it.
- **Datasets** — what is prepared locally, and official download instructions
  for IAM, IIIT5K, MJSynth and SynthText.

Every model carries a **provenance** badge: `trained in this project` or
`pretrained (third party)`. A downloaded baseline's numbers are never
presented as a result of this work.

---

## Setting up on a new machine

### What a fresh clone contains — and what it does not

The repository holds **source, configuration, documentation and 29 test
images**. It deliberately does **not** hold:

| Not in the repo | Why | How you get it |
|---|---|---|
| Trained models (`artifacts/checkpoints/`) | ~124 MB each, over GitHub's 100 MB file limit | Train them (step 4), or use TrOCR which downloads itself |
| Datasets (`data/processed/`) | ~1 GB of generated images | Generate in ~2 minutes (step 4) |
| `artifacts/ocr.db` | Your own local history | Created automatically on first run |
| `.venv/`, `node_modules/` | Environment, not source | Steps 2a and 2b |

So a fresh clone **cannot recognise anything until you either train a model or
let it download TrOCR**. The fastest working path is in step 3.

### Requirements

| | Minimum | Developed on |
|---|---|---|
| Python | 3.11 | 3.13.7 |
| Node.js | 18 | 24.22.0 |
| Disk | ~6 GB free | — |
| GPU | none (CPU works) | RTX 2050, 4 GB |

Everything runs on CPU; training is several times slower and inference is
roughly 10–20x slower, but nothing is GPU-only.

---

### 1. Clone

```bash
git clone https://github.com/pranavakumar01/efficient-text-recognition-system.git
```

```bash
cd efficient-text-recognition-system
```

### 2a. Python environment

```bash
python -m venv .venv
```

Then install PyTorch — **pick the line that matches your machine**:

```bash
.venv/Scripts/python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The first is the CUDA build (~2.7 GB, needs an NVIDIA GPU and driver 525+);
the second is CPU-only and far smaller. Then the rest:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> **On Linux or macOS**, every `.venv/Scripts/python.exe` in this README
> becomes `.venv/bin/python`.

> **Disk note.** The CUDA wheel and the Hugging Face model cache are large.
> The project redirects its caches into `./.cache/` automatically, so nothing
> lands on your system drive.

Check it worked:

```bash
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

If CUDA is unavailable even though Windows can see the NVIDIA GPU, the
environment may contain a newer CPU-only PyTorch wheel. Re-run the CUDA
install command above; `--force-reinstall` is needed to replace that wheel.

### 2b. Frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 3. Fastest path to seeing it work (no training)

TrOCR downloads itself on first use, so you can extract text immediately:

```bash
.venv/Scripts/python.exe scripts/extract_text.py data/test_images/03_page_clean_01.png --model trocr_base
```

The first run downloads ~1.3 GB and takes a few minutes; afterwards it is
cached. Use `--model trocr_handwritten` for a ~250 MB alternative that is
slightly less accurate but much quicker to fetch.

To run the web application instead, skip to step 5 — the Preprocessing Lab,
Datasets and Training pages all work with no trained model.

### 4. Train your own models (optional, reproduces the research result)

First generate the dataset — this needs no download, it renders text from the
fonts already installed on your machine:

```bash
.venv/Scripts/python.exe scripts/make_synthetic_data.py
```

Then the three experiment stages:

```bash
.venv/Scripts/python.exe scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml
```

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml
```

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml
```

```bash
.venv/Scripts/python.exe scripts/evaluate.py --all --split test
```

Measured on the development GPU (RTX 2050, 4 GB):

| Step | Time |
|---|---|
| Generate the dataset | ~2 min |
| SimCLR pretraining (12 epochs) | ~9 min |
| Control arm (25 epochs) | ~4 min |
| Treatment arm (25 epochs) | ~3 min |
| Evaluate both | ~1 min |

Expect roughly 5–10x longer on CPU. Add `--preset quick` to
`make_synthetic_data.py` and `--epochs 3` to the training commands for a fast
smoke test that exercises the whole pipeline without producing meaningful
accuracy.

### 5. Run the application

**On Windows, double-click `START.bat`** — it launches both servers and opens
your browser. `STOP.bat` shuts them down.

Otherwise, two terminals:

```bash
.venv/Scripts/python.exe scripts/run_api.py
```

```bash
cd frontend && npm run dev
```

Dashboard at <http://localhost:5173>, API docs at
<http://127.0.0.1:8000/docs>. The backend takes ~30 seconds to report
*Application startup complete* — start it before the frontend.

### 6. Verify the install

```bash
.venv/Scripts/python.exe -m pytest -q
```

299 tests should pass. Tests that need a trained model skip themselves
automatically, so this passes on a fresh clone too.

### Common first-run problems

| Symptom | Cause and fix |
|---|---|
| `No module named torch` | PyTorch not installed, or you used system `python` instead of `.venv/Scripts/python.exe` |
| `No recognition model is available` | Expected on a fresh clone. Train one (step 4) or use `--model trocr_base` |
| `Manifest not found` | Run `scripts/make_synthetic_data.py` first |
| `Cannot reach the backend` | The Python server is not running — step 5 |
| `address already in use` | A previous run is still going. `STOP.bat`, or kill whatever holds ports 8000 and 5173 |
| TrOCR fails with a tokenizer error | `pip install protobuf` — transformers 5 needs it |
| CUDA out of memory | Lower `train.batch_size`, or set `OCR_DEVICE=cpu` in `.env` |
| `python` not found on Windows | Use `py -3` instead, or add Python to PATH during installation |

---

## Using it

### Try it on the bundled test images

```bash
.venv/Scripts/python.exe scripts/make_test_images.py
```

This gathers 29 images into `data/test_images/` with an `EXPECTED.md` listing
the exact text of each one, so the output can be judged rather than guessed
at: printed lines, real photographed handwriting, full document pages at
three capture qualities, and four deliberately hard cases.

### Just get the text out of an image

No server or browser needed:

```bash
.venv/Scripts/python.exe scripts/extract_text.py page.png
```

```bash
.venv/Scripts/python.exe scripts/extract_text.py page.png --model trocr_base
```

Other things it does:

| Command | Effect |
|---|---|
| `scripts/extract_text.py --model list` | show which models are installed |
| `scripts/extract_text.py scans/ --save-next-to-image` | write `<image>.txt` beside every image in a folder |
| `scripts/extract_text.py scans/*.png --out all.txt` | everything into one file |
| `scripts/extract_text.py page.png --json` | structured output, for scripting |
| `--single-line` | the image is one cropped line; skip line detection |
| `--enhance` | contrast-enhance, for dim photographs |
| `--beam` | beam-search decoding, slightly more accurate |

`crnn_full` is this project's own recogniser and is fast (milliseconds per
line). `trocr_base` is the most accurate on arbitrary documents but takes a
few seconds per page and downloads ~1.3 GB the first time.

### Run the full application

**Easiest — double-click `START.bat`.** It opens the backend and frontend in
their own windows, waits for the backend to finish loading, and opens your
browser. `STOP.bat` shuts everything down if a port is left occupied.

Or start the two servers by hand, in two separate terminals.

**Windows `cmd`** (note the **back**slashes):

```bat
cd /d "D:\MAJORR PROJECTT" && .venv\Scripts\python.exe scripts\run_api.py
```

```bat
cd /d "D:\MAJORR PROJECTT\frontend" && npm run dev
```

**Git Bash / PowerShell / macOS / Linux** (forward slashes):

```bash
.venv/Scripts/python.exe scripts/run_api.py
```

```bash
cd frontend && npm run dev
```

Dashboard at <http://localhost:5173>, interactive API docs at
<http://127.0.0.1:8000/docs>.

The backend takes about 30 seconds to report *Application startup complete* —
it is scanning for checkpoints. Start it before the frontend.

The app is usable immediately: the Preprocessing Lab, Datasets and Training
pages work with no trained model. Recognition needs either a trained model
(next section) or `transformers` installed for the TrOCR baseline.

---

## Reproducing the experiment

Three stages. On the development GPU (RTX 2050, 4 GB) the whole sequence takes
about 25 minutes.

```bash
.venv/Scripts/python.exe scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml
```

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/experiments/baseline_scratch.yaml
```

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/experiments/simclr_finetune.yaml
```

```bash
.venv/Scripts/python.exe scripts/evaluate.py --all --split test
```

```bash
.venv/Scripts/python.exe scripts/compare_models.py
```

The last command writes `artifacts/reports/comparison.md` and report-ready
PNGs into `artifacts/figures/`.

### Compare the CNN and Mamba recognisers

The Mamba OCR model uses the same CNN image encoder, training data, CTC loss,
and decoder as the CRNN baseline, replacing its BiLSTM/attention sequence
layers with bidirectional selective state-space layers. It is implemented in
PyTorch and does not require the CUDA-only `mamba-ssm` package.

To make a controlled comparison, train both models with the same options and
seed, then evaluate them on the same held-out split:

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/default.yaml --run-name crnn_compare --max-samples 1000 --max-val-samples 200 --epochs 5
```

```bash
.venv/Scripts/python.exe scripts/train_mamba.py --run-name mamba_compare --max-samples 1000 --max-val-samples 200 --epochs 5
```

```bash
.venv/Scripts/python.exe scripts/evaluate.py --models crnn_compare mamba_compare --dataset synthetic --split test
```

```bash
.venv/Scripts/python.exe scripts/compare_ocr_models.py --models crnn_compare mamba_compare --dataset synthetic --split test --max-samples 200
```

Each evaluation writes a JSON report under `artifacts/reports/` with CER,
WER, exact-match accuracy, latency and example predictions against the same
ground-truth samples. The capped training run above is a quick comparison;
the comparison command also writes aligned reference/CRNN/Mamba outputs to
`artifacts/reports/recognizer_comparison.json`. Use the full training split and
longer schedules for final results.

### Train the CNN + Transformer hybrid

The hybrid uses the project's CNN visual encoder, a Transformer sequence
encoder, and the same CTC objective. Train it on a labeled OCR manifest with:

```bash
.venv/Scripts/python.exe scripts/train_crnn.py --config configs/cnn_transformer.yaml --run-name cnn_transformer --epochs 20
```

Its checkpoint appears in the second column of the Compare page after training.
Handwritten PDF page images without line transcriptions are not supervised
training data; prepare labeled line crops before using them to train or to
claim handwriting accuracy.

### What makes it a controlled experiment

`baseline_scratch.yaml` and `simclr_finetune.yaml` are identical except for
`model.init_encoder_from`. Everything that could otherwise explain a
difference is pinned:

| Held constant | How |
|---|---|
| Architecture | Same class, same config block, same 10,850,900 parameters |
| Labeled data | `max_train_samples: 2000` with the same `seed`, and the subset is a *seeded permutation*, so both arms see the identical 2,000 images |
| Augmentation | Same `augment` block |
| Optimiser & schedule | Same AdamW, learning rate, cosine schedule, epoch count |
| Evaluation | Same held-out split, same decoder, same batch size |

`tests/test_config.py` asserts this: it loads both configs and fails if they
differ anywhere except the encoder initialisation. If someone later tweaks one
arm's learning rate, the test breaks rather than the result quietly becoming
meaningless.

The unlabeled pool is also kept **disjoint from the evaluation splits** — it is
rendered (or drawn) from images that appear in no labeled split, so pretraining
cannot have seen a validation or test image.

---

## Results

All figures below were produced by running the models on this machine
(RTX 2050, 4 GB, CUDA 12.8). Nothing is quoted from a paper.

### Real handwriting — 2,998 held-out samples

| Model | Trained on | CER | WER | Exact | ms/img | Params |
|---|---|---:|---:|---:|---:|---:|
| **CRNN+Attention (SimCLR pretrained)** | handwriting | **0.1033** | **0.3776** | **0.6364** | 1.33 | 10.85 M |
| CRNN+Attention (from scratch) | handwriting | 0.4339 | 0.9312 | 0.0707 | 2.05 | 10.85 M |
| Transformer OCR — TrOCR small *(external baseline)* | SROIE / IAM | 1.6162 | 2.4748 | 0.0017 | 28.29 | 61.60 M |

### Synthetic — 1,000 held-out samples

| Model | Trained on | CER | CER (case-ins.) | Exact | ms/img | Params |
|---|---|---:|---:|---:|---:|---:|
| **CRNN+Attention (SimCLR pretrained)** | synthetic | **0.1152** | 0.1115 | **0.6090** | 1.30 | 10.85 M |
| CRNN+Attention (from scratch) | synthetic | 0.3377 | 0.3382 | 0.1630 | 1.36 | 10.85 M |
| Transformer OCR — TrOCR small *(external baseline)* | SROIE / IAM | 0.6342 | **0.0505** | 0.2620 | 27.24 | 61.60 M |

### Reading the TrOCR baseline fairly

TrOCR's strict CER looks poor, and it would be dishonest to leave it there.
Two things are going on, neither of which is "TrOCR is a bad model":

1. **Case convention.** TrOCR-printed was fine-tuned on SROIE receipts, which
   are all upper-case, so it outputs upper-case. Its strict CER on the
   synthetic set is 0.6342, but its **case-insensitive CER is 0.0505** — it
   reads the text almost perfectly and is being charged for a convention.
   This is why the project reports both.
2. **Annotation convention.** The handwriting images are photographed form
   fields that include a printed label (`NOM :`). The ground truth is only the
   handwritten name. TrOCR transcribes everything it sees — `'now : LORD'` for
   a reference of `'LORAS'` — because it has no way to know the convention.
   The project's own models learned it from 2,000 labeled examples.

TrOCR is a genuinely strong model being used out of its domain. The honest
conclusion is **not** "our 10.85 M model beats a 61.6 M transformer"; it is
that a small task-adapted model trained on a few thousand in-domain labels
beats a large general model used zero-shot, at ~20× the speed.

### Self-supervised pretraining

![SimCLR pretraining](artifacts/figures/simclr_pretraining.png)

Contrastive accuracy — the fraction of augmented views matched to their correct
pair — rose from 0.30 to **0.996** against a chance level of 0.0079, with
held-out accuracy tracking it at 0.9957. The encoder learned genuine
invariances rather than memorising crops.

### Training behaviour

![Validation CER during training](artifacts/figures/training_curves.png)

The pattern is the same on both datasets: the from-scratch arms (orange) sit at
CER ≈ 1.0 for ten or more epochs — the model emits only blanks while CTC finds
its alignment — then descend slowly. The pretrained arms (blue) escape that
plateau almost immediately, because the encoder already produces useful
features and only the recognition head has to be learned.

**That is the mechanism**: self-supervised pretraining does not merely give a
better final number, it removes the hardest part of low-label CTC training.

---

## Architecture

```
image ─► preprocessing ─► line segmentation ─► recogniser ─► CTC decode ─► text
         (OpenCV)         (projection profile)
```

### The recogniser (model A)

```
image (B, 1, 32, 256)
  → CNNEncoder                 → (B, 63, 512)   visual column features
  → BiLSTM × 2                 → (B, 63, 512)   local bidirectional context
  → Multi-head self-attention  → (B, 63, 512)   global context, length preserving
  → Linear                     → (B, 63, 84)    per-frame class logits
  → CTC loss / CTC decoding
```

**Why self-attention and not an attention decoder.** A Bahdanau-style decoder
emits its own output sequence, whose length is unrelated to the input width —
incompatible with CTC, which needs one output frame per input frame in
monotonic order. Placing multi-head self-attention *between* the BiLSTM and the
CTC classifier keeps `T` unchanged while giving every frame a global receptive
field. `tests/test_models.py` asserts the length-preserving property.

### SimCLR pretraining (model C/D)

The same `CNNEncoder` class, wrapped with a projection head and trained with
NT-Xent on unlabeled crops. Because it is literally the same module, the
transfer into the recogniser is an exact `state_dict` load — verified tensor by
tensor in `tests/test_models.py::test_simclr_weights_transfer_exactly`.

**OCR-aware augmentation.** The canonical SimCLR recipe (random resized crop +
colour jitter + horizontal flip) is wrong for text: flipping or aggressively
cropping a word changes what it says. This project substitutes blur, noise,
stroke erosion/dilation, JPEG artefacts, slant and mild width cropping —
nuisances a scanner or camera genuinely introduces. `horizontal_flip` is not
merely defaulted off; requesting it raises.

### Mamba OCR (model B)

The optional Mamba recogniser shares the CNN image encoder and CTC decoding
pipeline, using bidirectional selective state-space layers for sequence
context. Train it with `scripts/train_mamba.py`; its checkpoint is then
available to the application and comparison tools.

### CNN + Transformer OCR

`CNNTransformerOCR` uses the CNN visual encoder followed by a Transformer
sequence encoder and CTC classifier. Train it with
`configs/cnn_transformer.yaml`; it is a separate architecture from the
pretrained TrOCR model below.

### Transformer OCR (model C)

TrOCR (`microsoft/trocr-small-*`), loaded lazily on first use. Inference only:
4 GB of VRAM does not permit fine-tuning it at a useful batch size.

### Decoding

Greedy best-path and a real CTC prefix beam search, both in
`ml/decoding/ctc.py`. The collapse rule is tested explicitly — `[1,1,0,1]` must
decode to `"aa"`, not `"a"`, because the blank is what separates a genuine
double letter.

---

## Datasets

### Synthetic (bundled generator, no download)

`scripts/make_synthetic_data.py` renders text lines from the typefaces
installed on your system — 39 printed and 18 handwriting faces were found on
the development machine. Varied backgrounds, lighting, slant and per-character
jitter.

It **is** genuine training data: real glyph shapes, labels correct by
construction. It is **not** a substitute for IAM when reporting handwriting
results — the "handwritten" styles are script *typefaces*, more regular than
real cursive.

### Real datasets

```bash
.venv/Scripts/python.exe scripts/prepare_dataset.py --dataset handwriting --source "path/to/archive.zip" --train 20000 --val 3000 --test 3000 --unlabeled 30000
```

| Dataset | Source | Size | Notes |
|---|---|---|---|
| Kaggle Handwriting Recognition | [kaggle.com](https://www.kaggle.com/datasets/landlord/handwriting-recognition) | 1.3 GB | 413k handwritten name fields. Read straight from the zip. **Used for the headline result.** |
| IAM Handwriting | [fki.tic.heia-fr.ch](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | ~1.5 GB | Registration required. Line images + `ascii/lines.txt`. |
| IIIT 5K-Word | [cvit.iiit.ac.in](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) | ~100 MB | Cropped scene text, `.mat` annotations (needs SciPy). |
| MJSynth | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/text/) | ~10 GB | 9M synthetic words. |
| SynthText | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) | ~41 GB | Scene images with word boxes. |

Every importer reports exactly what it discarded and why — unreadable labels,
characters outside the charset, missing or corrupt images, labels too long for
the CTC time-step budget. On the Kaggle set it correctly dropped 888 rows the
annotators had marked `UNREADABLE`; training on those would teach the model to
output that literal word.

All datasets convert to one manifest format, so everything downstream is
identical regardless of source.

---

## Project layout

```
backend/            FastAPI application
  api/routes/       health, models, ocr, history, evaluation, datasets, training
  core/             settings (pydantic-settings), error handling
  db/               SQLAlchemy models and session management
  schemas/          pydantic request/response contracts
  services/         runtime, storage, ocr, history, evaluation, charts, training
ml/                 standalone deep-learning library (no backend imports)
  config.py         YAML loading with `extends` inheritance
  text/             charset and label encoding
  preprocessing/    OpenCV ops, pipeline, OCR-aware augmentation
  datasets/         synthetic renderer, external importers, manifests, Datasets
  models/           CNN encoders, attention, CRNN recogniser, SimCLR, TrOCR
  decoding/         CTC greedy + prefix beam search
  engine/           training loops
  evaluation/       CER/WER metrics, benchmarking
  inference/        line segmentation, predictors, model registry
frontend/           React 18 + Vite + Tailwind v4
configs/            default.yaml + experiment configs
scripts/            data prep, training, evaluation, reporting, server
tests/              pytest suite
artifacts/          checkpoints, logs, reports, figures, ocr.db
data/               generated and imported datasets, uploads, demo images
```

`ml/` never imports from `backend/`. The library is usable on its own, and the
API is one consumer of it.

---

## Configuration

Two separate concerns, deliberately kept apart:

- **`configs/*.yaml`** — ML hyperparameters. These belong to an experiment and
  must be versioned with it.
- **`.env`** (see `.env.example`) — deployment settings: host, port, device,
  upload limits, database URL. Every value has a working default, so `.env` is
  optional.

Experiment configs use single-key inheritance:

```yaml
extends: default.yaml
name: baseline_scratch
train:
  epochs: 25        # every other `train` key comes from default.yaml
```

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest
```

```bash
.venv/Scripts/python.exe -m pytest -m "not slow" -q
```

The suite covers charset encoding, CTC decoding, metrics, preprocessing,
segmentation, the model architectures, the SimCLR transfer, dataset handling,
config integrity and the HTTP API end to end. API tests run the real
application against a temporary database — nothing is mocked.

Three tests exist specifically to protect the research claim:

- `test_simclr_weights_transfer_exactly` — the transfer is verified tensor by
  tensor, not trusted from a log line.
- `test_experiment_arms_differ_only_in_the_encoder_initialisation` — fails if
  the two arms ever diverge in anything else.
- `test_max_samples_is_deterministic_across_instances` — both arms really do
  see the identical 2,000 labeled images.

Several bugs in this codebase were found by these tests rather than by
inspection, including a polarity check that could never fire and a skew
estimator blind to half its input range.

---

## Honest limitations

- **The synthetic corpus is rendered from system typefaces.** "Handwritten"
  samples are script fonts with per-character jitter — more regular than real
  cursive. The real-handwriting result is the one to quote.
- **The handwriting dataset is single-word name fields**, not full lines of
  prose. Results do not directly transfer to paragraph-level handwriting, and
  are not comparable to published IAM line-level numbers.
- **TrOCR is a zero-shot baseline here.** It was not fine-tuned on either
  dataset, and its weights are third-party. Its numbers belong to its authors.
- **4 GB of VRAM** ruled out fine-tuning TrOCR and capped SimCLR at 12 epochs.
  The contrastive objective was still improving when the budget ran out; a
  longer schedule on a larger GPU should widen the gap, not narrow it.
- **One seed per arm.** The effect is large (65–76 % relative) and consistent
  across two independent datasets, but no variance over seeds is reported.
- **English only**, as scoped in the synopsis.

---

## Troubleshooting

**`Cannot reach the backend`** — the Python server is not running. Start it
with `python scripts/run_api.py`.

**`No recognition model is available yet`** — nothing has been trained. Either
run the training stages, or install `transformers` to use the TrOCR baseline.

**TrOCR fails with a tokenizer error** — install `protobuf`
(`pip install protobuf`). transformers 5 needs it to convert TrOCR's
SentencePiece tokenizer.

**CUDA out of memory** — lower `train.batch_size`, or set `OCR_DEVICE=cpu`.
The comparison endpoint also refuses more than six models at once for this
reason.

**Manifest not found** — run `python scripts/make_synthetic_data.py` first.

**Only some lines of my document are recognised** — check that *Deskew* is on
(it is by default). A photographed page is rarely square, and even 3–4° of
rotation makes neighbouring text lines overlap vertically, which merges the
page into one unreadable row. The Preprocessing Lab shows the detected line
boxes, which makes this obvious immediately.

**Text is recognised but garbled** — make sure the selected model is
**CRNN+Attention (full labeled data)**. The two experiment arms
(`baseline_scratch`, `simclr_finetune`) are deliberately trained on only 2,000
labeled samples so the self-supervised effect is measurable; they are the
wrong choice for reading a real document. For photographs, also try enabling
*Enhance contrast (CLAHE)*.

**A long line comes back cut short** — this was a real bug and is fixed:
inference now sizes its canvas to the line rather than reusing the 256 px
training width. If you see it on a line longer than ~255 characters, that is
the architecture's ceiling (the attention stack holds 512 positions); split
the image or crop the line.

**Schema change after an update** — delete `artifacts/ocr.db`; it is recreated
on the next start. The project uses `create_all` rather than migrations, since
there is no deployed database to migrate.

---

## References

- B. Shi, X. Bai, C. Yao, *An End-to-End Trainable Neural Network for
  Image-Based Sequence Recognition* (CRNN), 2015.
- A. Graves et al., *Connectionist Temporal Classification*, ICML 2006.
- T. Chen et al., *A Simple Framework for Contrastive Learning of Visual
  Representations* (SimCLR), ICML 2020.
- M. Li et al., *TrOCR: Transformer-Based Optical Character Recognition with
  Pre-Trained Models*, arXiv:2109.10282.
- A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- C. Penarrubia, J. J. Valero-Mas, J. Calvo-Zaragoza, *Self-Supervised Learning
  for Text Recognition: A Critical Survey*, IJCV 2025.
- G. Kim et al., *OCR-Free Document Understanding Transformer* (Donut),
  arXiv:2111.15664.
- Y. Xu et al., *LayoutLMv2*, arXiv:2012.14740.

---

## Team

| Name | USN |
|---|---|
| Vishnumoorthy Raghavendra Bhat | NNM23IS215 |
| Tejaswi Shankar P | NNM23IS194 |
| Pranava Kumara K | NNM23IS132 |
| Simran Kasim Sayed | NNM23IS179 |

Guide: **Dr. Rashmi Naveen**
