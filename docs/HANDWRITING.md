# Handwritten documents

Photographed handwritten pages (notebook notes, letters, prescriptions) need
extra work before any recogniser can read them. The Recognize page and API
default to handwritten-document mode: page cleanup runs first, then the
detected lines are sent to the selected recogniser. Turn off **Handwritten
document** for printed pages, or pass `--handwritten` to
`scripts/extract_text.py`.

## Extract and transcribe a handwritten PDF

Extract PDF scans using the page placement transform so rotated embedded
images are saved upright. The default output is `data/handwritten/images/`:

```powershell
.venv\Scripts\python.exe scripts\prepare_handwritten_pdf.py "C:\path\to\handwritten.pdf" --force
```

Create editable OCR draft text and line crops for review:

```powershell
.venv\Scripts\python.exe scripts\transcribe_handwriting.py data\handwritten\images --model trocr_base
```

The script saves one `.txt` and `.json` file per page under
`data/handwritten/transcriptions/`, plus line crops and a combined
`draft_lines.tsv`. Existing page results are resumed by default. OCR drafts
are explicitly unverified and are kept outside `data/processed/`; review and
correct them before using any text for training or accuracy measurement.
The first TrOCR run downloads pretrained weights and requires internet access.

## Why the normal pipeline fails

On a real notebook photo (`data/test_images/real_notes/capital_budgeting_notes.jpg`)
the standard line detector returned **one region, the whole page**. Three
things cause that:

| Problem | Effect | Fix (`ml/inference/handwriting.py`) |
|---|---|---|
| Printed ruled lines and margin lines touch every word | The whole page becomes one ink component, so no gaps between lines | `remove_ruled_lines`: morphological detection of long thin rules, then inpainting |
| Shadows and uneven light in phone photos | One threshold erases faint ink or turns shadow into "ink" | `flatten_background`: divide by an estimated background |
| Desk or table visible at the edge | A huge dark blob that everything connects to | `whiten_dark_borders` |
| Hairlines (page edges underneath, binding) between rows | Fill the gaps the projection profile relies on | `remove_thin_strokes` |
| Descenders touching the next line | Two lines returned as one box | `split_tall_regions`: cut at the emptiest rows |

With handwriting mode on, the same photo segments into **29 separate lines**.

## Line detectors

| `line_detector` | What it is | Needs |
|---|---|---|
| `projection` | The project's own projection-profile detector, run on the cleaned page | nothing extra |
| `doctr` | DBNet neural text detector from [docTR](https://github.com/mindee/doctr); finds word boxes, which are chained into lines (`group_words_into_lines`) | `pip install python-doctr` (weights ~100 MB download on first use) |
| `auto` (default) | `doctr` when installed, otherwise `projection` | |

If docTR is requested but cannot load, the pipeline falls back to `projection`
and reports `"projection (docTR unavailable)"` in the response's
`segmentation.handwriting.line_detector`.

## Recognition

The line crops go to whichever model is selected, so the CNN-vs-Transformer
comparison stays fair. For handwriting, Transformer OCR trained on IAM lines
reads best:

* `trocr_finetuned` / `trocr_base` -- the app's Transformer model
* `trocr_large` -- `microsoft/trocr-large-handwritten`, the most accurate and
  slowest (~2.2 GB download). Hidden in the app by default; use it from the
  command line or add it to `models.visible` in `configs/default.yaml`.

The CNN model was trained mostly on single words and short lines, so expect
it to struggle on full cursive pages. That gap is a finding, not a bug.

The extracted `data/handwritten/images/` pages are unlabeled scans, not
supervised training samples. Training this project's recogniser on them, or
measuring its accuracy on them, requires ground-truth transcriptions paired
with the corresponding text-line crops. The pretrained TrOCR models can
recognise line crops without project-specific training, but handwriting,
image quality, and line-detection errors mean no OCR system can guarantee a
perfect transcription; review and correct the output.

## Usage

```bash
# command line
python scripts/extract_text.py notes.jpg --handwritten --model trocr_base
python scripts/extract_text.py notes.jpg --handwritten --detector doctr --model trocr_large

# API: options form field
{"handwritten": true, "line_detector": "auto"}
```

The Preprocessing Lab shows the cleaned page as an extra stage
("Handwriting clean-up") when handwriting mode is on.

## Tests

`tests/test_handwriting.py` (13 tests) builds a synthetic notebook page with
ruled lines, a margin, a shadow and a desk edge, and checks that every line is
found, that docTR failures fall back cleanly and that drifting word rows are
chained into the right lines.
