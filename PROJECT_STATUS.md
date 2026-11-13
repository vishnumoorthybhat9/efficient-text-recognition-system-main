# PROJECT STATUS

**Project:** Efficient Text Recognition from Images Using Deep Learning and Self-Supervised Learning
**Directory:** `D:\MAJORR PROJECTT`
**Last updated:** 2026-10-30

> Resume point. If development is interrupted, read this first, then continue
> from **Next steps**. Do not regenerate files listed as complete.

---

## 1. Summary

All twelve phases are implemented and the application runs end to end. The
research experiment was executed on **two independent datasets** and the result
reproduces on both.

| Dataset | From scratch (CER) | SimCLR pretrained (CER) | Relative reduction | Samples |
|---|---:|---:|---:|---:|
| Real handwriting | 0.4339 | **0.1033** | **76.2 %** | 2,998 |
| Synthetic | 0.3377 | **0.1152** | **65.9 %** | 1,000 |

**223 tests pass.**

---

## 2. Phase status

| Phase | Description | Status |
|---|---|---|
| 1 | Inspect synopsis, finalise architecture | **Complete** |
| 2 | Project folder structure | **Complete** |
| 3 | Backend, database, APIs | **Complete** — 34 endpoints, verified |
| 4 | Image preprocessing | **Complete, tested** |
| 5 | CNN + BiLSTM + Attention + CTC | **Complete, trained, evaluated** |
| 6 | SimCLR self-supervised learning | **Complete, run on both datasets** |
| 7 | Transformer OCR (TrOCR) | **Complete** — loads, runs, benchmarked |
| 8 | React frontend | **Complete** — 9 pages, builds clean |
| 9 | Frontend/backend integration | **Complete** — verified in browser |
| 10 | Model evaluation and comparison | **Complete** — 9 runs stored, reports + figures |
| 11 | Full application test pass | **Complete** — 223 tests pass |
| 12 | README and documentation | **Complete** — README, ARCHITECTURE, RESULTS, DATASETS, VIVA |

---

## 3. Environment (verified, not assumed)

| Item | Value |
|---|---|
| Python | 3.13.7 (`.venv` at project root) |
| Node.js / npm | v24.22.0 / 11.19.1 |
| GPU | NVIDIA GeForce RTX 2050, 4,095 MB, compute 8.6 |
| CUDA | driver 592.00, torch cu128, `cuda.is_available() == True`, bf16 supported |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| OpenCV | 5.0.0 (headless) |
| transformers | 5.22.1 (+ protobuf, required for TrOCR's tokenizer) |
| FastAPI / SQLAlchemy | 0.162.0 / 2.1.6 |
| CPU / RAM | i5-11400H (6C/12T) / 16 GB |

**Disk note:** `C:` had ~6 GB free, so pip/HF/temp caches are redirected to
`D:\MAJORR PROJECTT\.cache\`. `ml/__init__.py` sets `HF_HOME` automatically.

---

## 4. Trained artefacts on disk

| Run | Purpose | Result |
|---|---|---|
| `simclr_pretrain` | SimCLR on 11,400 unlabeled synthetic | contrastive acc 0.30 → 0.996 (chance 0.0079) |
| `baseline_scratch` | Control arm, synthetic | val CER 0.3406 |
| `simclr_finetune` | Treatment arm, synthetic | val CER 0.1179 |
| `hw_simclr_pretrain` | SimCLR on 28,481 unlabeled handwriting | contrastive acc 0.24 → 0.986 |
| `hw_baseline_scratch` | Control arm, handwriting | val CER 0.4362 |
| `hw_simclr_finetune` | Treatment arm, handwriting | val CER 0.1044 |
| `crnn_full` | **Production model** — all 27,992 printed + handwritten labels. Not part of the experiment | val CER 0.0575 · **test CER 0.0406, 83.5% exact** |

Evaluation runs are in `artifacts/ocr.db`; reports and figures in
`artifacts/reports/` and `artifacts/figures/`.

---

## 5. Datasets prepared

| Dataset | Train | Val | Test | Unlabeled | Source |
|---|---:|---:|---:|---:|---|
| `synthetic` | 8,000 | 1,000 | 1,000 | 12,000 | Generated from 39 printed + 18 handwriting system fonts |
| `handwriting` | 19,992 | 3,000 | 2,998 | 29,980 | Kaggle Handwriting Recognition (`D:/Downloads/archive (1).zip`) |

The unlabeled pools share no image file with any labeled split, so pretraining
cannot have seen an evaluation image.

---

## 6. Bugs found and fixed during development

Several were found by the test suite rather than by inspection.

| Problem | Resolution |
|---|---|
| **Long lines were silently truncated.** Inference reused the 32×256 *training* canvas, so anything wider was cropped. A 63-character sentence came back as 28 characters, looking like a complete result | Inference now sizes the canvas to the content (`prepare_variable_width_batch`). The network is width-agnostic, so no retraining was needed. Beyond a 2048 px cap it compresses rather than crops |
| `resize_with_padding` reported `content_ratio = 1.0` **after** cropping, so nothing downstream could detect the loss | Added `natural_width`, `truncated` and `lost_fraction` to `ResizeInfo` |
| **Table rows came out as separate lines.** `Laptop stand / 2 / 51.25` became three output lines | Columns stay separate *crops* (they recognise far better than one box spanning a blank gap), and `group_into_rows` rejoins them into one output line |
| Merging a whole table row into one box garbled recognition — the model never saw that much internal whitespace in training | Recognise tight fragments, assemble by layout afterwards |
| Two-column pages risked being joined into one line | `_detect_column_gutters` finds vertical bands empty across most rows and blocks joining across them; a table's gaps are *not* flagged because long lines cross them |
| **A skewed photo collapsed a whole page into one row** — rotated lines overlap vertically, so row grouping merged all of them. `deskew` defaulted to `false` | `deskew` now defaults to **true** everywhere (config, dataclass, API schema, UI). It is a no-op below 0.1°, so clean scans are unaffected |
| `stopwatch()` with no device issued a full `torch.cuda.synchronize()` per CPU stage — **22.4 ms/image** on the preprocessing hot path | `synchronize(None)` is now a no-op. **168× faster** (22.4 ms → 0.133 ms) |
| `normalise_polarity` compared border median to global median — a test that can *never* fire, since the background is the majority either way | Otsu-split the image and ask which class the border belongs to |
| `estimate_skew_angle` used `minAreaRect` and returned 0.0 for every **negative** rotation | Replaced with a projection-profile search; now accurate to ~0.2° in both directions |
| Sauvola binarisation shattered glyphs, collapsing the median component height so one cropped line segmented into 4 fragments | Row-projection profile is now the authority on line count |
| Checkpoints unloadable: `torch.__version__` is a `TorchVersion`, which `weights_only=True` rejects | Coerced to `str`, plus a narrow allowlist in `ml/utils/serialization.py` (protection kept on) |
| TrOCR crashed: `[..., ::-1]` makes a negative-stride array torch refuses | Use `cv2.cvtColor` |
| TrOCR looped emitting `000 000 …` to the token cap (CER 2.2142) | `no_repeat_ngram_size=3` → CER 1.6162 and 2× faster |
| TrOCR tokenizer failed to load on transformers 5 | Added `protobuf` dependency |
| Comparison could pair the synthetic control with the handwriting treatment | Arms are now grouped by `(dataset, split)` before pairing |
| `Charset` as a frozen **slotted** dataclass rejected its own lookup tables | Dropped `slots=True` |
| Beam search returned confidence 1.0 for an empty prediction; greedy returned 0.0 | Empty prefix scores 0.0 in both |
| Synthetic renderer dropped `background_kind` | Returned from `_draw` and recorded |
| Plotly figure titles collided with the legend and duplicated the card header | Figures render title-less; legend moved below |
| Scatter labels overflowed the plot area | Replaced with a legend |
| pydantic reserved the `model_` prefix, clashing with `model_id` / `model_name` | `protected_namespaces=()` on the base schema |

---

## 7. Known remaining items

Nothing is broken. These are scope decisions, documented in the README:

- **One seed per experiment arm.** The effect is large and reproduces across
  two datasets, but no variance over seeds is reported.
- **SimCLR capped at 12 epochs** by the 4 GB / 35 W GPU budget. The contrastive
  objective was still improving; a longer schedule should widen the gap.
- **TrOCR is zero-shot.** Fine-tuning it does not fit in 4 GB at a useful
  batch size.
- **IAM / IIIT5K / MJSynth / SynthText importers are written but unexercised** —
  no copy of those datasets is on this machine. `scripts/prepare_dataset.py`
  implements them and the Datasets page carries the official links.
- **Synthetic "handwriting" is script typefaces**, more regular than real
  cursive. The real-handwriting result is the one to quote.

---

## 8. How to run everything

```bash
# backend
.venv/Scripts/python.exe scripts/run_api.py

# frontend
cd frontend && npm run dev

# tests
.venv/Scripts/python.exe -m pytest -q
```

Full reproduction commands are in `README.md` and `docs/RESULTS.md`.

---

## 9. Next steps (optional extensions)

1. Repeat each arm over 3–5 seeds and report mean ± std.
2. Import IAM and rerun the experiment at line level, for numbers comparable
   to the published literature.
3. Longer SimCLR schedule on a larger GPU.
4. A label-efficiency curve: repeat the comparison at 500 / 1k / 2k / 5k / 10k
   labels to show where the pretraining advantage closes.
5. Fine-tune TrOCR on the handwriting train split for a like-for-like
   transformer comparison (needs >4 GB VRAM).
