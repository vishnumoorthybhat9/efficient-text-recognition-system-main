# Handwritten name OCR: CNN + Transformer (CTC)

Reads the text in an image of a handwritten name. The model combines a CNN
(for local stroke features) with a Transformer encoder (for context across the
whole word), and is trained with CTC loss so no per-character alignment is needed.

## Files

| File | Purpose |
|---|---|
| `ocr_utils.py` | Data loading, image preprocessing, augmentation, CTC decoding, CER metric (no torch needed) |
| `ocr_model.py` | `OCRNet`: CNN backbone + positional encoding + Transformer encoder + CTC head |
| `train.py` | Training, per-epoch validation (CER), best-checkpoint saving, final test score |
| `predict.py` | Read text from one or more images, or a whole folder |

## Setup

```bash
pip install torch torchvision opencv-python-headless numpy pandas
```

Put the data in this layout (it matches `C:\EFFICIENT\data`):

```
data/
  written_name_train_v2.csv
  train_v2/train/*.jpg
  written_name_validation_v2.csv
  validation_v2/validation/*.jpg
  written_name_test_v2.csv
  test_v2/test/*.jpg
```

## Train

```bash
python train.py --data-root data --out-dir runs/exp1 --epochs 30 --batch-size 128
```

Use a GPU if you have one (the script picks CUDA automatically and uses mixed precision).
On Google Colab, select a GPU runtime and point `--data-root` at your Drive copy of `data/`.
A quick check that the pipeline runs end to end:

```bash
python train.py --data-root data --out-dir runs/smoke --epochs 1 --limit 2000
```

Outputs in `--out-dir`:
- `best.pt`: best checkpoint by validation CER (weights, charset, config)
- `charset.json`: the character set, needed for decoding
- `history.json`: per-epoch losses, CER and exact-match accuracy
- `results.json`: test CER and exact-match with the best checkpoint

## Predict

```bash
python predict.py --ckpt runs/exp1/best.pt --image my_name.jpg
python predict.py --ckpt runs/exp1/best.pt --image-dir some_folder --json out.json
```

## Model summary

| Stage | Shape |
|---|---|
| Input (grayscale, resized to height 64, padded to width 384) | 1 × 64 × 384 |
| CNN backbone (7 conv blocks, 4 max-pools: all halve the height, the first 2 also halve the width; a final 4×1 conv collapses the height) | 256 × 1 × 96 |
| Sequence (96 steps, one per horizontal position) + sinusoidal positions | 96 × 256 |
| Transformer encoder (3 layers, 4 heads, pre-norm) | 96 × 256 |
| Linear + log-softmax over 50 characters + blank | 96 × 51 |
| CTC loss / greedy decoding | text |

## Metrics

- **CER** (character error rate): edit distance divided by the reference length. Lower is better.
- **Exact match**: fraction of names predicted exactly right.

## Notes

- Image size: the data images are about 45–50 px tall and 285–325 px wide, so every image is
  resized to height 64 and padded to width 384. Anything wider than 384 px is squeezed so no text is cut off.
- Augmentation (training only): small rotation, slant (shear), scale and contrast/brightness jitter.
- Labels with empty or NaN values are skipped (about 565 train rows).
- The CTC output has 96 steps, which is enough for names up to 34 characters.
- Unseen characters in validation or test are reported at the start of training. The model cannot predict them.
