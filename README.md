# Efficient Text Recognition

Reads **printed and handwritten** text from images with a small deep-learning
model (CNN + BiLSTM + CTC, 2.2 M parameters) that is **trained from scratch** on
the dataset shipped in this repository. No pretrained weights, no external
services, three Python packages.

## Run it

```bash
python run.py
```

That one command does everything:

1. installs `torch`, `numpy` and `Pillow` if they are missing,
2. extracts `data/dataset.zip` into `dataset/` (first run only),
3. asks **`Train again from scratch on the dataset images? [y/N]`**
   (answer `n` to use the trained model in `models/`, `y` to retrain),
4. opens the web app at <http://127.0.0.1:8000> - upload, paste or drop an
   image, click a sample, or write on the handwriting pad.

Other ways to run it:

| Command | What it does |
|---|---|
| `python run.py --train` | retrain from scratch without asking |
| `python run.py --no-train` | use the saved model without asking |
| `python run.py --epochs 10` | shorter (or longer) training |
| `python run.py --image photo.png` | print the text of one image and exit |
| `python run.py --port 9000 --no-browser` | different port / do not open the browser |

Training uses an NVIDIA GPU automatically when PyTorch can see one (about
15-20 minutes); on CPU only it takes a few hours, which is why a trained model
is included.

## Project layout

```
run.py                 the only file you run
requirements.txt       torch, numpy, Pillow
data/dataset.zip       the dataset (28 MB)
dataset/               extracted dataset - created by run.py, git-ignored
models/ocr_crnn.pt     trained model (4.5 MB) - overwritten when you retrain
samples/               example images shown in the web app
ocr/
  config.py            paths, character set, training defaults
  dataset.py           unzip, load, augment, build training batches
  model.py             the CRNN, CTC decoding, save / load
  train.py             training loop and evaluation
  predict.py           image clean-up, line splitting, recognition
  server.py            local web app (Python standard library)
  web/index.html       the page
tools/build_dataset.py how data/dataset.zip was built (not needed to run)
```

## Dataset

`data/dataset.zip` holds 27,373 small PNG images and one `labels.csv`
(`file,text,kind,font,split`):

| Part | Images | What it is |
|---|---:|---|
| `handwritten/` | 10,373 lines | **IAM Handwriting Database** lines (real handwriting by ~650 writers), from [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) (MIT licence), resized to 40 px height |
| `printed/` | 17,000 words | Dictionary words, numbers and punctuation rendered in 170 open fonts (the MJSynth recipe). The trainer joins words of one font into random text lines, so these become an unlimited supply of printed lines |

The images are stored with 2-4 grey levels, which is why 27 k images fit in
28 MB. Training adds slant, stretch, blur, stroke-thickness and noise
augmentation on the fly.

Held-out data used for the reported numbers: the official IAM validation set
(976 lines) and 5 % of the printed words. IAM's official train and test sets
are both used for training.

**Add your own images:** put them in a new folder inside `dataset/`, append
rows to `dataset/labels.csv` (`kind=handwritten` for whole-line images, `font`
empty, `split=train`), and run `python run.py --train`.

## Model

```
line image (40 px high, any width)
  -> 5 conv layers (32-64-96-128-192, batch norm, max pooling)
  -> 2-layer bidirectional LSTM (192 units)
  -> linear layer over 83 characters + CTC blank
  -> CTC best-path decoding
```

For a full image, `ocr/predict.py` evens out the background, inverts
light-on-dark text, splits the page into lines by their ink profile, and reads
each line.

## Results

RESULTS_PLACEHOLDER

## Limits

- Lines are found by horizontal ink profile, so it expects reasonably straight
  text in a single column. Multi-column pages, strongly rotated photos and
  text over busy backgrounds are not handled.
- English letters, digits and common punctuation only.
- Handwriting accuracy depends on the writing being roughly as legible as the
  IAM samples; very different styles need more training data of that style.
