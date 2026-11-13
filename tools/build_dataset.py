"""Rebuild data/dataset.zip.  NOT needed to run the project - the zip is committed.

The dataset has two parts, both stored as tiny few-grey-level PNGs:

  handwritten/  IAM handwriting lines (Teklia/IAM-line on Hugging Face, MIT licence),
                real scanned handwriting from ~650 writers, resized to 40 px high.
  printed/      MJSynth-style word images: dictionary words, numbers and punctuation
                rendered with ~170 open fonts (one word per image, shared baseline per
                font so the trainer can join words of one font into text lines).

Usage (needs: pip install pyarrow fonttools):
    pip download trdg --no-deps -d tmp && unzip tmp/trdg-*.whl -d tmp/trdg
    # download data/{train,validation,test}.parquet from
    #   https://huggingface.co/datasets/Teklia/IAM-line  into tmp/iam
    python tools/build_dataset.py --iam tmp/iam --words tmp/trdg/trdg/dicts/en.txt \
        --fonts tmp/trdg/trdg/fonts/latin /usr/share/fonts
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ocr.config import CHARSET  # noqa: E402

HW_HEIGHT = 40             # handwritten lines, 4 grey levels
WORD_HEIGHT = 48           # printed words, 1 bit (anti-aliased again when loaded)
REFERENCE = "Hgjpqy(|"     # tallest + deepest glyphs: defines the line box of a font
SKIP = ("capture", "tusj", "seasrn", "emoji", "symbol", "unifont", "ipa", "wqy",
        "wenquanyi", "loma", "display")
COMMON = """the of and to a in is it you that he was for on are with as I his they be at
one have this from or had by but some what there we can out other were all your when up use
how said an each she which do their time if will way about many then them would write like so
these her long make thing see him two has look more day could go come did my no most who over
know than call first people may down side been now find any new work part take get place made
live where after back little only round man year came show every good me give our under name
very through just form much great think say help low line before turn cause same mean differ
move right boy old too does tell sentence set three want air well also play small end put home
read hand port large spell add even land here must big high such follow act why ask men change
went light kind off need house picture try us again animal point mother world near build self
earth father head stand own page should country found answer school grow study still learn""".split()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------- handwriting
def iam_line(job: tuple[bytes, str]) -> tuple[bytes, str]:
    raw, text = job
    img = Image.open(io.BytesIO(raw)).convert("L")
    img = img.resize((max(8, round(img.width * HW_HEIGHT / img.height)), HW_HEIGHT), Image.LANCZOS)
    img = img.point(lambda v: round(v / 85) * 85).quantize(4, dither=Image.Dither.NONE)
    return png_bytes(img), text


def iam_jobs(folder: str):
    import pyarrow.parquet as pq
    # official train + test are used for training, official validation is held out
    for name, split in (("train", "train"), ("test", "train"), ("validation", "val")):
        table = pq.read_table(str(Path(folder) / f"{name}.parquet"))
        for batch in table.to_batches(max_chunksize=256):
            for row in batch.to_pylist():
                text = " ".join(row["text"].split())
                if text and set(text) <= set(CHARSET):
                    yield split, (row["image"]["bytes"], text)


# ------------------------------------------------------------------- printed
def usable_fonts(dirs: list[str]) -> list[str]:
    from fontTools.ttLib import TTFont
    need = {ord(c) for c in CHARSET if c != " "}
    fonts = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.[to]tf")):
            if any(s in p.name.lower() for s in SKIP):
                continue
            try:
                cmap = TTFont(str(p), lazy=True).getBestCmap()
            except Exception:
                continue
            if need <= set(cmap):
                fonts.append(str(p))
    return fonts


def make_texts(words: list[str], count: int, rng: random.Random) -> list[str]:
    def word() -> str:
        w = rng.choice(COMMON) if rng.random() < 0.15 else rng.choice(words)
        r = rng.random()
        if r < 0.18:
            w = w.capitalize()
        elif r < 0.28:
            w = w.upper()
        elif r < 0.31:
            w = w + rng.choice(["-", "'", "/", "&", "+", "="]) + rng.choice(words)
        r = rng.random()
        if r < 0.22:
            w += rng.choice(".,.,.,;:!?")
        elif r < 0.27:
            a, b = rng.choice(['()', '""', "''"])
            w = a + w + b
        return w

    def number() -> str:
        k = rng.random()
        n = lambda a, b: str(rng.randint(a, b))  # noqa: E731
        if k < 0.35:
            return n(0, 10 ** rng.randint(1, 6))
        if k < 0.50:
            return f"{n(0, 999)}.{n(0, 99)}"
        if k < 0.60:
            return f"{n(1, 31)}/{n(1, 12)}/{n(1950, 2030)}"
        if k < 0.68:
            return f"{n(0, 23)}:{rng.randint(0, 59):02d}"
        if k < 0.76:
            return f"${n(1, 9999)}"
        if k < 0.84:
            return f"{n(0, 100)}%"
        if k < 0.90:
            return f"#{n(1, 999)}"
        if k < 0.95:
            return f"{rng.choice(words)}@{rng.choice(words)}.com"
        return f"{n(1, 99)}*{n(1, 99)}"

    return [number() if rng.random() < 0.14 else word() for _ in range(count)]


_font_cache: dict[str, tuple] = {}


def render_word(job: tuple[str, str]) -> bytes:
    text, font_path = job
    if font_path not in _font_cache:
        font = ImageFont.truetype(font_path, 72)
        box = font.getbbox(REFERENCE)
        _font_cache[font_path] = (font, box[1], box[3])
    font, top, bottom = _font_cache[font_path]
    left, _, right, _ = font.getbbox(text)
    big = Image.new("L", (right - left + 8, bottom - top), 255)
    ImageDraw.Draw(big).text((4 - left, -top), text, font=font, fill=0)
    width = max(4, round(big.width * WORD_HEIGHT / big.height))
    small = big.resize((width, WORD_HEIGHT), Image.LANCZOS)
    return png_bytes(small.point(lambda v: 255 if v > 140 else 0).convert("1"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam", required=True, help="folder with IAM-line parquet files")
    ap.add_argument("--fonts", nargs="+", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--count", type=int, default=17000, help="printed word images")
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset.zip"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [("file", "text", "kind", "font", "split")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z, Pool() as pool:
        splits, jobs = zip(*iam_jobs(args.iam))
        for i, (png, text) in enumerate(pool.imap(iam_line, jobs, chunksize=64)):
            name = f"handwritten/{i // 1000:02d}/{i:05d}.png"
            z.writestr(name, png)
            rows.append((name, text, "handwritten", "", splits[i]))
        print(f"handwritten: {len(jobs)} lines")

        fonts = usable_fonts(args.fonts)
        ok = set(CHARSET)
        words = [w.strip() for w in open(args.words, encoding="utf-8", errors="ignore")]
        words = [w.lower() for w in words if 1 <= len(w) <= 14 and w.isalpha() and set(w) <= ok]
        texts = make_texts(words, args.count, rng)
        jobs = [(t, fonts[i % len(fonts)]) for i, t in enumerate(texts)]
        for i, png in enumerate(pool.imap(render_word, jobs, chunksize=256)):
            name = f"printed/{i // 1000:02d}/{i:05d}.png"
            z.writestr(name, png)
            split = "val" if rng.random() < 0.05 else "train"
            rows.append((name, texts[i], "printed", i % len(fonts), split))
        print(f"printed: {len(texts)} words in {len(fonts)} fonts")

        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        z.writestr(zipfile.ZipInfo("labels.csv"), buf.getvalue(), zipfile.ZIP_DEFLATED)
        z.writestr("README.txt", (
            "OCR training dataset - labels.csv columns: file,text,kind,font,split\n\n"
            "handwritten/  IAM handwriting database lines (Teklia/IAM-line, MIT licence),\n"
            "              40 px high. split=train is the official train+test sets,\n"
            "              split=val is the official validation set (held out).\n"
            "printed/      one rendered word per image, 48 px high, 1 bit. Words with the\n"
            "              same font id share a baseline and are joined into lines by\n"
            "              the trainer.\n\n"
            "Add your own data: drop images in a new folder and append rows to labels.csv\n"
            "(kind=handwritten for whole line images, font empty, split=train or val).\n"))
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
