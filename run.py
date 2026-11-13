#!/usr/bin/env python3
"""Efficient Text Recognition - the one file you run.

    python run.py                 extract dataset (first time), ask whether to
                                  train again, then open the web app
    python run.py --train         retrain from scratch without asking
    python run.py --no-train      never retrain (use the saved model)
    python run.py --image a.png   just print the text found in an image
    python run.py --epochs 10     shorter / longer training
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_packages() -> None:
    """Install torch / numpy / Pillow with pip if they are missing."""
    missing = [m for m in ("torch", "numpy", "PIL") if importlib.util.find_spec(m) is None]
    if not missing:
        return
    print(f"Installing required packages ({', '.join(missing)}) - one time only ...", flush=True)
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if "torch" in missing and sys.platform != "darwin" and shutil.which("nvidia-smi"):
        # NVIDIA GPU present: prefer the CUDA build, fall back to the default one below
        subprocess.call(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu128"])
    if subprocess.call(pip + ["-r", str(ROOT / "requirements.txt")]) != 0:
        raise SystemExit("Could not install the requirements. Run:  pip install -r requirements.txt")
    importlib.invalidate_caches()


def ask_yes_no(question: str) -> bool:
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:                                   # not an interactive terminal
        print("n")
        return False


def describe(info: dict) -> str:
    m = info.get("metrics", {})
    return (f"epoch {info.get('epoch', '?')}, validation character error rate: "
            f"handwritten {m.get('handwritten', {}).get('cer', 0):.1%}, "
            f"printed {m.get('printed', {}).get('cer', 0):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Efficient text recognition (CRNN + CTC, trained from scratch)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="retrain from scratch without asking")
    group.add_argument("--no-train", action="store_true", help="use the saved model without asking")
    ap.add_argument("--epochs", type=int, help="training epochs (default: see ocr/config.py)")
    ap.add_argument("--image", help="recognise this image in the terminal and exit")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9 or newer is required.")
    ensure_packages()

    import torch
    from ocr import config
    from ocr.dataset import ensure_extracted
    from ocr.model import load_model

    ensure_extracted()                                 # unzip data/dataset.zip once

    retrain = args.train
    if not config.MODEL_PATH.exists():
        if args.no_train:
            raise SystemExit(f"No trained model at {config.MODEL_PATH}; run without --no-train.")
        print("No trained model found - training one now.")
        retrain = True
    elif not (args.train or args.no_train or args.image):
        _, info = load_model(config.MODEL_PATH)
        print(f"Trained model found ({describe(info)}).")
        retrain = ask_yes_no("Train again from scratch on the dataset images?")

    if retrain:
        from ocr.train import train
        train(epochs=args.epochs or config.EPOCHS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_model(config.MODEL_PATH, device)
    print(f"Model ready on {device} ({describe(info)}).")

    if args.image:
        from ocr.predict import recognize
        result = recognize(model, args.image, device)
        print(result["text"] or "(no text found)")
        return

    from ocr.server import serve
    serve(model, info, device, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
