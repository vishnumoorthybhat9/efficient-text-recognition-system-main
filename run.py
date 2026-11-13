#!/usr/bin/env python3
"""Efficient Text Recognition - the one file you run.

    python run.py                 start screen: pick a model, train, compare
    python run.py --arch mamba    use that model (lstm, transformer, mamba) in the web app
    python run.py --train         train the chosen --arch from scratch, then open the app
    python run.py --resume        continue the training run that was interrupted
    python run.py --compare       score every trained model on the same held-out lines
    python run.py --image a.png   print the text found in an image and exit
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


def ask(question: str, default: str = "") -> str:
    try:
        answer = input(f"{question} [{default}]: " if default else f"{question}: ").strip().lower()
    except EOFError:                                   # not an interactive terminal
        print(default)
        return default
    return answer or default


def ask_yes_no(question: str, default: bool = False) -> bool:
    return ask(question, "y" if default else "n") in ("y", "yes")


def describe(info: dict) -> str:
    m = info.get("metrics", {})
    return (f"errors per character: handwriting {m.get('handwritten', {}).get('cer', 0):.1%}, "
            f"printed {m.get('printed', {}).get('cer', 0):.1%}  (best epoch {info.get('epoch', '?')})")


TRAIN_TIME = "about 20-40 min on an NVIDIA GPU, several hours on a CPU"


def start_screen(config, load_model, device: str):
    """Interactive menu -> (arch, train_first) or None to quit."""
    while True:
        trained = {a: load_model(config.MODEL_PATHS[a])[1] for a in config.ARCHS
                   if config.MODEL_PATHS[a].exists()}
        print("\nEvery model = the same CNN (looks at the pixels) + a sequence model "
              "(reads the line in order) + CTC (turns it into letters).\n")
        for i, arch in enumerate(config.ARCHS, 1):
            state = describe(trained[arch]) if arch in trained else "not trained yet"
            if config.STATE_PATHS[arch].exists():
                state += f"\n{'':24}training unfinished - continue it with:  python run.py --resume --arch {arch}"
            print(f"  {i}  {config.ARCH_NAMES[arch]:<19}{state}")
        print("\n  1-3  open the web app with that model\n"
              "  t    train a model from scratch\n"
              "  c    compare all trained models side by side\n"
              "  q    quit\n")
        def errors(a):                                # lower = more accurate model
            m = trained[a].get("metrics", {})
            return sum(m.get(k, {}).get("cer", 1.0) for k in ("handwritten", "printed"))
        default = str(config.ARCHS.index(min(trained, key=errors)) + 1) if trained else "t"
        choice = ask("Your choice", default)
        if choice == "q":
            return None
        if choice == "c":
            from ocr.train import compare
            print("\nScoring every trained model on the same held-out lines ...")
            compare(device)
            continue
        if choice == "t":
            choice = ask("Train which model? 1, 2 or 3")
            if choice not in ("1", "2", "3"):
                continue
            arch = config.ARCHS[int(choice) - 1]
            if arch in trained and not ask_yes_no(
                    f"This replaces the trained {config.ARCH_NAMES[arch]} model. Continue?"):
                continue
            return arch, True
        if choice in ("1", "2", "3"):
            arch = config.ARCHS[int(choice) - 1]
            if arch in trained:
                return arch, False
            if ask_yes_no(f"{config.ARCH_NAMES[arch]} is not trained yet ({TRAIN_TIME}). "
                          "Train it now?", default=True):
                return arch, True
            continue
        print("Please type 1, 2, 3, t, c or q.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Efficient text recognition: CNN + (BiLSTM | "
                                 "Transformer | Mamba) + CTC, trained from scratch")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="train --arch from scratch without asking")
    group.add_argument("--no-train", action="store_true", help="never train; skip the start screen")
    group.add_argument("--resume", action="store_true", help="continue an interrupted training run")
    group.add_argument("--compare", action="store_true", help="compare all trained models and exit")
    ap.add_argument("--arch", choices=("lstm", "transformer", "mamba"),
                    help="sequence model after the CNN (default: lstm)")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.compare:
        from ocr.train import compare
        compare(device)
        return

    arch, retrain = args.arch or "lstm", args.train or args.resume
    if args.resume and not args.arch:                  # resume whichever run was interrupted
        pending = [a for a in config.ARCHS if config.STATE_PATHS[a].exists()]
        if not pending:
            raise SystemExit("Nothing to resume: no interrupted training run was found.")
        arch = pending[0]
    flags_given = args.arch or args.train or args.resume or args.no_train or args.image
    if not flags_given and sys.stdin.isatty():
        print(f"Efficient Text Recognition  (running on {device})")
        picked = start_screen(config, load_model, device)
        if picked is None:
            return
        arch, retrain = picked

    model_path = config.MODEL_PATHS[arch]
    if not model_path.exists() and not retrain:
        if args.no_train or args.image:
            raise SystemExit(f"{config.ARCH_NAMES[arch]} is not trained yet; run:  "
                             f"python run.py --train --arch {arch}")
        print(f"{config.ARCH_NAMES[arch]} is not trained yet - training it now ({TRAIN_TIME}).")
        retrain = True

    if retrain:
        from ocr.train import train
        train(epochs=args.epochs or config.EPOCHS, resume=args.resume, arch=arch)

    model, info = load_model(model_path, device)
    print(f"{config.ARCH_NAMES[arch]} ready on {device} - {describe(info)}")

    if args.image:
        from ocr.predict import recognize
        result = recognize(model, args.image, device)
        print(result["text"] or "(no text found)")
        return

    from ocr.server import serve
    serve(model, info, device, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
