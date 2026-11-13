"""
Read the text in one or more images with a trained checkpoint.

Examples:
    python predict.py --ckpt runs/exp1/best.pt --image my_name.jpg
    python predict.py --ckpt runs/exp1/best.pt --image a.jpg b.png c.jpg --json out.json
    python predict.py --ckpt runs/exp1/best.pt --image-dir some_folder/

Each image is grayscaled, resized to height 64, and passed through the
CNN + Transformer model. The output is the greedy CTC decoding of the sequence.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import ocr_utils as U
from ocr_model import OCRNet

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    chars = ckpt["charset"]
    model = OCRNet(num_classes=len(chars), d_model=cfg["d_model"], nhead=cfg["heads"],
                   num_layers=cfg["layers"], ff_dim=4 * cfg["d_model"], dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, chars


@torch.no_grad()
def predict_paths(model, chars, paths, device, batch_size=32):
    """Return a list of (path, text, confidence) tuples."""
    results = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        x = torch.from_numpy(np.stack([U.preprocess(p) for p in chunk])).unsqueeze(1).to(device)
        probs = model(x).exp()                       # (B, T, K+1)
        best_prob, best_ids = probs.max(-1)
        best_ids = best_ids.cpu().numpy()
        best_prob = best_prob.cpu().numpy()
        for p, ids, pr in zip(chunk, best_ids, best_prob):
            text = U.decode_greedy(ids, chars)
            # confidence: mean probability of the non-blank steps that were emitted
            mask = ids != U.BLANK
            conf = float(pr[mask].mean()) if mask.any() else 0.0
            results.append((p, text, round(conf, 4)))
    return results


def collect_paths(args):
    paths = list(args.image or [])
    if args.image_dir:
        for name in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXT:
                paths.append(os.path.join(args.image_dir, name))
    if not paths:
        raise SystemExit("Give --image <file ...> or --image-dir <folder>.")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Extract text from handwritten-name images.")
    ap.add_argument("--ckpt", default="runs/exp1/best.pt")
    ap.add_argument("--image", nargs="+", help="one or more image files")
    ap.add_argument("--image-dir", help="folder of images")
    ap.add_argument("--json", help="also write results to this JSON file")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, chars = load_model(args.ckpt, device)
    paths = collect_paths(args)

    t0 = time.time()
    results = predict_paths(model, chars, paths, device)
    elapsed = time.time() - t0

    for path, text, conf in results:
        print(f"{os.path.basename(path)}\t{text}\t(confidence {conf:.2f})")
    print(f"\n{len(results)} image(s) in {elapsed:.2f}s "
          f"({1000 * elapsed / len(results):.1f} ms/image on {device})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"file": p, "text": t, "confidence": c} for p, t, c in results],
                      f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
