"""
Train the CNN + Transformer CTC model on the handwritten-name dataset.

Expected layout (the one in C:\\EFFICIENT\\data):
    data/written_name_train_v2.csv        FILENAME,IDENTITY
    data/train_v2/train/*.jpg
    data/written_name_validation_v2.csv
    data/validation_v2/validation/*.jpg
    data/written_name_test_v2.csv
    data/test_v2/test/*.jpg

Example:
    python train.py --data-root data --out-dir runs/exp1 --epochs 30 --batch-size 128
Quick smoke test:
    python train.py --data-root data --out-dir runs/smoke --epochs 1 --limit 2000
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import ocr_utils as U
from ocr_model import OCRNet, count_parameters


# ----------------------------------------------------------------- dataset
class NameDataset(Dataset):
    def __init__(self, items, char2idx, augment=False):
        self.items = items
        self.char2idx = char2idx
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        x = U.preprocess(path, augment=self.augment)
        target = [self.char2idx[c] for c in text]
        return torch.from_numpy(x).unsqueeze(0), target, text


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.tensor([t for b in batch for t in b[1]], dtype=torch.long)
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    texts = [b[2] for b in batch]
    return images, targets, target_lengths, texts


def worker_init(_):
    cv2.setNumThreads(0)  # avoid thread oversubscription inside DataLoader workers


# -------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, idx2char, device):
    model.eval()
    preds, refs = [], []
    total_loss, n_batches = 0.0, 0
    ctc = torch.nn.CTCLoss(blank=U.BLANK, zero_infinity=True)
    for images, targets, target_lengths, texts in loader:
        images = images.to(device, non_blocking=True)
        log_probs = model(images)                      # (B, T, K+1)
        T = log_probs.size(1)
        input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
        loss = ctc(log_probs.permute(1, 0, 2).float(), targets, input_lengths, target_lengths)
        total_loss += loss.item()
        n_batches += 1
        best = log_probs.argmax(-1).cpu().numpy()
        preds += [U.decode_greedy(seq, idx2char) for seq in best]
        refs += texts
    return {
        "loss": total_loss / max(1, n_batches),
        "cer": U.cer(preds, refs),
        "exact_match": U.exact_match(preds, refs),
        "samples": list(zip(refs[:10], preds[:10])),
    }


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows (debug)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"device={device}  amp={use_amp}")

    # ---- data
    d = args.data_root
    train_items = U.read_split(os.path.join(d, "written_name_train_v2.csv"),
                               os.path.join(d, "train_v2", "train"))
    val_items = U.read_split(os.path.join(d, "written_name_validation_v2.csv"),
                             os.path.join(d, "validation_v2", "validation"))
    test_items = U.read_split(os.path.join(d, "written_name_test_v2.csv"),
                              os.path.join(d, "test_v2", "test"))
    if args.limit:
        train_items = train_items[: args.limit]
        val_items = val_items[: max(200, args.limit // 10)]
    chars = U.build_charset([t for _, t in train_items])
    char2idx = {c: i + 1 for i, c in enumerate(chars)}   # 0 is the blank
    idx2char = chars
    with open(os.path.join(args.out_dir, "charset.json"), "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)} "
          f"chars={len(chars)}")

    # labels that appear only in val/test cannot be predicted; report them
    unseen = {c for _, t in val_items + test_items for c in t} - set(chars)
    if unseen:
        print(f"WARNING: characters not in training set (will be unpredictable): {sorted(unseen)}")

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     collate_fn=collate, worker_init_fn=worker_init,
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(NameDataset(train_items, char2idx, augment=True),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(NameDataset(val_items, char2idx), batch_size=256, shuffle=False, **loader_kw)
    test_loader = DataLoader(NameDataset(test_items, char2idx), batch_size=256, shuffle=False, **loader_kw)

    # ---- model
    model = OCRNet(num_classes=len(chars), d_model=args.d_model, nhead=args.heads,
                   num_layers=args.layers, ff_dim=4 * args.d_model, dropout=args.dropout).to(device)
    print(f"trainable parameters: {count_parameters(model) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos")
    ctc = torch.nn.CTCLoss(blank=U.BLANK, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(vars(args), num_classes=len(chars), img_h=U.IMG_H, img_w=U.IMG_W)
    history, best_cer = [], float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (images, targets, target_lengths, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                log_probs = model(images)
            T = log_probs.size(1)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
            loss = ctc(log_probs.float().permute(1, 0, 2), targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.4f}")

        val = evaluate(model, val_loader, idx2char, device)
        rec = dict(epoch=epoch, train_loss=running / max(1, seen), val_loss=val["loss"],
                   val_cer=val["cer"], val_exact=val["exact_match"],
                   lr=scheduler.get_last_lr()[0], seconds=round(time.time() - t0, 1))
        history.append(rec)
        print(f"epoch {epoch:3d} | train_loss {rec['train_loss']:.4f} | val_loss {rec['val_loss']:.4f} "
              f"| val_CER {rec['val_cer']:.4f} | val_exact {rec['val_exact']:.4f} | {rec['seconds']}s")
        for ref, pred in val["samples"][:3]:
            print(f"      ref={ref!r:28} pred={pred!r}")

        if val["cer"] < best_cer:
            best_cer = val["cer"]
            torch.save({"model": model.state_dict(), "charset": chars, "config": config,
                        "epoch": epoch, "val_cer": best_cer}, best_path)
            print(f"  saved best checkpoint (val_CER {best_cer:.4f}) -> {best_path}")

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ---- final test evaluation with the best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, idx2char, device)
    results = {"best_epoch": ckpt["epoch"], "val_cer": best_cer,
               "test_cer": test["cer"], "test_exact_match": test["exact_match"]}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nTEST RESULTS:", json.dumps(results, indent=2))
    for ref, pred in test["samples"]:
        print(f"      ref={ref!r:28} pred={pred!r}")


if __name__ == "__main__":
    main()
