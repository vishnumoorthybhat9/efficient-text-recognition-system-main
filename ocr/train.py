"""Train the CRNN from scratch (random initial weights) with CTC loss."""
from __future__ import annotations

import os
import sys
import time

import torch
from torch import nn

from . import config
from .dataset import LineBatches, ensure_extracted
from .model import CRNN, greedy_decode, save_model


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> dict:
    """Character error rate (CER) and exact-line accuracy per kind of text."""
    model.eval()
    stats = {k: [0, 0, 0, 0] for k in ("handwritten", "printed")}   # errs, chars, exact, lines
    for index, (images, _, _, texts) in enumerate(loader):
        kind = "handwritten" if index % 3 == 0 else "printed"
        preds = greedy_decode(model(images.to(device)))
        for (pred, _), truth in zip(preds, texts):
            s = stats[kind]
            s[0] += edit_distance(pred, truth)
            s[1] += len(truth)
            s[2] += pred == truth
            s[3] += 1
    return {k: {"cer": s[0] / max(1, s[1]), "exact": s[2] / max(1, s[3]), "lines": s[3]}
            for k, s in stats.items()}


def train(epochs: int = config.EPOCHS, steps: int = config.STEPS_PER_EPOCH,
          batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
          resume: bool = False) -> dict:
    """Train from random weights; resume=True continues an interrupted run."""
    ensure_extracted()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cores = os.cpu_count() or 2
    workers = 0 if sys.platform == "win32" and device == "cpu" else min(4, max(1, cores // 2))
    if device == "cpu":
        torch.set_num_threads(max(1, cores - workers // 2))

    def loader(split: str, batches: int, seed: int):
        ds = LineBatches(split, batches, batch_size, seed)
        return ds, torch.utils.data.DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=workers,
            persistent_workers=workers > 0)

    train_ds, train_loader = loader("train", steps, 0)
    val_ds, val_loader = loader("val", 0, 1)
    val_ds.batches = 3 * max(1, len(val_ds.hand) // batch_size)   # covers every val line once

    torch.manual_seed(0)
    model = CRNN().to(device)                      # random weights: trained from scratch
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps,
                                                pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    print(f"Training from scratch on {device}: {params / 1e6:.2f} M parameters, "
          f"{len(train_ds.hand)} handwritten lines + "
          f"{sum(len(v) for v in train_ds.fonts.values())} printed words "
          f"({len(train_ds.font_ids)} fonts), {epochs} epochs x {steps} steps", flush=True)

    best, best_score, first, started = {}, float("inf"), 1, time.time()
    if resume and config.STATE_PATH.exists():
        state = torch.load(config.STATE_PATH, map_location=device, weights_only=True)
        if state["total_steps"] == epochs * steps:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            best, best_score, first = state["best"], state["best_score"], state["epoch"] + 1
            print(f"Resuming interrupted training at epoch {first}.", flush=True)

    for epoch in range(first, epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for step, (images, targets, lengths, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(images)
            log_probs = logits.float().log_softmax(-1).transpose(0, 1)     # T x B x C
            in_lens = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, in_lens, lengths)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 25 == 0 or step == steps:
                print(f"\r  epoch {epoch:>2}/{epochs}  step {step:>4}/{steps}  "
                      f"loss {total / step:6.3f}  {(time.time() - t0) / step:4.2f}s/step",
                      end="", flush=True)

        m = evaluate(model, val_loader, device)
        score = m["handwritten"]["cer"] + m["printed"]["cer"]
        flag = ""
        if score < best_score:
            best_score, flag = score, "  * saved"
            best = {"metrics": m, "epoch": epoch, "params": params}
            save_model(model, config.MODEL_PATH, **best)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "best_score": best_score, "total_steps": epochs * steps}, config.STATE_PATH)
        print(f"\r  epoch {epoch:>2}/{epochs}  loss {total / steps:6.3f}  "
              f"val CER handwritten {m['handwritten']['cer']:.3f} "
              f"printed {m['printed']['cer']:.3f}  "
              f"[{(time.time() - started) / 60:5.1f} min]{flag}      ", flush=True)

    config.STATE_PATH.unlink(missing_ok=True)          # run complete: nothing to resume
    print(f"Best model (epoch {best['epoch']}) saved to {config.MODEL_PATH}", flush=True)
    return best
