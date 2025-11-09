"""
Training entry point for the segmentation assignment.

Supports toggling between the fully-convolutional network scaffold and a future
U-Net implementation (placeholder). Uses the PH2 dataloader by default and is
ready for GPU execution on the DTU HPC cluster (paths default to /dtu/datasets1/02516).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable

import contextlib

import torch
from packaging import version
from torch import nn
from torch.optim import Adam

from DataLoaders.PH2DataLoader import create_ph2_dataloaders
from DataLoaders.DRIVEDataLoader import create_drive_dataloaders
from models.segmentation_model import create_model
from models.unet import create_unet

# -- Default dataset roots on the DTU HPC --
_DEFAULT_DATA_ROOTS = {
    "ph2": os.environ.get("PH2_ROOT", "C:/Users/owner/Documents/DTU/Semester_1/comp_vision/PH2_Dataset_images"),
    "drive": os.environ.get("DRIVE_ROOT", "C:/Users/owner/Documents/DTU/Semester_1/comp_vision/DRIVE"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--dataset", choices=("ph2", "drive"), default="drive", help="Which dataset to use.")
    parser.add_argument("--model", choices=("fcn", "unet"), default="unet", help="Model architecture.")
    parser.add_argument("--data-root", default=None, help="Root directory for the chosen dataset.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker threads.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data reserved for validation.")
    parser.add_argument("--test-split", type=float, default=0.0, help="Fraction of data reserved for testing.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for inputs; <=0 keeps original.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num-classes", type=int, default=2, help="Segmentation classes (foreground/background).")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or specific device string.")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Limit training steps per epoch (0 = all).")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP).")
    return parser.parse_args(argv)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(name: str, num_classes: int) -> nn.Module:
    factories = {
        "fcn": lambda: create_model({"num_classes": num_classes}),
        "unet": lambda: create_unet(num_classes=num_classes),
    }
    if name not in factories:
        raise ValueError(f"Unknown model: {name}")
    return factories[name]()


def build_dataloaders(args: argparse.Namespace) -> Dict[str, Iterable[Dict[str, object]]]:
    size = (args.image_size, args.image_size) if args.image_size > 0 else None
    if args.dataset == "ph2":
        return create_ph2_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=args.num_workers,
            size=size,
        )
    if args.dataset == "drive":
        return create_drive_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=args.num_workers,
            size=size,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _prepare_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    masks = masks.to(device=device, dtype=torch.long, non_blocking=True)
    return {"image": images, "mask": masks, "meta": batch.get("meta")}


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    # Works for binary masks after argmax.
    pred_fg = pred == 1
    target_fg = target == 1
    intersection = (pred_fg & target_fg).sum().item()
    pred_sum = pred_fg.sum().item()
    target_sum = target_fg.sum().item()
    return (2 * intersection + eps) / (pred_sum + target_sum + eps)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    use_new_amp = version.parse(torch.__version__) >= version.parse("2.1.0")
    if use_amp and device.type != "cuda":
        raise ValueError("AMP is only supported on CUDA devices in this training script.")

    if use_amp:
        if use_new_amp:
            scaler = torch.amp.GradScaler(enabled=True)
            autocast_factory = lambda: torch.amp.autocast(device_type=device.type, enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            autocast_factory = lambda: torch.cuda.amp.autocast()
    else:
        scaler = None
        autocast_factory = contextlib.nullcontext

    total_loss = 0.0
    total_samples = 0
    total_pixels = 0
    total_correct = 0
    dice_sum = 0.0
    dice_count = 0
    num_batches = 0

    for step, raw_batch in enumerate(loader, start=1):
        batch = _prepare_batch(raw_batch, device)
        images, masks = batch["image"], batch["mask"]
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_factory():
            logits = model({"image": images})
            loss = criterion(logits, masks)

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()
        dice_sum += _dice_score(preds, masks)
        dice_count += 1
        num_batches += 1

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    pixel_acc = total_correct / total_pixels if total_pixels else 0.0
    avg_dice = dice_sum / dice_count if dice_count else 0.0
    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "dice": avg_dice,
        "num_batches": num_batches,
        "num_samples": total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Dict[str, object]],
    criterion: nn.Module,
    device: torch.device,
    max_steps: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_pixels = 0
    total_correct = 0
    dice_sum = 0.0
    dice_count = 0
    num_batches = 0

    for step, raw_batch in enumerate(loader, start=1):
        batch = _prepare_batch(raw_batch, device)
        images, masks = batch["image"], batch["mask"]
        batch_size = images.size(0)

        logits = model({"image": images})
        loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()
        dice_sum += _dice_score(preds, masks)
        dice_count += 1
        num_batches += 1

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    pixel_acc = total_correct / total_pixels if total_pixels else 0.0
    avg_dice = dice_sum / dice_count if dice_count else 0.0
    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "dice": avg_dice,
        "num_batches": num_batches,
        "num_samples": total_samples,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_root is None:
        args.data_root = _DEFAULT_DATA_ROOTS.get(args.dataset)

    if args.data_root is None:
        print(f"No default data root for dataset '{args.dataset}'. Please supply --data-root.", file=sys.stderr)
        return 2

    if not os.path.isdir(args.data_root):
        print(
            f"Data root '{args.data_root}' not found. Adjust --data-root (current machine may lack the HPC mount).",
            file=sys.stderr,
        )
        return 2

    device = select_device(args.device)
    print(f"Using device: {device}")

    try:
        model = build_model(args.model, args.num_classes).to(device)
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        loaders = build_dataloaders(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            max_steps=args.max_train_steps,
            use_amp=args.amp and device.type == "cuda",
        )
        val_stats = None
        if "val" in loaders:
            val_stats = evaluate(
                model,
                loaders["val"],
                criterion,
                device,
                max_steps=args.max_train_steps,
            )

        log_parts = [
            f"[Epoch {epoch}/{args.epochs}]",
            f"train_loss={train_stats['loss']:.4f}",
            f"train_acc={train_stats['pixel_acc']:.3f}",
            f"train_dice={train_stats['dice']:.3f}",
            f"train_batches={train_stats['num_batches']}",
        ]
        if val_stats is not None:
            log_parts.extend(
                [
                    f"val_loss={val_stats['loss']:.4f}",
                    f"val_acc={val_stats['pixel_acc']:.3f}",
                    f"val_dice={val_stats['dice']:.3f}",
                    f"val_batches={val_stats['num_batches']}",
                ]
            )
        print(" ".join(log_parts))

    if "val" in loaders:
        final_val = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            max_steps=0,
        )
        print(
            f"[Validation] loss={final_val['loss']:.4f} "
            f"acc={final_val['pixel_acc']:.3f} dice={final_val['dice']:.3f} "
            f"batches={final_val['num_batches']}"
        )

    if "test" in loaders:
        final_test = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
            max_steps=0,
        )
        print(
            f"[Test] loss={final_test['loss']:.4f} "
            f"acc={final_test['pixel_acc']:.3f} dice={final_test['dice']:.3f} "
            f"batches={final_test['num_batches']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
