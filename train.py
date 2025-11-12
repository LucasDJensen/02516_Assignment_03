"""
Training entry point for the segmentation assignment.

Supports toggling between the fully-convolutional network scaffold and a future
U-Net implementation (placeholder). Uses the PH2 dataloader by default and is
ready for GPU execution on the DTU HPC cluster (paths default to /dtu/datasets1/02516).
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, Iterable


import contextlib

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
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
    "ph2": os.environ.get("PH2_ROOT", "dtu/datasets1/02516/PH2_Dataset_images"),
    "drive": os.environ.get("DRIVE_ROOT", "dtu/datasets1/02516/DRIVE"),
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
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory for saving training curves/metrics (set empty string to disable).",
    )
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


def _prepare_artifact_dir(path: str | None) -> str | None:
    if not path:
        return None
    resolved = os.path.abspath(path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _sanitize_stats(stats: Dict[str, float | int] | None) -> Dict[str, float | int | None] | None:
    if stats is None:
        return None
    cleaned: Dict[str, float | int | None] = {}
    for key, value in stats.items():
        if isinstance(value, float):
            cleaned[key] = value if math.isfinite(value) else None
        elif isinstance(value, int):
            cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned


def _extract_metric(history: list[Dict[str, object]], split: str, metric: str) -> list[float | None]:
    values: list[float | None] = []
    for entry in history:
        split_stats = entry.get(split)
        if isinstance(split_stats, dict):
            values.append(split_stats.get(metric))
        else:
            values.append(None)
    return values


def _nanify(values: list[float | None]) -> list[float]:
    return [v if v is not None else float("nan") for v in values]


def _plot_metric(ax, epochs: list[int], train_vals: list[float | None], val_vals: list[float | None], ylabel: str, title: str) -> None:
    ax.plot(epochs, _nanify(train_vals), label="Train", marker="o")
    if any(v is not None for v in val_vals):
        ax.plot(epochs, _nanify(val_vals), label="Validation", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()


def _plot_training_curves(history: list[Dict[str, object]], artifact_dir: str) -> str:
    epochs = [entry.get("epoch") for entry in history]
    train_loss = _extract_metric(history, "train", "loss")
    val_loss = _extract_metric(history, "val", "loss")
    train_acc = _extract_metric(history, "train", "pixel_acc")
    val_acc = _extract_metric(history, "val", "pixel_acc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_metric(axes[0], epochs, train_loss, val_loss, "Loss", "Cross-Entropy Loss")
    _plot_metric(axes[1], epochs, train_acc, val_acc, "Pixel Accuracy", "Pixel Accuracy")
    fig.tight_layout()

    figure_path = os.path.join(artifact_dir, "training_curves.png")
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _save_history(history: list[Dict[str, object]], artifact_dir: str) -> str:
    history_path = os.path.join(artifact_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    return history_path


def _save_checkpoint(model: nn.Module, artifact_dir: str, filename: str) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, filename)
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            array = tensor.squeeze(0)
        else:
            array = tensor.permute(1, 2, 0)
    elif tensor.ndim == 2:
        array = tensor
    else:
        array = tensor
    array = array.numpy()
    min_val = np.min(array)
    max_val = np.max(array)
    # Avoid division by zero for constant images.
    if max_val - min_val > 1e-8:
        array = (array - min_val) / (max_val - min_val)
    else:
        array = np.zeros_like(array)
    return array


def _extract_sample_id(meta_entry) -> str | None:
    if meta_entry is None:
        return None
    if isinstance(meta_entry, dict):
        for key in ("id", "name", "filename", "file"):
            if key in meta_entry and meta_entry[key]:
                return str(meta_entry[key])
    else:
        return os.path.basename(str(meta_entry))
    return None


def _save_prediction_samples(
    model: nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    artifact_dir: str | None,
    split_name: str,
    max_images: int = 4,
) -> list[str]:
    if artifact_dir is None or loader is None or max_images <= 0:
        return []

    samples_dir = os.path.join(artifact_dir, "samples", split_name)
    os.makedirs(samples_dir, exist_ok=True)

    model_was_training = model.training
    model.eval()
    saved_paths: list[str] = []
    images_saved = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = _prepare_batch(raw_batch, device)
            images, masks = batch["image"], batch["mask"]
            logits = model({"image": images})
            preds = logits.argmax(dim=1)
            meta = batch.get("meta")

            for idx in range(images.size(0)):
                if images_saved >= max_images:
                    break

                input_img = _tensor_to_display(images[idx])
                target_mask = _tensor_to_display(masks[idx])
                pred_mask = _tensor_to_display(preds[idx])

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(input_img, cmap="gray" if input_img.ndim == 2 else None)
                axes[0].set_title("Input")
                axes[1].imshow(target_mask, cmap="gray")
                axes[1].set_title("Target")
                axes[2].imshow(pred_mask, cmap="gray")
                axes[2].set_title("Prediction")
                for ax in axes:
                    ax.axis("off")

                sample_id = None
                if isinstance(meta, (list, tuple)) and idx < len(meta):
                    sample_id = _extract_sample_id(meta[idx])
                if sample_id:
                    fig.suptitle(f"{split_name.capitalize()} sample: {sample_id}", fontsize=10)

                filename = f"{split_name}_sample_{images_saved + 1}.png"
                save_path = os.path.join(samples_dir, filename)
                fig.tight_layout()
                fig.savefig(save_path, dpi=200)
                plt.close(fig)

                saved_paths.append(save_path)
                images_saved += 1

            if images_saved >= max_images:
                break

    if model_was_training:
        model.train()
    return saved_paths


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
    # Confusion matrix components for binary segmentation (positive class = 1)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    eps = 1e-6

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

        # Update confusion matrix for binary metrics
        pred_fg = (preds == 1)
        true_fg = (masks == 1)
        pred_bg = ~pred_fg
        true_bg = ~true_fg
        tp += (pred_fg & true_fg).sum().item()
        tn += (pred_bg & true_bg).sum().item()
        fp += (pred_fg & true_bg).sum().item()
        fn += (pred_bg & true_fg).sum().item()

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    pixel_acc = total_correct / total_pixels if total_pixels else 0.0
    avg_dice = dice_sum / dice_count if dice_count else 0.0
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)  # Recall / TPR
    specificity = tn / (tn + fp + eps)  # TNR
    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "dice": avg_dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
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
    # Confusion matrix components for binary segmentation (positive class = 1)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    eps = 1e-6

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

        # Update confusion matrix for binary metrics
        pred_fg = (preds == 1)
        true_fg = (masks == 1)
        pred_bg = ~pred_fg
        true_bg = ~true_fg
        tp += (pred_fg & true_fg).sum().item()
        tn += (pred_bg & true_bg).sum().item()
        fp += (pred_fg & true_bg).sum().item()
        fn += (pred_bg & true_fg).sum().item()

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    pixel_acc = total_correct / total_pixels if total_pixels else 0.0
    avg_dice = dice_sum / dice_count if dice_count else 0.0
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "dice": avg_dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
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

    artifact_base = args.artifact_dir.strip() if args.artifact_dir else None
    artifact_path = None
    if artifact_base:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = os.path.join(artifact_base, timestamp)
    artifact_dir = _prepare_artifact_dir(artifact_path)
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
    history: list[Dict[str, object]] = []
    best_val_metric = -float("inf")
    best_checkpoint_path: str | None = None

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
            f"train_iou={train_stats.get('iou', float('nan')):.3f}",
            f"train_sens={train_stats.get('sensitivity', float('nan')):.3f}",
            f"train_spec={train_stats.get('specificity', float('nan')):.3f}",
            f"train_batches={train_stats['num_batches']}",
        ]
        if val_stats is not None:
            log_parts.extend(
                [
                    f"val_loss={val_stats['loss']:.4f}",
                    f"val_acc={val_stats['pixel_acc']:.3f}",
                    f"val_dice={val_stats['dice']:.3f}",
                    f"val_iou={val_stats.get('iou', float('nan')):.3f}",
                    f"val_sens={val_stats.get('sensitivity', float('nan')):.3f}",
                    f"val_spec={val_stats.get('specificity', float('nan')):.3f}",
                    f"val_batches={val_stats['num_batches']}",
                ]
            )
        print(" ".join(log_parts))
        history.append({"epoch": epoch, "train": _sanitize_stats(train_stats), "val": _sanitize_stats(val_stats)})
        if artifact_dir and val_stats is not None:
            candidate_metric = val_stats.get("dice")
            if isinstance(candidate_metric, float) and math.isfinite(candidate_metric) and candidate_metric > best_val_metric:
                best_val_metric = candidate_metric
                best_checkpoint_path = _save_checkpoint(model, artifact_dir, "best_model.pt")
                print(
                    f"[Checkpoint] Saved new best model to {best_checkpoint_path} "
                    f"(val_dice={candidate_metric:.3f})"
                )

    if artifact_dir and history:
        history_path = _save_history(history, artifact_dir)
        curves_path = _plot_training_curves(history, artifact_dir)
        print(f"[Artifacts] metrics={history_path} curves={curves_path}")
        last_checkpoint_path = _save_checkpoint(model, artifact_dir, "last_model.pt")
        info = f"[Checkpoint] last={last_checkpoint_path}"
        if best_checkpoint_path:
            info += f" best={best_checkpoint_path}"
        print(info)

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
            f"iou={final_val.get('iou', float('nan')):.3f} "
            f"sens={final_val.get('sensitivity', float('nan')):.3f} "
            f"spec={final_val.get('specificity', float('nan')):.3f} "
            f"batches={final_val['num_batches']}"
        )
        val_sample_paths = _save_prediction_samples(model, loaders["val"], device, artifact_dir, "val")
        if val_sample_paths:
            print(f"[Samples:val] saved {len(val_sample_paths)} preview(s): {val_sample_paths[0]}")

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
            f"iou={final_test.get('iou', float('nan')):.3f} "
            f"sens={final_test.get('sensitivity', float('nan')):.3f} "
            f"spec={final_test.get('specificity', float('nan')):.3f} "
            f"batches={final_test['num_batches']}"
        )
        test_sample_paths = _save_prediction_samples(model, loaders["test"], device, artifact_dir, "test")
        if test_sample_paths:
            print(f"[Samples:test] saved {len(test_sample_paths)} preview(s): {test_sample_paths[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
