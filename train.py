"""
Dummy training script that can toggle between the fully convolutional
segmentation model and an upcoming U-Net implementation.

Usage
-----
    python train.py --model fcn
    python train.py --model unet

The `"unet"` option currently raises `NotImplementedError`. Once `models/unet.py`
lands, add its factory function to `_MODEL_REGISTRY`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict

import torch

from models.segmentation_model import create_model


def _load_unet_model(**_: object) -> torch.nn.Module:
    """
    Placeholder loader for the upcoming U-Net architecture.

    Replace the body with:

        from models.unet import create_unet
        return create_unet(**overrides)
    """

    raise NotImplementedError("U-Net model is not yet implemented. Add it to models/unet.py.")


_MODEL_REGISTRY: Dict[str, Callable[..., torch.nn.Module]] = {
    "fcn": create_model,
    "unet": _load_unet_model,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation training entry point.")
    parser.add_argument(
        "--model",
        default="fcn",
        choices=sorted(_MODEL_REGISTRY),
        help="Which architecture to instantiate.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of semantic classes.")
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--height", type=int, default=256, help="Dummy image height.")
    parser.add_argument("--width", type=int, default=256, help="Dummy image width.")
    parser.add_argument("--device", default="cpu", help="Torch device string, e.g. 'cpu' or 'cuda:0'.")
    return parser.parse_args(argv)


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    factory = _MODEL_REGISTRY[name]
    try:
        model = factory({"num_classes": num_classes})
    except TypeError:
        model = factory(num_classes=num_classes)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Factory for model '{name}' must return an nn.Module.")
    return model


def run_dummy_iteration(
    model: torch.nn.Module, batch_size: int, height: int, width: int, num_classes: int, device: str
) -> None:
    model.to(device)
    model.train()

    dummy_batch = {
        "image": torch.randn(batch_size, 3, height, width, device=device),
        "mask": torch.randint(0, num_classes, (batch_size, height, width), device=device),
    }

    logits = model(dummy_batch)
    loss = torch.nn.functional.cross_entropy(logits, dummy_batch["mask"])
    loss.backward()

    print(
        f"[Dummy run] model={model.__class__.__name__} "
        f"logits_shape={tuple(logits.shape)} loss={loss.item():.4f} device={device}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = build_model(args.model, args.num_classes)
    except NotImplementedError as exc:
        print(exc)
        return 1

    run_dummy_iteration(model, args.batch_size, args.height, args.width, args.num_classes, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
