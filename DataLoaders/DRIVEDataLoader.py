import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

_ID_RE = re.compile(r"^(\d+)_")


def _extract_id(filename: str) -> str:
    match = _ID_RE.match(filename)
    if not match:
        raise ValueError(f"Cannot extract DRIVE sample id from '{filename}'")
    return match.group(1)


class DriveDataset(Dataset):
    """
    Dataset wrapper for the DRIVE retinal vessel segmentation data.

    Expected directory layout::

        DRIVE/
            training/
                images/
                1st_manual/
                mask/             # field-of-view mask (optional during training)
            test/
                ...
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        image_subdir: str = "images",
        mask_subdir: str = "1st_manual",
        image_tfms: Optional[T.Compose] = None,
        mask_tfms: Optional[T.Compose] = None,
        binarize_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / image_subdir
        self.mask_dir = self.root / split / mask_subdir

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"DriveDataset image directory missing: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"DriveDataset mask directory missing: {self.mask_dir}")

        image_files = { _extract_id(p.name): p for p in sorted(self.image_dir.glob("*.tif")) }
        mask_files = { _extract_id(p.name): p for p in sorted(self.mask_dir.glob("*.gif")) }

        common_ids = sorted(set(image_files) & set(mask_files), key=lambda x: int(x))
        if not common_ids:
            raise RuntimeError(f"No overlapping samples between images and masks under {self.split}")

        self.samples: List[Dict[str, Path]] = [
             {"id": sid, "image_path": image_files[sid], "mask_path": mask_files[sid]}
             for sid in common_ids
        ]

        self.image_tfms = image_tfms or T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.mask_tfms = mask_tfms or T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.binarize_threshold = binarize_threshold

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.samples[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        mask = Image.open(record["mask_path"]).convert("L")

        image_tensor = self.image_tfms(image)
        mask_tensor = self.mask_tfms(mask)
        mask_tensor = (mask_tensor >= self.binarize_threshold).long().squeeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": {
                "id": record["id"],
                "image_path": str(record["image_path"]),
                "mask_path": str(record["mask_path"]),
                "split": self.split,
            },
        }


def create_drive_dataloaders(
    root: str,
    batch_size: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    seed: int = 42,
    size: Optional[Tuple[int, int]] = None,
    image_tfms: Optional[T.Compose] = None,
    mask_tfms: Optional[T.Compose] = None,
) -> Dict[str, DataLoader]:
    if size is not None:
        image_tfms = image_tfms or T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        mask_tfms = mask_tfms or T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    dataset = DriveDataset(
        root=root,
        split="training",
        image_tfms=image_tfms,
        mask_tfms=mask_tfms,
    )

    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1).")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("test_split must be in [0, 1).")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must sum to < 1.")

    total = len(dataset)
    val_len = int(round(total * val_split))
    test_len = int(round(total * test_split))

    if val_split > 0 and val_len == 0:
        val_len = 1
    if test_split > 0 and test_len == 0:
        test_len = 1

    train_len = total - val_len - test_len
    if train_len <= 0:
        raise ValueError("Splits leave no samples for training.")

    lengths: List[int] = [train_len]
    if val_len > 0:
        lengths.append(val_len)
    if test_len > 0:
        lengths.append(test_len)

    generator = torch.Generator().manual_seed(seed)
    subsets = random_split(dataset, lengths, generator=generator)

    idx = 0
    train_dataset = subsets[idx]
    idx += 1
    val_dataset = subsets[idx] if val_len > 0 else None
    idx += 1 if val_len > 0 else 0
    test_dataset = subsets[idx] if test_len > 0 else None

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    loaders: Dict[str, DataLoader] = {"train": _make_loader(train_dataset, shuffle=True)}
    if val_dataset is not None:
        loaders["val"] = _make_loader(val_dataset, shuffle=False)
    if test_dataset is not None:
        loaders["test"] = _make_loader(test_dataset, shuffle=False)

    return loaders


def build_dataloaders(
    drive_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    transform: Optional[T.Compose] = None,
    mask_transform: Optional[T.Compose] = None,
):
    loaders = create_drive_dataloaders(
        root=drive_root,
        batch_size=batch_size,
        val_split=val_ratio,
        test_split=0.0,
        num_workers=num_workers,
        seed=seed,
        size=None,
        image_tfms=transform,
        mask_tfms=mask_transform,
    )
    return loaders["train"], loaders["val"]


__all__ = ["DriveDataset", "create_drive_dataloaders", "build_dataloaders"]
