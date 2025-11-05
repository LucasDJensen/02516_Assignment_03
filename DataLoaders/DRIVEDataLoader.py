import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

_ID_RE = re.compile(r"^(\d+)_")

def _id_from_name(name: str) -> str:
    """
    Extract leading numeric ID before the first underscore.
    Example: '21_manual1.gif' -> '21'
             '21_training.tif' -> '21'
    """
    m = _ID_RE.match(name)
    if not m:
        raise ValueError(f"Could not parse ID from: {name}")
    return m.group(1)

class DriveSegmentation(Dataset):
    """
    DRIVE-style dataset where:
      root/
        training/
          images/*.tif       (inputs)
          1st_manual/*.gif   (binary masks)
    """
    def __init__(
        self,
        root: str,
        split: str = "training",         # keep for future extensibility
        image_subdir: str = "images",
        mask_subdir: str = "1st_manual",
        transform: Optional[T.Compose] = None,
        mask_transform: Optional[T.Compose] = None,
        binarize_threshold: float = 0.5,  # applied AFTER ToTensor (0..1)
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / image_subdir
        self.mask_dir  = self.root / split / mask_subdir
        self.transform = transform
        self.mask_transform = mask_transform
        self.binarize_threshold = binarize_threshold

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Masks dir not found: {self.mask_dir}")

        # Index images and masks by numeric ID
        img_files = { _id_from_name(p.name): p for p in sorted(self.image_dir.glob("*.tif")) }
        msk_files = { _id_from_name(p.name): p for p in sorted(self.mask_dir.glob("*.gif")) }

        # Keep only IDs that appear in BOTH sets; warn on mismatches
        common_ids = sorted(set(img_files.keys()) & set(msk_files.keys()), key=lambda x: int(x))
        missing_imgs = sorted(set(msk_files.keys()) - set(img_files.keys()), key=lambda x: int(x))
        missing_msks = sorted(set(img_files.keys()) - set(msk_files.keys()), key=lambda x: int(x))

        if missing_imgs:
            print(f"[DriveSegmentation] Warning: missing images for IDs: {missing_imgs}")
        if missing_msks:
            print(f"[DriveSegmentation] Warning: missing masks for IDs: {missing_msks}")

        self.samples: List[Dict] = [
            {"id": sid, "image_path": img_files[sid], "mask_path": msk_files[sid]}
            for sid in common_ids
        ]

        # Default transforms (simple + safe). You can replace with Albumentations if you prefer.
        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),                         # -> [0,1], CxHxW
                # Normalize if you like; DRIVE is not ImageNet, but this helps training stability.
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        if self.mask_transform is None:
            # For masks: to tensor only; no normalization. Will binarize afterwards.
            self.mask_transform = T.Compose([
                T.ToTensor(),                         # -> [0,1], 1xHxW
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.samples[idx]
        img = Image.open(record["image_path"]).convert("RGB")
        msk = Image.open(record["mask_path"]).convert("L")   # single-channel

        img_t = self.transform(img) if self.transform else T.ToTensor()(img)
        msk_t = self.mask_transform(msk) if self.mask_transform else T.ToTensor()(msk)

        # binarize mask (handle masks saved with 0/255 or grayscale)
        msk_t = (msk_t >= self.binarize_threshold).float()

        return {
            "image": img_t,            # FloatTensor [3,H,W], normalized
            "mask": msk_t,             # FloatTensor [1,H,W], values {0.,1.}
            "id": record["id"],
            "image_path": str(record["image_path"]),
            "mask_path": str(record["mask_path"]),
        }

def build_dataloaders(
    drive_root: str,                      # path to the DRIVE root (folder that contains "training")
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    transform: Optional[T.Compose] = None,
    mask_transform: Optional[T.Compose] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders from the training split.
    """
    full_ds = DriveSegmentation(
        root=drive_root,
        split="training",
        transform=transform,
        mask_transform=mask_transform,
    )

    # Split reproducibly
    n_total = len(full_ds)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Point to the folder that contains `training/`
    drive_root = r"C:\Users\lucas\PycharmProjects\02516_Assignment_03\data\DRIVE"

    # Optional: add resizing/augmentations
    img_aug = T.Compose([
        T.Resize((256, 256)),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    mask_aug = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])

    train_loader, val_loader = build_dataloaders(
        drive_root,
        batch_size=4,
        num_workers=2,
        val_ratio=0.2,
        transform=img_aug,
        mask_transform=mask_aug,
    )

    # Iterate
    for batch in train_loader:
        images = batch["image"]  # [B,3,H,W]
        masks = batch["mask"]  # [B,1,H,W]
        ids = batch["id"]
        print(images.shape, masks.shape, ids)
        # plot images
        plt.figure(figsize=(10, 5))
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow(images[i].permute(1, 2, 0))
            plt.title(f"Image {ids[i]}")
            plt.axis('off')
            plt.subplot(2, 4, i+5)
            plt.imshow(masks[i, 0], cmap='gray')
            plt.title(f"Mask {ids[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        break
