import os
from glob import glob
from typing import Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T


class PH2Dataset(Dataset):
    """
    Dataset loader for PH2 skin lesion segmentation.

    Expected directory layout::

        PH2_Dataset_images/
            IMD002/
                IMD002_Dermoscopic_Image/IMD002.bmp
                IMD002_lesion/IMD002_lesion.bmp
                IMD002_roi/IMD002_R1_Label4.bmp
            IMD003/
                ...
    """

    def __init__(
        self,
        root: str,
        target: Literal["lesion", "roi"] = "lesion",
        image_tfms: Optional[T.Compose] = None,
        mask_tfms: Optional[T.Compose] = None,
        size: Optional[Tuple[int, int]] = (256, 256),
        roi_preference: Literal["R1", "largest"] = "R1",
    ) -> None:
        super().__init__()
        if target not in {"lesion", "roi"}:
            raise ValueError("PH2Dataset currently supports 'lesion' or 'roi' targets.")

        self.root = os.path.abspath(root)
        self.target = target
        self.size = size
        self.roi_preference = roi_preference
        self.image_tfms = image_tfms
        self.mask_tfms = mask_tfms

        self.items = self._index_cases()

        if self.image_tfms is None:
            img_tfms = []
            if size is not None:
                img_tfms.append(T.Resize(size, interpolation=T.InterpolationMode.BILINEAR))
            img_tfms.append(T.ToTensor())
            self.image_tfms = T.Compose(img_tfms)

        if self.mask_tfms is None:
            mask_tfms = []
            if size is not None:
                mask_tfms.append(T.Resize(size, interpolation=T.InterpolationMode.NEAREST))
            mask_tfms.append(T.PILToTensor())
            self.mask_tfms = T.Compose(mask_tfms)

    def _index_cases(self) -> List[Dict[str, Optional[str]]]:
        cases: List[Dict[str, Optional[str]]] = []
        entries = next(os.walk(self.root))[1]

        for case_dir in sorted(entries):
            case_root = os.path.join(self.root, case_dir)

            img_dir = os.path.join(case_root, f"{case_dir}_Dermoscopic_Image")
            img_candidates = glob(os.path.join(img_dir, "*.bmp"))
            if not img_candidates:
                continue
            image_path = img_candidates[0]

            lesion_path: Optional[str] = None
            lesion_dir = os.path.join(case_root, f"{case_dir}_lesion")
            lesion_candidates = glob(os.path.join(lesion_dir, f"{case_dir}_lesion.bmp"))
            if lesion_candidates:
                lesion_path = lesion_candidates[0]

            roi_path: Optional[str] = None
            roi_dir = os.path.join(case_root, f"{case_dir}_roi")
            roi_candidates = sorted(glob(os.path.join(roi_dir, f"{case_dir}_R*_Label*.bmp")))
            if roi_candidates:
                if self.roi_preference == "R1":
                    r1_candidates = [
                        candidate for candidate in roi_candidates if os.path.basename(candidate).startswith(f"{case_dir}_R1")
                    ]
                    roi_path = r1_candidates[0] if r1_candidates else roi_candidates[0]
                else:
                    roi_path = max(
                        roi_candidates,
                        key=lambda candidate: Image.open(candidate).size[0] * Image.open(candidate).size[1],
                    )

            cases.append(
                {
                    "case": case_dir,
                    "image": image_path,
                    "lesion": lesion_path,
                    "roi": roi_path,
                }
            )

        if not cases:
            raise RuntimeError(f"No PH2 cases found under {self.root}")
        return cases

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.items[idx]
        img = self._load_rgb(entry["image"])
        mask_path = entry[self.target]
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for case {entry['case']} (target={self.target}).")

        mask_img = Image.open(mask_path).convert("L")

        image_tensor = self.image_tfms(img)
        mask_tensor = self.mask_tfms(mask_img)
        # Convert mask to integer labels {0,1}
        mask_tensor = (mask_tensor > 0).long().squeeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": {
                "case": entry["case"],
                "image_path": entry["image"],
                "mask_path": mask_path,
                "target": self.target,
            },
        }


def create_ph2_dataloaders(
    root: str,
    batch_size: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    seed: int = 42,
    size: Optional[Tuple[int, int]] = (256, 256),
    target: Literal["lesion", "roi"] = "lesion",
    image_tfms: Optional[T.Compose] = None,
    mask_tfms: Optional[T.Compose] = None,
) -> Dict[str, DataLoader]:
    dataset = PH2Dataset(
        root=root,
        target=target,
        image_tfms=image_tfms,
        mask_tfms=mask_tfms,
        size=size,
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
    root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    transform: Optional[T.Compose] = None,
    mask_transform: Optional[T.Compose] = None,
):
    loaders = create_ph2_dataloaders(
        root=root,
        batch_size=batch_size,
        val_split=val_ratio,
        test_split=0.0,
        num_workers=num_workers,
        seed=seed,
        size=None,
        target="lesion",
        image_tfms=transform,
        mask_tfms=mask_transform,
    )
    return loaders["train"], loaders["val"]


__all__ = ["PH2Dataset", "create_ph2_dataloaders", "build_dataloaders"]
