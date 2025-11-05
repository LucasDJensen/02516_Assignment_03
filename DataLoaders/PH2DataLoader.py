import os
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
<<<<<<< HEAD
=======
from torch.utils.data import Dataset, DataLoader, random_split
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T


class PH2Dataset(Dataset):
    """
    PH2 dataset loader.

    Root structure (per your screenshot):
    PH2_Dataset_images/
      IMD002/
        IMD002_Dermoscopic_Image/IMD002.bmp
        IMD002_lesion/IMD002_lesion.bmp
        IMD002_roi/IMD002_R1_Label4.bmp (sometimes also *_R2_*.bmp)
      IMD003/
        ...

    Args:
        root: path to 'PH2_Dataset_images'
        target: which mask to return: 'lesion', 'roi', or 'both'
        image_tfms: torchvision transform applied to the image (after resize)
        mask_tfms:  torchvision transform applied to mask(s) (after resize)
        size: optional (w, h) to resize image & mask(s) to; if None, keep original size
        roi_preference: if multiple ROI masks exist, choose 'R1' first or 'largest'
                        ('R1' | 'largest')
        return_paths: if True, adds a 'paths' entry mirroring the meta dictionary.
                      Samples always follow {"image": tensor, "mask": tensor, "meta": dict}.
    """

    def __init__(
            self,
            root: str,
            target: Literal["lesion", "roi", "both"] = "lesion",
            image_tfms: Optional[Callable] = None,
            mask_tfms: Optional[Callable] = None,
            size: Optional[Tuple[int, int]] = (256, 256),
            roi_preference: Literal["R1", "largest"] = "R1",
            return_paths: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.target = target
        self.image_tfms = image_tfms
        self.mask_tfms = mask_tfms
        self.size = size
        self.roi_preference = roi_preference
        self.return_paths = return_paths

        self.items = self._index_cases()

        # default basic transforms (only if user didn't pass any)
        if self.image_tfms is None:
            tfms = []
            if size is not None:
                tfms.append(T.Resize(size, interpolation=T.InterpolationMode.BILINEAR))
            tfms += [T.ToTensor()]  # 0..1 float, CxHxW
            self.image_tfms = T.Compose(tfms)

        if self.mask_tfms is None:
            m_tfms = []
            if size is not None:
                m_tfms.append(T.Resize(size, interpolation=T.InterpolationMode.NEAREST))
            # convert to tensor without normalization; keep single channel
            m_tfms += [T.PILToTensor()]  # uint8, 1xHxW
            self.mask_tfms = T.Compose(m_tfms)

    def _index_cases(self) -> List[dict]:
        cases = []
        for case_dir in sorted(next(os.walk(self.root))[1]):  # IMDxxx folders
            croot = os.path.join(self.root, case_dir)

            # image
            img_dir = glob(os.path.join(croot, f"{case_dir}_Dermoscopic_Image"))[0]
            img_paths = glob(os.path.join(img_dir, "*.bmp"))
            if not img_paths:
                continue
            img_path = img_paths[0]

            # lesion mask
            lesion_dir = os.path.join(croot, f"{case_dir}_lesion")
            lesion_path = None
            cand = glob(os.path.join(lesion_dir, f"{case_dir}_lesion.bmp"))
            if cand:
                lesion_path = cand[0]

            # roi mask(s)
            roi_dir = os.path.join(croot, f"{case_dir}_roi")
            roi_paths = sorted(glob(os.path.join(roi_dir, f"{case_dir}_R*_Label*.bmp")))
            roi_path = None
            if roi_paths:
                if self.roi_preference == "R1":
                    # pick an R1 if present, else fall back to first
                    r1 = [p for p in roi_paths if os.path.basename(p).startswith(f"{case_dir}_R1")]
                    roi_path = r1[0] if r1 else roi_paths[0]
                else:  # largest area heuristic
                    roi_path = max(roi_paths, key=lambda p: Image.open(p).size[0] * Image.open(p).size[1])

            cases.append(
                {
                    "case": case_dir,
                    "image": img_path,
                    "lesion": lesion_path,
                    "roi": roi_path,
                }
            )
        if not cases:
            raise RuntimeError(f"No cases found under {self.root}")
        return cases

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _img_to_rgb(img_path: str) -> Image.Image:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

<<<<<<< HEAD
    @staticmethod
    def _mask_to_binary(mask_img: Image.Image) -> torch.Tensor:
        """
        Converts a grayscale mask PIL image to float tensor in {0,1}, shape 1xHxW.
        Assumes lesion/roi masks are white foreground on black background.
        """
        # After PILToTensor -> uint8 [0..255], shape 1xHxW
        # We'll threshold at >0
        tensor = T.functional.pil_to_tensor(mask_img)  # uint8, 1xHxW
        return (tensor > 0).float()

=======
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f
    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = self._img_to_rgb(item["image"])
        img = self.image_tfms(img)  # float [0..1], 3xHxW

        m = Image.open(item["lesion"]).convert("L")
        m = self.mask_tfms(m)  # uint8, 1xHxW
        lesion_t = (m > 0).float()  # binary mask {0,1}

        return {
            "image": img,
            "mask": lesion_t,
            "id": item["case"],
            "image_path": item["image"],
            "mask_path": item["lesion"],
        }


<<<<<<< HEAD
        if self.target == "lesion":
            if lesion_t is None:
                raise FileNotFoundError(f"No lesion mask found for case {item['case']}")
            mask = lesion_t.squeeze(0)
            meta = {"case": item["case"], "image_path": item["image"], "mask_path": item["lesion"]}
        elif self.target == "roi":
            if roi_t is None:
                raise FileNotFoundError(f"No ROI mask found for case {item['case']}")
            mask = roi_t.squeeze(0)
            meta = {"case": item["case"], "image_path": item["image"], "mask_path": item["roi"]}
        else:  # both
            if lesion_t is None or roi_t is None:
                raise FileNotFoundError(f"Incomplete masks for case {item['case']}")
            mask = torch.stack([lesion_t.squeeze(0), roi_t.squeeze(0)])
            meta = {
                "case": item["case"],
                "image_path": item["image"],
                "lesion_path": item["lesion"],
                "roi_path": item["roi"],
            }

        sample = {"image": img, "mask": mask.to(torch.long), "meta": meta}
        if self.return_paths:
            sample["paths"] = meta
        return sample
=======
def build_dataloaders(
        root: str,  # path to the DRIVE root (folder that contains "training")
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
    full_ds = PH2Dataset(
        root=root,
        target="lesion",  # 'lesion' | 'roi' | 'both'
        image_tfms=transform,  # or None to use defaults
        mask_tfms=mask_transform,  # or None to use defaults
        size=None,  # if you rely entirely on the custom transforms above
        roi_preference="R1",  # or "largest"
        return_paths=False
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
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f


def create_ph2_dataloaders(
    root: str,
    batch_size: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool | None = None,
    **dataset_kwargs: object,
) -> Dict[str, DataLoader]:
    """
    Convenience helper that instantiates PH2Dataset and returns train/val loaders.

<<<<<<< HEAD
    Parameters mirror PH2Dataset, with additional DataLoader hyperparameters.
    """
=======
if __name__ == "__main__":
    data_root = r"C:\Users\lucas\PycharmProjects\02516_Assignment_03\data\PH2_Dataset_images"  # <- change to your path
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f

    dataset_kwargs.setdefault("target", "lesion")
    dataset = PH2Dataset(root=root, **dataset_kwargs)

<<<<<<< HEAD
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1).")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("test_split must be in [0, 1).")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.")

    total_len = len(dataset)
    val_len = int(round(total_len * val_split))
    test_len = int(round(total_len * test_split))

    # ensure at least one sample when requested
    if val_split > 0 and val_len == 0:
        val_len = 1
    if test_split > 0 and test_len == 0:
        test_len = 1

    train_len = total_len - val_len - test_len
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
    val_dataset = None
    test_dataset = None
    if val_len > 0:
        val_dataset = subsets[idx]
        idx += 1
    if test_len > 0:
        test_dataset = subsets[idx]

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
=======
    train_loader, val_loader = build_dataloaders(
        data_root,
        batch_size=4,
        num_workers=2,
        val_ratio=0.2,
        transform=image_tfms,
        mask_transform=mask_tfms,
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f
    )
    loaders: Dict[str, DataLoader] = {"train": train_loader}

<<<<<<< HEAD
    if val_dataset is not None:
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return loaders


__all__ = ["PH2Dataset", "create_ph2_dataloaders"]
=======
    # Iterate
    for batch in train_loader:
        images = batch["image"]  # [B,3,H,W]
        masks = batch["mask"]  # [B,1,H,W]
        ids = batch["id"]
        print(images.shape, masks.shape, ids)
        # plot images
        plt.figure(figsize=(10, 5))
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))
            plt.title(f"Image {ids[i]}")
            plt.axis('off')
            plt.subplot(2, 4, i + 5)
            plt.imshow(masks[i, 0], cmap='gray')
            plt.title(f"Mask {ids[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        break
>>>>>>> d783e496fba88c51c8bf0ba0a618ab82d061c67f
