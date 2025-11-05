import os
from glob import glob
from typing import Callable, Optional, Tuple, List, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

import matplotlib.pyplot as plt


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
        return_paths: if True, also returns (img_path, mask_path_or_tuple)
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

    @staticmethod
    def _mask_to_binary(mask_img: Image.Image) -> torch.Tensor:
        """
        Converts a grayscale mask PIL image to float tensor in {0,1}, shape 1xHxW.
        Assumes lesion/roi masks are white foreground on black background.
        """
        # After PILToTensor -> uint8 [0..255], shape 1xHxW
        # We'll threshold at >0
        return None  # placeholder to satisfy lints (replaced below)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = self._img_to_rgb(item["image"])
        img = self.image_tfms(img)  # float [0..1], 3xHxW

        # Prepare masks
        lesion_t, roi_t = None, None

        if self.target in ("lesion", "both") and item["lesion"] is not None:
            m = Image.open(item["lesion"]).convert("L")
            m = self.mask_tfms(m)  # uint8, 1xHxW
            lesion_t = (m > 0).float()  # binary mask {0,1}

        if self.target in ("roi", "both") and item["roi"] is not None:
            m = Image.open(item["roi"]).convert("L")
            m = self.mask_tfms(m)
            roi_t = (m > 0).float()

        if self.target == "lesion":
            out = (img, lesion_t)
            if self.return_paths:
                out = (img, lesion_t, {"image": item["image"], "mask": item["lesion"]})
            return out
        elif self.target == "roi":
            out = (img, roi_t)
            if self.return_paths:
                out = (img, roi_t, {"image": item["image"], "mask": item["roi"]})
            return out
        else:  # both
            out = (img, {"lesion": lesion_t, "roi": roi_t})
            if self.return_paths:
                out = (img, {"lesion": lesion_t, "roi": roi_t},
                       {"image": item["image"], "lesion": item["lesion"], "roi": item["roi"]})
            return out


# ---------- Example usage ----------

if __name__ == "__main__":
    root = r"C:\Users\lucas\PycharmProjects\02516_Assignment_03\data\PH2_Dataset_images"  # <- change to your path

    # Optional: add augmentations on top of the defaults
    image_tfms = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
    ])
    mask_tfms = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor()
    ])

    ds = PH2Dataset(
        root=root,
        target="lesion",           # 'lesion' | 'roi' | 'both'
        image_tfms=image_tfms,   # or None to use defaults
        mask_tfms=mask_tfms,     # or None to use defaults
        size=None,               # if you rely entirely on the custom transforms above
        roi_preference="R1",     # or "largest"
        return_paths=False
    )

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    # Iterate
    for images, masks in loader:
        # images:  [B, 3, H, W]
        # masks:   dict with keys 'lesion' and 'roi', each [B, 1, H, W] (since target='both')
        # ... your training step ...
        print(images.shape, masks.shape) # torch.Size([1, 3, 384, 384]) torch.Size([1, 1, 384, 384]) torch.Size([1, 1, 384, 384])
        print(images.dtype, masks.dtype)
        # Take the first item in the batch
        img = images[0]  # [3, H, W]
        lesion = masks[0, 0]  # [H, W]

        # Convert image tensor (C,H,W) -> (H,W,C)
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Dermoscopic Image")
        axs[0].axis("off")

        axs[1].imshow(img_np)
        axs[1].imshow(lesion.cpu(), cmap="Reds", alpha=0.5)
        axs[1].set_title("Lesion Mask Overlay")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
        break

        break
