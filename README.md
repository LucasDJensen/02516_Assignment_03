# 02516_Assignment_03
02516 Introduction to Deep Learning in Computer Vision. Assignment 2. Image segmentation

## Model architecture scaffold

- `models/segmentation_model.py` contains a fully convolutional encoder-decoder that mirrors the assignment slide (defaults to binary foreground/background segmentation).
- The upcoming data loader should feed batches shaped like `{"image": tensor, "mask": tensor}`; see the docstring and the inline marker inside `forward`.
- Adjust `SegmentationModelConfig` to tune depth, channels, dropout, or the number of classes once dataset details arrive.
- `DataLoaders/PH2DataLoader.py` provides a dataset class and helper to build train/val/test loaders for PH2 (defaults to lesion masks, resized to 256×256).
- `DataLoaders/DRIVEDataLoader.py` mirrors the interface for the DRIVE retinal vessel dataset (with optional resizing and the correct vessel masks).
- `train.py` now supports both datasets and can switch between the FCN scaffold and the U-Net baseline while logging per-split metrics.

### Quick start (DTU HPC)

```bash
python train.py \
  --dataset ph2 \  # or drive
  --model fcn \    # or unet
  --data-root /dtu/datasets1/02516/PH2_Dataset_images \
  --device cuda \
  --val-split 0.2 \
  --test-split 0.1 \
  --epochs 5 \
  --batch-size 8 \
  --amp
```

If you're testing locally without the dataset mounted, the script will exit early and prompt you to point `--data-root` to an accessible copy.



```bash
uv sync
``` 
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
``` 


```bash
bsub -app c02516_1g.10gb < run_hpc_drive_fcn.sh
``` 
