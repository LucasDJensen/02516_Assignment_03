# 02516_Assignment_03
02516 Introduction to Deep Learning in Computer Vision. Assignment 2. Image segmentation

## Model architecture scaffold

- `models/segmentation_model.py` contains a fully convolutional encoder-decoder that mirrors the assignment slide (defaults to binary foreground/background segmentation).
- The upcoming data loader should feed batches shaped like `{"image": tensor, "mask": tensor}`; see the docstring and the inline marker inside `forward`.
- Adjust `SegmentationModelConfig` to tune depth, channels, dropout, or the number of classes once dataset details arrive.
- `train.py` is a dummy entry point that can switch between the FCN scaffold and the future `unet` implementation (currently a placeholder).
