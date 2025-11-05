# How to Use

## What It Is

The **U-Net** is a convolutional neural network designed for **pixel-wise image segmentation**.  
It has an **encoder** that captures context through downsampling and a **decoder** that restores spatial detail through upsampling and skip connections.  
The final 1×1 convolution outputs a segmentation mask (logits), which can later be passed through a sigmoid for probabilities.

---

## Import & Instantiate

```python
from models.unet import UNet

# 4 down/upsampling levels (depth=4 → input H and W should be divisible by 16)
model = UNet(in_channels=3, out_channels=1, base_c=64, depth=4)
```

## Forward Pass Example

```
import torch
x = torch.randn(2, 3, 576, 768)   # [batch, channels, height, width]
y = model(x)                      # Output: [2, 1, 576, 768] (logits)

```

Use torch.sigmoid(y) to obtain probabilities or BCEWithLogitsLoss during training.
