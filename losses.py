from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("BCELoss", "DiceLoss", "FocalLoss", "BCELoss_TotalVariation")


class BCELoss(nn.Module):
    """Binary cross entropy loss with logits."""

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Handle multi-class case
        if y_pred.dim() == 4 and y_pred.size(1) > 1:
            # Use CrossEntropyLoss for multi-class
            return F.cross_entropy(y_pred, y_true.long(), weight=self.weight)
        else:
            # Binary case
            return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=self.weight)


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Handle multi-class case: convert logits to probabilities
        if y_pred.dim() == 4 and y_pred.size(1) > 1:
            # Multi-class: y_pred shape [B, C, H, W], y_true shape [B, H, W]
            probs = torch.softmax(y_pred, dim=1)  # [B, C, H, W]
            # Convert y_true to one-hot encoding
            num_classes = y_pred.size(1)
            y_true_one_hot = F.one_hot(y_true.long(), num_classes=num_classes)  # [B, H, W, C]
            y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

            intersection = (probs * y_true_one_hot).sum(dim=(2, 3))  # [B, C]
            denominator = probs.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))  # [B, C]
            dice = (2 * intersection + self.eps) / (denominator + self.eps)  # [B, C]
            # Average over classes and batch
            return 1 - dice.mean()
        else:
            # Binary case with single channel: y_pred shape [B, 1, H, W]
            probs = torch.sigmoid(y_pred)
            intersection = (probs * y_true).sum(dim=(1, 2, 3))
            denominator = probs.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
            dice = (2 * intersection + self.eps) / (denominator + self.eps)
            return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss as introduced by Lin et al. (2017)."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Handle multi-class case
        if y_pred.dim() == 4 and y_pred.size(1) > 1:
            # Multi-class focal loss
            ce_loss = F.cross_entropy(y_pred, y_true.long(), reduction="none")
            probs = torch.softmax(y_pred, dim=1)
            # Gather probabilities of true class
            y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.size(1))
            p_t = (probs * y_true_one_hot.permute(0, 3, 1, 2)).sum(dim=1)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                focal_weight = focal_weight * self.alpha
            loss = focal_weight * ce_loss
            return loss.mean()
        else:
            # Binary case
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
            probs = torch.sigmoid(y_pred)
            p_t = y_true * probs + (1 - y_true) * (1 - probs)
            focal_weight = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
                focal_weight = focal_weight * alpha_t
            loss = focal_weight * bce
            return loss.mean()


class BCELoss_TotalVariation(nn.Module):
    """Binary cross entropy with total variation regularisation."""

    def __init__(self, tv_weight: float = 0.1) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tv_weight = tv_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(y_pred, y_true)
        probs = torch.sigmoid(y_pred)
        tv_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
        tv_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
        tv_loss = tv_h + tv_w
        return bce_loss + self.tv_weight * tv_loss
