from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    diff = prediction - target
    return torch.mean(torch.sqrt(diff * diff + epsilon * epsilon))


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    padding = kernel_size // 2

    mu_x = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(prediction * prediction, kernel_size, 1, padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, kernel_size, 1, padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size, 1, padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / (denominator + 1e-12)).mean()


class CombinedRestorationLoss(nn.Module):
    def __init__(self, base_loss: str = "l1", ssim_weight: float = 0.0) -> None:
        super().__init__()
        if base_loss not in {"l1", "charbonnier"}:
            raise ValueError("base_loss must be 'l1' or 'charbonnier'.")
        self.base_loss = base_loss
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "l1":
            base = F.l1_loss(prediction, target)
        else:
            base = charbonnier_loss(prediction, target)

        if self.ssim_weight <= 0:
            return base

        ssim_term = 1.0 - ssim(
            prediction.clamp(0.0, 1.0),
            target.clamp(0.0, 1.0),
        )
        return base + self.ssim_weight * ssim_term
