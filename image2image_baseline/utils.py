from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import copy
import json
import random

import numpy as np
import torch

from .losses import ssim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    psnr = 10.0 * torch.log10((data_range * data_range) / mse)
    return float(psnr.mean().item())


def batch_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    return float(ssim(prediction, target, data_range=data_range).item())


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    ema_state: dict[str, torch.Tensor] | None,
    scaler_state: dict | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "ema_state": ema_state,
            "scaler_state": scaler_state,
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def encode_npy_base64(array: np.ndarray) -> str:
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {key: value.clone() for key, value in state_dict.items()}

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None
