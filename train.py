from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from image2image_baseline.data import create_train_val_loaders
from image2image_baseline.losses import CombinedRestorationLoss
from image2image_baseline.model import BicubicResidualSR
from image2image_baseline.utils import (
    ModelEma,
    batch_psnr,
    batch_ssim,
    save_checkpoint,
    save_json,
    set_seed,
)


# ── Set your dataset path here ───────────────────────────────────────────────
DEFAULT_DATA_DIR: Path | None = None   # e.g. Path("/path/to/dataset")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Image2Image baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        required=DEFAULT_DATA_DIR is None,
        help="Extracted dataset root. Can also be set via DEFAULT_DATA_DIR in train.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "baseline")
    # More epochs to let the deeper model converge fully.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    # Slightly lower LR pairs better with warmup + larger model.
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Larger capacity: more features and more residual blocks.
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=12)
    # Charbonnier is smoother than L1 and typically gives better PSNR.
    parser.add_argument("--base-loss", choices=["l1", "charbonnier"], default="charbonnier")
    # Higher SSIM weight improves structural fidelity.
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    # Stronger EMA smoothing for a more stable best model.
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Linear LR warmup to stabilise the first few epochs.
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of linear-warmup epochs before cosine decay starts.")
    # Floor for cosine annealing so the LR never reaches zero.
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate at the end of cosine decay.")
    # Early stopping: stop if val PSNR does not improve for this many epochs.
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience in epochs (0 = disabled).")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision on CUDA for faster, lower-memory training.",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint, but use a fresh optimizer and scheduler.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ema: ModelEma | None,
    grad_clip: float,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    show_progress: bool,
) -> float:
    model.train()
    running_loss = 0.0
    num_items = 0
    for batch in tqdm(loader, desc="Train", leave=False, disable=not show_progress):
        lr = batch["lr"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(lr)
            loss = criterion(prediction, gt)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = lr.size(0)
        running_loss += float(loss.item()) * batch_size
        num_items += batch_size
    return running_loss / max(1, num_items)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ema: ModelEma | None = None,
    use_amp: bool = False,
    show_progress: bool = True,
) -> dict[str, float]:
    if ema is not None:
        ema.apply_to(model)
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_items = 0

    for batch in tqdm(loader, desc="Val", leave=False, disable=not show_progress):
        lr = batch["lr"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(lr)
            loss = criterion(prediction, gt)

        clipped_prediction = prediction.clamp(0.0, 1.0)
        clipped_gt = gt.clamp(0.0, 1.0)

        batch_size = lr.size(0)
        total_loss += float(loss.item()) * batch_size
        total_psnr += batch_psnr(clipped_prediction, clipped_gt) * batch_size
        total_ssim += batch_ssim(clipped_prediction, clipped_gt) * batch_size
        num_items += batch_size

    metrics = {
        "loss": total_loss / max(1, num_items),
        "psnr": total_psnr / max(1, num_items),
        "ssim": total_ssim / max(1, num_items),
    }
    if ema is not None:
        ema.restore(model)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    show_progress = sys.stderr.isatty()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_train_val_loaders(
        dataset_root=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = BicubicResidualSR(features=args.features, blocks=args.blocks).to(device)
    criterion = CombinedRestorationLoss(
        base_loss=args.base_loss,
        ssim_weight=args.ssim_weight,
    )
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Warmup phase: linear ramp from lr/1000 → lr over warmup_epochs.
    # Then cosine decay from lr down to min_lr over the remaining epochs.
    warmup_epochs = max(1, args.warmup_epochs)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_psnr = float("-inf")
    no_improve_epochs = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.load_state_dict(checkpoint["ema_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("metrics", {}).get("psnr", float("-inf")))
    elif args.init_model is not None:
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)
    config["amp_enabled"] = use_amp
    config["train_size"] = len(loaders.train_ids)
    config["val_size"] = len(loaders.val_ids)
    save_json(args.output_dir / "config.json", config)

    history_path = args.output_dir / "history.json"
    history: list[dict[str, float | int]] = []
    if args.resume is not None and history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle).get("history", [])
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=loaders.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            ema=ema,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            scaler=scaler,
            show_progress=show_progress,
        )
        metrics = evaluate(
            model=model,
            loader=loaders.val_loader,
            criterion=criterion,
            device=device,
            ema=ema,
            use_amp=use_amp,
            show_progress=show_progress,
        )
        scheduler.step()
        epoch_seconds = time.time() - start

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_psnr": metrics["psnr"],
            "val_ssim": metrics["ssim"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_summary)
        save_json(history_path, {"history": history})

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={metrics['loss']:.5f} "
            f"val_psnr={metrics['psnr']:.3f} "
            f"val_ssim={metrics['ssim']:.4f} "
            f"lr={current_lr:.2e} "
            f"time={epoch_seconds:.1f}s"
        )

        save_checkpoint(
            checkpoint_path=args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_state=ema.state_dict() if ema is not None else None,
            scaler_state=scaler.state_dict() if use_amp else None,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            no_improve_epochs = 0
            save_checkpoint(
                checkpoint_path=args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema_state=ema.state_dict() if ema is not None else None,
                scaler_state=scaler.state_dict() if use_amp else None,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            print(f"  --> New best PSNR: {best_psnr:.3f} dB (saved best.pt)")
        else:
            no_improve_epochs += 1
            if args.patience > 0 and no_improve_epochs >= args.patience:
                print(
                    f"Early stopping: val PSNR did not improve for {args.patience} epochs. "
                    f"Best PSNR: {best_psnr:.3f} dB"
                )
                break

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.3f} dB")


if __name__ == "__main__":
    main()
