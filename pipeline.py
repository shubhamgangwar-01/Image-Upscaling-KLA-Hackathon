"""
pipeline.py — end-to-end CLI for the Image Upscaling pipeline.

Stages:
  download   Download and extract the dataset
  train      Train the BicubicResidualSR model
  predict    Run inference and produce a submission CSV
  run        Execute all three stages in sequence

Examples
--------
# Full pipeline with defaults:
    python pipeline.py run --data-dir data --output-dir runs/exp1

# Individual stages:
    python pipeline.py download --extract-dir data
    python pipeline.py train --data-dir data --output-dir runs/exp1 --epochs 60 --amp
    python pipeline.py predict --data-dir data --checkpoint runs/exp1/best.pt --submission-path submission.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


# ── Stage: download ────────────────────────────────────────────────────────────

def add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        default=None,
        help="Dataset ZIP URL (defaults to Hugging Face).",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data") / "image2image.zip",
        help="Where to store the ZIP file. (default: data/image2image.zip)",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data"),
        help="Directory to extract the dataset into. (default: data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; only extract an existing ZIP.",
    )


def run_download(args: argparse.Namespace) -> None:
    cmd: list[str] = [sys.executable, "download_dataset.py"]
    if args.url:
        cmd += ["--url", args.url]
    cmd += ["--zip-path", str(args.zip_path)]
    cmd += ["--extract-dir", str(args.extract_dir)]
    if args.skip_download:
        cmd.append("--skip-download")
    _run(cmd)


# ── Stage: train ───────────────────────────────────────────────────────────────

def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "baseline",
                        help="Where to save checkpoints and logs. (default: runs/baseline)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs. (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size. (default: 32)")
    parser.add_argument("--features", type=int, default=96,
                        help="Feature channels in the network. (default: 96)")
    parser.add_argument("--blocks", type=int, default=12,
                        help="Number of residual blocks. (default: 12)")
    parser.add_argument("--base-loss", choices=["l1", "charbonnier"], default="charbonnier",
                        help="Base loss function. (default: charbonnier)")
    parser.add_argument("--ssim-weight", type=float, default=0.2,
                        help="SSIM loss weight (0 to disable). (default: 0.2)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Initial learning rate. (default: 5e-4)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs. (default: 5)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience (0 = off). (default: 15)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (CUDA only).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint path.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. (default: 0)")


def run_train(args: argparse.Namespace) -> None:
    cmd: list[str] = [
        sys.executable, "train.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--features", str(args.features),
        "--blocks", str(args.blocks),
        "--base-loss", args.base_loss,
        "--ssim-weight", str(args.ssim_weight),
        "--learning-rate", str(args.learning_rate),
        "--warmup-epochs", str(args.warmup_epochs),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
    ]
    if args.amp:
        cmd.append("--amp")
    if args.resume is not None:
        cmd += ["--resume", str(args.resume)]
    _run(cmd)


# ── Stage: predict ─────────────────────────────────────────────────────────────

def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted dataset root.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best.pt or last.pt.")
    parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"),
                        help="Output CSV path. (default: submission.csv)")
    parser.add_argument("--prediction-dir", type=Path, default=None,
                        help="Also save raw .npy predictions here.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size. (default: 16)")
    parser.add_argument("--tta", choices=["none", "x8"], default="none",
                        help="Test-time augmentation. (default: none)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision inference (CUDA only).")
    parser.add_argument("--template-submission", type=Path, default=None,
                        help="Optional CSV template to preserve row-ID order.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. (default: 0)")


def run_predict(args: argparse.Namespace) -> None:
    cmd: list[str] = [
        sys.executable, "predict.py",
        "--data-dir", str(args.data_dir),
        "--checkpoint", str(args.checkpoint),
        "--submission-path", str(args.submission_path),
        "--batch-size", str(args.batch_size),
        "--tta", args.tta,
        "--num-workers", str(args.num_workers),
    ]
    if args.prediction_dir is not None:
        cmd += ["--prediction-dir", str(args.prediction_dir)]
    if args.amp:
        cmd.append("--amp")
    if args.template_submission is not None:
        cmd += ["--template-submission", str(args.template_submission)]
    _run(cmd)


# ── Stage: run (all) ───────────────────────────────────────────────────────────

def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Extracted dataset root. (default: data)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "baseline",
                        help="Where to save checkpoints and logs. (default: runs/baseline)")
    parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"),
                        help="Output CSV path. (default: submission.csv)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--base-loss", choices=["l1", "charbonnier"], default="charbonnier")
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--tta", choices=["none", "x8"], default="none")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (assumes data already exists).")
    parser.add_argument("--num-workers", type=int, default=0)


def run_all(args: argparse.Namespace) -> None:
    # ── 1. Download ─────────────────────────────────────────────────────────────
    if not args.skip_download:
        print("=" * 60)
        print("STAGE 1 / 3 — Downloading dataset")
        print("=" * 60)
        download_cmd: list[str] = [
            sys.executable, "download_dataset.py",
            "--extract-dir", str(args.data_dir),
        ]
        _run(download_cmd)
    else:
        print("Skipping download stage.")

    # ── 2. Train ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 2 / 3 — Training model")
    print("=" * 60)
    train_cmd: list[str] = [
        sys.executable, "train.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--features", str(args.features),
        "--blocks", str(args.blocks),
        "--base-loss", args.base_loss,
        "--ssim-weight", str(args.ssim_weight),
        "--learning-rate", str(args.learning_rate),
        "--warmup-epochs", str(args.warmup_epochs),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
    ]
    if args.amp:
        train_cmd.append("--amp")
    _run(train_cmd)

    # ── 3. Predict ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 3 / 3 — Running inference")
    print("=" * 60)
    checkpoint = args.output_dir / "best.pt"
    predict_cmd: list[str] = [
        sys.executable, "predict.py",
        "--data-dir", str(args.data_dir),
        "--checkpoint", str(checkpoint),
        "--submission-path", str(args.submission_path),
        "--tta", args.tta,
        "--num-workers", str(args.num_workers),
    ]
    if args.amp:
        predict_cmd.append("--amp")
    _run(predict_cmd)

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Submission saved to: {args.submission_path}")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p_download = subparsers.add_parser("download", help="Download and extract the dataset.")
    add_download_args(p_download)

    p_train = subparsers.add_parser("train", help="Train the SR model.")
    add_train_args(p_train)

    p_predict = subparsers.add_parser("predict", help="Run inference and build a submission CSV.")
    add_predict_args(p_predict)

    p_run = subparsers.add_parser("run", help="Run all three stages end-to-end.")
    add_run_args(p_run)

    args = parser.parse_args()

    if args.stage == "download":
        run_download(args)
    elif args.stage == "train":
        run_train(args)
    elif args.stage == "predict":
        run_predict(args)
    elif args.stage == "run":
        run_all(args)


if __name__ == "__main__":
    main()
