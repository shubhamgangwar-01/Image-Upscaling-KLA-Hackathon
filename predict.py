from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from image2image_baseline.data import TestNpyDataset, discover_test_inputs
from image2image_baseline.model import BicubicResidualSR
from image2image_baseline.utils import encode_npy_base64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and build a submission file."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted dataset root.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Optional directory to also save raw predicted .npy arrays.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision on CUDA for faster inference.",
    )
    parser.add_argument(
        "--tta",
        choices=["none", "x8"],
        default="none",
        help="Optional test-time augmentation self-ensemble.",
    )
    parser.add_argument(
        "--template-submission",
        type=Path,
        default=None,
        help="Optional CSV template to reuse the exact id column ordering.",
    )
    return parser.parse_args()


def load_template_ids(path: Path) -> list[int]:
    csv.field_size_limit(10_000_000)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain an 'id' column.")
        return [int(row["id"]) for row in reader]


def apply_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode >= 4:
        x = x.transpose(-2, -1)
    if mode % 4 == 1:
        x = torch.flip(x, dims=[-1])
    elif mode % 4 == 2:
        x = torch.flip(x, dims=[-2])
    elif mode % 4 == 3:
        x = torch.flip(x, dims=[-2, -1])
    return x


def invert_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode % 4 == 1:
        x = torch.flip(x, dims=[-1])
    elif mode % 4 == 2:
        x = torch.flip(x, dims=[-2])
    elif mode % 4 == 3:
        x = torch.flip(x, dims=[-2, -1])
    if mode >= 4:
        x = x.transpose(-2, -1)
    return x


def predict_batch(
    model: torch.nn.Module,
    lr: torch.Tensor,
    tta: str,
    use_amp: bool = False,
) -> torch.Tensor:
    device_type = lr.device.type
    if tta == "none":
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            return model(lr)

    outputs = []
    for mode in range(8):
        augmented = apply_tta(lr, mode)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            prediction = model(augmented)
        outputs.append(invert_tta(prediction, mode))
    return torch.stack(outputs, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = BicubicResidualSR(
        features=int(config.get("features", 64)),
        blocks=int(config.get("blocks", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sample_ids = discover_test_inputs(args.data_dir)
    if args.template_submission is not None:
        row_ids = load_template_ids(args.template_submission)
        if len(row_ids) != len(sample_ids):
            raise ValueError(
                "Template submission row count does not match the number of test samples."
            )
    else:
        row_ids = list(range(args.start_id, args.start_id + len(sample_ids)))

    dataset = TestNpyDataset(args.data_dir, sample_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[tuple[int, str]] = []
    if args.prediction_dir is not None:
        args.prediction_dir.mkdir(parents=True, exist_ok=True)

    id_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            lr = batch["lr"].to(device, non_blocking=True)
            predictions = (
                predict_batch(model, lr, args.tta, use_amp=use_amp)
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            sample_id_batch = batch["sample_id"]

            for sample_name, prediction in zip(sample_id_batch, predictions):
                image = prediction[0].astype(np.float32)
                rows.append((row_ids[id_index], encode_npy_base64(image)))
                if args.prediction_dir is not None:
                    np.save(args.prediction_dir / f"{sample_name}.npy", image, allow_pickle=False)
                id_index += 1

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    with args.submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "npy_base64"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {args.submission_path.resolve()}")


if __name__ == "__main__":
    main()
