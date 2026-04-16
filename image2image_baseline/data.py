from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TRAIN_LR_DIR = Path("train") / "train" / "NoisyLR"
TRAIN_GT_DIR = Path("train") / "train" / "GT"
TEST_LR_DIR = Path("Test_NoisyLR") / "NoisyLR"


def _load_2d_npy(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array at {path}, got shape {array.shape}.")
    return array


def _apply_pair_augmentations(
    lr: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        lr = np.flip(lr, axis=1)
        gt = np.flip(gt, axis=1)
    if random.random() < 0.5:
        lr = np.flip(lr, axis=0)
        gt = np.flip(gt, axis=0)
    rotations = random.randint(0, 3)
    if rotations:
        lr = np.rot90(lr, rotations)
        gt = np.rot90(gt, rotations)
    return np.ascontiguousarray(lr), np.ascontiguousarray(gt)


def discover_training_pairs(dataset_root: Path) -> list[str]:
    lr_dir = dataset_root / TRAIN_LR_DIR
    gt_dir = dataset_root / TRAIN_GT_DIR
    if not lr_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(
            f"Expected paired training directories under {dataset_root}."
        )

    lr_ids = {path.stem for path in lr_dir.glob("*.npy")}
    gt_ids = {path.stem for path in gt_dir.glob("*.npy")}
    shared_ids = sorted(lr_ids & gt_ids)
    if not shared_ids:
        raise FileNotFoundError(
            f"No matching .npy pairs found in {lr_dir} and {gt_dir}."
        )
    return shared_ids


def discover_test_inputs(dataset_root: Path) -> list[str]:
    test_dir = dataset_root / TEST_LR_DIR
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test directory at {test_dir}.")
    sample_ids = sorted(path.stem for path in test_dir.glob("*.npy"))
    if not sample_ids:
        raise FileNotFoundError(f"No test .npy inputs found in {test_dir}.")
    return sample_ids


def split_train_val(
    sample_ids: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


class PairedNpyDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        sample_ids: list[str],
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sample_ids = list(sample_ids)
        self.augment = augment
        self.seed = seed
        self.lr_dir = self.dataset_root / TRAIN_LR_DIR
        self.gt_dir = self.dataset_root / TRAIN_GT_DIR

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        lr = _load_2d_npy(self.lr_dir / f"{sample_id}.npy")
        gt = _load_2d_npy(self.gt_dir / f"{sample_id}.npy")

        if self.augment:
            lr, gt = _apply_pair_augmentations(lr, gt)

        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
        return {"lr": lr_tensor, "gt": gt_tensor, "sample_id": sample_id}


class TestNpyDataset(Dataset):
    def __init__(self, dataset_root: Path, sample_ids: list[str]) -> None:
        self.dataset_root = Path(dataset_root)
        self.sample_ids = list(sample_ids)
        self.lr_dir = self.dataset_root / TEST_LR_DIR

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        lr = _load_2d_npy(self.lr_dir / f"{sample_id}.npy")
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        return {"lr": lr_tensor, "sample_id": sample_id}


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_ids: list[str]
    val_ids: list[str]


def create_train_val_loaders(
    dataset_root: Path,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> LoaderBundle:
    sample_ids = discover_training_pairs(dataset_root)
    train_ids, val_ids = split_train_val(sample_ids, val_ratio=val_ratio, seed=seed)
    train_dataset = PairedNpyDataset(
        dataset_root=dataset_root,
        sample_ids=train_ids,
        augment=True,
        seed=seed,
    )
    val_dataset = PairedNpyDataset(
        dataset_root=dataset_root,
        sample_ids=val_ids,
        augment=False,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_ids=train_ids,
        val_ids=val_ids,
    )
