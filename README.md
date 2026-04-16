# Image Upscaling — KLA Hackathon

A PyTorch pipeline for **x2 grayscale super-resolution + denoising** built for the KLA AI Hackathon @ IIT-H 2026.

- **Input:** 128×128 noisy low-resolution `.npy` arrays  
- **Target:** 256×256 clean high-resolution `.npy` arrays  
- **Architecture:** Bicubic-residual SR network with pixel-shuffle upsampling  
- **Loss:** Charbonnier + SSIM (configurable)  
- **Metrics:** PSNR and SSIM reported per epoch  

---

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are recommended. A CUDA-capable GPU is optional but speeds up training significantly.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/shubhamgangwar-01/Image-Upscaling-KLA-Hackathon.git
cd Image-Upscaling-KLA-Hackathon
pip install -r requirements.txt
```

### 2. One-command pipeline (recommended)

`pipeline.py` is a single entry point that runs all three stages — download, train, and predict — end-to-end:

```bash
python pipeline.py run --data-dir data --output-dir runs/exp1 --amp
```

You can also run each stage individually:

```bash
# Download only
python pipeline.py download --extract-dir data

# Train only
python pipeline.py train --data-dir data --output-dir runs/exp1 --epochs 60 --amp

# Predict only
python pipeline.py predict --data-dir data --checkpoint runs/exp1/best.pt --submission-path submission.csv
```

Run `python pipeline.py <stage> --help` for the full list of options for any stage.

---

### 3. Download and extract the dataset (manual)

```bash
python download_dataset.py --extract-dir data
```

This downloads the dataset ZIP from Hugging Face and extracts it. Expected layout after extraction:

```
data/
  train/train/NoisyLR/*.npy   ← noisy low-res training inputs (128×128)
  train/train/GT/*.npy        ← clean high-res training targets (256×256)
  Test_NoisyLR/NoisyLR/*.npy  ← test inputs (128×128)
```

If you already have the ZIP:

```bash
python download_dataset.py --skip-download --zip-path data/image2image.zip --extract-dir data
```

### 4. Train

```bash
python train.py --data-dir data --output-dir runs/baseline --epochs 60 --batch-size 32
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Extracted dataset root |
| `--output-dir` | `runs/baseline` | Where checkpoints and logs are saved |
| `--epochs` | `60` | Total training epochs |
| `--batch-size` | `32` | Batch size (reduce if OOM) |
| `--features` | `96` | Number of feature channels in the network |
| `--blocks` | `12` | Number of residual blocks |
| `--base-loss` | `charbonnier` | `l1` or `charbonnier` |
| `--ssim-weight` | `0.2` | Weight for SSIM loss term (0 to disable) |
| `--learning-rate` | `5e-4` | Initial learning rate |
| `--warmup-epochs` | `5` | Linear LR warmup before cosine decay |
| `--patience` | `15` | Early-stopping patience (0 = disabled) |
| `--amp` | *(flag)* | Enable mixed-precision training (CUDA only) |
| `--resume` | `None` | Resume from a checkpoint path |

Training outputs saved to `--output-dir`:

```
runs/baseline/
  config.json     ← hyperparameters used
  history.json    ← per-epoch metrics
  last.pt         ← latest checkpoint
  best.pt         ← best validation PSNR checkpoint
```

**Example with recommended settings:**

```bash
python train.py \
  --data-dir data \
  --output-dir runs/best \
  --epochs 60 \
  --batch-size 32 \
  --features 96 \
  --blocks 12 \
  --base-loss charbonnier \
  --ssim-weight 0.2 \
  --amp
```

### 5. Predict and generate submission

```bash
python predict.py \
  --data-dir data \
  --checkpoint runs/baseline/best.pt \
  --submission-path submission.csv
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Extracted dataset root |
| `--checkpoint` | *(required)* | Path to `best.pt` or `last.pt` |
| `--submission-path` | `submission.csv` | Output CSV path |
| `--prediction-dir` | `None` | Optional: also save predicted `.npy` files |
| `--batch-size` | `16` | Inference batch size |
| `--tta` | `none` | Test-time augmentation: `none` or `x8` |
| `--template-submission` | `None` | Use a fixed row-ID CSV template |
| `--amp` | *(flag)* | Enable mixed-precision inference (CUDA only) |

The output `submission.csv` has columns `id` and `npy_base64`.

**With TTA (slightly better predictions, ~8× slower):**

```bash
python predict.py \
  --data-dir data \
  --checkpoint runs/baseline/best.pt \
  --submission-path submission.csv \
  --tta x8
```

**Also save raw `.npy` predictions:**

```bash
python predict.py \
  --data-dir data \
  --checkpoint runs/baseline/best.pt \
  --submission-path submission.csv \
  --prediction-dir preds/
```

---

## Project Structure

```
.
├── pipeline.py                  # End-to-end CLI: download → train → predict
├── download_dataset.py          # Downloads and extracts the dataset
├── train.py                     # Training loop with EMA, LR warmup, early stopping
├── predict.py                   # Inference + CSV submission builder
├── requirements.txt             # Python dependencies
└── image2image_baseline/
    ├── __init__.py
    ├── data.py                  # Dataset classes and data loaders
    ├── model.py                 # BicubicResidualSR architecture
    ├── losses.py                # Charbonnier loss + SSIM
    └── utils.py                 # EMA, PSNR/SSIM metrics, checkpoint helpers
```

---

## Model Architecture

`BicubicResidualSR` is a lightweight residual super-resolution network:

1. **Bicubic baseline** — the input is upscaled 2× via bicubic interpolation  
2. **Residual body** — a stack of `ResidualBlock` modules (Conv → GELU → Conv + skip)  
3. **Pixel-shuffle upsampling** — sub-pixel convolution for learned upscaling  
4. **Residual prediction** — the network predicts a residual added to the bicubic baseline  

Default config: 96 features, 12 residual blocks (~5M parameters).

---

## Recommended Upgrades

| Upgrade | Expected gain |
|---|---|
| Replace model with SwinIR or Restormer | Large PSNR boost |
| Use `--tta x8` at inference | ~+0.2 dB PSNR |
| Increase `--features` and `--blocks` | Moderate gain with more GPU memory |
| Add perceptual / frequency loss | Better visual quality |
| Patch-based training with random crops | More diverse augmentation |

---

## Dataset

The dataset is hosted on Hugging Face:  
[shubhamgangwar-01/Image2Image](https://huggingface.co/datasets/shubhamgangwar-01/Image2Image)

It is a paired grayscale image restoration dataset:
- 3200 training pairs (NoisyLR 128×128 → GT 256×256)
- ~1600 test inputs (NoisyLR 128×128, GT withheld)
