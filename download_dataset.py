from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import requests
from tqdm import tqdm


DEFAULT_URL = (
    "https://huggingface.co/datasets/shubhamgangwar-01/Image2Image/resolve/main/"
    "kla-ai-hack-x-iit-h-2026.zip"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Downloading dataset",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Image2Image dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset ZIP URL.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data") / "image2image.zip",
        help="Where to store the ZIP file.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data"),
        help="Directory where the ZIP should be extracted.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the ZIP already exists and only extract it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        download_file(args.url, args.zip_path)
    extract_zip(args.zip_path, args.extract_dir)
    print(f"Dataset extracted under: {args.extract_dir.resolve()}")


if __name__ == "__main__":
    main()

