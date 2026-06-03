"""Prepare image and label files in YOLOv8 dataset format."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation arguments."""
    parser = argparse.ArgumentParser(
        description="Organize images and YOLO labels for YOLOv8 training."
    )
    parser.add_argument("--images", required=True, help="Directory containing source images.")
    parser.add_argument(
        "--labels", required=True, help="Directory containing YOLO .txt label files."
    )
    parser.add_argument("--output", default="dataset_yolo", help="Output dataset directory.")
    parser.add_argument(
        "--val-ratio", default=0.2, type=float, help="Validation split ratio from 0 to 1."
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Include images without labels by creating empty .txt files.",
    )
    return parser.parse_args()


def validate_args(images_dir: Path, labels_dir: Path, val_ratio: float) -> None:
    """Validate input paths and split ratio."""
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {labels_dir}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be greater than 0 and less than 1.")


def collect_images(images_dir: Path) -> list[Path]:
    """Collect supported image files from a directory."""
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def create_output_dirs(output_dir: Path) -> None:
    """Create YOLOv8 image and label split directories."""
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def split_images(images: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split image paths into train and validation sets."""
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_images = shuffled[:val_count]
    train_images = shuffled[val_count:]
    if not train_images:
        raise ValueError("Dataset is too small to create a non-empty train split.")
    return train_images, val_images


def copy_split(
    images: list[Path],
    labels_dir: Path,
    output_dir: Path,
    split: str,
    allow_empty_labels: bool,
) -> int:
    """Copy images and matching labels into one YOLOv8 split."""
    copied = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists() and not allow_empty_labels:
            print(f"[INFO] Skipping image without label: {image_path.name}")
            continue

        target_image = output_dir / "images" / split / image_path.name
        target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        if label_path.exists():
            shutil.copy2(label_path, target_label)
        else:
            target_label.write_text("", encoding="utf-8")
        copied += 1
    return copied


def main() -> int:
    """Prepare a YOLOv8 dataset directory."""
    args = parse_args()

    try:
        images_dir = Path(args.images)
        labels_dir = Path(args.labels)
        output_dir = Path(args.output)
        validate_args(images_dir, labels_dir, args.val_ratio)

        images = collect_images(images_dir)
        if not images:
            raise ValueError(f"No supported images found in: {images_dir}")

        train_images, val_images = split_images(images, args.val_ratio, args.seed)
        create_output_dirs(output_dir)

        train_count = copy_split(
            train_images, labels_dir, output_dir, "train", args.allow_empty_labels
        )
        val_count = copy_split(val_images, labels_dir, output_dir, "val", args.allow_empty_labels)

        print(f"[INFO] Prepared dataset at: {output_dir}")
        print(f"[INFO] Train images copied: {train_count}")
        print(f"[INFO] Validation images copied: {val_count}")
        print("[INFO] Update train/dataset.yaml if class names or dataset path differ.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
