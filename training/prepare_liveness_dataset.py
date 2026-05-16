from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import IMAGE_EXTENSIONS, label_from_name, parse_label

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
SPLIT_NAMES = {"train", "dev", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class RawSample:
    path: Path
    label: int
    source: str
    split: str | None = None
    subject: str = ""
    attack_type: str = ""
    device: str = ""
    lighting: str = ""
    environment: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert downloaded liveness datasets into TinyLiveness layout."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset source folder or CSV manifest. Repeat for multiple sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--video-stride", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=40)
    parser.add_argument("--crop-face", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--source must be formatted as NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("source name cannot be empty")
    source_path = Path(path.strip())
    if not source_path.exists():
        raise FileNotFoundError(f"source path not found: {source_path}")
    return name, source_path


def load_samples(source_name: str, source_path: Path) -> list[RawSample]:
    if source_path.is_file():
        return load_manifest(source_name, source_path)
    return scan_folder(source_name, source_path)


def load_manifest(source_name: str, manifest_path: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        for row in reader:
            path = Path(row["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            samples.append(
                RawSample(
                    path=path,
                    label=parse_label(row["label"]),
                    source=row.get("source") or source_name,
                    split=normalize_split(row.get("split", "")) or None,
                    subject=row.get("subject", ""),
                    attack_type=row.get("attack_type", ""),
                    device=row.get("device", ""),
                    lighting=row.get("lighting", ""),
                    environment=row.get("environment", ""),
                )
            )
    return samples


def scan_folder(source_name: str, root: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue

        parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
        label = first_label(parts)
        if label is None:
            continue
        split = first_split(parts)
        samples.append(
            RawSample(
                path=path,
                label=label,
                source=source_name,
                split=split,
                subject=guess_subject(parts),
                attack_type=guess_attack_type(parts, label),
                device=guess_tag(parts, {"phone", "mobile", "webcam", "laptop", "pc", "tablet", "camera"}),
                lighting=guess_tag(parts, {"bright", "dark", "low_light", "normal", "back", "indoor", "outdoor"}),
                environment=guess_tag(parts, {"indoor", "outdoor", "office", "home", "lab"}),
            )
        )
    return samples


def first_label(parts: Iterable[str]) -> int | None:
    for part in parts:
        label = label_from_name(part)
        if label is not None:
            return label
    return None


def first_split(parts: Iterable[str]) -> str | None:
    for part in parts:
        split = normalize_split(part)
        if split:
            return split
    return None


def normalize_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"dev", "valid", "validation"}:
        return "val"
    if normalized in {"train", "val", "test"}:
        return normalized
    return ""


def guess_subject(parts: list[str]) -> str:
    for part in parts:
        if any(token in part for token in ("subject", "client", "person", "user", "id")):
            return part
    return ""


def guess_attack_type(parts: list[str], label: int) -> str:
    if label == 1:
        return "live"
    attack_keywords = (
        "print",
        "photo",
        "poster",
        "screen",
        "replay",
        "video",
        "mask",
        "phone",
        "tablet",
        "pc",
        "laptop",
    )
    return guess_tag(parts, set(attack_keywords)) or "spoof"


def guess_tag(parts: list[str], options: set[str]) -> str:
    for part in parts:
        normalized = part.replace("-", "_")
        for option in options:
            if option in normalized:
                return normalized
    return ""


def split_for_sample(sample: RawSample, val_ratio: float, test_ratio: float) -> str:
    if sample.split in {"train", "val", "test"}:
        return sample.split

    key = sample.subject or str(sample.path)
    bucket = stable_bucket(f"{sample.source}:{key}")
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def output_stem(sample: RawSample, frame_index: int | None = None) -> str:
    digest = hashlib.sha1(str(sample.path.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = f"{sample.source}_{sample.path.stem}_{digest}"
    if frame_index is not None:
        stem = f"{stem}_f{frame_index:06d}"
    return sanitize(stem)


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def save_image(
    image: Image.Image,
    output_path: Path,
    image_size: int,
    crop_face: bool,
) -> None:
    image = image.convert("RGB")
    if crop_face:
        image = crop_or_center_face(image)
    image = image.resize((image_size, image_size), Image.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def crop_or_center_face(image: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError:
        return center_square(image)

    array = np.asarray(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
    if len(faces) == 0:
        return center_square(image)

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin = int(max(w, h) * 0.25)
    left = max(x - margin, 0)
    top = max(y - margin, 0)
    right = min(x + w + margin, image.width)
    bottom = min(y + h + margin, image.height)
    return image.crop((left, top, right, bottom))


def center_square(image: Image.Image) -> Image.Image:
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def convert_image_sample(
    sample: RawSample,
    split: str,
    output_dir: Path,
    image_size: int,
    crop_face: bool,
) -> list[dict[str, str]]:
    label_name = "live" if sample.label == 1 else "spoof"
    relative_path = Path(split) / label_name / f"{output_stem(sample)}.jpg"
    with Image.open(sample.path) as image:
        save_image(image, output_dir / relative_path, image_size, crop_face)
    return [manifest_row(sample, split, relative_path)]


def convert_video_sample(
    sample: RawSample,
    split: str,
    output_dir: Path,
    image_size: int,
    crop_face: bool,
    video_stride: int,
    max_frames: int,
) -> list[dict[str, str]]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("video conversion requires opencv-python") from exc

    rows: list[dict[str, str]] = []
    label_name = "live" if sample.label == 1 else "spoof"
    capture = cv2.VideoCapture(str(sample.path))
    frame_index = 0
    kept = 0
    try:
        while capture.isOpened():
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % max(video_stride, 1) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                relative_path = Path(split) / label_name / f"{output_stem(sample, frame_index)}.jpg"
                save_image(image, output_dir / relative_path, image_size, crop_face)
                rows.append(manifest_row(sample, split, relative_path, frame_index))
                kept += 1
                if kept >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
    return rows


def manifest_row(
    sample: RawSample,
    split: str,
    relative_path: Path,
    frame_index: int | None = None,
) -> dict[str, str]:
    return {
        "path": relative_path.as_posix(),
        "label": "live" if sample.label == 1 else "spoof",
        "split": split,
        "source": sample.source,
        "subject": sample.subject,
        "attack_type": sample.attack_type,
        "device": sample.device,
        "lighting": sample.lighting,
        "environment": sample.environment,
        "original_path": str(sample.path),
        "frame_index": "" if frame_index is None else str(frame_index),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "path",
        "label",
        "split",
        "source",
        "subject",
        "attack_type",
        "device",
        "lighting",
        "environment",
        "original_path",
        "frame_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[RawSample] = []
    for source_arg in args.source:
        source_name, source_path = parse_source(source_arg)
        samples.extend(load_samples(source_name, source_path))

    if not samples:
        raise ValueError("no live/spoof samples found; check labels or source manifests")

    rows: list[dict[str, str]] = []
    for sample in tqdm(samples, desc="converting"):
        split = split_for_sample(sample, args.val_ratio, args.test_ratio)
        suffix = sample.path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            rows.extend(
                convert_image_sample(
                    sample,
                    split,
                    args.output_dir,
                    args.image_size,
                    args.crop_face,
                )
            )
        elif suffix in VIDEO_EXTENSIONS:
            rows.extend(
                convert_video_sample(
                    sample,
                    split,
                    args.output_dir,
                    args.image_size,
                    args.crop_face,
                    args.video_stride,
                    args.max_frames_per_video,
                )
            )

    write_manifest(args.output_dir / "manifest.csv", rows)
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    print(f"output_dir: {args.output_dir}")
    print(f"raw_samples: {len(samples)}")
    print(f"converted_images: {len(rows)}")
    for split, count in sorted(split_counts.items()):
        print(f"{split}_images: {count}")
    print(f"manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
