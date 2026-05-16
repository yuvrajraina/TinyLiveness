from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SPOOF_TYPES = {
    0: "live",
    1: "photo",
    2: "poster",
    3: "a4",
    4: "face_mask",
    5: "upper_body_mask",
    6: "region_mask",
    7: "pc",
    8: "pad",
    9: "phone",
    10: "3d_mask",
}
ILLUMINATION = {
    0: "live",
    1: "normal",
    2: "strong",
    3: "back",
    4: "dark",
}
ENVIRONMENT = {
    0: "live",
    1: "indoor",
    2: "outdoor",
}


@dataclass(frozen=True)
class CelebASample:
    image_path: str
    attributes: list[int]
    split: str

    @property
    def label(self) -> int:
        # Official benchmark uses index 43 where 0=live and 1=spoof.
        # TinyLiveness uses 1=live and 0=spoof.
        return 1 if int(self.attributes[43]) == 0 else 0

    @property
    def label_name(self) -> str:
        return "live" if self.label == 1 else "spoof"

    @property
    def attack_type(self) -> str:
        return SPOOF_TYPES.get(int(self.attributes[40]), f"unknown_{self.attributes[40]}")

    @property
    def illumination(self) -> str:
        return ILLUMINATION.get(int(self.attributes[41]), f"unknown_{self.attributes[41]}")

    @property
    def environment(self) -> str:
        return ENVIRONMENT.get(int(self.attributes[42]), f"unknown_{self.attributes[42]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert extracted CelebA-Spoof into TinyLiveness layout."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/CelebA-Spoof/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/liveness_celeba_spoof"))
    parser.add_argument(
        "--label-json",
        action="append",
        type=Path,
        default=None,
        help="Optional label JSON file. Repeat for train/val/test. If omitted, all *label*.json files under source-dir are used.",
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-per-split", type=int, default=None)
    parser.add_argument("--max-per-label-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def find_label_jsons(source_dir: Path) -> list[Path]:
    candidates = sorted(source_dir.rglob("*label*.json"))
    if not candidates:
        raise FileNotFoundError(f"no *label*.json files found under {source_dir}")
    return candidates


def split_from_label_path(path: Path) -> str:
    normalized = path.as_posix().lower()
    if "train" in normalized:
        return "train"
    if "val" in normalized or "dev" in normalized:
        return "val"
    if "test" in normalized:
        return "test"
    return "train"


def load_samples(label_jsons: list[Path]) -> list[CelebASample]:
    samples: list[CelebASample] = []
    for label_path in label_jsons:
        split = split_from_label_path(label_path)
        data = json.loads(label_path.read_text(encoding="utf-8"))
        for image_path, attributes in data.items():
            if len(attributes) < 44:
                raise ValueError(f"expected 44 attributes for {image_path}, got {len(attributes)}")
            samples.append(
                CelebASample(
                    image_path=image_path,
                    attributes=[int(value) for value in attributes],
                    split=split,
                )
            )
    return samples


def resolve_image_path(source_dir: Path, image_path: str) -> Path:
    direct = source_dir / image_path
    if direct.exists():
        return direct

    # Some archives include one top-level folder; search by suffix once.
    matches = list(source_dir.rglob(Path(image_path).name))
    for candidate in matches:
        if candidate.as_posix().endswith(Path(image_path).as_posix()):
            return candidate
    if matches:
        return matches[0]
    raise FileNotFoundError(f"image not found for label row: {image_path}")


def read_bbox(image_file: Path, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    bbox_file = image_file.with_name(f"{image_file.stem}_BB.txt")
    if not bbox_file.exists():
        return None
    parts = bbox_file.read_text(encoding="utf-8").strip().split()
    if len(parts) < 4:
        return None
    x, y, width, height = [int(float(value)) for value in parts[:4]]
    x = int(x * (image_width / 224))
    y = int(y * (image_height / 224))
    width = int(width * (image_width / 224))
    height = int(height * (image_height / 224))
    left = max(x, 0)
    top = max(y, 0)
    right = min(x + width, image_width)
    bottom = min(y + height, image_height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def center_square(image: Image.Image) -> Image.Image:
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def prepare_image(source_dir: Path, sample: CelebASample, output_path: Path, image_size: int) -> None:
    image_file = resolve_image_path(source_dir, sample.image_path)
    with Image.open(image_file) as image:
        image = image.convert("RGB")
        bbox = read_bbox(image_file, image.width, image.height)
        if bbox is not None:
            image = image.crop(bbox)
        else:
            image = center_square(image)
        image = image.resize((image_size, image_size), Image.BILINEAR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)


def safe_stem(image_path: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in image_path).strip("_")


def manifest_row(sample: CelebASample, relative_path: Path) -> dict[str, str]:
    parts = Path(sample.image_path).parts
    subject = parts[-2] if len(parts) >= 2 else ""
    return {
        "path": relative_path.as_posix(),
        "label": sample.label_name,
        "split": sample.split,
        "source": "celeba_spoof",
        "subject": subject,
        "attack_type": sample.attack_type,
        "device": sample.attack_type if sample.attack_type in {"pc", "pad", "phone"} else "",
        "lighting": sample.illumination,
        "environment": sample.environment,
        "original_path": sample.image_path,
        "frame_index": "",
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

    label_jsons = args.label_json or find_label_jsons(args.source_dir)
    samples = load_samples(label_jsons)
    per_split_count: dict[str, int] = {}
    per_split_label_count: dict[tuple[str, str], int] = {}
    rows: list[dict[str, str]] = []

    for sample in tqdm(samples, desc="preparing CelebA-Spoof"):
        if args.max_per_split is not None:
            split_count = per_split_count.get(sample.split, 0)
            if split_count >= args.max_per_split:
                continue
        if args.max_per_label_per_split is not None:
            label_key = (sample.split, sample.label_name)
            label_count = per_split_label_count.get(label_key, 0)
            if label_count >= args.max_per_label_per_split:
                continue

        relative_path = (
            Path(sample.split)
            / sample.label_name
            / f"{safe_stem(sample.image_path)}.jpg"
        )
        prepare_image(args.source_dir, sample, args.output_dir / relative_path, args.image_size)
        rows.append(manifest_row(sample, relative_path))
        per_split_count[sample.split] = per_split_count.get(sample.split, 0) + 1
        label_key = (sample.split, sample.label_name)
        per_split_label_count[label_key] = per_split_label_count.get(label_key, 0) + 1

    write_manifest(args.output_dir / "manifest.csv", rows)
    print(f"source_dir: {args.source_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"label_jsons: {len(label_jsons)}")
    print(f"converted_images: {len(rows)}")
    for split, count in sorted(per_split_count.items()):
        print(f"{split}_images: {count}")
    for (split, label), count in sorted(per_split_label_count.items()):
        print(f"{split}_{label}_images: {count}")
    print(f"manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
