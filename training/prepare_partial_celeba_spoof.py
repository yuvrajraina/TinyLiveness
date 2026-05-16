from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    data: bytes | None = None


@dataclass
class PendingImage:
    name: str
    data: bytes
    split: str
    subject: str
    label: str


class MultiPartReader:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.index = 0
        self.handle = None
        self.offset = 0
        self._open_next()

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        if self.index >= len(self.paths):
            self.handle = None
            return
        self.handle = self.paths[self.index].open("rb")
        self.index += 1

    def read(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            if self.handle is None:
                raise EOFError
            data = self.handle.read(remaining)
            if not data:
                self._open_next()
                continue
            chunks.append(data)
            remaining -= len(data)
            self.offset += len(data)
        return b"".join(chunks)

    def skip(self, size: int) -> None:
        remaining = size
        while remaining > 0:
            if self.handle is None:
                raise EOFError
            chunk_size = min(remaining, 16 * 1024 * 1024)
            data = self.handle.read(chunk_size)
            if not data:
                self._open_next()
                continue
            remaining -= len(data)
            self.offset += len(data)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a usable TinyLiveness subset from partial CelebA-Spoof split zip parts."
    )
    parser.add_argument("--parts-dir", type=Path, default=Path("data/raw/CelebA-Spoof"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/liveness_celeba_partial"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--val-subject-ratio", type=float, default=0.15)
    parser.add_argument("--train-per-label", type=int, default=4000)
    parser.add_argument("--val-per-label", type=int, default=1000)
    parser.add_argument("--test-per-label", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def find_parts(parts_dir: Path) -> list[Path]:
    parts = sorted(parts_dir.glob("CelebA_Spoof.zip.*"))
    selected = [
        part
        for part in parts
        if part.suffix[1:].isdigit() or part.name.endswith(".part")
    ]
    if not selected:
        raise FileNotFoundError(f"no split zip parts found in {parts_dir}")
    return selected


def iter_local_zip_entries(parts: list[Path]):
    reader = MultiPartReader(parts)
    try:
        while True:
            try:
                header = reader.read(30)
            except EOFError:
                break
            if header[:4] != b"PK\x03\x04":
                break

            (
                _version,
                flags,
                compression,
                _mtime,
                _mdate,
                _crc,
                compressed_size,
                uncompressed_size,
                name_len,
                extra_len,
            ) = struct.unpack("<HHHHHIIIHH", header[4:30])
            name = reader.read(name_len).decode("utf-8", errors="replace")
            reader.skip(extra_len)
            if flags & 0x08:
                raise ValueError(f"unsupported zip data descriptor entry: {name}")

            should_read = is_image(name) or name.endswith("_BB.txt")
            try:
                data = reader.read(compressed_size) if should_read else None
                if not should_read:
                    reader.skip(compressed_size)
            except EOFError:
                break

            yield ZipEntry(
                name=name,
                compression=compression,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                data=data,
            )
    finally:
        reader.close()


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def parse_image_path(name: str) -> tuple[str, str, str] | None:
    parts = Path(name).parts
    if len(parts) < 6:
        return None
    split = next((part for part in parts if part in {"train", "test", "val"}), None)
    label = next((part for part in parts if part in {"live", "spoof"}), None)
    if split is None or label is None:
        return None
    label_index = parts.index(label)
    subject = parts[label_index - 1] if label_index > 0 else ""
    return split, subject, label


def target_split(source_split: str, subject: str, val_ratio: float) -> str:
    if source_split == "test":
        return "test"
    bucket = stable_bucket(subject)
    return "val" if bucket < val_ratio else "train"


def stable_bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def decompress_entry(entry: ZipEntry) -> bytes:
    if entry.data is None:
        raise ValueError(f"entry has no data: {entry.name}")
    if entry.compression == 0:
        return entry.data
    if entry.compression == 8:
        return zlib.decompress(entry.data, -15)
    raise ValueError(f"unsupported zip compression {entry.compression} for {entry.name}")


def read_bbox(text: str, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    parts = text.strip().split()
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


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def should_keep(counts: dict[tuple[str, str], int], split: str, label: str, args: argparse.Namespace) -> bool:
    limit = {
        "train": args.train_per_label,
        "val": args.val_per_label,
        "test": args.test_per_label,
    }[split]
    return counts.get((split, label), 0) < limit


def all_targets_met(counts: dict[tuple[str, str], int], args: argparse.Namespace) -> bool:
    for split, limit in (
        ("train", args.train_per_label),
        ("val", args.val_per_label),
        ("test", args.test_per_label),
    ):
        for label in ("live", "spoof"):
            if counts.get((split, label), 0) < limit:
                return False
    return True


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


def save_pending(
    pending: PendingImage,
    bbox_text: str | None,
    output_dir: Path,
    image_size: int,
    target: str,
) -> dict[str, str]:
    from io import BytesIO

    with Image.open(BytesIO(pending.data)) as image:
        image = image.convert("RGB")
        bbox = read_bbox(bbox_text or "", image.width, image.height)
        if bbox is not None:
            image = image.crop(bbox)
        else:
            image = center_square(image)
        image = image.resize((image_size, image_size), Image.BILINEAR)
        relative = Path(target) / pending.label / f"{safe_name(pending.name)}.jpg"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=95)

    return {
        "path": relative.as_posix(),
        "label": pending.label,
        "split": target,
        "source": "celeba_spoof_partial",
        "subject": pending.subject,
        "attack_type": "live" if pending.label == "live" else "spoof",
        "device": "",
        "lighting": "",
        "environment": "",
        "original_path": pending.name,
        "frame_index": "",
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parts = find_parts(args.parts_dir)
    counts: dict[tuple[str, str], int] = {}
    rows: list[dict[str, str]] = []
    pending: PendingImage | None = None

    for entry in tqdm(iter_local_zip_entries(parts), desc="streaming CelebA-Spoof"):
        if is_image(entry.name):
            parsed = parse_image_path(entry.name)
            if parsed is None:
                pending = None
                continue
            source_split, subject, label = parsed
            target = target_split(source_split, subject, args.val_subject_ratio)
            if not should_keep(counts, target, label, args):
                pending = None
                continue
            pending = PendingImage(
                name=entry.name,
                data=decompress_entry(entry),
                split=target,
                subject=subject,
                label=label,
            )
            continue

        if entry.name.endswith("_BB.txt") and pending is not None:
            bbox_stem = entry.name.removesuffix("_BB.txt")
            image_stem = pending.name.rsplit(".", 1)[0]
            if bbox_stem == image_stem:
                row = save_pending(
                    pending,
                    decompress_entry(entry).decode("utf-8", errors="replace"),
                    args.output_dir,
                    args.image_size,
                    pending.split,
                )
                rows.append(row)
                key = (pending.split, pending.label)
                counts[key] = counts.get(key, 0) + 1
                pending = None
                if all_targets_met(counts, args):
                    break

    write_manifest(args.output_dir / "manifest.csv", rows)
    print(f"output_dir: {args.output_dir}")
    print(f"converted_images: {len(rows)}")
    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            print(f"{split}_{label}: {counts.get((split, label), 0)}")
    print(f"manifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
