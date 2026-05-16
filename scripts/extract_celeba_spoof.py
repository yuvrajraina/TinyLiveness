from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine and extract the official split CelebA-Spoof zip archive."
    )
    parser.add_argument("--parts-dir", type=Path, default=Path("data/raw/CelebA-Spoof"))
    parser.add_argument("--combined-zip", type=Path, default=Path("data/raw/CelebA-Spoof/CelebA_Spoof.zip"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw/CelebA-Spoof/extracted"))
    parser.add_argument("--skip-combine", action="store_true")
    return parser


def find_parts(parts_dir: Path) -> list[Path]:
    parts = sorted(
        path
        for path in parts_dir.glob("CelebA_Spoof.zip.*")
        if path.suffix[1:].isdigit() and not path.name.endswith(".part")
    )
    if not parts:
        raise FileNotFoundError(f"no CelebA_Spoof.zip.NNN parts found in {parts_dir}")
    return parts


def validate_parts(parts: list[Path]) -> None:
    expected = 1
    for part in parts:
        suffix = int(part.suffix[1:])
        if suffix != expected:
            raise ValueError(f"missing split archive part {expected:03d} before {part.name}")
        expected += 1


def combine_parts(parts: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out_handle:
        for part in parts:
            print(f"adding {part.name}")
            with part.open("rb") as in_handle:
                while True:
                    chunk = in_handle.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    out_handle.write(chunk)


def extract_zip(path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)


def main() -> None:
    args = build_parser().parse_args()
    if not args.skip_combine:
        parts = find_parts(args.parts_dir)
        validate_parts(parts)
        print(f"parts: {len(parts)}")
        combine_parts(parts, args.combined_zip)
    print(f"combined_zip: {args.combined_zip}")
    print(f"combined_size_gb: {args.combined_zip.stat().st_size / (1024 ** 3):.2f}")
    extract_zip(args.combined_zip, args.extract_dir)
    print(f"extracted_to: {args.extract_dir}")


if __name__ == "__main__":
    main()
