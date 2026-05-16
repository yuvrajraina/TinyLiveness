from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OFFICIAL_GOOGLE_DRIVE = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume the official CelebA-Spoof Google Drive download."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/CelebA-Spoof"))
    parser.add_argument("--accept-noncommercial-terms", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.accept_noncommercial_terms:
        raise SystemExit(
            "CelebA-Spoof is restricted to non-commercial research/education and "
            "must not be redistributed. Re-run with --accept-noncommercial-terms "
            "only if that is acceptable for this project."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        OFFICIAL_GOOGLE_DRIVE,
        "-O",
        str(args.output_dir),
        "--continue",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
