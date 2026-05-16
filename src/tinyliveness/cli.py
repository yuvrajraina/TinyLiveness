from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_NORMALIZATION,
    DEFAULT_POLICY_FILENAME,
    create_default_onnx_detector,
    get_default_model_path,
    get_default_policy_path,
)
from .inference import load_threshold_policy, resolve_decision_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info() -> dict[str, Any]:
    model_path = get_default_model_path()
    policy_path = get_default_policy_path()
    reject_threshold, accept_threshold, threshold_policy = resolve_decision_policy(
        threshold=0.5,
        thresholds_path=policy_path,
    )
    return {
        "package": "tinyliveness",
        "model": DEFAULT_MODEL_FILENAME,
        "policy": DEFAULT_POLICY_FILENAME,
        "normalization": DEFAULT_NORMALIZATION,
        "image_size": DEFAULT_IMAGE_SIZE,
        "threshold_policy": threshold_policy,
        "reject_threshold": reject_threshold,
        "accept_threshold": accept_threshold,
        "model_size_mb": round(model_path.stat().st_size / (1024.0 * 1024.0), 3),
        "model_sha256": _sha256(model_path),
        "policy_payload": load_threshold_policy(policy_path),
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_info(_args: argparse.Namespace) -> int:
    _print_json(_artifact_info())
    return 0


def _run_smoke(_args: argparse.Namespace) -> int:
    detector = create_default_onnx_detector()
    image = np.full((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), 127, dtype=np.uint8)
    result = detector.predict_image(image)
    payload = _artifact_info()
    payload["smoke_result"] = {
        "live_probability": result.live_probability,
        "spoof_probability": result.spoof_probability,
        "decision": result.decision,
        "is_live": result.is_live,
        "is_spoof": result.is_spoof,
    }
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyliveness",
        description="Inspect and smoke-test bundled TinyLiveness release artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("info", help="Print bundled model and policy metadata.")
    info_parser.set_defaults(func=_run_info)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Load the bundled ONNX model and run one synthetic inference.",
    )
    smoke_parser.set_defaults(func=_run_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args = parser.parse_args(["info"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
