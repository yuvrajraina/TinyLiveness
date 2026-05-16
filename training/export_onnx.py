from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tinyliveness import TinyLivenessProbability, build_liveness_model

MODEL_SIZE_TARGET_MB = 20.0


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch_load(checkpoint_path, device)
    config = checkpoint.get("config", {})
    model = build_liveness_model(
        str(config.get("architecture", "tiny")),
        width_mult=float(config.get("width_mult", 0.5)),
        dropout=0.0,
        pretrained=False,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def verify_onnx(path: Path, input_size: int) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample = np.random.randn(2, 3, input_size, input_size).astype(np.float32)
    output = session.run(None, {"input": sample})[0]
    if output.shape != (2, 1):
        raise RuntimeError(f"unexpected ONNX output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("ONNX output contains NaN or infinite values")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TinyLiveness to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinyliveness.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tinyliveness.onnx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--raw-logits",
        action="store_true",
        help="export raw logits instead of sigmoid live probability",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    model, config = load_model(args.checkpoint, device)
    export_model = model if args.raw_logits else TinyLivenessProbability(model)
    export_model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_size = int(config.get("input_size", 112))
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    output_name = "live_logit" if args.raw_logits else "live_probability"
    torch.onnx.export(
        export_model,
        dummy,
        args.output,
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={
            "input": {0: "batch"},
            output_name: {0: "batch"},
        },
    )
    verify_onnx(args.output, input_size)
    size_mb = file_size_mb(args.output)
    print(f"exported ONNX model: {args.output}")
    print(f"model_size_mb: {size_mb:.3f}")
    print(f"size_target_under_{MODEL_SIZE_TARGET_MB:.0f}mb: {size_mb < MODEL_SIZE_TARGET_MB}")
    print(f"input: NCHW float32 RGB face tensor, shape Nx3x{input_size}x{input_size}")
    print(f"output: {output_name}, shape Nx1")


if __name__ == "__main__":
    main()
