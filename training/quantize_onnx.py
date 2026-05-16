from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

MODEL_SIZE_TARGET_MB = 20.0


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-int8{input_path.suffix}")


def verify_onnx_runtime(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if not output_names:
        raise RuntimeError(f"ONNX model has no outputs: {path}")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for TinyLiveness ONNX.")
    parser.add_argument("--input", type=Path, default=Path("checkpoints/tinyliveness.onnx"))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or default_output_path(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(output),
        weight_type=QuantType.QInt8,
    )
    verify_onnx_runtime(output)
    size_mb = file_size_mb(output)
    print(f"quantized ONNX model: {output}")
    print(f"model_size_mb: {size_mb:.3f}")
    print(f"size_target_under_{MODEL_SIZE_TARGET_MB:.0f}mb: {size_mb < MODEL_SIZE_TARGET_MB}")


if __name__ == "__main__":
    main()
