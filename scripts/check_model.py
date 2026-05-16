from __future__ import annotations

import argparse
import time

import torch

from tinyliveness import TinyLiveness, count_parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check TinyLiveness model size and CPU latency.")
    parser.add_argument("--width-mult", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    model = TinyLiveness(width_mult=args.width_mult, dropout=0.0).eval()
    sample = torch.randn(args.batch_size, 3, 112, 112)

    for _ in range(args.warmup):
        model(sample)

    start = time.perf_counter()
    for _ in range(args.runs):
        model(sample)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / max(args.runs * args.batch_size, 1) * 1000.0
    params = count_parameters(model)

    output = torch.sigmoid(model(sample))
    print(f"params: {params:,}")
    print(f"estimated_fp32_size_mb: {params * 4 / (1024 * 1024):.3f}")
    print(f"avg_cpu_ms_per_image: {avg_ms:.3f}")
    print(f"output_shape: {tuple(output.shape)}")
    print(f"output_range: {float(output.min()):.4f}..{float(output.max()):.4f}")


if __name__ == "__main__":
    main()
