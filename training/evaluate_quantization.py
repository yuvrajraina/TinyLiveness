from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import load_model
from evaluate_production import ProductionEvalDataset, collate_batch, load_threshold_policy
from metrics import binary_roc_auc, metrics_at_threshold, write_json


APPROVAL = {
    "max_auc_drop": 0.005,
    "max_acer_increase": 0.01,
    "max_apcer_increase": 0.01,
    "min_score_correlation": 0.99,
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def score_onnx(
    model_path: Path,
    loader: DataLoader,
    *,
    output_is_probability: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_count = 0
    elapsed_seconds = 0.0
    for images, batch_labels, _ in tqdm(loader, desc=f"scoring {model_path.name}", leave=False):
        batch = images.numpy().astype(np.float32, copy=False)
        start = time.perf_counter()
        output = session.run([output_name], {input_name: batch})[0]
        elapsed_seconds += time.perf_counter() - start
        batch_scores = np.asarray(output, dtype=np.float32).reshape(-1)
        if not output_is_probability:
            batch_scores = sigmoid(batch_scores)
        scores.append(np.clip(batch_scores, 0.0, 1.0))
        labels.append(batch_labels.numpy().reshape(-1))
        image_count += batch.shape[0]
    return (
        np.concatenate(scores),
        np.concatenate(labels).astype(np.int64),
        (elapsed_seconds / max(image_count, 1)) * 1000.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare FP32 and INT8 TinyLiveness ONNX safety.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--fp32-onnx", type=Path, required=True)
    parser.add_argument("--int8-onnx", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-is-logits", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("reports/int8_quantization_eval.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, config = load_model(args.checkpoint, torch.device("cpu"))
    image_size = int(args.image_size or config.get("input_size", 112))
    normalization = str(config.get("normalization", "tinyliveness"))
    manifest = args.manifest or args.data_dir / "manifest.csv"
    if not manifest.exists():
        manifest = None
    dataset = ProductionEvalDataset(
        data_dir=args.data_dir,
        split=args.split,
        manifest=manifest,
        image_size=image_size,
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    fp32_scores, labels, fp32_latency = score_onnx(
        args.fp32_onnx,
        loader,
        output_is_probability=not args.output_is_logits,
    )
    int8_scores, _, int8_latency = score_onnx(
        args.int8_onnx,
        loader,
        output_is_probability=not args.output_is_logits,
    )
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_policy = "cli_threshold"
    else:
        policy = load_threshold_policy(args.thresholds_json)
        if policy is not None:
            threshold = float(policy["accept_threshold"])
            threshold_policy = str(policy["threshold_policy"])
        else:
            threshold = float(config.get("threshold", 0.5))
            threshold_policy = "checkpoint_threshold"

    fp32_metrics = metrics_at_threshold(fp32_scores, labels, threshold)
    int8_metrics = metrics_at_threshold(int8_scores, labels, threshold)
    fp32_auc = binary_roc_auc(fp32_scores, labels)
    int8_auc = binary_roc_auc(int8_scores, labels)
    score_delta = np.abs(fp32_scores - int8_scores)
    correlation = float(np.corrcoef(fp32_scores, int8_scores)[0, 1])
    auc_drop = float(fp32_auc - int8_auc)
    acer_increase = float(int8_metrics.acer - fp32_metrics.acer)
    apcer_increase = float(int8_metrics.apcer - fp32_metrics.apcer)
    approved = (
        auc_drop <= APPROVAL["max_auc_drop"]
        and acer_increase <= APPROVAL["max_acer_increase"]
        and apcer_increase <= APPROVAL["max_apcer_increase"]
        and correlation >= APPROVAL["min_score_correlation"]
    )
    result = {
        "split": args.split,
        "threshold": threshold,
        "threshold_policy": threshold_policy,
        "mean_absolute_score_difference": float(score_delta.mean()),
        "max_score_difference": float(score_delta.max()),
        "score_correlation": correlation,
        "fp32": {
            "onnx": str(args.fp32_onnx),
            "latency_ms_per_image": fp32_latency,
            "roc_auc": fp32_auc,
            "apcer": fp32_metrics.apcer,
            "bpcer": fp32_metrics.bpcer,
            "acer": fp32_metrics.acer,
        },
        "int8": {
            "onnx": str(args.int8_onnx),
            "latency_ms_per_image": int8_latency,
            "roc_auc": int8_auc,
            "apcer": int8_metrics.apcer,
            "bpcer": int8_metrics.bpcer,
            "acer": int8_metrics.acer,
        },
        "deltas": {
            "roc_auc_drop": auc_drop,
            "apcer_change": apcer_increase,
            "bpcer_change": float(int8_metrics.bpcer - fp32_metrics.bpcer),
            "acer_change": acer_increase,
        },
        "approval_criteria": APPROVAL,
        "status": "INT8_APPROVED" if approved else "INT8_NOT_APPROVED",
    }
    write_json(args.output_json, result)
    print(f"status: {result['status']}")
    print(f"mean_abs_score_diff: {result['mean_absolute_score_difference']:.6f}")
    print(f"max_score_diff: {result['max_score_difference']:.6f}")
    print(f"score_correlation: {correlation:.6f}")
    print(f"auc_drop: {auc_drop:.6f}")
    print(f"apcer_change: {apcer_increase:.6f}")
    print(f"acer_change: {acer_increase:.6f}")
    print(f"json: {args.output_json}")


if __name__ == "__main__":
    main()
