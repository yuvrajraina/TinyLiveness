from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_model, pick_device
from evaluate_production import ProductionEvalDataset, collate_batch, load_threshold_policy, score_dataset
from metrics import brier_score, ece_score, metrics_at_threshold, write_json


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def nll(scores: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    logits = logit(scores)
    candidates = np.concatenate(
        [
            np.linspace(0.25, 2.0, 120),
            np.linspace(2.05, 8.0, 120),
        ]
    )
    losses = [nll(sigmoid(logits / temperature), labels) for temperature in candidates]
    return float(candidates[int(np.argmin(losses))])


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> dict[str, object] | None:
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(scores, labels)
    return {
        "x_thresholds": [float(value) for value in model.X_thresholds_],
        "y_thresholds": [float(value) for value in model.y_thresholds_],
    }


def apply_isotonic(scores: np.ndarray, model: dict[str, object]) -> np.ndarray:
    x = np.asarray(model["x_thresholds"], dtype=np.float64)
    y = np.asarray(model["y_thresholds"], dtype=np.float64)
    return np.interp(scores, x, y).astype(np.float64)


def metric_block(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = metrics_at_threshold(scores, labels, threshold)
    return {
        "nll": nll(scores, labels),
        "ece": ece_score(scores, labels),
        "brier_score": brier_score(scores, labels),
        "accuracy": metrics.accuracy,
        "apcer": metrics.apcer,
        "bpcer": metrics.bpcer,
        "acer": metrics.acer,
        "threshold": threshold,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate TinyLiveness live probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/score_calibration.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    model, config = load_model(args.checkpoint, device)
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
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    scores, labels, _, _ = score_dataset(model, loader, device)
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        policy = load_threshold_policy(args.thresholds_json)
        threshold = float(policy["accept_threshold"]) if policy else float(config.get("threshold", 0.5))

    temperature = fit_temperature(scores, labels)
    temp_scores = sigmoid(logit(scores) / temperature)
    isotonic = fit_isotonic(scores, labels)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "threshold": threshold,
        "temperature_scaling": {
            "temperature": temperature,
            "metrics": metric_block(temp_scores, labels, threshold),
        },
        "before": metric_block(scores, labels, threshold),
        "isotonic": None,
        "warnings": [],
    }
    if payload["temperature_scaling"]["metrics"]["apcer"] > payload["before"]["apcer"]:
        payload["warnings"].append("temperature scaling worsened APCER at the selected threshold")
    if isotonic is not None:
        isotonic_scores = apply_isotonic(scores, isotonic)
        isotonic_metrics = metric_block(isotonic_scores, labels, threshold)
        payload["isotonic"] = {
            "model": isotonic,
            "metrics": isotonic_metrics,
        }
        if isotonic_metrics["apcer"] > payload["before"]["apcer"]:
            payload["warnings"].append("isotonic calibration worsened APCER at the selected threshold")
    write_json(args.output, payload)
    print(f"temperature: {temperature:.6f}")
    print(f"before_ece: {payload['before']['ece']:.6f}")
    print(f"temperature_ece: {payload['temperature_scaling']['metrics']['ece']:.6f}")
    print(f"warnings: {len(payload['warnings'])}")
    print(f"json: {args.output}")


if __name__ == "__main__":
    main()
