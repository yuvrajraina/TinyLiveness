from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LivenessImageDataset
from metrics import binary_roc_auc, metrics_at_threshold, select_threshold, true_live_rate_at_apcer
from tinyliveness import build_liveness_model


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
        dropout=float(config.get("dropout", 0.0)),
        pretrained=False,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_count = 0
    elapsed_seconds = 0.0

    for images, batch_labels in tqdm(loader, desc="scoring", leave=False):
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        probabilities = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds += time.perf_counter() - start

        image_count += images.shape[0]
        scores.append(probabilities)
        labels.append(batch_labels.numpy().reshape(-1))

    avg_ms = (elapsed_seconds / max(image_count, 1)) * 1000.0
    return np.concatenate(scores), np.concatenate(labels).astype(np.int64), avg_ms


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TinyLiveness.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinyliveness.pt"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-apcer", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    model, config = load_model(args.checkpoint, device)
    dataset = LivenessImageDataset(
        args.data_dir / args.split,
        image_size=args.image_size,
        augment=False,
        normalization=str(config.get("normalization", "tinyliveness")),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    scores, labels, inference_ms = score_dataset(model, loader, device)
    auc = binary_roc_auc(scores, labels)
    tlr_targets = true_live_rate_at_apcer(scores, labels)

    threshold = args.threshold
    if threshold is None:
        threshold = config.get("threshold")
    if threshold is None:
        selected = select_threshold(scores, labels, max_apcer=args.max_apcer)
    else:
        selected = metrics_at_threshold(scores, labels, float(threshold))

    counts = dataset.class_counts()
    print(f"split: {args.split}")
    print(f"images: {len(dataset)}")
    print(f"live_images: {counts['live']}")
    print(f"spoof_images: {counts['spoof']}")
    print(f"roc_auc: {auc:.4f}")
    print(f"threshold: {selected.threshold:.4f}")
    print(f"accuracy: {selected.accuracy:.4f}")
    print(f"balanced_accuracy: {selected.balanced_accuracy:.4f}")
    print(f"apcer: {selected.apcer:.4f}")
    print(f"bpcer: {selected.bpcer:.4f}")
    print(f"acer: {selected.acer:.4f}")
    print(f"true_live_rate: {selected.true_live_rate:.4f}")
    print(f"true_spoof_rate: {selected.true_spoof_rate:.4f}")
    for apcer_target, true_live_rate in tlr_targets.items():
        print(f"true_live_rate_at_apcer_{apcer_target:g}: {true_live_rate:.4f}")
    print(f"false_live: {selected.false_live}")
    print(f"false_spoof: {selected.false_spoof}")
    print(f"true_live: {selected.true_live}")
    print(f"true_spoof: {selected.true_spoof}")
    print(f"checkpoint_size_mb: {file_size_mb(args.checkpoint):.3f}")
    print(f"avg_inference_ms_per_image: {inference_ms:.3f}")


if __name__ == "__main__":
    main()
