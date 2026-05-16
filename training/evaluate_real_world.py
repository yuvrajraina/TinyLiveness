from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import image_to_tensor, load_rgb_image, parse_label
from evaluate import load_model
from metrics import (
    binary_roc_auc,
    metrics_at_threshold,
    select_threshold,
    true_live_rate_at_apcer,
)


@dataclass(frozen=True)
class ManifestItem:
    path: Path
    label: int
    metadata: dict[str, str]


class ManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        data_dir: Path,
        split: str,
        image_size: int = 112,
        normalization: str = "tinyliveness",
    ) -> None:
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.normalization = normalization
        self.items = load_manifest_items(manifest_path, data_dir, split)
        if not self.items:
            raise ValueError(f"no rows found for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        item = self.items[index]
        image = load_rgb_image(item.path, self.image_size)
        tensor = image_to_tensor(
            image,
            augment=False,
            normalization=self.normalization,
        )
        label = torch.tensor([item.label], dtype=torch.float32)
        return tensor, label, item.metadata


def load_manifest_items(manifest_path: Path, data_dir: Path, split: str) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        for row in reader:
            row_split = row.get("split", "")
            if row_split != split:
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = data_dir / path
            metadata = {key: value for key, value in row.items() if key not in {"path", "label"}}
            items.append(
                ManifestItem(
                    path=path,
                    label=parse_label(row["label"]),
                    metadata=metadata,
                )
            )
    return items


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, str]]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return images, labels, metadata


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], float]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    image_count = 0
    elapsed_seconds = 0.0

    for images, batch_labels, batch_metadata in tqdm(loader, desc="scoring", leave=False):
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
        metadata.extend(batch_metadata)

    avg_ms = (elapsed_seconds / max(image_count, 1)) * 1000.0
    return np.concatenate(scores), np.concatenate(labels).astype(np.int64), metadata, avg_ms


def summarize(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    group_name: str = "overall",
    group_value: str = "all",
) -> dict[str, float | int | str]:
    metrics = metrics_at_threshold(scores, labels, threshold)
    live_count = int((labels == 1).sum())
    spoof_count = int((labels == 0).sum())
    try:
        auc = binary_roc_auc(scores, labels)
    except ValueError:
        auc = float("nan")

    row: dict[str, float | int | str] = {
        "group": group_name,
        "value": group_value,
        "samples": int(len(scores)),
        "live_samples": live_count,
        "spoof_samples": spoof_count,
        "roc_auc": auc,
        **asdict(metrics),
    }
    if live_count == 0:
        row["bpcer"] = float("nan")
        row["true_live_rate"] = float("nan")
    if spoof_count == 0:
        row["apcer"] = float("nan")
        row["true_spoof_rate"] = float("nan")
    if live_count == 0 or spoof_count == 0:
        row["acer"] = float("nan")
        row["balanced_accuracy"] = float("nan")
    return row


def grouped_summaries(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    threshold: float,
    group_columns: list[str],
    min_group_size: int,
) -> list[dict[str, float | int | str]]:
    rows = [summarize(scores, labels, threshold)]
    for column in group_columns:
        values = sorted({item.get(column, "") or "unknown" for item in metadata})
        for value in values:
            indices = [index for index, item in enumerate(metadata) if (item.get(column, "") or "unknown") == value]
            if len(indices) < min_group_size:
                continue
            index_array = np.asarray(indices, dtype=np.int64)
            rows.append(
                summarize(
                    scores[index_array],
                    labels[index_array],
                    threshold,
                    group_name=column,
                    group_value=value,
                )
            )
    return rows


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "value",
        "samples",
        "live_samples",
        "spoof_samples",
        "roc_auc",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "apcer",
        "bpcer",
        "acer",
        "true_live_rate",
        "true_spoof_rate",
        "false_live",
        "false_spoof",
        "true_live",
        "true_spoof",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TinyLiveness with real-world grouped reporting."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-apcer", type=float, default=0.05)
    parser.add_argument(
        "--group-by",
        default="source,attack_type,device,lighting,environment",
        help="Comma-separated manifest columns to report separately.",
    )
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/real_world_eval.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/real_world_eval.json"))
    return parser


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(dataset: ManifestDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest or args.data_dir / "manifest.csv"
    device = pick_device(args.device)
    model, config = load_model(args.checkpoint, device)

    calibration_dataset = ManifestDataset(
        manifest_path,
        data_dir=args.data_dir,
        split=args.calibration_split,
        image_size=args.image_size,
        normalization=str(config.get("normalization", "tinyliveness")),
    )
    eval_dataset = ManifestDataset(
        manifest_path,
        data_dir=args.data_dir,
        split=args.eval_split,
        image_size=args.image_size,
        normalization=str(config.get("normalization", "tinyliveness")),
    )

    calibration_scores, calibration_labels, _, _ = score_dataset(
        model,
        make_loader(calibration_dataset, args, device),
        device,
    )
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "cli"
    else:
        selected = select_threshold(
            calibration_scores,
            calibration_labels,
            max_apcer=args.max_apcer,
        )
        threshold = selected.threshold
        threshold_source = f"{args.calibration_split}_calibration"

    eval_scores, eval_labels, eval_metadata, avg_ms = score_dataset(
        model,
        make_loader(eval_dataset, args, device),
        device,
    )
    group_columns = [column.strip() for column in args.group_by.split(",") if column.strip()]
    rows = grouped_summaries(
        eval_scores,
        eval_labels,
        eval_metadata,
        threshold,
        group_columns,
        args.min_group_size,
    )

    write_report(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_threshold": config.get("threshold"),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "calibration_split": args.calibration_split,
                "eval_split": args.eval_split,
                "avg_inference_ms_per_image": avg_ms,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overall = rows[0]
    try:
        tlr_targets = true_live_rate_at_apcer(eval_scores, eval_labels)
    except ValueError:
        tlr_targets = {}
    print(f"threshold: {threshold:.4f} ({threshold_source})")
    print(f"eval_split: {args.eval_split}")
    print(f"samples: {overall['samples']}")
    print(f"roc_auc: {overall['roc_auc']:.4f}")
    print(f"accuracy: {overall['accuracy']:.4f}")
    print(f"apcer: {overall['apcer']:.4f}")
    print(f"bpcer: {overall['bpcer']:.4f}")
    print(f"acer: {overall['acer']:.4f}")
    for apcer_target, true_live_rate in tlr_targets.items():
        print(f"true_live_rate_at_apcer_{apcer_target:g}: {true_live_rate:.4f}")
    print(f"avg_inference_ms_per_image: {avg_ms:.3f}")
    print(f"csv_report: {args.output_csv}")
    print(f"json_report: {args.output_json}")


if __name__ == "__main__":
    main()
