from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import image_to_tensor, load_rgb_image, parse_label
from evaluate import load_model, pick_device
from metrics import (
    choose_security_threshold,
    full_evaluation,
    metrics_at_threshold,
    subgroup_evaluation,
    write_csv,
    write_json,
    write_markdown_summary,
)


DEFAULT_GROUP_COLUMNS = [
    "spoof_type",
    "attack_type",
    "illumination",
    "lighting",
    "environment",
    "source",
    "domain",
    "device",
    "subject",
    "subject_id",
    "split",
]


@dataclass(frozen=True)
class EvalItem:
    path: Path
    label: int
    metadata: dict[str, str]


class ProductionEvalDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        manifest: Path | None,
        image_size: int,
        normalization: str,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.normalization = normalization
        if manifest is not None and manifest.exists():
            self.items = self._load_manifest(manifest)
        else:
            self.items = self._load_folder(data_dir / split)
        if not self.items:
            raise ValueError(f"no samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        item = self.items[index]
        image = load_rgb_image(item.path, self.image_size)
        tensor = image_to_tensor(image, augment=False, normalization=self.normalization)
        label = torch.tensor([item.label], dtype=torch.float32)
        return tensor, label, dict(item.metadata)

    def _load_manifest(self, manifest: Path) -> list[EvalItem]:
        items: list[EvalItem] = []
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"path", "label"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")
            for row in reader:
                if row.get("split", "") != self.split:
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = self.data_dir / path
                metadata = {key: value for key, value in row.items() if key not in {"label"}}
                metadata["path"] = str(path)
                metadata["split"] = row.get("split", self.split) or self.split
                if "source" not in metadata and "domain" in metadata:
                    metadata["source"] = metadata["domain"]
                if "spoof_type" not in metadata and "attack_type" in metadata:
                    metadata["spoof_type"] = metadata["attack_type"]
                if "illumination" not in metadata and "lighting" in metadata:
                    metadata["illumination"] = metadata["lighting"]
                items.append(
                    EvalItem(
                        path=path,
                        label=parse_label(row["label"]),
                        metadata=metadata,
                    )
                )
        return items

    def _load_folder(self, root: Path) -> list[EvalItem]:
        if not root.exists():
            raise FileNotFoundError(f"split folder not found and no manifest available: {root}")
        items: list[EvalItem] = []
        for label_name in ("live", "real", "genuine", "bona_fide", "spoof", "fake", "attack"):
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            label = 1 if label_name in {"live", "real", "genuine", "bona_fide"} else 0
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file():
                    continue
                metadata = {
                    "path": str(path),
                    "split": self.split,
                    "source": "folder",
                    "label_name": "live" if label == 1 else "spoof",
                }
                items.append(EvalItem(path=path, label=label, metadata=metadata))
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

    model.eval()
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

    return (
        np.concatenate(scores),
        np.concatenate(labels).astype(np.int64),
        metadata,
        (elapsed_seconds / max(image_count, 1)) * 1000.0,
    )


def load_threshold_policy(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision_policy = payload.get("decision_policy") if isinstance(payload, dict) else None
    if isinstance(decision_policy, dict):
        return {
            "reject_threshold": float(decision_policy.get("reject_threshold", decision_policy.get("threshold", 0.5))),
            "accept_threshold": float(decision_policy.get("accept_threshold", payload.get("threshold", 0.5))),
            "threshold_policy": str(decision_policy.get("threshold_policy", payload.get("policy", path.stem))),
            "source": str(path),
        }
    if isinstance(payload, dict) and isinstance(payload.get("production"), dict):
        production = payload["production"]
        threshold = float(production["threshold"])
        return {
            "reject_threshold": threshold,
            "accept_threshold": threshold,
            "threshold_policy": str(production.get("policy", "production_threshold")),
            "source": str(path),
        }
    if isinstance(payload, dict) and "threshold" in payload:
        threshold = float(payload["threshold"])
        return {
            "reject_threshold": threshold,
            "accept_threshold": threshold,
            "threshold_policy": str(payload.get("policy", "threshold_json")),
            "source": str(path),
        }
    raise ValueError(f"could not read threshold policy from {path}")


def flatten_metrics(evaluation: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in evaluation.items():
        if isinstance(value, (str, int, float)) or value is None:
            rows.append({"section": "overall", "metric": key, "value": value})
    selected = evaluation.get("selected_threshold", {})
    if isinstance(selected, dict):
        for key, value in selected.items():
            if isinstance(value, (str, int, float)) or value is None:
                rows.append({"section": "selected_threshold", "metric": key, "value": value})
    band = evaluation.get("decision_band", {})
    if isinstance(band, dict):
        for key, value in band.items():
            rows.append({"section": "decision_band", "metric": key, "value": value})
    return rows


def sample_rows(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score, label, meta in zip(scores, labels, metadata, strict=True):
        prediction = int(float(score) >= threshold)
        row: dict[str, object] = {
            "path": meta.get("path", ""),
            "label": int(label),
            "label_name": "live" if int(label) == 1 else "spoof",
            "live_probability": float(score),
            "prediction": prediction,
            "prediction_name": "live" if prediction == 1 else "spoof",
            "is_false_accept": int(label) == 0 and prediction == 1,
            "is_false_reject": int(label) == 1 and prediction == 0,
        }
        for key in DEFAULT_GROUP_COLUMNS:
            if key in meta:
                row[key] = meta.get(key, "")
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-style TinyLiveness evaluation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--reject-threshold", type=float, default=None)
    parser.add_argument("--accept-threshold", type=float, default=None)
    parser.add_argument("--threshold-policy", default=None)
    parser.add_argument(
        "--group-by",
        default=",".join(DEFAULT_GROUP_COLUMNS),
        help="Comma-separated metadata columns for subgroup metrics.",
    )
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/production_eval"))
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
    scores, labels, metadata, latency_ms = score_dataset(model, loader, device)

    threshold_policy = load_threshold_policy(args.thresholds_json)
    if args.threshold is not None:
        accept_threshold = float(args.threshold)
        reject_threshold = float(args.reject_threshold if args.reject_threshold is not None else args.threshold)
        policy_name = args.threshold_policy or "cli_threshold"
    elif args.accept_threshold is not None:
        accept_threshold = float(args.accept_threshold)
        reject_threshold = float(args.reject_threshold if args.reject_threshold is not None else args.accept_threshold)
        policy_name = args.threshold_policy or "cli_decision_band"
    elif threshold_policy is not None:
        accept_threshold = float(threshold_policy["accept_threshold"])
        reject_threshold = float(threshold_policy["reject_threshold"])
        policy_name = str(threshold_policy["threshold_policy"])
    elif "threshold" in config:
        accept_threshold = float(config["threshold"])
        reject_threshold = accept_threshold
        policy_name = "checkpoint_threshold"
    else:
        selected = choose_security_threshold(scores, labels)
        accept_threshold = float(selected["threshold"])
        reject_threshold = accept_threshold
        policy_name = str(selected["policy"])

    if reject_threshold > accept_threshold:
        raise ValueError("reject threshold must be <= accept threshold")

    evaluation = full_evaluation(
        scores,
        labels,
        threshold=accept_threshold,
        decision_reject_threshold=reject_threshold,
        decision_accept_threshold=accept_threshold,
    )
    group_columns = [item.strip() for item in args.group_by.split(",") if item.strip()]
    subgroup_rows = subgroup_evaluation(
        scores,
        labels,
        metadata,
        threshold=accept_threshold,
        group_columns=group_columns,
        min_group_size=args.min_group_size,
    )
    rows = sample_rows(scores, labels, metadata, threshold=accept_threshold)
    false_accepts = [row for row in rows if row["is_false_accept"]]

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "normalization": normalization,
        "image_size": image_size,
        "latency_ms_per_image": latency_ms,
        "threshold_policy": {
            "reject_threshold": reject_threshold,
            "accept_threshold": accept_threshold,
            "threshold_policy": policy_name,
            "source": threshold_policy.get("source") if threshold_policy else None,
        },
        "checkpoint_config": config,
        "evaluation": evaluation,
        "subgroups": subgroup_rows,
        "false_accept_count": len(false_accepts),
    }

    json_path = args.output_prefix.with_suffix(".json")
    metrics_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    subgroup_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_subgroups.csv")
    samples_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_scores.csv")
    false_accepts_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_false_accepts.csv")
    md_path = args.output_prefix.with_suffix(".md")

    write_json(json_path, report)
    write_csv(metrics_csv_path, flatten_metrics(evaluation))
    write_csv(subgroup_csv_path, subgroup_rows)
    write_csv(samples_csv_path, rows)
    write_csv(false_accepts_csv_path, false_accepts)
    write_markdown_summary(
        md_path,
        title=f"TinyLiveness Production Evaluation ({args.split})",
        evaluation=evaluation,
        subgroup_rows=subgroup_rows,
        extra_lines=[
            "## Runtime",
            "",
            f"- Average model inference latency: {latency_ms:.6f} ms/image",
            f"- Threshold policy: {policy_name}",
        ],
    )

    selected = evaluation["selected_threshold"]
    assert isinstance(selected, dict)
    print(f"split: {args.split}")
    print(f"samples: {evaluation['samples']}")
    print(f"roc_auc: {float(evaluation['roc_auc']):.6f}")
    print(f"apcer: {float(selected['apcer']):.6f}")
    print(f"bpcer: {float(selected['bpcer']):.6f}")
    print(f"acer: {float(selected['acer']):.6f}")
    print(f"threshold: {accept_threshold:.6f} ({policy_name})")
    print(f"latency_ms_per_image: {latency_ms:.6f}")
    print(f"json_report: {json_path}")
    print(f"markdown_report: {md_path}")
    print(f"false_accepts_csv: {false_accepts_csv_path}")


if __name__ == "__main__":
    main()
