from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import load_model, pick_device
from evaluate_production import ProductionEvalDataset, collate_batch, score_dataset
from metrics import choose_security_threshold, full_evaluation, write_json, write_markdown_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Security-first TinyLiveness threshold calibration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/thresholds.json"))
    parser.add_argument("--decision-policy-output", type=Path, default=Path("checkpoints/decision_policy.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/threshold_calibration.md"))
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
    scores, labels, _, latency_ms = score_dataset(model, loader, device)
    selected = choose_security_threshold(scores, labels, preferred_targets=(0.01, 0.03, 0.05))
    accept_threshold = float(selected["threshold"])
    initial_eval = full_evaluation(scores, labels, threshold=accept_threshold)
    thresholds = initial_eval["thresholds"]
    assert isinstance(thresholds, dict)
    best_balanced = thresholds["best_balanced_accuracy"]
    assert isinstance(best_balanced, dict)
    reject_threshold = min(float(best_balanced["threshold"]), accept_threshold)
    evaluation = full_evaluation(
        scores,
        labels,
        threshold=accept_threshold,
        decision_reject_threshold=reject_threshold,
        decision_accept_threshold=accept_threshold,
    )
    decision_policy = {
        "reject_threshold": reject_threshold,
        "accept_threshold": accept_threshold,
        "threshold_policy": str(selected["policy"]),
        "calibration_split": args.split,
        "live_label": 1,
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "calibration_split": args.split,
        "normalization": normalization,
        "image_size": image_size,
        "latency_ms_per_image": latency_ms,
        "production": {
            "threshold": accept_threshold,
            "target_apcer": selected["target_apcer"],
            "policy": selected["policy"],
            "apcer": selected["apcer"],
            "bpcer": selected["bpcer"],
            "acer": selected["acer"],
        },
        "decision_policy": decision_policy,
        "thresholds": thresholds,
        "evaluation": evaluation,
    }
    write_json(args.output, payload)
    write_json(args.decision_policy_output, decision_policy)
    write_markdown_summary(
        args.report_md,
        title=f"TinyLiveness Threshold Calibration ({args.split})",
        evaluation=evaluation,
        extra_lines=[
            "## Selected Policy",
            "",
            f"- Policy: {selected['policy']}",
            f"- Accept threshold: {accept_threshold:.6f}",
            f"- Reject threshold: {reject_threshold:.6f}",
            f"- Calibration latency: {latency_ms:.6f} ms/image",
        ],
    )
    print(f"calibration_split: {args.split}")
    print(f"policy: {selected['policy']}")
    print(f"accept_threshold: {accept_threshold:.6f}")
    print(f"reject_threshold: {reject_threshold:.6f}")
    print(f"apcer: {float(selected['apcer']):.6f}")
    print(f"bpcer: {float(selected['bpcer']):.6f}")
    print(f"acer: {float(selected['acer']):.6f}")
    print(f"thresholds_json: {args.output}")
    print(f"decision_policy_json: {args.decision_policy_output}")


if __name__ == "__main__":
    main()
