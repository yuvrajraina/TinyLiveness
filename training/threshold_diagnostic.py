from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_scores(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "path": row.get("path", ""),
                    "label": int(row["label"]),
                    "score": float(row["live_probability"]),
                }
            )
    if not rows:
        raise ValueError(f"empty scores file: {path}")
    return rows


def sorted_arrays(rows: list[dict[str, float | int | str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    live_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 1)])
    spoof_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 0)])
    return sorted_scores, live_prefix, spoof_prefix, int((labels == 1).sum()), int((labels == 0).sum())


def boundary_threshold(scores: np.ndarray, index: int) -> float:
    if index <= 0:
        return float(np.nextafter(scores[0], -np.inf))
    if index >= len(scores):
        return float(np.nextafter(scores[-1], np.inf))
    return float((scores[index - 1] + scores[index]) / 2.0)


def single_metrics(data: tuple[np.ndarray, np.ndarray, np.ndarray, int, int], index: int) -> dict[str, Any]:
    scores, live_prefix, spoof_prefix, live_count, spoof_count = data
    sample_count = len(scores)
    false_rejects = int(live_prefix[index])
    false_accepts = int(spoof_count - spoof_prefix[index])
    apcer = false_accepts / max(spoof_count, 1)
    bpcer = false_rejects / max(live_count, 1)
    return {
        "threshold": boundary_threshold(scores, index),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "accuracy": (sample_count - false_accepts - false_rejects) / max(sample_count, 1),
    }


def band_metrics(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, int, int],
    reject_index: int,
    accept_index: int,
) -> dict[str, Any]:
    scores, live_prefix, spoof_prefix, live_count, spoof_count = data
    sample_count = len(scores)
    false_rejects = int(live_prefix[reject_index])
    false_accepts = int(spoof_count - spoof_prefix[accept_index])
    manual_count = int(accept_index - reject_index)
    apcer = false_accepts / max(spoof_count, 1)
    bpcer = false_rejects / max(live_count, 1)
    return {
        "reject_threshold": boundary_threshold(scores, reject_index),
        "accept_threshold": boundary_threshold(scores, accept_index),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "manual_count": manual_count,
        "auto_accept_count": int(sample_count - accept_index),
        "auto_reject_count": int(reject_index),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "manual_review_rate": manual_count / max(sample_count, 1),
        "auto_accept_rate": (sample_count - accept_index) / max(sample_count, 1),
        "auto_reject_rate": reject_index / max(sample_count, 1),
    }


def best_single(
    rows: list[dict[str, float | int | str]],
    *,
    apcer_target: float,
    bpcer_target: float,
) -> dict[str, Any]:
    data = sorted_arrays(rows)
    candidates = [single_metrics(data, index) for index in range(len(data[0]) + 1)]
    feasible = [item for item in candidates if item["apcer"] < apcer_target and item["bpcer"] < bpcer_target]
    under_apcer = [item for item in candidates if item["apcer"] < apcer_target]
    return {
        "feasible": min(feasible, key=lambda item: (item["apcer"], item["bpcer"], item["threshold"]), default=None),
        "best_with_apcer_target": min(
            under_apcer,
            key=lambda item: (item["bpcer"], item["apcer"], item["threshold"]),
            default=None,
        ),
        "best_acer": min(candidates, key=lambda item: (item["acer"], item["apcer"], item["bpcer"])),
    }


def evaluate_single_threshold(rows: list[dict[str, float | int | str]], threshold: float) -> dict[str, Any]:
    live = [row for row in rows if int(row["label"]) == 1]
    spoof = [row for row in rows if int(row["label"]) == 0]
    false_accepts = sum(1 for row in spoof if float(row["score"]) >= threshold)
    false_rejects = sum(1 for row in live if float(row["score"]) < threshold)
    apcer = false_accepts / max(len(spoof), 1)
    bpcer = false_rejects / max(len(live), 1)
    return {
        "threshold": threshold,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "accuracy": (len(rows) - false_accepts - false_rejects) / max(len(rows), 1),
    }


def evaluate_band_thresholds(
    rows: list[dict[str, float | int | str]],
    *,
    reject_threshold: float,
    accept_threshold: float,
) -> dict[str, Any]:
    live = [row for row in rows if int(row["label"]) == 1]
    spoof = [row for row in rows if int(row["label"]) == 0]
    false_accepts = sum(1 for row in spoof if float(row["score"]) >= accept_threshold)
    false_rejects = sum(1 for row in live if float(row["score"]) < reject_threshold)
    manual_count = sum(1 for row in rows if reject_threshold <= float(row["score"]) < accept_threshold)
    apcer = false_accepts / max(len(spoof), 1)
    bpcer = false_rejects / max(len(live), 1)
    return {
        "reject_threshold": reject_threshold,
        "accept_threshold": accept_threshold,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "manual_count": manual_count,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "manual_review_rate": manual_count / max(len(rows), 1),
        "auto_accept_rate": sum(1 for row in rows if float(row["score"]) >= accept_threshold) / max(len(rows), 1),
        "auto_reject_rate": sum(1 for row in rows if float(row["score"]) < reject_threshold) / max(len(rows), 1),
    }


def evaluate_dev_selection_on_test(dev_selection: dict[str, Any], test_rows: list[dict[str, float | int | str]]) -> dict[str, Any]:
    single_results: dict[str, Any] = {}
    for name, item in dev_selection["single_threshold"].items():
        single_results[name] = None if item is None else evaluate_single_threshold(test_rows, float(item["threshold"]))

    band_results: dict[str, Any] = {}
    for name, item in dev_selection["manual_review_band"].items():
        band_results[name] = (
            None
            if item is None
            else evaluate_band_thresholds(
                test_rows,
                reject_threshold=float(item["reject_threshold"]),
                accept_threshold=float(item["accept_threshold"]),
            )
        )

    return {
        "single_threshold": single_results,
        "manual_review_band": band_results,
    }


def best_band(
    rows: list[dict[str, float | int | str]],
    *,
    apcer_target: float,
    bpcer_target: float,
    manual_review_cap: float,
) -> dict[str, Any]:
    data = sorted_arrays(rows)
    sample_count = len(data[0])
    max_manual = int(math.floor(manual_review_cap * sample_count + 1e-12))
    feasible: dict[str, Any] | None = None
    under_apcer: dict[str, Any] | None = None
    closest: tuple[tuple[float, float, float, float, float], dict[str, Any]] | None = None

    for accept_index in range(sample_count + 1):
        min_reject = max(0, accept_index - max_manual)
        for reject_index in range(min_reject, accept_index + 1):
            item = band_metrics(data, reject_index, accept_index)
            if item["apcer"] < apcer_target:
                if under_apcer is None or (
                    item["bpcer"],
                    item["manual_review_rate"],
                    item["apcer"],
                ) < (
                    under_apcer["bpcer"],
                    under_apcer["manual_review_rate"],
                    under_apcer["apcer"],
                ):
                    under_apcer = item
            if item["apcer"] < apcer_target and item["bpcer"] < bpcer_target:
                if feasible is None or (
                    item["apcer"],
                    item["bpcer"],
                    item["manual_review_rate"],
                ) < (
                    feasible["apcer"],
                    feasible["bpcer"],
                    feasible["manual_review_rate"],
                ):
                    feasible = item

            key = (
                max(0.0, item["apcer"] - apcer_target),
                max(0.0, item["bpcer"] - bpcer_target),
                item["apcer"],
                item["bpcer"],
                item["manual_review_rate"],
            )
            if closest is None or key < closest[0]:
                closest = (key, item)

    return {
        "feasible": feasible,
        "best_with_apcer_target": under_apcer,
        "closest": closest[1] if closest else None,
    }


def render_item(item: dict[str, Any] | None) -> str:
    if item is None:
        return "no feasible point"
    if "accept_threshold" in item:
        return (
            f"reject={item['reject_threshold']:.9f}, accept={item['accept_threshold']:.9f}, "
            f"APCER={100 * item['apcer']:.2f}%, BPCER={100 * item['bpcer']:.2f}%, "
            f"ACER={100 * item['acer']:.2f}%, manual={100 * item['manual_review_rate']:.2f}%, "
            f"FA={item['false_accepts']}, FR={item['false_rejects']}, manual_n={item['manual_count']}"
        )
    return (
        f"threshold={item['threshold']:.9f}, APCER={100 * item['apcer']:.2f}%, "
        f"BPCER={100 * item['bpcer']:.2f}%, ACER={100 * item['acer']:.2f}%, "
        f"FA={item['false_accepts']}, FR={item['false_rejects']}"
    )


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# TinyLiveness Threshold Diagnostic",
        "",
        "Test-selected thresholds are diagnostic only and must not be used as production calibration.",
        "",
    ]
    for section_name in ["dev_selected", "test_selected_diagnostic"]:
        section = payload[section_name]
        title = "Dev-selected" if section_name == "dev_selected" else "Test-selected Diagnostic"
        lines.extend([f"## {title}", ""])
        lines.append("### Single Threshold")
        for name, item in section["single_threshold"].items():
            lines.append(f"- {name}: {render_item(item)}")
        lines.append("")
        lines.append("### Manual Review Band")
        for name, item in section["manual_review_band"].items():
            lines.append(f"- {name}: {render_item(item)}")
        lines.append("")
    if "dev_selected_evaluated_on_test" in payload:
        lines.extend(["## Dev-Selected Evaluated On Test", ""])
        section = payload["dev_selected_evaluated_on_test"]
        lines.append("### Single Threshold")
        for name, item in section["single_threshold"].items():
            lines.append(f"- {name}: {render_item(item)}")
        lines.append("")
        lines.append("### Manual Review Band")
        for name, item in section["manual_review_band"].items():
            lines.append(f"- {name}: {render_item(item)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find strict security threshold operating points.")
    parser.add_argument("--dev-scores", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--apcer-target", type=float, default=0.02)
    parser.add_argument("--bpcer-target", type=float, default=0.05)
    parser.add_argument("--manual-review-cap", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dev_rows = load_scores(args.dev_scores)
    test_rows = load_scores(args.test_scores)
    dev_selection = {
        "single_threshold": best_single(dev_rows, apcer_target=args.apcer_target, bpcer_target=args.bpcer_target),
        "manual_review_band": best_band(
            dev_rows,
            apcer_target=args.apcer_target,
            bpcer_target=args.bpcer_target,
            manual_review_cap=args.manual_review_cap,
        ),
    }
    payload = {
        "note": "Threshold diagnostic. Test-selected thresholds are analysis only.",
        "targets": {
            "apcer_strictly_less_than": args.apcer_target,
            "bpcer_strictly_less_than": args.bpcer_target,
            "manual_review_at_or_below": args.manual_review_cap,
        },
        "sample_counts": {
            "dev": len(dev_rows),
            "test": len(test_rows),
        },
        "dev_selected": dev_selection,
        "dev_selected_evaluated_on_test": evaluate_dev_selection_on_test(dev_selection, test_rows),
        "test_selected_diagnostic": {
            "single_threshold": best_single(test_rows, apcer_target=args.apcer_target, bpcer_target=args.bpcer_target),
            "manual_review_band": best_band(
                test_rows,
                apcer_target=args.apcer_target,
                bpcer_target=args.bpcer_target,
                manual_review_cap=args.manual_review_cap,
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_md)
    print(args.output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
