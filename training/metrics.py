from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


APCER_TARGETS = (0.10, 0.05, 0.03, 0.01)


@dataclass(frozen=True)
class LivenessMetrics:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    apcer: float
    bpcer: float
    acer: float
    true_live_rate: float
    true_spoof_rate: float
    false_live: int
    false_spoof: int
    true_live: int
    true_spoof: int

    @property
    def tar(self) -> float:
        return self.true_live_rate

    @property
    def far(self) -> float:
        return self.apcer

    @property
    def frr(self) -> float:
        return self.bpcer


def _as_scores_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"scores and labels length mismatch: {scores.shape[0]} != {labels.shape[0]}")
    if scores.size == 0:
        raise ValueError("scores and labels must be non-empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values")
    return scores, labels


def metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> LivenessMetrics:
    scores, labels = _as_scores_labels(scores, labels)
    predictions = (scores >= threshold).astype(np.int64)

    live = labels == 1
    spoof = labels == 0
    live_count = max(int(live.sum()), 1)
    spoof_count = max(int(spoof.sum()), 1)

    true_live = int(((predictions == 1) & live).sum())
    false_spoof = int(((predictions == 0) & live).sum())
    true_spoof = int(((predictions == 0) & spoof).sum())
    false_live = int(((predictions == 1) & spoof).sum())

    true_live_rate = true_live / live_count
    true_spoof_rate = true_spoof / spoof_count
    apcer = false_live / spoof_count
    bpcer = false_spoof / live_count
    acer = (apcer + bpcer) / 2.0
    accuracy = float((predictions == labels).mean())
    balanced_accuracy = (true_live_rate + true_spoof_rate) / 2.0

    return LivenessMetrics(
        threshold=float(threshold),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        apcer=apcer,
        bpcer=bpcer,
        acer=acer,
        true_live_rate=true_live_rate,
        true_spoof_rate=true_spoof_rate,
        false_live=false_live,
        false_spoof=false_spoof,
        true_live=true_live,
        true_spoof=true_spoof,
    )


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(np.asarray(scores, dtype=np.float64))
    return np.concatenate(
        [
            np.array([-1e-12], dtype=np.float64),
            unique_scores,
            np.array([1.0 + 1e-12], dtype=np.float64),
        ]
    )


def threshold_curve(scores: np.ndarray, labels: np.ndarray) -> list[LivenessMetrics]:
    scores, labels = _as_scores_labels(scores, labels)
    return [metrics_at_threshold(scores, labels, float(threshold)) for threshold in candidate_thresholds(scores)]


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_apcer: float = 0.05,
) -> LivenessMetrics:
    all_metrics = threshold_curve(scores, labels)
    allowed = [metrics for metrics in all_metrics if metrics.apcer <= max_apcer]
    if allowed:
        return min(
            allowed,
            key=lambda item: (
                item.acer,
                item.bpcer,
                -item.balanced_accuracy,
                item.threshold,
            ),
        )

    return min(
        all_metrics,
        key=lambda item: (
            item.apcer,
            item.acer,
            item.bpcer,
            -item.balanced_accuracy,
        ),
    )


def best_acer_threshold(scores: np.ndarray, labels: np.ndarray) -> LivenessMetrics:
    return min(threshold_curve(scores, labels), key=lambda item: (item.acer, item.apcer, item.bpcer))


def best_balanced_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> LivenessMetrics:
    return max(
        threshold_curve(scores, labels),
        key=lambda item: (item.balanced_accuracy, -item.apcer, -item.bpcer),
    )


def binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = _as_scores_labels(scores, labels)
    positive = labels == 1
    negative = labels == 0
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both live and spoof samples")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end

    positive_rank_sum = float(ranks[positive].sum())
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = _as_scores_labels(scores, labels)
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        raise ValueError("average precision requires at least one live sample")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision * (sorted_labels == 1)).sum() / positive_count)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = _as_scores_labels(scores, labels)
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        raise ValueError("PR AUC requires at least one live sample")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / positive_count
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [float(positive_count / len(labels))]])
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(precision, recall))


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    curve = threshold_curve(scores, labels)
    selected = min(curve, key=lambda item: abs(item.apcer - item.bpcer))
    return {
        "eer": float((selected.apcer + selected.bpcer) / 2.0),
        "threshold": selected.threshold,
        "apcer": selected.apcer,
        "bpcer": selected.bpcer,
    }


def true_live_rate_at_apcer(
    scores: np.ndarray,
    labels: np.ndarray,
    apcer_targets: tuple[float, ...] = (0.01, 0.005, 0.001),
) -> dict[float, float]:
    scores, labels = _as_scores_labels(scores, labels)
    if int((labels == 1).sum()) == 0 or int((labels == 0).sum()) == 0:
        raise ValueError("TLR@APCER requires both live and spoof samples")

    return {
        target: select_threshold(scores, labels, max_apcer=target).true_live_rate
        for target in apcer_targets
    }


def bpcer_at_apcer_targets(
    scores: np.ndarray,
    labels: np.ndarray,
    targets: tuple[float, ...] = APCER_TARGETS,
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for target in targets:
        selected = select_threshold(scores, labels, max_apcer=target)
        key = f"bpcer{int(round(1 / target))}" if target > 0 else "bpcer"
        results[key] = selected.bpcer if selected.apcer <= target else None
    return results


def security_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    targets: tuple[float, ...] = APCER_TARGETS,
) -> dict[str, dict[str, float | int]]:
    thresholds = {
        "best_acer": metrics_to_dict(best_acer_threshold(scores, labels)),
        "best_balanced_accuracy": metrics_to_dict(best_balanced_accuracy_threshold(scores, labels)),
    }
    for target in targets:
        selected = select_threshold(scores, labels, max_apcer=target)
        thresholds[f"apcer_le_{target:g}"] = metrics_to_dict(selected)
    return thresholds


def choose_security_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    preferred_targets: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> dict[str, float | str]:
    for target in preferred_targets:
        selected = select_threshold(scores, labels, max_apcer=target)
        if selected.apcer <= target:
            return {
                "threshold": selected.threshold,
                "target_apcer": target,
                "policy": f"apcer_le_{target:g}",
                "apcer": selected.apcer,
                "bpcer": selected.bpcer,
                "acer": selected.acer,
            }
    selected = select_threshold(scores, labels, max_apcer=preferred_targets[-1])
    return {
        "threshold": selected.threshold,
        "target_apcer": float("nan"),
        "policy": "lowest_available_apcer",
        "apcer": selected.apcer,
        "bpcer": selected.bpcer,
        "acer": selected.acer,
    }


def confusion_matrix_dict(metrics: LivenessMetrics) -> dict[str, int]:
    return {
        "true_live": metrics.true_live,
        "false_spoof": metrics.false_spoof,
        "true_spoof": metrics.true_spoof,
        "false_live": metrics.false_live,
    }


def metrics_to_dict(metrics: LivenessMetrics) -> dict[str, float | int]:
    values = asdict(metrics)
    values["tar"] = metrics.tar
    values["far"] = metrics.far
    values["frr"] = metrics.frr
    values["confusion_matrix"] = confusion_matrix_dict(metrics)
    return values


def ece_score(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    scores, labels = _as_scores_labels(scores, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / total) * abs(confidence - accuracy)
    return float(ece)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = _as_scores_labels(scores, labels)
    return float(np.mean((scores - labels) ** 2))


def decision_band_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    reject_threshold: float,
    accept_threshold: float,
) -> dict[str, float | int]:
    scores, labels = _as_scores_labels(scores, labels)
    if reject_threshold > accept_threshold:
        raise ValueError("reject_threshold must be <= accept_threshold")
    auto_reject = scores < reject_threshold
    auto_accept = scores >= accept_threshold
    manual = ~(auto_reject | auto_accept)
    live = labels == 1
    spoof = labels == 0
    auto_accepted_spoof = int((auto_accept & spoof).sum())
    auto_rejected_live = int((auto_reject & live).sum())
    return {
        "reject_threshold": float(reject_threshold),
        "accept_threshold": float(accept_threshold),
        "auto_accept_rate": float(auto_accept.mean()),
        "auto_reject_rate": float(auto_reject.mean()),
        "manual_review_rate": float(manual.mean()),
        "auto_accept_count": int(auto_accept.sum()),
        "auto_reject_count": int(auto_reject.sum()),
        "manual_review_count": int(manual.sum()),
        "apcer_auto_accepted": auto_accepted_spoof / max(int(spoof.sum()), 1),
        "bpcer_auto_rejected": auto_rejected_live / max(int(live.sum()), 1),
        "auto_accepted_spoof": auto_accepted_spoof,
        "auto_rejected_live": auto_rejected_live,
    }


def full_evaluation(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float | None = None,
    decision_reject_threshold: float | None = None,
    decision_accept_threshold: float | None = None,
) -> dict[str, object]:
    scores, labels = _as_scores_labels(scores, labels)
    best_acer = best_acer_threshold(scores, labels)
    selected = metrics_at_threshold(scores, labels, threshold) if threshold is not None else best_acer
    thresholds = security_thresholds(scores, labels)
    result: dict[str, object] = {
        "samples": int(len(scores)),
        "live_samples": int((labels == 1).sum()),
        "spoof_samples": int((labels == 0).sum()),
        "roc_auc": safe_metric(binary_roc_auc, scores, labels),
        "pr_auc": safe_metric(pr_auc, scores, labels),
        "average_precision": safe_metric(average_precision, scores, labels),
        "eer": equal_error_rate(scores, labels),
        "selected_threshold": metrics_to_dict(selected),
        "thresholds": thresholds,
        "bpcer10": thresholds["apcer_le_0.1"]["bpcer"],
        "bpcer20": thresholds["apcer_le_0.05"]["bpcer"],
        "bpcer100": thresholds["apcer_le_0.01"]["bpcer"],
        "ece": ece_score(scores, labels),
        "brier_score": brier_score(scores, labels),
    }
    if decision_reject_threshold is not None and decision_accept_threshold is not None:
        result["decision_band"] = decision_band_metrics(
            scores,
            labels,
            reject_threshold=decision_reject_threshold,
            accept_threshold=decision_accept_threshold,
        )
    return result


def safe_metric(function, scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(function(scores, labels))
    except ValueError:
        return float("nan")


def subgroup_evaluation(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
    *,
    threshold: float,
    group_columns: Iterable[str],
    min_group_size: int = 5,
) -> list[dict[str, object]]:
    scores, labels = _as_scores_labels(scores, labels)
    if len(metadata) != len(scores):
        raise ValueError("metadata length must match scores")
    rows: list[dict[str, object]] = []
    for column in group_columns:
        values = sorted({item.get(column, "") or "unknown" for item in metadata})
        for value in values:
            indices = [index for index, item in enumerate(metadata) if (item.get(column, "") or "unknown") == value]
            if len(indices) < min_group_size:
                continue
            idx = np.asarray(indices, dtype=np.int64)
            group_scores = scores[idx]
            group_labels = labels[idx]
            metrics = metrics_at_threshold(group_scores, group_labels, threshold)
            live_count = int((group_labels == 1).sum())
            spoof_count = int((group_labels == 0).sum())
            row: dict[str, object] = {
                "group": column,
                "value": value,
                "samples": int(len(idx)),
                "live_samples": live_count,
                "spoof_samples": spoof_count,
                "roc_auc": safe_metric(binary_roc_auc, group_scores, group_labels),
                "apcer": metrics.apcer if spoof_count else None,
                "bpcer": metrics.bpcer if live_count else None,
                "acer": metrics.acer if live_count and spoof_count else None,
                "false_accepts": metrics.false_live,
                "false_rejects": metrics.false_spoof,
                "threshold": float(threshold),
            }
            rows.append(row)
    return rows


def worst_groups(rows: list[dict[str, object]], limit: int = 20) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -1.0 if row.get("apcer") is None else float(row["apcer"]),
            int(row.get("samples", 0)),
        ),
        reverse=True,
    )[:limit]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    path: Path,
    *,
    title: str,
    evaluation: dict[str, object],
    subgroup_rows: list[dict[str, object]] | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    selected = evaluation["selected_threshold"]
    assert isinstance(selected, dict)
    thresholds = evaluation["thresholds"]
    assert isinstance(thresholds, dict)
    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
        f"- Samples: {evaluation['samples']}",
        f"- Live samples: {evaluation['live_samples']}",
        f"- Spoof samples: {evaluation['spoof_samples']}",
        f"- ROC AUC: {format_metric(evaluation['roc_auc'])}",
        f"- PR AUC: {format_metric(evaluation['pr_auc'])}",
        f"- Average precision: {format_metric(evaluation['average_precision'])}",
        f"- EER: {format_metric(cast_float_dict(evaluation['eer'])['eer'])}",
        f"- Accuracy: {format_metric(selected['accuracy'])}",
        f"- Balanced accuracy: {format_metric(selected['balanced_accuracy'])}",
        f"- APCER: {format_metric(selected['apcer'])}",
        f"- BPCER: {format_metric(selected['bpcer'])}",
        f"- ACER: {format_metric(selected['acer'])}",
        f"- Threshold: {format_metric(selected['threshold'])}",
        f"- BPCER10: {format_metric(evaluation['bpcer10'])}",
        f"- BPCER20: {format_metric(evaluation['bpcer20'])}",
        f"- BPCER100: {format_metric(evaluation['bpcer100'])}",
        f"- ECE: {format_metric(evaluation['ece'])}",
        f"- Brier score: {format_metric(evaluation['brier_score'])}",
        "",
        "## Security Thresholds",
        "",
        "| Policy | Threshold | APCER | BPCER | ACER |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, values in thresholds.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {policy} | {format_metric(values['threshold'])} | "
            f"{format_metric(values['apcer'])} | {format_metric(values['bpcer'])} | "
            f"{format_metric(values['acer'])} |"
        )

    if "decision_band" in evaluation:
        band = evaluation["decision_band"]
        assert isinstance(band, dict)
        lines.extend(
            [
                "",
                "## Decision Band",
                "",
                f"- Auto accept rate: {format_metric(band['auto_accept_rate'])}",
                f"- Auto reject rate: {format_metric(band['auto_reject_rate'])}",
                f"- Manual review rate: {format_metric(band['manual_review_rate'])}",
                f"- APCER among auto-accepted: {format_metric(band['apcer_auto_accepted'])}",
                f"- BPCER among auto-rejected: {format_metric(band['bpcer_auto_rejected'])}",
            ]
        )

    if subgroup_rows:
        lines.extend(["", "## Worst Groups", "", "| Group | Value | Samples | APCER | BPCER | ACER | False accepts | False rejects |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in worst_groups(subgroup_rows, limit=20):
            lines.append(
                f"| {row['group']} | {row['value']} | {row['samples']} | "
                f"{format_metric(row.get('apcer'))} | {format_metric(row.get('bpcer'))} | "
                f"{format_metric(row.get('acer'))} | {row['false_accepts']} | {row['false_rejects']} |"
            )

    if extra_lines:
        lines.extend(["", *extra_lines])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cast_float_dict(value: object) -> dict[str, float]:
    assert isinstance(value, dict)
    return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))}


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return str(value)
