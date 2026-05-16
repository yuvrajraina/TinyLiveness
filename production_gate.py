from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


GATES = {
    "beta": {
        "apcer": 0.10,
        "bpcer": 0.10,
        "acer": 0.08,
        "roc_auc": 0.95,
        "max_subgroup_apcer": None,
        "bpcer100": None,
    },
    "production": {
        "apcer": 0.03,
        "bpcer": 0.10,
        "acer": 0.05,
        "roc_auc": 0.98,
        "max_subgroup_apcer": 0.05,
        "bpcer100": None,
    },
    "strong-production": {
        "apcer": 0.01,
        "bpcer": 0.05,
        "acer": 0.03,
        "roc_auc": 0.99,
        "max_subgroup_apcer": 0.03,
        "bpcer100": 0.10,
    },
}


def as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def load_metrics(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if "evaluation" in report and isinstance(report["evaluation"], dict):
        evaluation = report["evaluation"]
        selected = evaluation.get("selected_threshold", {})
        if not isinstance(selected, dict):
            raise ValueError("report evaluation is missing selected_threshold")
        metrics = {
            "roc_auc": evaluation.get("roc_auc"),
            "apcer": selected.get("apcer"),
            "bpcer": selected.get("bpcer"),
            "acer": selected.get("acer"),
            "bpcer100": evaluation.get("bpcer100"),
        }
        subgroups = report.get("subgroups", [])
        return metrics, subgroups if isinstance(subgroups, list) else []

    if "rows" in report and isinstance(report["rows"], list) and report["rows"]:
        overall = report["rows"][0]
        metrics = {
            "roc_auc": overall.get("roc_auc"),
            "apcer": overall.get("apcer"),
            "bpcer": overall.get("bpcer"),
            "acer": overall.get("acer"),
            "bpcer100": report.get("bpcer100"),
        }
        return metrics, report["rows"][1:]

    raise ValueError("unsupported report format")


def gate_result(metrics: dict[str, Any], subgroups: list[dict[str, Any]], level: str) -> dict[str, Any]:
    spec = GATES[level]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: Any, op: str, limit: float | None) -> None:
        if limit is None:
            return
        metric_value = as_float(value)
        if op == "<=":
            passed = not math.isnan(metric_value) and metric_value <= limit
        elif op == ">=":
            passed = not math.isnan(metric_value) and metric_value >= limit
        else:
            raise ValueError(op)
        checks.append(
            {
                "metric": name,
                "value": metric_value,
                "operator": op,
                "limit": limit,
                "passed": passed,
            }
        )

    add_check("test_apcer", metrics.get("apcer"), "<=", spec["apcer"])
    add_check("test_bpcer", metrics.get("bpcer"), "<=", spec["bpcer"])
    add_check("test_acer", metrics.get("acer"), "<=", spec["acer"])
    add_check("test_roc_auc", metrics.get("roc_auc"), ">=", spec["roc_auc"])
    add_check("bpcer100", metrics.get("bpcer100"), "<=", spec["bpcer100"])

    subgroup_limit = spec["max_subgroup_apcer"]
    subgroup_failures: list[dict[str, Any]] = []
    if subgroup_limit is not None and subgroups:
        for row in subgroups:
            apcer = row.get("apcer")
            spoof_samples = int(row.get("spoof_samples") or 0)
            if apcer is None or spoof_samples <= 0:
                continue
            apcer_value = float(apcer)
            if not math.isnan(apcer_value) and apcer_value > subgroup_limit:
                subgroup_failures.append(row)
        checks.append(
            {
                "metric": "max_subgroup_apcer",
                "value": max([float(row.get("apcer") or 0.0) for row in subgroups], default=float("nan")),
                "operator": "<=",
                "limit": subgroup_limit,
                "passed": not subgroup_failures,
            }
        )

    passed = all(bool(check["passed"]) for check in checks)
    return {
        "level": level,
        "passed": passed,
        "checks": checks,
        "subgroup_failures": subgroup_failures,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# TinyLiveness Production Gate: {result['level']}",
        "",
        f"Result: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        "| Metric | Value | Rule | Limit | Status |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for check in result["checks"]:
        lines.append(
            f"| {check['metric']} | {float(check['value']):.6f} | "
            f"{check['operator']} | {float(check['limit']):.6f} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    failures = result.get("subgroup_failures", [])
    if failures:
        lines.extend(["", "## Subgroup Failures", ""])
        for row in failures[:30]:
            lines.append(
                f"- {row.get('group')}={row.get('value')}: APCER={float(row.get('apcer')):.6f}, "
                f"samples={row.get('samples')}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyLiveness production gate.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--level", choices=list(GATES), default="production")
    parser.add_argument("--output-json", type=Path, default=Path("reports/production_gate.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/production_gate.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metrics, subgroups = load_metrics(report)
    result = gate_result(metrics, subgroups, args.level)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(args.output_md, result)
    print(f"level: {args.level}")
    print(f"passed: {result['passed']}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_md}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
