from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_LIVE_THRESHOLD",
    "FACE_IMAGE_SIZE",
    "LivenessResult",
    "OnnxLivenessDetector",
    "SequenceLivenessResult",
    "MobileNetV3Liveness",
    "TinyLiveness",
    "TinyLivenessProbability",
    "TorchLivenessDetector",
    "aggregate_probabilities",
    "apply_score_calibration",
    "build_liveness_model",
    "count_parameters",
    "create_default_onnx_detector",
    "decision_from_probability",
    "ensure_nchw_batch",
    "get_default_model_path",
    "get_default_policy_path",
    "get_default_thresholds_path",
    "load_threshold_policy",
    "make_liveness_result",
    "normalize_rgb_image",
    "normalize_rgb_images",
    "resolve_decision_policy",
    "sigmoid",
]

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "DEFAULT_LIVE_THRESHOLD": ("tinyliveness.inference", "DEFAULT_LIVE_THRESHOLD"),
    "FACE_IMAGE_SIZE": ("tinyliveness.inference", "FACE_IMAGE_SIZE"),
    "LivenessResult": ("tinyliveness.inference", "LivenessResult"),
    "OnnxLivenessDetector": ("tinyliveness.inference", "OnnxLivenessDetector"),
    "SequenceLivenessResult": ("tinyliveness.inference", "SequenceLivenessResult"),
    "MobileNetV3Liveness": ("tinyliveness.model", "MobileNetV3Liveness"),
    "TinyLiveness": ("tinyliveness.model", "TinyLiveness"),
    "TinyLivenessProbability": ("tinyliveness.model", "TinyLivenessProbability"),
    "TorchLivenessDetector": ("tinyliveness.inference", "TorchLivenessDetector"),
    "aggregate_probabilities": ("tinyliveness.inference", "aggregate_probabilities"),
    "apply_score_calibration": ("tinyliveness.inference", "apply_score_calibration"),
    "build_liveness_model": ("tinyliveness.model", "build_liveness_model"),
    "count_parameters": ("tinyliveness.model", "count_parameters"),
    "create_default_onnx_detector": ("tinyliveness.artifacts", "create_default_onnx_detector"),
    "decision_from_probability": ("tinyliveness.inference", "decision_from_probability"),
    "ensure_nchw_batch": ("tinyliveness.inference", "ensure_nchw_batch"),
    "get_default_model_path": ("tinyliveness.artifacts", "get_default_model_path"),
    "get_default_policy_path": ("tinyliveness.artifacts", "get_default_policy_path"),
    "get_default_thresholds_path": ("tinyliveness.artifacts", "get_default_thresholds_path"),
    "load_threshold_policy": ("tinyliveness.inference", "load_threshold_policy"),
    "make_liveness_result": ("tinyliveness.inference", "make_liveness_result"),
    "normalize_rgb_image": ("tinyliveness.inference", "normalize_rgb_image"),
    "normalize_rgb_images": ("tinyliveness.inference", "normalize_rgb_images"),
    "resolve_decision_policy": ("tinyliveness.inference", "resolve_decision_policy"),
    "sigmoid": ("tinyliveness.inference", "sigmoid"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'tinyliveness' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])
