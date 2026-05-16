from __future__ import annotations

import json

import numpy as np
import pytest

from tinyliveness import (
    create_default_onnx_detector,
    get_default_model_path,
    get_default_policy_path,
    normalize_rgb_image,
    resolve_decision_policy,
)


def test_default_release_artifacts_exist() -> None:
    model_path = get_default_model_path()
    policy_path = get_default_policy_path()

    assert model_path.name == "tinyliveness_main_apcer1_224.onnx"
    assert policy_path.name == "decision_policy_main_apcer1_224.json"
    assert model_path.is_file()
    assert model_path.stat().st_size > 10 * 1024 * 1024
    assert policy_path.is_file()


def test_default_policy_resolves_apcer1_thresholds() -> None:
    reject_threshold, accept_threshold, threshold_policy = resolve_decision_policy(
        threshold=0.5,
        thresholds_path=get_default_policy_path(),
    )

    assert reject_threshold == pytest.approx(0.99)
    assert accept_threshold == pytest.approx(0.99)
    assert threshold_policy == "main_apcer1_threshold"


def test_invalid_threshold_policy_is_rejected(tmp_path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"decision_policy": {"reject_threshold": 1.2, "accept_threshold": 1.2}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite probability"):
        resolve_decision_policy(threshold=0.5, thresholds_path=policy_path)


def test_normalize_rgb_image_supports_release_image_size() -> None:
    image = np.full((224, 224, 3), 127, dtype=np.uint8)

    tensor = normalize_rgb_image(image, image_size=224, normalization="imagenet")

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_default_onnx_detector_smoke() -> None:
    pytest.importorskip("onnxruntime")
    image = np.full((224, 224, 3), 127, dtype=np.uint8)

    detector = create_default_onnx_detector()
    result = detector.predict_image(image)

    assert 0.0 <= result.live_probability <= 1.0
    assert result.decision in {"spoof", "manual_review", "live"}
    assert result.threshold_policy == "main_apcer1_threshold"
