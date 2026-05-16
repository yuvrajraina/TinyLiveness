from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_MODEL_FILENAME = "tinyliveness_main_apcer1_224.onnx"
DEFAULT_POLICY_FILENAME = "decision_policy_main_apcer1_224.json"
DEFAULT_THRESHOLDS_FILENAME = "thresholds_main_apcer1_224.json"
DEFAULT_NORMALIZATION = "imagenet"
DEFAULT_IMAGE_SIZE = 224

_ASSET_PACKAGE = "tinyliveness.assets"
_RESOURCE_STACK = ExitStack()
_RESOURCE_LOCK = Lock()
_PATH_CACHE: dict[str, Path] = {}


def _asset_path(filename: str) -> Path:
    with _RESOURCE_LOCK:
        cached = _PATH_CACHE.get(filename)
        if cached is not None:
            return cached

        resource = files(_ASSET_PACKAGE).joinpath(filename)
        path = Path(_RESOURCE_STACK.enter_context(as_file(resource)))
        if not path.exists():
            raise FileNotFoundError(f"bundled TinyLiveness artifact not found: {filename}")
        _PATH_CACHE[filename] = path
        return path


def get_default_model_path() -> Path:
    """Return the bundled APCER 1% FP32 ONNX model path."""

    return _asset_path(DEFAULT_MODEL_FILENAME)


def get_default_policy_path() -> Path:
    """Return the bundled APCER 1% decision policy JSON path."""

    return _asset_path(DEFAULT_POLICY_FILENAME)


def get_default_thresholds_path() -> Path:
    """Return the bundled APCER 1% thresholds JSON path."""

    return _asset_path(DEFAULT_THRESHOLDS_FILENAME)


def create_default_onnx_detector(**kwargs: Any) -> Any:
    """Create an ONNX detector using the bundled release model and policy."""

    from .inference import OnnxLivenessDetector

    kwargs.setdefault("thresholds_path", get_default_policy_path())
    kwargs.setdefault("normalization", DEFAULT_NORMALIZATION)
    kwargs.setdefault("image_size", DEFAULT_IMAGE_SIZE)
    return OnnxLivenessDetector(get_default_model_path(), **kwargs)
