from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

FACE_IMAGE_SIZE = 112
RGB_CHANNELS = 3
PROJECT_MEAN = 127.5
PROJECT_SCALE = 128.0
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_LIVE_THRESHOLD = 0.5
DEFAULT_MAX_BATCH_SIZE = 64
UNSAFE_TORCH_LOAD_ENV = "TINYLIVENESS_ALLOW_UNSAFE_TORCH_LOAD"
AggregationMethod = Literal["mean", "median", "min", "p10"]
NormalizationMode = Literal["tinyliveness", "imagenet"]


@dataclass(frozen=True)
class LivenessResult:
    live_probability: float
    threshold: float
    is_live: bool
    decision: str = ""
    threshold_policy: str = "single_threshold"
    reject_threshold: float | None = None
    accept_threshold: float | None = None

    @property
    def spoof_probability(self) -> float:
        return 1.0 - self.live_probability

    @property
    def is_spoof(self) -> bool:
        if self.decision:
            return self.decision == "spoof"
        return not self.is_live

    @property
    def needs_manual_review(self) -> bool:
        return self.decision == "manual_review"


@dataclass(frozen=True)
class SequenceLivenessResult(LivenessResult):
    frame_probabilities: tuple[float, ...] = ()
    aggregation: str = "mean"


def aggregate_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    method: AggregationMethod = "mean",
) -> float:
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("at least one frame probability is required")
    if not np.isfinite(scores).all():
        raise ValueError("frame probabilities contain NaN or infinite values")

    if method == "mean":
        return float(np.mean(scores))
    if method == "median":
        return float(np.median(scores))
    if method == "min":
        return float(np.min(scores))
    if method == "p10":
        return float(np.percentile(scores, 10))
    raise ValueError(f"unknown aggregation method: {method!r}")


def normalize_rgb_image(
    image: np.ndarray,
    *,
    image_size: int = FACE_IMAGE_SIZE,
    normalization: NormalizationMode = "tinyliveness",
) -> np.ndarray:
    """Convert an aligned RGB face image to a CHW float32 tensor.

    Inputs may be uint8/float pixels in 0..255 or float pixels in 0..1. Face
    detection and alignment should happen before this function is called.
    """

    array = np.asarray(image)
    expected_shape = (image_size, image_size, RGB_CHANNELS)
    if array.shape != expected_shape:
        raise ValueError(f"expected RGB image shape {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image must contain numeric pixels, got {array.dtype}")

    image_f32 = array.astype(np.float32, copy=False)
    if not np.isfinite(image_f32).all():
        raise ValueError("image contains NaN or infinite values")

    min_value = float(np.min(image_f32))
    max_value = float(np.max(image_f32))
    if np.issubdtype(array.dtype, np.floating) and 0.0 <= min_value and max_value <= 1.0:
        image_f32 = image_f32 * 255.0
    elif min_value < 0.0 or max_value > 255.0:
        raise ValueError("image pixels must be in 0..255 or 0..1 range")

    if normalization == "tinyliveness":
        normalized = (image_f32 - PROJECT_MEAN) / PROJECT_SCALE
    elif normalization == "imagenet":
        normalized = ((image_f32 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(f"unknown normalization: {normalization!r}")
    return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)), dtype=np.float32)


def ensure_nchw_batch(
    face_tensor: np.ndarray,
    *,
    image_size: int = FACE_IMAGE_SIZE,
) -> np.ndarray:
    array = np.asarray(face_tensor, dtype=np.float32)
    if array.ndim == 3:
        array = array[np.newaxis, ...]

    expected_tail = (RGB_CHANNELS, image_size, image_size)
    if array.ndim != 4 or tuple(array.shape[1:]) != expected_tail:
        raise ValueError(
            "expected face tensor shape "
            f"(N, {RGB_CHANNELS}, {image_size}, {image_size}) or {expected_tail}, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("face tensor contains NaN or infinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_rgb_images(
    images: Iterable[np.ndarray],
    *,
    image_size: int = FACE_IMAGE_SIZE,
    normalization: NormalizationMode = "tinyliveness",
) -> np.ndarray:
    tensors = [
        normalize_rgb_image(image, image_size=image_size, normalization=normalization)
        for image in images
    ]
    if not tensors:
        raise ValueError("at least one image is required")
    return np.stack(tensors, axis=0).astype(np.float32, copy=False)


def sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = np.empty_like(array, dtype=np.float32)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def load_threshold_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"threshold policy must be a JSON object: {path}")
    return payload


def _validate_probability(name: str, value: Any) -> float:
    probability = float(value)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}")
    return probability


def resolve_decision_policy(
    *,
    threshold: float,
    thresholds_path: str | Path | None = None,
    reject_threshold: float | None = None,
    accept_threshold: float | None = None,
    threshold_policy: str = "single_threshold",
) -> tuple[float, float, str]:
    resolved_reject = _validate_probability(
        "reject_threshold",
        threshold if reject_threshold is None else reject_threshold,
    )
    resolved_accept = _validate_probability(
        "accept_threshold",
        threshold if accept_threshold is None else accept_threshold,
    )
    resolved_policy = str(threshold_policy).strip()
    if thresholds_path is not None:
        payload = load_threshold_policy(thresholds_path)
        decision_policy = payload.get("decision_policy")
        if isinstance(decision_policy, dict):
            resolved_reject = _validate_probability(
                "decision_policy.reject_threshold",
                decision_policy.get("reject_threshold", resolved_reject),
            )
            resolved_accept = _validate_probability(
                "decision_policy.accept_threshold",
                decision_policy.get("accept_threshold", payload.get("threshold", resolved_accept)),
            )
            resolved_policy = str(
                decision_policy.get("threshold_policy", payload.get("policy", resolved_policy))
            ).strip()
        elif isinstance(payload.get("production"), dict):
            production = payload["production"]
            resolved_accept = _validate_probability("production.threshold", production["threshold"])
            resolved_reject = resolved_accept
            resolved_policy = str(production.get("policy", "production_threshold")).strip()
        elif "threshold" in payload:
            resolved_accept = _validate_probability("threshold", payload["threshold"])
            resolved_reject = resolved_accept
            resolved_policy = str(payload.get("policy", "threshold_json")).strip()
        else:
            raise ValueError(f"could not read threshold policy from {thresholds_path}")
    if resolved_reject > resolved_accept:
        raise ValueError("reject_threshold must be <= accept_threshold")
    if not resolved_policy:
        raise ValueError("threshold_policy must be a non-empty string")
    return resolved_reject, resolved_accept, resolved_policy


def load_score_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"score calibration must be a JSON object: {path}")
    return payload


def apply_score_calibration(probabilities: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    if calibration is None:
        return probabilities
    scores = np.asarray(probabilities, dtype=np.float32)
    if isinstance(calibration.get("isotonic"), dict) and isinstance(calibration["isotonic"].get("model"), dict):
        model = calibration["isotonic"]["model"]
        x = np.asarray(model["x_thresholds"], dtype=np.float32)
        y = np.asarray(model["y_thresholds"], dtype=np.float32)
        if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.size != y.size:
            raise ValueError("isotonic calibration thresholds must be equal-length 1D arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("isotonic calibration contains NaN or infinite values")
        if np.any(np.diff(x) < 0):
            raise ValueError("isotonic x_thresholds must be sorted ascending")
        return np.interp(scores, x, y).astype(np.float32)
    temperature_block = calibration.get("temperature_scaling")
    if isinstance(temperature_block, dict) and "temperature" in temperature_block:
        temperature = max(float(temperature_block["temperature"]), 1e-6)
        if not np.isfinite(temperature):
            raise ValueError("temperature calibration must be finite")
        clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        return sigmoid(logits / temperature).astype(np.float32)
    return scores


def decision_from_probability(
    probability: float,
    *,
    reject_threshold: float,
    accept_threshold: float,
) -> str:
    resolved_probability = _validate_probability("probability", probability)
    if resolved_probability < reject_threshold:
        return "spoof"
    if resolved_probability >= accept_threshold:
        return "live"
    return "manual_review"


def make_liveness_result(
    probability: float,
    *,
    reject_threshold: float,
    accept_threshold: float,
    threshold_policy: str,
) -> LivenessResult:
    decision = decision_from_probability(
        probability,
        reject_threshold=reject_threshold,
        accept_threshold=accept_threshold,
    )
    return LivenessResult(
        live_probability=probability,
        threshold=accept_threshold,
        is_live=decision == "live",
        decision=decision,
        threshold_policy=threshold_policy,
        reject_threshold=reject_threshold,
        accept_threshold=accept_threshold,
    )


class OnnxLivenessDetector:
    """ONNX Runtime wrapper for TinyLiveness models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = DEFAULT_LIVE_THRESHOLD,
        providers: Sequence[str] | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
        session_options: Any | None = None,
        output_is_probability: bool = True,
        normalization: NormalizationMode = "tinyliveness",
        image_size: int = FACE_IMAGE_SIZE,
        thresholds_path: str | Path | None = None,
        reject_threshold: float | None = None,
        accept_threshold: float | None = None,
        threshold_policy: str = "single_threshold",
        score_calibration_path: str | Path | None = None,
        max_batch_size: int | None = DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "OnnxLivenessDetector requires the 'onnx' extra: "
                "pip install 'tinyliveness[onnx]'"
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.reject_threshold, self.accept_threshold, self.threshold_policy = resolve_decision_policy(
            threshold=float(threshold),
            thresholds_path=thresholds_path,
            reject_threshold=reject_threshold,
            accept_threshold=accept_threshold,
            threshold_policy=threshold_policy,
        )
        self.threshold = self.accept_threshold
        self.output_is_probability = output_is_probability
        self.normalization = normalization
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive or None")
        self.score_calibration = load_score_calibration(score_calibration_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=list(providers or ["CPUExecutionProvider"]),
        )
        session_inputs = self.session.get_inputs()
        session_outputs = self.session.get_outputs()
        if not session_inputs:
            raise RuntimeError(f"ONNX model has no inputs: {self.model_path}")
        if not session_outputs:
            raise RuntimeError(f"ONNX model has no outputs: {self.model_path}")

        self.input_name = input_name or session_inputs[0].name
        self.output_name = output_name or session_outputs[0].name

    def predict_batch(self, face_tensor: np.ndarray) -> np.ndarray:
        batch = ensure_nchw_batch(face_tensor, image_size=self.image_size)
        if self.max_batch_size is not None and batch.shape[0] > self.max_batch_size:
            raise ValueError(
                f"batch size {batch.shape[0]} exceeds max_batch_size={self.max_batch_size}"
            )
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        if not self.output_is_probability:
            scores = sigmoid(scores)
        scores = apply_score_calibration(scores, self.score_calibration)
        if not np.isfinite(scores).all():
            raise RuntimeError("ONNX liveness output contains NaN or infinite values")
        return np.clip(scores, 0.0, 1.0)

    def predict_image(self, image: np.ndarray) -> LivenessResult:
        probability = float(
            self.predict_batch(
                normalize_rgb_image(image, normalization=self.normalization)
                if self.image_size == FACE_IMAGE_SIZE
                else normalize_rgb_image(
                    image,
                    image_size=self.image_size,
                    normalization=self.normalization,
                )
            )[0]
        )
        return make_liveness_result(
            probability,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
            threshold_policy=self.threshold_policy,
        )

    def predict_images(self, images: Iterable[np.ndarray]) -> np.ndarray:
        tensors = [
            normalize_rgb_image(
                image,
                image_size=self.image_size,
                normalization=self.normalization,
            )
            for image in images
        ]
        if not tensors:
            raise ValueError("at least one image is required")
        return self.predict_batch(np.stack(tensors, axis=0).astype(np.float32, copy=False))

    def predict_sequence(
        self,
        frames: Iterable[np.ndarray],
        *,
        aggregation: AggregationMethod = "mean",
    ) -> SequenceLivenessResult:
        frame_probabilities = self.predict_images(frames)
        probability = aggregate_probabilities(frame_probabilities, method=aggregation)
        decision = decision_from_probability(
            probability,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
        )
        return SequenceLivenessResult(
            live_probability=probability,
            threshold=self.accept_threshold,
            is_live=decision == "live",
            decision=decision,
            threshold_policy=self.threshold_policy,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
            frame_probabilities=tuple(float(score) for score in frame_probabilities),
            aggregation=aggregation,
        )


class TorchLivenessDetector:
    """PyTorch checkpoint wrapper for TinyLiveness models."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        threshold: float = DEFAULT_LIVE_THRESHOLD,
        device: str = "cpu",
        width_mult: float | None = None,
        thresholds_path: str | Path | None = None,
        reject_threshold: float | None = None,
        accept_threshold: float | None = None,
        threshold_policy: str = "single_threshold",
        score_calibration_path: str | Path | None = None,
        max_batch_size: int | None = DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchLivenessDetector requires PyTorch") from exc

        from .model import build_liveness_model

        self.torch = torch
        self.device = torch.device(device)
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights not found: {self.weights_path}")

        checkpoint = _torch_load(self.weights_path, self.device, torch)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        resolved_width_mult = float(
            width_mult if width_mult is not None else config.get("width_mult", 0.5)
        )
        configured_threshold = float(config.get("threshold", threshold))
        self.reject_threshold, self.accept_threshold, self.threshold_policy = resolve_decision_policy(
            threshold=configured_threshold,
            thresholds_path=thresholds_path,
            reject_threshold=reject_threshold,
            accept_threshold=accept_threshold,
            threshold_policy=threshold_policy,
        )
        self.threshold = self.accept_threshold
        dropout = float(config.get("dropout", 0.0))
        architecture = str(config.get("architecture", "tiny"))
        self.normalization: NormalizationMode = config.get("normalization", "tinyliveness")
        self.image_size = int(config.get("input_size", FACE_IMAGE_SIZE))
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive or None")
        self.score_calibration = load_score_calibration(score_calibration_path)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model = build_liveness_model(
            architecture,
            width_mult=resolved_width_mult,
            dropout=dropout,
            pretrained=False,
        )
        self.model.load_state_dict(_clean_state_dict_keys(state_dict))
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, face_tensor: np.ndarray) -> np.ndarray:
        batch = ensure_nchw_batch(face_tensor, image_size=self.image_size)
        if self.max_batch_size is not None and batch.shape[0] > self.max_batch_size:
            raise ValueError(
                f"batch size {batch.shape[0]} exceeds max_batch_size={self.max_batch_size}"
            )
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(batch).to(self.device)
            logits = self.model(tensor).detach().cpu().numpy()
        scores = sigmoid(logits).reshape(-1)
        scores = apply_score_calibration(scores, self.score_calibration)
        return np.clip(scores, 0.0, 1.0)

    def predict_image(self, image: np.ndarray) -> LivenessResult:
        probability = float(
            self.predict_batch(
                normalize_rgb_image(image, normalization=self.normalization)
                if self.image_size == FACE_IMAGE_SIZE
                else normalize_rgb_image(
                    image,
                    image_size=self.image_size,
                    normalization=self.normalization,
                )
            )[0]
        )
        return make_liveness_result(
            probability,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
            threshold_policy=self.threshold_policy,
        )

    def predict_images(self, images: Iterable[np.ndarray]) -> np.ndarray:
        tensors = [
            normalize_rgb_image(
                image,
                image_size=self.image_size,
                normalization=self.normalization,
            )
            for image in images
        ]
        if not tensors:
            raise ValueError("at least one image is required")
        return self.predict_batch(np.stack(tensors, axis=0).astype(np.float32, copy=False))

    def predict_sequence(
        self,
        frames: Iterable[np.ndarray],
        *,
        aggregation: AggregationMethod = "mean",
    ) -> SequenceLivenessResult:
        frame_probabilities = self.predict_images(frames)
        probability = aggregate_probabilities(frame_probabilities, method=aggregation)
        decision = decision_from_probability(
            probability,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
        )
        return SequenceLivenessResult(
            live_probability=probability,
            threshold=self.accept_threshold,
            is_live=decision == "live",
            decision=decision,
            threshold_policy=self.threshold_policy,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
            frame_probabilities=tuple(float(score) for score in frame_probabilities),
            aggregation=aggregation,
        )


def _torch_load(path: Path, device: Any, torch: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        if os.environ.get(UNSAFE_TORCH_LOAD_ENV) == "1":
            return torch.load(path, map_location=device)
        raise RuntimeError(
            "this PyTorch version does not support safe weights_only loading; "
            f"set {UNSAFE_TORCH_LOAD_ENV}=1 only for trusted checkpoints"
        ) from exc
    except Exception:
        if os.environ.get(UNSAFE_TORCH_LOAD_ENV) == "1":
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=device)
        raise


def _clean_state_dict_keys(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned
