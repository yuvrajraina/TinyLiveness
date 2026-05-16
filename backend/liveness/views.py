from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageOps, UnidentifiedImageError

from tinyliveness import OnnxLivenessDetector, get_default_model_path, get_default_policy_path

MODEL_NAME = "tinyliveness_main_apcer1_224.onnx"
POLICY_NAME = "decision_policy_main_apcer1_224.json"
MODEL_VARIANT = "apcer1"
MODEL_IMAGE_SIZE = int(os.environ.get("TINYLIVENESS_IMAGE_SIZE", "224"))
MODEL_NORMALIZATION = os.environ.get("TINYLIVENESS_NORMALIZATION", "imagenet")
RATE_LIMIT_SECONDS = float(os.environ.get("TINYLIVENESS_RATE_LIMIT_SECONDS", "1.0"))
MAX_SEQUENCE_FRAMES = int(os.environ.get("TINYLIVENESS_MAX_SEQUENCE_FRAMES", "10"))
MAX_UPLOAD_BYTES = int(os.environ.get("TINYLIVENESS_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("TINYLIVENESS_MAX_IMAGE_PIXELS", "12000000"))
TRUST_PROXY_HEADERS = os.environ.get("TINYLIVENESS_TRUST_PROXY_HEADERS", "0") == "1"
EXPOSE_ERRORS = os.environ.get("TINYLIVENESS_EXPOSE_ERRORS", "0") == "1"
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = REPO_ROOT / "checkpoints" / MODEL_NAME
LOCAL_POLICY_PATH = REPO_ROOT / "checkpoints" / POLICY_NAME
DEFAULT_MODEL_PATH = os.environ.get("TINYLIVENESS_MODEL_PATH", "")
DEFAULT_POLICY_PATH = os.environ.get("TINYLIVENESS_POLICY_PATH", "")

_last_request_by_client: dict[str, float] = {}
_rate_limit_lock = threading.Lock()
_model_lock = threading.Lock()
_default_detector: OnnxLivenessDetector | None = None
_default_model_path: Path | None = None
_default_policy_path: Path | None = None


def _with_cors(response: HttpResponse) -> HttpResponse:
    cors_origin = os.environ.get("TINYLIVENESS_CORS_ORIGIN", "*" if settings.DEBUG else "")
    if cors_origin:
        response["Access-Control-Allow-Origin"] = cors_origin
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Accept, Content-Type"
    return response


def _json(data: dict[str, Any], status: int = 200) -> JsonResponse:
    return _with_cors(JsonResponse(data, status=status))


def _error_payload(error: str, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": error}
    if detail and (settings.DEBUG or EXPOSE_ERRORS):
        payload["detail"] = detail
    return payload


def _client_key(request: HttpRequest) -> str:
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if TRUST_PROXY_HEADERS and forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def _check_rate_limit(request: HttpRequest) -> tuple[bool, float]:
    if RATE_LIMIT_SECONDS <= 0:
        return True, 0.0
    client = _client_key(request)
    now = time.monotonic()
    with _rate_limit_lock:
        last_request = _last_request_by_client.get(client, 0.0)
        elapsed = now - last_request
        if elapsed < RATE_LIMIT_SECONDS:
            return False, RATE_LIMIT_SECONDS - elapsed
        _last_request_by_client[client] = now
    return True, 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_path() -> Path:
    if DEFAULT_MODEL_PATH:
        path = Path(DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"TINYLIVENESS_MODEL_PATH does not exist: {path}")
        return path
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH
    return get_default_model_path()


def _resolve_policy_path() -> Path:
    if DEFAULT_POLICY_PATH:
        path = Path(DEFAULT_POLICY_PATH)
        if not path.exists():
            raise FileNotFoundError(f"TINYLIVENESS_POLICY_PATH does not exist: {path}")
        return path
    if LOCAL_POLICY_PATH.exists():
        return LOCAL_POLICY_PATH
    return get_default_policy_path()


def _get_default_detector() -> tuple[OnnxLivenessDetector, Path, Path]:
    global _default_detector, _default_model_path, _default_policy_path

    with _model_lock:
        if _default_detector is None or _default_model_path is None or _default_policy_path is None:
            _default_model_path = _resolve_model_path()
            _default_policy_path = _resolve_policy_path()
            _default_detector = OnnxLivenessDetector(
                _default_model_path,
                thresholds_path=_default_policy_path,
                normalization=MODEL_NORMALIZATION,  # type: ignore[arg-type]
                image_size=MODEL_IMAGE_SIZE,
            )
        return _default_detector, _default_model_path, _default_policy_path


def _uploaded_frames(request: HttpRequest) -> list[Any]:
    allowed_keys = {"image", "frame", "frames"}
    extra_keys = set(request.FILES) - allowed_keys
    if extra_keys:
        raise ValueError(f"unexpected file field(s): {', '.join(sorted(extra_keys))}")

    image_files = request.FILES.getlist("image")
    frame_files = [*request.FILES.getlist("frame"), *request.FILES.getlist("frames")]
    if image_files and frame_files:
        raise ValueError("send either one 'image' or a sequence of 'frames', not both")
    if len(image_files) > 1:
        raise ValueError("upload exactly one file for 'image'")

    files = image_files or frame_files
    if not files:
        raise ValueError("upload one face image as 'image' or frames as 'frames'")
    if len(files) > MAX_SEQUENCE_FRAMES:
        raise ValueError(f"sequence upload is limited to {MAX_SEQUENCE_FRAMES} frames")
    for uploaded_file in files:
        size = int(getattr(uploaded_file, "size", 0) or 0)
        if size <= 0:
            raise ValueError(f"empty upload: {uploaded_file.name}")
        if size > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES / (1024.0 * 1024.0)
            raise ValueError(f"upload exceeds {limit_mb:.1f} MB: {uploaded_file.name}")
        content_type = str(getattr(uploaded_file, "content_type", "") or "").lower()
        if content_type and content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise ValueError(f"unsupported content type for {uploaded_file.name}: {content_type}")
    return files


def _parse_aggregation(request: HttpRequest) -> str:
    aggregation = request.POST.get("aggregation", "mean").strip().lower()
    if aggregation not in {"mean", "median", "min", "p10"}:
        raise ValueError("'aggregation' must be one of: mean, median, min, p10")
    return aggregation


def _load_face_image(uploaded_file: Any) -> np.ndarray:
    try:
        with Image.open(uploaded_file) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"invalid image: {uploaded_file.name}") from exc

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = ImageOps.fit(
        image,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        method=resampling,
        centering=(0.5, 0.5),
    )
    return np.asarray(image, dtype=np.uint8)


@csrf_exempt
def predict(request: HttpRequest) -> HttpResponse:
    if request.method == "OPTIONS":
        return _with_cors(HttpResponse(status=204))
    if request.method != "POST":
        return _json({"error": "POST one image or a short frame sequence to this endpoint"}, status=405)

    allowed, retry_after = _check_rate_limit(request)
    if not allowed:
        return _json(
            {
                "error": "rate_limited",
                "retry_after_seconds": round(retry_after, 2),
                "detail": "The public demo API accepts one request at a time per client.",
            },
            status=429,
        )

    started = time.perf_counter()
    try:
        uploads = _uploaded_frames(request)
        aggregation = _parse_aggregation(request)
        frames = [_load_face_image(upload) for upload in uploads]
        detector, model_path, policy_path = _get_default_detector()

        if len(frames) == 1:
            result = detector.predict_image(frames[0])
            frame_probabilities: list[float] = [result.live_probability]
            result_type = "single_image"
        else:
            result = detector.predict_sequence(frames, aggregation=aggregation)  # type: ignore[arg-type]
            frame_probabilities = [float(value) for value in result.frame_probabilities]
            result_type = "sequence"

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _json(
            {
                "live_probability": round(float(result.live_probability), 6),
                "spoof_probability": round(float(result.spoof_probability), 6),
                "decision": result.decision,
                "is_live": result.is_live,
                "is_spoof": result.is_spoof,
                "needs_manual_review": result.needs_manual_review,
                "threshold": result.threshold,
                "reject_threshold": result.reject_threshold,
                "accept_threshold": result.accept_threshold,
                "threshold_policy": result.threshold_policy,
                "model_variant": MODEL_VARIANT,
                "model_name": model_path.name,
                "policy_name": policy_path.name,
                "result_type": result_type,
                "aggregation": aggregation if len(frames) > 1 else None,
                "frame_count": len(frames),
                "frame_probabilities": [round(value, 6) for value in frame_probabilities],
                "processing_ms": round(elapsed_ms, 2),
            }
        )
    except ValueError as exc:
        return _json({"error": str(exc)}, status=400)
    except Exception as exc:
        return _json(_error_payload("liveness_prediction_failed", str(exc)), status=500)


def model_info(request: HttpRequest) -> HttpResponse:
    if request.method == "OPTIONS":
        return _with_cors(HttpResponse(status=204))
    if request.method != "GET":
        return _json({"error": "GET this endpoint to inspect the liveness model"}, status=405)

    started = time.perf_counter()
    try:
        detector, model_path, policy_path = _get_default_detector()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _json(
            {
                "status": "ready",
                "model_name": model_path.name,
                "model_variant": MODEL_VARIANT,
                "model_type": "onnx",
                "policy_name": policy_path.name,
                "normalization": detector.normalization,
                "image_size": detector.image_size,
                "threshold_policy": detector.threshold_policy,
                "reject_threshold": detector.reject_threshold,
                "accept_threshold": detector.accept_threshold,
                "size_mb": round(model_path.stat().st_size / (1024.0 * 1024.0), 3),
                "sha256": _file_sha256(model_path),
                "load_ms": round(elapsed_ms, 2),
            }
        )
    except Exception as exc:
        return _json(
            {
                **_error_payload("model_unavailable", str(exc)),
                "status": "unavailable",
                "model_name": MODEL_NAME,
                "model_variant": MODEL_VARIANT,
                "policy_name": POLICY_NAME,
            },
            status=503,
        )
