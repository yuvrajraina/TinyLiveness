# TinyLiveness Runtime Constraints

Target contract:

```text
Input: aligned face image or short webcam frame sequence
Output: real/spoof score plus decision
Model size: under 20 MB FP32 ONNX
Runtime: Python + JavaScript + ONNX
```

## Input

Single frame:

```text
RGB face crop -> 224x224 -> ImageNet normalization -> NCHW float32
```

Short sequence:

```text
3-10 aligned RGB face crops -> per-frame live probabilities -> aggregate score
```

Supported aggregation methods in Python and JS:

```text
mean
median
min
p10
```

`mean` is the default. `min` and `p10` are stricter because one suspicious frame
can lower the sequence score.

## Output

The runtime supports a decision band:

```text
live_probability < reject_threshold      -> spoof
between reject and accept thresholds     -> manual_review
live_probability >= accept_threshold     -> live
```

Applications should load thresholds from the matching JSON policy file.

## Python ONNX

```python
from tinyliveness import OnnxLivenessDetector

detector = OnnxLivenessDetector(
    "checkpoints/tinyliveness_main_apcer1_224.onnx",
    thresholds_path="checkpoints/decision_policy_main_apcer1_224.json",
    normalization="imagenet",
    image_size=224,
)

single = detector.predict_image(aligned_face_rgb_224)
sequence = detector.predict_sequence(aligned_face_rgb_frames, aggregation="mean")
```

## JavaScript ONNX

See:

```text
runtime/js/tinyliveness.js
```

The JS runtime uses `onnxruntime-web` and accepts browser `ImageData` frames that
are already cropped/resized to `224x224`.
