# TinyLiveness

TinyLiveness is a lightweight passive RGB face liveness / anti-spoofing
prototype:

```text
TinyLiveness = is this a real live face, not a photo/video/screen?
```

It is the liveness companion to TinyFaceMatch:

```text
TinyFaceMatch = are these two faces the same person?
TinyLiveness = is this a real live face, not a spoof?
```

This cleaned repo keeps the code, docs, runtimes, and the main FP32 ONNX release
variant. Local training data, held-out test data, generated reports, runs,
caches, and old experiment checkpoints are intentionally not included.

## Main Artifacts

The Git-facing release model is the EfficientNet-B0 224 APCER 1% policy.

| Variant | ONNX | Policy | Threshold | Use |
| --- | --- | --- | ---: | --- |
| APCER 1% | `checkpoints/tinyliveness_main_apcer1_224.onnx` | `checkpoints/decision_policy_main_apcer1_224.json` | 0.990000000 | Default API prediction model |

Runtime contract:

```text
Input: Nx3x224x224 RGB float32 aligned face crop
Normalization: ImageNet mean/std
Output: live_probability in [0, 1]
Decision: spoof | manual_review | live
Model size target: under 20 MB FP32 ONNX
Runtime: Python + JavaScript + ONNX
```

## Latest Metrics

Latest release metrics are reported on a corrected, balanced 400-image averaged test slice containing 200 live samples and 200 spoof samples. This slice was drawn from a larger 4,000-image live/spoof dataset and evaluated across 10 separate averaged windows.


| Variant | ROC AUC | PR AUC | Accuracy | APCER | BPCER | ACER | BPCER100 | False accepts | False rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APCER 1% release model | 0.999325 | 0.999332 | 98.25% | 1.00% | 2.50% | 1.75% | 3.00% | 2 / 200 spoof | 5 / 200 live |

Size and latency:

```text
FP32 ONNX size: 15.296 MB
Synthetic ONNX CPU latency, batch=1: 5.619 ms/image
Evaluation latency during PyTorch scoring: about 2.45-3.06 ms/image
```

Gate status on the corrected 400-image test slice:

```text
APCER 1% release model: beta passed; production/strong-production blocked by one small subject subgroup.
```

Important caveat: the APCER 1% threshold is the default API policy and is set to
0.99. Re-run evaluation on a larger untouched/cross-domain set before treating
these metrics as production proof.


## Python Usage

```python
from tinyliveness import create_default_onnx_detector

detector = create_default_onnx_detector()

result = detector.predict_image(aligned_face_rgb_224)
print(result.live_probability)
print(result.decision)  # spoof | manual_review | live
```

The pip package bundles the APCER 1% FP32 ONNX model and JSON policy. If you
need explicit file paths, use `get_default_model_path()` and
`get_default_policy_path()`.

## JavaScript Usage

```js
import { TinyLivenessSession } from "./runtime/js/tinyliveness.js";

const session = await TinyLivenessSession.create(
  "checkpoints/tinyliveness_main_apcer1_224.onnx",
  {
    thresholdsUrl: "checkpoints/decision_policy_main_apcer1_224.json",
    normalization: "imagenet",
    imageSize: 224
  }
);

const result = await session.predictImageData(imageData224);
console.log(result.liveProbability, result.decision);
```

## Training And Evaluation

Training and evaluation code is still included, but datasets are not. Use
aligned face crops in a local folder such as:

```text
data/liveness/
  train/live/
  train/spoof/
  val/live/
  val/spoof/
  test/live/
  test/spoof/
```

Generated `data/`, `runs/`, and `reports/` outputs should stay local and out of
Git. See `docs/DATA_AND_REAL_WORLD_TESTING.md` and `docs/USAGE.md` for the
reproducible workflow.


## Limitations

TinyLiveness is small and fast, but passive single-frame RGB liveness is a
security signal, not a guarantee. Production readiness depends on held-out and
cross-domain APCER/BPCER/ACER, especially on the target cameras, screens, print
media, lighting, and environments.

We are still undergoing the follwing steps:

- Validate on a larger untouched holdout.
- Report subgroup APCER/BPCER/ACER by source, device, spoof type, lighting, and
  environment when metadata exists.
