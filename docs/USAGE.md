# TinyLiveness Usage

## Install From GitHub Release

After publishing the release to PyPI:

```bash
pip install "tinyliveness[onnx]"
```

Or install the release tag directly with pip:

```bash
pip install "tinyliveness[onnx] @ git+https://github.com/yuvrajraina/TinyLiveness.git@v0.1.0"
```

For local development from a repo checkout:

```bash
pip install -e ".[onnx]"
```

## Main Release Model

Use the included FP32 ONNX release model:

```text
checkpoints/tinyliveness_main_apcer1_224.onnx
```

The backend API defaults to the APCER 1% variant:

```text
checkpoints/tinyliveness_main_apcer1_224.onnx
checkpoints/decision_policy_main_apcer1_224.json
```

Load the matching policy JSON instead of hardcoding thresholds:

```text
checkpoints/decision_policy_main_apcer1_224.json
```

## Python Runtime

```python
from tinyliveness import create_default_onnx_detector

detector = create_default_onnx_detector()

result = detector.predict_image(aligned_face_rgb_224)
print(result.live_probability, result.decision)
```

The pip package includes the APCER 1% ONNX model and policy. Use
`get_default_model_path()` and `get_default_policy_path()` if your integration
needs explicit filesystem paths.

Verify an installation with:

```bash
tinyliveness smoke
```

## JavaScript Runtime

```js
import { TinyLivenessSession } from "../runtime/js/tinyliveness.js";

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

## Training From A Folder Dataset

Datasets are not included in the cleaned repo. For local experiments, place
aligned face crops in:

```text
data/liveness/
  train/live/
  train/spoof/
  val/live/
  val/spoof/
  test/live/
  test/spoof/
```

Then run:

```bash
python training/train.py --data-dir data/liveness --image-size 224
```

## Training From CSV Manifests

Pass the CSV as the split path by setting `--data-dir` to the parent and using
CSV names for `--train-split` and `--val-split`:

```bash
python training/train.py ^
  --data-dir data/liveness ^
  --train-split train.csv ^
  --val-split val.csv ^
  --image-size 224
```

CSV format:

```csv
path,label,split,source,subject,attack_type,device,lighting,environment
train/live/a.jpg,live,train,celeba_spoof,001,,,,
train/spoof/b.jpg,spoof,train,celeba_spoof,001,print,,,
```

## Threshold Selection

For security-sensitive deployments, calibrate thresholds on train/dev or
validation data only. Do not tune thresholds on held-out test data.

Lower APCER means fewer spoof attacks accepted as live. That usually increases
BPCER, so expect more real users to retry or enter manual review.

## Deployment Flow

```text
camera frame
  -> face detection
  -> face alignment/crop to 224x224 RGB
  -> TinyLiveness live probability
  -> manual review or step-up if uncertain
  -> TinyFaceMatch identity verification, if live
```

## Real-World Evaluation

```bash
python training/evaluate_real_world.py ^
  --data-dir data/liveness ^
  --checkpoint checkpoints/best_tinyliveness.pt ^
  --calibration-split val ^
  --eval-split test ^
  --group-by source,attack_type,device,lighting,environment
```

The script writes local reports under `reports/`. Generated data and reports
should stay out of Git.
