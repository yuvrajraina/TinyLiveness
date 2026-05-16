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

Do not hardcode thresholds in applications. Load the JSON policy with the ONNX
model.

## Latest Metrics

Latest release metrics are from the corrected 400-image test slice:
200 live and 200 spoof images. The original 10,000-image test split was reduced
per request, with the remaining samples moved into train/dev. These metrics are
useful for the current repo release, but they are not a replacement for a larger
untouched or cross-domain production test.

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

## Install

After publishing the release to PyPI:

```bash
pip install "tinyliveness[onnx]"
```

Or install the GitHub release/tag directly with pip:

```bash
pip install "tinyliveness[onnx] @ git+https://github.com/yuvrajraina/TinyLiveness.git@v0.1.0"
```

For development from a local checkout:

```bash
cd tinyliveness
pip install -e ".[onnx]"
```

For training or evaluation scripts:

```bash
pip install -e ".[training,onnx]"
```

To create release artifacts for GitHub Releases:

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

Upload the generated `dist/tinyliveness-0.1.0-py3-none-any.whl` and
`dist/tinyliveness-0.1.0.tar.gz` files to the `v0.1.0` release.

After install, verify the bundled model and policy:

```bash
tinyliveness info
tinyliveness smoke
```

See `docs/RELEASE.md` for the complete GitHub/PyPI checklist.

## Backend API

The Django demo backend uses the APCER 1% model by default.

```bash
cd tinyliveness
pip install -r backend/requirements.txt
python backend/manage.py runserver
```

Routes:

```text
GET  /api/liveness/model/
POST /api/liveness/predict/
```

Upload one image as multipart field `image`. Short frame sequences are accepted
with repeated `frames` fields and optional `aggregation=mean|median|min|p10`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/liveness/predict/ ^
  -F image=@face.jpg
```

The route returns `live_probability`, `spoof_probability`, `decision`,
`reject_threshold`, `accept_threshold`, `threshold_policy`, and timing metadata.

## Netlify Frontend

The static website lives in `frontend/`. It is a multi-page landing site and
API-only model tester. The model does not run in the browser; the tester calls
the Django route above.

```bash
cd tinyliveness/frontend
npm install
npm run dev
```

For Netlify, use the repo root as the site base. The included `netlify.toml`
builds `frontend/dist`. Configure `VITE_TINYLIVENESS_API_URL` for the deployed
API endpoint, or let users enter the API URL on the test page.

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

## INT8 Status

INT8 is not shipped as a main model. Earlier INT8 exports showed score shift and
must be considered experimental until quantization evaluation passes:

```text
AUC drop <= 0.005
ACER increase <= 1 percentage point
APCER increase <= 1 percentage point
score correlation >= 0.99
```

Use FP32 ONNX for the current release.

## Limitations

TinyLiveness is small and fast, but passive single-frame RGB liveness is a
security signal, not a guarantee. Production readiness depends on held-out and
cross-domain APCER/BPCER/ACER, especially on the target cameras, screens, print
media, lighting, and environments.

Before making production claims:

- Validate on a larger untouched holdout.
- Validate on at least one unseen source/domain.
- Report subgroup APCER/BPCER/ACER by source, device, spoof type, lighting, and
  environment when metadata exists.
- Keep threshold tuning on train/dev only.
- Re-run the production gate after each data or model change.

Current launch assessment is documented in `docs/PRODUCTION_READINESS.md`: the
package is ready as a hardened beta/release candidate, while broad production
claims still need larger untouched and cross-domain validation.
