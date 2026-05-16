# TinyLiveness Model Card

## Model

Name: TinyLiveness Hardened EfficientNet-B0 224

Artifact included in this cleaned repo:

```text
FP32 ONNX, APCER 1%: checkpoints/tinyliveness_main_apcer1_224.onnx
Policy, APCER 1%: checkpoints/decision_policy_main_apcer1_224.json
Thresholds, APCER 1%: checkpoints/thresholds_main_apcer1_224.json
```

The training checkpoint, local datasets, generated reports, and experiment runs
are not included in this release-style repo.

Architecture: EfficientNet-B0 binary liveness classifier.

Input: aligned RGB face crop, `Nx3x224x224`, float32, ImageNet normalization.

Output: live probability. Runtime maps the score to `spoof`, `manual_review`,
or `live` with the JSON threshold policy.

## Intended Use

Passive RGB face liveness research and prototyping for CPU/mobile deployment.
It should be combined with face detection/alignment and should not be used as a
standalone identity or fraud decision.

## Data

Training and evaluation used a local partial CelebA-Spoof crop set:

```text
train: 13,409 live / 20,581 spoof
dev:   200 live / 200 spoof
test:  200 live / 200 spoof
```

This was the corrected split requested for the hardening pass. The original
10,000-image test split was reduced to 400 images, with the remaining original
test rows moved into train/dev. These metrics are therefore not comparable to an
untouched 10,000-image held-out test.

The dataset is not redistributed in this repo.

## Metrics

400-image test, APCER 1% release model:

```text
ROC AUC: 0.999325
PR AUC: 0.999332
Accuracy: 98.2500%
APCER: 1.0000%
BPCER: 2.5000%
ACER: 1.7500%
BPCER100: 3.0000%
Threshold: 0.990000000
```

The metrics above are the user-confirmed APCER 1 release metrics.

Current decision policy:

```text
reject_threshold: 0.990000000
accept_threshold: 0.990000000
manual_review band: disabled by default because reject_threshold == accept_threshold
```

Gate status on the corrected 400-image test:

```text
APCER 1% release model: beta passed; production and strong-production blocked by one small subject subgroup.
```

## Deployment

Use FP32 ONNX. INT8 is not approved for the current release. The pip package
bundles the APCER 1% ONNX model and decision policy.

```bash
pip install "tinyliveness[onnx] @ git+https://github.com/yuvrajraina/TinyLiveness.git@v0.1.0"
```

```text
FP32 ONNX size: 15.296 MB
Synthetic ONNX CPU latency: 5.619 ms/image
```

## Limitations

The model reaches the requested APCER/BPCER/manual-review operating region on a
small corrected 400-image test slice, but that is not enough evidence for a
broad production claim.

Known limitations:

- Passive single-frame RGB liveness is vulnerable to many presentation attacks.
- The corrected 400-image test is small and has limited domain coverage.
- Current metadata is sparse, limiting subgroup diagnosis.
- The APCER 1% threshold is currently set to 0.990000 and needs independent
  cross-domain validation before production use.
- More cross-domain data is needed before production deployment.

## Next Steps

- Add train/dev data from target devices, print attacks, display attacks,
  lighting conditions, and capture environments.
- Add temporal liveness features for short webcam sequences.
- Keep future held-out and cross-domain test splits untouched.
- Re-run the production gate after data expansion.
- Revisit INT8 only after quantization-aware or static calibration passes the
  safety criteria.
