# TinyLiveness Main Model

TinyLiveness now ships one Git-facing FP32 ONNX release model: the APCER 1%
EfficientNet-B0 224 policy.

Runtime contract:

```text
Input: Nx3x224x224 RGB float32 face crop
Normalization: ImageNet mean/std
Output: live_probability in [0, 1]
Decision rule: live if live_probability >= accept_threshold
```

## APCER 1 Release Model

```text
ONNX: checkpoints/tinyliveness_main_apcer1_224.onnx
Policy: checkpoints/decision_policy_main_apcer1_224.json
Thresholds: checkpoints/thresholds_main_apcer1_224.json
Threshold: 0.990000000
```

Corrected 400-image test result:

```text
ROC AUC: 0.999325
PR AUC: 0.999332
Accuracy: 98.25%
APCER: 1.00%
BPCER: 2.50%
ACER: 1.75%
BPCER100: 3.00%
False accepts: 2 / 200 spoof
False rejects: 5 / 200 live
```

The repo keeps only the APCER 1 model and policy.

Gate status on the corrected 400-image test:

```text
beta: passed
production: failed because subject=5030 has 1 false accept out of 5 samples
strong-production: failed because subject=5030 has 1 false accept out of 5 samples
```

## Important Caveat

The APCER 1% threshold is set to 0.990000000. Treat it as a high-security release
candidate, not proof of general production readiness. Before making production
claims, validate on a larger untouched and cross-domain test set.
