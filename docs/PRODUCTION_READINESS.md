# Production Readiness

## Current Status

Release package status: ready for hardened beta / release candidate.

Broad production launch status: not fully cleared yet.

Reason: the bundled EfficientNet-B0 224 APCER 1% ONNX model passes the beta
gate on the corrected 400-image test slice, but the production and
strong-production gates are blocked by one small subject subgroup. The current
test slice is also too small to prove generalization across target cameras,
screens, print media, lighting, and environments.

## What Is Ready

- Pip package metadata and bundled artifacts are configured.
- Default runtime uses the strongest included model:
  `tinyliveness_main_apcer1_224.onnx`.
- The JSON decision policy is bundled and loaded by default.
- ONNX smoke testing is available with `tinyliveness smoke`.
- Backend upload limits, CORS defaults, production secret handling, and error
  exposure have been hardened.
- The release checklist covers pytest, build, twine validation, Git tag, and
  PyPI upload.

## Required Before A High-Risk Production Launch

- Validate on a larger untouched holdout.
- Validate on at least one unseen source/domain.
- Report APCER, BPCER, ACER, and BPCER100 by source, device, spoof type,
  lighting, environment, and subject where metadata exists.
- Keep threshold selection on train/dev data only.
- Re-run `production_gate.py` and require at least the `production` gate to pass.
- Run a live API load test with target upload sizes and expected request rate.

## Recommended Runtime Policy

Use `decision` instead of only `is_live`. For security-sensitive flows, route
`manual_review` and low-confidence cases into retry, step-up verification, or
human review rather than treating them as hard pass/fail.
