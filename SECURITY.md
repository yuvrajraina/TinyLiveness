# Security Policy

TinyLiveness is a passive RGB liveness signal. Treat it as one layer in a
verification flow, not as a standalone fraud decision.

## Production Defaults

- Use the bundled FP32 ONNX EfficientNet-B0 224 APCER 1% model and matching JSON
  policy unless you have revalidated a newer artifact.
- Keep `DJANGO_DEBUG=0` in deployed backends.
- Set `DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS`, and a specific
  `TINYLIVENESS_CORS_ORIGIN` in production.
- Keep upload limits enabled with `TINYLIVENESS_MAX_UPLOAD_BYTES` and
  `TINYLIVENESS_MAX_IMAGE_PIXELS`.
- Do not expose backend exception details unless temporarily debugging a private
  deployment with `TINYLIVENESS_EXPOSE_ERRORS=1`.
- PyTorch checkpoint loading uses `weights_only=True` by default. Set
  `TINYLIVENESS_ALLOW_UNSAFE_TORCH_LOAD=1` only for checkpoints you fully trust.

## Model Risk

Single-frame passive liveness can be bypassed by presentation attacks outside
the validation domain. Before a high-risk launch, validate on target devices,
lighting, print attacks, replay/display attacks, and capture environments.

## Reporting Issues

Open a private security advisory or contact the maintainer before publishing a
working bypass, remote crash, or deployment hardening issue.
