# TinyLiveness Backend

Django API backend for the APCER 1% TinyLiveness model.

## Run Locally

```bash
cd tinyliveness
pip install -r backend/requirements.txt
python backend/manage.py runserver
```

For an application that only needs the Python runtime and bundled release model:

```bash
pip install "tinyliveness[onnx]"
```

## Routes

```text
GET  /health/
GET  /api/liveness/model/
POST /api/liveness/predict/
```

`POST /api/liveness/predict/` accepts:

```text
image: one JPEG/PNG/WEBP image
frames: repeated frame files for a short sequence
aggregation: mean, median, min, or p10
```

Default model:

```text
checkpoints/tinyliveness_main_apcer1_224.onnx
checkpoints/decision_policy_main_apcer1_224.json
```

Environment overrides:

```text
TINYLIVENESS_MODEL_PATH
TINYLIVENESS_POLICY_PATH
TINYLIVENESS_RATE_LIMIT_SECONDS
TINYLIVENESS_MAX_UPLOAD_BYTES
TINYLIVENESS_MAX_IMAGE_PIXELS
TINYLIVENESS_CORS_ORIGIN
TINYLIVENESS_EXPOSE_ERRORS
TINYLIVENESS_TRUST_PROXY_HEADERS
DJANGO_ALLOWED_HOSTS
DJANGO_CSRF_TRUSTED_ORIGINS
DJANGO_SECRET_KEY
DJANGO_DEBUG
DJANGO_SECURE_SSL_REDIRECT
DJANGO_TRUST_PROXY_SSL_HEADER
```

For production, set `DJANGO_DEBUG=0`, provide `DJANGO_SECRET_KEY`, restrict
`DJANGO_ALLOWED_HOSTS`, and use a specific `TINYLIVENESS_CORS_ORIGIN`.
