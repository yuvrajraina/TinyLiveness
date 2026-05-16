# TinyLiveness Frontend

Netlify-ready multi-page landing site and API-only model tester.

## Pages

```text
index.html  - Model overview and MIT license
stats.html  - Metrics, size, latency, and caveats
usage.html  - Install, run, input preparation, and curl flow
api.html    - Endpoint details, response schema, and deployment settings
test.html   - Free API-only tester
```

The frontend does not run ONNX in the browser. The model runs on the
TinyLiveness backend, and the test page sends a single cropped frame to
`/api/liveness/predict/`.

## Local Dev

Start the backend first:

```bash
cd tinyliveness
python backend/manage.py runserver
```

Then run the frontend:

```bash
cd tinyliveness/frontend
npm install
npm run dev
```

Set `VITE_TINYLIVENESS_API_URL` at build time if your hosted API is not the
default local endpoint.

## Netlify

Use the `netlify.toml` at the TinyLiveness root. It publishes `frontend/dist`.
The hosted tester requires a deployed backend URL with CORS enabled for the
Netlify domain.
