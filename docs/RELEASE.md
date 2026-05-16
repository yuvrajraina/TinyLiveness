# Release Checklist

This repo is prepared for a pip-installable GitHub/PyPI release.

## Local Verification

```bash
python -m pytest
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall "dist/tinyliveness-0.1.0-py3-none-any.whl[onnx]"
tinyliveness smoke
```

## GitHub Release

```bash
git add .
git commit -m "Prepare TinyLiveness v0.1.0 release"
git tag -a v0.1.0 -m "TinyLiveness v0.1.0"
git push origin main
git push origin v0.1.0
```

Attach these files to the GitHub release:

```text
dist/tinyliveness-0.1.0-py3-none-any.whl
dist/tinyliveness-0.1.0.tar.gz
```

Users can install the GitHub tag with:

```bash
pip install "tinyliveness[onnx] @ git+https://github.com/yuvrajraina/TinyLiveness.git@v0.1.0"
```

## PyPI Release

After the GitHub release is verified:

```bash
python -m twine upload dist/*
```

Users can then install from PyPI with:

```bash
pip install "tinyliveness[onnx]"
```

## Launch Gate

The included release model passes the beta gate on the corrected 400-image test
slice. Production and strong-production gates remain blocked by a small subgroup
failure, so use this as a hardened beta/release-candidate package until larger
untouched and cross-domain validation passes.
