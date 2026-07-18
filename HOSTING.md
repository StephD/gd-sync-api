# gd-sync-api: hosting

**2026-07-18 update:** Path B (below) is done. The OCR engine is now
RapidOCR (PP-OCRv6, pure ONNX Runtime) - cross-platform, no Windows
dependency, runs in-process. `gd-sync-api/` is fully self-contained (no
sibling-folder imports into `gd-sync-ocr/` anymore). Render and any other
Linux PaaS work now.

## Render setup

- **Root Directory:** `gd-sync-api`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

Notes:
- `main.py` (not `gd-sync-api.py`, which this project's start file used to be
  named) - `module:app` import syntax used by both uvicorn and gunicorn can't
  parse a hyphenated module name.
- RapidOCR's cold init (~4s) happens once per worker process at startup, not
  per-request. Default to 1-2 gunicorn workers on a small instance - each
  worker loads its own model set into memory.
- No GPU needed or used; CPU inference, ~1-2s per screenshot.
- First deploy downloads the PP-OCRv6 ONNX models (a few MB) into the
  container's local cache on first use - after that they're warm for the
  life of the instance, but a fresh deploy re-downloads them (Render's
  filesystem isn't persistent across deploys). If cold-start time on every
  deploy becomes a problem, look at bundling the models into the repo /
  Docker image instead of relying on RapidOCR's auto-download.

## What used to block this (obsolete, kept for context)

The original engine was `Windows.Media.Ocr` via a PowerShell bridge
(`gd-sync-ocr/ocr_worker.ps1`) - Windows-only, which is why every Linux PaaS
(Render, Railway, Fly.io) and every "free VM" option (Oracle/GCP free tiers
are Linux-only unless you pay for a Windows guest) was a dead end. The
"Cloudflare Tunnel to your own PC" workaround documented here previously is
no longer necessary - the API can just be deployed like a normal Python
service now.
