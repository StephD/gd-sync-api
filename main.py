"""FastAPI wrapper around RapidOCR (PP-OCRv6, pure ONNX Runtime) for
parsing game-screen screenshots.

Run locally:  py main.py
Run in production (Render, etc):  gunicorn main:app -k uvicorn.workers.UvicornWorker
  (filename has to be a valid Python module name for either uvicorn's or
  gunicorn's `module:app` import syntax to work - "gd-sync-api.py" broke
  that, hence "main.py")
Then POST a screenshot to /parse, either:
  - multipart/form-data, fields "image" = the image file, "type" = optional
  - application/json, {"image": "<base64, data-URI prefix OK>", "type": "..."}
"type" is one of the names in screen_types.REGISTRY (GET /types lists them).
Omit it and the screen type is auto-detected from the same OCR pass used to
parse it - see screen_types.py. PNG or JPEG both fine (Pillow reads by
content, not extension).

Cross-platform (Linux/Render included) - this project's original engine
(Windows OCR via PowerShell + WinRT, see git history) only ran on Windows,
which is why it couldn't be deployed. RapidOCR runs in-process, no OS-level
dependency, no subprocess/pipe protocol to manage.

Each screen type has its own parser module - player_paddle.py,
leaderboard_paddle.py - registered in screen_types.py. Unlike the old
crop-coordinate approach, these classify OCR boxes by content pattern +
relative position instead of hand-measured pixel offsets, so a differently
scaled/cropped/padded image doesn't need new coordinates measured for it
(confirmed live across every real capture collected this session, at 3+
different resolutions, with zero per-resolution tuning needed).
"""
import asyncio
import base64
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import ocr_engine
import screen_types as st
import leaderboard_paddle
import player_paddle
import preprocess

st.register_type(st.ScreenType(name="player", keyword="info", parse=player_paddle.parse))
st.register_type(st.ScreenType(name="leaderboard", keyword="leaderboard", parse=leaderboard_paddle.parse))

# guild_leaderboard: not wired up yet - the one real sample seen so far
# (guild-raid-lb.png) is podium+damage-value, not the rank/atk/cards/stage
# layout leaderboard_paddle.py already handles well; its own parser needs
# more samples to resolve the portrait-section duplicate-name edge case
# (see leaderboard_paddle.py's docstring) before it's worth registering.


@asynccontextmanager
async def _lifespan(app: FastAPI):
    ocr_engine.warm()
    yield


app = FastAPI(title="gd-sync-api", lifespan=_lifespan)

# ponytail: wide-open CORS for a localhost-only dev tool your own site calls
# from the browser. Tighten allow_origins when/if this gets hosted publicly.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Crop coords/box-pattern heuristics everywhere in this API are tuned
# against specific sample screenshots - the single most common failure mode
# is a caller sending a differently-shaped image than whatever was tested,
# which reads wrong with no other clue what went wrong. Every /parse call
# gets saved (not just failures) so real traffic keeps building up a
# test/calibration set instead of only capturing the ones that already broke.
DEBUG_DIR = Path(__file__).parent / "_debug_captures"

# ponytail: RENDER is set automatically by Render, absent everywhere else -
# only downscale where the CPU is actually the bottleneck (see preprocess.downscale)
ON_RENDER = bool(os.environ.get("RENDER"))

# One worker, one CPU core to work with - running requests concurrently
# doesn't make them faster, it just means N images' worth of buffers alive
# at once. Confirmed live: 6 simultaneous calls OOM-killed the whole
# instance (512MB cap). Serializing costs nothing on a single-core box and
# means a burst of calls queues instead of crashing the process.
_parse_lock = asyncio.Lock()


def _save_debug(shot: Path, reason: str) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    dest = DEBUG_DIR / f"{int(time.time())}_{reason}.png"
    shutil.copy(shot, dest)
    print(f"[debug] saved screenshot -> {dest}", flush=True)


async def _read_image(request: Request) -> tuple[bytes, str | None]:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        b64 = body.get("image")
        if not b64:
            raise HTTPException(400, "JSON body needs an 'image' base64 field")
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64), body.get("type")
        except (ValueError, base64.binascii.Error):
            raise HTTPException(400, "'image' is not valid base64")
    form = await request.form()
    image = form.get("image")
    if image is None:
        raise HTTPException(400, "multipart form needs an 'image' file field")
    type_field = form.get("type")
    return await image.read(), (str(type_field) if type_field else None)


def _blank_count(rec: dict) -> int:
    if "rows" in rec:  # leaderboard-shaped record, not field-shaped
        return 0 if rec["rows"] else len(rec)
    return sum(1 for k, v in rec.items() if k != "type" and v in (None, "", []))


def _is_weak(rec: dict) -> bool:
    return _blank_count(rec) > max(1, (len(rec) - 1) // 2)


def _attempt(image_path: str, type_name: str | None):
    """One detect+parse pass against a given image - never raises, so the
    caller can run this against two candidate images (trimmed vs original)
    and compare instead of committing to whichever ran first."""
    t0 = time.perf_counter()
    boxes = ocr_engine.detect(image_path)
    print(f"[timing] ocr_engine.detect: {time.perf_counter() - t0:.2f}s, {len(boxes)} boxes", flush=True)
    if type_name:
        screen = st.get(type_name)
        if screen is None:
            return None, None, "bad_type"
    else:
        screen = st.detect(boxes)
        if screen is None:
            return None, None, "undetected"
    rec = screen.parse(boxes, image_path)
    rec["type"] = screen.name
    return rec, screen, None


@app.post("/parse")
async def parse(request: Request):
    t_start = time.perf_counter()
    data, type_name = await _read_image(request)
    print(f"[timing] read body: {time.perf_counter() - t_start:.2f}s, {len(data)} bytes", flush=True)
    async with _parse_lock:
        return await _parse_locked(data, type_name, t_start)


async def _parse_locked(data: bytes, type_name: str | None, t_start: float):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
        shot = Path(f.name)
    if ON_RENDER:
        preprocess.downscale(shot)
    trimmed = None
    try:
        # Black letterboxing varies unpredictably between real captures and
        # is pure noise for OCR (see preprocess.py) - try the trimmed image first.
        trimmed = preprocess.trim_black_borders(shot)
        rec, screen, err = _attempt(str(trimmed or shot), type_name)

        # Self-correction: trimming is a threshold heuristic, and a parse
        # that came back undetected or mostly-null on the trimmed image
        # might just mean the trim was wrong for this particular screen (or
        # ate into real dark UI content) - not that the popup isn't there.
        # One bounded retry against the untrimmed original, keep whichever
        # result is actually better; never loops beyond this single fallback.
        if trimmed is not None and err != "bad_type" and (rec is None or _is_weak(rec)):
            rec2, screen2, err2 = _attempt(str(shot), type_name)
            if rec2 is not None and (rec is None or _blank_count(rec2) < _blank_count(rec)):
                rec, screen, err = rec2, screen2, err2

        if err == "bad_type":
            known = [t.name for t in st.REGISTRY]
            raise HTTPException(400, f"Unknown type '{type_name}'. Known: {known}")
        if rec is None:
            _save_debug(shot, "undetected_type")
            raise HTTPException(422, "Could not identify screen type from image")

        _save_debug(shot, f"{screen.name}_weak" if _is_weak(rec) else f"{screen.name}_ok")
        print(f"[timing] total: {time.perf_counter() - t_start:.2f}s", flush=True)
        return rec
    finally:
        shot.unlink(missing_ok=True)
        if trimmed is not None:
            trimmed.unlink(missing_ok=True)


@app.get("/types")
def types():
    return [t.name for t in st.REGISTRY]


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
