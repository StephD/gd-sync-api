"""Persistent RapidOCR (PP-OCRv6, pure ONNX Runtime) engine - cross-platform,
no Windows/WinRT dependency (unlike this project's original engine, see git
history), and no PaddlePaddle framework either: the official `paddleocr`
PyPI package was tried first and crashed on real inference (a oneDNN/PIR
"Unimplemented" error inside PaddlePaddle's inference engine itself, not
this code) - confirmed live, not worth debugging a third-party framework's
CPU-backend compatibility further when `rapidocr` (same PP-OCRv6 models,
plain ONNX Runtime, no PaddlePaddle) worked first try and scored better.

Confirmed live against 3 real player-popup captures at 3 different
resolutions/scales: **every field correct**, in one whole-image call, no
hand-tuned crop coordinates at all - including the level badge digit and a
CJK guild name, both of which needed extra work (or never fully worked) in
every previous approach tried this session (Windows OCR, PaddleOCR-JS
recognition-only, PaddleOCR-JS full detection). This is why player_paddle.py
and leaderboard.py classify boxes by content pattern + relative position
instead of cropping - the coordinates come from real detection every time,
so there's nothing to re-measure when a caller's image is a different shape
than whatever was tested last.

One process-wide instance, loaded once (~4s cold init) and reused - same
idea as the old OcrWorker's persistent-subprocess pattern, but simpler:
RapidOCR runs in-process, no subprocess/pipe protocol needed.
"""
from dataclasses import dataclass

from rapidocr import RapidOCR

_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


@dataclass(frozen=True)
class Box:
    text: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


def detect(image_path: str) -> list[Box]:
    """Every recognized text box on the image, y-sorted (top to bottom) -
    callers classify rows/fields from this instead of hand-tuned crops."""
    result = _get_engine()(image_path)
    if result.boxes is None or len(result.boxes) == 0:  # boxes is a numpy array - no bare truthiness
        return []
    out = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append(Box(text=text, confidence=score,
                        x=round(min(xs)), y=round(min(ys)),
                        w=round(max(xs) - min(xs)), h=round(max(ys) - min(ys))))
    out.sort(key=lambda b: b.y)
    return out
