"""Black-border trimming, run before every /parse.

Real captures have varying amounts of black letterboxing around the actual
popup (confirmed live: two samples with visibly different black padding).
That padding is pure noise for OCR detection, and worse, it changes the
image's effective width/height unpredictably between captures. Stripping it
first means detection always starts from the same kind of tightly-cropped
image the calibration samples were, instead of whatever margin a given
capture happened to have.
"""
from pathlib import Path

from PIL import Image

BLACK_THRESHOLD = 24  # 0-255 luminance; real UI content is never this dark


def trim_black_borders(path: Path) -> Path | None:
    """Returns a new file cropped to the bounding box of non-black content,
    or None if there was nothing worth trimming (already tight, or the
    whole image is black - not this function's problem to solve)."""
    im = Image.open(path)
    mask = im.convert("L").point(lambda p: 255 if p > BLACK_THRESHOLD else 0)
    bbox = mask.getbbox()
    if bbox is None or bbox == (0, 0, im.width, im.height):
        return None
    dest = path.with_name(path.stem + "_trimmed" + path.suffix)
    im.crop(bbox).save(dest)
    return dest
