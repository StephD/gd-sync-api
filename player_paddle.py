"""Player Info popup parser for ocr_engine.detect()'s box list - no crop
coordinates, classifies each box by content pattern + relative position.

Confirmed live across 3 real captures at 3 different resolutions/scales
(gd-sync-api/_debug_captures/): after removing the title ("Info") and the
two left-column pure-digit boxes (level badge, atk - see _level_atk()), the
REMAINING boxes are always in this exact y-order: name, id-line(s), guild,
stage-line. id wraps to 1 or 2 boxes depending on id length/font, so instead
of guessing which, everything between name and guild is joined as the id.

Stage is found by content (regex), not by "last box in the sequence" - a 4th
real capture's image happened to crop in just enough of the gear grid below
the info card to pick up a stray "T7" tier-badge box after the real stage
line, and a purely positional "last item is stage" broke immediately (guild
got the stage line's text, stage came back null). Searching for the
Stage-matching box and treating guild as "whatever's immediately before it"
survives any amount of trailing noise the same way.
"""
import colorsys
import re

from PIL import Image

RE_ID_START = re.compile(r"^id[.:]", re.I)
RE_STAGE = re.compile(r"stage\D*(\d+)", re.I)
RE_DIGITS = re.compile(r"\d+")
KEYWORDS = {"info", "guardian", "ranger", "tap to close", "close", "x"}


def _level_atk(boxes, below_y: int | None) -> tuple[int | None, int | None, set[int]]:
    # Level (badge ring, upper card) and atk (lower card) are the only
    # digit-ish boxes in the left avatar column - level's y is always less
    # than atk's on every real sample seen, so sorting separates them
    # without needing to know either's absolute position. Extracting the
    # digit run (not requiring the WHOLE box text to be digits) matters:
    # confirmed live, the level badge ring occasionally reads a trailing
    # junk character into the box ("117)") - a bare .isdigit() check
    # silently drops the box entirely, and losing level then cascades into
    # the OTHER box (really atk) getting misassigned as level instead.
    #
    # below_y (the stage line's own y, from parse()) bounds the search to
    # "above the stage line" - confirmed live, a capture that includes even
    # a sliver of the gear grid below the info card produces a tier-badge
    # box like "T7" in the SAME x<150 column, which is exactly one digit
    # run just like a real stat and would otherwise outrank the real atk
    # value as "whichever digit box has the highest y."
    candidates = []
    for b in boxes:
        if b.x >= 150 or (below_y is not None and b.y >= below_y):
            continue
        runs = RE_DIGITS.findall(b.text.strip())
        if len(runs) == 1:  # exactly one digit run - junk chars elsewhere are fine, a second number isn't
            candidates.append((b, int(runs[0])))
    candidates.sort(key=lambda pair: pair[0].y)
    level = candidates[0][1] if candidates else None
    atk = candidates[-1][1] if len(candidates) >= 2 else None
    used = {id(b) for b, _ in candidates[:2]}
    return level, atk, used


def _clean_id(text: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]{2,}", RE_ID_START.sub("", text, count=1))
    pid = "".join(toks)
    return pid if len(pid) >= 12 else ""


def _detect_tier(image_path, name_box) -> str | None:
    # Name-text color encodes account tier - not OCR'd, sampled directly.
    # Same hue thresholds as the old gdocr.detect_tier() (svip ~33-40 gold,
    # vip ~178-180 cyan, normal ~192 blue) - only the sample box changed,
    # since it's now the ACTUAL detected name box, not a hand-measured
    # offset that could drift on a differently-scaled image.
    with Image.open(image_path) as im:
        px = im.convert("RGB").crop((name_box.x, name_box.y,
                                      name_box.x + name_box.w, name_box.y + name_box.h)).getdata()
    hues = []
    for r, g, b in px:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v > 0.6 and s > 0.15:
            hues.append(h * 360)
    if not hues:
        return None
    hues.sort()
    med = hues[len(hues) // 2]
    if med < 90:
        return "svip"
    if med < 186:
        return "vip"
    return "normal"


def parse(boxes, image_path) -> dict:
    stage_box = next((b for b in boxes if RE_STAGE.search(b.text)), None)
    level, atk, used_ids = _level_atk(boxes, stage_box.y if stage_box else None)

    seq = [b for b in boxes
           if id(b) not in used_ids
           and b.text.strip().lower() not in KEYWORDS
           and b.text.strip().lower() != "info"]

    rec = {"name": None, "level": level, "atk": atk, "guild_name": None,
           "normal_stage": None, "tier": None}
    if len(seq) < 3:
        return rec  # not enough of the popup read to say anything useful

    rec["name"] = seq[0].text.strip() or None
    rec["tier"] = _detect_tier(image_path, seq[0])

    stage_idx = next((i for i, b in enumerate(seq) if b is stage_box), None)
    if stage_idx is None or stage_idx < 2:
        # no stage line found, or found somewhere that leaves no room for a
        # guild box before it (shouldn't happen on a real popup, but don't
        # guess) - still return name/level/atk/tier, which don't depend on it
        return rec

    m = RE_STAGE.search(seq[stage_idx].text)
    rec["normal_stage"] = int(m.group(1))
    rec["guild_name"] = seq[stage_idx - 1].text.strip() or None

    id_text = "".join(b.text for b in seq[1:stage_idx - 1])
    pid = _clean_id(id_text)
    if pid:
        rec["_playerId_ocr"] = pid  # best-effort only, same caveat as the old parser's

    return rec
