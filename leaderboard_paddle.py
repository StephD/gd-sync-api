"""Leaderboard parser for ocr_engine.detect()'s box list - covers every
layout seen so far (rank/name/atk/cards/stage; rank/name/stage-only "Normal
Stage NNN"; podium name+damage "90.24t") with one row-clustering + per-row
content classifier, instead of a separate hand-tuned parser per layout.

Row clustering anchors on NAME boxes, not a y-gap/window. Tried a "growing
window" (a box joins the row if its top is within the row's current bottom
edge + a tolerance) first - broke on a real sample (ark-lb.png, 506x735):
two adjacent rows' OWN internal vertical spread (57px, rank/name/stat/stage
lines stacked within one row) was bigger than the actual GAP between that
row and the next one (5px), so no fixed-fraction-of-height tolerance could
tell "still this row" from "next row" - the window kept growing and ate the
next row whole. Name boxes don't have that problem: they're one line each,
reliably present (even podium/medal rows, which lack a rank *number*, still
have a name), and stay well-separated from each other even when everything
else in the row is packed tight. So: find every name-shaped box first, then
assign every other box to whichever name is closest in y - independent
per-box, so one bad assignment can't cascade into merging two whole rows
the way the growing window did.

Per row: leftmost pure-digit box is rank (top-3 podium rows have a medal
icon instead of a digit - no box, so rank comes back None, not wrong).
Everything else classifies by pattern: "ago"/time -> dropped,
`Stage\\D*(\\d+)` or a bare number following a "...Stage" label -> stage,
`\\d+\\.\\d+[a-z]` -> dmg (guild-raid style), remaining digit-ish boxes
(icon-merge single-letter prefix stripped) sorted by x -> atk then cards.
"""
import re

RE_STAGE_INLINE = re.compile(r"stage\D*(\d+)", re.I)
RE_STAGE_LABEL = re.compile(r"stage$", re.I)
RE_DMG = re.compile(r"^\d+(\.\d+)?[a-zA-Z]$")
RE_TIME = re.compile(r"ago$|^\d+/\d+$", re.I)
RE_NUM = re.compile(r"^[A-Za-z]?(\d[\d,]*)$")  # strips a single icon-merge leading letter
ROW_KEYWORDS = {"ranking", "stage requirement", "player ranking", "rank rewards",
                 "leaderboard", "tap to close", "total damage", "guild damage"}
NAME_STOPWORDS = {"x", "not", "ranked", "tap", "close", "info"}


def _is_chrome(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ROW_KEYWORDS) or not t


def _is_name_like(text: str) -> bool:
    t = text.strip()
    # len >= 2 excludes a lone icon-merge leftover letter (confirmed live:
    # VoodooKid's flame icon detected as a standalone "M" box, separate
    # from its "3340" digit box - without this it wins the anchor slot by
    # simply existing, splitting VoodooKid's row into two)
    if len(t) < 2 or _is_chrome(t) or t.lower() in NAME_STOPWORDS:
        return False
    if RE_STAGE_LABEL.search(t) or t.lower().replace(" ", "") in ("normalstage", "stage"):
        return False
    if RE_STAGE_INLINE.search(t) or RE_TIME.search(t) or RE_NUM.match(t.replace(",", "")):
        return False
    return bool(re.search(r"[A-Za-z一-鿿가-힣]", t))


def _cluster_rows(boxes) -> list[list]:
    anchors = [b for b in boxes if _is_name_like(b.text)]
    if not anchors:
        return []
    anchor_ids = {id(a) for a in anchors}
    members: dict[int, list] = {id(a): [a] for a in anchors}
    for b in boxes:
        if id(b) in anchor_ids:
            continue
        nearest = min(anchors, key=lambda a: abs(b.y - a.y))
        members[id(nearest)].append(b)
    return [members[id(a)] for a in sorted(anchors, key=lambda a: a.y)]


def _parse_row(row: list) -> dict | None:
    row = sorted(row, key=lambda b: b.x)
    cand = [b for b in row if not _is_chrome(b.text) and not RE_TIME.search(b.text.strip())]
    if not cand:
        return None

    rank = None
    if cand[0].text.strip().isdigit() and len(cand[0].text.strip()) <= 3:
        rank = int(cand[0].text.strip())
        cand = cand[1:]

    name = None
    stage = None
    dmg = None
    nums: list[int] = []
    skip_next_num_as_stage = False
    for b in cand:
        t = b.text.strip()
        m = RE_STAGE_INLINE.search(t)
        if m:
            stage = int(m.group(1))
            continue
        if RE_STAGE_LABEL.search(t) or t.lower().replace(" ", "") in ("normalstage", "stage"):
            skip_next_num_as_stage = True
            continue
        if RE_DMG.match(t):
            dmg = t
            continue
        nm = RE_NUM.match(t.replace(",", ""))
        if nm:
            if skip_next_num_as_stage and stage is None:
                stage = int(nm.group(1))
                skip_next_num_as_stage = False
            else:
                nums.append(int(nm.group(1)))
            continue
        if re.search(r"[A-Za-z一-鿿가-힣]", t) and len(t) >= 2 and name is None and t.lower() not in NAME_STOPWORDS:
            name = t

    if name is None and rank is None and stage is None and dmg is None and not nums:
        return None  # nothing row-shaped survived - nav chrome slipped through _is_chrome

    return {
        "rank": rank,
        "name": name,
        "atk": nums[0] if len(nums) >= 1 else None,
        "cards": nums[1] if len(nums) >= 2 else None,
        "stage": stage,
        "dmg": dmg,
    }


def parse(boxes, image_path=None) -> dict:
    rows = [r for row in _cluster_rows(boxes) if (r := _parse_row(row)) is not None]
    return {"rows": rows}
