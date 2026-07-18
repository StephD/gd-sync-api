"""Leaderboard parser for ocr_engine.detect()'s box list.

Two real layouts confirmed live (other test/lb ark.jpg, other test/lb
ember.png), different enough they need their own row schema instead of one
shared one:

- "ark": rank/name/atk/cards/stage per row, identified by "Stage NN"
  appearing on every row.
- "ember": rank/name/atk/chips/tier per row, identified by tier keywords
  ("Vanguard", more later) appearing on every row - PLUS "Ranking: 1-10
  Required Tier: Vanguard x50" section-header lines interleaved between
  rows, which are chrome, not player rows, and have to be stripped before
  row-clustering (their own "Vanguard x50" text would otherwise look
  exactly like a real player's tier and either spawn a fake row of its own
  or contaminate whichever real row is nearest in y).

A third style (podium/guild-raid, name+damage like "90.24t", no rank
number) was the original shape this parser was built for - kept as a
fallback for anything that isn't ark or ember, so it doesn't regress.

Row clustering anchors on NAME boxes, not a y-gap/window. Tried a "growing
window" (a box joins the row if its top is within the row's current bottom
edge + a tolerance) first - broke on a real sample (ark-lb.png, 506x735):
two adjacent rows' OWN internal vertical spread (57px, rank/name/stat/stage
lines stacked within one row) was bigger than the actual GAP between that
row and the next one (5px), so no fixed-fraction-of-height tolerance could
tell "still this row" from "next row" - the window kept growing and ate the
next row whole. Name boxes don't have that problem: they're one line each,
reliably present, and stay well-separated from each other even when
everything else in the row is packed tight. So: find every name-shaped box
first, then assign every other box to whichever name is closest in y -
independent per-box, so one bad assignment can't cascade into merging two
whole rows the way the growing window did.

Rank is expected sequential and increasing (confirmed live: real captures
never show 14-19-15, only gaps like 14-15-19) - one exception: the caller's
own rank, always the LAST row in the screenshot, sits far below the visible
top-N block and is exempt from that check.
"""
import re

RE_STAGE_INLINE = re.compile(r"stage\D*(\d+)", re.I)
RE_STAGE_LABEL = re.compile(r"stage$", re.I)
RE_DMG = re.compile(r"^\d+(\.\d+)?[a-zA-Z]$")
RE_TIME = re.compile(r"ago$|^\d+/\d+$", re.I)
RE_NUM = re.compile(r"^[A-Za-z]?(\d[\d,]*)$")  # strips a single icon-merge leading letter
RE_TIER_COUNT = re.compile(r"^x\d+$", re.I)  # e.g. "x89" - a tier's reward count, not atk/chips
RE_LB_ID = re.compile(r"leaderboard\s*id[:\s]*([A-Z0-9]{4,10})", re.I)
RE_HEADER_LINE = re.compile(r"ranking\s*:|required\s*tier", re.I)

ROW_KEYWORDS = {"ranking", "stage requirement", "player ranking", "rank rewards",
                 "leaderboard", "tap to close", "total damage", "guild damage",
                 "required tier"}
NAME_STOPWORDS = {"x", "not", "ranked", "tap", "close", "info"}
TIER_KEYWORDS = {"vanguard"}  # diamond/platinum/gold/silver/bronze later - same pattern


def _is_chrome(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in ROW_KEYWORDS) or not t


def _is_name_like(text: str) -> bool:
    t = text.strip()
    # len >= 2 excludes a lone icon-merge leftover letter (confirmed live:
    # VoodooKid's flame icon detected as a standalone "M" box, separate
    # from its "3340" digit box - without this it wins the anchor slot by
    # simply existing, splitting VoodooKid's row into two)
    if len(t) < 2 or _is_chrome(t) or t.lower() in NAME_STOPWORDS or t.lower() in TIER_KEYWORDS:
        return False
    if RE_STAGE_LABEL.search(t) or t.lower().replace(" ", "") in ("normalstage", "stage"):
        return False
    if RE_STAGE_INLINE.search(t) or RE_TIME.search(t) or RE_NUM.match(t.replace(",", "")):
        return False
    return bool(re.search(r"[A-Za-z一-鿿가-힣]", t))


def _strip_header_lines(boxes):
    """Drop ember's "Ranking: 1-10  Required Tier: Vanguard x50" section
    lines - same y-band as the label box that names them, so this only
    removes that one line, not a real player row above/below it."""
    headers = [b for b in boxes if RE_HEADER_LINE.search(b.text)]
    if not headers:
        return boxes
    return [b for b in boxes
            if not any(abs(b.y - h.y) <= max(h.h, b.h) * 1.5 for h in headers)]


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


def _find_leaderboard_id(boxes) -> str | None:
    for b in boxes:
        m = RE_LB_ID.search(b.text)
        if m:
            return m.group(1).upper()
    # OCR sometimes splits "Leaderboard ID:" and the code into two boxes -
    # take the nearest same-line box that's a bare alnum code
    label = next((b for b in boxes if re.search(r"leaderboard\s*id", b.text, re.I)), None)
    if label is None:
        return None
    for b in sorted((b for b in boxes if b is not label
                      and abs(b.y - label.y) <= max(label.h, 20)), key=lambda b: b.x):
        code = b.text.strip().upper()
        if re.fullmatch(r"[A-Z0-9]{4,10}", code):
            return code
    return None


def _detect_type(boxes, rows: list[list]) -> str | None:
    # own row (bottom-most, always present) is the most reliable single
    # place to check - confirmed live, the field that identifies the type
    # (Stage / a tier keyword) is guaranteed there even if noise elsewhere
    # in the popup were ever ambiguous
    if rows:
        own = " ".join(b.text.lower() for b in rows[-1])
        if "stage" in own:
            return "ark"
        if any(k in own for k in TIER_KEYWORDS):
            return "ember"
    everything = " ".join(b.text.lower() for b in boxes)
    if "stage" in everything:
        return "ark"
    if any(k in everything for k in TIER_KEYWORDS):
        return "ember"
    return None


def _leading_rank(cand: list) -> tuple[int | None, list]:
    if cand and cand[0].text.strip().isdigit() and len(cand[0].text.strip()) <= 3:
        return int(cand[0].text.strip()), cand[1:]
    return None, cand


def _parse_row_ark(row: list) -> dict | None:
    row = sorted(row, key=lambda b: b.x)
    cand = [b for b in row if not _is_chrome(b.text) and not RE_TIME.search(b.text.strip())]
    rank, cand = _leading_rank(cand)

    name = None
    stage = None
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

    if name is None or stage is None:
        return None  # name/stage are mandatory - rank's checked after leading-rank inference
    return {"rank": rank, "name": name,
            "atk": nums[0] if len(nums) >= 1 else None,
            "cards": nums[1] if len(nums) >= 2 else None,
            "stage": stage}


def _parse_row_ember(row: list) -> dict | None:
    row = sorted(row, key=lambda b: b.x)
    cand = [b for b in row if not _is_chrome(b.text) and not RE_TIME.search(b.text.strip())]
    rank, cand = _leading_rank(cand)

    name = None
    tier_name = None
    tier_count = None
    nums: list[int] = []
    for b in cand:
        t = b.text.strip()
        tl = t.lower()
        matched = next((k for k in TIER_KEYWORDS if k in tl), None)
        if matched:
            tier_name = matched.capitalize()
            continue
        if RE_TIER_COUNT.match(t):
            tier_count = t.lower()  # e.g. "x89" - the reward count, not atk/chips
            continue
        nm = RE_NUM.match(t.replace(",", ""))
        if nm:
            nums.append(int(nm.group(1)))
            continue
        if re.search(r"[A-Za-z一-鿿가-힣]", t) and len(t) >= 2 and name is None and t.lower() not in NAME_STOPWORDS:
            name = t

    if name is None or tier_name is None:
        return None  # name/tier are mandatory - rank's checked after leading-rank inference
    tier = f"{tier_name} {tier_count}" if tier_count else tier_name
    return {"rank": rank, "name": name,
            "atk": nums[0] if len(nums) >= 1 else None,
            "chips": nums[1] if len(nums) >= 2 else None,
            "tier": tier}


def _parse_row_generic(row: list) -> dict | None:
    """Fallback for anything that isn't ark or ember - the original
    unified parser (podium/guild-raid style: name + damage like "90.24t",
    no rank number). Kept so those samples don't regress."""
    row = sorted(row, key=lambda b: b.x)
    cand = [b for b in row if not _is_chrome(b.text) and not RE_TIME.search(b.text.strip())]
    rank, cand = _leading_rank(cand)

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
        return None
    return {"rank": rank, "name": name,
            "atk": nums[0] if len(nums) >= 1 else None,
            "cards": nums[1] if len(nums) >= 2 else None,
            "stage": stage, "dmg": dmg}


def _infer_leading_ranks(rows: list[dict]) -> list[dict]:
    """Top-3 rows show a medal badge instead of a plain digit - RapidOCR
    doesn't detect a text box there at all (confirmed live: no box where
    "1"/"2" should be, next to XIANG_PING/MANGO_MUSA). Those badges are
    always exactly rank 1/2/3 in order, so infer sequential rank for the
    leading run of rank-less rows instead of losing real player data over
    a missing icon-digit."""
    for i, r in enumerate(rows):
        if r["rank"] is not None:
            break
        r["rank"] = i + 1
    return rows


def _enforce_rank_order(rows: list[dict]) -> list[dict]:
    """Rank is mandatory and expected strictly increasing - a row that
    breaks that is an OCR misread, and there's no safe way to guess the
    right value, so drop the row rather than report a wrong rank. The last
    row (the caller's own rank) is exempt - it's real, just far below the
    visible top-N block."""
    if not rows:
        return rows
    fixed = []
    last_good = 0
    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            fixed.append(r)
            continue
        if r["rank"] is None or r["rank"] <= last_good:
            continue
        last_good = r["rank"]
        fixed.append(r)
    return fixed


def parse(boxes, image_path=None) -> dict:
    boxes = _strip_header_lines(boxes)
    rows_raw = _cluster_rows(boxes)
    lb_type = _detect_type(boxes, rows_raw)
    lb_id = _find_leaderboard_id(boxes)

    row_parser = {"ark": _parse_row_ark, "ember": _parse_row_ember}.get(lb_type, _parse_row_generic)
    rows = [r for row in rows_raw if (r := row_parser(row)) is not None]
    if lb_type in ("ark", "ember"):
        rows = _infer_leading_ranks(rows)
        rows = _enforce_rank_order(rows)
        rows = [r for r in rows if r["rank"] is not None]  # mandatory, after inference attempt

    return {"leaderboard_type": lb_type, "leaderboard_id": lb_id, "rows": rows}
