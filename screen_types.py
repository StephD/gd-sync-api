"""Screen-type registry for /parse - lets one API tell a player Info popup
apart from a leaderboard, guild leaderboard, etc.

Old version (Windows OCR) did a tiny cheap-crop keyword check before the
expensive multi-scale field OCR, since those were two very different costs.
ocr_engine.detect() doesn't have that asymmetry - one whole-image call
already returns every box needed for BOTH deciding the type and parsing the
fields (confirmed live: player_paddle.py and leaderboard_paddle.py both
classify boxes from the exact same detect() call) - so detection is just
"does any box contain the keyword," checked against boxes already in hand,
no second OCR call.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ScreenType:
    name: str
    keyword: str     # case-insensitive substring that confirms this type
    parse: Callable    # (boxes, image_path) -> dict, the field reader

    def matches(self, boxes) -> bool:
        return any(self.keyword in b.text.lower() for b in boxes)


REGISTRY: list[ScreenType] = []  # checked in registration order - cheapest/most-common first


def register_type(t: ScreenType) -> None:
    REGISTRY.append(t)


def detect(boxes) -> ScreenType | None:
    for t in REGISTRY:
        if t.matches(boxes):
            return t
    return None


def get(name: str) -> ScreenType | None:
    return next((t for t in REGISTRY if t.name == name), None)
