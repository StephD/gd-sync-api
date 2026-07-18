"""One-off: time+print ocr_engine.detect()+leaderboard_paddle.parse() over
every LB/misc sample in "other test/" - no ground-truth manifest exists for
these, so output is for eyeballing against the source image, not auto-scored.
Run: py _bench_rapidocr_lb.py"""
import time
from pathlib import Path

import ocr_engine
import leaderboard_paddle

HERE = Path(__file__).parent
SAMPLES = sorted((HERE / "other test").glob("*.png")) + \
          sorted((HERE / "other test").glob("*.jpg"))

for path in SAMPLES:
    t0 = time.perf_counter()
    boxes = ocr_engine.detect(str(path))
    result = leaderboard_paddle.parse(boxes, str(path))
    dt = time.perf_counter() - t0
    print(f"\n=== {path.name}  ({dt*1000:.0f}ms, {len(boxes)} boxes) ===")
    for row in result["rows"]:
        print(" ", row)
    if not result["rows"]:
        print("  (no rows parsed - raw boxes:)")
        for b in boxes:
            print("   ", b.text)
