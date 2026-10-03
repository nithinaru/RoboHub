"""v8 feasibility probe: run real (scraped, licensed) footage through the unchanged RoboHub clip path
(MediaPipe extractor -> events -> Claude scene audit -> clip gates -> retarget -> MuJoCo replay gates) and print
which gates pass and why. Nothing is tuned for real footage; this measures how far the current gates are from it.

Footage lives in data/scraped/ (gitignored) and is never embedded in the film or site; only the per-gate verdicts
(numbers) are kept, in data/scraped/probe/<clip>/results/*.json.

    PYTHONPATH=src nice -n 19 taskpolicy -b .venv/bin/python scripts/scraped_probe.py data/scraped/*.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rohub.pipeline import process_clip
from rohub.scene import load


def main() -> None:
    scene = load(task="pick")
    for clip in map(Path, sys.argv[1:]):
        run_dir = clip.parent / "probe" / clip.stem
        for sub in ("pose", "results", "media"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        r = process_clip(
            clip, {"clip_id": clip.stem, "note": "scraped real footage"}, run_dir, scene
        )["res"]
        ev = r["events"]
        print(
            f"\n== {clip.name}: {'ACCEPTED' if r['accepted'] else 'REJECTED'} ({r['reason']})"
        )
        print(
            f"   frames {r['frames']} @ {r['fps']:.2f} fps; events ok={ev.get('ok')} why={ev.get('why')!r} grasp={ev.get('t_grasp')} "
            f"release={ev.get('t_release')}; audit={json.dumps({k: r['audit'].get(k) for k in ('available', 'red_blocks_first', 'bowls_first', 'hands_first', 'block_in_bowl_last', 'note')})}"
        )
        for g in r["gates"]:
            print(
                f"   [{ {True: 'PASS', False: 'FAIL', None: 'n/a '}[g.get('passed')] }] {g.get('id')} {g.get('name')}: {g.get('value')} "
                f"(limit {g.get('limit')}){'' if g.get('passed') else '  <- ' + str(g.get('why'))}"
            )


if __name__ == "__main__":
    main()
