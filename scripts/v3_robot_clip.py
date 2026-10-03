"""Film v3 evidence shot: ask Runway for a video of the SO-101 itself doing the task, once.

Same cheap path as the human clips: gen4_image_turbo first frame (2 credits) + gen4_turbo 5 s (25 credits).
References: a MuJoCo render of the SO-101 (from its CAD meshes, @arm) and clip v01's first frame (@scene).
Hard cap for film v3: 30 credits, checked against the live balance before each call. One attempt only.
The key comes from the environment (filled from the Keychain by the caller), never written anywhere.
Every call is logged in data/runway-ledger.jsonl by rohub.runway.

usage: RUNWAY_API_KEY=... .venv/bin/python scripts/v3_robot_clip.py <arm-reference.png>
Writes data/runs/robot-clip/{r00.png,r00.mp4,r00.json}.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rohub import runway  # noqa: E402

V3_CAP = 30
OUT = REPO / "data" / "runs" / "robot-clip"
SCENE = REPO / "data" / "runs" / "put-the-red-block-in-the-bowl" / "clips" / "v01.png"
SEED = 31

IMAGE_PROMPT = (
    "A photorealistic photo in the style of @scene: the same light oak table, the same single red wooden block "
    "and the same white bowl, but no person and no hand. Instead, the small robot arm from @arm, an SO-101 "
    "open-source desktop arm made of 3D-printed plastic links with black servo motors at every joint and a "
    "two-finger gripper, is mounted on the table behind the block, its open gripper hovering just above the "
    "red block. The whole arm, the block and the bowl are fully in frame. Sharp focus, no text."
)
VIDEO_PROMPT = (
    "The robot arm lowers its open gripper onto the red block, closes the gripper on it, lifts it about 10 cm, "
    "carries it to the right and releases it into the bowl, then rises back to its starting pose. One continuous "
    "smooth motion. Static camera, locked off. The arm keeps the same rigid links and joints the whole time, the "
    "block stays one solid red cube and the bowl does not move."
)


def spend_check(start: int, cost: int) -> None:
    bal = runway.balance()
    if (start - bal) + cost > V3_CAP:
        raise SystemExit(
            f"v3 cap: would spend {start - bal + cost} > {V3_CAP} (balance {bal})"
        )


def main() -> None:
    arm = Path(sys.argv[1]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    start = runway.balance()
    print(f"[robot-clip] balance before: {start}")
    spend_check(start, 2 + 25)
    img = runway.generate(
        "text_to_image",
        {
            "model": "gen4_image_turbo",
            "promptText": IMAGE_PROMPT,
            "ratio": "1280:720",
            "seed": SEED,
            "referenceImages": [
                {"uri": runway.image_data_uri(arm), "tag": "arm"},
                {"uri": runway.image_data_uri(SCENE), "tag": "scene"},
            ],
        },
        ".png",
        "v3 robot-clip first frame",
    )
    shutil.copy(img, OUT / "r00.png")
    spend_check(start, 25)
    vid = runway.generate(
        "image_to_video",
        {
            "model": "gen4_turbo",
            "promptImage": runway.image_data_uri(img),
            "promptText": VIDEO_PROMPT,
            "ratio": "1280:720",
            "duration": 5,
            "seed": SEED,
        },
        ".mp4",
        "v3 robot-clip video",
    )
    shutil.copy(vid, OUT / "r00.mp4")
    end = runway.balance()
    # the ledger is the record of what was paid (a rerun hits the cache and pays nothing)
    paid = [e for e in runway.ledger() if e["label"].startswith("v3 robot-clip") and e["status"] == "SUCCEEDED"]
    rec = {
        "image_model": "gen4_image_turbo",
        "video_model": "gen4_turbo",
        "image_prompt": IMAGE_PROMPT,
        "video_prompt": VIDEO_PROMPT,
        "references": {
            "arm": str(arm.relative_to(REPO)),
            "scene": str(SCENE.relative_to(REPO)),
        },
        "seed": SEED,
        "balance_before": paid[0]["balance_before"],
        "balance_after": paid[-1]["balance_after"],
        "spent": sum(e["cost"] for e in paid),
        "balance_now": end,
    }
    (OUT / "r00.json").write_text(json.dumps(rec, indent=2))
    print(f"[robot-clip] done, spent {start - end}, balance {end}")


if __name__ == "__main__":
    main()
