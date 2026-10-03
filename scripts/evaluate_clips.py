"""Re-evaluate every clip of a run with the current code (no Runway calls, no writes). Prints each verdict."""

import glob
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rohub import gates, retarget, scene  # noqa: E402
from rohub.audit import audit  # noqa: E402
from rohub.pipeline import POSE_CACHE, POSE_VERSION  # noqa: E402
from rohub.track import find_events, load_track, to_robot  # noqa: E402

run = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/runs/put-the-red-block-in-the-bowl"
sc = scene.load(task="pick")
for c in sorted(glob.glob(str(run / "clips/*.mp4"))):
    h = hashlib.sha256(open(c, "rb").read()).hexdigest()[:20]
    tr = load_track(POSE_CACHE / f"{h}_mediapipe_v{POSE_VERSION}.json")
    ev = find_events(tr)
    au = audit(Path(c))
    bb = au["bowl_bbox_first"]
    demo = to_robot(tr, ev, (bb[0] + bb[2]) / 2) if ev.ok else None
    rows = gates.clip_gates(tr, ev, au, demo)
    if demo is not None:
        rows += gates.robot_gates(retarget.replay(sc, demo))
    print(Path(c).stem, *gates.verdict(rows))
