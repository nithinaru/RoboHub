"""Unique frames per second of a video: how many frames actually change, second by second.

A frame counts as a repeat when it differs from the previous frame by less than THRESH (mean absolute
difference of a 320x180 greyscale copy, 0-255 scale). Encoder noise on a duplicated frame sits well
under 0.05; a real change (even a slow push) sits above it.

usage: python scripts/fps_probe.py <video> [sections "0,4.2,16.3,..."] [--per-second]
"""

import subprocess
import sys

import numpy as np

W, H = 320, 180
THRESH = 0.08


def diffs(path: str) -> tuple[np.ndarray, float]:
    fps = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    num, den = fps.split("/")
    rate = float(num) / float(den)
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-vf",
            f"scale={W}:{H}:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[255.0], d]), rate


def main() -> None:
    path = sys.argv[1]
    d, rate = diffs(path)
    changed = d > THRESH
    n = len(d)
    dur = n / rate
    print(
        f"{path}: {n} frames, container {rate:.2f} fps, {dur:.1f} s, unique {changed.sum() / dur:.1f} fps overall"
    )
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    if args:
        cuts = [float(x) for x in args[0].split(",")] + [dur]
        for a, b in zip(cuts, cuts[1:]):
            i, j = int(a * rate), int(b * rate)
            seg = changed[i:j]
            # median over whole seconds shows the steady rate, not diluted by deliberate holds
            secs = [
                changed[k : k + int(rate)].sum()
                for k in range(i, j - int(rate) + 1, int(rate))
            ]
            med = float(np.median(secs)) if secs else float("nan")
            print(
                f"  {a:6.1f}-{b:6.1f}s  unique {seg.sum() / max(b - a, 1e-6):5.1f} fps  (median per-second {med:4.0f}, max {max(secs) if secs else 0})"
            )
    if "--per-second" in sys.argv:
        print(
            " ".join(
                str(int(changed[k : k + int(rate)].sum()))
                for k in range(0, n, int(rate))
            )
        )


if __name__ == "__main__":
    main()
