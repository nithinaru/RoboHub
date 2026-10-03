"""Web copy of the SO-101 model for the in-browser MuJoCo preview (site-v2/preview.js).

The vendored MJCF (vendor/so101) plus its STL meshes decimated by vertex clustering (2 mm cells, ~15 MB -> ~1.2 MB).
The preview only draws the arm, so coarse meshes are fine; the physics runs keep using vendor/so101.

Usage: uv run python scripts/web_meshes.py [out_dir]   (default deploy-v2/sim/so101)
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "vendor" / "so101"
CELL = 0.002  # metres
STL = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def decimate(src: Path, dst: Path, cell: float = CELL) -> int:
    b = src.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    v = (
        np.frombuffer(b[84 : 84 + n * 50], dtype=STL)["v"]
        .reshape(-1, 3)
        .astype(np.float64)
    )
    keys, inv = np.unique(
        np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True
    )
    inv = inv.reshape(-1)
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inv, v)
    pts = sums / np.bincount(inv, minlength=len(keys))[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    tri = pts[f[np.sort(keep)]].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    out = np.zeros(len(tri), dtype=STL)
    out["n"], out["v"] = nrm, tri
    data = b"\0" * 80 + struct.pack("<I", len(tri)) + out.tobytes()
    dst.write_bytes(data)
    return len(data)


def main(out: Path) -> None:
    (out / "assets").mkdir(parents=True, exist_ok=True)
    for f in ("so101_new_calib.xml", "LICENSE"):
        shutil.copy(SRC / f, out / f)
    total = sum(
        decimate(f, out / "assets" / f.name)
        for f in sorted((SRC / "assets").glob("*.stl"))
    )
    print(f"{out}: meshes {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "deploy-v2" / "sim" / "so101"
    )
