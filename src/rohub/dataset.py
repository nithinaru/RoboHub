"""Stage 5: accepted episodes -> a LeRobot v3.0 dataset (lerobot 0.6.1, LeRobotDataset.create).

Keys, names and units match a real so101_follower in degrees mode (see units.py), so these episodes can sit
next to real teleop data: observation.state and action (6 motors), observation.images.front (the sim's front
camera, AV1 video), plus observation.environment_state (the cube's xyz in metres), which the state-based policy
reads. Each episode's origin (which Runway clip, original or re-anchored) goes to rohub_episodes.json.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np

from .retarget import replay
from .scene import JOINTS, RECORD_FPS, Scene
from .units import to_lerobot_units

IMG_H, IMG_W = 240, 320


def features(cameras: tuple[str, ...] = ("front",)) -> dict:
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    motors = {f"{j}.pos": float for j in JOINTS}
    feats = {
        **hw_to_dataset_features(motors, ACTION, True),
        **hw_to_dataset_features({**motors, **{c: (IMG_H, IMG_W, 3) for c in cameras}}, OBS_STR, True),
    }
    feats["observation.environment_state"] = {
        "dtype": "float32",
        "shape": (3,),
        "names": ["cube_x", "cube_y", "cube_z"],
    }
    return feats


def trash(path: Path) -> None:
    """Move an old output to the macOS Trash (never a permanent delete)."""
    dest = Path.home() / ".Trash" / f"{path.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(path), str(dest))


def write(
    scene: Scene,
    episodes: list[dict],
    root: Path,
    repo_id: str,
    task_text: str,
    on_episode=None,
    cameras: tuple[str, ...] = ("front",),
) -> dict:
    """episodes: [{demo, origin}], each already accepted by the gates (replay is deterministic).

    on_episode(i, n, origin, frames) is called after each episode is saved (live progress)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        trash(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=RECORD_FPS,
        features=features(cameras),
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
    )
    renderer = mujoco.Renderer(scene.model, IMG_H, IMG_W)
    meta, frames = [], 0
    t0 = time.time()

    def on_frame(state, action):
        imgs = {}
        for cam in cameras:
            renderer.update_scene(scene.data, camera=cam)
            imgs[f"observation.images.{cam}"] = renderer.render().copy()
        ds.add_frame(
            {
                "observation.state": to_lerobot_units(state),
                "action": to_lerobot_units(action),
                **imgs,
                "observation.environment_state": scene.cube_pos().astype(np.float32),
                "task": task_text,
            }
        )

    try:
        for i, ep in enumerate(episodes):
            r = replay(scene, ep["demo"], on_frame=on_frame)
            assert r.success, "an accepted episode failed on the recording pass"
            ds.save_episode()
            frames += len(r.states)
            meta.append({"episode_index": i, **ep["origin"], "frames": len(r.states)})
            if on_episode is not None:
                on_episode(i, len(episodes), ep["origin"], len(r.states))
    finally:
        ds.finalize()
    info = {
        "repo_id": repo_id,
        "root": str(root),
        "episodes": len(meta),
        "frames": frames,
        "fps": RECORD_FPS,
        "codebase_version": json.loads((root / "meta" / "info.json").read_text()).get(
            "codebase_version"
        ),
        "seconds": round(time.time() - t0, 1),
    }
    (root / "rohub_episodes.json").write_text(
        json.dumps({"info": info, "episodes": meta}, indent=1)
    )
    return info


def load_arrays(root: Path, repo_id: str) -> dict:
    """Read state/action/env/episode columns back through LeRobotDataset (proves the dataset loads)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    cols = hf[:]
    return {
        "state": np.asarray(cols["observation.state"], np.float32),
        "action": np.asarray(cols["action"], np.float32),
        "env": np.asarray(cols["observation.environment_state"], np.float32),
        "episode": np.asarray(cols["episode_index"]),
        "frame": np.asarray(cols["frame_index"]),
        "num_episodes": ds.num_episodes,
        "fps": ds.fps,
        "codebase_version": ds.meta.info.get("codebase_version"),
    }
