"""Stage 6: a small behaviour-cloning policy (ACT-lite) trained on the LeRobot dataset, on the CPU, in minutes.

Observation: the 6 joint positions (LeRobot units) + the cube's xyz (observation.environment_state).
Output: a chunk of the next CHUNK joint targets. At run time chunks are blended with ACT's temporal ensembling
(exponential weights, oldest prediction weighted most). No images: this is a state-based policy, said plainly
on the site and in the README.

Evaluation: N seeds, each a fresh cube position drawn from the same region the dataset covers, a full physics
rollout in MuJoCo, success = the cube rests in the bowl after being lifted. The rate is reported with a
Wilson 95% interval.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from .pick_expert import in_bowl
from .scene import CUBE_HALF, GRIPPER_CLOSED, HOME, STEPS_PER_FRAME, Scene
from .units import from_lerobot_units

CHUNK = 20
HIDDEN = 256
ENSEMBLE_M = 0.05


class MLPChunk(nn.Module):
    def __init__(self, obs_dim: int = 9, act_dim: int = 6, chunk: int = CHUNK):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, chunk * act_dim),
        )

    def forward(self, x):
        return self.net(x).view(-1, self.chunk, self.act_dim)


def make_windows(arr: dict):
    obs = np.concatenate([arr["state"], arr["env"]], 1)
    act = arr["action"]
    ep = arr["episode"]
    X, Y = [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        o, a = obs[idx], act[idx]
        T = len(idx)
        for t in range(T):
            j = np.minimum(np.arange(t, t + CHUNK), T - 1)  # pad with the last action
            X.append(o[t])
            Y.append(a[j])
    return np.array(X, np.float32), np.array(Y, np.float32)


def train(
    arr: dict,
    out: Path,
    steps: int = 4000,
    seed: int = 0,
    log=print,
    on_step=None,
    every: int = 50,
) -> dict:
    """on_step(step, mean_l1_of_last_`every`_steps, seconds) is called every `every` steps (live progress)."""
    torch.manual_seed(seed)
    X, Y = make_windows(arr)
    stats = {
        "obs_mean": X.mean(0),
        "obs_std": X.std(0) + 1e-3,
        "act_mean": Y.reshape(-1, 6).mean(0),
        "act_std": Y.reshape(-1, 6).std(0) + 1e-3,
    }
    Xn = torch.tensor((X - stats["obs_mean"]) / stats["obs_std"])
    Yn = torch.tensor((Y - stats["act_mean"]) / stats["act_std"])
    model = MLPChunk()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    losses = []
    for step in range(steps):
        b = torch.randint(0, len(Xn), (256,), generator=g)
        x = Xn[b] + 0.02 * torch.randn(
            len(b), Xn.shape[1], generator=g
        )  # small state noise for robustness
        loss = nn.functional.l1_loss(model(x), Yn[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if on_step is not None and (step % every == every - 1 or step == steps - 1):
            on_step(step + 1, float(np.mean(losses[-every:])), time.time() - t0)
        if step % 1000 == 0 or step == steps - 1:
            log(f"[train] step {step} l1 {np.mean(losses[-100:]):.4f}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stats": {k: v.tolist() for k, v in stats.items()},
        },
        out / "policy.pt",
    )
    info = {
        "samples": int(len(X)),
        "steps": steps,
        "final_l1": round(float(np.mean(losses[-200:])), 4),
        "train_seconds": round(time.time() - t0, 1),
        "params": sum(p.numel() for p in model.parameters()),
        "chunk": CHUNK,
    }
    (out / "train.json").write_text(json.dumps(info, indent=1))
    return info


class Runner:
    def __init__(self, path: Path):
        ck = torch.load(path / "policy.pt", weights_only=False)
        self.model = MLPChunk()
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.s = {k: np.array(v, np.float32) for k, v in ck["stats"].items()}

    def chunk(self, state_lr: np.ndarray, env: np.ndarray) -> np.ndarray:
        o = (
            np.concatenate([state_lr, env]).astype(np.float32) - self.s["obs_mean"]
        ) / self.s["obs_std"]
        with torch.no_grad():
            y = self.model(torch.tensor(o)[None])[0].numpy()
        return y * self.s["act_std"] + self.s["act_mean"]


def rollout(
    scene: Scene, runner: Runner, cube_xy, max_s: float = 25.0, on_frame=None
) -> dict:
    from .units import to_lerobot_units

    m, d = scene.model, scene.data
    scene.reset(HOME, GRIPPER_CLOSED, cube_xy, 0.0)
    T = int(max_s * 30)
    preds = []  # (t0, chunk)
    lifted = False
    settled = 0
    for t in range(T):
        st = to_lerobot_units(scene.joint_pos())
        preds.append((t, runner.chunk(st, scene.cube_pos())))
        preds = [(t0, c) for t0, c in preds if t - t0 < CHUNK]
        acts = np.array([c[t - t0] for t0, c in preds])
        w = np.exp(-ENSEMBLE_M * np.arange(len(acts)))  # preds[0] is the oldest
        a = (acts * w[:, None]).sum(0) / w.sum()
        d.ctrl[:] = from_lerobot_units(a)
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(m, d)
        c = scene.cube_pos()
        lifted |= bool(c[2] > CUBE_HALF + 0.02)
        if on_frame is not None:
            on_frame(t)
        # stop early once the cube has rested in the bowl for 1 s after a lift
        settled = (
            settled + 1
            if (lifted and in_bowl(c) and np.linalg.norm(d.qvel[-6:-3]) < 1e-3)
            else 0
        )
        if settled > 30:
            break
    c = scene.cube_pos()
    return {
        "success": bool(lifted and in_bowl(c)),
        "lifted": lifted,
        "cube_end": c.round(4).tolist(),
        "frames": t + 1,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
