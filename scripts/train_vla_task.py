"""Train SmolVLA on a LeRobot dataset on a rented RunPod GPU, on the fly, and stream progress.

usage: train_vla.py DATASET_ROOT STEPS [--run-dir RUN] [--cameras front] [--no-eval]

Prints machine-readable lines the RoboHub app shows live:
  [gpu] ...            pod lifecycle
  [train] step 1200/3000 0.157 s/step cost $0.07
  [train] done checkpoint=<local path> seconds=<n>
  [eval] 41/50
  [tiles] <dir>
  [cost] total $<x> gpu_seconds=<n>
The pod is always terminated at the end, even on failure. The RunPod key comes from the Keychain (RUNPOD_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from runpod import gql  # noqa: E402

REPLAY = Path(__file__).resolve().parents[1]
HERE = REPLAY / "gpu"
KEY = Path(os.environ.get("ROBOHUB_SSH_KEY", str(Path.home() / ".ssh/vultr_robohub")))  # its .pub is registered with RunPod
GPUS = [
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def deploy() -> dict:
    pub = KEY.with_suffix(".pub").read_text().strip()
    q = "mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) { id costPerHr machine { gpuDisplayName } } }"
    for gpu in GPUS:
        inp = {
            "cloudType": "ALL",
            "gpuCount": 1,
            "gpuTypeId": gpu,
            "name": "robohub-onthefly",
            "minVcpuCount": 16,
            "imageName": IMAGE,
            "containerDiskInGb": 60,
            "volumeInGb": 0,
            "ports": "22/tcp",
            "startSsh": True,
            "env": [{"key": "PUBLIC_KEY", "value": pub}],
        }
        d = (gql(q, {"in": inp}).get("data") or {}).get("podFindAndDeployOnDemand")
        if d:
            say(
                "gpu",
                f"rented {d['machine']['gpuDisplayName']} at ${d['costPerHr']}/h (pod {d['id']})",
            )
            return d
        say("gpu", f"{gpu}: none available")
    raise SystemExit("[gpu] no GPU available on RunPod")


def ssh_addr(pod_id: str) -> tuple[str, int]:
    for _ in range(60):
        rt = gql(
            '{ pod(input:{podId:"%s"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }'
            % pod_id
        )["data"]["pod"]["runtime"]
        ports = [
            p
            for p in ((rt or {}).get("ports") or [])
            if p["privatePort"] == 22 and p["isIpPublic"]
        ]
        if ports:
            return ports[0]["ip"], ports[0]["publicPort"]
        time.sleep(5)
    raise SystemExit("[gpu] pod never exposed SSH")


class Pod:
    def __init__(self, ip: str, port: int):
        self.ip, self.port = ip, port
        self.ssh = [
            "ssh",
            "-i",
            str(KEY),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ConnectTimeout=20",
            "-p",
            str(port),
            f"root@{ip}",
        ]

    def run(self, cmd: str, check: bool = True) -> str:
        for attempt in range(4):
            r = subprocess.run(self.ssh + [cmd], capture_output=True, text=True)
            if (
                r.returncode != 255
            ):  # 255 = ssh transport error (RunPod proxies reset now and then)
                break
            time.sleep(3)
        if check and r.returncode != 0:
            raise RuntimeError(
                f"remote failed ({r.returncode}): {cmd[:80]}\n{r.stderr[-800:]}"
            )
        return r.stdout

    def rsync(self, src: str, dst: str) -> None:
        e = f"ssh -i {KEY} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -p {self.port}"
        for _ in range(15):
            if (
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--partial",
                        "--inplace",
                        "--timeout=30",
                        "-e",
                        e,
                        src,
                        dst,
                    ],
                    capture_output=True,
                ).returncode
                == 0
            ):
                return
            time.sleep(3)
        raise RuntimeError(f"rsync failed: {src} -> {dst}")

    def push(self, local: Path, remote: str) -> None:
        self.rsync(str(local), f"root@{self.ip}:{remote}")

    def pull(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.rsync(f"root@{self.ip}:{remote}", str(local))


def bundle(dataset: Path, run_dir: Path | None) -> Path:
    """One tar (resumable upload) with the dataset, the replay code and the eval run dir; no macOS ._ files."""
    tmp = Path(tempfile.mkdtemp()) / "bundle.tar"

    def skip(ti):  # drop AppleDouble files and caches
        name = Path(ti.name).name
        return None if name.startswith("._") or name == "__pycache__" else ti

    with tarfile.open(tmp, "w") as t:
        t.add(dataset, arcname="data/train", filter=skip)
        for d in ("src", "scripts", "vendor"):
            t.add(REPLAY / d, arcname=d, filter=skip)
        if run_dir:
            for f in (
                "train.json",
                "data.json",
                "task.txt",
                "pose",
                "results",
                "policy",
            ):
                p = run_dir / f
                if p.exists():
                    t.add(p, arcname=f"data/run/{f}", filter=skip)
    return tmp


SETUP = """set -e
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync libegl1 libgl1 libglib2.0-0 ffmpeg >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
cd /work && uv venv -q .venv --python 3.12
uv pip install -q --python .venv/bin/python "lerobot[dataset,smolvla]==0.6.1" mujoco==3.14.0
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
if [ "$drv" -lt 580 ]; then uv pip install -q --python .venv/bin/python --reinstall "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match; fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; [s(r) for r in ['lerobot/smolvla_base','HuggingFaceTB/SmolVLM2-500M-Video-Instruct']]"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("steps", type=int)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=REPLAY / "data/web-runs/put-the-red-block-in-the-bowl",
    )
    ap.add_argument("--cameras", default="front")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--task", required=True, choices=["push", "stack", "tower"])
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="local dir for the checkpoint, eval and tiles",
    )
    a = ap.parse_args()
    ds = a.dataset.resolve()
    assert (ds / "meta" / "info.json").exists(), f"not a LeRobot dataset: {ds}"
    info = json.loads((ds / "meta/info.json").read_text())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = (a.out or HERE / "runs" / f"vla-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    say(
        "gpu",
        f"dataset {info.get('total_episodes')} episodes, {info.get('total_frames')} frames; {a.steps} steps",
    )

    t0 = time.time()
    pod = deploy()
    rate = float(pod["costPerHr"])
    cost = lambda: rate * (time.time() - t0) / 3600  # noqa: E731
    try:
        ip, port = ssh_addr(pod["id"])
        p = Pod(ip, port)
        for _ in range(20):  # sshd comes up a little after the port is published
            try:
                p.run("mkdir -p /work")
                break
            except RuntimeError:
                time.sleep(5)
        say("gpu", f"pod up in {time.time() - t0:.0f} s; installing LeRobot + SmolVLA")
        tar = bundle(ds, None)
        p.run(
            "apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync >/dev/null 2>&1"
        )
        p.push(tar, "/work/bundle.tar")
        p.run("cd /work && tar -xf bundle.tar && rm bundle.tar")
        p.run(SETUP)
        say("gpu", f"ready in {time.time() - t0:.0f} s (cost so far ${cost():.2f})")

        ren = {"observation.images.front": "observation.images.camera1"}
        if "wrist" in a.cameras:
            ren["observation.images.wrist"] = "observation.images.camera2"
        repo = info.get("repo_id") or "local/rohub"
        train = (
            f"cd /work && HF_HUB_OFFLINE=1 nohup .venv/bin/lerobot-train --policy.path=lerobot/smolvla_base "
            f"--dataset.repo_id={shlex.quote(repo)} --dataset.root=/work/data/train --dataset.video_backend=pyav "
            f"--rename_map={shlex.quote(json.dumps(ren))} --batch_size=8 --steps={a.steps} --log_freq=100 "
            f"--save_freq={a.steps} --seed=1000 --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false "
            f"--output_dir=/work/out --job_name=onthefly > /work/train.log 2>&1; echo $? > /work/train.exit"
        )
        p.run(f"(setsid bash -c {shlex.quote(train)} >/dev/null 2>&1 </dev/null &)")
        t_train, last = time.time(), -1
        while True:
            time.sleep(10)
            tail = p.run(
                "tr '\\r' '\\n' < /work/train.log 2>/dev/null | grep -E '^Training' | tail -1; cat /work/train.exit 2>/dev/null",
                check=False,
            )
            m = re.search(r"(\d+)/(\d+) \[[^<]*<[^,]*,\s*([\d.]+)(step/s|s/step)", tail)
            if m and int(m.group(1)) != last:
                last = int(m.group(1))
                v = float(m.group(3))
                sps = 1 / v if m.group(4) == "step/s" else v
                say(
                    "train",
                    f"step {last}/{a.steps} {sps:.3f} s/step cost ${cost():.2f}",
                )
            code = tail.strip().splitlines()[-1] if tail.strip() else ""
            if code.isdigit() and "Training" not in code:
                if code != "0":
                    raise RuntimeError(
                        "training failed:\n"
                        + p.run("tail -c 1500 /work/train.log", check=False)
                    )
                break
        ck_remote = f"/work/out/checkpoints/{a.steps:06d}/pretrained_model"
        train_s = time.time() - t_train
        p.pull(ck_remote, out)  # -> out/pretrained_model
        say(
            "train", f"done checkpoint={out / 'pretrained_model'} seconds={train_s:.0f}"
        )

        if not a.no_eval:
            ev = (
                f"cd /work && MUJOCO_GL=egl PYTHONPATH=/work/src HF_HUB_OFFLINE=1 .venv/bin/python -m rohub.task_eval "
                f"{a.task} {ck_remote} --device cuda --seeds {a.eval_seeds} --out /work/eval > /work/eval.log 2>&1; "
                f"MUJOCO_GL=egl PYTHONPATH=/work/src .venv/bin/python -c 'from pathlib import Path; import json; "
                f"from rohub.task_eval import film; print(json.dumps(film(\"{a.task}\", Path(\"/work/eval\"), Path(\"/work/eval/hero.mp4\"))))' "
                f"> /work/eval/hero.log 2>&1"
            )
            p.run(ev, check=False)
            p.pull("/work/eval/", out / "eval")
            evj = out / "eval" / "eval.json"
            if evj.exists():
                e = json.loads(evj.read_text())
                say("eval", f"{e['successes']}/{e['seeds']}")
            else:
                say("eval", "failed: " + p.run("grep -v Warning /work/eval.log | tail -c 4000", check=False))
    finally:
        gql('mutation { podTerminate(input:{podId:"%s"}) }' % pod["id"])
        say("gpu", "pod terminated")
        say("cost", f"total ${cost():.2f} gpu_seconds={time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
