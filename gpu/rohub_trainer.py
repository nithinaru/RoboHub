"""Train SmolVLA on a LeRobot dataset on a rented RunPod GPU and store the checkpoint in Supabase.

usage: python gpu/rohub_trainer.py DATASET_ROOT STEPS [--task-id UUID] [--run-dir RUN]

Prints machine-readable lines the RoboHub worker and dashboard show live:
  [gpu] ...            pod lifecycle
  [train] step 1200/3000 0.157 s/step cost $0.07
  [train] done checkpoint=<local path> seconds=<n>
  [eval] 41/50
  [cost] total $<x> gpu_seconds=<n>
The pod is always terminated at the end, even on failure. RUNPOD_API_KEY comes from the environment.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "src"))
from rohub.supabase_client import get_store  # noqa: E402
from runpod import gql  # noqa: E402

HERE = Path(__file__).parent
KEY = Path(os.environ.get("ROBOHUB_SSH_KEY", Path.home() / ".ssh/robohub"))
GPUS = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def deploy(prefer: str | None = None) -> dict:
    pub = KEY.with_suffix(".pub").read_text().strip()
    q = "mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) { id costPerHr machine { gpuDisplayName } } }"
    for gpu in ([prefer] if prefer else []) + [g for g in GPUS if g != prefer]:
        inp = {
            "cloudType": "ALL",
            "gpuCount": 1,
            "gpuTypeId": gpu,
            "name": "robohub-smolvla",
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
        for d in ("src", "vendor"):
            t.add(ROOT / d, arcname=d, filter=skip)
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


def publish_checkpoint(task_id: str, out: Path, eval_score: float | None, training_cost: float) -> str | None:
    """Zip the SmolVLA checkpoint and eval rollouts, then write the model row."""
    store = get_store()
    bundle_path = out / "smolvla_checkpoint.tar"
    with tarfile.open(bundle_path, "w") as archive:
        if (out / "pretrained_model").exists():
            archive.add(out / "pretrained_model", arcname="pretrained_model")
        if (out / "eval.json").exists():
            archive.add(out / "eval.json", arcname="eval.json")
        tiles = out / "tiles"
        if tiles.exists():
            archive.add(tiles, arcname="tiles")
    weights_url = store.upload_artifact(
        bundle_path,
        f"{task_id}/models/smolvla.tar",
        content_type="application/x-tar",
    )
    store.record_model(
        task_id,
        "smolvla",
        weights_url=weights_url,
        eval_score=eval_score,
        training_cost=training_cost,
    )
    store.set_task_status(task_id, "done")
    say("store", f"checkpoint {weights_url}")
    return weights_url


def train(
    dataset: Path,
    steps: int,
    *,
    run_dir: Path | None = None,
    cameras: str = "front",
    no_eval: bool = False,
    gpu: str | None = None,
    out: Path | None = None,
    task_id: str | None = None,
) -> dict:
    """Rent a pod, fine-tune SmolVLA, evaluate, and optionally persist artifacts."""
    ds = dataset.resolve()
    assert (ds / "meta" / "info.json").exists(), f"not a LeRobot dataset: {ds}"
    info = json.loads((ds / "meta/info.json").read_text())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = (out or HERE / "runs" / f"vla-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    say(
        "gpu",
        f"dataset {info.get('total_episodes')} episodes, {info.get('total_frames')} frames; {steps} steps",
    )

    t0 = time.time()
    pod = deploy(gpu)
    result: dict = {"out": str(out), "eval_score": None, "cost": None, "weights_url": None}
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
        tar = bundle(ds, run_dir if not no_eval else None)
        p.run(
            "apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync >/dev/null 2>&1"
        )
        p.push(tar, "/work/bundle.tar")
        p.run("cd /work && tar -xf bundle.tar && rm bundle.tar")
        p.run(SETUP)
        say("gpu", f"ready in {time.time() - t0:.0f} s (cost so far ${cost():.2f})")

        ren = {"observation.images.front": "observation.images.camera1"}
        if "wrist" in cameras:
            ren["observation.images.wrist"] = "observation.images.camera2"
        repo = info.get("repo_id") or "local/rohub"
        train_cmd = (
            f"cd /work && HF_HUB_OFFLINE=1 nohup .venv/bin/lerobot-train --policy.path=lerobot/smolvla_base "
            f"--dataset.repo_id={shlex.quote(repo)} --dataset.root=/work/data/train --dataset.video_backend=pyav "
            f"--rename_map={shlex.quote(json.dumps(ren))} --batch_size=8 --steps={steps} --log_freq=100 "
            f"--save_freq={steps} --seed=1000 --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false "
            f"--output_dir=/work/out --job_name=onthefly > /work/train.log 2>&1; echo $? > /work/train.exit"
        )
        p.run(f"(setsid bash -c {shlex.quote(train_cmd)} >/dev/null 2>&1 </dev/null &)")
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
                    f"step {last}/{steps} {sps:.3f} s/step cost ${cost():.2f}",
                )
            code = tail.strip().splitlines()[-1] if tail.strip() else ""
            if code.isdigit() and "Training" not in code:
                if code != "0":
                    raise RuntimeError(
                        "training failed:\n"
                        + p.run("tail -c 1500 /work/train.log", check=False)
                    )
                break
        ck_remote = f"/work/out/checkpoints/{steps:06d}/pretrained_model"
        train_s = time.time() - t_train
        p.pull(ck_remote, out)  # -> out/pretrained_model
        say(
            "train", f"done checkpoint={out / 'pretrained_model'} seconds={train_s:.0f}"
        )

        eval_score = None
        if not no_eval:
            ev = (
                f"cd /work && MUJOCO_GL=egl PYTHONPATH=/work/src HF_HUB_OFFLINE=1 .venv/bin/python -m rohub.vla "
                f"{ck_remote} /work/data/run --device cuda --cameras {cameras} --out /work/eval/eval.json > /work/eval.log 2>&1"
            )
            p.run(ev, check=False)
            evj = p.run("cat /work/eval/eval.json 2>/dev/null", check=False)
            if evj.strip():
                e = json.loads(evj)
                (out / "eval.json").write_text(evj)
                eval_score = (e["successes"] / e["seeds"]) if e.get("seeds") else None
                say(
                    "eval",
                    f"{e['successes']}/{e['seeds']} (Wilson 95% {e['wilson95'][0]:.2f}-{e['wilson95'][1]:.2f})",
                )
                tiles = (
                    "cd /work && MUJOCO_GL=egl PYTHONPATH=/work/src .venv/bin/python -m rohub.vla_tiles /work/data/run "
                    "--from-eval /work/eval/eval.json /work/eval/eval_traj.npz --out /work/tiles > /work/tiles.log 2>&1"
                )
                p.run(tiles, check=False)
                p.pull("/work/tiles/", out / "tiles")
                say("tiles", str(out / "tiles"))
            else:
                say(
                    "eval",
                    "failed: "
                    + p.run("tail -c 600 /work/eval.log", check=False).replace(
                        "\n", " "
                    )[-300:],
                )
    finally:
        gql('mutation { podTerminate(input:{podId:"%s"}) }' % pod["id"])
        say("gpu", "pod terminated")
        spent = cost()
        say("cost", f"total ${spent:.2f} gpu_seconds={time.time() - t0:.0f}")
    result["eval_score"] = eval_score
    result["cost"] = spent
    if task_id:
        result["weights_url"] = publish_checkpoint(task_id, out, eval_score, spent)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("steps", type=int)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--cameras", default="front")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--gpu", default=None, help='RunPod gpuTypeId to try first, e.g. "NVIDIA GeForce RTX 4090"')
    ap.add_argument("--out", type=Path, default=None, help="local dir for the checkpoint, eval and tiles")
    ap.add_argument("--task-id", default=os.environ.get("ROBOHUB_TASK_ID"))
    args = ap.parse_args()
    train(
        args.dataset,
        args.steps,
        run_dir=args.run_dir,
        cameras=args.cameras,
        no_eval=args.no_eval,
        gpu=args.gpu,
        out=args.out,
        task_id=args.task_id,
    )


if __name__ == "__main__":
    main()
