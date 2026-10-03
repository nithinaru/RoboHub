"""One pipeline stage as a child process of the web server. Prints every progress event as a line
`@@EV {json}` on stdout; anything else it prints is a plain log line. The server relays both to the page.

  python -m rohub.webjob footage '{"task": "...", "run_dir": "...", "clips": 5}'
  python -m rohub.webjob data    '{"task": "...", "run_dir": "..."}'
  python -m rohub.webjob vla     '{"run_dir": "...", "ckpt": "...", "clips": ["v08"]}'
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

_OUT = threading.Lock()


def emit(ev: dict) -> None:
    ev = {"ts": round(time.time(), 3), **ev}
    with _OUT:
        sys.stdout.write("@@EV " + json.dumps(ev, default=float) + "\n")
        sys.stdout.flush()


def log(msg: str) -> None:
    with _OUT:
        print(msg, flush=True)


def parse_train_line(line: str) -> dict:
    """One trainer line -> fields. Tags: [gpu] pod lifecycle, [train] step k/n s/step cost $x | done checkpoint=,
    [eval] k/n, [tiles] <dir>, [cost] total $x gpu_seconds=n."""
    import re

    ev: dict = {}
    m = re.match(r"\s*\[(\w+)\]\s*(.*)", line)
    if not m:
        return ev
    tag, rest = m[1], m[2]
    ev["tag"] = tag
    if (st := re.search(r"step\s+(\d+)\s*/\s*(\d+)", rest)) and tag == "train":
        ev.update(step=int(st[1]), steps=int(st[2]))
    if sp := re.search(r"([\d.]+)\s*s/step", rest):
        ev["s_per_step"] = float(sp[1])
    if c := re.search(r"(?:cost(?: so far)?|total)\s*\$([\d.]+)", rest):
        ev["usd"] = float(c[1])
    if r := re.search(r"\$([\d.]+)/h", rest):
        ev["usd_per_hour"] = float(r[1])
    if g := re.search(r"gpu_seconds=([\d.]+)", rest):
        ev["gpu_seconds"] = float(g[1])
    if tag == "train" and rest.startswith("done"):
        ev["done"] = True
        if ck := re.search(r"checkpoint=(\S+)", rest):
            ev["checkpoint"] = ck[1]
        if sec := re.search(r"seconds=([\d.]+)", rest):
            ev["train_seconds"] = float(sec[1])
    if tag == "eval" and (e := re.search(r"(\d+)\s*/\s*(\d+)", rest)):
        ev.update(eval_ok=int(e[1]), eval_n=int(e[2]))
    if tag == "tiles":
        ev["tiles_dir"] = rest.strip()
    if tag == "gpu" and (rent := re.search(r"rented (.+?) at", rest)):
        ev["gpu"] = rent[1]
    return ev


def train_stage(args: dict) -> None:
    """Run the GPU trainer script (~/helloworld/so101/train_vla.sh <dataset root> <steps>) and relay every line as
    a train_line event with the parsed fields (parse_train_line)."""
    import subprocess

    script = Path(args["script"])
    t0 = time.time()
    emit({"type": "train_start", "script": str(script), "root": args["root"], "steps": int(args["steps"])})
    p = subprocess.Popen(
        ["/bin/bash", str(script), args["root"], str(int(args["steps"]))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert p.stdout is not None
    last: dict = {}
    for line in p.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        f = parse_train_line(line)
        last.update({k: v for k, v in f.items() if k in ("usd", "checkpoint", "eval_ok", "eval_n", "tiles_dir", "gpu")})
        emit({"type": "train_line", "line": line[:400], "seconds": round(time.time() - t0, 1), **f})
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"{script.name} exited {code}")
    emit({"type": "train_end", "seconds": round(time.time() - t0, 1), **last})


def main() -> None:
    stage, args = sys.argv[1], json.loads(sys.argv[2])
    from . import stages
    from .plan import parse_task

    run_dir = Path(args["run_dir"])
    emit({"type": "job_start", "stage": stage})
    t0 = time.time()
    try:
        if stage == "footage":
            task = parse_task(args["task"])
            fixes = None
            if args.get("refine"):  # PhyT2V Step 3: the last take's rejections rewrite the prompts
                from .plan import refine_fixes

                verdicts = json.loads((run_dir / "data.json").read_text())["clips"]
                fixes = refine_fixes(verdicts, task)
            recs = stages.footage(
                task, run_dir, int(args["clips"]), emit=emit, fixes=fixes, start=int(args.get("start", 0)),
                router=args.get("router") or None,
            )
            from . import runway

            emit(
                {
                    "type": "footage_done",
                    "clips": [r["clip_id"] for r in recs],
                    "balance": runway.balance(),
                }
            )
        elif stage == "data":
            stages.training_data(
                parse_task(args["task"]),
                run_dir,
                reanchor_n=int(args.get("reanchor", 15)),
                emit=emit,
                log=log,
            )
        elif stage == "train":  # the GPU trainer hook: ~/helloworld/so101/train_vla.sh <dataset root> <steps>
            train_stage(args)
        elif stage == "vla":  # the fine-tuned SmolVLA runs each scenario in MuJoCo, filmed for the tiles
            from . import vla_tiles

            vla_tiles.web_stage(run_dir, args, emit=emit, log=log)
        else:
            raise ValueError(f"unknown stage {stage}")
    except Exception as e:  # reported to the page, then the job exits non-zero
        log(traceback.format_exc())
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        emit(
            {
                "type": "job_end",
                "stage": stage,
                "ok": False,
                "seconds": round(time.time() - t0, 1),
            }
        )
        sys.exit(1)
    emit(
        {
            "type": "job_end",
            "stage": stage,
            "ok": True,
            "seconds": round(time.time() - t0, 1),
        }
    )


if __name__ == "__main__":
    main()
