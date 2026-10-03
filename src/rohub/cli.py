"""RoboHub command line.

  bin/robohub run "put the red block in the bowl" --clips 8     # the whole pipeline, cached stage by stage
  bin/robohub generate "put the red block in the bowl" --clips 8 # Runway only
  bin/robohub budget                                             # live balance and credits spent
  bin/robohub serve                                              # the web app on http://127.0.0.1:8765
  bin/robohub train put-the-red-block-in-the-bowl               # the small MLP on a run's dataset + MuJoCo eval

Every stage caches: Runway outputs by request hash, pose by clip hash, the Claude audit by clip hash. Re-running
the same prompt spends nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import runway
from .plan import Infeasible, parse_task

REPO = runway.REPO


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def cmd_budget(_a) -> None:
    bal = runway.balance()
    spent = runway.spent_so_far(bal)
    print(
        json.dumps(
            {
                "balance": bal,
                "spent_by_this_build": spent,
                "floor_balance": runway.FLOOR_BALANCE,
                "left_above_floor": bal - runway.FLOOR_BALANCE,
                "ledger_sum": runway.spent_so_far(None),
                "ledger_calls": len(runway.ledger()),
            }
        )
    )


def _task_or_exit(text: str):
    try:
        return parse_task(text)
    except Infeasible as e:
        print(f"[plan] refused before spending any credit: {e}")
        sys.exit(2)


def cmd_generate(a) -> None:
    from .generate import generate

    task = _task_or_exit(a.task)
    run_dir = REPO / "data" / "runs" / slug(a.task)
    recs = generate(task, run_dir, a.clips)
    print(f"[generate] {len(recs)} clips in {run_dir / 'clips'}")


def cmd_run(a) -> None:
    from .pipeline import run

    task = _task_or_exit(a.task)
    run(task, REPO / "data" / "runs" / slug(a.task), a)


def cmd_serve(a) -> None:
    import uvicorn

    print(f"RoboHub on http://{a.host}:{a.port}", flush=True)
    uvicorn.run("rohub.web:app", host=a.host, port=a.port, log_level="warning")


def cmd_train(a) -> None:
    """Train the small MLP baseline on a run's LeRobot dataset (the clips that passed the gates), evaluate it on 50
    unseen cube positions in MuJoCo and film the hero rollout. The same code the site's trainer used."""
    from .stages import train_and_eval

    run_dir = REPO / "data" / "web-runs" / a.run
    clips = a.clips or [c["clip_id"] for c in json.loads((run_dir / "data.json").read_text())["clips"] if c["accepted"]]

    def emit(ev: dict) -> None:
        t = ev["type"]
        if t == "train_start":
            print(f"[train] {ev['episodes']} episodes, {ev['frames']} frames from {', '.join(ev['clips'])}; {ev['steps']} steps", flush=True)
        elif t == "trained":
            c = ev["card"]
            print(f"[train] done: {c['params']:,} params, L1 {c['final_l1']}, {c['train_seconds']} s on the {c['device']}", flush=True)
        elif t == "eval":
            print(f"[eval] {ev['i']:>2}/{ev['n']} seed {ev['seed']} cube {tuple(ev['cube_xy'])} {'IN BOWL' if ev['success'] else 'miss'}", flush=True)
        elif t == "rollout_film":
            print(f"[film] {ev['file']} (seed {ev['seed']}, {ev['frames']} frames)", flush=True)

    train_and_eval(run_dir, clips, steps=a.steps, eval_seeds=a.eval_seeds, emit=emit, log=lambda m: print(m, flush=True))


def main() -> None:
    ap = argparse.ArgumentParser(prog="robohub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.set_defaults(fn=cmd_budget)
    g = sub.add_parser("generate")
    g.add_argument("task")
    g.add_argument("--clips", type=int, default=8)
    g.set_defaults(fn=cmd_generate)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(fn=cmd_serve)
    t = sub.add_parser("train")
    t.add_argument("run", help="a run under data/web-runs/, e.g. put-the-red-block-in-the-bowl")
    t.add_argument("--clips", nargs="*", default=None, help="default: every clip of the run that passed the gates")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--eval-seeds", type=int, default=50)
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--clips", type=int, default=8)
    r.add_argument(
        "--reanchor",
        type=int,
        default=15,
        help="sim re-anchored copies per accepted clip",
    )
    r.add_argument("--eval-seeds", type=int, default=50)
    r.add_argument("--train-steps", type=int, default=4000)
    r.add_argument(
        "--no-generate", action="store_true", help="use only clips already on disk"
    )
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
