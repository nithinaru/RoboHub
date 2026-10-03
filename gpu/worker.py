"""Watch the Supabase tasks queue and start SmolVLA once enough clips pass the gates.

usage: python gpu/worker.py
Stops after one training dispatch when ROBOHUB_WORKER_ONCE=1 (useful in tests).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rohub_trainer import train  # noqa: E402
from rohub.supabase_client import get_store  # noqa: E402


def dispatch_ready(min_accepted: int | None = None, steps: int | None = None) -> list[dict]:
    """Train every task that already has enough accepted demonstrations."""
    store = get_store()
    need = min_accepted if min_accepted is not None else int(os.environ.get("ROBOHUB_MIN_ACCEPTED", "4"))
    train_steps = steps if steps is not None else int(os.environ.get("ROBOHUB_TRAIN_STEPS", "3000"))
    finished = []
    for task in store.list_ready_tasks(need):
        paths = store.accepted_dataset_paths(task["id"])
        dataset = next((Path(p) for p in paths if Path(p).exists()), None)
        if dataset is None:
            print(f"[worker] task {task['id']} has {need}+ accepts but no local dataset yet", flush=True)
            continue
        store.set_task_status(task["id"], "training")
        try:
            result = train(dataset, train_steps, task_id=task["id"])
        except Exception as exc:
            store.set_task_status(task["id"], "failed")
            print(f"[worker] training failed for {task['id']}: {exc}", flush=True)
            continue
        finished.append({"task_id": task["id"], **result})
    return finished


def main() -> None:
    once = os.environ.get("ROBOHUB_WORKER_ONCE", "") == "1"
    poll_s = float(os.environ.get("ROBOHUB_WORKER_POLL_S", "20"))
    print("[worker] watching tasks for accepted demonstrations", flush=True)
    while True:
        dispatch_ready()
        if once:
            return
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
