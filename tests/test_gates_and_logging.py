"""Gate catalog and Supabase gate-log behavior. Physics imports are optional."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATES_PY = ROOT / "src" / "rohub" / "gates.py"

EXPECTED = {
    "fps",
    "length",
    "scene",
    "one_object",
    "no_duplicate",
    "hand_visible",
    "block_visible",
    "events",
    "coupled",
    "lift",
    "carry",
    "video_in_bowl",
    "ik",
    "vel",
    "acc",
    "jerk",
    "close_before_lift",
    "open_over_bowl",
    "sim_success",
    "self_collision",
    "episode_len",
}


def _gate_ids() -> set[str]:
    tree = ast.parse(GATES_PY.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in EXPECTED:
            found.add(node.value)
    return found


def test_gemini_budget_picks_veo_tier():
    from rohub.gemini import VEO, VEO_FAST, model_for_budget

    assert model_for_budget(2) == VEO_FAST
    assert model_for_budget(4) == VEO_FAST
    assert model_for_budget(6) == VEO
    assert model_for_budget(10) == VEO
    assert model_for_budget(None) == VEO_FAST


def test_twenty_one_gate_ids_are_declared():
    found = _gate_ids()
    assert found == EXPECTED
    assert len(found) == 21


def test_offline_store_records_gate_json():
    from rohub.supabase_client import RoboHubStore

    store = RoboHubStore(url="", key="")
    gates = [{"id": gid, "passed": gid != "jerk", "value": 1, "limit": "spec", "why": "peak jerk"} for gid in sorted(EXPECTED)]
    row = store.log_gate_audit("local-task", gates, passed=False, video_storage_path="clips/v00.mp4")
    assert row["passed_gates"] is False
    assert row["gate_audit"]["gates"][0]["id"]
    assert store.accepted_count("local-task") == 0
    store.log_gate_audit("local-task", gates, passed=True)
    assert store.accepted_count("local-task") == 1
    assert store.search_trajectories([0.0] * 8) == []


def test_worker_idle_without_supabase(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
    import importlib
    import sys

    sys.path.insert(0, str(ROOT / "gpu"))
    sys.path.insert(0, str(ROOT / "src"))
    import worker

    importlib.reload(worker)
    assert worker.dispatch_ready(min_accepted=4, steps=10) == []
