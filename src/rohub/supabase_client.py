"""Supabase helpers: run state, gate telemetry, artifact uploads, trajectory search.

Missing credentials do not crash the pipeline. Calls land in an in-memory log so
local runs and tests still record every gate verdict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "robohub-artifacts")


class RoboHubStore:
    """Thin wrapper around supabase-py. One store per process."""

    def __init__(self, url: str | None = None, key: str | None = None) -> None:
        self.url = url or os.environ.get("SUPABASE_URL", "")
        self.key = key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")
        self.offline_log: list[dict[str, Any]] = []
        self._client = None
        if self.url and self.key and "YOUR_PROJECT" not in self.url:
            from supabase import create_client

            self._client = create_client(self.url, self.key)

    @property
    def live(self) -> bool:
        return self._client is not None

    def _remember(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = {"kind": kind, **payload}
        self.offline_log.append(row)
        return row

    def create_task(self, prompt: str, arm_type: str = "so101", status: str = "queued") -> dict[str, Any]:
        body = {"prompt": prompt, "arm_type": arm_type, "status": status}
        if not self.live:
            return self._remember("task", {**body, "id": f"local-{len(self.offline_log)}"})
        res = self._client.table("tasks").insert(body).execute()
        return res.data[0]

    def set_task_status(self, task_id: str, status: str) -> dict[str, Any]:
        if not self.live or str(task_id).startswith("local-"):
            return self._remember("task_status", {"id": task_id, "status": status})
        res = self._client.table("tasks").update({"status": status}).eq("id", task_id).execute()
        return res.data[0] if res.data else {"id": task_id, "status": status}

    def log_gate_audit(
        self,
        task_id: str,
        gate_audit: dict[str, Any] | list[dict[str, Any]],
        *,
        passed: bool,
        video_storage_path: str | None = None,
        dataset_path: str | None = None,
        trajectory_vector: list[float] | None = None,
        demonstration_id: str | None = None,
    ) -> dict[str, Any]:
        """Write one clip's 21-gate JSON. Upsert when demonstration_id is set."""
        body: dict[str, Any] = {
            "task_id": task_id,
            "passed_gates": passed,
            "gate_audit": {"gates": gate_audit} if isinstance(gate_audit, list) else gate_audit,
        }
        if video_storage_path:
            body["video_storage_path"] = video_storage_path
        if dataset_path:
            body["dataset_path"] = dataset_path
        if trajectory_vector is not None:
            body["trajectory_vector"] = trajectory_vector
        if not self.live or str(task_id).startswith("local-"):
            return self._remember("gate_audit", {**body, "id": demonstration_id or f"local-demo-{len(self.offline_log)}"})
        if demonstration_id:
            res = self._client.table("demonstrations").update(body).eq("id", demonstration_id).execute()
        else:
            res = self._client.table("demonstrations").insert(body).execute()
        return res.data[0]

    def record_model(
        self,
        task_id: str,
        model_name: str,
        *,
        weights_url: str | None = None,
        eval_score: float | None = None,
        training_cost: float | None = None,
    ) -> dict[str, Any]:
        body = {
            "task_id": task_id,
            "model_name": model_name,
            "weights_url": weights_url,
            "eval_score": eval_score,
            "training_cost": training_cost,
        }
        if not self.live or str(task_id).startswith("local-"):
            return self._remember("model", body)
        res = self._client.table("models").insert(body).execute()
        return res.data[0]

    def search_trajectories(
        self,
        query_embedding: list[float],
        match_count: int = 8,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.live:
            return []
        res = self._client.rpc(
            "search_similar_trajectories",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "task_filter": task_id,
            },
        ).execute()
        return list(res.data or [])

    def accepted_count(self, task_id: str) -> int:
        if not self.live or str(task_id).startswith("local-"):
            return sum(
                1
                for row in self.offline_log
                if row.get("kind") == "gate_audit" and row.get("task_id") == task_id and row.get("passed_gates")
            )
        res = (
            self._client.table("demonstrations")
            .select("id", count="exact")
            .eq("task_id", task_id)
            .eq("passed_gates", True)
            .execute()
        )
        return int(res.count or 0)

    def accepted_dataset_paths(self, task_id: str) -> list[str]:
        if not self.live or str(task_id).startswith("local-"):
            return [
                row["dataset_path"]
                for row in self.offline_log
                if row.get("kind") == "gate_audit"
                and row.get("task_id") == task_id
                and row.get("passed_gates")
                and row.get("dataset_path")
            ]
        res = (
            self._client.table("demonstrations")
            .select("dataset_path")
            .eq("task_id", task_id)
            .eq("passed_gates", True)
            .execute()
        )
        return [row["dataset_path"] for row in (res.data or []) if row.get("dataset_path")]

    def list_ready_tasks(self, min_accepted: int) -> list[dict[str, Any]]:
        """Tasks still queued or gating that already have enough accepted clips."""
        if not self.live:
            return []
        tasks = self._client.table("tasks").select("*").in_("status", ["queued", "gating", "accepted"]).execute()
        ready = []
        for task in tasks.data or []:
            if self.accepted_count(task["id"]) >= min_accepted:
                ready.append(task)
        return ready

    def upload_artifact(self, local_path: str | Path, object_path: str, content_type: str = "application/octet-stream") -> str:
        path = Path(local_path)
        data = path.read_bytes()
        if not self.live:
            self._remember("upload", {"object_path": object_path, "bytes": len(data)})
            return f"offline://{BUCKET}/{object_path}"
        self._client.storage.from_(BUCKET).upload(
            object_path,
            data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return f"{self.url}/storage/v1/object/{BUCKET}/{object_path}"


def get_store() -> RoboHubStore:
    return RoboHubStore()
