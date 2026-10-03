/**
 * Browser and worker client for RoboHub tables, Storage, and Realtime.
 * Uses the Supabase REST and Realtime endpoints. No bundler required.
 */

const BUCKET = "robohub-artifacts";

export function robohubConfig(env = globalThis.process?.env ?? {}) {
  const url = env.SUPABASE_URL || globalThis.ROBOHUB_SUPABASE_URL || "";
  const key = env.SUPABASE_ANON_KEY || globalThis.ROBOHUB_SUPABASE_ANON_KEY || "";
  return { url: url.replace(/\/$/, ""), key, bucket: env.SUPABASE_STORAGE_BUCKET || BUCKET };
}

function headers(key, extra = {}) {
  return {
    apikey: key,
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

async function rest(cfg, path, { method = "GET", body, prefer } = {}) {
  if (!cfg.url || !cfg.key) {
    return { offline: true, path, body };
  }
  const res = await fetch(`${cfg.url}/rest/v1/${path}`, {
    method,
    headers: headers(cfg.key, prefer ? { Prefer: prefer } : {}),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`Supabase ${method} ${path} failed: ${res.status} ${await res.text()}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export async function createTask(cfg, { prompt, armType = "so101", status = "queued" }) {
  const rows = await rest(cfg, "tasks", {
    method: "POST",
    body: { prompt, arm_type: armType, status },
    prefer: "return=representation",
  });
  return Array.isArray(rows) ? rows[0] : rows;
}

export async function setTaskStatus(cfg, taskId, status) {
  return rest(cfg, `tasks?id=eq.${taskId}`, {
    method: "PATCH",
    body: { status },
    prefer: "return=representation",
  });
}

export async function logGateAudit(cfg, {
  taskId,
  gateAudit,
  passed,
  videoStoragePath = null,
  datasetPath = null,
  trajectoryVector = null,
}) {
  const body = {
    task_id: taskId,
    passed_gates: passed,
    gate_audit: Array.isArray(gateAudit) ? { gates: gateAudit } : gateAudit,
    video_storage_path: videoStoragePath,
    dataset_path: datasetPath,
    trajectory_vector: trajectoryVector,
  };
  const rows = await rest(cfg, "demonstrations", {
    method: "POST",
    body,
    prefer: "return=representation",
  });
  return Array.isArray(rows) ? rows[0] : rows;
}

export async function recordModel(cfg, { taskId, modelName, weightsUrl, evalScore, trainingCost }) {
  const rows = await rest(cfg, "models", {
    method: "POST",
    body: {
      task_id: taskId,
      model_name: modelName,
      weights_url: weightsUrl ?? null,
      eval_score: evalScore ?? null,
      training_cost: trainingCost ?? null,
    },
    prefer: "return=representation",
  });
  return Array.isArray(rows) ? rows[0] : rows;
}

export async function searchTrajectories(cfg, queryEmbedding, matchCount = 8, taskId = null) {
  return rest(cfg, "rpc/search_similar_trajectories", {
    method: "POST",
    body: {
      query_embedding: queryEmbedding,
      match_count: matchCount,
      task_filter: taskId,
    },
  });
}

export async function uploadArtifact(cfg, objectPath, bytes, contentType = "application/octet-stream") {
  if (!cfg.url || !cfg.key) return `offline://${cfg.bucket}/${objectPath}`;
  const res = await fetch(`${cfg.url}/storage/v1/object/${cfg.bucket}/${objectPath}`, {
    method: "POST",
    headers: headers(cfg.key, {
      "Content-Type": contentType,
      "x-upsert": "true",
    }),
    body: bytes,
  });
  if (!res.ok) {
    throw new Error(`Storage upload failed: ${res.status} ${await res.text()}`);
  }
  return `${cfg.url}/storage/v1/object/${cfg.bucket}/${objectPath}`;
}

/**
 * Subscribe to demonstration inserts for one task.
 * onEvent receives the new row (gate_audit, passed_gates).
 * Returns an unsubscribe function. Uses Supabase Realtime websocket when configured.
 */
export function subscribeGateProgress(cfg, taskId, onEvent) {
  if (!cfg.url || !cfg.key || typeof WebSocket === "undefined") {
    return () => {};
  }
  const wsUrl = cfg.url.replace(/^http/, "ws") + `/realtime/v1/websocket?apikey=${encodeURIComponent(cfg.key)}&vsn=1.0.0`;
  const ws = new WebSocket(wsUrl);
  const topic = `realtime:public:demonstrations:task_id=eq.${taskId}`;
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({
      topic,
      event: "phx_join",
      payload: {
        config: {
          broadcast: { self: false },
          postgres_changes: [{ event: "INSERT", schema: "public", table: "demonstrations", filter: `task_id=eq.${taskId}` }],
        },
      },
      ref: "1",
    }));
  });
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    const record = msg?.payload?.data?.record || msg?.payload?.record;
    if (record) onEvent(record);
  });
  const beat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ topic: "phoenix", event: "heartbeat", payload: {}, ref: "hb" }));
    }
  }, 25000);
  return () => {
    clearInterval(beat);
    ws.close();
  };
}
