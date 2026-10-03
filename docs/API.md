# RoboHub local API

Server: `src/rohub/web.py` (FastAPI). Start it with the Runway key from the Keychain in the environment:

```
RUNWAY_API_KEY="$(security find-generic-password -s RUNWAY_API_KEY -w)" PYTHONPATH=src \
  .venv/bin/python -m uvicorn rohub.web:app --host 127.0.0.1 --port 8766
```

All bodies are JSON. Errors: `422 {"ok": false, "reason": "..."}` for a refused request, `404 {"detail": "..."}` for a
missing run, `409 {"detail": "the <stage> stage is still running"}` when a local stage is already running (one local
stage at a time; GPU training runs alongside them, one training job at a time).

Run files are served statically under `/runs/<slug>/...` (for example `/runs/put-the-red-block-in-the-bowl/clips/v08.mp4`).
Runway credits cost $0.01 each.

## Model Router

### GET /api/routers
The account's Runway Model Router configs.
```json
{"routers": [
  {"slug": "demo-cheap", "description": "RoboHub robot demonstrations, optimize for cost", "optimize_for": "cost", "max_credits": null, "fallback": true},
  {"slug": "demo-fast", "optimize_for": "latency", "max_credits": null, "fallback": true, "description": "..."},
  {"slug": "demo-best", "optimize_for": "quality", "max_credits": null, "fallback": true, "description": "..."},
  {"slug": "robot", "optimize_for": "cost", "max_credits": null, "fallback": false, "description": null}
]}
```
`max_credits` is the router's price ceiling (`settings.maxCreditsPerGeneration.video` on Runway), null if none.

### POST /api/route
Automatic routing from the task text (rubric in `src/rohub/route.py`), plus Runway's free dry run on the chosen
router with the run's first frame. Nothing is generated or billed.

Rubric, in order: a hard-task cue (cloth, laundry, fabric, fold; rope, cable, knot; pour, water, liquid; both hands,
bimanual; then, next, sequence, sort, tidy; insert, peg, plug, precise) -> `demo-best`; else `live: true` ->
`demo-fast`; else (one rigid object pick, place, push, stack) -> `demo-cheap`. `claude: true` asks the headless
Claude CLI to confirm or overrule; any error or more than 10 s keeps the rubric.

Request:
```json
{"task": "put the red block in the bowl", "live": false, "claude": false}
```
Response:
```json
{"task": "put the red block in the bowl", "slug": "put-the-red-block-in-the-bowl",
 "router": "demo-cheap",
 "reasons": ["single pick, place, push or stack", "one rigid object (block)", "easy task: the cheapest model is enough"],
 "source": "rubric",
 "feasible": true, "infeasible_reason": null,
 "dryRun": {"model": "gen4_turbo", "provider": "runway", "credits": 25, "optimize_for": "cost",
            "resolved": {"duration": 5, "ratio": "1280:720", "resolution": "720p"}},
 "first_frame": "/runs/put-the-red-block-in-the-bowl/clips/v01.png"}
```
`feasible: false` means the planner refuses the task for a 5-DOF two-finger arm (the footage job will refuse it too);
the routing decision is still returned. `dryRun` is `{"error": "..."}` if Runway could not be reached.

### POST /api/router/dryrun
Free dry run on several routers at once (default the three demo routers, run in parallel, about 4 s).

Request: `{"task": "put the red block in the bowl", "routers": ["demo-cheap", "demo-fast", "demo-best"]}` (routers optional)

Response:
```json
{"ok": true, "task": "put the red block in the bowl", "slug": "put-the-red-block-in-the-bowl", "clip": "v26",
 "free": true, "seconds": 3.9,
 "picks": [
  {"router": "demo-cheap", "model": "gen4_turbo", "provider": "runway", "optimize_for": "cost", "price_ceiling": null,
   "resolved": {"duration": 5, "ratio": "1280:720", "resolution": "720p"}, "estimated_credits": 25, "usd": 0.25},
  {"router": "demo-fast", "model": "seedance2_fast", "provider": "bytedance", "estimated_credits": 145, "usd": 1.45, "...": "..."},
  {"router": "demo-best", "model": "seedance2_5", "provider": "bytedance", "estimated_credits": 150, "usd": 1.5, "...": "..."}
 ]}
```
A router that fails returns `{"router": "...", "error": "..."}` in its slot.

## Generation

### POST /api/footage
Start a footage job (Runway). A sentence whose run already has footage gets NEW clips after its highest id.

Request:
```json
{"task": "put the red block in the bowl", "clips": 1, "router": null, "live": false}
```
- `router` omitted or null: chosen automatically by the rubric (as `/api/route`), `live` feeds it.
- `router: "demo-best"` (any router slug): use that router.
- `router: "direct"`: the old fixed path (gen4_turbo image-to-video, no router).
- `clips`: 1 to 8. Each new clip's first frame is a gen4_image_turbo restyle (2 credits) of the run's base frame.

Response:
```json
{"job": "3f2a9c1d0b7e", "slug": "put-the-red-block-in-the-bowl", "stage": "footage", "clips": 1,
 "clip_ids": ["v26"], "existing": ["v01", "v03", "...", "v25"],
 "router": "demo-cheap",
 "route": {"router": "demo-cheap", "reasons": ["..."], "source": "rubric"}}
```
`route` is null when the router was given explicitly.

### GET /api/jobs/{job}/events  (Server-Sent Events)
`data: {json}` per event, in order; the stream ends after `job_end`. Every event has `type` and `ts` (unix seconds).

Footage job events:
```json
{"type": "job_start", "stage": "footage"}
{"type": "runway", "clip": "v26", "stage": "frame", "model": "gen4_image_turbo", "status": "RUNNING", "progress": 0.4, "cost": 2, "seconds": 6.1}
{"type": "frame_ready", "clip": "v26", "file": "clips/v26.png"}
{"type": "runway", "clip": "v26", "stage": "video", "model": "gen4_turbo", "status": "ROUTED", "progress": 0, "cost": 25,
 "routing": {"router": "demo-cheap", "model": "gen4_turbo", "provider": "runway", "optimize_for": "cost", "price_ceiling": null, "estimated_credits": 25, "dry_run": {"...": "..."}}}
{"type": "runway", "clip": "v26", "stage": "video", "status": "RUNNING", "progress": 0.55, "task_id": "...", "cost": 25, "seconds": 14.2, "routing": {"...": "..."}}
{"type": "runway", "clip": "v26", "stage": "video", "status": "SUCCEEDED", "progress": 1, "cost": 25, "seconds": 24.0,
 "routing": {"router": "demo-cheap", "model": "gen4_turbo", "realized_credits": 25, "...": "..."}}
{"type": "routed", "clip": "v26", "router": "demo-cheap", "model": "gen4_turbo", "provider": "runway",
 "estimated_credits": 25, "realized_credits": 25, "seconds": 24.0, "cached": false}
{"type": "clip_ready", "clip": "v26", "file": "clips/v26.mp4"}
{"type": "footage_done", "clips": ["v26"], "balance": 48081}
{"type": "job_end", "stage": "footage", "ok": true, "seconds": 52.3}
```
`status` runs ROUTED (free dry run done) -> PENDING -> RUNNING -> SUCCEEDED (or FAILED / CANCELLED / TIMEOUT);
`CACHED` means the identical request was made before and cost 0. `budget_stop` (`message`) and `error` (`message`)
may appear; `log` events (`line`) carry plain output.

For the video stage, the event's `model` field is `router:<slug>` until the routing is known; use `routing.model`.
`realized_credits` is the live balance drop across task creation.

### POST /api/data
`{"slug": "put-the-red-block-in-the-bowl"}` -> `{"job", "slug", "stage": "data"}`. Tracks, audits, gates, retargets
every clip of the run and writes the LeRobot dataset. Events: `data_start {clips}`, `clip_stage {clip, stage:
track|audit|gates|retarget}`, `tracked {clip, frames, fps, grasp_frame, release_frame, file}`, `verdict {clip,
accepted, reason, gates_total, gates_passed, failed, gates[], demo}`, `robot_film {clip, file}`, `reanchor {clip,
tried, accepted}`, `episodes {total, reanchor}`, `dataset_progress {episode, total, clip, kind, frames}`,
`data_done {accepted[], episodes, frames, codebase_version, loaded_back, seconds}`, `job_end`.

## Run views

### GET /api/runs/{slug}/tiles
Every clip of a run as a tile: footage, routing, physics verdict, SmolVLA film.
```json
{"slug": "put-the-red-block-in-the-bowl", "task": "put the red block in the bowl",
 "tiles": [
  {"clip": "v01", "footage": "/runs/put-the-red-block-in-the-bowl/clips/v01.mp4", "first_frame": "/runs/.../clips/v01.png",
   "video_model": "gen4_turbo", "variant": "v01", "routing": null,
   "verdict": {"accepted": true, "reason": "accepted", "gates_total": 21, "gates_passed": 21},
   "vla": {"clip": "v01", "success": true, "frames": 294, "cube_xy": [0.251, -0.0881], "model": "SmolVLA (...)",
           "video": "/runs/put-the-red-block-in-the-bowl/media/vla/v01.mp4", "...": "..."},
   "skeleton": "/runs/.../media/v01_skeleton.mp4"},
  {"clip": "v20", "footage": "/runs/.../clips/v20.mp4", "video_model": "seedance2_5", "variant": "v08",
   "routing": {"router": "demo-best", "model": "seedance2_5", "provider": "bytedance", "optimize_for": "quality",
               "estimated_credits": 150, "realized_credits": 150, "seconds": 170.6, "cached": false},
   "verdict": {"accepted": false, "reason": "...", "gates_total": 21, "gates_passed": 17}, "vla": null, "skeleton": "..."}
 ],
 "totals": {"clips": 22, "accepted": 6, "routed_credits": 1920, "routed_usd": 19.2, "routed_seconds": 1834.2, "credits_per_accepted": 320.0}}
```
`verdict` is null until the clip went through the gates; `vla` is null until `media/vla/<clip>.mp4` + `.json` exist.
`variant` is the scene the clip was made from (router bench clips share scenes across routers).

### GET /api/runs/{slug}/routing
Just the routed clips: `{"clips": {"v20": {"router", "model", "provider", "optimize_for", "estimated_credits",
"realized_credits", "seconds", "cached", "verdict": {"accepted", "reason"} | null}}}`.

### GET /api/runs/{slug}/router_bench
`data/web-runs/<slug>/router_bench.json` (written by `scripts/router_bench.py`, updated live while it runs).
```json
{"slug": "put-the-red-block-in-the-bowl", "task": "put the red block in the bowl",
 "routers": ["demo-cheap", "demo-fast", "demo-best"], "clips_per_router": 6, "budget": 2500,
 "started": "2026-09-30T10:05:12", "generated": "...", "judged": "...", "updated": "...",
 "credits_spent_balance": 1932, "method": "...",
 "table": [
  {"router": "demo-cheap", "optimize_for": "cost", "models": ["gen4_turbo"], "clips": 6, "judged": 6, "accepted": 2,
   "acceptance_rate": 0.333, "credits": 150, "usd": 1.5, "credits_per_accepted": 75.0,
   "accepted_per_1000_credits": 13.33, "mean_generation_s": 24.9, "winner": true}
 ],
 "clips": [{"clip_id": "v08", "router": "demo-cheap", "variant": "v08", "status": "SUCCEEDED", "model": "gen4_turbo",
            "estimated_credits": 25, "realized_credits": 25, "seconds": 23.8, "accepted": true, "reason": "accepted",
            "failed": [], "gates_total": 21, "gates_passed": 21}],
 "frames": [{"variant": "v08", "model": "gen4_image_turbo", "credits": 2}],
 "datasets": {"demo-cheap": {"path": "data/probes/router-demo-cheap", "abs_path": "...", "repo_id": "local/rohub_router_demo_cheap",
                             "clips": ["v08", "v12"], "episodes": 44, "frames": 15000, "per_clip": 25, "cameras": "front,wrist"}}}
```
`winner` marks the most physics-accepted demos per 1,000 credits. `accepted` is null on a clip until it is judged;
clip `status`: QUEUED, RUNNING, SUCCEEDED, FAILED, SKIPPED_BUDGET. (The numbers above are the shape, not the result;
see README "Model Router" for the measured table.)

### Other run endpoints (unchanged)
- `GET /api/runs/{slug}/events`: a saved run's finished footage and data events, to redraw it.
- `GET /api/runs/{slug}/vla`: `{accepted[], films{clip: rec}, missing[], ckpt, running, command}`.
- `POST /api/vla {slug, clips?}`: film the missing scenarios with the local SmolVLA checkpoint (needs `ROBOHUB_VLA_CKPT`).
- `GET /api/runs/{slug}/dataset`: the LeRobot dataset's counts; `GET /api/runs/{slug}/dataset.zip`: the dataset.
- `GET /api/runs/{slug}/refine`: prompt fixes from the last rejections (text only, free).
- `GET /api/budget`: `{balance, floor, clips, cost_first, cost_rest, cost_run}`; `POST /api/plan {task, clips?}`: feasibility and cost of the direct path.

## GPU training (SmolVLA on RunPod)

### GET /api/trainer
`{"connected": true, "script": "/Users/pranavachar/helloworld/so101/train_vla.sh", "running": null}`.
`connected: false` means the script is missing: show "GPU trainer not connected". Override the path with
`ROBOHUB_TRAINER`.

### POST /api/train
Request: `{"slug": "put-the-red-block-in-the-bowl", "steps": 3000, "dataset": "data/probes/router-demo-cheap"}`
(`dataset` optional: a LeRobot root under `data/`; default is the run's own `lerobot/`). Steps are clamped to 100..100000.

Response: `{"job": "a1b2c3d4e5f6", "slug": "...", "stage": "train"}`, or 422 `{"ok": false, "reason": "GPU trainer not connected"}`
/ `"no LeRobot dataset for this run yet"`.

Runs `train_vla.sh <dataset root> <steps>`. Events on `/api/jobs/{job}/events`:
```json
{"type": "train_start", "script": "...", "root": "...", "steps": 3000}
{"type": "train_line", "line": "[gpu] rented L40S at $1.09/h (pod chf5bnk7x1459w)", "seconds": 21.0, "tag": "gpu", "gpu": "L40S", "usd_per_hour": 1.09}
{"type": "train_line", "line": "[gpu] ready in 95 s (cost so far $0.03)", "seconds": 95.2, "tag": "gpu", "usd": 0.03}
{"type": "train_line", "line": "[train] step 1200/3000 0.157 s/step cost $0.07", "seconds": 300.4, "tag": "train", "step": 1200, "steps": 3000, "s_per_step": 0.157, "usd": 0.07}
{"type": "train_line", "line": "[train] done checkpoint=/path/pretrained_model seconds=112", "tag": "train", "done": true, "checkpoint": "/path/pretrained_model", "train_seconds": 112}
{"type": "train_line", "line": "[eval] 41/50", "tag": "eval", "eval_ok": 41, "eval_n": 50}
{"type": "train_line", "line": "[tiles] /path/tiles", "tag": "tiles", "tiles_dir": "/path/tiles"}
{"type": "train_line", "line": "[cost] total $0.12 gpu_seconds=300", "tag": "cost", "usd": 0.12, "gpu_seconds": 300}
{"type": "train_end", "seconds": 412.0, "usd": 0.12, "checkpoint": "...", "eval_ok": 41, "eval_n": 50, "tiles_dir": "...", "gpu": "L40S"}
{"type": "job_end", "stage": "train", "ok": true, "seconds": 412.3}
```
Lines without a `[tag]` arrive as `train_line` with only `line` and `seconds`. On a non-zero exit: `error` then
`job_end {ok: false}`. The meter's GPU $ is the latest `usd`; training time is the job's `seconds`.
