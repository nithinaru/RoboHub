# Benchmark: Runway video demonstrations vs scripted expert demonstrations

## Question

Given the same budget of demonstrations, does SmolVLA trained on demonstrations extracted from Runway-generated
human video do about as well as SmolVLA trained on conventional robot demonstrations (the kind labs pay people to
teleoperate)?

## Controlled variable

Where the demonstrations came from. Nothing else.

| Arm | Dataset | How the motion was produced |
|---|---|---|
| runway | `data/probes/aug25` (`local/rohub_aug`) | Runway-generated human video, hand tracked, retargeted to the SO-101, re-anchored to new cube positions, gated in MuJoCo replay (`scripts/augment_dataset.py`) |
| expert | `data/probes/expert74` (`local/rohub_expert`) | Scripted expert `pick_expert.run_episode` in the same simulator (`scripts/make_expert_dataset.py`), the stand-in for teleoperated demonstrations |

## Held fixed

- Model: SmolVLA, lerobot 0.6.1, same base weights and training recipe (steps, batch size, camera set) per row.
- Task and scene: MuJoCo SO-101 "pick" scene, task string `put the red block in the bowl` on every frame.
- Episodes: 74 in both arms. Each expert episode starts from the same cube position as the aug25 episode with the
  same index (exact positions recovered from the augment RNG, not the 0.1 mm rounded log), yaw 0 in both.
- Recording: same writer (`rohub.dataset.write`), 30 fps, 240x320 AV1 video, cameras front and wrist, same
  feature keys, names, units and robot type. The two roots are drop-in replacements for each other in
  `lerobot-train` (`--dataset.repo_id`, `--dataset.root`).
- Acceptance: an expert episode is kept only if the cube was lifted and ends in the bowl (the eval's rule). Any
  failed start, retry or replacement is logged in `expert_episodes.json`.

Not held fixed, by construction: episode length and motion style. The expert is faster and smoother than the
retargeted human motion (smoke test: 268 to 271 frames per expert episode vs 375 for the same two aug25 episodes;
aug25 averages 340 frames), so the expert arm has fewer total frames. Report both frame totals next to the results.

## Eval protocol

`python -m rohub.vla <checkpoint>/pretrained_model data/web-runs/put-the-red-block-in-the-bowl --seeds 50`
(add `--cameras front,wrist` for the two-camera rows). 50 unseen cube positions, seeds 10000 to 10049, the same
positions the MLP baseline was scored on and disjoint from both training sets. Success = the cube was lifted, then
settled in the bowl. SmolVLA sees only the camera frames, the 6 joint angles and the task sentence, never the
cube position.

## Results (successes out of 50)

| Training | runway | expert |
|---|---|---|
| 3000 steps, front | 34/50 (0.55-0.79) | 44/50 (0.76-0.94) |
| 20000 steps, front | 43/50 (0.74-0.93) | 44/50 (0.76-0.94) |
| 6000 steps, front + wrist | 41/50 (0.69-0.90) | not run |

## Caveats

- This is a simulation benchmark: one task, one simulated arm, one scene. It says nothing yet about a real robot.
- The expert arm is a scripted planner with privileged cube pose, which is an optimistic stand-in for human
  teleoperation (no hesitation, no operator error). If Runway demonstrations match it, that is a strong result; if
  they trail it, the gap is an upper bound on what better human data could buy.
- 50 seeds give wide intervals (roughly plus or minus 10 to 14 points at these rates, Wilson 95%). Small
  differences between cells are not meaningful.
- The SmolVLA paper reports results on LIBERO, Meta-World and real SO-100 tasks. Those are different tasks,
  scenes and protocols and are not directly comparable to these numbers; no paper numbers are quoted here.

## Result (2026-09-30, one RTX 5090 on RunPod)

- With short training (3000 steps) the scripted expert's demonstrations win clearly: 44/50 against 34/50.
- With longer training (20000 steps) the Runway arm reaches 43/50 against the expert's 44/50. The 95% intervals overlap almost completely, so at this sample size the two are statistically indistinguishable.
- The expert is a strong baseline: it knows the exact cube pose and never hesitates, and its episodes are shorter (about 20k frames against 25,190). Real teleoperation is noisier than this stand-in.
- Raw numbers: data/probes/bench/*.eval.json. Wilson 95% intervals in brackets.
