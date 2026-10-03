# RoboHub: research-backed pipeline for the Runway Hackathon (Wed 2026-09-30)

Written 2026-09-24 evening. 0 Runway credits spent (balance 48). Literature details with sources: docs/research/LIT-NOTES.md
(primary sources only, read 2026-09-24). Every number below is either measured here (file named) or read from a
source (URL named). Probe results marked PENDING are running now and are filled in below when they finish.

## 1. What we have (measured)

| | Model | Data | Compute | Unseen positions in the bowl |
|---|---|---|---|---|
| BEFORE | `lerobot/smolvla_base`, no fine-tune | none | none | 0/50 (Wilson 0.00-0.07) |
| AFTER | same, 1000 steps, batch 8 | 44 sim episodes from 4 accepted Runway clips | 49 min 28 s, M4 MacBook, MPS | 16/50 (0.21-0.46) |
| AFTER, shorter | same, 650 steps | same | 39 min | 17/50 |
| Privileged reference (off screen) | 2-layer MLP | same, plus the cube position | 26.5 s CPU | 46/50 |

## 2. Diagnosis: why SmolVLA stops at 16/50

### 2a. The 34 failures, sorted (scripts/eval_failures.py on data/smolvla/robohub-1h/checkpoints/last/eval_traj.npz)

| Outcome (1000-step model, 50 seeds) | n | median closest gripper-to-cube | median max lift |
|---|---|---|---|
| Success | 16 | 1.5 cm | 3.0 cm |
| Never moved the cube | 4 | 1.5 cm | 0.05 cm |
| Pushed the cube, never lifted it | 14 | 2.3 cm | 0.15 cm |
| Lifted, ended at the bowl rim (within 3 cm of the bowl wall) | 10 | 1.3 cm | 2.75 cm |
| Lifted, dropped away from the bowl | 6 | 1.2 cm | 2.65 cm |

So 18 failures are **grasp** failures (it reaches the cube but closes off-centre or late) and 16 are **carry/place**
failures (it lifts, then clips the bowl rim or drops). Success is spread over all 4 source clips' regions (2, 6, 3, 5
of 12-13 each): no single clip is broken.

### 2b. Causes, ranked by evidence

1. **Undertrained by two orders of magnitude.** 1000 steps x batch 8 = 8,000 samples = 0.53 epoch (the log says
   `epch:0.53`); every frame was seen at most once. SmolVLA's docs: "Training the model for 20k steps will roughly
   take ~4 hrs on a single A100 GPU", command `--batch_size=64 --steps=20000` = 1.28M samples (160x ours). The paper
   fine-tunes 100,000 steps at batch 64 for sim benchmarks and 200,000 for real tasks (Sec 4.3,
   https://arxiv.org/abs/2506.01844; https://huggingface.co/docs/lerobot/smolvla). The 650 vs 1000 comparison
   (17 vs 16) does NOT show "more steps don't help": both runs are under one epoch, and lerobot compresses the same
   cosine schedule into each run (lr 1e-4 down to 2.5e-6), so they are nearly the same run.
2. **Open-loop chunks.** Our eval executes all 50 actions of a chunk (1.67 s) before looking again
   (`n_action_steps: 50`, the checkpoint's default). SmolVLA's paper: "In simulation, we perform inference by sampling
   new observations and predicting a new action after each executed action" (Sec 4.3). Their LIBERO ablation
   (Table 13) has 50 action steps as the worst setting (51.8%) and 10 steps at 82.8%. Grasp failures (18) are exactly
   what a 1.67 s blind window produces: the gripper arrives slightly off and cannot correct. Probe P1 below.
3. **Thin demonstrations: 4 motions, 44 positions, carry margin under 1 cm.** All 44 episodes are copies of 4 human
   motions. Measured on the dataset (observation.environment_state): the cube's maximum lift is 2.8-3.6 cm (median
   3.3) and the bowl wall is 2.5 cm, so the demonstrations clear the rim by 0.3-1.1 cm. A policy that carries 5 mm
   lower than the demo hits the rim: that is the 10 "rim" failures. The MLP copes (it is handed the cube position and
   replays the motion precisely); a camera policy after half an epoch does not. The docs' own warning: "We tried
   similar dataset with 25 episodes, and it was not enough leading to a bad performance" and they recommend 50
   episodes as 5 positions x 10 demos.
4. **One small, far camera.** 320x240 front camera, upscaled into SmolVLA's 512x512 input; the cube is a few dozen
   pixels. SmolVLA's SO-100 tasks used a top and a wrist camera (LIT-NOTES, SmolVLA section). Our scene already has a
   wrist camera; the dataset never recorded it. Probe P3.
5. **Not the cause (checked):** normalisation (BEFORE is 0/50 with either our stats or the base model's own, and the
   AFTER model uses the same stats lerobot-train computed); the eval positions (they sit inside the same +-3 cm x
   +-2 cm boxes the training copies were drawn from, so this is not an out-of-distribution test); the task text
   (identical in training and eval).

## 3. Interventions ranked by expected gain per credit and per hour

| # | Intervention | Credits | Laptop time | Expected effect | Verdict |
|---|---|---|---|---|---|
| 1 | Replan every 10 actions at eval (P1) | 0 | 0 training, eval 3 h 40 min under load (measured) | paper ablation said large | **Measured: no gain (11/50 vs 16/50, p = 0.37).** Dropped |
| 2 | Train longer (2-5 epochs instead of 0.5) | 0 | 4 h 36 min for 3000 steps under load (measured); quiet-machine time unmeasured | the recipe gap is 160x | **Measured: 22 -> 33/50 on the same data (McNemar p = 0.007).** The biggest lever |
| 3 | More positions from the same clips (MimicGen-style re-anchoring, re-gated in physics) (P2; measured 16 -> 22/50, p = 0.07) | 0 | ~2 h for 70 episodes under load (109 s each, 2 cameras, measured) | more position diversity; same 4 motions | **Probably moves it a little**; honest as "MimicGen's step", not new demos |
| 4 | Wrist camera (P3) | 0 | same dataset, eval renders 2 cameras | grasp precision | **Plausible**; SmolVLA's real tasks used one |
| 5 | More accepted clips (new motions): prompt planner v2 on Runway | 27 per clip; acceptance on the new sentence was 0/5 with v1 prompts, 4/7 on the old sentence | ~2 min data stage per clip | new motions, higher carries, more variety | **Moves it if acceptance is decent**; the only lever that needs credits |
| 6 | Raise the carry height in retargeting (clear the rim by 2 cm) | 0 | re-write dataset | removes the rim failures' cause | Effective but it edits the human motion; say so if used |
| 7 | Unfreeze the vision encoder | 0 | slower, more memory | paper keeps VLM frozen | **Theatre** for this budget |
| 8 | Visual randomisation (lighting, table colour) | 0 | same | sim eval has one fixed look | **Theatre for the sim number**; matters only for a real arm |
| 9 | aleph2 photoreal pass (V8-PLAN Option 2) | 56 minimum, 28/s | none | the sim eval renders the sim camera, so photoreal frames cannot raise the sim score | **Theatre for the number**, real value is the demo visual and future real-arm transfer. Never claim it raised 16/50 |

## 4. Probes (0 credits, run tonight at nice 19 + taskpolicy -b, one at a time)

- P1 replan: same 1000-step checkpoint, `--n-action-steps 10`, same 50 seeds (scripts/probe_replan.sh ->
  data/smolvla/probe-replan10/eval.json). RESULT (2026-09-24 22:06): **11/50 (Wilson 0.13-0.35) vs 16/50 with 50-action
  chunks: no gain** (Fisher two-sided p = 0.37, so "worse" is not shown either). Per seed: 5 succeed in both, 11 only
  with 50-step chunks, 6 only with 10. Failures: 21 grasp (18 pushed, 3 untouched), 10 rim, 8 dropped away. Reading:
  with this undertrained model, replanning does not fix grasp precision; each fresh chunk starts from new flow-matching
  noise, and switching chunks every 1/3 s shows up as more pushes. Keep the checkpoint default (50) for the stage
  number; the paper's ablation is for a fully trained model on LIBERO, not ours.
- P2 augmented data: 4 accepted clips x 25 re-anchor tries each, re-gated by the unchanged robot gates: 70/100 copies
  passed (v01 23, v03 12, v06 22, v07 13; rejects: gripper closes before lift 14, IK limits 11, opens over bowl 5) -> 74
  episodes, 25,190 frames, front + wrist cameras (scripts/augment_dataset.py -> data/probes/aug25). SmolVLA 1000 steps,
  batch 8, front camera only (same budget as the film run; 2 h 4 min wall under swap).
  RESULT (2026-09-26 02:32): **22/50 (Wilson 0.31-0.58) vs 16/50** on the same 50 seeds. Unpaired Fisher p = 0.30;
  paired by seed: 15 succeed in both, 7 only with 74 episodes, 1 only with 44 (exact McNemar p = 0.07). Suggestive, not
  significant. Where it helped: carry/place failures fell (rim 10 -> 6, dropped 6 -> 3); grasp failures did not (18 -> 19:
  17 pushed, 2 untouched). vs BEFORE 0/50: p = 2e-8.
- P3 wrist: aug25, front + wrist (wrist -> camera2), 1000 steps. NOT RUN to completion: started 13:08 09-26, stalled at 31 s/step
  under swap (132/1000 steps in 2 h 47 min), stopped at 15:59 to keep a promised quiet window for another job. No result;
  the wrist camera's effect is untested.
- P4 longer: aug25, front camera, **3000 steps** (2.4 epochs; 4 h 36 min wall at 5.5 s/step on the M4 under the research
  render; loss 0.66 -> ~0.06). RESULT (2026-09-26 12:14): **33/50 (Wilson 0.52-0.78)**.
  - vs the film model (44 episodes, 1000 steps, 16/50): paired by seed, 17 positions succeed only with P4 and 0 only with
    the film model (every film success is also a P4 success); exact McNemar p < 0.0001; unpaired Fisher p = 0.001.
  - vs P2 (same 74 episodes, 1000 steps, 22/50): 13 only-P4, 2 only-P2, McNemar p = 0.007; Fisher p = 0.04. So the
    extra training epochs are a real effect on top of the extra positions.
  - vs BEFORE (0/50): Fisher p = 7e-14.
  - Failures left (17): 10 grasp misses (pushed, no lift), 7 lifted but ended at the bowl rim; 0 dropped away, 0 untouched.
    Success is even across the 4 source clips (8, 9, 8, 8 of 12-13).
  - Eval-noise check (2026-09-27 04:25): same checkpoint, same 50 positions, flow-matching noise redrawn (torch seed +1000):
    **29/50 (0.44-0.71)**. Paired with the recorded 33/50: 22 succeed in both, 11 only in the first draw, 7 only in the
    second, 10 in neither. So SmolVLA's sampling noise alone moves the score by several points and flips 18 of 50
    positions; pooled over both draws it is **62/100 = 0.62 (Wilson 0.52-0.71)**. The honest headline is "about 30 of 50
    (33 and 29 on two draws)", not "33" alone. Both draws still beat the film model (16/50) and BEFORE (0/50) clearly.
  - Single training run (one seed); training-seed spread is untested on top of this sampling spread.
All four: scripts/probe_chain.sh (log data/probes/chain.log), lean checkpoints (no optimizer state; disk is 2 GB above
the 20 GB floor, the chain stops itself under 21 GB).

## 5. The recommended pipeline

1. Sentence -> prompt planner v2 (PhyT2V-style rules) -> Runway gen4_image_turbo first frame + gen4_turbo 5 s.
2. MediaPipe + OpenCV tracking -> the 21 unchanged gates -> retarget (IK) -> MuJoCo replay gate (+ PyBullet cross-check).
3. MimicGen-style re-anchoring of every accepted clip, each copy re-gated in physics (report the pass rate, as MimicGen
   does).
4. LeRobot v3.0 dataset with front + wrist cameras (if P3 wins).
5. SmolVLA fine-tune: as many epochs as the time allows (not 0.5).
6. Eval in MuJoCo on the same 50 fixed seeds, 50-action chunks (the checkpoint default; P1 showed no gain from replanning).
7. Optional visual: one aleph2 restyle of a sim episode, gated by silhouette IoU, labelled "appearance only".

## 6. Odds of 30+/50 by Wednesday

**Reached in a probe: 33/50, and 29/50 on a second noise draw of the same model (62/100 pooled).** Same 4 accepted clips,
same eval. Odds that a day-of retrain with the same recipe scores 30+/50 on one draw: about 50% (judgement; the model's
own draws straddle 30). Safe plan: this checkpoint is the fallback; report it as "about 30 of 50 (29-33)". A new model
replaces it only if it is higher on the same 50 seeds over two noise draws. History: before P1 35-45%; after P1 25-35%;
after P2 35-45%; after P4 65-75%; after the noise check about 50%.

## 7. Runway credit plan

**APPROVED by Pranav 2026-09-24 (chat): the ~1,000-credit plan below, to run once Runway grants the credits.**
Scope is exactly the table below and its stop rule; anything beyond it (and the 50,000 day-of plan) still needs his yes.
Balance checked 2026-09-24 evening: 48 (grant not arrived).

**The ~1,000 test credits (if Runway grants them):**
| Use | Calls | Credits |
|---|---|---|
| Measure prompt planner v2: same sentence, 10 clips (turbo frame 2 + 5 s video 25) | 10 | 270 |
| Second sentence for task variety (e.g. a different colour/object) | 6 | 162 |
| aleph2 Option 2 probe: 2 sim episodes, 5 s each, 720p | 2 | 280 |
| Reserve (live run on stage + retries) | | ~290 |
Stop rule: if v2 acceptance is under 2 of 10, stop generating and spend nothing more on clips before the day.

**The 50,000 on the day:** training time, not credits, is the constraint (1 epoch of 44 episodes = 1,873 steps = ~1.4 h on this
laptop at batch 8). Plan: ~80 clips across 2-3 phrasings (~2,200 credits), which at 40-60% acceptance gives 30-50
accepted motions, 10x today; 4-8 aleph2 restyles for the visual (560-1,120); keep the rest unspent. More clips than
the laptop can train in the day buys nothing; say so rather than burn credits for a bigger number.

## 8. Day-of plan (Wednesday), with times

| Time | Step | Duration (measured or estimated) |
|---|---|---|
| Start | Generate ~80 clips in parallel batches | Runway: ~1-2 min per clip, parallel (v5 take: 3 videos in 0:42) |
| +0:30 | Data stage on all clips (niced off: the render is done by then) | ~2 min per clip measured under load (15:33 for 7) -> ~2.5 h for 80; start on the first batch while the rest generate |
| +1:30 | Re-anchor + write dataset | ~20 s per episode measured |
| +3:00 | SmolVLA training | as long as the schedule allows; best-probe config |
| +end-1:00 | Eval (50 seeds) + hero film | ~30 min |
Fallback at every step: the current 16/50 model and film exist; the new number replaces it only if it is higher on
the same 50 seeds.

## 9. Sentences he can say (all true today; each says what it compares)

1. "Same foundation model, before and after our data: 0 of 50 to 16 of 50 unseen cube positions, in 49 minutes on a
   laptop, from 4 generated clips. In simulation."
2. "SmolVLA's own docs recommend 20,000 training steps at batch 64, about 4 hours on an A100. We used 1,000 steps at
   batch 8 on a MacBook, under 1 percent of that."
3. "SmolVLA's paper fine-tunes on 50 real teleoperated demos per task. Our 44 simulated episodes come from 4 video
   clips, and no robot was teleoperated."
4. "Like MimicGen, we re-anchor each human demo to new cube positions and keep only the copies that still succeed
   in physics."
5. "NVIDIA's DreamGen also trains robots on video-model output, but it starts from thousands of real robot demos; we
   start from a sentence." (DreamGen: 2,884 GR1 pick-and-place demos seed the video model, arXiv 2505.12705.)
6. (P4, measured) "Same foundation model, same 4 generated clips: 0 of 50 before, about 30 of 50 after (33 and 29 on two
   runs of the test), trained on a MacBook. In simulation." (3000 steps took 4 h 36 min on the M4 while another job ran; do not say "under an hour" for this one.)
7. (After P2, if it holds on the day) "Same 4 clips, more simulated cube positions checked in physics: 16 to 22 of 50
   at the same 1,000 training steps." Say "suggestive" if asked about significance (paired p = 0.07).
Never: "real-world", "matches SmolVLA's 78%", "44 demonstrations" without "from 4 clips", or any published number as
a benchmark we beat (LIT-NOTES comparability table: no work shares our task, embodiment and metric).
